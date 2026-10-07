//! [`VulkanHevcDecoder`]: HEVC (Main, Main 10) decoded on the GPU through Vulkan Video, driven by
//! this workspace's own front-end (`filmcraft_hevc::accel`), so pictures come out exactly when, and
//! in the order, the software decoder outputs them (RASL pictures it skips are skipped too).
//!
//! Each `Decode` event becomes one decode into a DPB slot ([`Slots`]) with the RPS lists as slot
//! indices; each `Output` event copies the slot's NV12 / P010 planes to the host and into a planar
//! `Yuv8` / `Yuv16` frame cropped to the conformance window.

use std::sync::Arc;

use filmcraft_codecs::hw::{NalCodec, NalStreamInfo};
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};
use filmcraft_hevc::accel::{DecodePicture, Event, Frontend, OutputPicture};
use filmcraft_hevc::params::{Pps, Sps, Vps};

use super::ffi::{Codec, DecodeJob, Params, PictureInfo, RefInfo, Session, SessionSpec};
use super::frames::{self, Presentation};
use super::h265;
use super::slots::Slots;

/// The Vulkan Video decoder for one `hvcC` stream.
pub struct VulkanHevcDecoder {
    session: Session,
    hvcc: Vec<u8>,
    frontend: Frontend,
    info: NalStreamInfo,
    slots: Slots,
    /// The next decode resets the session's DPB state (after creation and every seek).
    reset_session: bool,
    max_refs: usize,
    /// Test hook: fail every decode once this many samples were fed.
    fail_after: Option<u64>,
    fed: u64,
    name: String,
}

/// The base-layer VPS, SPS and PPS of a stream's configuration record; a later set with the same
/// id replaces an earlier one, as in the software decoder (the session takes each id once).
pub(super) fn parse_parameter_sets(info: &NalStreamInfo) -> std::result::Result<(Vec<Vps>, Vec<Sps>, Vec<Pps>), String> {
    let (mut vpss, mut spss, mut ppss): (Vec<Vps>, Vec<Sps>, Vec<Pps>) = (Vec::new(), Vec::new(), Vec::new());
    for nal in &info.parameter_sets {
        let (Some(&header), Some(&second)) = (nal.first(), nal.get(1)) else { continue };
        // nuh_layer_id: other layers (MV-HEVC views) are not decoded, as in the software decoder
        if (header & 1) != 0 || (second >> 3) != 0 {
            continue;
        }
        let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(2..).unwrap_or_default());
        match (header >> 1) & 0x3f {
            32 => {
                let v = Vps::parse(&rbsp).map_err(|e| format!("VPS: {e}"))?;
                vpss.retain(|x| x.id != v.id);
                vpss.push(v);
            }
            33 => {
                let s = Sps::parse(&rbsp).map_err(|e| format!("SPS: {e}"))?;
                spss.retain(|x| x.id != s.id);
                spss.push(s);
            }
            34 => {
                let p = Pps::parse(&rbsp).map_err(|e| format!("PPS: {e}"))?;
                ppss.retain(|x| x.id != p.id);
                ppss.push(p);
            }
            _ => {}
        }
    }
    if vpss.is_empty() || spss.is_empty() || ppss.is_empty() {
        return Err("no VPS / SPS / PPS in the configuration record".into());
    }
    Ok((vpss, spss, ppss))
}

impl VulkanHevcDecoder {
    /// A decoder for the stream, or why the GPU does not take it (format, profile, level, size,
    /// no Vulkan Video HEVC device).
    pub fn new(info: NalStreamInfo, hvcc: Vec<u8>) -> std::result::Result<Self, String> {
        if info.codec != NalCodec::Hevc {
            return Err("not an HEVC stream".into());
        }
        let depth = info.bit_depth_luma;
        if info.chroma_format_idc != 1 || depth != info.bit_depth_chroma || !matches!(depth, 8 | 10) {
            return Err(format!("chroma_format_idc {} at {}/{}-bit (8- or 10-bit 4:2:0 only)", info.chroma_format_idc, depth, info.bit_depth_chroma));
        }
        let (vpss, spss, ppss) = parse_parameter_sets(&info)?;
        let mut profile = None;
        let (mut level, mut w, mut h, mut dpb) = (0, 0, 0, 1);
        for sps in &spss {
            // the software decoder (the fallback) must take the stream too
            sps.check_supported().map_err(|e| e.to_string())?;
            if sps.bit_depth_luma != depth || sps.bit_depth_chroma != depth {
                return Err("SPSs of different bit depths".into());
            }
            let p = h265::std_profile(&sps.ptl, depth)?;
            if profile.is_some_and(|q| q != p) {
                return Err("SPSs of different profiles".into());
            }
            profile = Some(p);
            level = level.max(h265::std_level(sps.ptl.level_idc)?);
            // every picture is decoded at the session's coded size
            if (w, h) != (0, 0) && (w, h) != (sps.width, sps.height) {
                return Err("SPSs of different picture sizes".into());
            }
            (w, h) = (sps.width, sps.height);
            dpb = dpb.max(sps.max_dec_pic_buffering as usize);
        }
        for pps in &ppss {
            pps.check_supported().map_err(|e| e.to_string())?;
        }
        let profile = profile.ok_or("no SPS")?;
        let sets = h265::parameter_sets(&vpss, &spss, &ppss)?;
        let as_u32 = |v: usize| u32::try_from(v).map_err(|_| "DPB size overflows".to_string());
        let spec = SessionSpec { codec: Codec::H265(profile, depth), level, coded: (w, h), slots: as_u32(dpb + 1)?, max_refs: as_u32(dpb)? };
        let dev = super::device()?;
        let name = format!("Vulkan Video HEVC ({})", dev.name);
        let session = Session::new(Arc::clone(&dev), &spec, Params::H265(&sets))?;
        let frontend = Frontend::from_hvcc(&hvcc).map_err(|e| e.to_string())?;
        Ok(Self { session, hvcc, frontend, info, slots: Slots::new(dpb + 1), reset_session: true, max_refs: dpb, fail_after: None, fed: 0, name })
    }

    /// Test hook: from the `n`-th sample on, every decode fails as a lost session would.
    pub fn fail_after(&mut self, n: u64) {
        self.fail_after = Some(n);
    }

    fn run(&mut self, events: Vec<Event>) -> std::result::Result<Vec<DecodedFrame>, String> {
        let mut out = Vec::new();
        for e in events {
            match e {
                Event::Decode(p) => self.decode_picture(&p)?,
                Event::Output(o) => out.push(self.output(&o)?),
            }
        }
        Ok(out)
    }

    fn decode_picture(&mut self, p: &DecodePicture) -> std::result::Result<(), String> {
        if p.refs.iter().any(|r| r.non_existing) {
            return Err("missing reference pictures".into());
        }
        if p.refs.len() > self.max_refs {
            return Err(format!("{} reference pictures (the DPB holds {})", p.refs.len(), self.max_refs));
        }
        let layer = self.slots.assign(&p.dpb, p.id)?;
        let mut refs = Vec::with_capacity(p.refs.len());
        for r in &p.refs {
            let slot = self.slots.slot_of(r.id).ok_or_else(|| format!("reference picture {} was never decoded", r.id))?;
            refs.push((slot, RefInfo::H265(h265::reference_info(r))));
        }
        let picture = h265::picture_info(p, |id| self.slots.slot_of(id))?;
        let (bitstream, offsets) = super::bitstream(&p.slices)?;
        let job = DecodeJob {
            bitstream: &bitstream,
            slice_offsets: &offsets,
            picture: PictureInfo::H265(picture),
            setup: (layer, RefInfo::H265(h265::setup_info(p))),
            refs: &refs,
            reset: self.reset_session,
        };
        self.session.decode(&job)?;
        self.reset_session = false;
        Ok(())
    }

    fn output(&mut self, o: &OutputPicture) -> std::result::Result<DecodedFrame, String> {
        let layer = self.slots.slot_of(o.id).ok_or_else(|| format!("picture {} is output but was never decoded", o.id))?;
        let (iw, ih) = self.session.image_size();
        let bytes = self.session.bytes_per_sample() as usize;
        let presentation =
            Presentation { crop: o.crop, matrix: o.color.matrix, transfer: o.color.transfer, full_range: o.color.full_range, sar: o.sar, bits: o.bit_depth };
        let planes = self.session.read_picture(layer)?;
        let frame = frames::frame(planes, iw, ih, bytes, &presentation)?;
        Ok(DecodedFrame { pts: o.pts, frame, draft: false })
    }
}

impl VideoDecoder for VulkanHevcDecoder {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        self.fed += 1;
        if self.fail_after.is_some_and(|n| self.fed > n) {
            return Err(CodecError::Decode("Vulkan Video session failed (test hook)".into()));
        }
        let events = self.frontend.decode(sample, pts).map_err(|e| CodecError::Decode(e.to_string()))?;
        self.run(events).map_err(CodecError::Decode)
    }

    fn flush(&mut self) -> Vec<DecodedFrame> {
        let events = self.frontend.flush();
        self.run(events).unwrap_or_else(|e| {
            log::warn!("{}: {e} while flushing", self.name);
            Vec::new()
        })
    }

    fn reset(&mut self) {
        // The session stays; only the front-end and the slot assignment start over, and the next
        // decode resets the session's DPB state.
        match Frontend::from_hvcc(&self.hvcc) {
            Ok(f) => self.frontend = f,
            Err(e) => log::warn!("{}: {e} while resetting", self.name),
        }
        self.slots.clear();
        self.reset_session = true;
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_random_access(&self, sample: &[u8]) -> Option<bool> {
        self.info.is_random_access(sample)
    }

    fn is_disposable(&self, sample: &[u8]) -> bool {
        self.info.is_disposable(sample)
    }
}
