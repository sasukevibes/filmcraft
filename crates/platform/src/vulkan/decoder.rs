//! [`VulkanH264Decoder`]: H.264 decoded on the GPU through Vulkan Video, driven by this
//! workspace's own front-end (`filmcraft_h264::accel`), so pictures come out exactly when, and in
//! the order, the software decoder outputs them.
//!
//! Each `Decode` event becomes one decode into a DPB slot ([`Slots`]); each `Output` event copies
//! the slot's NV12 planes to the host and into a planar `Yuv8` frame cropped to the conformance
//! window, with the colour and pixel aspect the software decoder reports.

use filmcraft_codecs::hw::{NalCodec, NalStreamInfo};
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};
use filmcraft_h264::accel::{DecodePicture, Event, Frontend, OutputPicture};
use filmcraft_h264::params::{Pps, Sps};
use std::sync::Arc;

use super::ffi::{Codec, DecodeJob, Params, PictureInfo, RefInfo, Session, SessionSpec};
use super::frames::{self, Presentation};
use super::h264;
use super::slots::Slots;

/// The Vulkan Video decoder for one `avcC` stream.
pub struct VulkanH264Decoder {
    session: Session,
    avcc: Vec<u8>,
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

/// The SPS and PPS of a stream's configuration record (each PPS parsed against its SPS).
pub(super) fn parse_parameter_sets(info: &NalStreamInfo) -> std::result::Result<(Vec<Sps>, Vec<Pps>), String> {
    let mut table: Vec<Option<Sps>> = vec![None; 32];
    let mut ppss = Vec::new();
    for nal in &info.parameter_sets {
        let Some(&header) = nal.first() else { continue };
        let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(1..).unwrap_or_default());
        match header & 0x1f {
            7 => {
                let sps = Sps::parse(&rbsp).map_err(|e| format!("SPS: {e}"))?;
                if let Some(slot) = table.get_mut(sps.id as usize) {
                    *slot = Some(sps);
                }
            }
            8 => {
                let pps = Pps::parse(&rbsp, &table).map_err(|e| format!("PPS: {e}"))?;
                // a later PPS with the same id replaces an earlier one, as in the software decoder
                ppss.retain(|p: &Pps| p.id != pps.id);
                ppss.push(pps);
            }
            _ => {}
        }
    }
    let spss: Vec<Sps> = table.into_iter().flatten().collect();
    if spss.is_empty() || ppss.is_empty() {
        return Err("no SPS / PPS in the configuration record".into());
    }
    Ok((spss, ppss))
}

impl VulkanH264Decoder {
    /// A decoder for the stream, or why the GPU does not take it (format, profile, level, size,
    /// no Vulkan Video device).
    pub fn new(info: NalStreamInfo, avcc: Vec<u8>) -> std::result::Result<Self, String> {
        if info.codec != NalCodec::H264 {
            return Err("not an H.264 stream".into());
        }
        if info.interlaced {
            return Err("field-coded H.264".into());
        }
        if info.chroma_format_idc != 1 || info.bit_depth_luma != 8 || info.bit_depth_chroma != 8 {
            return Err(format!("chroma_format_idc {} at {}/{}-bit (8-bit 4:2:0 only)", info.chroma_format_idc, info.bit_depth_luma, info.bit_depth_chroma));
        }
        let (spss, ppss) = parse_parameter_sets(&info)?;
        let mut profile = None;
        let mut level = 0;
        let (mut w, mut h, mut slots, mut max_refs) = (0, 0, 0, 1);
        for sps in &spss {
            // the software decoder (the fallback) must take the stream too
            sps.check_supported().map_err(|e| e.to_string())?;
            let p = h264::std_profile(sps)?;
            if profile.is_some_and(|q| q != p) {
                return Err("SPSs of different profiles".into());
            }
            profile = Some(p);
            level = level.max(h264::std_level(sps.level_idc)?);
            // every picture is decoded at the session's coded size
            if (w, h) != (0, 0) && (w, h) != (sps.width(), sps.height()) {
                return Err("SPSs of different picture sizes".into());
            }
            (w, h) = (sps.width(), sps.height());
            slots = slots.max(sps.max_dpb_frames() + 1);
            max_refs = max_refs.max(sps.max_num_ref_frames as usize);
        }
        let profile = profile.ok_or("no SPS")?;
        let sets = h264::parameter_sets(&spss, &ppss)?;
        let spec = SessionSpec {
            codec: Codec::H264(profile),
            level,
            coded: (w, h),
            slots: u32::try_from(slots).map_err(|_| "DPB size overflows".to_string())?,
            max_refs: u32::try_from(max_refs).map_err(|_| "reference count overflows".to_string())?,
        };
        let dev = super::device()?;
        let name = format!("Vulkan Video H.264 ({})", dev.name);
        let session = Session::new(Arc::clone(&dev), &spec, Params::H264(&sets))?;
        let frontend = Frontend::from_avcc(&avcc).map_err(|e| e.to_string())?;
        Ok(Self { session, avcc, frontend, info, slots: Slots::new(slots), reset_session: true, max_refs, fail_after: None, fed: 0, name })
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
            return Err("frame_num gap (non-existing reference frames)".into());
        }
        if p.refs.len() > self.max_refs {
            return Err(format!("{} reference pictures (the stream allows {})", p.refs.len(), self.max_refs));
        }
        let layer = self.slots.assign(&p.dpb, p.id)?;
        let mut refs = Vec::with_capacity(p.refs.len());
        for r in &p.refs {
            let slot = self.slots.slot_of(r.id).ok_or_else(|| format!("reference picture {} was never decoded", r.id))?;
            refs.push((slot, RefInfo::H264(h264::reference_info(r)?)));
        }
        let (bitstream, offsets) = super::bitstream(&p.slices)?;
        let job = DecodeJob {
            bitstream: &bitstream,
            slice_offsets: &offsets,
            picture: PictureInfo::H264(h264::picture_info(p)?),
            setup: (layer, RefInfo::H264(h264::setup_info(p)?)),
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
        let presentation =
            Presentation { crop: o.crop, matrix: o.color.matrix, transfer: o.color.transfer, full_range: o.color.full_range, sar: o.sar, bits: 8 };
        let planes = self.session.read_picture(layer)?;
        let frame = frames::frame(planes, iw, ih, 1, &presentation)?;
        Ok(DecodedFrame { pts: o.pts, frame, draft: false })
    }
}

impl VideoDecoder for VulkanH264Decoder {
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
        match Frontend::from_avcc(&self.avcc) {
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
