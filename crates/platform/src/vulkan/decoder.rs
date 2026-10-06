//! [`VulkanH264Decoder`]: H.264 decoded on the GPU through Vulkan Video, driven by this
//! workspace's own front-end (`filmcraft_h264::accel`), so pictures come out exactly when, and in
//! the order, the software decoder outputs them.
//!
//! Each `Decode` event becomes one decode into a DPB slot ([`Slots`]); each `Output` event copies
//! the slot's NV12 planes to the host and into a planar `Yuv8` frame cropped to the conformance
//! window, with the colour and pixel aspect the software decoder reports.

use filmcraft_codecs::hw::{NalCodec, NalStreamInfo};
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};
use filmcraft_frame::{Chroma, PixelData, VideoFrame, pool};
use filmcraft_h264::accel::{DecodePicture, Event, Frontend, OutputPicture};
use filmcraft_h264::params::{Pps, Sps};
use std::sync::Arc;

use super::ffi::{DecodeJob, Session, SessionSpec};
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
fn parse_parameter_sets(info: &NalStreamInfo) -> std::result::Result<(Vec<Sps>, Vec<Pps>), String> {
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
            8 => ppss.push(Pps::parse(&rbsp, &table).map_err(|e| format!("PPS: {e}"))?),
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
            w = w.max(sps.width());
            h = h.max(sps.height());
            slots = slots.max(sps.max_dpb_frames() + 1);
            max_refs = max_refs.max(sps.max_num_ref_frames as usize);
        }
        let profile = profile.ok_or("no SPS")?;
        let sets = h264::parameter_sets(&spss, &ppss)?;
        let spec = SessionSpec {
            profile,
            level,
            coded: (w, h),
            slots: u32::try_from(slots).map_err(|_| "DPB size overflows".to_string())?,
            max_refs: u32::try_from(max_refs).map_err(|_| "reference count overflows".to_string())?,
        };
        let dev = super::device()?;
        let name = format!("Vulkan Video H.264 ({})", dev.name);
        let session = Session::new(Arc::clone(&dev), &spec, &sets)?;
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
            refs.push((slot, h264::reference_info(r)?));
        }
        let (bitstream, offsets) = h264::bitstream(&p.slices)?;
        let job = DecodeJob {
            bitstream: &bitstream,
            slice_offsets: &offsets,
            picture: h264::picture_info(p)?,
            setup: (layer, h264::setup_info(p)?),
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
        let planes = self.session.read_picture(layer)?;
        let frame = nv12_frame(planes, iw, ih, o)?;
        Ok(DecodedFrame { pts: o.pts, frame, draft: false })
    }
}

/// A planar 4:2:0 frame from tightly packed NV12 planes of an `iw`×`ih` image, cropped to the
/// output's window.
fn nv12_frame(data: &[u8], iw: u32, ih: u32, o: &OutputPicture) -> std::result::Result<VideoFrame, String> {
    let (cx, cy, cw, ch) = o.crop;
    if cw == 0 || ch == 0 || cx.checked_add(cw).is_none_or(|r| r > iw) || cy.checked_add(ch).is_none_or(|b| b > ih) {
        return Err(format!("crop {cw}x{ch}+{cx}+{cy} outside the {iw}x{ih} picture"));
    }
    let (iw, ih) = (iw as usize, ih as usize);
    let (cx, cy, w, h) = (cx as usize, cy as usize, cw as usize, ch as usize);
    let luma = iw * ih;
    if data.len() < luma + luma / 2 {
        return Err("decoded picture is smaller than expected".into());
    }
    let row = |start: usize, len: usize| data.get(start..start + len).ok_or_else(|| "decoded picture row out of range".to_string());
    let mut y = pool::take_u8(w * h);
    for r in 0..h {
        y.extend_from_slice(row((cy + r) * iw + cx, w)?);
    }
    let (cw2, ch2) = (w.div_ceil(2), h.div_ceil(2));
    let (cox, coy) = (cx / 2, cy / 2);
    let (mut u, mut v) = (pool::take_u8(cw2 * ch2), pool::take_u8(cw2 * ch2));
    for r in 0..ch2 {
        for pair in row(luma + (coy + r) * iw + cox * 2, cw2 * 2)?.as_chunks::<2>().0 {
            u.push(pair[0]);
            v.push(pair[1]);
        }
    }
    Ok(VideoFrame {
        width: cw,
        height: ch,
        data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, alpha: None },
        color: filmcraft_codecs::video::vui_color(cw, ch, o.color.matrix, o.color.transfer, o.color.full_range),
        par: filmcraft_codecs::video::sar_par(o.sar),
        pts: filmcraft_time::Tick::ZERO,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn output(crop: (u32, u32, u32, u32)) -> OutputPicture {
        OutputPicture {
            id: 1,
            pts: 7,
            poc: 0,
            key: true,
            crop,
            color: filmcraft_h264::ColorInfo { full_range: false, primaries: 1, transfer: 1, matrix: 1 },
            sar: (0, 0),
        }
    }

    /// NV12 planes become cropped planar planes: luma rows, chroma deinterleaved.
    #[test]
    fn nv12_planes_are_cropped_and_deinterleaved() {
        let (iw, ih) = (8u32, 4u32);
        let mut data: Vec<u8> = (0..32).collect();
        // chroma rows: (U, V) pairs 100+, 200+
        for r in 0..2 {
            for c in 0..4 {
                data.push(100 + (r * 4 + c) as u8);
                data.push(200 + (r * 4 + c) as u8);
            }
        }
        let f = nv12_frame(&data, iw, ih, &output((2, 2, 4, 2))).unwrap();
        assert_eq!((f.width, f.height), (4, 2));
        let PixelData::Yuv8 { planes, chroma, .. } = &f.data else { panic!("8-bit planar") };
        assert_eq!(*chroma, Chroma::C420);
        assert_eq!(*planes[0], [18, 19, 20, 21, 26, 27, 28, 29]);
        assert_eq!(*planes[1], [105, 106]);
        assert_eq!(*planes[2], [205, 206]);
        assert_eq!(f.par, (1, 1));
        // crops outside the picture and short buffers are errors
        assert!(nv12_frame(&data, iw, ih, &output((6, 0, 4, 2))).is_err());
        assert!(nv12_frame(&data, iw, ih, &output((0, 0, 0, 2))).is_err());
        assert!(nv12_frame(&data[..40], iw, ih, &output((0, 0, 8, 4))).is_err());
        assert!(nv12_frame(&data, iw, ih, &output((u32::MAX, 0, 2, 2))).is_err());
    }
}
