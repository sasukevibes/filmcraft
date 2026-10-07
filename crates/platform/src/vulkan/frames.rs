//! Copied-out pictures to FilmCraft frames: tightly packed biplanar 4:2:0 (NV12, or P010 with the
//! samples in the high bits of 16-bit words) to planar `Yuv8` / `Yuv16`, cropped to the output
//! window, with the colour and pixel aspect the software decoders report.

use std::sync::Arc;

use filmcraft_frame::{Chroma, PixelData, VideoFrame, pool};

/// How an output picture is presented (from the front-end's output event).
pub(crate) struct Presentation {
    /// Output rectangle in luma samples: x, y, width, height.
    pub(crate) crop: (u32, u32, u32, u32),
    /// VUI matrix and transfer code points, full range.
    pub(crate) matrix: u8,
    pub(crate) transfer: u8,
    pub(crate) full_range: bool,
    pub(crate) sar: (u16, u16),
    /// Significant bits per sample (8: `Yuv8`, else `Yuv16`).
    pub(crate) bits: u32,
}

/// A planar 4:2:0 frame from the copied-out planes of an `iw`×`ih` picture with `bytes`-byte
/// samples (1: NV12, 2: P010).
pub(crate) fn frame(data: &[u8], iw: u32, ih: u32, bytes: usize, p: &Presentation) -> Result<VideoFrame, String> {
    let (cx, cy, cw, ch) = p.crop;
    if cw == 0 || ch == 0 || cx.checked_add(cw).is_none_or(|r| r > iw) || cy.checked_add(ch).is_none_or(|b| b > ih) {
        return Err(format!("crop {cw}x{ch}+{cx}+{cy} outside the {iw}x{ih} picture"));
    }
    if !matches!(bytes, 1 | 2) || (bytes == 1) != (p.bits <= 8) || p.bits > 16 {
        return Err(format!("{bytes}-byte samples for {}-bit video", p.bits));
    }
    let (iw, ih) = (iw as usize, ih as usize);
    let (cx, cy, w, h) = (cx as usize, cy as usize, cw as usize, ch as usize);
    let stride = iw * bytes;
    let luma = stride * ih;
    if data.len() < luma + luma / 2 {
        return Err("decoded picture is smaller than expected".into());
    }
    let row = |start: usize, len: usize| data.get(start..start + len).ok_or_else(|| "decoded picture row out of range".to_string());
    let (cw2, ch2) = (w.div_ceil(2), h.div_ceil(2));
    let (cox, coy) = (cx / 2, cy / 2);
    let data = if bytes == 1 {
        let mut y = pool::take_u8(w * h);
        for r in 0..h {
            y.extend_from_slice(row((cy + r) * stride + cx, w)?);
        }
        let (mut u, mut v) = (pool::take_u8(cw2 * ch2), pool::take_u8(cw2 * ch2));
        for r in 0..ch2 {
            for pair in row(luma + (coy + r) * stride + cox * 2, cw2 * 2)?.as_chunks::<2>().0 {
                u.push(pair[0]);
                v.push(pair[1]);
            }
        }
        PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, alpha: None }
    } else {
        let shift = 16 - p.bits;
        let word = |b: &[u8; 2]| u16::from_le_bytes(*b) >> shift;
        let mut y = pool::take_u16(w * h);
        for r in 0..h {
            y.extend(row((cy + r) * stride + cx * 2, w * 2)?.as_chunks::<2>().0.iter().map(word));
        }
        let (mut u, mut v) = (pool::take_u16(cw2 * ch2), pool::take_u16(cw2 * ch2));
        for r in 0..ch2 {
            for q in row(luma + (coy + r) * stride + cox * 4, cw2 * 4)?.as_chunks::<4>().0 {
                u.push(word(&[q[0], q[1]]));
                v.push(word(&[q[2], q[3]]));
            }
        }
        PixelData::Yuv16 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, bits: p.bits, alpha: None }
    };
    Ok(VideoFrame {
        width: cw,
        height: ch,
        data,
        color: filmcraft_codecs::video::vui_color(cw, ch, p.matrix, p.transfer, p.full_range),
        par: filmcraft_codecs::video::sar_par(p.sar),
        pts: filmcraft_time::Tick::ZERO,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn presentation(crop: (u32, u32, u32, u32), bits: u32) -> Presentation {
        Presentation { crop, matrix: 1, transfer: 1, full_range: false, sar: (0, 0), bits }
    }

    /// NV12 planes become cropped planar planes: luma rows, chroma deinterleaved.
    #[test]
    fn nv12_planes_are_cropped_and_deinterleaved() {
        let (iw, ih) = (8u32, 4u32);
        let mut data: Vec<u8> = (0..32).collect();
        for r in 0..2 {
            for c in 0..4 {
                data.push(100 + (r * 4 + c) as u8);
                data.push(200 + (r * 4 + c) as u8);
            }
        }
        let f = frame(&data, iw, ih, 1, &presentation((2, 2, 4, 2), 8)).unwrap();
        assert_eq!((f.width, f.height), (4, 2));
        let PixelData::Yuv8 { planes, chroma, .. } = &f.data else { panic!("8-bit planar") };
        assert_eq!(*chroma, Chroma::C420);
        assert_eq!(*planes[0], [18, 19, 20, 21, 26, 27, 28, 29]);
        assert_eq!(*planes[1], [105, 106]);
        assert_eq!(*planes[2], [205, 206]);
        assert_eq!(f.par, (1, 1));
        // crops outside the picture, short buffers and mismatched sample sizes are errors
        assert!(frame(&data, iw, ih, 1, &presentation((6, 0, 4, 2), 8)).is_err());
        assert!(frame(&data, iw, ih, 1, &presentation((0, 0, 0, 2), 8)).is_err());
        assert!(frame(&data[..40], iw, ih, 1, &presentation((0, 0, 8, 4), 8)).is_err());
        assert!(frame(&data, iw, ih, 1, &presentation((u32::MAX, 0, 2, 2), 8)).is_err());
        assert!(frame(&data, iw, ih, 2, &presentation((0, 0, 2, 2), 8)).is_err());
    }

    /// P010 words (10 bits in the high bits, little endian) become 10-bit samples.
    #[test]
    fn p010_planes_are_shifted_down() {
        let (iw, ih) = (4u32, 2u32);
        let mut data = Vec::new();
        for i in 0..8u16 {
            data.extend_from_slice(&((i * 100) << 6).to_le_bytes());
        }
        // one chroma row of two (U, V) pairs
        for (u, v) in [(512u16, 1023u16), (1, 2)] {
            data.extend_from_slice(&(u << 6).to_le_bytes());
            data.extend_from_slice(&(v << 6).to_le_bytes());
        }
        let f = frame(&data, iw, ih, 2, &presentation((0, 0, 4, 2), 10)).unwrap();
        let PixelData::Yuv16 { planes, bits, .. } = &f.data else { panic!("16-bit planar") };
        assert_eq!(*bits, 10);
        assert_eq!(*planes[0], [0, 100, 200, 300, 400, 500, 600, 700]);
        assert_eq!(*planes[1], [512, 1]);
        assert_eq!(*planes[2], [1023, 2]);
    }
}
