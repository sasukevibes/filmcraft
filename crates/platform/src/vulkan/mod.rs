//! Vulkan Video hardware decoding (Linux, docs/adr/0002-linux-vulkan-video.md): H.264 and HEVC
//! decoded by the GPU's video engine, driven by this workspace's own front-ends
//! (`filmcraft_h264::accel`, `filmcraft_hevc::accel`), pictures copied back into planar frames.
//!
//! The decode device is opened once, on first use, from the system Vulkan loader
//! (`libvulkan.so.1`, loaded at runtime): a GPU with a video decode queue that takes H.264 or HEVC.
//! Everything that can fail is a `Result`; the factory declines (the software decoder is used)
//! whenever the GPU cannot take a stream.

mod decoder;
mod decoder_h265;
#[allow(unsafe_code)]
mod ffi;
mod frames;
mod h264;
mod h265;
mod slots;
#[cfg(test)]
mod trace_headers;

use std::sync::{Arc, OnceLock};

use filmcraft_codecs::hw::{NalCodec, NalStreamInfo};

pub use decoder::VulkanH264Decoder;
pub use decoder_h265::VulkanHevcDecoder;

/// A slice NAL unit's prefix in the bitstream buffer: the decoders take Annex B start codes.
const START_CODE: [u8; 3] = [0, 0, 1];

/// Whether the system Vulkan loader can be opened (no GPU driver is started).
pub fn loader_available() -> Result<(), String> {
    ffi::loader_available()
}

/// The shared decode device, opened on first use; the error says why there is none.
pub(crate) fn device() -> Result<Arc<ffi::Device>, String> {
    static DEVICE: OnceLock<Result<Arc<ffi::Device>, String>> = OnceLock::new();
    let d = DEVICE.get_or_init(|| {
        let opened = std::panic::catch_unwind(ffi::Device::open).unwrap_or_else(|_| Err("panic while opening the Vulkan device".into()));
        match &opened {
            Ok(d) => log::info!("Vulkan Video: decoding {} on {}", d.codec_names(), d.name),
            Err(e) => log::info!("Vulkan Video unavailable: {e}"),
        }
        opened.map(Arc::new)
    });
    match d {
        Ok(d) if d.is_lost() => Err("the GPU was lost".into()),
        Ok(d) => Ok(Arc::clone(d)),
        Err(e) => Err(e.clone()),
    }
}

/// The GPU hardware decoding would use and the codecs it decodes ("NVIDIA GeForce RTX 4070: H.264
/// and HEVC"), or why there is none (opens the device on first use).
pub fn probe() -> Result<String, String> {
    device().map(|d| format!("{}: {}", d.name, d.codec_names()))
}

/// Whether a stream's quantisation matrices reach the GPU: H.264 scaling matrices that are not
/// flat, HEVC scaling lists. Parameter sets that do not parse count as matrices (the stricter
/// check; the GPU declines such a stream anyway).
pub(crate) fn scaling_in_use(info: &NalStreamInfo) -> bool {
    match info.codec {
        NalCodec::H264 => decoder::parse_parameter_sets(info)
            .map_or(true, |(spss, ppss)| spss.iter().any(|s| !s.scaling.is_flat()) || ppss.iter().any(|p| !p.scaling.is_flat())),
        NalCodec::Hevc => decoder_h265::parse_parameter_sets(info).map_or(true, |(_, spss, _)| spss.iter().any(|s| s.scaling_list_enabled)),
    }
}

/// A picture's bitstream: each slice NAL unit after a start code, and where each starts.
pub(crate) fn bitstream(slices: &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u32>), String> {
    let len = slices.iter().try_fold(0usize, |a, s| a.checked_add(s.len())?.checked_add(START_CODE.len())).ok_or("bitstream too large")?;
    let mut data = Vec::with_capacity(len);
    let mut offsets = Vec::with_capacity(slices.len());
    for s in slices {
        offsets.push(u32::try_from(data.len()).map_err(|_| "slice offset overflows".to_string())?);
        data.extend_from_slice(&START_CODE);
        data.extend_from_slice(s);
    }
    Ok((data, offsets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitstream_puts_a_start_code_before_each_slice() {
        let (data, offsets) = bitstream(&[vec![0x65, 1, 2], vec![0x41, 3]]).unwrap();
        assert_eq!(data, [0, 0, 1, 0x65, 1, 2, 0, 0, 1, 0x41, 3]);
        assert_eq!(offsets, [0, 6]);
        assert_eq!(bitstream(&[]).unwrap(), (Vec::new(), Vec::new()));
    }
}
