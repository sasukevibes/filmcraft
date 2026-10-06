//! Vulkan Video hardware decoding (Linux, docs/adr/0002-linux-vulkan-video.md): H.264 decoded by
//! the GPU's video engine, driven by this workspace's own front-end
//! (`filmcraft_h264::accel`), pictures copied back into planar frames.
//!
//! The decode device is opened once, on first use, from the system Vulkan loader
//! (`libvulkan.so.1`, loaded at runtime): a GPU with a video decode queue that takes H.264.
//! Everything that can fail is a `Result`; the factory declines (the software decoder is used)
//! whenever the GPU cannot take a stream.

mod decoder;
#[allow(unsafe_code)]
mod ffi;
mod h264;
mod slots;

use std::sync::{Arc, OnceLock};

pub use decoder::VulkanH264Decoder;

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
            Ok(d) => log::info!("Vulkan Video: H.264 decoding on {}", d.name),
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

/// The GPU hardware decoding would use, or why there is none (opens the device on first use).
pub fn probe() -> Result<String, String> {
    device().map(|d| d.name.clone())
}
