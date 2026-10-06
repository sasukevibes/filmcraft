//! OS media integration (layer L5): hardware video decoding through the operating system's codecs.
//!
//! [`register`] puts the platform's hardware decoder factory in front of FilmCraft's own decoders
//! (`filmcraft_codecs::register_video_decoder`): VideoToolbox on macOS for H.264 (`avcC`) and HEVC
//! (`hvcC`) streams, 8- and 10-bit, 4:2:0 and 4:2:2; Vulkan Video on Linux for 8-bit 4:2:0
//! progressive H.264 ([`vulkan`], docs/adr/0002-linux-vulkan-video.md). Elsewhere registration
//! does nothing and reports [`Availability::Unavailable`].
//!
//! Hardware decoding never makes a file undecodable:
//!
//! - the factory declines (falls through to the software decoder) when Settings ▸ Playback ▸
//!   Hardware decoding is Off (`filmcraft_codecs::hw::set_hardware_decoding`), when the stream's
//!   format is one the hardware path does not take, or when the OS cannot create a hardware
//!   session for it (profile, size, no hardware decoder);
//! - a decoder that fails mid-stream switches to the software decoder transparently
//!   ([`HybridDecoder`]) and logs it.
//!
//! Pictures are the software decoder's: same planes (bit-exact on the parity fixtures), colour,
//! pixel aspect, pts and presentation order, so the two are interchangeable.
//!
//! This is the one crate allowed to use `unsafe` (OS FFI), and only in its FFI modules
//! (docs/adr/0001-platform-ffi.md).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod hybrid;
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub mod videotoolbox;
// No `unsafe` outside its FFI submodule, which carries the allowance itself.
#[cfg(target_os = "linux")]
pub mod vulkan;

pub use hybrid::HybridDecoder;

/// What [`register`] made available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// A hardware decoder factory was registered (its name).
    Available(&'static str),
    /// Nothing to register on this system (why).
    Unavailable(&'static str),
}

/// Register the platform's hardware video decoders (call once at startup; repeated calls are
/// harmless). Streams they do not take, and every stream while hardware decoding is Off, keep
/// using FilmCraft's own decoders.
pub fn register() -> Availability {
    #[cfg(target_os = "macos")]
    {
        filmcraft_codecs::register_video_decoder(videotoolbox_factory);
        Availability::Available("VideoToolbox")
    }
    #[cfg(target_os = "linux")]
    {
        // Only the loader is checked here: the GPU is opened when the first stream needs it, so
        // starting up never waits for (or depends on) a GPU driver.
        match vulkan::loader_available() {
            Ok(()) => {
                filmcraft_codecs::register_video_decoder(vulkan_factory);
                Availability::Available("Vulkan Video")
            }
            Err(e) => {
                log::info!("hardware decoding unavailable: {e}");
                Availability::Unavailable("no Vulkan loader (libvulkan.so.1)")
            }
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Availability::Unavailable("no hardware video decoder for this system yet")
    }
}

/// Whether [`register`] has put a hardware decoder factory in front of our decoders.
pub fn registered() -> bool {
    #[cfg(target_os = "macos")]
    {
        filmcraft_codecs::video_decoder_registered(videotoolbox_factory)
    }
    #[cfg(target_os = "linux")]
    {
        filmcraft_codecs::video_decoder_registered(vulkan_factory)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Whether this system's hardware decoder takes the stream of `entry` (whatever the Hardware
/// decoding setting says): diagnostics and tests.
pub fn hardware_decoder_for(entry: &filmcraft_isobmff::SampleEntry) -> bool {
    #[cfg(target_os = "macos")]
    {
        filmcraft_codecs::hw::NalStreamInfo::from_entry(entry).and_then(|r| r.ok()).is_some_and(|info| videotoolbox::VtDecoder::new(info).is_ok())
    }
    #[cfg(target_os = "linux")]
    {
        vulkan_decoder(entry).is_some_and(|r| r.is_ok())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = entry;
        false
    }
}

/// The VideoToolbox factory: a [`HybridDecoder`] around [`videotoolbox::VtDecoder`] for `avcC` /
/// `hvcC` streams VideoToolbox can decode in hardware, `None` otherwise.
#[cfg(target_os = "macos")]
pub fn videotoolbox_factory(entry: &filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>> {
    if !filmcraft_codecs::hw::hardware_decoding() {
        return None;
    }
    let info = filmcraft_codecs::hw::NalStreamInfo::from_entry(entry)?.ok()?;
    match videotoolbox::VtDecoder::new(info.clone()) {
        Ok(vt) => Some(Ok(Box::new(HybridDecoder::new(Box::new(vt), entry.clone(), info)))),
        Err(why) => {
            log::info!("hardware decoding declined for {} video: {why}", entry.codec.name());
            filmcraft_codecs::hw::note_hw_declined();
            None
        }
    }
}

/// The Vulkan Video decoder for an `avcC` stream: `None` for other codecs, an error when the GPU
/// does not take the stream.
#[cfg(target_os = "linux")]
fn vulkan_decoder(
    entry: &filmcraft_isobmff::SampleEntry,
) -> Option<std::result::Result<(vulkan::VulkanH264Decoder, filmcraft_codecs::hw::NalStreamInfo), String>> {
    let filmcraft_isobmff::CodecConfig::Avc(avc) = &entry.codec else { return None };
    let info = match filmcraft_codecs::hw::NalStreamInfo::from_entry(entry)? {
        Ok(info) => info,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(vulkan::VulkanH264Decoder::new(info.clone(), avc.to_bytes()).map(|d| (d, info)))
}

/// The Vulkan Video factory (Linux): a [`HybridDecoder`] around [`vulkan::VulkanH264Decoder`] for
/// `avcC` streams the GPU decodes, `None` otherwise. Until one stream has been checked against the
/// software decoder in this process, the first pictures are decoded both ways and compared
/// ([`HybridDecoder::verifying`]); after a mismatch the factory declines for the rest of the run.
#[cfg(target_os = "linux")]
pub fn vulkan_factory(entry: &filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>> {
    if !filmcraft_codecs::hw::hardware_decoding() || !matches!(entry.codec, filmcraft_isobmff::CodecConfig::Avc(_)) {
        return None;
    }
    let check = match hybrid::verification(VULKAN_H264) {
        hybrid::Verification::Verified => None,
        hybrid::Verification::Claimed => Some(VULKAN_H264),
        hybrid::Verification::Busy | hybrid::Verification::Failed => return None,
    };
    let Some(made) = vulkan_decoder(entry) else {
        if let Some(key) = check {
            hybrid::release(key);
        }
        return None;
    };
    match made {
        Ok((d, info)) => {
            let hybrid = match check {
                Some(key) => HybridDecoder::verifying(Box::new(d), entry.clone(), info, key),
                None => Ok(HybridDecoder::new(Box::new(d), entry.clone(), info)),
            };
            match hybrid {
                Ok(h) => Some(Ok(Box::new(h))),
                Err(e) => {
                    log::info!("hardware decoding declined for H.264 video: {e}");
                    filmcraft_codecs::hw::note_hw_declined();
                    None
                }
            }
        }
        Err(why) => {
            if let Some(key) = check {
                hybrid::release(key);
            }
            log::info!("hardware decoding declined for H.264 video: {why}");
            filmcraft_codecs::hw::note_hw_declined();
            None
        }
    }
}

/// The first-use verification key of the Vulkan Video H.264 path ([`hybrid::verification`]).
#[cfg(target_os = "linux")]
pub const VULKAN_H264: &str = "Vulkan Video H.264";
