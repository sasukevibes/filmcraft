//! OS media integration (layer L5): hardware video decoding through the operating system's codecs.
//!
//! [`register`] puts the platform's hardware decoder factory in front of FilmCraft's own decoders
//! (`filmcraft_codecs::register_video_decoder`): VideoToolbox on macOS for H.264 (`avcC`) and HEVC
//! (`hvcC`) streams, 8- and 10-bit, 4:2:0 and 4:2:2; Vulkan Video on Linux for 8-bit 4:2:0
//! progressive H.264 and 8- / 10-bit 4:2:0 HEVC (Main, Main 10) ([`vulkan`],
//! docs/adr/0002-linux-vulkan-video.md). Elsewhere registration does nothing and reports
//! [`Availability::Unavailable`].
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
        vulkan_stream(entry).is_some_and(|(info, _)| vulkan_decoder(entry, &info).is_ok())
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

/// An `avcC` / `hvcC` stream's info and the first-use verification key of its Vulkan Video path;
/// `None` for other codecs and configuration records that do not parse (the software decoder
/// reports those).
#[cfg(target_os = "linux")]
fn vulkan_stream(entry: &filmcraft_isobmff::SampleEntry) -> Option<(filmcraft_codecs::hw::NalStreamInfo, &'static str)> {
    use filmcraft_codecs::hw::NalCodec;
    let info = filmcraft_codecs::hw::NalStreamInfo::from_entry(entry)?.ok()?;
    let key = match (info.codec, info.bit_depth_luma > 8, vulkan::scaling_in_use(&info)) {
        (NalCodec::H264, _, false) => VULKAN_H264,
        (NalCodec::H264, _, true) => VULKAN_H264_MATRICES,
        (NalCodec::Hevc, false, false) => VULKAN_HEVC,
        (NalCodec::Hevc, false, true) => VULKAN_HEVC_LISTS,
        (NalCodec::Hevc, true, false) => VULKAN_HEVC_10,
        (NalCodec::Hevc, true, true) => VULKAN_HEVC_10_LISTS,
    };
    Some((info, key))
}

/// The first-use verification key of the Vulkan Video path a stream would take (`None`: not an
/// `avcC` / `hvcC` stream): one per codec, bit depth and use of quantisation matrices, so each
/// kind is checked against the software decoder on its own and a mistake that only shows with one
/// of them cannot hide behind a stream that verified without it.
#[cfg(target_os = "linux")]
pub fn vulkan_verification_key(entry: &filmcraft_isobmff::SampleEntry) -> Option<&'static str> {
    vulkan_stream(entry).map(|(_, key)| key)
}

/// The Vulkan Video decoder for an `avcC` / `hvcC` stream, or why the GPU does not take it.
#[cfg(target_os = "linux")]
fn vulkan_decoder(
    entry: &filmcraft_isobmff::SampleEntry,
    info: &filmcraft_codecs::hw::NalStreamInfo,
) -> std::result::Result<Box<dyn filmcraft_codecs::VideoDecoder>, String> {
    match &entry.codec {
        filmcraft_isobmff::CodecConfig::Avc(c) => Ok(Box::new(vulkan::VulkanH264Decoder::new(info.clone(), c.to_bytes())?)),
        filmcraft_isobmff::CodecConfig::Hevc(c) => Ok(Box::new(vulkan::VulkanHevcDecoder::new(info.clone(), c.to_bytes())?)),
        _ => Err("not an H.264 or HEVC stream".into()),
    }
}

/// The Vulkan Video factory (Linux): a [`HybridDecoder`] around [`vulkan::VulkanH264Decoder`] /
/// [`vulkan::VulkanHevcDecoder`] for `avcC` / `hvcC` streams the GPU decodes, `None` otherwise.
/// Until one stream of a kind (H.264, HEVC 8-bit, HEVC 10-bit) has been checked against the
/// software decoder in this process, the first pictures are decoded both ways and compared
/// ([`HybridDecoder::verifying`]); after a mismatch the factory declines that kind for the rest of
/// the run.
#[cfg(target_os = "linux")]
pub fn vulkan_factory(entry: &filmcraft_isobmff::SampleEntry) -> Option<filmcraft_codecs::Result<Box<dyn filmcraft_codecs::VideoDecoder>>> {
    if !filmcraft_codecs::hw::hardware_decoding() {
        return None;
    }
    let (info, key) = vulkan_stream(entry)?;
    let check = match hybrid::verification(key) {
        hybrid::Verification::Verified => None,
        hybrid::Verification::Claimed => Some(key),
        hybrid::Verification::Busy | hybrid::Verification::Failed => return None,
    };
    let made = match vulkan_decoder(entry, &info) {
        Ok(d) => match check {
            // `verifying` gives the claim back itself when it fails
            Some(key) => HybridDecoder::verifying(d, entry.clone(), info, key).map_err(|e| e.to_string()),
            None => Ok(HybridDecoder::new(d, entry.clone(), info)),
        },
        Err(why) => {
            if let Some(key) = check {
                hybrid::release(key);
            }
            Err(why)
        }
    };
    match made {
        Ok(h) => Some(Ok(Box::new(h))),
        Err(why) => {
            log::info!("hardware decoding declined for {} video: {why}", entry.codec.name());
            filmcraft_codecs::hw::note_hw_declined();
            None
        }
    }
}

/// The first-use verification keys of the Vulkan Video paths ([`hybrid::verification`],
/// [`vulkan_verification_key`]).
#[cfg(target_os = "linux")]
pub const VULKAN_H264: &str = "Vulkan Video H.264";
#[cfg(target_os = "linux")]
pub const VULKAN_H264_MATRICES: &str = "Vulkan Video H.264 with scaling matrices";
#[cfg(target_os = "linux")]
pub const VULKAN_HEVC: &str = "Vulkan Video HEVC";
#[cfg(target_os = "linux")]
pub const VULKAN_HEVC_LISTS: &str = "Vulkan Video HEVC with scaling lists";
#[cfg(target_os = "linux")]
pub const VULKAN_HEVC_10: &str = "Vulkan Video HEVC 10-bit";
#[cfg(target_os = "linux")]
pub const VULKAN_HEVC_10_LISTS: &str = "Vulkan Video HEVC 10-bit with scaling lists";
