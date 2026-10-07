//! Vulkan Video decoding against our software decoders (Linux, docs/adr/0002-linux-vulkan-video.md).
//!
//! On a machine with a Vulkan Video device: every picture of libx264 H.264 fixtures (High with a
//! B-pyramid and cropping, Main with four references and weighted prediction, Baseline with four
//! slices, custom scaling matrices, 1080p) and libx265 HEVC fixtures (Main with open GOPs, CRA and
//! RASL pictures; Main 10; three slices with default scaling lists; custom scaling lists; transform
//! skip and transquant bypass at 10 bits; 1080p) must be identical to the software decoder's
//! (planes, pts order, colour, pixel aspect), also after `reset` + reseek to every later sync sample
//! and a mid-stream `flush`; a forced mid-stream failure continues in software with the software
//! output; the factory verifies each kind of stream on first use; damaged samples give errors or
//! fall back, never a crash or a hang.
//!
//! Without one (CI, the cloud): H.264 and HEVC streams go to the software decoders and are counted
//! as declined. Fixtures are made with ffmpeg (generator only).
#![cfg(target_os = "linux")]

mod common;

use std::time::Duration;

use common::*;
use filmcraft_codecs::VideoDecoder;
use filmcraft_codecs::hw::{NalCodec, NalStreamInfo};
use filmcraft_isobmff::CodecConfig;
use filmcraft_platform::HybridDecoder;
use filmcraft_platform::hybrid::{self, Verification};
use filmcraft_platform::vulkan::{VulkanH264Decoder, VulkanHevcDecoder};

const GOP: &str = "keyint=24:min-keyint=24:scenecut=0";

/// An HM-format scaling list file for libx265 (raster-order matrices, the DC values of the 16x16
/// and 32x32 lists, some below 8).
fn scaling_list_file() -> std::path::PathBuf {
    let mut out = String::new();
    for (size, n) in [("4X4", 4u32), ("8X8", 8), ("16X16", 8), ("32X32", 8)] {
        for (k, mode) in ["INTRA", "INTER"].into_iter().enumerate() {
            let comps: &[&str] = if size == "32X32" { &["LUMA"] } else { &["LUMA", "CHROMAU", "CHROMAV"] };
            for (c, comp) in comps.iter().enumerate() {
                let seed = (k * 3 + c) as u32;
                out += &format!("{mode}{size}_{comp} =\n");
                for y in 0..n {
                    out += &(0..n).map(|x| format!("{},", 8 + 3 * seed + 2 * (x + y) + (x * 7 + y * 3 + seed) % 5)).collect::<String>();
                    out += "\n";
                }
                if matches!(size, "16X16" | "32X32") {
                    out += &format!("{mode}{size}_{comp}_DC =\n{},\n", 5 + 6 * seed);
                }
            }
        }
    }
    let path = dir().join("vk_hevc_lists.txt");
    std::fs::write(&path, out).unwrap();
    path
}

/// Fixtures beyond `common::FIXTURES`: (file, ffmpeg arguments).
fn fixtures() -> Vec<(&'static str, Vec<String>)> {
    let src = |size: &str, secs: u32| format!("testsrc2=s={size}:r=24:d={secs},noise=alls=12:allf=t");
    let strings = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
    let x264 = |size: &str, secs: u32, profile: &str, params: String| -> Vec<String> {
        strings(&[
            "-f",
            "lavfi",
            "-i",
            &src(size, secs),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-profile:v",
            profile,
            "-x264-params",
            &params,
            "-pix_fmt",
            "yuv420p",
        ])
    };
    let x265 = |size: &str, secs: u32, pix_fmt: &str, params: String| -> Vec<String> {
        let params = format!("log-level=error:{params}");
        strings(&["-f", "lavfi", "-i", &src(size, secs), "-c:v", "libx265", "-preset", "fast", "-tag:v", "hvc1", "-x265-params", &params, "-pix_fmt", pix_fmt])
    };
    let lists = scaling_list_file();
    vec![
        ("vk_h264_main_ref4.mp4", x264("640x360", 3, "main", format!("ref=4:bframes=2:weightp=2:{GOP}"))),
        ("vk_h264_baseline_slices.mp4", x264("480x270", 3, "baseline", format!("slices=4:ref=2:{GOP}"))),
        ("vk_h264_high_cqm.mp4", x264("640x360", 3, "high", format!("cqm=jvt:bframes=3:b-pyramid=normal:weightb=1:{GOP}"))),
        ("vk_h264_1080p.mp4", x264("1920x1080", 2, "high", format!("bframes=3:{GOP}"))),
        (
            "vk_hevc_slices_default_lists.mp4",
            x265("638x358", 3, "yuv420p", format!("slices=3:bframes=3:b-pyramid=1:ref=3:weightb=1:scaling-list=default:{GOP}")),
        ),
        ("vk_hevc_custom_lists.mp4", x265("640x360", 3, "yuv420p", format!("bframes=2:scaling-list={}:{GOP}", lists.display()))),
        (
            "vk_hevc_main10_tools.mp4",
            x265(
                "640x360",
                3,
                "yuv420p10le",
                format!("bframes=2:amp=1:rect=1:tskip=1:cu-lossless=1:constrained-intra=1:signhide=0:deblock=-2,1:cbqpoffs=2:crqpoffs=-1:{GOP}"),
            ),
        ),
        ("vk_hevc_1080p.mp4", x265("1920x1080", 2, "yuv420p", format!("bframes=4:open-gop=1:{GOP}"))),
    ]
}

/// Every fixture as a stream (generated on first use).
fn streams(ff: &std::path::Path) -> Vec<(String, Stream)> {
    let mut out = Vec::new();
    for name in ["h264_high.mp4", "hevc_main.mp4", "hevc_main10.mp4"] {
        if let Some(p) = named(ff, name) {
            out.push((name.to_string(), read_stream(&p)));
        }
    }
    for (name, args) in fixtures() {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        if let Some(p) = fixture(ff, name, &args) {
            out.push((name.to_string(), read_stream(&p)));
        }
    }
    assert_eq!(out.len(), 11, "all fixtures generated");
    out
}

/// The device name, or `None` (skip) when this machine has no Vulkan Video device.
fn gpu() -> Option<String> {
    match filmcraft_platform::vulkan::probe() {
        Ok(name) => {
            eprintln!("Vulkan Video device: {name}");
            Some(name)
        }
        Err(e) => {
            eprintln!("SKIPPED: no Vulkan Video device ({e})");
            None
        }
    }
}

fn info(s: &Stream) -> NalStreamInfo {
    NalStreamInfo::from_entry(&s.entry).unwrap().unwrap()
}

/// The stream's configuration record (`avcC` / `hvcC` payload).
fn record(s: &Stream) -> Vec<u8> {
    match &s.entry.codec {
        CodecConfig::Avc(a) => a.to_bytes(),
        CodecConfig::Hevc(c) => c.to_bytes(),
        _ => panic!("not H.264 / HEVC"),
    }
}

/// The Vulkan Video decoder for a stream's info and configuration record; `fail_after` arms its
/// test hook.
fn vulkan(info: NalStreamInfo, record: Vec<u8>, fail_after: Option<u64>) -> Result<Box<dyn VideoDecoder>, String> {
    Ok(match info.codec {
        NalCodec::H264 => {
            let mut d = VulkanH264Decoder::new(info, record)?;
            if let Some(n) = fail_after {
                d.fail_after(n);
            }
            Box::new(d)
        }
        NalCodec::Hevc => {
            let mut d = VulkanHevcDecoder::new(info, record)?;
            if let Some(n) = fail_after {
                d.fail_after(n);
            }
            Box::new(d)
        }
    })
}

fn hardware(name: &str, s: &Stream) -> Box<dyn VideoDecoder> {
    vulkan(info(s), record(s), None).unwrap_or_else(|e| panic!("{name}: the GPU declined the stream: {e}"))
}

fn software(s: &Stream) -> Box<dyn VideoDecoder> {
    filmcraft_codecs::software_video_decoder(&s.entry).unwrap()
}

#[test]
fn without_a_video_device_streams_go_to_software() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    if filmcraft_platform::vulkan::probe().is_ok() {
        eprintln!("SKIPPED: this machine has a Vulkan Video device");
        return;
    }
    let available = filmcraft_platform::register();
    assert_eq!(filmcraft_platform::registered(), matches!(available, filmcraft_platform::Availability::Available(_)));
    filmcraft_codecs::hw::set_hardware_decoding(true);
    for (name, software_name) in [("h264_high.mp4", "FilmCraft H.264"), ("hevc_main.mp4", "FilmCraft HEVC"), ("hevc_main10.mp4", "FilmCraft HEVC")] {
        let Some(path) = named(&ff, name) else { return };
        let s = read_stream(&path);
        for _ in 0..3 {
            let before = filmcraft_codecs::hw::hw_stats().declined;
            let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
            assert_eq!(d.name(), software_name, "{name}");
            if filmcraft_platform::registered() {
                assert!(filmcraft_codecs::hw::hw_stats().declined > before, "{name}: declined and counted");
            }
            assert!(!decode_all(d.as_mut(), &s.samples[..10]).is_empty(), "{name}");
        }
        assert!(!filmcraft_platform::hardware_decoder_for(&s.entry), "{name}");
        assert!(vulkan(info(&s), record(&s), None).is_err(), "{name}");
        // a declined stream never leaves its first-use check claimed
        let key = filmcraft_platform::vulkan_verification_key(&s.entry).unwrap();
        assert_ne!(hybrid::verification(key), Verification::Busy, "{name}");
        hybrid::release(key);
    }
}

/// Each kind of stream has its own first-use check: codec, bit depth and quantisation matrices
/// (no GPU needed).
#[test]
fn verification_keys_follow_the_stream_kind() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let fixtures = fixtures();
    let path = |name: &str| match fixtures.iter().find(|f| f.0 == name) {
        Some((_, args)) => fixture(&ff, name, &args.iter().map(String::as_str).collect::<Vec<_>>()),
        None => named(&ff, name),
    };
    for (name, key) in [
        ("h264_high.mp4", filmcraft_platform::VULKAN_H264),
        ("vk_h264_high_cqm.mp4", filmcraft_platform::VULKAN_H264_MATRICES),
        ("hevc_main.mp4", filmcraft_platform::VULKAN_HEVC),
        ("hevc_main10.mp4", filmcraft_platform::VULKAN_HEVC_10),
        ("vk_hevc_slices_default_lists.mp4", filmcraft_platform::VULKAN_HEVC_LISTS),
        ("vk_hevc_custom_lists.mp4", filmcraft_platform::VULKAN_HEVC_LISTS),
        ("vk_hevc_main10_tools.mp4", filmcraft_platform::VULKAN_HEVC_10),
    ] {
        let s = read_stream(&path(name).unwrap_or_else(|| panic!("{name}: fixture")));
        assert_eq!(filmcraft_platform::vulkan_verification_key(&s.entry), Some(key), "{name}");
    }
}

#[test]
fn bit_exact_with_the_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    for (name, s) in streams(&ff) {
        let mut hw = hardware(&name, &s);
        let mut sw = software(&s);
        assert!(hw.name().starts_with("Vulkan Video"), "{name}: {}", hw.name());
        let a = decode_all(hw.as_mut(), &s.samples);
        let b = decode_all(sw.as_mut(), &s.samples);
        assert!(a.len() >= 48, "{name}: {} pictures", a.len());
        assert_same(&name, &a, &b);

        // reset + reseek to each later sync sample, decode a stretch, flush mid-stream
        let syncs: Vec<usize> = (1..s.samples.len()).filter(|&i| s.sync[i]).collect();
        assert!(!syncs.is_empty(), "{name}: more than one GOP");
        for &k in syncs.iter().rev() {
            let end = (k + 17).min(s.samples.len());
            hw.reset();
            sw.reset();
            let a = decode_all(hw.as_mut(), &s.samples[k..end]);
            let b = decode_all(sw.as_mut(), &s.samples[k..end]);
            assert!(!a.is_empty(), "{name}: pictures after seeking to {k}");
            assert_same(&format!("{name} from sample {k}"), &a, &b);
        }
        // back to the start after seeking around: the whole stream again
        hw.reset();
        sw.reset();
        assert_same(&format!("{name} after resets"), &decode_all(hw.as_mut(), &s.samples), &decode_all(sw.as_mut(), &s.samples));
        for (smp, _) in &s.samples {
            assert_eq!(hw.is_random_access(smp), sw.is_random_access(smp), "{name}");
            assert_eq!(hw.is_disposable(smp), sw.is_disposable(smp), "{name}");
        }
    }
}

#[test]
fn mid_stream_failure_continues_in_software() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    // H.264, HEVC with open GOPs, HEVC Main 10
    for (name, s) in streams(&ff).into_iter().take(3) {
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        let k = (1..s.samples.len()).find(|&i| s.sync[i]).unwrap();
        for fail_at in [0, 5, k, k + 1, k + 9] {
            let vk = vulkan(info(&s), record(&s), Some(fail_at as u64)).unwrap_or_else(|e| panic!("{name}: the GPU declined the stream: {e}"));
            let mut d = HybridDecoder::new(vk, s.entry.clone(), info(&s));
            let before = filmcraft_codecs::hw::hw_stats().fallbacks;
            let out = decode_all(&mut d, &s.samples);
            assert!(!d.is_hardware(), "{name}: switched to software");
            assert!(filmcraft_codecs::hw::hw_stats().fallbacks > before, "{name}: fallback counted");
            assert_same(&format!("{name} failing at sample {fail_at}"), &out, &reference);
        }
    }
}

/// The registered factory hands H.264 and HEVC to the GPU, checks the first pictures of each kind
/// of stream against the software decoder and then trusts that kind (only that kind); Off gives
/// the software decoders.
#[test]
fn factory_verifies_on_first_use() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    assert_eq!(filmcraft_platform::register(), filmcraft_platform::Availability::Available("Vulkan Video"));
    hybrid::reset_verification();
    filmcraft_codecs::hw::set_hardware_decoding(true);
    for (name, hardware_name, software_name) in [
        ("h264_high.mp4", "Vulkan Video H.264", "FilmCraft H.264"),
        ("hevc_main.mp4", "Vulkan Video HEVC", "FilmCraft HEVC"),
        ("hevc_main10.mp4", "Vulkan Video HEVC", "FilmCraft HEVC"),
    ] {
        let Some(path) = named(&ff, name) else { return };
        let s = read_stream(&path);
        let key = filmcraft_platform::vulkan_verification_key(&s.entry).unwrap();
        assert!(filmcraft_platform::hardware_decoder_for(&s.entry), "{name}");
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
        assert!(d.name().starts_with(hardware_name), "{name}: {}", d.name());
        let out = decode_all(d.as_mut(), &s.samples);
        assert_same(&format!("{name}: factory decoder"), &out, &reference);
        assert_eq!(hybrid::verification(key), Verification::Verified, "{name}: verified on first use");
        let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
        assert!(d.name().starts_with(hardware_name), "{name}");
        assert_same(&format!("{name}: verified path"), &decode_all(d.as_mut(), &s.samples), &reference);
        filmcraft_codecs::hw::set_hardware_decoding(false);
        assert_eq!(filmcraft_codecs::make_video_decoder(&s.entry).unwrap().name(), software_name, "{name}");
        filmcraft_codecs::hw::set_hardware_decoding(true);
    }
    // the kinds not seen yet are still to be checked
    for key in [filmcraft_platform::VULKAN_H264_MATRICES, filmcraft_platform::VULKAN_HEVC_LISTS, filmcraft_platform::VULKAN_HEVC_10_LISTS] {
        assert_eq!(hybrid::verification(key), Verification::Claimed, "{key}");
        hybrid::release(key);
    }
}

/// Run `f` on a thread; fail if it panics or takes longer than `limit`.
fn bounded(what: &str, limit: Duration, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let _ = tx.send(r.is_ok());
    });
    match rx.recv_timeout(limit) {
        Ok(true) => {}
        Ok(false) => panic!("{what}: panicked"),
        Err(_) => panic!("{what}: hung"),
    }
}

/// Bit flips, truncation and corrupt length prefixes in samples, and damaged parameter sets: the
/// GPU decoder returns errors (or pictures), never crashes or hangs; through the hybrid the
/// output continues in software.
#[test]
fn damaged_input_never_crashes_or_hangs() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    for name in ["h264_high.mp4", "hevc_main.mp4"] {
        let Some(path) = named(&ff, name) else { return };
        damaged(&read_stream(&path));
    }
}

fn damaged(s: &Stream) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for round in 0..24 {
        let mut samples = s.samples.clone();
        for (i, (smp, _)) in samples.iter_mut().enumerate() {
            if smp.is_empty() || (i + round) % 4 != 0 {
                continue;
            }
            match xorshift(&mut seed) % 3 {
                0 => {
                    for _ in 0..1 + round % 6 {
                        let k = (xorshift(&mut seed) as usize) % smp.len();
                        smp[k] ^= 1 << (xorshift(&mut seed) % 8);
                    }
                }
                1 => smp.truncate((xorshift(&mut seed) as usize) % smp.len()),
                _ => smp[0] ^= 0x7f,
            }
        }
        let (stream_info, record, entry) = (info(s), record(s), s.entry.clone());
        bounded(&format!("damaged samples, round {round}"), Duration::from_secs(60), move || {
            if let Ok(mut d) = vulkan(stream_info.clone(), record, None) {
                for (smp, pts) in &samples {
                    let _ = d.decode(smp, *pts);
                }
                d.flush();
                let mut h = HybridDecoder::new(d, entry, stream_info);
                h.reset();
                for (smp, pts) in &samples {
                    let _ = h.decode(smp, *pts);
                }
                h.flush();
            }
        });
    }
    // damaged parameter sets: declined or decoded, never a crash
    for round in 0..16 {
        let mut stream_info = info(s);
        let mut record = record(s);
        let k = 6 + (xorshift(&mut seed) as usize) % (record.len() - 6);
        record[k] ^= 1 << (round % 8);
        if let Some(p) = stream_info.parameter_sets.get_mut(round % 3) {
            let k = (xorshift(&mut seed) as usize) % p.len();
            p[k] ^= 1 << (round % 8);
        }
        let samples = s.samples.clone();
        bounded(&format!("damaged parameter sets, round {round}"), Duration::from_secs(60), move || {
            if let Ok(mut d) = vulkan(stream_info, record, None) {
                for (smp, pts) in samples.iter().take(30) {
                    let _ = d.decode(smp, *pts);
                }
                d.flush();
            }
        });
    }
}
