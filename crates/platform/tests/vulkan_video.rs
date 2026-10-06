//! Vulkan Video decoding against our software decoder (Linux, docs/adr/0002-linux-vulkan-video.md).
//!
//! On a machine with a Vulkan Video device: every picture of libx264 H.264 fixtures (High with a
//! B-pyramid and cropping, Main with four references and weighted prediction, Baseline with four
//! slices, custom scaling matrices, 1080p) must be identical to the software decoder's (planes,
//! pts order, colour, pixel aspect), also after `reset` + reseek to every later sync sample and a
//! mid-stream `flush`; a forced mid-stream failure continues in software with the software output;
//! the factory verifies the path on first use; damaged samples give errors or fall back, never a
//! crash or a hang.
//!
//! Without one (CI, the cloud): H.264 streams go to the software decoder and are counted as
//! declined. Fixtures are made with ffmpeg (generator only).
#![cfg(target_os = "linux")]

mod common;

use std::time::Duration;

use common::*;
use filmcraft_codecs::VideoDecoder;
use filmcraft_codecs::hw::NalStreamInfo;
use filmcraft_isobmff::CodecConfig;
use filmcraft_platform::HybridDecoder;
use filmcraft_platform::hybrid::{self, Verification};
use filmcraft_platform::vulkan::VulkanH264Decoder;

const GOP: &str = "keyint=24:min-keyint=24:scenecut=0";

/// H.264 fixtures beyond `common::FIXTURES`' `h264_high.mp4`: (file, ffmpeg arguments).
fn fixtures() -> Vec<(&'static str, Vec<String>)> {
    let src = |size: &str, secs: u32| format!("testsrc2=s={size}:r=24:d={secs},noise=alls=12:allf=t");
    let args = |size: &str, secs: u32, profile: &str, params: String| -> Vec<String> {
        ["-f", "lavfi", "-i", &src(size, secs), "-c:v", "libx264", "-preset", "fast", "-profile:v", profile, "-x264-params", &params, "-pix_fmt", "yuv420p"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    vec![
        ("vk_h264_main_ref4.mp4", args("640x360", 3, "main", format!("ref=4:bframes=2:weightp=2:{GOP}"))),
        ("vk_h264_baseline_slices.mp4", args("480x270", 3, "baseline", format!("slices=4:ref=2:{GOP}"))),
        ("vk_h264_high_cqm.mp4", args("640x360", 3, "high", format!("cqm=jvt:bframes=3:b-pyramid=normal:weightb=1:{GOP}"))),
        ("vk_h264_1080p.mp4", args("1920x1080", 2, "high", format!("bframes=3:{GOP}"))),
    ]
}

/// Every H.264 fixture as a stream (generated on first use).
fn streams(ff: &std::path::Path) -> Vec<(String, Stream)> {
    let mut out = Vec::new();
    if let Some(p) = named(ff, "h264_high.mp4") {
        out.push(("h264_high.mp4".to_string(), read_stream(&p)));
    }
    for (name, args) in fixtures() {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        if let Some(p) = fixture(ff, name, &args) {
            out.push((name.to_string(), read_stream(&p)));
        }
    }
    assert_eq!(out.len(), 5, "all fixtures generated");
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

fn avcc(s: &Stream) -> Vec<u8> {
    let CodecConfig::Avc(a) = &s.entry.codec else { panic!("not H.264") };
    a.to_bytes()
}

fn hardware(name: &str, s: &Stream) -> VulkanH264Decoder {
    VulkanH264Decoder::new(info(s), avcc(s)).unwrap_or_else(|e| panic!("{name}: the GPU declined the stream: {e}"))
}

fn software(s: &Stream) -> Box<dyn VideoDecoder> {
    filmcraft_codecs::software_video_decoder(&s.entry).unwrap()
}

#[test]
fn without_a_video_device_h264_goes_to_software() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    if filmcraft_platform::vulkan::probe().is_ok() {
        eprintln!("SKIPPED: this machine has a Vulkan Video device");
        return;
    }
    let Some(path) = named(&ff, "h264_high.mp4") else { return };
    let s = read_stream(&path);
    let available = filmcraft_platform::register();
    assert_eq!(filmcraft_platform::registered(), matches!(available, filmcraft_platform::Availability::Available(_)));
    filmcraft_codecs::hw::set_hardware_decoding(true);
    for _ in 0..3 {
        let before = filmcraft_codecs::hw::hw_stats().declined;
        let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
        assert_eq!(d.name(), "FilmCraft H.264");
        if filmcraft_platform::registered() {
            assert!(filmcraft_codecs::hw::hw_stats().declined > before, "declined and counted");
        }
        assert!(!decode_all(d.as_mut(), &s.samples[..10]).is_empty());
    }
    assert!(!filmcraft_platform::hardware_decoder_for(&s.entry));
    assert!(VulkanH264Decoder::new(info(&s), avcc(&s)).is_err());
    // a declined stream never leaves the first-use check claimed
    assert_ne!(hybrid::verification(filmcraft_platform::VULKAN_H264), Verification::Busy);
}

#[test]
fn bit_exact_with_the_software_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    for (name, s) in streams(&ff) {
        let mut hw = hardware(&name, &s);
        let mut sw = software(&s);
        let a = decode_all(&mut hw, &s.samples);
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
            let a = decode_all(&mut hw, &s.samples[k..end]);
            let b = decode_all(sw.as_mut(), &s.samples[k..end]);
            assert!(!a.is_empty(), "{name}: pictures after seeking to {k}");
            assert_same(&format!("{name} from sample {k}"), &a, &b);
        }
        // back to the start after seeking around: the whole stream again
        hw.reset();
        sw.reset();
        assert_same(&format!("{name} after resets"), &decode_all(&mut hw, &s.samples), &decode_all(sw.as_mut(), &s.samples));
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
    for (name, s) in streams(&ff).into_iter().take(2) {
        let reference = decode_all(software(&s).as_mut(), &s.samples);
        let k = (1..s.samples.len()).find(|&i| s.sync[i]).unwrap();
        for fail_at in [0, 5, k, k + 1, k + 9] {
            let mut vk = hardware(&name, &s);
            vk.fail_after(fail_at as u64);
            let mut d = HybridDecoder::new(Box::new(vk), s.entry.clone(), info(&s));
            let before = filmcraft_codecs::hw::hw_stats().fallbacks;
            let out = decode_all(&mut d, &s.samples);
            assert!(!d.is_hardware(), "{name}: switched to software");
            assert!(filmcraft_codecs::hw::hw_stats().fallbacks > before, "{name}: fallback counted");
            assert_same(&format!("{name} failing at sample {fail_at}"), &out, &reference);
        }
    }
}

/// The registered factory hands H.264 to the GPU, checks the first pictures against the
/// software decoder and then trusts the path; Off gives the software decoder.
#[test]
fn factory_verifies_on_first_use() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(_) = gpu() else { return };
    let Some(path) = named(&ff, "h264_high.mp4") else { return };
    let s = read_stream(&path);
    assert_eq!(filmcraft_platform::register(), filmcraft_platform::Availability::Available("Vulkan Video"));
    assert!(filmcraft_platform::hardware_decoder_for(&s.entry));
    hybrid::reset_verification();
    filmcraft_codecs::hw::set_hardware_decoding(true);
    let reference = decode_all(software(&s).as_mut(), &s.samples);
    let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
    assert!(d.name().starts_with("Vulkan Video H.264"), "{}", d.name());
    let out = decode_all(d.as_mut(), &s.samples);
    assert_same("factory decoder", &out, &reference);
    assert_eq!(hybrid::verification(filmcraft_platform::VULKAN_H264), Verification::Verified, "verified on first use");
    let mut d = filmcraft_codecs::make_video_decoder(&s.entry).unwrap();
    assert!(d.name().starts_with("Vulkan Video H.264"));
    assert_same("verified path", &decode_all(d.as_mut(), &s.samples), &reference);
    filmcraft_codecs::hw::set_hardware_decoding(false);
    assert_eq!(filmcraft_codecs::make_video_decoder(&s.entry).unwrap().name(), "FilmCraft H.264");
    filmcraft_codecs::hw::set_hardware_decoding(true);
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
    let Some(path) = named(&ff, "h264_high.mp4") else { return };
    let s = read_stream(&path);
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
        let (stream_info, record, entry) = (info(&s), avcc(&s), s.entry.clone());
        bounded(&format!("damaged samples, round {round}"), Duration::from_secs(60), move || {
            if let Ok(mut d) = VulkanH264Decoder::new(stream_info.clone(), record) {
                for (smp, pts) in &samples {
                    let _ = d.decode(smp, *pts);
                }
                d.flush();
                let mut h = HybridDecoder::new(Box::new(d), entry, stream_info);
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
        let mut stream_info = info(&s);
        let mut record = avcc(&s);
        let k = 6 + (xorshift(&mut seed) as usize) % (record.len() - 6);
        record[k] ^= 1 << (round % 8);
        if let Some(p) = stream_info.parameter_sets.first_mut() {
            let k = (xorshift(&mut seed) as usize) % p.len();
            p[k] ^= 1 << (round % 8);
        }
        let samples = s.samples.clone();
        bounded(&format!("damaged parameter sets, round {round}"), Duration::from_secs(60), move || {
            if let Ok(mut d) = VulkanH264Decoder::new(stream_info, record) {
                for (smp, pts) in samples.iter().take(30) {
                    let _ = d.decode(smp, *pts);
                }
                d.flush();
            }
        });
    }
}
