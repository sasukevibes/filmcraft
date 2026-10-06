//! First-use verification in [`HybridDecoder`] on every platform, with stand-in "hardware"
//! decoders (our own reference decoder, optionally damaging one picture or failing): matching
//! pictures verify the path after a few pictures; a damaged picture never reaches the caller (the
//! output is exactly the software decoder's), turns the path off for the run and is counted; a
//! hardware error or an early drop gives the claim back. Its own test binary: verdicts are
//! process-wide.

mod common;

use std::sync::Arc;

use common::*;
use filmcraft_codecs::hw::NalStreamInfo;
use filmcraft_codecs::{DecodedFrame, Result, VideoDecoder};
use filmcraft_frame::PixelData;
use filmcraft_platform::HybridDecoder;
use filmcraft_platform::hybrid::{self, Verification};

/// The reference decoder, damaging the picture output `damage` (0-based) or failing from sample
/// `fail_at` on.
struct StandIn {
    inner: Box<dyn VideoDecoder>,
    damage: Option<usize>,
    fail_at: Option<usize>,
    outputs: usize,
    fed: usize,
}

impl StandIn {
    fn new(s: &Stream, damage: Option<usize>, fail_at: Option<usize>) -> Self {
        Self { inner: filmcraft_codecs::reference_video_decoder(&s.entry).unwrap(), damage, fail_at, outputs: 0, fed: 0 }
    }
    fn mark(&mut self, mut frames: Vec<DecodedFrame>) -> Vec<DecodedFrame> {
        for f in &mut frames {
            if Some(self.outputs) == self.damage
                && let PixelData::Yuv8 { planes, .. } = &mut f.frame.data
            {
                let mut y = planes[0].as_ref().clone();
                y[0] ^= 1;
                planes[0] = Arc::new(y);
            }
            self.outputs += 1;
        }
        frames
    }
}

impl VideoDecoder for StandIn {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        self.fed += 1;
        if self.fail_at.is_some_and(|n| self.fed > n) {
            return Err(filmcraft_codecs::CodecError::Decode("session lost (test)".into()));
        }
        let frames = self.inner.decode(sample, pts)?;
        Ok(self.mark(frames))
    }
    fn flush(&mut self) -> Vec<DecodedFrame> {
        let frames = self.inner.flush();
        self.mark(frames)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn name(&self) -> &str {
        "stand-in hardware"
    }
}

const KEY: &str = "test path";

fn verifying(s: &Stream, hw: StandIn) -> HybridDecoder {
    let info = NalStreamInfo::from_entry(&s.entry).unwrap().unwrap();
    assert_eq!(hybrid::verification(KEY), Verification::Claimed);
    assert_eq!(hybrid::verification(KEY), Verification::Busy, "one verifier at a time");
    HybridDecoder::verifying(Box::new(hw), s.entry.clone(), info, KEY).unwrap()
}

/// One test function: the verdicts are process-wide state.
#[test]
fn first_use_verification() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(path) = named(&ff, "h264_high.mp4") else { return };
    let s = read_stream(&path);
    let reference = decode_all(filmcraft_codecs::software_video_decoder(&s.entry).unwrap().as_mut(), &s.samples);

    // matching pictures: verified after the first few, hardware throughout
    hybrid::reset_verification();
    let mut d = verifying(&s, StandIn::new(&s, None, None));
    assert!(d.is_verifying());
    let out = decode_all(&mut d, &s.samples);
    assert_same("verified", &out, &reference);
    assert!(d.is_hardware() && !d.is_verifying(), "verified and still in hardware");
    assert_eq!(hybrid::verification(KEY), Verification::Verified);

    // a damaged picture inside the check: the caller gets the software picture, the path is off
    for damage in [0, 3] {
        hybrid::reset_verification();
        let before = filmcraft_codecs::hw::hw_stats().mismatches;
        let mut d = verifying(&s, StandIn::new(&s, Some(damage), None));
        let out = decode_all(&mut d, &s.samples);
        assert_same(&format!("damaged picture {damage}"), &out, &reference);
        assert!(!d.is_hardware(), "switched to software");
        assert_eq!(hybrid::verification(KEY), Verification::Failed);
        assert_eq!(filmcraft_codecs::hw::hw_stats().mismatches, before + 1, "counted");
        // seeking keeps the instance in software with exact output
        d.reset();
        assert_same("after a reset", &decode_all(&mut d, &s.samples), &reference);
    }

    // a hardware error during the check: software output, the claim is given back
    hybrid::reset_verification();
    let mut d = verifying(&s, StandIn::new(&s, None, Some(2)));
    let out = decode_all(&mut d, &s.samples);
    assert_same("failing during the check", &out, &reference);
    assert!(!d.is_hardware());
    assert_eq!(hybrid::verification(KEY), Verification::Claimed, "free to verify again");

    // dropped before the check finished: the claim is given back
    hybrid::reset_verification();
    let mut d = verifying(&s, StandIn::new(&s, None, None));
    d.decode(&s.samples[0].0, s.samples[0].1).unwrap();
    assert!(d.is_verifying());
    drop(d);
    assert_eq!(hybrid::verification(KEY), Verification::Claimed);

    // the check continues across seeks
    hybrid::reset_verification();
    let mut d = verifying(&s, StandIn::new(&s, None, None));
    let k = (1..s.samples.len()).find(|&i| s.sync[i]).unwrap();
    d.decode(&s.samples[0].0, s.samples[0].1).unwrap();
    d.reset();
    let tail = &s.samples[k..];
    let expect = decode_all(filmcraft_codecs::software_video_decoder(&s.entry).unwrap().as_mut(), tail);
    assert_same("seek during the check", &decode_all(&mut d, tail), &expect);
    assert_eq!(hybrid::verification(KEY), Verification::Verified);
    hybrid::reset_verification();
}
