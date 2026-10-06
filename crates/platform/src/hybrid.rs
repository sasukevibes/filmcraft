//! [`HybridDecoder`]: a hardware decoder that turns into the software decoder for the same stream
//! when it fails mid-stream (decode error, session lost to a GPU change or sleep, an in-band
//! parameter-set change it was not set up for), without the caller noticing.
//!
//! The hybrid keeps the samples fed since the last point decoding can restart from (the run's
//! first sample, then every IDR / IRAP access unit). On a failure it builds the software decoder
//! ([`filmcraft_codecs::software_video_decoder`], skipping registered factories), replays those
//! samples through it and carries on, dropping pictures the hardware already returned and keeping
//! the ones it had decoded but not yet returned, so the output is the software decoder's own
//! sequence. That replay log is bounded; past the bound a failure is reported as an error once
//! and the instance continues in software from the next seek ([`VideoDecoder::reset`]).
//!
//! A hardware path that has not been checked on real hardware yet can be verified on first use
//! ([`HybridDecoder::verifying`]): the first decoder of that path in the process runs our
//! reference decoder in lockstep and compares every picture of the first calls bit for bit. On a
//! match the path is trusted for the rest of the run; on a mismatch the instance continues with the
//! (already synchronised) reference decoder and the path is turned off for the rest of the run, so
//! a wrong hardware picture never reaches the caller.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock, PoisonError};

use filmcraft_codecs::hw::NalStreamInfo;
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};
use filmcraft_isobmff::SampleEntry;

/// Replay log bounds: samples and bytes since the last restart point.
const LOG_MAX_SAMPLES: usize = 600;
const LOG_MAX_BYTES: usize = 256 << 20;
/// Presentation times remembered as returned (older ones are forgotten first).
const EMITTED_MAX: usize = 4096;
/// Pictures compared before a hardware path counts as verified.
const VERIFY_PICTURES: usize = 8;

/// Where a hardware path's first-use verification stands ([`verification`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verification {
    /// Verified in this process: use it.
    Verified,
    /// Not verified yet and now claimed by the caller, who verifies it
    /// ([`HybridDecoder::verifying`]) or gives the claim back ([`release`]).
    Claimed,
    /// Another decoder is verifying it right now: use the software decoder meanwhile.
    Busy,
    /// Its pictures differed from ours: off for the rest of the run.
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Verifying,
    Verified,
    Failed,
}

fn states() -> std::sync::MutexGuard<'static, HashMap<&'static str, State>> {
    static S: OnceLock<Mutex<HashMap<&'static str, State>>> = OnceLock::new();
    S.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner)
}

/// The state of hardware path `key`; an unverified path is claimed by the caller.
pub fn verification(key: &'static str) -> Verification {
    let mut m = states();
    match m.get(key) {
        Some(State::Verified) => Verification::Verified,
        Some(State::Failed) => Verification::Failed,
        Some(State::Verifying) => Verification::Busy,
        None => {
            m.insert(key, State::Verifying);
            Verification::Claimed
        }
    }
}

/// Give back a claim without a verdict (the stream could not be checked); a later decoder verifies.
pub fn release(key: &'static str) {
    let mut m = states();
    if m.get(key) == Some(&State::Verifying) {
        m.remove(key);
    }
}

fn settle(key: &'static str, ok: bool) {
    states().insert(key, if ok { State::Verified } else { State::Failed });
}

/// Forget every verdict (tests).
pub fn reset_verification() {
    states().clear();
}

/// A first-use check in progress: the reference decoder in lockstep with the hardware.
struct Verify {
    key: &'static str,
    reference: Box<dyn VideoDecoder>,
    /// Pictures still to compare.
    left: usize,
}

/// Bit-exact equality of two decoders' outputs (pictures, order, pts, colour, aspect).
fn same_frames(a: &[DecodedFrame], b: &[DecodedFrame]) -> bool {
    use filmcraft_codecs::DecodedFrame as F;
    fn same(x: &F, y: &F) -> bool {
        let (p, q) = (&x.frame, &y.frame);
        let pixels = match (&p.data, &q.data) {
            (filmcraft_frame::PixelData::Yuv8 { planes: a, chroma: c, alpha: al }, filmcraft_frame::PixelData::Yuv8 { planes: b, chroma: d, alpha: bl }) => {
                c == d && al == bl && a.iter().zip(b).all(|(a, b)| a == b)
            }
            (
                filmcraft_frame::PixelData::Yuv16 { planes: a, chroma: c, bits: n, alpha: al },
                filmcraft_frame::PixelData::Yuv16 { planes: b, chroma: d, bits: m, alpha: bl },
            ) => c == d && n == m && al == bl && a.iter().zip(b).all(|(a, b)| a == b),
            (filmcraft_frame::PixelData::Rgba8(a), filmcraft_frame::PixelData::Rgba8(b)) => a == b,
            _ => false,
        };
        x.pts == y.pts && x.draft == y.draft && p.width == q.width && p.height == q.height && p.color == q.color && p.par == q.par && pixels
    }
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
}

/// A hardware decoder with a transparent software fallback.
pub struct HybridDecoder {
    hw: Option<Box<dyn VideoDecoder>>,
    sw: Option<Box<dyn VideoDecoder>>,
    entry: SampleEntry,
    info: NalStreamInfo,
    draft: bool,
    /// Samples (and pts) since the last restart point; `None` once the bound was passed.
    log: Option<Vec<(Vec<u8>, i64)>>,
    log_bytes: usize,
    /// Index in the log of the most recent restart point.
    last_irap: usize,
    /// Pictures returned since the last reset (by pts).
    emitted: BTreeSet<i64>,
    /// Pictures the hardware had decoded but not returned when it failed.
    carry: BTreeMap<i64, DecodedFrame>,
    /// First-use verification in progress.
    verify: Option<Verify>,
}

impl HybridDecoder {
    /// Wrap `hw`, a decoder for `entry` (whose stream is described by `info`).
    pub fn new(hw: Box<dyn VideoDecoder>, entry: SampleEntry, info: NalStreamInfo) -> Self {
        filmcraft_codecs::hw::note_hw_session();
        Self {
            hw: Some(hw),
            sw: None,
            entry,
            info,
            draft: false,
            log: Some(Vec::new()),
            log_bytes: 0,
            last_irap: 0,
            emitted: BTreeSet::new(),
            carry: BTreeMap::new(),
            verify: None,
        }
    }

    /// As [`HybridDecoder::new`], checking the hardware against our reference decoder
    /// (`filmcraft_codecs::reference_video_decoder`) on the first pictures, for the caller that
    /// claimed path `key` ([`verification`]). On an error the claim is given back.
    pub fn verifying(hw: Box<dyn VideoDecoder>, entry: SampleEntry, info: NalStreamInfo, key: &'static str) -> Result<Self> {
        let reference = match filmcraft_codecs::reference_video_decoder(&entry) {
            Ok(d) => d,
            Err(e) => {
                release(key);
                return Err(e);
            }
        };
        let mut h = Self::new(hw, entry, info);
        h.verify = Some(Verify { key, reference, left: VERIFY_PICTURES });
        Ok(h)
    }

    /// Whether a first-use check is still running.
    pub fn is_verifying(&self) -> bool {
        self.verify.is_some()
    }

    /// Continue with `sw` (in step with the stream) instead of the hardware.
    fn switch_to(&mut self, mut sw: Box<dyn VideoDecoder>) {
        sw.set_draft(self.draft);
        self.hw = None;
        self.sw = Some(sw);
    }

    /// Settle one lockstep step of the first-use check (`hw` / `sw`: both decoders' results for
    /// the same call).
    fn checked(&mut self, hw: Result<Vec<DecodedFrame>>, sw: Result<Vec<DecodedFrame>>) -> Result<Vec<DecodedFrame>> {
        let Some(v) = self.verify.take() else { return hw };
        match (hw, sw) {
            (Ok(h), Ok(s)) if same_frames(&h, &s) => {
                filmcraft_codecs::hw::note_hw_frames(h.len());
                let left = v.left.saturating_sub(h.len());
                if left == 0 {
                    settle(v.key, true);
                    log::info!("{}: pictures match the software decoder; using the hardware", v.key);
                } else {
                    self.verify = Some(Verify { left, ..v });
                }
                self.note_emitted(&h);
                Ok(h)
            }
            (Ok(_), Ok(s)) => {
                log::error!("{}: pictures differ from the software decoder's; hardware decoding of this kind is off for the rest of the run", v.key);
                settle(v.key, false);
                filmcraft_codecs::hw::note_hw_mismatch();
                self.switch_to(v.reference);
                self.note_emitted(&s);
                Ok(s)
            }
            (Err(e), Ok(s)) => {
                log::warn!("{}: failed while being checked ({e}); continuing with the software decoder", v.key);
                release(v.key);
                filmcraft_codecs::hw::note_hw_fallback();
                self.switch_to(v.reference);
                self.note_emitted(&s);
                Ok(s)
            }
            (_, Err(e)) => {
                // A stream our own decoder rejects cannot be checked: give the claim back and
                // behave as the software decoder does.
                release(v.key);
                self.switch_to(v.reference);
                Err(e)
            }
        }
    }

    /// Whether the hardware decoder is still in use (false after a fallback).
    pub fn is_hardware(&self) -> bool {
        self.hw.is_some()
    }

    fn remember(&mut self, sample: &[u8], pts: i64) {
        if self.info.is_irap(sample) {
            // An HEVC CRA keeps the previous restart point: replaying from the CRA itself would
            // treat it as a first picture and drop its RASL pictures, which the hardware decodes.
            let cra = self.info.codec == filmcraft_codecs::hw::NalCodec::Hevc && self.info.nal_types(sample).contains(&21);
            match self.log.as_mut() {
                Some(log) if cra => {
                    log.drain(..self.last_irap.min(log.len()));
                    self.log_bytes = log.iter().map(|s| s.0.len()).sum();
                }
                _ => {
                    self.log = Some(Vec::new());
                    self.log_bytes = 0;
                }
            }
            self.last_irap = self.log.as_ref().map_or(0, Vec::len);
        }
        let Some(log) = self.log.as_mut() else { return };
        self.log_bytes = self.log_bytes.saturating_add(sample.len());
        if log.len() >= LOG_MAX_SAMPLES || self.log_bytes > LOG_MAX_BYTES {
            self.log = None;
            return;
        }
        log.push((sample.to_vec(), pts));
    }

    fn note_emitted(&mut self, out: &[DecodedFrame]) {
        for f in out {
            self.emitted.insert(f.pts);
        }
        while self.emitted.len() > EMITTED_MAX {
            self.emitted.pop_first();
        }
    }

    /// In-band parameter sets that differ from the sample entry's (the hardware session was set
    /// up from those).
    fn parameter_sets_changed(&self, sample: &[u8]) -> bool {
        self.info
            .nals(sample)
            .into_iter()
            .any(|n| n.first().is_some_and(|&h| self.info.is_parameter_set(self.info.nal_type(h))) && !self.info.parameter_sets.iter().any(|p| p == n))
    }

    /// Software output after a fallback: pictures already returned are dropped, carried hardware
    /// pictures are merged in presentation order (a software picture replaces the carried one with
    /// the same pts).
    fn merge(&mut self, frames: Vec<DecodedFrame>) -> Vec<DecodedFrame> {
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            if self.emitted.contains(&f.pts) {
                self.carry.remove(&f.pts);
                continue;
            }
            while let Some(e) = self.carry.first_entry() {
                if *e.key() >= f.pts {
                    break;
                }
                out.push(e.remove());
            }
            self.carry.remove(&f.pts);
            out.push(f);
        }
        self.note_emitted(&out);
        out
    }

    /// Switch to the software decoder after `err`, replaying the log (which ends with the sample
    /// that failed).
    fn fall_back(&mut self, err: CodecError) -> Result<Vec<DecodedFrame>> {
        let Some(mut hw) = self.hw.take() else { return Err(err) };
        log::warn!("{} failed ({err}); continuing with the software decoder", hw.name());
        filmcraft_codecs::hw::note_hw_fallback();
        for f in hw.flush() {
            if !self.emitted.contains(&f.pts) {
                self.carry.insert(f.pts, f);
            }
        }
        drop(hw);
        let mut sw = filmcraft_codecs::software_video_decoder(&self.entry).map_err(|e| CodecError::Decode(format!("{err}; no software decoder: {e}")))?;
        sw.set_draft(self.draft);
        self.sw = Some(sw);
        let Some(log) = self.log.take() else {
            // Too far from a restart point to replay: the next seek restarts in software.
            self.carry.clear();
            return Err(CodecError::Decode(format!("{err} (continuing in software after the next seek)")));
        };
        let mut frames = Vec::new();
        for (s, p) in &log {
            let Some(sw) = self.sw.as_mut() else { break };
            frames.extend(sw.decode(s, *p)?);
        }
        self.log = Some(log);
        Ok(self.merge(frames))
    }
}

impl VideoDecoder for HybridDecoder {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        if let Some(sw) = self.sw.as_mut() {
            let frames = sw.decode(sample, pts)?;
            return Ok(self.merge(frames));
        }
        self.remember(sample, pts);
        if self.parameter_sets_changed(sample) {
            if let Some(v) = self.verify.take() {
                // The reference decoder of the check is in step with the stream: continue with it.
                log::warn!("{}: in-band parameter sets differ from the sample entry; continuing with the software decoder", v.key);
                release(v.key);
                filmcraft_codecs::hw::note_hw_fallback();
                self.switch_to(v.reference);
                let frames = self.sw.as_mut().map(|sw| sw.decode(sample, pts)).unwrap_or_else(|| Ok(Vec::new()))?;
                return Ok(self.merge(frames));
            }
            return self.fall_back(CodecError::Decode("in-band parameter sets differ from the sample entry".into()));
        }
        if let (Some(v), Some(hw)) = (self.verify.as_mut(), self.hw.as_mut()) {
            let s = v.reference.decode(sample, pts);
            let h = hw.decode(sample, pts);
            return self.checked(h, s);
        }
        let Some(hw) = self.hw.as_mut() else {
            return Err(CodecError::Decode("no decoder".into()));
        };
        match hw.decode(sample, pts) {
            Ok(out) => {
                filmcraft_codecs::hw::note_hw_frames(out.len());
                self.note_emitted(&out);
                Ok(out)
            }
            Err(e) => self.fall_back(e),
        }
    }

    fn flush(&mut self) -> Vec<DecodedFrame> {
        if let (Some(v), Some(hw)) = (self.verify.as_mut(), self.hw.as_mut()) {
            let s = v.reference.flush();
            let h = hw.flush();
            return self.checked(Ok(h), Ok(s)).unwrap_or_default();
        }
        if let Some(hw) = self.hw.as_mut() {
            let out = hw.flush();
            filmcraft_codecs::hw::note_hw_frames(out.len());
            self.note_emitted(&out);
            return out;
        }
        let frames = self.sw.as_mut().map(|d| d.flush()).unwrap_or_default();
        let mut out = self.merge(frames);
        out.extend(std::mem::take(&mut self.carry).into_values());
        out
    }

    fn reset(&mut self) {
        if let Some(hw) = self.hw.as_mut() {
            hw.reset();
        }
        if let Some(v) = self.verify.as_mut() {
            v.reference.reset();
        }
        if let Some(sw) = self.sw.as_mut() {
            sw.reset();
        }
        self.log = Some(Vec::new());
        self.log_bytes = 0;
        self.last_irap = 0;
        self.emitted.clear();
        self.carry.clear();
    }

    fn name(&self) -> &str {
        match (&self.hw, &self.sw) {
            (Some(hw), _) => hw.name(),
            (None, Some(sw)) => sw.name(),
            (None, None) => "hardware decoder",
        }
    }

    fn is_random_access(&self, sample: &[u8]) -> Option<bool> {
        self.info.is_random_access(sample)
    }

    fn is_disposable(&self, sample: &[u8]) -> bool {
        self.info.is_disposable(sample)
    }

    fn set_draft(&mut self, on: bool) {
        // The reference decoder of a first-use check stays exact (the hardware ignores draft mode).
        self.draft = on;
        if let Some(sw) = self.sw.as_mut() {
            sw.set_draft(on);
        }
    }
}

impl Drop for HybridDecoder {
    fn drop(&mut self) {
        // An unfinished check gives its claim back, so a later decoder verifies the path.
        if let Some(v) = self.verify.take() {
            release(v.key);
        }
    }
}
