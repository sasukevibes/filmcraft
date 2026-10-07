//! Front-end for hardware ("stateless") decoders: this crate's own NAL handling, slice segment
//! header parsing, picture order counts (8.3.1), reference picture sets (8.3.2), RASL handling and
//! DPB output process (C.5.2), without reconstructing any pixels.
//!
//! Stateless hardware APIs (Vulkan Video, VA-API, D3D11 video decoding) decode one picture at a
//! time from what the application derives from the bitstream: the active VPS / SPS / PPS, the
//! picture's order count and IRAP flags, its slice segments, how many bits its slice header's
//! short-term RPS took, and which decoded pictures form its RPS lists. [`Frontend`] reports that as
//! [`Event::Decode`] and reports output as [`Event::Output`], in the calls and order in which a
//! single-threaded [`crate::Decoder`] outputs pictures.
//!
//! Pictures are named by an `id`, unique within one [`Frontend`]. A picture's storage must be kept
//! while its id is listed in a later [`DecodePicture::dpb`] or until it has been output, whichever
//! is later; pictures dropped by `no_output_of_prior_pics_flag` are never output. RASL pictures
//! the software decoder skips are not reported.

use std::sync::Arc;

use crate::decoder::Decoder;
use crate::error::Result;
use crate::params::{Pps, Sps, Vps};

/// What a hardware decoder has to do next, in order.
#[derive(Clone, Debug)]
pub enum Event {
    Decode(DecodePicture),
    Output(OutputPicture),
}

/// One picture to decode.
#[derive(Clone, Debug)]
pub struct DecodePicture {
    pub id: u32,
    /// The VPS the SPS names, when it was received.
    pub vps: Option<Arc<Vps>>,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    /// PicOrderCntVal.
    pub poc: i32,
    /// IRAP picture (BLA, IDR, CRA).
    pub irap: bool,
    pub idr: bool,
    /// The picture's short-term RPS came from the SPS (`short_term_ref_pic_set_sps_flag`).
    pub st_rps_sps: bool,
    /// Bits of `st_ref_pic_set()` in the slice header (0 when it came from the SPS).
    pub st_rps_bits: u32,
    /// NumDeltaPocs[RefRpsIdx] of a slice-header RPS predicted from another one, else 0.
    pub st_rps_ref_delta_pocs: u32,
    /// Slice segment NAL units in decoding order: NAL header and payload, emulation prevention
    /// bytes kept (without length prefix or start code).
    pub slices: Vec<Vec<u8>>,
    /// RefPicSetStCurrBefore, RefPicSetStCurrAfter and RefPicSetLtCurr as picture ids.
    pub st_curr_before: Vec<u32>,
    pub st_curr_after: Vec<u32>,
    pub lt_curr: Vec<u32>,
    /// The pictures marked as references once this picture's RPS is applied (current and
    /// following ones): what must stay available. Missing references the software decoder
    /// generated (8.3.3) are listed with `non_existing`.
    pub refs: Vec<Reference>,
    /// Every picture held in the decoded picture buffer when this picture is decoded.
    pub dpb: Vec<u32>,
}

/// A picture of the decoded picture buffer as a reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    pub id: u32,
    pub poc: i32,
    pub long_term: bool,
    /// Generated for a missing reference (8.3.3): it has no decoded samples.
    pub non_existing: bool,
}

/// A picture leaving the decoded picture buffer for output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputPicture {
    pub id: u32,
    pub pts: i64,
    pub poc: i32,
    /// IRAP picture.
    pub key: bool,
    /// Output (cropped) rectangle in luma samples: x, y, width, height.
    pub crop: (u32, u32, u32, u32),
    pub color: crate::ColorInfo,
    pub sar: (u16, u16),
    pub bit_depth: u32,
}

/// The decoder front-end: feed access units in decoding order, act on the events in order.
pub struct Frontend {
    dec: Decoder,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend {
    /// A front-end for Annex B byte-stream input.
    pub fn new() -> Self {
        Self { dec: Decoder::accel() }
    }

    /// A front-end for length-prefixed samples described by an `hvcC` record.
    pub fn from_hvcc(hvcc: &[u8]) -> Result<Self> {
        let mut dec = Decoder::accel();
        dec.configure_hvcc(hvcc)?;
        Ok(Self { dec })
    }

    /// One access unit. On an error the events of this access unit are dropped and the front-end
    /// should be recreated before it is used again.
    pub fn decode(&mut self, data: &[u8], pts: i64) -> Result<Vec<Event>> {
        self.dec.decode_accel(data, pts)
    }

    /// End of stream: the outputs of every picture still waiting.
    pub fn flush(&mut self) -> Vec<Event> {
        self.dec.flush_accel()
    }

    /// Length of NAL length prefixes (from `hvcC`), `None` for Annex B input.
    pub fn nal_length_size(&self) -> Option<usize> {
        self.dec.nal_length_size()
    }
}
