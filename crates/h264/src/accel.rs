//! Front-end for hardware ("stateless") decoders: this crate's own NAL handling, slice-header
//! parsing, picture order counts (8.2.1), reference marking (8.2.5) and DPB output process (C.4),
//! without reconstructing any pixels.
//!
//! Stateless hardware APIs (Vulkan Video, VA-API, D3D11 video decoding) decode one picture at a
//! time from what the application derives from the bitstream: the active SPS and PPS, the
//! picture's `frame_num`, `idr_pic_id` and picture order counts, its slices, and the pictures of the
//! decoded picture buffer it may reference. [`Frontend`] reports exactly that as [`Event::Decode`],
//! and reports output as [`Event::Output`] in the order [`crate::Decoder`] outputs pictures, so a
//! hardware decoder built on it is interchangeable with the software decoder.
//!
//! Pictures are named by an `id`, unique within one [`Frontend`]. A picture's storage must be kept
//! while its id is listed in a later [`DecodePicture::dpb`] or until it has been output, whichever
//! is later; pictures dropped by `no_output_of_prior_pics_flag` are never output.

use std::sync::Arc;

use crate::decoder::Decoder;
use crate::error::Result;
use crate::params::{Pps, Sps};

/// What a hardware decoder has to do next, in order.
#[derive(Clone, Debug)]
pub enum Event {
    /// Decode a picture.
    Decode(DecodePicture),
    /// Output a decoded picture (presentation order).
    Output(OutputPicture),
}

/// One picture to decode.
#[derive(Clone, Debug)]
pub struct DecodePicture {
    pub id: u32,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    pub frame_num: u32,
    pub idr: bool,
    pub idr_pic_id: u32,
    /// `nal_ref_idc` != 0.
    pub reference: bool,
    /// Every slice is an I or SI slice.
    pub intra: bool,
    /// TopFieldOrderCnt and BottomFieldOrderCnt while the picture is decoded (before an MMCO 5
    /// makes them relative to the picture itself).
    pub poc: (i32, i32),
    /// The picture as it is stored for reference after decoding (its reference marking applied),
    /// or `None` when it is not kept as a reference.
    pub stored: Option<Reference>,
    /// Slice NAL units in decoding order: NAL header byte and payload, emulation prevention bytes
    /// kept (as in the bitstream, without length prefix or start code).
    pub slices: Vec<Vec<u8>>,
    /// The pictures marked "used for reference" when this picture is decoded, before its own
    /// reference marking: what its inter prediction may use.
    pub refs: Vec<Reference>,
    /// Every picture held in the decoded picture buffer when this picture is decoded (references
    /// and pictures waiting for output).
    pub dpb: Vec<u32>,
}

/// A picture of the decoded picture buffer as a reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    pub id: u32,
    pub long_term: bool,
    /// FrameNum of a short-term reference, LongTermFrameIdx of a long-term one.
    pub frame_num: u32,
    /// TopFieldOrderCnt and BottomFieldOrderCnt.
    pub poc: (i32, i32),
    /// A "non-existing" frame inferred for a gap in `frame_num` (8.2.5.2): it has no samples.
    pub non_existing: bool,
}

/// A picture leaving the decoded picture buffer for output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputPicture {
    pub id: u32,
    /// The `pts` passed to [`Frontend::decode`] with the picture's access unit.
    pub pts: i64,
    pub poc: i32,
    /// IDR picture.
    pub key: bool,
    /// Output (cropped) rectangle in luma samples: x, y, width, height.
    pub crop: (u32, u32, u32, u32),
    pub color: crate::ColorInfo,
    /// Sample aspect ratio ((0, 0) when unspecified).
    pub sar: (u16, u16),
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
    /// A front-end for Annex B byte-stream input (start codes, in-band parameter sets).
    pub fn new() -> Self {
        Self { dec: Decoder::accel() }
    }

    /// A front-end for length-prefixed samples described by an `avcC` record (its SPS and PPS are
    /// parsed up front).
    pub fn from_avcc(avcc: &[u8]) -> Result<Self> {
        let mut dec = Decoder::accel();
        dec.configure_avcc(avcc)?;
        Ok(Self { dec })
    }

    /// One access unit (all NAL units of one picture). On an error the events of this access unit
    /// are dropped and the front-end should be reset (recreated) before it is used again.
    pub fn decode(&mut self, data: &[u8], pts: i64) -> Result<Vec<Event>> {
        self.dec.decode_accel(data, pts)
    }

    /// End of stream: the outputs of every picture still waiting.
    pub fn flush(&mut self) -> Vec<Event> {
        self.dec.flush_accel()
    }

    /// Length of NAL length prefixes (from `avcC`), `None` for Annex B input.
    pub fn nal_length_size(&self) -> Option<usize> {
        self.dec.nal_length_size()
    }
}
