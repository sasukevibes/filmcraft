//! H.264 parameters as Vulkan Video's std structures (`StdVideoH264*` / `StdVideoDecodeH264*`, the
//! Khronos `vulkan_video_codec_h264std` headers as ash generates them), built from this
//! workspace's own parsed SPS / PPS and the front-end's picture events (`filmcraft_h264::accel`).
//!
//! Safe code: the structures carry raw pointers into storage that [`ParameterSets`] owns, and that
//! storage outlives every Vulkan call reading them (the FFI module only passes them on).

use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags, StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH264ReferenceInfoFlags,
    StdVideoH264ChromaFormatIdc, StdVideoH264LevelIdc, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1, StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2, StdVideoH264PictureParameterSet, StdVideoH264PocType,
    StdVideoH264PpsFlags, StdVideoH264ProfileIdc, StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
    StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH, StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN, StdVideoH264ScalingLists,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags, StdVideoH264WeightedBipredIdc,
};
use filmcraft_h264::accel::{DecodePicture, Reference};
use filmcraft_h264::params::{Pps, ScalingMatrices, Sps};

/// The std header these structures follow: its name and `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)`.
pub(crate) const STD_HEADER_NAME: &std::ffi::CStr = c"VK_STD_vulkan_video_codec_h264_decode";
pub(crate) const STD_HEADER_VERSION: u32 = 1 << 22;

/// A slice NAL unit's prefix in the bitstream buffer: the decoder takes Annex B start codes.
const START_CODE: [u8; 3] = [0, 0, 1];

/// Std SPS / PPS structures and the storage their pointers point into.
pub(crate) struct ParameterSets {
    pub(crate) sps: Vec<StdVideoH264SequenceParameterSet>,
    pub(crate) pps: Vec<StdVideoH264PictureParameterSet>,
    // Pointed to by `sps` / `pps`: filled before any pointer is taken and never changed after, and
    // moving a vector does not move its heap storage.
    _sps_lists: Vec<Option<StdVideoH264ScalingLists>>,
    _pps_lists: Vec<Option<StdVideoH264ScalingLists>>,
    _offsets: Vec<Box<[i32]>>,
}

fn err(what: &str) -> impl Fn(std::num::TryFromIntError) -> String + '_ {
    move |_| format!("{what} out of range")
}

/// The std profile for an SPS (8-bit 4:2:0 progressive streams; the caller checks those).
pub(crate) fn std_profile(sps: &Sps) -> Result<StdVideoH264ProfileIdc, String> {
    match sps.profile_idc {
        66 => Ok(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE),
        77 => Ok(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN),
        // High 10 / 4:2:2 / 4:4:4 streams coded with High tools only at 8-bit 4:2:0 (the format
        // check comes first) decode as High.
        100 | 110 | 122 | 244 => Ok(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH),
        p => Err(format!("H.264 profile_idc {p}")),
    }
}

/// The std level for a `level_idc` (level 1b, coded as 9, decodes as 1.1).
pub(crate) fn std_level(level_idc: u8) -> Result<StdVideoH264LevelIdc, String> {
    Ok(match level_idc {
        10 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        9 | 11 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        20 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2,
        30 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2,
        40 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2,
        50 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2,
        60 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        62 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
        l => return Err(format!("H.264 level_idc {l}")),
    })
}

/// Effective scaling matrices as std scaling lists: every list of a 4:2:0 stream (six 4x4, the two
/// luma 8x8) marked present, in coded order, so no fall-back rule is left to the driver.
fn scaling_lists(m: &ScalingMatrices) -> StdVideoH264ScalingLists {
    let (l4, l8) = m.zigzag();
    StdVideoH264ScalingLists { scaling_list_present_mask: 0xff, use_default_scaling_matrix_mask: 0, ScalingList4x4: l4, ScalingList8x8: l8 }
}

fn std_sps(sps: &Sps) -> Result<StdVideoH264SequenceParameterSet, String> {
    let mut flags = StdVideoH264SpsFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: 0 };
    let c = sps.constraint_flags as u32;
    flags.set_constraint_set0_flag((c >> 7) & 1);
    flags.set_constraint_set1_flag((c >> 6) & 1);
    flags.set_constraint_set2_flag((c >> 5) & 1);
    flags.set_constraint_set3_flag((c >> 4) & 1);
    flags.set_constraint_set4_flag((c >> 3) & 1);
    flags.set_constraint_set5_flag((c >> 2) & 1);
    flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference as u32);
    flags.set_mb_adaptive_frame_field_flag(sps.mb_adaptive_frame_field as u32);
    flags.set_frame_mbs_only_flag(sps.frame_mbs_only as u32);
    flags.set_delta_pic_order_always_zero_flag(sps.delta_pic_order_always_zero as u32);
    flags.set_separate_colour_plane_flag(sps.separate_colour_plane as u32);
    flags.set_gaps_in_frame_num_value_allowed_flag(sps.gaps_in_frame_num_allowed as u32);
    flags.set_qpprime_y_zero_transform_bypass_flag(sps.qpprime_y_zero_transform_bypass as u32);
    flags.set_frame_cropping_flag(sps.frame_crop.is_some() as u32);
    flags.set_seq_scaling_matrix_present_flag(!sps.scaling.is_flat() as u32);
    // VUI is not used by the decoding process.
    flags.set_vui_parameters_present_flag(0);
    let (l, r, t, b) = sps.frame_crop.unwrap_or((0, 0, 0, 0));
    let minus = |v: u32, by: u32, what: &str| v.checked_sub(by).ok_or_else(|| format!("{what} out of range"));
    Ok(StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: std_profile(sps)?,
        level_idc: std_level(sps.level_idc)?,
        chroma_format_idc: sps.chroma_format_idc as StdVideoH264ChromaFormatIdc,
        seq_parameter_set_id: u8::try_from(sps.id).map_err(err("seq_parameter_set_id"))?,
        bit_depth_luma_minus8: u8::try_from(minus(sps.bit_depth_luma, 8, "bit_depth_luma")?).map_err(err("bit_depth_luma"))?,
        bit_depth_chroma_minus8: u8::try_from(minus(sps.bit_depth_chroma, 8, "bit_depth_chroma")?).map_err(err("bit_depth_chroma"))?,
        log2_max_frame_num_minus4: u8::try_from(minus(sps.log2_max_frame_num, 4, "log2_max_frame_num")?).map_err(err("log2_max_frame_num"))?,
        pic_order_cnt_type: sps.pic_order_cnt_type as StdVideoH264PocType,
        offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
        offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
        // only present (and at least 4) for pic_order_cnt_type 0
        log2_max_pic_order_cnt_lsb_minus4: u8::try_from(sps.log2_max_poc_lsb.saturating_sub(4)).map_err(err("log2_max_pic_order_cnt_lsb"))?,
        num_ref_frames_in_pic_order_cnt_cycle: u8::try_from(sps.offset_for_ref_frame.len()).map_err(err("num_ref_frames_in_pic_order_cnt_cycle"))?,
        max_num_ref_frames: u8::try_from(sps.max_num_ref_frames).map_err(err("max_num_ref_frames"))?,
        reserved1: 0,
        pic_width_in_mbs_minus1: minus(sps.pic_width_in_mbs, 1, "pic_width_in_mbs")?,
        pic_height_in_map_units_minus1: minus(sps.pic_height_in_map_units, 1, "pic_height_in_map_units")?,
        frame_crop_left_offset: l,
        frame_crop_right_offset: r,
        frame_crop_top_offset: t,
        frame_crop_bottom_offset: b,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
    })
}

/// `explicit_lists`: give the PPS its effective lists explicitly (whenever the SPS or the PPS has
/// a non-flat matrix), so the driver never has to apply the SPS / PPS fall-back.
fn std_pps(pps: &Pps, explicit_lists: bool) -> Result<StdVideoH264PictureParameterSet, String> {
    let mut flags = StdVideoH264PpsFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_transform_8x8_mode_flag(pps.transform_8x8_mode as u32);
    flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present as u32);
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred as u32);
    flags.set_deblocking_filter_control_present_flag(pps.deblocking_filter_control_present as u32);
    flags.set_weighted_pred_flag(pps.weighted_pred as u32);
    flags.set_bottom_field_pic_order_in_frame_present_flag(pps.bottom_field_pic_order_in_frame_present as u32);
    flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode as u32);
    flags.set_pic_scaling_matrix_present_flag(explicit_lists as u32);
    let minus1 = |v: u32, what: &str| v.checked_sub(1).and_then(|v| u8::try_from(v).ok()).ok_or_else(|| format!("{what} out of range"));
    let i8_of = |v: i32, what: &str| i8::try_from(v).map_err(|_| format!("{what} out of range"));
    if pps.weighted_bipred_idc > 2 {
        return Err(format!("weighted_bipred_idc {}", pps.weighted_bipred_idc));
    }
    Ok(StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: u8::try_from(pps.sps_id).map_err(err("seq_parameter_set_id"))?,
        pic_parameter_set_id: u8::try_from(pps.id).map_err(err("pic_parameter_set_id"))?,
        num_ref_idx_l0_default_active_minus1: minus1(pps.num_ref_idx_l0_default_active, "num_ref_idx_l0_default_active")?,
        num_ref_idx_l1_default_active_minus1: minus1(pps.num_ref_idx_l1_default_active, "num_ref_idx_l1_default_active")?,
        weighted_bipred_idc: pps.weighted_bipred_idc as StdVideoH264WeightedBipredIdc,
        pic_init_qp_minus26: i8_of(pps.pic_init_qp.saturating_sub(26), "pic_init_qp")?,
        pic_init_qs_minus26: i8_of(pps.pic_init_qs.saturating_sub(26), "pic_init_qs")?,
        chroma_qp_index_offset: i8_of(pps.chroma_qp_index_offset, "chroma_qp_index_offset")?,
        second_chroma_qp_index_offset: i8_of(pps.second_chroma_qp_index_offset, "second_chroma_qp_index_offset")?,
        pScalingLists: std::ptr::null(),
    })
}

/// The std structures for a stream's parameter sets (each PPS with its SPS).
pub(crate) fn parameter_sets(spss: &[Sps], ppss: &[Pps]) -> Result<ParameterSets, String> {
    // Storage first; pointers into it are taken once it is complete.
    let sps_lists: Vec<Option<StdVideoH264ScalingLists>> = spss.iter().map(|s| (!s.scaling.is_flat()).then(|| scaling_lists(&s.scaling))).collect();
    let offsets: Vec<Box<[i32]>> = spss.iter().map(|s| s.offset_for_ref_frame.clone().into_boxed_slice()).collect();
    let mut explicit = Vec::with_capacity(ppss.len());
    for pps in ppss {
        let sps = spss.iter().find(|s| s.id == pps.sps_id).ok_or_else(|| format!("PPS {} refers to a missing SPS {}", pps.id, pps.sps_id))?;
        explicit.push(!(pps.scaling.is_flat() && sps.scaling.is_flat()));
    }
    let pps_lists: Vec<Option<StdVideoH264ScalingLists>> = ppss.iter().zip(&explicit).map(|(p, &e)| e.then(|| scaling_lists(&p.scaling))).collect();
    let mut sps = Vec::with_capacity(spss.len());
    for ((s, lists), offs) in spss.iter().zip(&sps_lists).zip(&offsets) {
        let mut std = std_sps(s)?;
        if !offs.is_empty() {
            std.pOffsetForRefFrame = offs.as_ptr();
        }
        if let Some(l) = lists {
            std.pScalingLists = l;
        }
        sps.push(std);
    }
    let mut pps = Vec::with_capacity(ppss.len());
    for ((p, &e), lists) in ppss.iter().zip(&explicit).zip(&pps_lists) {
        let mut std = std_pps(p, e)?;
        if let Some(l) = lists {
            std.pScalingLists = l;
        }
        pps.push(std);
    }
    Ok(ParameterSets { sps, pps, _sps_lists: sps_lists, _pps_lists: pps_lists, _offsets: offsets })
}

/// The picture's std decode info.
pub(crate) fn picture_info(p: &DecodePicture) -> Result<StdVideoDecodeH264PictureInfo, String> {
    let mut flags = StdVideoDecodeH264PictureInfoFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_is_intra(p.intra as u32);
    flags.set_IdrPicFlag(p.idr as u32);
    flags.set_is_reference(p.reference as u32);
    Ok(StdVideoDecodeH264PictureInfo {
        flags,
        seq_parameter_set_id: u8::try_from(p.sps.id).map_err(err("seq_parameter_set_id"))?,
        pic_parameter_set_id: u8::try_from(p.pps.id).map_err(err("pic_parameter_set_id"))?,
        reserved1: 0,
        reserved2: 0,
        frame_num: u16::try_from(p.frame_num).map_err(err("frame_num"))?,
        idr_pic_id: u16::try_from(p.idr_pic_id).map_err(err("idr_pic_id"))?,
        PicOrderCnt: [p.poc.0, p.poc.1],
    })
}

/// A reference picture's std info (a frame: neither field flag set).
pub(crate) fn reference_info(r: &Reference) -> Result<StdVideoDecodeH264ReferenceInfo, String> {
    let mut flags = StdVideoDecodeH264ReferenceInfoFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_used_for_long_term_reference(r.long_term as u32);
    flags.set_is_non_existing(r.non_existing as u32);
    Ok(StdVideoDecodeH264ReferenceInfo { flags, FrameNum: u16::try_from(r.frame_num).map_err(err("FrameNum"))?, reserved: 0, PicOrderCnt: [r.poc.0, r.poc.1] })
}

/// The std info of the slot the picture is decoded into: the picture as it is stored for reference,
/// or (a non-reference picture, whose slot the decode does not activate) its own numbers.
pub(crate) fn setup_info(p: &DecodePicture) -> Result<StdVideoDecodeH264ReferenceInfo, String> {
    let r = p.stored.unwrap_or(Reference { id: p.id, long_term: false, frame_num: p.frame_num, poc: p.poc, non_existing: false });
    reference_info(&r)
}

/// The picture's bitstream: each slice NAL unit after a start code, and where each starts.
pub(crate) fn bitstream(slices: &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u32>), String> {
    let len = slices.iter().try_fold(0usize, |a, s| a.checked_add(s.len())?.checked_add(START_CODE.len())).ok_or("bitstream too large")?;
    let mut data = Vec::with_capacity(len);
    let mut offsets = Vec::with_capacity(slices.len());
    for s in slices {
        offsets.push(u32::try_from(data.len()).map_err(err("slice offset"))?);
        data.extend_from_slice(&START_CODE);
        data.extend_from_slice(s);
    }
    Ok((data, offsets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_profiles() {
        assert_eq!(std_level(51), Ok(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1));
        assert_eq!(std_level(9), Ok(StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1));
        assert!(std_level(0).is_err());
        assert!(std_level(63).is_err());
    }

    /// ffmpeg's `trace_headers` dump of a stream (external oracle): (block title, fields in order).
    fn trace(ff: &std::path::Path, path: &std::path::Path) -> Vec<(String, Vec<(String, i64)>)> {
        let out = std::process::Command::new(ff)
            .args(["-hide_banner", "-loglevel", "info", "-i"])
            .arg(path)
            .args(["-c:v", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stderr);
        let mut blocks: Vec<(String, Vec<(String, i64)>)> = Vec::new();
        for line in text.lines() {
            let Some((_, body)) = line.strip_prefix("[trace_headers @ ").and_then(|r| r.split_once("] ")) else { continue };
            let body = body.trim();
            let field = body.split_once(" = ").and_then(|(lhs, v)| {
                let mut t = lhs.split_whitespace();
                let pos = t.next()?;
                pos.chars().all(|c| c.is_ascii_digit()).then(|| (t.next().unwrap_or_default().to_string(), v.trim().parse::<i64>().ok()))
            });
            match field {
                Some((name, Some(v))) => {
                    if let Some(b) = blocks.last_mut() {
                        b.1.push((name, v));
                    }
                }
                Some((_, None)) => {}
                None => blocks.push((body.to_string(), Vec::new())),
            }
        }
        blocks
    }

    fn get(fields: &[(String, i64)], name: &str, absent: i64) -> i64 {
        fields.iter().find(|(n, _)| n == name).map_or(absent, |(_, v)| *v)
    }

    /// The std SPS / PPS and per-picture structures carry what the bitstream says, field by
    /// field, on libx264 streams covering custom scaling matrices, weighted prediction, cropping,
    /// CAVLC / CABAC, picture order count types 0 and 2 and several reference frames; ffmpeg's
    /// header trace is the external reference (generator and oracle only).
    #[test]
    fn std_structures_match_ffmpeg_trace_headers() {
        let Some(ff) = filmcraft_testkit::ffmpeg() else {
            eprintln!("SKIPPED: no ffmpeg");
            return;
        };
        let streams: [(&str, &str, &[&str]); 3] = [
            ("trace_high_cqm.h264", "352x288", &["-profile:v", "high", "-x264-params", "cqm=jvt:bframes=3:b-pyramid=normal:weightp=2:weightb=1:ref=3"]),
            ("trace_baseline_crop.h264", "318x238", &["-profile:v", "baseline", "-x264-params", "ref=2:keyint=8"]),
            ("trace_main_ref4.h264", "320x240", &["-profile:v", "main", "-x264-params", "ref=4:bframes=2:keyint=10:8x8dct=0"]),
        ];
        for (name, size, args) in streams {
            let out = filmcraft_testkit::fixtures_dir("platform").join(name);
            let made = filmcraft_testkit::fixtures::generate(&out, |tmp| {
                std::process::Command::new(&ff)
                    .args(["-y", "-v", "error", "-f", "lavfi", "-i", &format!("testsrc2=s={size}:r=25:d=1,noise=alls=10:allf=t")])
                    .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
                    .args(args)
                    .args(["-f", "h264"])
                    .arg(tmp)
                    .status()
                    .is_ok_and(|s| s.success())
            });
            let Some(path) = made else { panic!("{name}: fixture generation failed") };
            let data = std::fs::read(&path).unwrap();
            let blocks = trace(&ff, &path);
            let block = |title: &str| blocks.iter().find(|b| b.0 == title).map(|b| b.1.clone()).unwrap_or_else(|| panic!("{name}: no {title} in the trace"));
            let (ts, tp) = (block("Sequence Parameter Set"), block("Picture Parameter Set"));

            let mut table: Vec<Option<Sps>> = vec![None; 32];
            let mut ppss = Vec::new();
            for nal in filmcraft_bitstream::annexb_nals(&data) {
                let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[1..]);
                match nal[0] & 0x1f {
                    7 if table.iter().all(Option::is_none) => {
                        let sps = Sps::parse(&rbsp).unwrap();
                        let id = sps.id as usize;
                        table[id] = Some(sps);
                    }
                    8 if ppss.is_empty() => ppss.push(Pps::parse(&rbsp, &table).unwrap()),
                    _ => {}
                }
            }
            let spss: Vec<Sps> = table.into_iter().flatten().collect();
            let sets = parameter_sets(&spss, &ppss).unwrap();
            let (s, p) = (&sets.sps[0], &sets.pps[0]);
            let f = &s.flags;
            let got_sps = [
                ("profile_idc", s.profile_idc as i64),
                ("constraint_set0_flag", f.constraint_set0_flag() as i64),
                ("constraint_set1_flag", f.constraint_set1_flag() as i64),
                ("constraint_set2_flag", f.constraint_set2_flag() as i64),
                ("constraint_set3_flag", f.constraint_set3_flag() as i64),
                ("constraint_set4_flag", f.constraint_set4_flag() as i64),
                ("constraint_set5_flag", f.constraint_set5_flag() as i64),
                ("seq_parameter_set_id", s.seq_parameter_set_id as i64),
                ("chroma_format_idc", s.chroma_format_idc as i64),
                ("bit_depth_luma_minus8", s.bit_depth_luma_minus8 as i64),
                ("bit_depth_chroma_minus8", s.bit_depth_chroma_minus8 as i64),
                ("qpprime_y_zero_transform_bypass_flag", f.qpprime_y_zero_transform_bypass_flag() as i64),
                ("seq_scaling_matrix_present_flag", f.seq_scaling_matrix_present_flag() as i64),
                ("log2_max_frame_num_minus4", s.log2_max_frame_num_minus4 as i64),
                ("pic_order_cnt_type", s.pic_order_cnt_type as i64),
                ("log2_max_pic_order_cnt_lsb_minus4", s.log2_max_pic_order_cnt_lsb_minus4 as i64),
                ("delta_pic_order_always_zero_flag", f.delta_pic_order_always_zero_flag() as i64),
                ("offset_for_non_ref_pic", s.offset_for_non_ref_pic as i64),
                ("offset_for_top_to_bottom_field", s.offset_for_top_to_bottom_field as i64),
                ("num_ref_frames_in_pic_order_cnt_cycle", s.num_ref_frames_in_pic_order_cnt_cycle as i64),
                ("max_num_ref_frames", s.max_num_ref_frames as i64),
                ("gaps_in_frame_num_allowed_flag", f.gaps_in_frame_num_value_allowed_flag() as i64),
                ("pic_width_in_mbs_minus1", s.pic_width_in_mbs_minus1 as i64),
                ("pic_height_in_map_units_minus1", s.pic_height_in_map_units_minus1 as i64),
                ("frame_mbs_only_flag", f.frame_mbs_only_flag() as i64),
                ("mb_adaptive_frame_field_flag", f.mb_adaptive_frame_field_flag() as i64),
                ("direct_8x8_inference_flag", f.direct_8x8_inference_flag() as i64),
                ("frame_cropping_flag", f.frame_cropping_flag() as i64),
                ("frame_crop_left_offset", s.frame_crop_left_offset as i64),
                ("frame_crop_right_offset", s.frame_crop_right_offset as i64),
                ("frame_crop_top_offset", s.frame_crop_top_offset as i64),
                ("frame_crop_bottom_offset", s.frame_crop_bottom_offset as i64),
            ];
            for (field, v) in got_sps {
                let absent = if field == "chroma_format_idc" { 1 } else { 0 };
                assert_eq!(v, get(&ts, field, absent), "{name}: SPS {field}");
            }
            assert_eq!(s.level_idc, std_level(get(&ts, "level_idc", -1) as u8).unwrap(), "{name}: SPS level_idc");
            let f = &p.flags;
            let got_pps = [
                ("pic_parameter_set_id", p.pic_parameter_set_id as i64),
                ("seq_parameter_set_id", p.seq_parameter_set_id as i64),
                ("entropy_coding_mode_flag", f.entropy_coding_mode_flag() as i64),
                ("bottom_field_pic_order_in_frame_present_flag", f.bottom_field_pic_order_in_frame_present_flag() as i64),
                ("num_ref_idx_l0_default_active_minus1", p.num_ref_idx_l0_default_active_minus1 as i64),
                ("num_ref_idx_l1_default_active_minus1", p.num_ref_idx_l1_default_active_minus1 as i64),
                ("weighted_pred_flag", f.weighted_pred_flag() as i64),
                ("weighted_bipred_idc", p.weighted_bipred_idc as i64),
                ("pic_init_qp_minus26", p.pic_init_qp_minus26 as i64),
                ("pic_init_qs_minus26", p.pic_init_qs_minus26 as i64),
                ("chroma_qp_index_offset", p.chroma_qp_index_offset as i64),
                ("deblocking_filter_control_present_flag", f.deblocking_filter_control_present_flag() as i64),
                ("constrained_intra_pred_flag", f.constrained_intra_pred_flag() as i64),
                ("redundant_pic_cnt_present_flag", f.redundant_pic_cnt_present_flag() as i64),
                ("transform_8x8_mode_flag", f.transform_8x8_mode_flag() as i64),
            ];
            for (field, v) in got_pps {
                assert_eq!(v, get(&tp, field, 0), "{name}: PPS {field}");
            }
            // absent: equal to chroma_qp_index_offset
            assert_eq!(p.second_chroma_qp_index_offset as i64, get(&tp, "second_chroma_qp_index_offset", get(&tp, "chroma_qp_index_offset", 0)), "{name}");
            // lists are given explicitly whenever any matrix is in play
            let lists_in_play = get(&ts, "seq_scaling_matrix_present_flag", 0) == 1 || get(&tp, "pic_scaling_matrix_present_flag", 0) == 1;
            assert_eq!(f.pic_scaling_matrix_present_flag() == 1, lists_in_play, "{name}: PPS scaling lists");
            assert_eq!(!p.pScalingLists.is_null(), lists_in_play, "{name}");
            assert_eq!(name == "trace_high_cqm.h264", lists_in_play, "{name}: only the cqm stream has matrices");

            // pictures, in decoding order: the trace's first slice of each picture
            let pictures: Vec<&Vec<(String, i64)>> =
                blocks.iter().filter(|b| b.0 == "Slice Header" && get(&b.1, "first_mb_in_slice", -1) == 0).map(|b| &b.1).collect();
            let mut fe = filmcraft_h264::accel::Frontend::new();
            let mut events = fe.decode(&data, 0).unwrap();
            events.extend(fe.flush());
            let decodes: Vec<&DecodePicture> = events
                .iter()
                .filter_map(|e| match e {
                    filmcraft_h264::accel::Event::Decode(p) => Some(p),
                    filmcraft_h264::accel::Event::Output(_) => None,
                })
                .collect();
            assert_eq!(decodes.len(), pictures.len(), "{name}: picture count");
            assert!(decodes.len() >= 20, "{name}: {} pictures", decodes.len());
            for (i, (d, t)) in decodes.iter().zip(&pictures).enumerate() {
                let info = picture_info(d).unwrap();
                let idr = get(t, "nal_unit_type", -1) == 5;
                assert_eq!(info.flags.IdrPicFlag() == 1, idr, "{name}: picture {i} IDR");
                assert_eq!(info.flags.is_reference() == 1, get(t, "nal_ref_idc", -1) != 0, "{name}: picture {i} reference");
                assert_eq!(info.frame_num as i64, get(t, "frame_num", -1), "{name}: picture {i} frame_num");
                if idr {
                    assert_eq!(info.idr_pic_id as i64, get(t, "idr_pic_id", -1), "{name}: picture {i} idr_pic_id");
                }
                assert_eq!((info.seq_parameter_set_id, info.pic_parameter_set_id), (0, 0), "{name}");
                assert_eq!(info.PicOrderCnt, [d.poc.0, d.poc.1]);
                let setup = setup_info(d).unwrap();
                assert_eq!(setup.FrameNum as u32, d.frame_num, "{name}: setup slot FrameNum");
                for r in &d.refs {
                    let ri = reference_info(r).unwrap();
                    assert_eq!(
                        (ri.FrameNum as u32, ri.PicOrderCnt, ri.flags.used_for_long_term_reference()),
                        (r.frame_num, [r.poc.0, r.poc.1], r.long_term as u32)
                    );
                }
            }
        }
    }

    #[test]
    fn bitstream_puts_a_start_code_before_each_slice() {
        let (data, offsets) = bitstream(&[vec![0x65, 1, 2], vec![0x41, 3]]).unwrap();
        assert_eq!(data, [0, 0, 1, 0x65, 1, 2, 0, 0, 1, 0x41, 3]);
        assert_eq!(offsets, [0, 6]);
        assert_eq!(bitstream(&[]).unwrap(), (Vec::new(), Vec::new()));
    }
}
