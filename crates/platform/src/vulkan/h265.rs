//! H.265 parameters as Vulkan Video's std structures (`StdVideoH265*` / `StdVideoDecodeH265*`, the
//! Khronos `vulkan_video_codec_h265std` headers as ash generates them), built from this
//! workspace's own parsed VPS / SPS / PPS and the front-end's picture events
//! (`filmcraft_hevc::accel`).
//!
//! Safe code: the structures carry raw pointers into storage that [`ParameterSets`] owns; it is
//! filled completely before any pointer is taken, never changed afterwards, and outlives every
//! Vulkan call reading the structures. Short-term RPSs are given in their explicit form (the same
//! DeltaPoc values a predicted set resolves to), scaling lists with prediction already resolved.

use ash::vk::native::{
    StdVideoDecodeH265PictureInfo, StdVideoDecodeH265PictureInfoFlags, StdVideoDecodeH265ReferenceInfo, StdVideoDecodeH265ReferenceInfoFlags,
    StdVideoH265ChromaFormatIdc, StdVideoH265DecPicBufMgr, StdVideoH265LevelIdc, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    StdVideoH265LongTermRefPicsSps, StdVideoH265PictureParameterSet, StdVideoH265PpsFlags, StdVideoH265ProfileIdc,
    StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN, StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10, StdVideoH265ProfileTierLevel,
    StdVideoH265ProfileTierLevelFlags, StdVideoH265ScalingLists, StdVideoH265SequenceParameterSet, StdVideoH265ShortTermRefPicSet,
    StdVideoH265ShortTermRefPicSetFlags, StdVideoH265SpsFlags, StdVideoH265VideoParameterSet, StdVideoH265VpsFlags,
};
use filmcraft_hevc::accel::{DecodePicture, Reference};
use filmcraft_hevc::params::{Pps, ProfileTierLevel, ScalingList, Sps, StRps, Vps};

/// The std header these structures follow: its name and `VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)`.
pub(crate) const STD_HEADER_NAME: &std::ffi::CStr = c"VK_STD_vulkan_video_codec_h265_decode";
pub(crate) const STD_HEADER_VERSION: u32 = 1 << 22;
/// Entries of each RPS list of the picture info (`STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE`).
const RPS_LIST: usize = 8;
/// An unused RPS list entry.
const NO_SLOT: u8 = 0xff;

fn err(what: &str) -> impl Fn(std::num::TryFromIntError) -> String + '_ {
    move |_| format!("{what} out of range")
}

fn u8_of(v: u32, what: &str) -> Result<u8, String> {
    u8::try_from(v).map_err(err(what))
}

fn i8_of(v: i32, what: &str) -> Result<i8, String> {
    i8::try_from(v).map_err(err(what))
}

fn minus(v: u32, by: u32, what: &str) -> Result<u8, String> {
    v.checked_sub(by).ok_or_else(|| format!("{what} out of range")).and_then(|v| u8_of(v, what))
}

/// The std profile for a stream: Main (8-bit) or Main 10 (8- or 10-bit), as `general_profile_idc`
/// or its compatibility flags say.
pub(crate) fn std_profile(ptl: &ProfileTierLevel, bit_depth: u32) -> Result<StdVideoH265ProfileIdc, String> {
    let compatible = |j: u32| ptl.compatibility & (1 << (31 - j)) != 0;
    let main = ptl.profile_idc == 1 || (ptl.profile_idc == 0 && compatible(1));
    let main10 = ptl.profile_idc == 2 || (ptl.profile_idc == 0 && compatible(2));
    match bit_depth {
        8 if main => Ok(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN),
        8 | 10 if main10 => Ok(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10),
        _ => Err(format!("HEVC general_profile_idc {} at {bit_depth}-bit", ptl.profile_idc)),
    }
}

/// The std level for a `general_level_idc` (30 × the level number).
pub(crate) fn std_level(level_idc: u8) -> Result<StdVideoH265LevelIdc, String> {
    Ok(match level_idc {
        30 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        60 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        63 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        90 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        93 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        120 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        123 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        150 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        153 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        156 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        180 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        183 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        186 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
        l => return Err(format!("HEVC general_level_idc {l}")),
    })
}

fn std_ptl(ptl: &ProfileTierLevel, bit_depth: u32) -> Result<StdVideoH265ProfileTierLevel, String> {
    let mut flags = StdVideoH265ProfileTierLevelFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_general_tier_flag(ptl.tier as u32);
    flags.set_general_progressive_source_flag(ptl.progressive_source as u32);
    flags.set_general_interlaced_source_flag(ptl.interlaced_source as u32);
    flags.set_general_non_packed_constraint_flag(ptl.non_packed_constraint as u32);
    flags.set_general_frame_only_constraint_flag(ptl.frame_only_constraint as u32);
    Ok(StdVideoH265ProfileTierLevel { flags, general_profile_idc: std_profile(ptl, bit_depth)?, general_level_idc: std_level(ptl.level_idc)? })
}

/// The SPS's (highest sub-layer) DPB values for every sub-layer: what an absent
/// `sub_layer_ordering_info` means.
fn dec_pic_buf_mgr(sps: &Sps) -> Result<StdVideoH265DecPicBufMgr, String> {
    Ok(StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: [sps.max_latency_increase_plus1; 7],
        max_dec_pic_buffering_minus1: [minus(sps.max_dec_pic_buffering, 1, "sps_max_dec_pic_buffering")?; 7],
        max_num_reorder_pics: [u8_of(sps.max_num_reorder, "sps_max_num_reorder_pics")?; 7],
    })
}

/// A resolved short-term RPS in its explicit (not predicted) form.
fn std_st_rps(rps: &StRps) -> Result<StdVideoH265ShortTermRefPicSet, String> {
    let flags = StdVideoH265ShortTermRefPicSetFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    let mut out = StdVideoH265ShortTermRefPicSet {
        flags,
        delta_idx_minus1: 0,
        use_delta_flag: 0,
        abs_delta_rps_minus1: 0,
        used_by_curr_pic_flag: 0,
        used_by_curr_pic_s0_flag: 0,
        used_by_curr_pic_s1_flag: 0,
        reserved1: 0,
        reserved2: 0,
        reserved3: 0,
        num_negative_pics: u8::try_from(rps.s0.len()).map_err(err("num_negative_pics"))?,
        num_positive_pics: u8::try_from(rps.s1.len()).map_err(err("num_positive_pics"))?,
        delta_poc_s0_minus1: [0; 16],
        delta_poc_s1_minus1: [0; 16],
    };
    if rps.s0.len() > 16 || rps.s1.len() > 16 {
        return Err("short-term RPS too large".into());
    }
    let step =
        |prev: i32, d: i32| -> Result<u16, String> { u16::try_from(i64::from(prev) - i64::from(d) - 1).map_err(|_| "RPS delta out of range".to_string()) };
    let mut prev = 0;
    for (i, &(d, used)) in rps.s0.iter().enumerate() {
        out.delta_poc_s0_minus1[i] = step(prev, d)?;
        out.used_by_curr_pic_s0_flag |= (used as u16) << i;
        prev = d;
    }
    prev = 0;
    for (i, &(d, used)) in rps.s1.iter().enumerate() {
        out.delta_poc_s1_minus1[i] = step(d, prev)?;
        out.used_by_curr_pic_s1_flag |= (used as u16) << i;
        prev = d;
    }
    Ok(out)
}

/// Resolved scaling lists (`ScalingList[sizeId][matrixId][i]` of 7.3.4, prediction applied) in
/// coded up-right diagonal order, as the std structure takes them: 4x4, 8x8, 16x16 and the two
/// 32x32 lists (matrixId 0 and 3). The DC entries carry the DC values themselves
/// (`scaling_list_dc_coef_minus8 + 8`, 8..=255; the field is unsigned and named for the
/// coefficient); streams with scaling lists are verified on the GPU separately
/// (`filmcraft_platform::vulkan_verification_key`).
fn std_scaling(l: &ScalingList) -> StdVideoH265ScalingLists {
    let mut s = StdVideoH265ScalingLists {
        ScalingList4x4: [[0; 16]; 6],
        ScalingList8x8: l.lists[1],
        ScalingList16x16: l.lists[2],
        ScalingList32x32: [l.lists[3][0], l.lists[3][3]],
        ScalingListDCCoef16x16: l.dc[0],
        ScalingListDCCoef32x32: [l.dc[1][0], l.dc[1][3]],
    };
    for (m, list) in s.ScalingList4x4.iter_mut().enumerate() {
        list.copy_from_slice(&l.lists[0][m][..16]);
    }
    s
}

fn std_lt(sps: &Sps) -> Result<StdVideoH265LongTermRefPicsSps, String> {
    let mut out = StdVideoH265LongTermRefPicsSps { used_by_curr_pic_lt_sps_flag: 0, lt_ref_pic_poc_lsb_sps: [0; 32] };
    if sps.lt_ref_pics.len() > 32 {
        return Err("too many long-term pictures in the SPS".into());
    }
    for (i, &(lsb, used)) in sps.lt_ref_pics.iter().enumerate() {
        out.lt_ref_pic_poc_lsb_sps[i] = lsb;
        out.used_by_curr_pic_lt_sps_flag |= (used as u32) << i;
    }
    Ok(out)
}

fn std_sps(sps: &Sps) -> Result<StdVideoH265SequenceParameterSet, String> {
    let mut flags = StdVideoH265SpsFlags { _bitfield_align_1: [], _bitfield_1: Default::default() };
    flags.set_sps_temporal_id_nesting_flag(sps.temporal_id_nesting as u32);
    flags.set_separate_colour_plane_flag(sps.separate_colour_plane as u32);
    flags.set_conformance_window_flag((sps.conf_win != (0, 0, 0, 0)) as u32);
    flags.set_sps_sub_layer_ordering_info_present_flag(1);
    flags.set_scaling_list_enabled_flag(sps.scaling_list_enabled as u32);
    // as coded: without SPS lists the driver applies the default ones itself
    flags.set_sps_scaling_list_data_present_flag((sps.scaling_list_enabled && sps.scaling_list_data_present) as u32);
    flags.set_amp_enabled_flag(sps.amp as u32);
    flags.set_sample_adaptive_offset_enabled_flag(sps.sao as u32);
    flags.set_pcm_enabled_flag(sps.pcm as u32);
    flags.set_pcm_loop_filter_disabled_flag(sps.pcm_loop_filter_disabled as u32);
    flags.set_long_term_ref_pics_present_flag(sps.long_term_refs_present as u32);
    flags.set_sps_temporal_mvp_enabled_flag(sps.temporal_mvp as u32);
    flags.set_strong_intra_smoothing_enabled_flag(sps.strong_intra_smoothing as u32);
    let (l, r, t, b) = sps.conf_win;
    let pcm = |v: u32, by: u32, what: &str| if sps.pcm { minus(v, by, what) } else { Ok(0) };
    Ok(StdVideoH265SequenceParameterSet {
        flags,
        chroma_format_idc: sps.chroma_format_idc as StdVideoH265ChromaFormatIdc,
        pic_width_in_luma_samples: sps.width,
        pic_height_in_luma_samples: sps.height,
        sps_video_parameter_set_id: sps.vps_id,
        sps_max_sub_layers_minus1: u8_of(sps.max_sub_layers_minus1, "sps_max_sub_layers_minus1")?,
        sps_seq_parameter_set_id: u8_of(sps.id, "sps_seq_parameter_set_id")?,
        bit_depth_luma_minus8: minus(sps.bit_depth_luma, 8, "bit_depth_luma")?,
        bit_depth_chroma_minus8: minus(sps.bit_depth_chroma, 8, "bit_depth_chroma")?,
        log2_max_pic_order_cnt_lsb_minus4: minus(sps.log2_max_poc_lsb, 4, "log2_max_pic_order_cnt_lsb")?,
        log2_min_luma_coding_block_size_minus3: minus(sps.log2_min_cb, 3, "log2_min_luma_coding_block_size")?,
        log2_diff_max_min_luma_coding_block_size: minus(sps.log2_ctb, sps.log2_min_cb, "log2_diff_max_min_luma_coding_block_size")?,
        log2_min_luma_transform_block_size_minus2: minus(sps.log2_min_tb, 2, "log2_min_luma_transform_block_size")?,
        log2_diff_max_min_luma_transform_block_size: minus(sps.log2_max_tb, sps.log2_min_tb, "log2_diff_max_min_luma_transform_block_size")?,
        max_transform_hierarchy_depth_inter: u8_of(sps.max_th_depth_inter, "max_transform_hierarchy_depth_inter")?,
        max_transform_hierarchy_depth_intra: u8_of(sps.max_th_depth_intra, "max_transform_hierarchy_depth_intra")?,
        num_short_term_ref_pic_sets: u8::try_from(sps.st_rps.len()).map_err(err("num_short_term_ref_pic_sets"))?,
        num_long_term_ref_pics_sps: u8::try_from(sps.lt_ref_pics.len()).map_err(err("num_long_term_ref_pics_sps"))?,
        pcm_sample_bit_depth_luma_minus1: pcm(sps.pcm_bit_depth_luma, 1, "pcm_sample_bit_depth_luma")?,
        pcm_sample_bit_depth_chroma_minus1: pcm(sps.pcm_bit_depth_chroma, 1, "pcm_sample_bit_depth_chroma")?,
        log2_min_pcm_luma_coding_block_size_minus3: pcm(sps.log2_min_pcm, 3, "log2_min_pcm_luma_coding_block_size")?,
        log2_diff_max_min_pcm_luma_coding_block_size: pcm(sps.log2_max_pcm, sps.log2_min_pcm, "log2_diff_max_min_pcm_luma_coding_block_size")?,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: l,
        conf_win_right_offset: r,
        conf_win_top_offset: t,
        conf_win_bottom_offset: b,
        pProfileTierLevel: std::ptr::null(),
        pDecPicBufMgr: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: std::ptr::null(),
        pLongTermRefPicsSps: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    })
}

fn std_pps(pps: &Pps, sps: &Sps) -> Result<StdVideoH265PictureParameterSet, String> {
    let mut flags = StdVideoH265PpsFlags { _bitfield_align_1: [], _bitfield_1: Default::default() };
    flags.set_dependent_slice_segments_enabled_flag(pps.dependent_slice_segments_enabled as u32);
    flags.set_output_flag_present_flag(pps.output_flag_present as u32);
    flags.set_sign_data_hiding_enabled_flag(pps.sign_data_hiding as u32);
    flags.set_cabac_init_present_flag(pps.cabac_init_present as u32);
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred as u32);
    flags.set_transform_skip_enabled_flag(pps.transform_skip as u32);
    flags.set_cu_qp_delta_enabled_flag(pps.cu_qp_delta_enabled as u32);
    flags.set_pps_slice_chroma_qp_offsets_present_flag(pps.slice_chroma_qp_offsets_present as u32);
    flags.set_weighted_pred_flag(pps.weighted_pred as u32);
    flags.set_weighted_bipred_flag(pps.weighted_bipred as u32);
    flags.set_transquant_bypass_enabled_flag(pps.transquant_bypass as u32);
    flags.set_tiles_enabled_flag(pps.tiles_enabled as u32);
    flags.set_entropy_coding_sync_enabled_flag(pps.entropy_coding_sync as u32);
    flags.set_uniform_spacing_flag(pps.uniform_spacing as u32);
    flags.set_loop_filter_across_tiles_enabled_flag(pps.loop_filter_across_tiles as u32);
    flags.set_pps_loop_filter_across_slices_enabled_flag(pps.loop_filter_across_slices as u32);
    flags.set_deblocking_filter_control_present_flag(pps.deblocking_control_present as u32);
    flags.set_deblocking_filter_override_enabled_flag(pps.deblocking_override_enabled as u32);
    flags.set_pps_deblocking_filter_disabled_flag(pps.deblocking_disabled as u32);
    flags.set_pps_scaling_list_data_present_flag(pps.scaling_list.is_some() as u32);
    flags.set_lists_modification_present_flag(pps.lists_modification_present as u32);
    flags.set_slice_segment_header_extension_present_flag(pps.slice_header_extension_present as u32);
    let mut out = StdVideoH265PictureParameterSet {
        flags,
        pps_pic_parameter_set_id: u8_of(pps.id, "pps_pic_parameter_set_id")?,
        pps_seq_parameter_set_id: u8_of(pps.sps_id, "pps_seq_parameter_set_id")?,
        sps_video_parameter_set_id: sps.vps_id,
        num_extra_slice_header_bits: u8_of(pps.num_extra_slice_header_bits, "num_extra_slice_header_bits")?,
        num_ref_idx_l0_default_active_minus1: minus(pps.num_ref_idx_l0_default, 1, "num_ref_idx_l0_default_active")?,
        num_ref_idx_l1_default_active_minus1: minus(pps.num_ref_idx_l1_default, 1, "num_ref_idx_l1_default_active")?,
        init_qp_minus26: i8_of(pps.init_qp.saturating_sub(26), "init_qp")?,
        diff_cu_qp_delta_depth: u8_of(pps.diff_cu_qp_delta_depth, "diff_cu_qp_delta_depth")?,
        pps_cb_qp_offset: i8_of(pps.cb_qp_offset, "pps_cb_qp_offset")?,
        pps_cr_qp_offset: i8_of(pps.cr_qp_offset, "pps_cr_qp_offset")?,
        pps_beta_offset_div2: i8_of(pps.beta_offset_div2, "pps_beta_offset_div2")?,
        pps_tc_offset_div2: i8_of(pps.tc_offset_div2, "pps_tc_offset_div2")?,
        log2_parallel_merge_level_minus2: minus(pps.log2_parallel_merge_level, 2, "log2_parallel_merge_level")?,
        log2_max_transform_skip_block_size_minus2: 0,
        diff_cu_chroma_qp_offset_depth: 0,
        chroma_qp_offset_list_len_minus1: 0,
        cb_qp_offset_list: [0; 6],
        cr_qp_offset_list: [0; 6],
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        reserved3: 0,
        pScalingLists: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    };
    if pps.tiles_enabled {
        out.num_tile_columns_minus1 = minus(pps.num_tile_columns, 1, "num_tile_columns")?;
        out.num_tile_rows_minus1 = minus(pps.num_tile_rows, 1, "num_tile_rows")?;
        if !pps.uniform_spacing {
            for (slot, &w) in out.column_width_minus1.iter_mut().zip(&pps.column_widths) {
                *slot = u16::try_from(w.saturating_sub(1)).map_err(err("column_width"))?;
            }
            for (slot, &h) in out.row_height_minus1.iter_mut().zip(&pps.row_heights) {
                *slot = u16::try_from(h.saturating_sub(1)).map_err(err("row_height"))?;
            }
        }
    }
    Ok(out)
}

/// Std VPS / SPS / PPS structures and the storage their pointers point into.
pub(crate) struct ParameterSets {
    pub(crate) vps: Vec<StdVideoH265VideoParameterSet>,
    pub(crate) sps: Vec<StdVideoH265SequenceParameterSet>,
    pub(crate) pps: Vec<StdVideoH265PictureParameterSet>,
    // Pointed to by the structures above: complete before any pointer is taken, never changed
    // after (moving a vector does not move its heap storage).
    _vps_ptl: Vec<StdVideoH265ProfileTierLevel>,
    _vps_dpb: Vec<StdVideoH265DecPicBufMgr>,
    _sps_ptl: Vec<StdVideoH265ProfileTierLevel>,
    _sps_dpb: Vec<StdVideoH265DecPicBufMgr>,
    _sps_lists: Vec<Option<StdVideoH265ScalingLists>>,
    _sps_st: Vec<Vec<StdVideoH265ShortTermRefPicSet>>,
    _sps_lt: Vec<Option<StdVideoH265LongTermRefPicsSps>>,
    _pps_lists: Vec<Option<StdVideoH265ScalingLists>>,
}

/// The std structures for a stream's parameter sets.
pub(crate) fn parameter_sets(vpss: &[Vps], spss: &[Sps], ppss: &[Pps]) -> Result<ParameterSets, String> {
    let depth = |s: &Sps| s.bit_depth_luma;
    // storage first
    let mut vps_ptl = Vec::with_capacity(vpss.len());
    let mut vps_dpb = Vec::with_capacity(vpss.len());
    // VPSs no SPS uses are left out
    let vpss: Vec<&Vps> = vpss.iter().filter(|v| spss.iter().any(|s| s.vps_id == v.id)).collect();
    for s in spss {
        if !vpss.iter().any(|v| v.id == s.vps_id) {
            return Err(format!("SPS {} refers to a missing VPS {}", s.id, s.vps_id));
        }
    }
    for v in &vpss {
        // the VPS's own sub-layer DPB values are not kept; those of an SPS using it are equivalent
        let sps = spss.iter().find(|s| s.vps_id == v.id).ok_or_else(|| format!("VPS {} is not used by any SPS", v.id))?;
        vps_ptl.push(std_ptl(&v.ptl, depth(sps))?);
        vps_dpb.push(dec_pic_buf_mgr(sps)?);
    }
    let mut sps_ptl = Vec::with_capacity(spss.len());
    let mut sps_dpb = Vec::with_capacity(spss.len());
    let mut sps_st = Vec::with_capacity(spss.len());
    let mut sps_lt = Vec::with_capacity(spss.len());
    for s in spss {
        sps_ptl.push(std_ptl(&s.ptl, depth(s))?);
        sps_dpb.push(dec_pic_buf_mgr(s)?);
        sps_st.push(s.st_rps.iter().map(std_st_rps).collect::<Result<Vec<_>, _>>()?);
        sps_lt.push(if s.long_term_refs_present { Some(std_lt(s)?) } else { None });
    }
    let sps_lists: Vec<Option<StdVideoH265ScalingLists>> =
        spss.iter().map(|s| s.scaling_list.as_ref().filter(|_| s.scaling_list_enabled && s.scaling_list_data_present).map(std_scaling)).collect();
    let pps_lists: Vec<Option<StdVideoH265ScalingLists>> = ppss.iter().map(|p| p.scaling_list.as_ref().map(std_scaling)).collect();

    let mut vps = Vec::with_capacity(vpss.len());
    for ((v, ptl), dpb) in vpss.iter().zip(&vps_ptl).zip(&vps_dpb) {
        let mut flags = StdVideoH265VpsFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
        flags.set_vps_temporal_id_nesting_flag(v.temporal_id_nesting as u32);
        flags.set_vps_sub_layer_ordering_info_present_flag(1);
        vps.push(StdVideoH265VideoParameterSet {
            flags,
            vps_video_parameter_set_id: v.id,
            vps_max_sub_layers_minus1: u8_of(v.max_sub_layers_minus1, "vps_max_sub_layers_minus1")?,
            reserved1: 0,
            reserved2: 0,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            reserved3: 0,
            pDecPicBufMgr: dpb,
            pHrdParameters: std::ptr::null(),
            pProfileTierLevel: ptl,
        });
    }
    let mut sps = Vec::with_capacity(spss.len());
    for (i, s) in spss.iter().enumerate() {
        let mut std = std_sps(s)?;
        if let (Some(ptl), Some(dpb), Some(st), Some(lt), Some(lists)) = (sps_ptl.get(i), sps_dpb.get(i), sps_st.get(i), sps_lt.get(i), sps_lists.get(i)) {
            std.pProfileTierLevel = ptl;
            std.pDecPicBufMgr = dpb;
            if !st.is_empty() {
                std.pShortTermRefPicSet = st.as_ptr();
            }
            if let Some(lt) = lt {
                std.pLongTermRefPicsSps = lt;
            }
            if let Some(l) = lists {
                std.pScalingLists = l;
            }
        }
        sps.push(std);
    }
    let mut pps = Vec::with_capacity(ppss.len());
    for (p, lists) in ppss.iter().zip(&pps_lists) {
        let s = spss.iter().find(|s| s.id == p.sps_id).ok_or_else(|| format!("PPS {} refers to a missing SPS {}", p.id, p.sps_id))?;
        let mut std = std_pps(p, s)?;
        if let Some(l) = lists {
            std.pScalingLists = l;
        }
        pps.push(std);
    }
    Ok(ParameterSets {
        vps,
        sps,
        pps,
        _vps_ptl: vps_ptl,
        _vps_dpb: vps_dpb,
        _sps_ptl: sps_ptl,
        _sps_dpb: sps_dpb,
        _sps_lists: sps_lists,
        _sps_st: sps_st,
        _sps_lt: sps_lt,
        _pps_lists: pps_lists,
    })
}

/// The picture's std decode info; `slot` maps a picture id to its DPB slot.
pub(crate) fn picture_info(p: &DecodePicture, slot: impl Fn(u32) -> Option<u32>) -> Result<StdVideoDecodeH265PictureInfo, String> {
    let mut flags = StdVideoDecodeH265PictureInfoFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_IrapPicFlag(p.irap as u32);
    flags.set_IdrPicFlag(p.idr as u32);
    // Every decoded picture is marked "used for short-term reference" (8.1.3); later RPSs unmark it.
    flags.set_IsReference(1);
    flags.set_short_term_ref_pic_set_sps_flag(p.st_rps_sps as u32);
    let list = |ids: &[u32], what: &str| -> Result<[u8; RPS_LIST], String> {
        if ids.len() > RPS_LIST {
            return Err(format!("{} pictures in {what}", ids.len()));
        }
        let mut out = [NO_SLOT; RPS_LIST];
        for (o, id) in out.iter_mut().zip(ids) {
            let s = slot(*id).ok_or_else(|| format!("{what} picture {id} was never decoded"))?;
            *o = u8::try_from(s).map_err(err("DPB slot"))?;
        }
        Ok(out)
    };
    Ok(StdVideoDecodeH265PictureInfo {
        flags,
        sps_video_parameter_set_id: p.sps.vps_id,
        pps_seq_parameter_set_id: u8_of(p.sps.id, "pps_seq_parameter_set_id")?,
        pps_pic_parameter_set_id: u8_of(p.pps.id, "pps_pic_parameter_set_id")?,
        NumDeltaPocsOfRefRpsIdx: u8_of(p.st_rps_ref_delta_pocs, "NumDeltaPocsOfRefRpsIdx")?,
        PicOrderCntVal: p.poc,
        NumBitsForSTRefPicSetInSlice: u16::try_from(p.st_rps_bits).map_err(err("NumBitsForSTRefPicSetInSlice"))?,
        reserved: 0,
        RefPicSetStCurrBefore: list(&p.st_curr_before, "RefPicSetStCurrBefore")?,
        RefPicSetStCurrAfter: list(&p.st_curr_after, "RefPicSetStCurrAfter")?,
        RefPicSetLtCurr: list(&p.lt_curr, "RefPicSetLtCurr")?,
    })
}

/// A reference picture's std info.
pub(crate) fn reference_info(r: &Reference) -> StdVideoDecodeH265ReferenceInfo {
    let mut flags = StdVideoDecodeH265ReferenceInfoFlags { _bitfield_align_1: [], _bitfield_1: Default::default(), __bindgen_padding_0: [0; 3] };
    flags.set_used_for_long_term_reference(r.long_term as u32);
    StdVideoDecodeH265ReferenceInfo { flags, PicOrderCntVal: r.poc }
}

/// The std info of the slot the picture is decoded into (a short-term reference).
pub(crate) fn setup_info(p: &DecodePicture) -> StdVideoDecodeH265ReferenceInfo {
    reference_info(&Reference { id: p.id, poc: p.poc, long_term: false, non_existing: false })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use filmcraft_hevc::accel::{Event, Frontend};

    use super::*;
    use crate::vulkan::trace_headers::{Field, get, trace};

    #[test]
    fn levels_profiles_and_rps() {
        assert_eq!(std_level(153), Ok(StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1));
        assert!(std_level(0).is_err() && std_level(255).is_err());
        let ptl = |idc, compat| ProfileTierLevel { profile_idc: idc, compatibility: compat, ..Default::default() };
        assert_eq!(std_profile(&ptl(1, 0), 8), Ok(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN));
        assert_eq!(std_profile(&ptl(2, 0), 10), Ok(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10));
        assert_eq!(std_profile(&ptl(0, 1 << 29), 10), Ok(StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10), "by compatibility flag");
        assert!(std_profile(&ptl(1, 0), 10).is_err(), "Main is 8-bit");
        assert!(std_profile(&ptl(4, 0), 8).is_err(), "range extensions");
        // DeltaPocS0 -1, -3, -4 / S1 +2: explicit differences
        let rps = StRps { s0: vec![(-1, true), (-3, false), (-4, true)], s1: vec![(2, true)] };
        let s = std_st_rps(&rps).unwrap();
        assert_eq!((s.num_negative_pics, s.num_positive_pics), (3, 1));
        assert_eq!(&s.delta_poc_s0_minus1[..3], &[0, 1, 0]);
        assert_eq!(s.delta_poc_s1_minus1[0], 1);
        assert_eq!((s.used_by_curr_pic_s0_flag, s.used_by_curr_pic_s1_flag), (0b101, 0b1));
        assert_eq!(s.flags.inter_ref_pic_set_prediction_flag(), 0);
    }

    /// An HM-format scaling list file for libx265: raster-order matrices and the DC values of the
    /// 16x16 / 32x32 lists (some below 8, so `scaling_list_dc_coef_minus8` is negative), with one
    /// 8x8 list equal to the one before it so the encoder codes it as predicted.
    fn scaling_list_file() -> String {
        let mut out = String::new();
        let mut prev: Vec<Vec<u32>> = Vec::new();
        for (size, n) in [("4X4", 4u32), ("8X8", 8), ("16X16", 8), ("32X32", 8)] {
            for (k, mode) in ["INTRA", "INTER"].into_iter().enumerate() {
                let comps: &[&str] = if size == "32X32" { &["LUMA"] } else { &["LUMA", "CHROMAU", "CHROMAV"] };
                for (c, comp) in comps.iter().enumerate() {
                    let seed = (k * 3 + c) as u32;
                    let mut m: Vec<Vec<u32>> = (0..n).map(|y| (0..n).map(|x| 8 + 3 * seed + 2 * (x + y) + (x * 7 + y * 3 + seed) % 5).collect()).collect();
                    if size == "8X8" && mode == "INTER" && *comp == "CHROMAV" {
                        m = prev.clone();
                    }
                    out += &format!("{mode}{size}_{comp} =\n");
                    for row in &m {
                        out += &row.iter().map(|v| format!("{v},")).collect::<String>();
                        out += "\n";
                    }
                    if matches!(size, "16X16" | "32X32") {
                        out += &format!("{mode}{size}_{comp}_DC =\n{},\n", 5 + 6 * seed);
                    }
                    prev = m;
                }
            }
        }
        out
    }

    /// `ScalingList[sizeId][matrixId]` and the DC value of every list an SPS codes, derived from
    /// the trace's syntax elements (7.3.4 / 7.4.5: delta-coded lists, prediction from a reference
    /// list; the default lists of Tables 7-5 / 7-6 by way of `ScalingList::default_lists`).
    fn coded_lists(t: &[Field]) -> HashMap<(usize, usize), (Vec<u8>, u8)> {
        let defaults = ScalingList::default_lists();
        let mut out: HashMap<(usize, usize), (Vec<u8>, u8)> = HashMap::new();
        for size in 0..4usize {
            let step = if size == 3 { 3 } else { 1 };
            let n = if size == 0 { 16 } else { 64 };
            for m in (0..6).step_by(step) {
                let list = if get(t, &format!("scaling_list_pred_mode_flag[{size}][{m}]"), -1) == 0 {
                    let delta = get(t, &format!("scaling_list_pred_matrix_id_delta[{size}][{m}]"), -1) as usize * step;
                    if delta == 0 { (defaults.lists[size][m][..n].to_vec(), 16) } else { out[&(size, m - delta)].clone() }
                } else {
                    let mut next = if size > 1 { get(t, &format!("scaling_list_dc_coef_minus8[{}][{m}]", size - 2), -100) + 8 } else { 8 };
                    let dc = if size > 1 { next as u8 } else { 16 };
                    let list = (0..n)
                        .map(|i| {
                            next = (next + get(t, &format!("scaling_list_delta_coeff[{size}][{m}][{i}]"), -1000) + 256).rem_euclid(256);
                            next as u8
                        })
                        .collect();
                    (list, dc)
                };
                out.insert((size, m), list);
            }
        }
        out
    }

    /// Bits of a slice header's short-term RPS (`st_ref_pic_set(num_short_term_ref_pic_sets)`):
    /// from the element after `short_term_ref_pic_set_sps_flag` to the first one after the set.
    fn st_rps_bits(t: &[Field]) -> u64 {
        const RPS: [&str; 12] = [
            "inter_ref_pic_set_prediction_flag",
            "delta_idx_minus1",
            "delta_rps_sign",
            "abs_delta_rps_minus1",
            "used_by_curr_pic_flag",
            "use_delta_flag",
            "num_negative_pics",
            "num_positive_pics",
            "delta_poc_s0_minus1",
            "used_by_curr_pic_s0_flag",
            "delta_poc_s1_minus1",
            "used_by_curr_pic_s1_flag",
        ];
        let Some(i) = t.iter().position(|f| f.name == "short_term_ref_pic_set_sps_flag") else { return 0 };
        if t[i].value == 1 {
            return 0;
        }
        let end = t[i + 1..].iter().find(|f| !RPS.contains(&f.name.split('[').next().unwrap_or_default())).unwrap();
        end.pos - t[i].pos - 1
    }

    /// DeltaPocS0 / DeltaPocS1 with their used_by_curr_pic flags, of a slice header's explicitly
    /// coded short-term RPS (7.4.8).
    fn explicit_rps(t: &[Field]) -> (Vec<(i32, bool)>, Vec<(i32, bool)>) {
        let side = |count: &str, delta: &str, used: &str, sign: i32| {
            let mut poc = 0;
            (0..get(t, count, 0))
                .map(|i| {
                    poc += sign * (get(t, &format!("{delta}[{i}]"), -1) as i32 + 1);
                    (poc, get(t, &format!("{used}[{i}]"), -1) == 1)
                })
                .collect::<Vec<_>>()
        };
        (
            side("num_negative_pics", "delta_poc_s0_minus1", "used_by_curr_pic_s0_flag", -1),
            side("num_positive_pics", "delta_poc_s1_minus1", "used_by_curr_pic_s1_flag", 1),
        )
    }

    /// The std VPS / SPS / PPS, scaling lists and per-picture structures carry what the bitstream
    /// says, field by field, on libx265 streams covering open GOPs (CRA with RASL pictures), a
    /// conformance window, several slices, B-pyramids, Main 10, transform skip, transquant bypass,
    /// constrained intra, deblocking and chroma QP offsets, default and custom scaling lists (DC
    /// values below 8, a predicted list); ffmpeg's header trace is the external reference
    /// (generator and oracle only). libx265 codes every short-term RPS in the slice headers, so the
    /// SPS-coded and predicted RPS forms are covered by `levels_profiles_and_rps` and the GPU tests.
    #[test]
    fn std_structures_match_ffmpeg_trace_headers() {
        let Some(ff) = filmcraft_testkit::ffmpeg() else {
            eprintln!("SKIPPED: no ffmpeg");
            return;
        };
        let dir = filmcraft_testkit::fixtures_dir("platform");
        let lists = dir.join("trace_hevc_lists.txt");
        std::fs::write(&lists, scaling_list_file()).unwrap();
        let streams: [(&str, &str, &str, String); 3] = [
            (
                "trace_hevc_open_gop.hevc",
                "318x238",
                "yuv420p",
                "keyint=12:min-keyint=12:scenecut=0:bframes=3:b-pyramid=1:open-gop=1:ref=3:slices=2:weightb=1:scaling-list=default".into(),
            ),
            (
                "trace_hevc_main10_tools.hevc",
                "320x240",
                "yuv420p10le",
                "keyint=8:bframes=2:amp=1:rect=1:tskip=1:cu-lossless=1:constrained-intra=1:wpp=0:signhide=0:sao=0:strong-intra-smoothing=0:deblock=-2,1:cbqpoffs=2:crqpoffs=-1".into(),
            ),
            ("trace_hevc_lists.hevc", "320x240", "yuv420p", format!("keyint=8:bframes=2:scaling-list={}", lists.display())),
        ];
        for (name, size, pix_fmt, params) in streams {
            let made = filmcraft_testkit::fixtures::generate(&dir.join(name), |tmp| {
                std::process::Command::new(&ff)
                    .args(["-y", "-v", "error", "-f", "lavfi", "-i", &format!("testsrc2=s={size}:r=25:d=1.2,noise=alls=10:allf=t")])
                    .args(["-c:v", "libx265", "-pix_fmt", pix_fmt, "-x265-params", &format!("log-level=error:{params}"), "-f", "hevc"])
                    .arg(tmp)
                    .status()
                    .is_ok_and(|s| s.success())
            });
            let Some(path) = made else { panic!("{name}: fixture generation failed") };
            let data = std::fs::read(&path).unwrap();
            let blocks = trace(&ff, &path);
            let block =
                |title: &str| blocks.iter().find(|b| b.title == title).map(|b| b.fields.clone()).unwrap_or_else(|| panic!("{name}: no {title} in the trace"));
            let (tv, ts, tp) = (block("Video Parameter Set"), block("Sequence Parameter Set"), block("Picture Parameter Set"));

            let (mut vps, mut sps, mut pps) = (None, None, None);
            for nal in filmcraft_bitstream::annexb_nals(&data) {
                let rbsp = filmcraft_bitstream::unescape_rbsp(&nal[2..]);
                match (nal[0] >> 1) & 0x3f {
                    32 if vps.is_none() => vps = Some(Vps::parse(&rbsp).unwrap()),
                    33 if sps.is_none() => sps = Some(Sps::parse(&rbsp).unwrap()),
                    34 if pps.is_none() => pps = Some(Pps::parse(&rbsp).unwrap()),
                    _ => {}
                }
            }
            let (vps, sps, pps) = (vps.unwrap(), sps.unwrap(), pps.unwrap());
            let sets = parameter_sets(std::slice::from_ref(&vps), std::slice::from_ref(&sps), std::slice::from_ref(&pps)).unwrap();

            let check_ptl = |what: &str, ptl: &StdVideoH265ProfileTierLevel, t: &[Field]| {
                assert_eq!(ptl.general_profile_idc as i64, get(t, "general_profile_idc", -1), "{name}: {what} general_profile_idc");
                assert_eq!(ptl.general_level_idc, std_level(get(t, "general_level_idc", -1) as u8).unwrap(), "{name}: {what} general_level_idc");
                let f = &ptl.flags;
                for (field, v) in [
                    ("general_tier_flag", f.general_tier_flag()),
                    ("general_progressive_source_flag", f.general_progressive_source_flag()),
                    ("general_interlaced_source_flag", f.general_interlaced_source_flag()),
                    ("general_non_packed_constraint_flag", f.general_non_packed_constraint_flag()),
                    ("general_frame_only_constraint_flag", f.general_frame_only_constraint_flag()),
                ] {
                    assert_eq!(v as i64, get(t, field, -1), "{name}: {what} {field}");
                }
            };
            // the values of the highest sub-layer, the one decoded
            let check_dpb = |what: &str, dpb: &StdVideoH265DecPicBufMgr, t: &[Field], prefix: &str, top: usize| {
                for (field, v) in [
                    ("max_dec_pic_buffering_minus1", dpb.max_dec_pic_buffering_minus1[top] as i64),
                    ("max_num_reorder_pics", dpb.max_num_reorder_pics[top] as i64),
                    ("max_latency_increase_plus1", dpb.max_latency_increase_plus1[top] as i64),
                ] {
                    assert_eq!(v, get(t, &format!("{prefix}_{field}[{top}]"), -1), "{name}: {what} {field}");
                }
            };

            let v = &sets.vps[0];
            assert_eq!(v.vps_video_parameter_set_id as i64, get(&tv, "vps_video_parameter_set_id", -1), "{name}");
            assert_eq!(v.vps_max_sub_layers_minus1 as i64, get(&tv, "vps_max_sub_layers_minus1", -1), "{name}");
            assert_eq!(v.flags.vps_temporal_id_nesting_flag() as i64, get(&tv, "vps_temporal_id_nesting_flag", -1), "{name}");
            assert!(std::ptr::eq(v.pProfileTierLevel, &sets._vps_ptl[0]) && std::ptr::eq(v.pDecPicBufMgr, &sets._vps_dpb[0]), "{name}");
            check_ptl("VPS", &sets._vps_ptl[0], &tv);
            check_dpb("VPS", &sets._vps_dpb[0], &tv, "vps", v.vps_max_sub_layers_minus1 as usize);

            let s = &sets.sps[0];
            let f = &s.flags;
            assert!(std::ptr::eq(s.pProfileTierLevel, &sets._sps_ptl[0]) && std::ptr::eq(s.pDecPicBufMgr, &sets._sps_dpb[0]), "{name}");
            check_ptl("SPS", &sets._sps_ptl[0], &ts);
            check_dpb("SPS", &sets._sps_dpb[0], &ts, "sps", s.sps_max_sub_layers_minus1 as usize);
            let got_sps = [
                ("sps_video_parameter_set_id", s.sps_video_parameter_set_id as i64),
                ("sps_max_sub_layers_minus1", s.sps_max_sub_layers_minus1 as i64),
                ("sps_temporal_id_nesting_flag", f.sps_temporal_id_nesting_flag() as i64),
                ("sps_seq_parameter_set_id", s.sps_seq_parameter_set_id as i64),
                ("chroma_format_idc", s.chroma_format_idc as i64),
                ("separate_colour_plane_flag", f.separate_colour_plane_flag() as i64),
                ("pic_width_in_luma_samples", s.pic_width_in_luma_samples as i64),
                ("pic_height_in_luma_samples", s.pic_height_in_luma_samples as i64),
                ("conformance_window_flag", f.conformance_window_flag() as i64),
                ("conf_win_left_offset", s.conf_win_left_offset as i64),
                ("conf_win_right_offset", s.conf_win_right_offset as i64),
                ("conf_win_top_offset", s.conf_win_top_offset as i64),
                ("conf_win_bottom_offset", s.conf_win_bottom_offset as i64),
                ("bit_depth_luma_minus8", s.bit_depth_luma_minus8 as i64),
                ("bit_depth_chroma_minus8", s.bit_depth_chroma_minus8 as i64),
                ("log2_max_pic_order_cnt_lsb_minus4", s.log2_max_pic_order_cnt_lsb_minus4 as i64),
                ("log2_min_luma_coding_block_size_minus3", s.log2_min_luma_coding_block_size_minus3 as i64),
                ("log2_diff_max_min_luma_coding_block_size", s.log2_diff_max_min_luma_coding_block_size as i64),
                ("log2_min_luma_transform_block_size_minus2", s.log2_min_luma_transform_block_size_minus2 as i64),
                ("log2_diff_max_min_luma_transform_block_size", s.log2_diff_max_min_luma_transform_block_size as i64),
                ("max_transform_hierarchy_depth_inter", s.max_transform_hierarchy_depth_inter as i64),
                ("max_transform_hierarchy_depth_intra", s.max_transform_hierarchy_depth_intra as i64),
                ("scaling_list_enabled_flag", f.scaling_list_enabled_flag() as i64),
                ("sps_scaling_list_data_present_flag", f.sps_scaling_list_data_present_flag() as i64),
                ("amp_enabled_flag", f.amp_enabled_flag() as i64),
                ("sample_adaptive_offset_enabled_flag", f.sample_adaptive_offset_enabled_flag() as i64),
                ("pcm_enabled_flag", f.pcm_enabled_flag() as i64),
                ("pcm_sample_bit_depth_luma_minus1", s.pcm_sample_bit_depth_luma_minus1 as i64),
                ("pcm_sample_bit_depth_chroma_minus1", s.pcm_sample_bit_depth_chroma_minus1 as i64),
                ("log2_min_pcm_luma_coding_block_size_minus3", s.log2_min_pcm_luma_coding_block_size_minus3 as i64),
                ("log2_diff_max_min_pcm_luma_coding_block_size", s.log2_diff_max_min_pcm_luma_coding_block_size as i64),
                ("pcm_loop_filter_disabled_flag", f.pcm_loop_filter_disabled_flag() as i64),
                ("num_short_term_ref_pic_sets", s.num_short_term_ref_pic_sets as i64),
                ("long_term_ref_pics_present_flag", f.long_term_ref_pics_present_flag() as i64),
                ("num_long_term_ref_pics_sps", s.num_long_term_ref_pics_sps as i64),
                ("sps_temporal_mvp_enabled_flag", f.sps_temporal_mvp_enabled_flag() as i64),
                ("strong_intra_smoothing_enabled_flag", f.strong_intra_smoothing_enabled_flag() as i64),
            ];
            for (field, v) in got_sps {
                assert_eq!(v, get(&ts, field, 0), "{name}: SPS {field}");
            }
            assert_eq!(s.pShortTermRefPicSet.is_null(), s.num_short_term_ref_pic_sets == 0, "{name}");
            assert!(s.pSequenceParameterSetVui.is_null() && f.vui_parameters_present_flag() == 0, "{name}: no VUI (decoding does not need it)");

            // scaling lists: given exactly when the SPS codes them, resolved, in coded order
            let coded = get(&ts, "sps_scaling_list_data_present_flag", 0) == 1;
            assert_eq!(coded, name == "trace_hevc_lists.hevc", "{name}: only the custom-list stream codes lists");
            assert_eq!(!s.pScalingLists.is_null(), coded, "{name}: SPS lists given exactly when coded");
            if coded {
                let l = sets._sps_lists[0].as_ref().unwrap();
                assert!(std::ptr::eq(s.pScalingLists, l), "{name}");
                let want = coded_lists(&ts);
                for m in 0..6 {
                    assert_eq!(l.ScalingList4x4[m].to_vec(), want[&(0, m)].0, "{name}: 4x4 list {m}");
                    assert_eq!(l.ScalingList8x8[m].to_vec(), want[&(1, m)].0, "{name}: 8x8 list {m}");
                    assert_eq!(l.ScalingList16x16[m].to_vec(), want[&(2, m)].0, "{name}: 16x16 list {m}");
                    assert_eq!(l.ScalingListDCCoef16x16[m], want[&(2, m)].1, "{name}: 16x16 DC {m}");
                }
                for (k, m) in [0, 3].into_iter().enumerate() {
                    assert_eq!(l.ScalingList32x32[k].to_vec(), want[&(3, m)].0, "{name}: 32x32 list {m}");
                    assert_eq!(l.ScalingListDCCoef32x32[k], want[&(3, m)].1, "{name}: 32x32 DC {m}");
                }
                assert!(want.values().any(|(_, dc)| *dc < 8), "{name}: a DC value below 8");
                assert!(ts.iter().any(|f| f.name.starts_with("scaling_list_pred_mode_flag") && f.value == 0), "{name}: a predicted list");
            }

            let p = &sets.pps[0];
            let f = &p.flags;
            let got_pps = [
                ("pps_pic_parameter_set_id", p.pps_pic_parameter_set_id as i64, 0),
                ("pps_seq_parameter_set_id", p.pps_seq_parameter_set_id as i64, 0),
                ("dependent_slice_segments_enabled_flag", f.dependent_slice_segments_enabled_flag() as i64, 0),
                ("output_flag_present_flag", f.output_flag_present_flag() as i64, 0),
                ("num_extra_slice_header_bits", p.num_extra_slice_header_bits as i64, 0),
                ("sign_data_hiding_enabled_flag", f.sign_data_hiding_enabled_flag() as i64, 0),
                ("cabac_init_present_flag", f.cabac_init_present_flag() as i64, 0),
                ("num_ref_idx_l0_default_active_minus1", p.num_ref_idx_l0_default_active_minus1 as i64, 0),
                ("num_ref_idx_l1_default_active_minus1", p.num_ref_idx_l1_default_active_minus1 as i64, 0),
                ("init_qp_minus26", p.init_qp_minus26 as i64, 0),
                ("constrained_intra_pred_flag", f.constrained_intra_pred_flag() as i64, 0),
                ("transform_skip_enabled_flag", f.transform_skip_enabled_flag() as i64, 0),
                ("cu_qp_delta_enabled_flag", f.cu_qp_delta_enabled_flag() as i64, 0),
                ("diff_cu_qp_delta_depth", p.diff_cu_qp_delta_depth as i64, 0),
                ("pps_cb_qp_offset", p.pps_cb_qp_offset as i64, 0),
                ("pps_cr_qp_offset", p.pps_cr_qp_offset as i64, 0),
                ("pps_slice_chroma_qp_offsets_present_flag", f.pps_slice_chroma_qp_offsets_present_flag() as i64, 0),
                ("weighted_pred_flag", f.weighted_pred_flag() as i64, 0),
                ("weighted_bipred_flag", f.weighted_bipred_flag() as i64, 0),
                ("transquant_bypass_enabled_flag", f.transquant_bypass_enabled_flag() as i64, 0),
                ("tiles_enabled_flag", f.tiles_enabled_flag() as i64, 0),
                ("entropy_coding_sync_enabled_flag", f.entropy_coding_sync_enabled_flag() as i64, 0),
                ("uniform_spacing_flag", f.uniform_spacing_flag() as i64, 1),
                ("loop_filter_across_tiles_enabled_flag", f.loop_filter_across_tiles_enabled_flag() as i64, 1),
                ("pps_loop_filter_across_slices_enabled_flag", f.pps_loop_filter_across_slices_enabled_flag() as i64, 0),
                ("deblocking_filter_control_present_flag", f.deblocking_filter_control_present_flag() as i64, 0),
                ("deblocking_filter_override_enabled_flag", f.deblocking_filter_override_enabled_flag() as i64, 0),
                ("pps_deblocking_filter_disabled_flag", f.pps_deblocking_filter_disabled_flag() as i64, 0),
                ("pps_beta_offset_div2", p.pps_beta_offset_div2 as i64, 0),
                ("pps_tc_offset_div2", p.pps_tc_offset_div2 as i64, 0),
                ("pps_scaling_list_data_present_flag", f.pps_scaling_list_data_present_flag() as i64, 0),
                ("lists_modification_present_flag", f.lists_modification_present_flag() as i64, 0),
                ("log2_parallel_merge_level_minus2", p.log2_parallel_merge_level_minus2 as i64, 0),
                ("slice_segment_header_extension_present_flag", f.slice_segment_header_extension_present_flag() as i64, 0),
            ];
            for (field, v, absent) in got_pps {
                assert_eq!(v, get(&tp, field, absent), "{name}: PPS {field}");
            }
            assert_eq!(p.sps_video_parameter_set_id, s.sps_video_parameter_set_id, "{name}: PPS key");
            assert!(p.pScalingLists.is_null(), "{name}: no PPS lists");

            // pictures, in decoding order: the trace's first slice segment of each picture
            let pictures: Vec<&Vec<Field>> = blocks
                .iter()
                .filter(|b| b.title == "Slice Segment Header" && get(&b.fields, "first_slice_segment_in_pic_flag", -1) == 1)
                .map(|b| &b.fields)
                .collect();
            let mut fe = Frontend::new();
            let mut events = fe.decode(&data, 0).unwrap();
            events.extend(fe.flush());
            let decodes: Vec<&DecodePicture> = events
                .iter()
                .filter_map(|e| match e {
                    Event::Decode(p) => Some(p),
                    Event::Output(_) => None,
                })
                .collect();
            assert_eq!(decodes.len(), pictures.len(), "{name}: picture count");
            assert!(decodes.len() >= 25, "{name}: {} pictures", decodes.len());
            let max_lsb = 1i64 << (get(&ts, "log2_max_pic_order_cnt_lsb_minus4", -1) + 4);
            let slot = |id: u32| id % 15;
            let (mut rasl, mut multi_slice) = (0, 0);
            for (i, (d, t)) in decodes.iter().zip(&pictures).enumerate() {
                let at = format!("{name}: picture {i}");
                let info = picture_info(d, |id| Some(slot(id))).unwrap();
                let nal_type = get(t, "nal_unit_type", -1);
                rasl += usize::from(matches!(nal_type, 8 | 9));
                multi_slice += usize::from(d.slices.len() > 1);
                assert_eq!(info.flags.IrapPicFlag() == 1, (16..=23).contains(&nal_type), "{at}: IRAP");
                assert_eq!(info.flags.IdrPicFlag() == 1, matches!(nal_type, 19 | 20), "{at}: IDR");
                assert_eq!(info.flags.IsReference(), 1, "{at}: every decoded picture is a short-term reference");
                assert_eq!(info.flags.short_term_ref_pic_set_sps_flag() as i64, get(t, "short_term_ref_pic_set_sps_flag", 0), "{at}");
                assert_eq!(info.pps_pic_parameter_set_id as i64, get(t, "slice_pic_parameter_set_id", -1), "{at}");
                assert_eq!(
                    (info.sps_video_parameter_set_id, info.pps_seq_parameter_set_id),
                    (p.sps_video_parameter_set_id, p.pps_seq_parameter_set_id),
                    "{at}"
                );
                assert_eq!(i64::from(info.PicOrderCntVal).rem_euclid(max_lsb), get(t, "slice_pic_order_cnt_lsb", 0), "{at}: POC");
                assert_eq!(info.NumBitsForSTRefPicSetInSlice as u64, st_rps_bits(t), "{at}: NumBitsForSTRefPicSetInSlice");
                assert!(t.iter().all(|f| f.name != "inter_ref_pic_set_prediction_flag" || f.value == 0), "{at}: libx265 codes RPSs explicitly");
                assert_eq!(info.NumDeltaPocsOfRefRpsIdx, 0, "{at}: no predicted RPS");

                // the RPS: current lists in the bitstream's order; the references are the whole RPS
                let (s0, s1) = explicit_rps(t);
                let poc_of = |id: &u32| d.refs.iter().find(|r| r.id == *id).map(|r| r.poc).unwrap();
                let used = |side: &[(i32, bool)]| side.iter().filter(|(_, u)| *u).map(|(dp, _)| d.poc + dp).collect::<Vec<_>>();
                assert_eq!(d.st_curr_before.iter().map(poc_of).collect::<Vec<_>>(), used(&s0), "{at}: StCurrBefore");
                assert_eq!(d.st_curr_after.iter().map(poc_of).collect::<Vec<_>>(), used(&s1), "{at}: StCurrAfter");
                assert!(d.lt_curr.is_empty(), "{at}");
                let mut have: Vec<i32> = d.refs.iter().map(|r| r.poc).collect();
                let mut want: Vec<i32> = s0.iter().chain(&s1).map(|(dp, _)| d.poc + dp).collect();
                have.sort_unstable();
                want.sort_unstable();
                assert_eq!(have, want, "{at}: the references are the RPS");
                let slots = |ids: &[u32]| {
                    let mut v = [NO_SLOT; RPS_LIST];
                    for (o, id) in v.iter_mut().zip(ids) {
                        *o = slot(*id) as u8;
                    }
                    v
                };
                assert_eq!(info.RefPicSetStCurrBefore, slots(&d.st_curr_before), "{at}");
                assert_eq!(info.RefPicSetStCurrAfter, slots(&d.st_curr_after), "{at}");
                assert_eq!(info.RefPicSetLtCurr, [NO_SLOT; RPS_LIST], "{at}");
                assert_eq!(info.PicOrderCntVal, d.poc, "{at}");
                let setup = setup_info(d);
                assert_eq!((setup.PicOrderCntVal, setup.flags.used_for_long_term_reference()), (d.poc, 0), "{at}");
                for r in &d.refs {
                    let ri = reference_info(r);
                    assert_eq!((ri.PicOrderCntVal, ri.flags.used_for_long_term_reference()), (r.poc, r.long_term as u32), "{at}");
                }
            }
            if name == "trace_hevc_open_gop.hevc" {
                assert!(rasl > 0 && multi_slice == decodes.len(), "{name}: RASL pictures ({rasl}) and two slices per picture");
            }
        }
    }
}
