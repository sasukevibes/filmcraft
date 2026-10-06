//! The hardware front-end (`accel::Frontend`) against the software decoder on the libx264
//! fixtures: the same pictures come out in the same order with the same pts, POC, key flag, crop,
//! colour and aspect; every picture is decoded once before it is output, and output at most once;
//! references and the DPB list only name decoded pictures; and a hardware decoder with
//! `max_dpb_frames + 1` picture slots, freeing a slot when its picture leaves the DPB list (the
//! rule documented in `accel`), never runs out of slots and never overwrites a picture that is
//! still referenced or not yet output. Skipped without ffmpeg (fixture generator).

mod common;

use std::collections::{HashMap, HashSet};

use filmcraft_h264::accel::{Event, Frontend};

/// Run the front-end over a whole Annex B stream (pts = access unit index).
fn frontend_events(data: &[u8], name: &str) -> Vec<Event> {
    let mut fe = Frontend::new();
    let mut events = Vec::new();
    for (i, au) in common::split_access_units(data).into_iter().enumerate() {
        events.extend(fe.decode(au, i as i64).unwrap_or_else(|e| panic!("{name}: access unit {i}: {e}")));
    }
    events.extend(fe.flush());
    events
}

/// Check the event contract and simulate a hardware decoder's picture slots.
fn check_contract(name: &str, events: &[Event]) {
    let mut decoded: HashSet<u32> = HashSet::new();
    let mut output: HashSet<u32> = HashSet::new();
    // slot index -> picture id it holds
    let mut slots: Vec<Option<u32>> = Vec::new();
    let mut slot_of: HashMap<u32, usize> = HashMap::new();
    let mut decodes = 0;
    for e in events {
        match e {
            Event::Decode(p) => {
                decodes += 1;
                assert!(!decoded.contains(&p.id), "{name}: picture {} decoded twice", p.id);
                assert!(!p.slices.is_empty(), "{name}: picture {} has slices", p.id);
                for s in &p.slices {
                    let t = s.first().map(|h| h & 0x1f);
                    assert!(matches!(t, Some(1 | 5)), "{name}: slice NAL type {t:?}");
                    assert_eq!(t == Some(5), p.idr, "{name}: IDR flag matches the NAL type");
                }
                let max_dpb = p.sps.max_dpb_frames();
                assert!(p.dpb.len() <= max_dpb, "{name}: DPB holds {} > {max_dpb}", p.dpb.len());
                if slots.is_empty() {
                    slots = vec![None; max_dpb + 1];
                }
                assert_eq!(slots.len(), max_dpb + 1, "{name}: one SPS");
                let held: HashSet<u32> = p.dpb.iter().copied().collect();
                assert_eq!(held.len(), p.dpb.len(), "{name}: DPB ids are unique");
                for id in &p.dpb {
                    assert!(decoded.contains(id), "{name}: DPB names undecoded picture {id}");
                }
                for r in &p.refs {
                    assert!(held.contains(&r.id), "{name}: reference {} is in the DPB", r.id);
                    assert!(!r.non_existing, "{name}: no frame_num gaps in the fixtures");
                    let s = slot_of.get(&r.id).copied().unwrap_or_else(|| panic!("{name}: reference {} has a slot", r.id));
                    assert_eq!(slots[s], Some(r.id), "{name}: reference {} still in its slot", r.id);
                }
                if p.idr {
                    assert!(p.refs.is_empty(), "{name}: an IDR picture references nothing");
                }
                assert_eq!(p.stored.is_some(), p.reference, "{name}: x264 keeps every reference picture");
                if let Some(st) = p.stored {
                    assert_eq!(st.id, p.id);
                    assert_eq!(st.poc, p.poc, "{name}: no MMCO 5");
                    assert!(!st.long_term, "{name}: no long-term references");
                    assert_eq!(st.frame_num, p.frame_num);
                }
                // free every slot whose picture left the DPB, then take one
                for s in slots.iter_mut() {
                    if s.is_some_and(|id| !held.contains(&id)) {
                        *s = None;
                    }
                }
                let free = slots.iter().position(Option::is_none).unwrap_or_else(|| panic!("{name}: out of slots at picture {}", p.id));
                slots[free] = Some(p.id);
                slot_of.insert(p.id, free);
                decoded.insert(p.id);
            }
            Event::Output(o) => {
                assert!(decoded.contains(&o.id), "{name}: picture {} output before it was decoded", o.id);
                assert!(output.insert(o.id), "{name}: picture {} output twice", o.id);
                let s = slot_of[&o.id];
                assert_eq!(slots[s], Some(o.id), "{name}: picture {} was overwritten before its output", o.id);
            }
        }
    }
    assert!(decodes > 0, "{name}: pictures decoded");
    assert_eq!(output.len(), decoded.len(), "{name}: every picture is output");
}

#[test]
fn frontend_matches_the_software_decoder() {
    let mut checked = 0;
    for f in common::FIXTURES {
        if f.args.contains(&"-c:v") {
            continue; // platform encoders (macOS only)
        }
        let Some((h264, _)) = common::ensure(f) else { continue };
        let data = std::fs::read(&h264).unwrap();
        let sw = common::decode_file_threads(&h264, 1).unwrap_or_else(|(au, e, _)| panic!("{}: software decoder failed at {au}: {e}", f.name));
        let events = frontend_events(&data, f.name);
        check_contract(f.name, &events);
        let outs: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::Output(o) => Some(o),
                Event::Decode(_) => None,
            })
            .collect();
        assert_eq!(outs.len(), sw.len(), "{}: picture count", f.name);
        for (i, (o, p)) in outs.iter().zip(&sw).enumerate() {
            let got = (o.pts, o.poc, o.key, o.crop.2, o.crop.3, o.color, o.sar);
            let want = (p.pts, p.poc, p.key, p.width, p.height, p.color, p.sar);
            assert_eq!(got, want, "{}: output {i}", f.name);
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: no fixtures (ffmpeg missing)");
    }
}

/// Per call, the front-end outputs exactly the pictures a single-threaded software decoder
/// returns from the same call: hardware decoders built on it can be checked against that decoder
/// in lockstep (first-use verification in `filmcraft-platform`).
#[test]
fn outputs_per_call_match_a_single_threaded_decoder() {
    for name in ["bpyramid", "ref4", "keyint10", "open_gop", "baseline_qcif"] {
        let f = common::fixture(name);
        let Some((h264, _)) = common::ensure(f) else { continue };
        let data = std::fs::read(&h264).unwrap();
        let mut fe = Frontend::new();
        let mut sw = filmcraft_h264::Decoder::with_threads(1);
        let outputs = |events: Vec<Event>| -> Vec<i64> {
            events
                .into_iter()
                .filter_map(|e| match e {
                    Event::Output(o) => Some(o.pts),
                    Event::Decode(_) => None,
                })
                .collect()
        };
        for (i, au) in common::split_access_units(&data).into_iter().enumerate() {
            let a = outputs(fe.decode(au, i as i64).unwrap());
            let b: Vec<i64> = sw.decode(au, i as i64).unwrap().iter().map(|p| p.pts).collect();
            assert_eq!(a, b, "{name}: access unit {i}");
        }
        let b: Vec<i64> = sw.flush().iter().map(|p| p.pts).collect();
        assert_eq!(outputs(fe.flush()), b, "{name}: flush");
    }
}

/// Slice NAL units are passed through byte for byte: the picture's slices are exactly the
/// access unit's slice NAL units.
#[test]
fn slices_are_the_access_units_slice_nal_units() {
    let f = common::fixture("slices4");
    let Some((h264, _)) = common::ensure(f) else { return };
    let data = std::fs::read(&h264).unwrap();
    let mut fe = Frontend::new();
    let mut decodes = 0;
    for (i, au) in common::split_access_units(&data).into_iter().enumerate() {
        let want: Vec<&[u8]> = filmcraft_bitstream::annexb_nals(au).into_iter().filter(|n| n.first().is_some_and(|h| matches!(h & 0x1f, 1 | 5))).collect();
        for e in fe.decode(au, i as i64).unwrap() {
            if let Event::Decode(p) = e {
                let got: Vec<&[u8]> = p.slices.iter().map(Vec::as_slice).collect();
                assert_eq!(got, want, "access unit {i}");
                assert_eq!(p.slices.len(), 4, "four slices per picture");
                decodes += 1;
            }
        }
    }
    assert!(decodes > 0);
}

/// Damaged input gives errors, never a panic, and the front-end never reports a picture twice.
#[test]
fn hostile_input_is_an_error_not_a_crash() {
    let f = common::fixture("main_cabac_b");
    let Some((h264, _)) = common::ensure(f) else { return };
    let data = std::fs::read(&h264).unwrap();
    let aus = common::split_access_units(&data);
    let mut seed = 0x2545_f491_u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for round in 0..64 {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut fe = Frontend::new();
            let mut ids = HashSet::new();
            for (i, au) in aus.iter().enumerate() {
                let mut au = au.to_vec();
                if !au.is_empty() && (i + round) % 3 == 0 {
                    for _ in 0..4 {
                        let k = next() as usize % au.len();
                        au[k] ^= 1 << (next() % 8);
                    }
                    if round % 4 == 0 {
                        au.truncate(next() as usize % au.len());
                    }
                }
                if let Ok(events) = fe.decode(&au, i as i64) {
                    for e in events {
                        if let Event::Decode(p) = e {
                            assert!(ids.insert(p.id), "picture {} reported twice", p.id);
                        }
                    }
                }
            }
            fe.flush();
        }));
        assert!(r.is_ok(), "round {round} panicked");
    }
}
