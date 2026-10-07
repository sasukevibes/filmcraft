//! The hardware front-end (`accel::Frontend`) against the software decoder on the libx265
//! fixtures: per call it outputs exactly what a single-threaded decoder returns (pts, POC, key
//! flag, crop, colour, aspect, bit depth); every picture is decoded once before it is output and
//! output at most once; RPS lists and the DPB list only name decoded pictures still held; a
//! hardware decoder with `max_dec_pic_buffering + 1` picture slots, freeing a slot when its picture
//! leaves the DPB list, never runs out of slots and never overwrites a picture still referenced or
//! not yet output. Skipped without ffmpeg (fixture generator).

mod common;

use std::collections::{HashMap, HashSet};

use filmcraft_hevc::accel::{Event, Frontend};

fn check_contract(name: &str, events: &[Event]) {
    let mut decoded: HashSet<u32> = HashSet::new();
    let mut output: HashSet<u32> = HashSet::new();
    let mut slots: Vec<Option<u32>> = Vec::new();
    let mut slot_of: HashMap<u32, usize> = HashMap::new();
    for e in events {
        match e {
            Event::Decode(p) => {
                assert!(!decoded.contains(&p.id), "{name}: picture {} decoded twice", p.id);
                assert!(!p.slices.is_empty(), "{name}: picture {} has slice segments", p.id);
                for s in &p.slices {
                    let t = s.first().map(|h| (h >> 1) & 0x3f);
                    assert!(t.is_some_and(|t| t < 32), "{name}: slice NAL type {t:?}");
                }
                let max_dpb = p.sps.max_dec_pic_buffering as usize;
                assert!(p.dpb.len() <= max_dpb, "{name}: DPB holds {} > {max_dpb}", p.dpb.len());
                if slots.is_empty() {
                    slots = vec![None; max_dpb + 1];
                }
                let held: HashSet<u32> = p.dpb.iter().copied().collect();
                for r in &p.refs {
                    assert!(!r.non_existing, "{name}: no missing references in the fixtures");
                    assert!(held.contains(&r.id), "{name}: reference {} is in the DPB", r.id);
                    let s = slot_of[&r.id];
                    assert_eq!(slots[s], Some(r.id), "{name}: reference {} still in its slot", r.id);
                }
                let ref_ids: HashSet<u32> = p.refs.iter().map(|r| r.id).collect();
                for id in p.st_curr_before.iter().chain(&p.st_curr_after).chain(&p.lt_curr) {
                    assert!(ref_ids.contains(id), "{name}: RPS picture {id} is a reference");
                }
                if p.idr {
                    assert!(p.refs.is_empty() && p.st_curr_before.is_empty(), "{name}: an IDR picture references nothing");
                    assert!(p.irap);
                }
                if p.st_rps_sps {
                    assert_eq!(p.st_rps_bits, 0, "{name}: no RPS bits in the header when it comes from the SPS");
                }
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
                assert_eq!(slots[slot_of[&o.id]], Some(o.id), "{name}: picture {} overwritten before its output", o.id);
            }
        }
    }
    assert!(!decoded.is_empty(), "{name}: pictures decoded");
}

#[test]
fn frontend_matches_a_single_threaded_decoder_per_call() {
    let mut checked = 0;
    for f in common::FIXTURES {
        if f.args.contains(&"-c:v") {
            continue; // platform encoders (macOS only)
        }
        let Some((hevc, _)) = common::ensure(f) else { continue };
        let data = std::fs::read(&hevc).unwrap();
        let mut fe = Frontend::new();
        let mut sw = filmcraft_hevc::Decoder::with_threads(1);
        let mut events = Vec::new();
        let outputs = |events: &[Event]| -> Vec<filmcraft_hevc::accel::OutputPicture> {
            events
                .iter()
                .filter_map(|e| match e {
                    Event::Output(o) => Some(o.clone()),
                    Event::Decode(_) => None,
                })
                .collect()
        };
        let same = |name: &str, at: &str, a: &[filmcraft_hevc::accel::OutputPicture], b: &[filmcraft_hevc::Picture]| {
            let a: Vec<_> = a.iter().map(|o| (o.pts, o.poc, o.key, o.crop.2, o.crop.3, o.color, o.sar, o.bit_depth)).collect();
            let b: Vec<_> = b.iter().map(|p| (p.pts, p.poc, p.key, p.width, p.height, p.color, p.sar, p.bit_depth)).collect();
            assert_eq!(a, b, "{name}: {at}");
        };
        for (i, au) in common::split_access_units(&data).into_iter().enumerate() {
            let ev = fe.decode(au, i as i64).unwrap_or_else(|e| panic!("{}: access unit {i}: {e}", f.name));
            let pics = sw.decode(au, i as i64).unwrap_or_else(|e| panic!("{}: software, access unit {i}: {e}", f.name));
            same(f.name, &format!("access unit {i}"), &outputs(&ev), &pics);
            events.extend(ev);
        }
        let ev = fe.flush();
        same(f.name, "flush", &outputs(&ev), &sw.flush());
        events.extend(ev);
        check_contract(f.name, &events);
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: no fixtures (ffmpeg missing)");
    }
}

/// Damaged input gives errors, never a panic, and no picture is reported twice.
#[test]
fn hostile_input_is_an_error_not_a_crash() {
    let f = common::fixture("bframes");
    let Some((hevc, _)) = common::ensure(f) else { return };
    let data = std::fs::read(&hevc).unwrap();
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
