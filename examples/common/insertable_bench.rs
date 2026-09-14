//! Shared scaffolding for the block-list insertable-storage benchmarks:
//! `examples/bench_insertable.rs` (every variant side by side, one
//! process -- fast to run, but `examples/common/metrics.rs`'s
//! `mem_size_bytes()`/allocated-capacity numbers are what's comparable
//! there, not peak RSS: a shared-process peak-RSS reading would be
//! contaminated by whichever earlier structure used the most memory,
//! since the kernel's `VmHWM` high-water mark never decreases even after
//! that structure is dropped) and `examples/rss_*.rs` (one variant per
//! *process*, so each gets its own honest, uncontaminated peak-RSS
//! reading -- see their own doc comments). Both sets of binaries need
//! the exact same dataset generation, probe sampling, and timing
//! helpers; this module is the one place those live, so the two can't
//! silently drift apart.

use crate::common::metrics::{current_rss_kb, human_bytes, peak_rss_kb};
use crate::common::rdf_gen::{zipfian_entities, zipfian_predicates, SplitMix64, Triple};
use std::path::{Path, PathBuf};

/// Block capacity used throughout these benchmarks -- the sweet spot
/// found by the capacity sweep documented in the unrolled variants'
/// own history (reads within ~1.1-1.6x of the flat/rebuild-only
/// baseline, inserts still ~10^5-10^6x faster than that baseline's only
/// option, a full rebuild).
pub const BLOCK_CAPACITY: usize = 65536;
pub const NUM_INSERTS: usize = 1000;

const LARGE_S: u32 = 2_000_000;
const LARGE_P: u32 = 19;
const LARGE_O: u32 = 2_000_000;
const LARGE_TRIPLES: usize = 20_000_000;

pub const LARGE_SCALE_SCENARIO_NAMES: [&str; 4] = ["large_scale_pred_skew_only", "large_scale_entity_skew_1.0", "large_scale_entity_skew_1.5", "large_scale_entity_skew_2.0"];

/// Same shape as `large_scale_pred_skew_only` (same S/O domain, same
/// 20M-triple count, same skew exponent) but with far more predicates --
/// specifically targeting `Mk2Forest`'s documented weakness (an unbound-
/// predicate query like `predicates_for` must scan every one of its
/// `num_predicates` independent trees) and `Ik2Tree`'s headline
/// counter-design (interleaving is supposed to make that same query
/// O(depth), not O(predicates)). `RadixTree`'s own `predicates_for` is a
/// genuine tree descent regardless of predicate count, so it's the
/// control: its cost shouldn't move as these scenarios' `num_predicates`
/// grows, unlike `Mk2Forest`'s (and, if the interleaving story holds,
/// unlike `Ik2Tree`'s either).
pub const LARGE_SCALE_MANY_PRED_SCENARIO_NAMES: [&str; 2] = ["large_scale_many_predicates_200", "large_scale_many_predicates_2000"];

/// Builds one of this project's standard large-scale (20M-triple)
/// scenarios by name -- either the original four
/// ([`LARGE_SCALE_SCENARIO_NAMES`]) or the many-predicate pair
/// ([`LARGE_SCALE_MANY_PRED_SCENARIO_NAMES`]).
pub fn large_scale_scenario(name: &str) -> Vec<Triple> {
    match name {
        "large_scale_pred_skew_only" => zipfian_predicates(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.2, 2026),
        "large_scale_entity_skew_1.0" => zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.0, 1.2, 1.0, 2026),
        "large_scale_entity_skew_1.5" => zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.5, 1.2, 1.5, 2026),
        "large_scale_entity_skew_2.0" => zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 2.0, 1.2, 2.0, 2026),
        "large_scale_many_predicates_200" => zipfian_predicates(LARGE_S, 200, LARGE_O, LARGE_TRIPLES, 1.2, 2026),
        "large_scale_many_predicates_2000" => zipfian_predicates(LARGE_S, 2000, LARGE_O, LARGE_TRIPLES, 1.2, 2026),
        _ => panic!("unknown scenario {name:?}; expected one of {LARGE_SCALE_SCENARIO_NAMES:?} or {LARGE_SCALE_MANY_PRED_SCENARIO_NAMES:?}"),
    }
}

pub fn new_triple(rng: &mut SplitMix64, max_s: u32, max_p: u32, max_o: u32) -> Triple {
    (rng.next_range(max_s as u64 + 1) as u32, rng.next_range(max_p as u64 + 1) as u32, rng.next_range(max_o as u64 + 1) as u32)
}

pub fn domain_bounds(triples: &[Triple]) -> (u32, u32, u32) {
    triples.iter().fold((0u32, 0u32, 0u32), |(ms, mp, mo), &(s, p, o)| (ms.max(s), mp.max(p), mo.max(o)))
}

/// Same sampling density every large-scale scenario in this project
/// uses: ~10,000 `contains` probes, ~2,000 row-query probes.
pub fn contains_and_row_probes(triples: &[Triple]) -> (Vec<Triple>, Vec<(u32, u32)>) {
    let contains_probes: Vec<Triple> = triples.iter().step_by((triples.len() / 10_000).max(1)).copied().collect();
    let row_probes: Vec<(u32, u32)> = triples.iter().step_by((triples.len() / 2_000).max(1)).map(|&(s, p, _)| (s, p)).collect();
    (contains_probes, row_probes)
}

pub fn measure_contains_and_row(contains: impl Fn(u32, u32, u32) -> bool, row: impl Fn(u32, u32) -> usize, contains_probes: &[Triple], row_probes: &[(u32, u32)]) -> (std::time::Duration, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let mut hits = 0u64;
    for &(s, p, o) in contains_probes {
        if contains(s, p, o) {
            hits += 1;
        }
    }
    let contains_avg = t0.elapsed() / contains_probes.len() as u32;
    std::hint::black_box(hits);

    let t0 = std::time::Instant::now();
    let mut total = 0u64;
    for &(s, p) in row_probes {
        total += row(s, p) as u64;
    }
    let row_avg = t0.elapsed() / row_probes.len() as u32;
    std::hint::black_box(total);

    (contains_avg, row_avg)
}

/// Times an unbound-predicate `predicates_for(s, o)`-shaped query --
/// the one this project's `large_scale_many_predicates_*` scenarios
/// (see [`LARGE_SCALE_MANY_PRED_SCENARIO_NAMES`]) exist to stress, since
/// its cost is exactly where `Mk2Forest`/`Ik2Tree`/`RadixTree` are
/// expected to diverge as predicate count grows. `probes` is `(s, o)`
/// pairs -- callers derive these from their already-loaded
/// [`ProbeData::contains_probes`] (dropping the bound predicate) rather
/// than this module sampling a separate probe set, so no `ProbeData`
/// format change was needed to add this measurement.
pub fn measure_predicates_for(f: impl Fn(u32, u32) -> usize, probes: &[(u32, u32)]) -> std::time::Duration {
    let t0 = std::time::Instant::now();
    let mut total = 0u64;
    for &(s, o) in probes {
        total += f(s, o) as u64;
    }
    let avg = t0.elapsed() / probes.len() as u32;
    std::hint::black_box(total);
    avg
}

/// Everything a query or insert phase needs from the original triple
/// dataset, *without* needing the dataset itself -- see
/// `examples/phase_*.rs`'s own doc comment for why. Saved once by the
/// `build` phase (which does hold the full dataset, legitimately, since
/// building genuinely requires it) via [`save_probe_data`], then loaded
/// by the `query`/`insert` phases via [`load_probe_data`] instead of
/// regenerating the ~20M-triple dataset just to re-derive these -- doing
/// that would reintroduce exactly the build-time RSS pollution phase
/// separation exists to avoid.
pub struct ProbeData {
    pub max_s: u32,
    pub max_p: u32,
    pub max_o: u32,
    pub contains_probes: Vec<Triple>,
    pub row_probes: Vec<(u32, u32)>,
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    v
}

pub fn save_probe_data(path: &Path, data: &ProbeData) {
    let mut out = Vec::new();
    write_u32(&mut out, data.max_s);
    write_u32(&mut out, data.max_p);
    write_u32(&mut out, data.max_o);
    write_u32(&mut out, data.contains_probes.len() as u32);
    for &(s, p, o) in &data.contains_probes {
        write_u32(&mut out, s);
        write_u32(&mut out, p);
        write_u32(&mut out, o);
    }
    write_u32(&mut out, data.row_probes.len() as u32);
    for &(s, p) in &data.row_probes {
        write_u32(&mut out, s);
        write_u32(&mut out, p);
    }
    std::fs::write(path, &out).unwrap_or_else(|e| panic!("failed to write probe data to {path:?}: {e}"));
}

pub fn load_probe_data(path: &Path) -> ProbeData {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read probe data from {path:?} (run the `build` phase first): {e}"));
    let mut pos = 0usize;
    let max_s = read_u32(&buf, &mut pos);
    let max_p = read_u32(&buf, &mut pos);
    let max_o = read_u32(&buf, &mut pos);
    let n = read_u32(&buf, &mut pos) as usize;
    let mut contains_probes = Vec::with_capacity(n);
    for _ in 0..n {
        let s = read_u32(&buf, &mut pos);
        let p = read_u32(&buf, &mut pos);
        let o = read_u32(&buf, &mut pos);
        contains_probes.push((s, p, o));
    }
    let m = read_u32(&buf, &mut pos) as usize;
    let mut row_probes = Vec::with_capacity(m);
    for _ in 0..m {
        let s = read_u32(&buf, &mut pos);
        let p = read_u32(&buf, &mut pos);
        row_probes.push((s, p));
    }
    ProbeData { max_s, max_p, max_o, contains_probes, row_probes }
}

/// `(tree_path, probes_path)` for one `structure`/`scenario` pair --
/// where `examples/phase_*.rs`'s `build` phase writes and its
/// `query`/`insert` phases read. `std::env::temp_dir()` rather than this
/// session's scratchpad, since these paths need to work for a `build`
/// then a *later, independent* `cargo run` of the `query`/`insert`
/// phase, not just within one script.
pub fn phase_paths(structure: &str, scenario: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir();
    (dir.join(format!("sgraph_phase_{structure}_{scenario}.tree.bin")), dir.join(format!("sgraph_phase_{structure}_{scenario}.probes.bin")))
}

/// Prints this process's steady-state and peak RSS -- call once, at the
/// very end of a `phase_*.rs` binary's `main`, so the reading reflects
/// everything that phase actually did (deserialize, run its workload)
/// and nothing from a different phase, since each phase is its own
/// process.
pub fn report_rss(label: &str) {
    match current_rss_kb() {
        Some(kb) => println!("  {label} steady-state RSS (VmRSS) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  {label} steady-state RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
    match peak_rss_kb() {
        Some(kb) => println!("  {label} peak RSS (VmHWM) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  {label} peak RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
}
