//! Driver for `s_graph::path_fst::PathFst`, the experimental minimized-
//! FST prototype -- see `DESIGN.md`'s "path FST prototype" entry and
//! `src/path_fst.rs`'s own doc comment. Builds both `PathFst` and the
//! production `Heap` tree over the *same* triples, then checks:
//! 1. correctness -- `contains` must agree with both `Heap` and a naive
//!    `HashSet` ground truth, on both real triples and random probes
//!    (including out-of-domain ones);
//! 2. memory -- `PathFst::mem_size_bytes()` vs `Heap::mem_size_bytes()`;
//! 3. state count -- `PathFst::node_count()` (post-dedup) vs
//!    `PathFst::visited_count()` (what a non-deduplicated build would
//!    have produced, i.e. directly comparable to `Heap::node_word_count`);
//! 4. a rough wall-clock `contains` timing comparison (not `perf stat`
//!    rigor -- a first signal, not a final number).
//!
//! Kept in the repository (not deleted after measuring), same convention
//! as the subtree-dedup prototypes. Run with:
//!   cargo run --release --example path_fst_proto --features heap

use std::collections::HashSet;
use std::time::Instant;

use s_graph::path_fst::{PathFst, PathFstDelta, PathFstDeltaPow2};
use s_graph::{Heap, Triple};

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::{uniform, zipfian_entities, SplitMix64};

fn human_bytes(n: usize) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.2} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2} MB", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.2} KB", n / 1e3)
    } else {
        format!("{n:.0} B")
    }
}

fn report(name: &str, triples: Vec<Triple>, num_s: u32, num_p: u32, num_o: u32) {
    println!("=== {name} (domain={num_s}x{num_p}x{num_o}, {} triples) ===", triples.len());

    let ground_truth: HashSet<Triple> = triples.iter().copied().collect();
    let heap_tree = Heap::build_dynamic(triples.clone(), s_graph::DEFAULT_BLOCK_CAPACITY);
    let path_fst = PathFst::build(triples.clone(), [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
    let path_fst_delta = PathFstDelta::build(triples.clone(), [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
    let path_fst_pow2 = PathFstDeltaPow2::build(triples.clone(), [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);

    // --- correctness ---
    let mut rng = SplitMix64::new(99);
    let mut checked = 0u64;
    let mut mismatches = 0u64;
    // real triples (sample if huge)
    let sample_n = triples.len().min(20_000);
    for i in 0..sample_n {
        let idx = if triples.len() <= sample_n { i } else { rng.next_range(triples.len() as u64) as usize };
        let (s, p, o) = triples[idx];
        let expected = true; // it's a real triple
        let heap_ok = heap_tree.contains(s, p, o);
        let fst_ok = path_fst.contains(s, p, o);
        let delta_ok = path_fst_delta.contains(s, p, o);
        let pow2_ok = path_fst_pow2.contains(s, p, o);
        let gt_ok = ground_truth.contains(&(s, p, o));
        checked += 1;
        if heap_ok != expected || fst_ok != expected || delta_ok != expected || pow2_ok != expected || gt_ok != expected {
            mismatches += 1;
            eprintln!("  MISMATCH (positive) ({s},{p},{o}): heap={heap_ok} fst={fst_ok} delta={delta_ok} pow2={pow2_ok} gt={gt_ok}");
        }
    }
    // random probes, in and out of domain
    for _ in 0..20_000 {
        let (s, p, o) = if rng.next_f64() < 0.1 {
            // deliberately out-of-domain sometimes
            (rng.next_range(num_s as u64 * 2) as u32, rng.next_range(num_p as u64 * 2) as u32, rng.next_range(num_o as u64 * 2) as u32)
        } else {
            (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32, rng.next_range(num_o as u64) as u32)
        };
        let heap_ok = heap_tree.contains(s, p, o);
        let fst_ok = path_fst.contains(s, p, o);
        let delta_ok = path_fst_delta.contains(s, p, o);
        let pow2_ok = path_fst_pow2.contains(s, p, o);
        let gt_ok = ground_truth.contains(&(s, p, o));
        checked += 1;
        if heap_ok != gt_ok || fst_ok != gt_ok || delta_ok != gt_ok || pow2_ok != gt_ok {
            mismatches += 1;
            eprintln!("  MISMATCH (random) ({s},{p},{o}): heap={heap_ok} fst={fst_ok} delta={delta_ok} pow2={pow2_ok} gt={gt_ok}");
        }
    }
    println!("  correctness: {checked} probes checked, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "PathFst/PathFstDelta/PathFstDeltaPow2/Heap/ground-truth disagreed on {mismatches} probes");

    // --- memory ---
    let heap_mem = heap_tree.mem_size_bytes();
    let fst_mem = path_fst.mem_size_bytes();
    let delta_mem = path_fst_delta.mem_size_bytes();
    let pow2_mem = path_fst_pow2.mem_size_bytes();
    println!(
        "  memory: Heap={} ({} nodes) vs PathFst(Pos)={} vs PathFstDelta={} vs PathFstDeltaPow2={} ({} distinct states, {} pre-dedup visits, {:.3} dedup ratio)",
        human_bytes(heap_mem),
        heap_tree.node_word_count(),
        human_bytes(fst_mem),
        human_bytes(delta_mem),
        human_bytes(pow2_mem),
        path_fst.node_count(),
        path_fst.visited_count(),
        path_fst.node_count() as f64 / path_fst.visited_count().max(1) as f64,
    );
    println!(
        "    saved vs Heap: PathFst(Pos)={:.2}%, PathFstDelta={:.2}%, PathFstDeltaPow2={:.2}%",
        100.0 * (1.0 - fst_mem as f64 / heap_mem.max(1) as f64),
        100.0 * (1.0 - delta_mem as f64 / heap_mem.max(1) as f64),
        100.0 * (1.0 - pow2_mem as f64 / heap_mem.max(1) as f64),
    );
    println!(
        "  storage: Heap={} blocks, PathFst(Pos)={} blocks, PathFstDelta={} blocks, PathFstDeltaPow2={} blocks (all block_capacity={})",
        heap_tree.block_count(),
        path_fst.block_count(),
        path_fst_delta.block_count(),
        path_fst_pow2.block_count(),
        path_fst.block_capacity()
    );

    // --- rough contains timing (wall clock, not perf stat) ---
    let probe_n = 200_000;
    let mut heap_probes = Vec::with_capacity(probe_n);
    for _ in 0..probe_n {
        heap_probes.push(if rng.next_f64() < 0.5 {
            triples[rng.next_range(triples.len() as u64) as usize]
        } else {
            (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32, rng.next_range(num_o as u64) as u32)
        });
    }
    let t0 = Instant::now();
    let mut sink = 0u64;
    for &(s, p, o) in &heap_probes {
        sink += heap_tree.contains(s, p, o) as u64;
    }
    let heap_elapsed = t0.elapsed();
    let t1 = Instant::now();
    for &(s, p, o) in &heap_probes {
        sink += path_fst.contains(s, p, o) as u64;
    }
    let fst_elapsed = t1.elapsed();
    let t2 = Instant::now();
    for &(s, p, o) in &heap_probes {
        sink += path_fst_delta.contains(s, p, o) as u64;
    }
    let delta_elapsed = t2.elapsed();
    let t3 = Instant::now();
    for &(s, p, o) in &heap_probes {
        sink += path_fst_pow2.contains(s, p, o) as u64;
    }
    let pow2_elapsed = t3.elapsed();
    std::hint::black_box(sink);
    println!(
        "  contains() wall time over {probe_n} probes: Heap={:.1}ns/call, PathFst(Pos)={:.1}ns/call ({:.2}x), PathFstDelta={:.1}ns/call ({:.2}x), PathFstDeltaPow2={:.1}ns/call ({:.2}x)",
        heap_elapsed.as_nanos() as f64 / probe_n as f64,
        fst_elapsed.as_nanos() as f64 / probe_n as f64,
        fst_elapsed.as_nanos() as f64 / heap_elapsed.as_nanos() as f64,
        delta_elapsed.as_nanos() as f64 / probe_n as f64,
        delta_elapsed.as_nanos() as f64 / heap_elapsed.as_nanos() as f64,
        pow2_elapsed.as_nanos() as f64 / probe_n as f64,
        pow2_elapsed.as_nanos() as f64 / heap_elapsed.as_nanos() as f64,
    );
    println!();
}

fn main() {
    report("uniform (small, symmetric)", uniform(5_000, 200, 5_000, 100_000, 42), 5_000, 200, 5_000);
    report(
        "uniform (symmetric, full scale)",
        uniform(500_000, 200, 500_000, 10_000_000, 42),
        500_000,
        200,
        500_000,
    );
    report(
        "zipfian S+P+O (symmetric, high skew, full scale)",
        zipfian_entities(500_000, 200, 500_000, 10_000_000, 1.5, 1.5, 1.5, 42),
        500_000,
        200,
        500_000,
    );
    report(
        "uniform (asymmetric, full scale)",
        uniform(20_000, 200, 2_000_000, 3_000_000, 42),
        20_000,
        200,
        2_000_000,
    );
}
