//! `perf`-measurement driver for `PathFst`/`PathFstDelta` vs `Heap`,
//! isolating one `contains()` loop per process invocation so
//! `perf stat`/`perf record --call-graph dwarf` attributes cleanly to a
//! single structure's code, not a mix of three. See `DESIGN.md`'s "path
//! FST prototype" entry.
//!
//! Usage: `cargo run --profile profiling --example path_fst_perf --features heap -- <which> [iters]`
//! where `<which>` is one of `heap`, `fst`, `delta`, `pow2`.
//!
//! Example:
//!   perf stat -e cycles,instructions,cache-references,cache-misses \
//!     target/profiling/examples/path_fst_perf fst 5000000
//!   perf record -F 999 --call-graph dwarf -o /tmp/fst.data -- \
//!     target/profiling/examples/path_fst_perf fst 5000000
//!   perf report -i /tmp/fst.data

use s_graph::path_fst::{PathFst, PathFstDelta, PathFstDeltaPow2};
use s_graph::{Heap, Triple};

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::uniform;
use common::rdf_gen::SplitMix64;

const NUM_S: u32 = 500_000;
const NUM_P: u32 = 200;
const NUM_O: u32 = 500_000;
const NUM_TRIPLES: usize = 10_000_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("fst");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);

    let triples = uniform(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 42);
    let mut rng = SplitMix64::new(7);
    let mut probes: Vec<Triple> = Vec::with_capacity(iters);
    for _ in 0..iters {
        probes.push(if rng.next_f64() < 0.5 {
            triples[rng.next_range(triples.len() as u64) as usize]
        } else {
            (rng.next_range(NUM_S as u64) as u32, rng.next_range(NUM_P as u64) as u32, rng.next_range(NUM_O as u64) as u32)
        });
    }

    let mut sink = 0u64;
    match which {
        "heap" => {
            let heap_tree = Heap::build_dynamic(triples, s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built Heap: {} bytes, {} blocks", heap_tree.mem_size_bytes(), heap_tree.block_count());
            let t0 = std::time::Instant::now();
            for &(s, p, o) in &probes {
                sink += heap_tree.contains(s, p, o) as u64;
            }
            eprintln!("Heap: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters as f64);
        }
        "fst" => {
            let path_fst = PathFst::build(triples, [NUM_S as u64, NUM_P as u64, NUM_O as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFst: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let t0 = std::time::Instant::now();
            for &(s, p, o) in &probes {
                sink += path_fst.contains(s, p, o) as u64;
            }
            eprintln!("PathFst: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters as f64);
        }
        "delta" => {
            let path_fst_delta = PathFstDelta::build(triples, [NUM_S as u64, NUM_P as u64, NUM_O as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDelta: {} bytes, {} blocks", path_fst_delta.mem_size_bytes(), path_fst_delta.block_count());
            let t0 = std::time::Instant::now();
            for &(s, p, o) in &probes {
                sink += path_fst_delta.contains(s, p, o) as u64;
            }
            eprintln!("PathFstDelta: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters as f64);
        }
        "pow2" => {
            let path_fst_pow2 = PathFstDeltaPow2::build(triples, [NUM_S as u64, NUM_P as u64, NUM_O as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst_pow2.mem_size_bytes(), path_fst_pow2.block_count());
            let t0 = std::time::Instant::now();
            for &(s, p, o) in &probes {
                sink += path_fst_pow2.contains(s, p, o) as u64;
            }
            eprintln!("PathFstDeltaPow2: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters as f64);
        }
        other => panic!("unknown structure {other:?}, expected heap/fst/delta/pow2"),
    }
    std::hint::black_box(sink);
}
