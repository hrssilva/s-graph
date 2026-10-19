//! `perf`-measurement driver for `PathFstDeltaPow2::restricted_descent_visit`
//! vs `Heap::restricted_descent_visit` on the "one-Exact"/"two-Exact" BGP
//! shapes -- see `DESIGN.md`'s "BGP-style queries on the FST" entry and its
//! follow-ups. `path_fst_bgp_proto.rs` already measured wall-clock near
//! parity (1.06x-1.07x) for `one-Exact` on the two largest domain shapes
//! (10M-triple symmetric, 3M-triple asymmetric) after wiring in the
//! precomputed-mask fix; this isolates one structure's loop per process
//! invocation, same baseline-subtraction methodology as `path_fst_perf.rs`,
//! so `perf stat`/`perf record` attributes cleanly to one structure's code,
//! to find out *where* the remaining margin is instead of guessing.
//!
//! Usage: `cargo run --profile profiling --example path_fst_bgp_perf --features heap -- <which> <shape> <domain> [iters]`
//! where `<which>` is `heap`|`pow2`, `<shape>` is `one`|`two`, `<domain>` is
//! `small`|`sym`|`zipf`|`asym` (the same four domain shapes
//! `path_fst_bgp_proto.rs` benchmarks).
//!
//! Example:
//!   perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
//!     target/profiling/examples/path_fst_bgp_perf pow2 one sym 50000
//!   perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
//!     target/profiling/examples/path_fst_bgp_perf pow2 one sym 0
//!   perf record -F 999 --call-graph dwarf -o /tmp/bgp.data -- \
//!     target/profiling/examples/path_fst_bgp_perf pow2 one sym 50000
//!   perf report -i /tmp/bgp.data

use s_graph::path_fst::PathFstDeltaPow2;
use s_graph::tree::DimRestriction;
use s_graph::{Heap, Triple};

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::{uniform, zipfian_entities};
use common::rdf_gen::SplitMix64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("pow2");
    let shape = args.get(2).map(String::as_str).unwrap_or("one");
    let domain = args.get(3).map(String::as_str).unwrap_or("sym");
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let (num_s, num_p, num_o, num_triples): (u32, u32, u32, usize) = match domain {
        "small" => (5_000, 200, 5_000, 100_000),
        "sym" => (500_000, 200, 500_000, 10_000_000),
        "zipf" => (500_000, 200, 500_000, 10_000_000),
        "asym" => (20_000, 200, 2_000_000, 3_000_000),
        other => panic!("unknown domain {other:?}, expected small/sym/zipf/asym"),
    };

    let triples: Vec<Triple> = if domain == "zipf" {
        zipfian_entities(num_s, num_p, num_o, num_triples, 1.5, 1.5, 1.5, 42)
    } else {
        uniform(num_s, num_p, num_o, num_triples, 42)
    };
    let mut rng = SplitMix64::new(777);

    let mut sink = 0u64;
    match (which, shape) {
        ("heap", "one") => {
            let heap_tree = Heap::build_dynamic(triples, s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built Heap: {} bytes, {} blocks", heap_tree.mem_size_bytes(), heap_tree.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = heap_tree.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("Heap one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("heap", "two") => {
            let heap_tree = Heap::build_dynamic(triples, s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built Heap: {} bytes, {} blocks", heap_tree.mem_size_bytes(), heap_tree.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = heap_tree.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("Heap two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("pow2", "one") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2 one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("pow2", "two") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2 two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        (w, s) => panic!("unknown combination {w:?}/{s:?}, expected heap|pow2 and one|two"),
    }
    std::hint::black_box(sink);
}
