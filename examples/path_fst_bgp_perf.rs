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
//! where `<which>` is `heap`|`pow2`|`frontier`|`bfs`|`layout`|`layoutbfs`|`layoutmasked`|`sweep`|`leafsweep`|`seq`|`automatonwalk`|`composedautomaton`|`parallel`|`parallelcutoff`|`parallelsizecutoff`|`parallelscoped`|`parallelchunked`|`parallellazy`|`parallelqueue`|`parallelmatrix`
//! (the `parallel*` variants need `--features heap,parallel`; `parallelcutoff`/`parallelscoped`/`parallelchunked`/`parallellazy`
//! read a 5th arg, `parallel_depth` (default 1); `parallelsizecutoff` reads the same 5th arg as
//! `min_parallel_size` (default 8) -- only fan out when a node has more real children than that;
//! `parallelchunked` additionally reads a 6th arg, `max_branch` (default 8) -- spawn at most this
//! many tasks per node, chunking wider fan-outs instead of one task per child; `parallelmatrix`
//! reads the same 5th arg as `steal_mode`, `"shallow"` (default) or `"deep"` -- which open column
//! the steal scan prefers when a thread runs out of its own work),
//! `<shape>` is `one`|`two`, `<domain>` is
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

use s_graph::path_fst::{PathFstDeltaPow2, PathFstDeltaPow2Bfs};
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
    let parallel_depth: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    let min_parallel_size: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_branch: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
    let prefer_shallow: bool = args.get(5).map(String::as_str).unwrap_or("shallow") != "deep";

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
        ("frontier", "one") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_frontier_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2(frontier) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("frontier", "two") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_frontier_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2(frontier) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("bfs", "one") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2(bfs) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("bfs", "two") => {
            let path_fst = PathFstDeltaPow2::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2(bfs) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layout", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(dfs) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layout", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(dfs) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layoutbfs", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(bfs) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layoutbfs", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(bfs) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layoutmasked", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_bfs_masked_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(masked) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("layoutmasked", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_bfs_masked_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(masked) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("sweep", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            // Correctness first, on a handful of probes -- the whole-level
            // sweep is expensive enough (see DESIGN.md) that running it
            // against the full proto driver's 500-probe loop isn't worth it.
            for _ in 0..20 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_sweep([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                assert_eq!(expected, got, "sweep traversal disagreed with the default on s={s}");
            }
            eprintln!("sweep one-Exact: 20/20 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let _ = path_fst.restricted_descent_sweep_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(sweep) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("sweep", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..20 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_sweep([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                assert_eq!(expected, got, "sweep traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("sweep two-Exact: 20/20 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let _ = path_fst.restricted_descent_sweep_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
                    sink += v as u64;
                    std::ops::ControlFlow::Continue::<()>(())
                });
            }
            eprintln!("PathFstDeltaPow2Bfs(sweep) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("leafsweep", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!(
                "built PathFstDeltaPow2Bfs: {} bytes, {} blocks, {} leaf nodes",
                path_fst.mem_size_bytes(),
                path_fst.block_count(),
                path_fst.leaf_level_node_count()
            );
            let t0 = std::time::Instant::now();
            let parent_index = path_fst.build_parent_index();
            eprintln!("build_parent_index: {:?} ({} entries)", t0.elapsed(), parent_index.len());
            // Measured at ~3.2-3.6 SECONDS per call on this project's
            // sym/asym domains (see DESIGN.md) -- `iters`'s normal
            // default (50000) would take hours. Capped hard, same
            // reasoning `sweep` already applies to its own correctness
            // probes, just more aggressively given the cost difference.
            let capped_iters = iters.min(3);
            for _ in 0..capped_iters {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_leaf_sweep_with_index([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, &parent_index);
                assert_eq!(expected, got, "leafsweep traversal disagreed with the default on s={s}");
            }
            eprintln!("leafsweep one-Exact: {capped_iters}/{capped_iters} correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..capped_iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_leaf_sweep_with_index([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, &parent_index);
                sink += out.len() as u64;
            }
            eprintln!(
                "PathFstDeltaPow2Bfs(leafsweep, amortized, index excluded) one-Exact: {:.1} ns/call",
                t0.elapsed().as_nanos() as f64 / capped_iters.max(1) as f64
            );
        }
        ("leafsweep", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!(
                "built PathFstDeltaPow2Bfs: {} bytes, {} blocks, {} leaf nodes",
                path_fst.mem_size_bytes(),
                path_fst.block_count(),
                path_fst.leaf_level_node_count()
            );
            let t0 = std::time::Instant::now();
            let parent_index = path_fst.build_parent_index();
            eprintln!("build_parent_index: {:?} ({} entries)", t0.elapsed(), parent_index.len());
            let capped_iters = iters.min(3);
            for _ in 0..capped_iters {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_leaf_sweep_with_index([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, &parent_index);
                assert_eq!(expected, got, "leafsweep traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("leafsweep two-Exact: {capped_iters}/{capped_iters} correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..capped_iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_leaf_sweep_with_index([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, &parent_index);
                sink += out.len() as u64;
            }
            eprintln!(
                "PathFstDeltaPow2Bfs(leafsweep, amortized, index excluded) two-Exact: {:.1} ns/call",
                t0.elapsed().as_nanos() as f64 / capped_iters.max(1) as f64
            );
        }
        ("seq", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(seq) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("seq", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(seq) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("automatonwalk", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..50 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_via_automaton([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                assert_eq!(expected, got, "automatonwalk traversal disagreed with the default on s={s}");
            }
            eprintln!("automatonwalk one-Exact: 50/50 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_via_automaton([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(automatonwalk) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("automatonwalk", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..50 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_via_automaton([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                assert_eq!(expected, got, "automatonwalk traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("automatonwalk two-Exact: 50/50 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_via_automaton([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(automatonwalk) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("composedautomaton", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..50 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_via_composed_automaton([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                assert_eq!(expected, got, "composedautomaton traversal disagreed with the default on s={s}");
            }
            eprintln!("composedautomaton one-Exact: 50/50 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_via_composed_automaton([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(composedautomaton) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        ("composedautomaton", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..50 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_via_composed_automaton([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                assert_eq!(expected, got, "composedautomaton traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("composedautomaton two-Exact: 50/50 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_via_composed_automaton([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(composedautomaton) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallel", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                assert_eq!(expected, got, "parallel traversal disagreed with the default on s={s}");
            }
            eprintln!("parallel one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallel) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallel", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                assert_eq!(expected, got, "parallel traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallel two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallel) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelcutoff", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_cutoff([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallelcutoff traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelcutoff(depth={parallel_depth}) one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_cutoff([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelcutoff depth={parallel_depth}) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelcutoff", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_cutoff([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallelcutoff traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelcutoff(depth={parallel_depth}) two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_cutoff([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelcutoff depth={parallel_depth}) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelsizecutoff", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_size_cutoff([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, min_parallel_size);
                assert_eq!(expected, got, "parallelsizecutoff traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelsizecutoff(x={min_parallel_size}) one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_size_cutoff([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, min_parallel_size);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelsizecutoff x={min_parallel_size}) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelsizecutoff", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_size_cutoff([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, min_parallel_size);
                assert_eq!(expected, got, "parallelsizecutoff traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelsizecutoff(x={min_parallel_size}) two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_size_cutoff([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, min_parallel_size);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelsizecutoff x={min_parallel_size}) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelscoped", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_scoped([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallelscoped traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelscoped(depth={parallel_depth}) one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_scoped([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelscoped depth={parallel_depth}) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelscoped", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_scoped([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallelscoped traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelscoped(depth={parallel_depth}) two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_scoped([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelscoped depth={parallel_depth}) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelchunked", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_scoped_chunked([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth, max_branch);
                assert_eq!(expected, got, "parallelchunked traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelchunked(depth={parallel_depth},branch={max_branch}) one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_scoped_chunked([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth, max_branch);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelchunked depth={parallel_depth},branch={max_branch}) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelchunked", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_scoped_chunked([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth, max_branch);
                assert_eq!(expected, got, "parallelchunked traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelchunked(depth={parallel_depth},branch={max_branch}) two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_scoped_chunked([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth, max_branch);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelchunked depth={parallel_depth},branch={max_branch}) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallellazy", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_lazy([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallellazy traversal disagreed with the default on s={s}");
            }
            eprintln!("parallellazy one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_lazy([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallellazy) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallellazy", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_lazy([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                assert_eq!(expected, got, "parallellazy traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallellazy two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_lazy([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, parallel_depth);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallellazy) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelqueue", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_queue([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                assert_eq!(expected, got, "parallelqueue traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelqueue one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_queue([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelqueue) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelqueue", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_queue([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                assert_eq!(expected, got, "parallelqueue traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelqueue two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_queue([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelqueue) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelmatrix", "one") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_matrix([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, prefer_shallow);
                assert_eq!(expected, got, "parallelmatrix traversal disagreed with the default on s={s}");
            }
            eprintln!("parallelmatrix(prefer_shallow={prefer_shallow}) one-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<u32> = (0..iters).map(|_| rng.next_range(num_s as u64) as u32).collect();
            let t0 = std::time::Instant::now();
            for &s in &probes {
                let out = path_fst.restricted_descent_parallel_matrix([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, prefer_shallow);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelmatrix prefer_shallow={prefer_shallow}) one-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(feature = "parallel")]
        ("parallelmatrix", "two") => {
            let path_fst = PathFstDeltaPow2Bfs::build(triples, [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
            eprintln!("built PathFstDeltaPow2Bfs: {} bytes, {} blocks", path_fst.mem_size_bytes(), path_fst.block_count());
            for _ in 0..5 {
                let s = rng.next_range(num_s as u64) as u32;
                let p = rng.next_range(num_p as u64) as u32;
                let expected = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
                let got = path_fst.restricted_descent_parallel_matrix([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, prefer_shallow);
                assert_eq!(expected, got, "parallelmatrix traversal disagreed with the default on s={s},p={p}");
            }
            eprintln!("parallelmatrix(prefer_shallow={prefer_shallow}) two-Exact: 5/5 correctness probes matched restricted_descent");
            let probes: Vec<(u32, u32)> = (0..iters).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32)).collect();
            let t0 = std::time::Instant::now();
            for &(s, p) in &probes {
                let out = path_fst.restricted_descent_parallel_matrix([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, prefer_shallow);
                sink += out.len() as u64;
            }
            eprintln!("PathFstDeltaPow2Bfs(parallelmatrix prefer_shallow={prefer_shallow}) two-Exact: {:.1} ns/call", t0.elapsed().as_nanos() as f64 / iters.max(1) as f64);
        }
        #[cfg(not(feature = "parallel"))]
        ("parallel" | "parallelcutoff" | "parallelsizecutoff" | "parallelscoped" | "parallelchunked" | "parallellazy" | "parallelqueue" | "parallelmatrix", _) => panic!("rebuild with --features heap,parallel to use this variant"),
        (w, s) => panic!("unknown combination {w:?}/{s:?}, expected heap|pow2|frontier|bfs|layout|layoutbfs|layoutmasked|sweep|leafsweep|seq|automatonwalk|composedautomaton|parallel|parallelcutoff|parallelsizecutoff|parallelscoped|parallelchunked|parallellazy|parallelqueue|parallelmatrix and one|two"),
    }
    std::hint::black_box(sink);
}
