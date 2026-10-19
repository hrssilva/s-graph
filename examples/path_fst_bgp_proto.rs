//! BGP-style ("two-Exact": `[Exact, Exact, Free]`, "one-Exact":
//! `[Exact, Free, Free]`) query comparison between `Heap::restricted_descent`
//! and `PathFstDeltaPow2::restricted_descent` -- see `DESIGN.md`'s "BGP-style
//! queries on the FST" entry (and its two follow-up corrections) for the full
//! history: an `(address, level)`-keyed caching experiment that never paid
//! off (`restricted_descent_with_stats`/`restricted_descent_fastpath*`,
//! still kept below as a documented negative result), and the real fix --
//! `restricted_descent`/`restricted_descent_visit` now precompute one bitmask
//! per tree level up front (mirroring `Heap`'s own `precompute_level_masks`/
//! `query_rec` fast path) instead of recomputing a mask from the full
//! restrictions array on every node, which is what actually closed most of
//! the `one-Exact` gap against `Heap`. This driver measures both: the cache
//! hit ratio and `use_cache=true` vs `false` ablation for the (still negative)
//! caching experiments, and the default traversal's timing against `Heap`, on
//! top of the already-established `contains()`-only comparison in
//! `path_fst_proto.rs`.
//!
//! Kept in the repository (not deleted after measuring), same convention as
//! every other prototype this project. Run with:
//!   cargo run --release --example path_fst_bgp_proto --features heap

use std::ops::ControlFlow;
use std::time::Instant;

use s_graph::path_fst::{PathFstDeltaPow2, PathFstDeltaPow2Bfs};
use s_graph::tree::DimRestriction;
use s_graph::{Heap, Triple};

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::{uniform, zipfian_entities, SplitMix64};

fn report(name: &str, triples: Vec<Triple>, num_s: u32, num_p: u32, num_o: u32) {
    println!("=== {name} (domain={num_s}x{num_p}x{num_o}, {} triples) ===", triples.len());

    let heap_tree = Heap::build_dynamic(triples.clone(), s_graph::DEFAULT_BLOCK_CAPACITY);
    let path_fst = PathFstDeltaPow2::build(triples.clone(), [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);
    let path_fst_bfs = PathFstDeltaPow2Bfs::build(triples.clone(), [num_s as u64, num_p as u64, num_o as u64], s_graph::DEFAULT_BLOCK_CAPACITY);

    let mut rng = SplitMix64::new(777);

    // ---- two-Exact: Exact(S), Exact(P), Free(O), emit O ----
    let two_exact_n = 2_000usize;
    let mut two_exact_probes: Vec<(u32, u32)> = Vec::with_capacity(two_exact_n);
    for _ in 0..two_exact_n {
        two_exact_probes.push((rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32));
    }

    let mut mismatches = 0u64;
    let mut heap_results_total = 0u64;
    for &(s, p) in &two_exact_probes {
        let heap_got = heap_tree.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_got = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let mut fst_stream_got = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            fst_stream_got.push(v);
            ControlFlow::Continue(())
        });
        fst_stream_got.sort_unstable();
        fst_stream_got.dedup();
        let fst_fast_got = path_fst.restricted_descent_fastpath([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_prefetch_got = path_fst.restricted_descent_prefetch([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_frontier_got = path_fst.restricted_descent_frontier([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_bfs_got = path_fst.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_bfs_layout_dfs_got = path_fst_bfs.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_bfs_layout_bfs_got = path_fst_bfs.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        let fst_bfs_layout_masked_got = path_fst_bfs.restricted_descent_bfs_masked([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2);
        heap_results_total += heap_got.len() as u64;
        if heap_got != fst_got
            || heap_got != fst_stream_got
            || heap_got != fst_fast_got
            || heap_got != fst_prefetch_got
            || heap_got != fst_frontier_got
            || heap_got != fst_bfs_got
            || heap_got != fst_bfs_layout_dfs_got
            || heap_got != fst_bfs_layout_bfs_got
            || heap_got != fst_bfs_layout_masked_got
        {
            mismatches += 1;
            eprintln!(
                "  TWO-EXACT MISMATCH (s={s},p={p}): heap={} values, fst={} values, fst_stream={} values, fst_fast={} values, fst_prefetch={} values, fst_frontier={} values, fst_bfs={} values, fst_bfs_layout_dfs={} values, fst_bfs_layout_bfs={} values, fst_bfs_layout_masked={} values",
                heap_got.len(),
                fst_got.len(),
                fst_stream_got.len(),
                fst_fast_got.len(),
                fst_prefetch_got.len(),
                fst_frontier_got.len(),
                fst_bfs_got.len(),
                fst_bfs_layout_dfs_got.len(),
                fst_bfs_layout_bfs_got.len(),
                fst_bfs_layout_masked_got.len()
            );
        }
    }
    println!("  two-Exact correctness: {two_exact_n} probes, {mismatches} mismatches, {heap_results_total} total results");
    assert_eq!(mismatches, 0, "two-Exact PathFstDeltaPow2 disagreed with Heap on {mismatches} probes");

    let t0 = Instant::now();
    for &(s, p) in &two_exact_probes {
        std::hint::black_box(heap_tree.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2));
    }
    let heap_elapsed = t0.elapsed();

    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let t1 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let (vals, hits, misses) = path_fst.restricted_descent_with_stats([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, true);
        std::hint::black_box(vals);
        total_hits += hits;
        total_misses += misses;
    }
    let fst_cached_elapsed = t1.elapsed();

    let t2 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let (vals, _, _) = path_fst.restricted_descent_with_stats([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, false);
        std::hint::black_box(vals);
    }
    let fst_nocache_elapsed = t2.elapsed();

    let t3 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_stream_elapsed = t3.elapsed();

    let mut fast_hits = 0u64;
    let mut fast_misses = 0u64;
    let t4 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let mut out = Vec::new();
        let (_, hits, misses) = path_fst.restricted_descent_fastpath_with_stats([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue::<()>(())
        });
        std::hint::black_box(out);
        fast_hits += hits;
        fast_misses += misses;
    }
    let fst_fast_elapsed = t4.elapsed();

    let t5 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_prefetch_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_prefetch_elapsed = t5.elapsed();

    let t6 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_frontier_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_frontier_elapsed = t6.elapsed();

    let t7 = Instant::now();
    for &(s, p) in &two_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_bfs_elapsed = t7.elapsed();

    let t8 = Instant::now();
    for &(s, p) in &two_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2));
    }
    let fst_bfs_layout_dfs_elapsed = t8.elapsed();

    let t9 = Instant::now();
    for &(s, p) in &two_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2));
    }
    let fst_bfs_layout_bfs_elapsed = t9.elapsed();

    let t10 = Instant::now();
    for &(s, p) in &two_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent_bfs_masked([DimRestriction::Exact(s), DimRestriction::Exact(p), DimRestriction::Free], 2));
    }
    let fst_bfs_layout_masked_elapsed = t10.elapsed();

    println!(
        "  two-Exact timing over {two_exact_n} probes: Heap={:.1}us/call, FST(cached)={:.1}us/call ({:.2}x), FST(no-cache)={:.1}us/call ({:.2}x), FST(stream, precomputed masks)={:.1}us/call ({:.2}x), FST(fastpath)={:.1}us/call ({:.2}x), FST(prefetch)={:.1}us/call ({:.2}x), FST(frontier)={:.1}us/call ({:.2}x), FST(bfs, no heap buffers)={:.1}us/call ({:.2}x), FST(bfs layout, dfs)={:.1}us/call ({:.2}x), FST(bfs layout, bfs)={:.1}us/call ({:.2}x), FST(bfs layout, masked)={:.1}us/call ({:.2}x)",
        heap_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_cached_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_cached_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_nocache_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_nocache_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_stream_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_stream_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_fast_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_fast_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_prefetch_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_prefetch_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_frontier_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_frontier_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_bfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_dfs_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_bfs_layout_dfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_bfs_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_bfs_layout_bfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_masked_elapsed.as_micros() as f64 / two_exact_n as f64,
        fst_bfs_layout_masked_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
    );
    println!(
        "  two-Exact FST cache (whole-descent): {total_hits} hits, {total_misses} misses ({:.2}% hit rate)",
        100.0 * total_hits as f64 / (total_hits + total_misses).max(1) as f64
    );
    println!(
        "  two-Exact FST cache (fully-open-only): {fast_hits} hits, {fast_misses} misses ({:.2}% hit rate)",
        100.0 * fast_hits as f64 / (fast_hits + fast_misses).max(1) as f64
    );

    // ---- one-Exact: Exact(S), Free(P), Free(O), emit O ----
    let one_exact_n = 500usize;
    let mut one_exact_probes: Vec<u32> = Vec::with_capacity(one_exact_n);
    for _ in 0..one_exact_n {
        one_exact_probes.push(rng.next_range(num_s as u64) as u32);
    }

    let mut mismatches = 0u64;
    let mut heap_results_total = 0u64;
    for &s in &one_exact_probes {
        let heap_got = heap_tree.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_got = path_fst.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let mut fst_stream_got = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            fst_stream_got.push(v);
            ControlFlow::Continue(())
        });
        fst_stream_got.sort_unstable();
        fst_stream_got.dedup();
        let fst_fast_got = path_fst.restricted_descent_fastpath([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_prefetch_got = path_fst.restricted_descent_prefetch([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_frontier_got = path_fst.restricted_descent_frontier([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_bfs_got = path_fst.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_bfs_layout_dfs_got = path_fst_bfs.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_bfs_layout_bfs_got = path_fst_bfs.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        let fst_bfs_layout_masked_got = path_fst_bfs.restricted_descent_bfs_masked([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2);
        heap_results_total += heap_got.len() as u64;
        if heap_got != fst_got
            || heap_got != fst_stream_got
            || heap_got != fst_fast_got
            || heap_got != fst_prefetch_got
            || heap_got != fst_frontier_got
            || heap_got != fst_bfs_got
            || heap_got != fst_bfs_layout_dfs_got
            || heap_got != fst_bfs_layout_bfs_got
            || heap_got != fst_bfs_layout_masked_got
        {
            mismatches += 1;
            eprintln!(
                "  ONE-EXACT MISMATCH (s={s}): heap={} values, fst={} values, fst_stream={} values, fst_fast={} values, fst_prefetch={} values, fst_frontier={} values, fst_bfs={} values, fst_bfs_layout_dfs={} values, fst_bfs_layout_bfs={} values, fst_bfs_layout_masked={} values",
                heap_got.len(),
                fst_got.len(),
                fst_stream_got.len(),
                fst_fast_got.len(),
                fst_prefetch_got.len(),
                fst_frontier_got.len(),
                fst_bfs_got.len(),
                fst_bfs_layout_dfs_got.len(),
                fst_bfs_layout_bfs_got.len(),
                fst_bfs_layout_masked_got.len()
            );
        }
    }
    println!("  one-Exact correctness: {one_exact_n} probes, {mismatches} mismatches, {heap_results_total} total results");
    assert_eq!(mismatches, 0, "one-Exact PathFstDeltaPow2 disagreed with Heap on {mismatches} probes");

    let t0 = Instant::now();
    for &s in &one_exact_probes {
        std::hint::black_box(heap_tree.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2));
    }
    let heap_elapsed = t0.elapsed();

    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let t1 = Instant::now();
    for &s in &one_exact_probes {
        let (vals, hits, misses) = path_fst.restricted_descent_with_stats([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, true);
        std::hint::black_box(vals);
        total_hits += hits;
        total_misses += misses;
    }
    let fst_cached_elapsed = t1.elapsed();

    let t2 = Instant::now();
    for &s in &one_exact_probes {
        let (vals, _, _) = path_fst.restricted_descent_with_stats([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, false);
        std::hint::black_box(vals);
    }
    let fst_nocache_elapsed = t2.elapsed();

    let t3 = Instant::now();
    for &s in &one_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_stream_elapsed = t3.elapsed();

    let mut fast_hits = 0u64;
    let mut fast_misses = 0u64;
    let t4 = Instant::now();
    for &s in &one_exact_probes {
        let mut out = Vec::new();
        let (_, hits, misses) = path_fst.restricted_descent_fastpath_with_stats([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue::<()>(())
        });
        std::hint::black_box(out);
        fast_hits += hits;
        fast_misses += misses;
    }
    let fst_fast_elapsed = t4.elapsed();

    let t5 = Instant::now();
    for &s in &one_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_prefetch_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_prefetch_elapsed = t5.elapsed();

    let t6 = Instant::now();
    for &s in &one_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_frontier_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_frontier_elapsed = t6.elapsed();

    let t7 = Instant::now();
    for &s in &one_exact_probes {
        let mut out = Vec::new();
        let _: ControlFlow<()> = path_fst.restricted_descent_bfs_visit([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2, |v| {
            out.push(v);
            ControlFlow::Continue(())
        });
        std::hint::black_box(out);
    }
    let fst_bfs_elapsed = t7.elapsed();

    let t8 = Instant::now();
    for &s in &one_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2));
    }
    let fst_bfs_layout_dfs_elapsed = t8.elapsed();

    let t9 = Instant::now();
    for &s in &one_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent_bfs([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2));
    }
    let fst_bfs_layout_bfs_elapsed = t9.elapsed();

    let t10 = Instant::now();
    for &s in &one_exact_probes {
        std::hint::black_box(path_fst_bfs.restricted_descent_bfs_masked([DimRestriction::Exact(s), DimRestriction::Free, DimRestriction::Free], 2));
    }
    let fst_bfs_layout_masked_elapsed = t10.elapsed();

    println!(
        "  one-Exact timing over {one_exact_n} probes: Heap={:.1}us/call, FST(cached)={:.1}us/call ({:.2}x), FST(no-cache)={:.1}us/call ({:.2}x), FST(stream, precomputed masks)={:.1}us/call ({:.2}x), FST(fastpath)={:.1}us/call ({:.2}x), FST(prefetch)={:.1}us/call ({:.2}x), FST(frontier)={:.1}us/call ({:.2}x), FST(bfs, no heap buffers)={:.1}us/call ({:.2}x), FST(bfs layout, dfs)={:.1}us/call ({:.2}x), FST(bfs layout, bfs)={:.1}us/call ({:.2}x), FST(bfs layout, masked)={:.1}us/call ({:.2}x)",
        heap_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_cached_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_cached_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_nocache_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_nocache_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_stream_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_stream_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_fast_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_fast_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_prefetch_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_prefetch_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_frontier_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_frontier_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_bfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_dfs_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_bfs_layout_dfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_bfs_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_bfs_layout_bfs_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
        fst_bfs_layout_masked_elapsed.as_micros() as f64 / one_exact_n as f64,
        fst_bfs_layout_masked_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64().max(1e-12),
    );
    println!(
        "  one-Exact FST cache (whole-descent): {total_hits} hits, {total_misses} misses ({:.2}% hit rate)",
        100.0 * total_hits as f64 / (total_hits + total_misses).max(1) as f64
    );
    println!(
        "  one-Exact FST cache (fully-open-only): {fast_hits} hits, {fast_misses} misses ({:.2}% hit rate)",
        100.0 * fast_hits as f64 / (fast_hits + fast_misses).max(1) as f64
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
