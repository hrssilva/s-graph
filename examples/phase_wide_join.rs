//! Isolates the one structural asymmetry between backends' join streams:
//! only `Mk2Forest` (via `K2Tree`'s row/col iterators) has a genuine
//! lazy, subtree-pruning `SortedNeighbors::seek` -- `Ik2Tree`, `heap`,
//! and `file` all use `SortedVecIter`, which materializes the whole
//! row/col result eagerly and can only seek *within* that already-
//! computed list (see `crate::common::sorted_neighbors`'s own doc
//! comment). `phase_join.rs`'s star/triangle/4-cycle patterns never
//! stress this, since every intermediate result there is tiny.
//!
//! Dataset: `crate::common::rdf_gen::wide_intersection_dataset` -- each
//! hub gets a *wide* relation (`hub -GP_WIDE-> x`, `wide_fanout` triples)
//! and a *sparse* one reusing `mark_count` of those same `x` values
//! (`hub -GP_MARK-> x`). The join `hub GP_WIDE ?x . hub GP_MARK ?x`
//! always has exactly `mark_count` true answers, however large
//! `wide_fanout` is. This binary measures both that join's latency
//! *and* the cost of just materializing the wide stream alone
//! (`wide_materialize_avg`) -- for a genuinely lazy backend, the join
//! should cost much *less* than materializing the wide side on its own
//! (it never needs to); for an eager backend, the join can't cost less,
//! since materializing the wide side is unavoidable work it does as
//! part of answering the join at all.
//!
//! Build and query run in separate processes, same rationale as every
//! other `phase_*.rs` binary in this project.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_wide_join heap build [num_hubs] [wide_fanout] [mark_count]
//! cargo run --release --example phase_wide_join heap query
//! cargo run --release --example phase_wide_join file build
//! cargo run --release --example phase_wide_join file query
//! cargo run --release --example phase_wide_join mk2 build
//! cargo run --release --example phase_wide_join mk2 query
//! cargo run --release --example phase_wide_join ik2 build
//! cargo run --release --example phase_wide_join ik2 query
//! ```
//! Defaults: 1,000 hubs, wide_fanout 20,000, mark_count 3 (~20M triples).

#[path = "common/mod.rs"]
mod common;

use common::ik2tree::Ik2Tree;
use common::insertable_bench::phase_paths;
use common::join::{ltj_run, JoinQuery, Term, TriplePattern};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::mk2tree::Mk2Forest;
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
use common::rdf_gen::{wide_intersection_dataset, wide_intersection_dataset_spread_s, GP_MARK, GP_WIDE};
use common::sorted_neighbors::GraphBackend;

const DEFAULT_NUM_HUBS: u32 = 1_000;
const DEFAULT_WIDE_FANOUT: u32 = 20_000;
const DEFAULT_MARK_COUNT: u32 = 3;
const NUM_HUB_PROBES: usize = 300;

fn wide_intersection_query(hub: u32) -> JoinQuery {
    JoinQuery {
        num_vars: 1,
        patterns: vec![
            TriplePattern { subject: Term::Const(hub), predicate: GP_WIDE, object: Term::Var(0) },
            TriplePattern { subject: Term::Const(hub), predicate: GP_MARK, object: Term::Var(0) },
        ],
    }
}

/// Fully drains a `stream_row(GP_WIDE, hub)` stream, counting its
/// length -- the cost of materializing the wide relation alone, with no
/// intersection at all. This is the floor every eager (`SortedVecIter`)
/// backend's join must pay at minimum; a genuinely lazy backend's join
/// can beat it.
fn measure_wide_materialize<B: GraphBackend>(backend: &B, hubs: &[u32]) -> (std::time::Duration, f64) {
    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    for &hub in hubs {
        let mut s = backend.stream_row(GP_WIDE, hub);
        while s.current().is_some() {
            total += 1;
            s.advance();
        }
    }
    let avg = t0.elapsed() / hubs.len() as u32;
    (avg, total as f64 / hubs.len() as f64)
}

/// DIAGNOSTIC: splits `measure_wide_materialize`'s total into
/// "construct the stream" (for an eager `SortedVecIter` backend, this is
/// where the *entire* row gets computed -- `GraphBackend::stream_row`
/// itself) vs. "drain it one element at a time" (walking the
/// already-materialized result through the `Box<dyn SortedNeighbors>`
/// vtable, `wide_fanout` times) -- to see how much of each backend's
/// total materialize cost is which.
fn measure_construct_vs_drain<B: GraphBackend>(backend: &B, hubs: &[u32]) -> (std::time::Duration, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let mut streams: Vec<_> = hubs.iter().map(|&hub| backend.stream_row(GP_WIDE, hub)).collect();
    let construct_avg = t0.elapsed() / hubs.len() as u32;

    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    for s in streams.iter_mut() {
        while s.current().is_some() {
            total += 1;
            s.advance();
        }
    }
    let drain_avg = t0.elapsed() / hubs.len() as u32;
    std::hint::black_box(total);
    (construct_avg, drain_avg)
}

fn measure_join<B: GraphBackend>(backend: &B, hubs: &[u32], build_query: impl Fn(u32) -> JoinQuery) -> (std::time::Duration, f64) {
    let t0 = std::time::Instant::now();
    let mut total_results = 0usize;
    for &hub in hubs {
        total_results += ltj_run(&build_query(hub), backend).len();
    }
    let avg = t0.elapsed() / hubs.len() as u32;
    (avg, total_results as f64 / hubs.len() as f64)
}

fn run_wide_join_benchmark<B: GraphBackend>(backend: &B, hub_probes: &[u32]) {
    let (join_avg, join_results) = measure_join(backend, hub_probes, wide_intersection_query);
    let (materialize_avg, wide_len) = measure_wide_materialize(backend, hub_probes);
    println!(
        "  wide_intersection_join_avg={join_avg:>10.2?} (avg {join_results:.2} results, expect {DEFAULT_MARK_COUNT})   wide_materialize_avg={materialize_avg:>10.2?} (avg {wide_len:.1} results)   join/materialize_ratio={:.3}x",
        join_avg.as_secs_f64() / materialize_avg.as_secs_f64()
    );
    let (construct_avg, drain_avg) = measure_construct_vs_drain(backend, hub_probes);
    println!("  DIAGNOSTIC: construct_avg={construct_avg:>10.2?}  drain_avg={drain_avg:>10.2?}  construct_share={:.1}%", 100.0 * construct_avg.as_secs_f64() / (construct_avg.as_secs_f64() + drain_avg.as_secs_f64()));
}

fn save_hub_probes(path: &std::path::Path, hubs: &[u32]) {
    let mut out = Vec::with_capacity(4 + hubs.len() * 4);
    out.extend_from_slice(&(hubs.len() as u32).to_le_bytes());
    for &h in hubs {
        out.extend_from_slice(&h.to_le_bytes());
    }
    std::fs::write(path, &out).unwrap_or_else(|e| panic!("failed to write {path:?}: {e}"));
}

fn load_hub_probes(path: &std::path::Path) -> Vec<u32> {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?} (run the `build` phase first): {e}"));
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    (0..n).map(|i| u32::from_le_bytes(buf[4 + i * 4..8 + i * 4].try_into().unwrap())).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let structure = args.next().unwrap_or_else(|| panic!("usage: phase_wide_join <heap|file|mk2|ik2> <build|query> [num_hubs] [wide_fanout] [mark_count] [spread]"));
    let phase = args.next().unwrap_or_else(|| "build".to_string());
    let num_hubs: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_NUM_HUBS);
    let wide_fanout: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_WIDE_FANOUT);
    let mark_count: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MARK_COUNT);
    // `spread`: use `wide_intersection_dataset_spread_s` (hub ids spread
    // across the same magnitude as the object domain) instead of the
    // default packed-into-[0,num_hubs) dataset -- see that function's
    // own doc comment for why this specifically tests whether
    // `Mk2Forest`'s high cost on the packed dataset traces to `K2Tree`'s
    // square-matrix row-discrimination problem.
    let spread = args.next().is_some_and(|s| s == "spread" || s == "1" || s == "true");
    let stride = wide_fanout.max(1);
    let (tree_path, probes_path) = phase_paths(&format!("wide_join_{structure}"), &format!("h{num_hubs}_w{wide_fanout}_m{mark_count}{}", if spread { "_spread" } else { "" }));

    macro_rules! run_build {
        ($build_fn:expr, $struct_name:literal) => {{
            let triples = if spread {
                wide_intersection_dataset_spread_s(num_hubs, wide_fanout, mark_count, 2026)
            } else {
                wide_intersection_dataset(num_hubs, wide_fanout, mark_count, 2026)
            };
            println!(
                "=== wide_intersection(hubs={num_hubs}, wide_fanout={wide_fanout}, mark_count={mark_count}, spread={spread}) ({} triples), structure={} phase=build ===",
                triples.len(),
                $struct_name
            );
            let hub_probes: Vec<u32> = (0..num_hubs)
                .step_by((num_hubs as usize / NUM_HUB_PROBES).max(1))
                .map(|i| if spread { i * stride } else { i })
                .collect();
            save_hub_probes(&probes_path, &hub_probes);
            let t0 = std::time::Instant::now();
            let structure = $build_fn(&triples);
            let build_time = t0.elapsed();
            let mem = structure.mem_size_bytes();
            let bytes = structure.serialize();
            std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
            println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={}", human_bytes(mem), human_bytes(bytes.len()));
            common::insertable_bench::report_rss("build");
        }};
    }

    macro_rules! run_query {
        ($deserialize_fn:expr) => {{
            let hub_probes = load_hub_probes(&probes_path);
            let process_baseline_kb = current_rss_kb().unwrap_or(0);
            let structure = $deserialize_fn(&tree_path);
            println!(
                "  DIAGNOSTIC: height={} blocks={} mem_size_bytes={} (spread={spread}, S spans up to ~{})",
                structure.height(),
                structure.block_count(),
                human_bytes(structure.mem_size_bytes()),
                hub_probes.iter().copied().max().unwrap_or(0)
            );
            println!("=== wide_intersection(hubs={num_hubs}, wide_fanout={wide_fanout}, mark_count={mark_count}, spread={spread}), phase=query ===");
            run_wide_join_benchmark(&structure, &hub_probes);
            common::insertable_bench::report_rss("query");
            let peak_kb = peak_rss_kb().unwrap_or(0);
            println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
        }};
    }

    match (structure.as_str(), phase.as_str()) {
        ("heap", "build") => run_build!(|t: &_| RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(t, 65536), "heap"),
        ("heap", "query") => run_query!(|p: &std::path::Path| RadixTreeDfsInterleavedVarSkipUnrolled::deserialize_from_path(p)),
        ("file", "build") => run_build!(|t: &_| RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(t, 65536), "file"),
        ("file", "query") => run_query!(|p: &std::path::Path| RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize_from_path(p)),
        ("mk2", "build") => run_build!(|t: &_| Mk2Forest::build(t), "mk2"),
        ("mk2", "query") => {
            let hub_probes = load_hub_probes(&probes_path);
            let forest = Mk2Forest::deserialize_from_path(&tree_path);
            // DIAGNOSTIC: raw K2Tree::row_iter().next() cost, bypassing
            // GraphBackend/Box<dyn SortedNeighbors>/Stream entirely, to
            // isolate whether mk2's high `stream_row` cost is intrinsic
            // to RowIter's own descent or an artifact of that wrapping.
            let tree = forest.tree_for(GP_WIDE).expect("GP_WIDE tree must exist");
            let breakdown = tree.size_breakdown();
            println!(
                "  DIAGNOSTIC: GP_WIDE K2Tree n={} height={} k={} mem_size_bytes={} (t={}, l={}) (spread={spread}, S spans up to ~{} out of [0,{}))",
                tree.n(),
                tree.height(),
                tree.k(),
                human_bytes(tree.mem_size_bytes()),
                human_bytes(breakdown[0].1),
                human_bytes(breakdown[1].1),
                hub_probes.iter().copied().max().unwrap_or(0),
                tree.n()
            );
            let t0 = std::time::Instant::now();
            let mut sum = 0u64;
            for &hub in &hub_probes {
                if let Some(v) = tree.row_iter(hub).next() {
                    sum += v as u64;
                }
            }
            let raw_first_next_avg = t0.elapsed() / hub_probes.len() as u32;
            std::hint::black_box(sum);
            println!("  DIAGNOSTIC: raw K2Tree::row_iter(hub).next() avg={raw_first_next_avg:>10.2?} (bypasses Box<dyn SortedNeighbors>/Stream)");

            run_wide_join_benchmark(&forest, &hub_probes);
            common::insertable_bench::report_rss("query");
        }
        ("ik2", "build") => run_build!(|t: &_| Ik2Tree::build(8, t), "ik2"),
        ("ik2", "query") => {
            let hub_probes = load_hub_probes(&probes_path);
            let process_baseline_kb = current_rss_kb().unwrap_or(0);
            let tree = Ik2Tree::deserialize_from_path(&tree_path);
            println!(
                "  DIAGNOSTIC: Ik2Tree n={} height={} mem_size_bytes={} (spread={spread}, S spans up to ~{} out of [0,{}))",
                tree.n(),
                tree.height(),
                human_bytes(tree.mem_size_bytes()),
                hub_probes.iter().copied().max().unwrap_or(0),
                tree.n()
            );
            println!("=== wide_intersection(hubs={num_hubs}, wide_fanout={wide_fanout}, mark_count={mark_count}, spread={spread}), phase=query ===");
            run_wide_join_benchmark(&tree, &hub_probes);
            common::insertable_bench::report_rss("query");
            let peak_kb = peak_rss_kb().unwrap_or(0);
            println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
        }
        (other_structure, other_phase) => panic!("unknown structure/phase {other_structure:?}/{other_phase:?}"),
    }
}
