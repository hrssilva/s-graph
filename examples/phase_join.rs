//! Join-pattern (star / triangle / 4-cycle) latency comparison across
//! `heap`, `file`, `mk2`, `ik2` -- the SPARQL-basic-graph-pattern-shaped
//! workload `examples/phase_heap.rs`'s sibling binaries don't measure at
//! all (they only ever time `contains`/row/`predicates_for`, all
//! single-triple-pattern queries; a real SPARQL query is a *conjunction*
//! of patterns joined on shared variables, which is what
//! `crate::common::join::ltj_run` actually answers).
//!
//! Dataset: `crate::common::rdf_gen::graph_patterns_dataset` -- unlike
//! this project's `large_scale_*` scenarios (unstructured random
//! triples, no guaranteed join matches), every hub gets its own private
//! star, triangle, and 4-cycle instance under a fixed predicate schema,
//! so a query anchored at any hub id is guaranteed exactly one match per
//! shape (star: `~fanout` matches) -- see that function's own doc
//! comment.
//!
//! Build and query run in separate processes, same rationale as every
//! other `phase_*.rs` binary in this project (see `phase_heap.rs`'s doc
//! comment). No `insert` phase: this binary is about join latency, not
//! this project's separate insert-cost story.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_join heap build [num_hubs] [fanout]
//! cargo run --release --example phase_join heap query
//! cargo run --release --example phase_join file build
//! cargo run --release --example phase_join file query
//! cargo run --release --example phase_join compressed build
//! cargo run --release --example phase_join compressed query
//! cargo run --release --example phase_join mk2 build
//! cargo run --release --example phase_join mk2 query
//! cargo run --release --example phase_join ik2 build
//! cargo run --release --example phase_join ik2 query
//! ```
//! (`num_hubs`/`fanout` only matter for `build`; `query` reads whatever
//! `build` last wrote for that structure. Defaults: 700,000 hubs,
//! fanout 20 -- about the same 19-20M-triple order of magnitude as this
//! project's `large_scale_*` scenarios.)

#[path = "common/mod.rs"]
mod common;

use common::ik2tree::Ik2Tree;
use common::insertable_bench::phase_paths;
use common::join::{ltj_run, JoinQuery, Term, TriplePattern};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::mk2tree::Mk2Forest;
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
use common::radixtree_dfs_interleaved_varskip_unrolled_generic::Compressed as GenericCompressed;
use common::rdf_gen::{graph_patterns_dataset, GP_CYC_1, GP_CYC_2, GP_CYC_3, GP_CYC_4, GP_STAR, GP_TRI_1, GP_TRI_2, GP_TRI_3};
use common::sorted_neighbors::GraphBackend;

const DEFAULT_NUM_HUBS: u32 = 700_000;
const DEFAULT_FANOUT: u32 = 20;
const NUM_HUB_PROBES: usize = 500;

fn star_query(hub: u32) -> JoinQuery {
    JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Const(hub), predicate: GP_STAR, object: Term::Var(0) }] }
}

fn triangle_query(hub: u32) -> JoinQuery {
    JoinQuery {
        num_vars: 2,
        patterns: vec![
            TriplePattern { subject: Term::Const(hub), predicate: GP_TRI_1, object: Term::Var(0) },
            TriplePattern { subject: Term::Var(0), predicate: GP_TRI_2, object: Term::Var(1) },
            TriplePattern { subject: Term::Var(1), predicate: GP_TRI_3, object: Term::Const(hub) },
        ],
    }
}

fn four_cycle_query(hub: u32) -> JoinQuery {
    JoinQuery {
        num_vars: 3,
        patterns: vec![
            TriplePattern { subject: Term::Const(hub), predicate: GP_CYC_1, object: Term::Var(0) },
            TriplePattern { subject: Term::Var(0), predicate: GP_CYC_2, object: Term::Var(1) },
            TriplePattern { subject: Term::Var(1), predicate: GP_CYC_3, object: Term::Var(2) },
            TriplePattern { subject: Term::Var(2), predicate: GP_CYC_4, object: Term::Const(hub) },
        ],
    }
}

/// Times `hubs.len()` calls to `ltj_run` for one query shape, returning
/// (average latency per call, average result-set size per call -- a
/// correctness sanity check, not just a performance number: triangle and
/// 4-cycle should average ~1.0, star should average close to `fanout`).
fn measure_join<B: GraphBackend>(backend: &B, hubs: &[u32], build_query: impl Fn(u32) -> JoinQuery) -> (std::time::Duration, f64) {
    let t0 = std::time::Instant::now();
    let mut total_results = 0usize;
    for &hub in hubs {
        total_results += ltj_run(&build_query(hub), backend).len();
    }
    let avg = t0.elapsed() / hubs.len() as u32;
    (avg, total_results as f64 / hubs.len() as f64)
}

fn run_join_benchmarks<B: GraphBackend>(backend: &B, hub_probes: &[u32]) {
    let (star_avg, star_results) = measure_join(backend, hub_probes, star_query);
    let (triangle_avg, triangle_results) = measure_join(backend, hub_probes, triangle_query);
    let (cycle_avg, cycle_results) = measure_join(backend, hub_probes, four_cycle_query);
    println!("  star_avg={star_avg:>10.2?} (avg {star_results:.1} results)   triangle_avg={triangle_avg:>10.2?} (avg {triangle_results:.2} results)   four_cycle_avg={cycle_avg:>10.2?} (avg {cycle_results:.2} results)");
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
    let structure = args.next().unwrap_or_else(|| panic!("usage: phase_join <heap|file|compressed|mk2|ik2> <build|query> [num_hubs] [fanout]"));
    let phase = args.next().unwrap_or_else(|| "build".to_string());
    let num_hubs: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_NUM_HUBS);
    let fanout: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_FANOUT);
    let block_capacity: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(65536);
    let (tree_path, probes_path) = phase_paths(&format!("join_{structure}"), &format!("h{num_hubs}_f{fanout}_bc{block_capacity}"));

    match structure.as_str() {
        "heap" => match phase.as_str() {
            "build" => {
                let triples = graph_patterns_dataset(num_hubs, fanout, 2026);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}) ({} triples), structure=heap, phase=build ===", triples.len());
                let hub_probes: Vec<u32> = (0..num_hubs).step_by((num_hubs as usize / NUM_HUB_PROBES).max(1)).collect();
                save_hub_probes(&probes_path, &hub_probes);
                let t0 = std::time::Instant::now();
                let tree = RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(&triples, block_capacity);
                let build_time = t0.elapsed();
                let bytes = tree.serialize();
                std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
                println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={}", human_bytes(tree.mem_size_bytes()), human_bytes(bytes.len()));
                common::insertable_bench::report_rss("build");
            }
            "query" => {
                let hub_probes = load_hub_probes(&probes_path);
                let process_baseline_kb = current_rss_kb().unwrap_or(0);
                let tree = RadixTreeDfsInterleavedVarSkipUnrolled::deserialize_from_path(&tree_path);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}), structure=heap, phase=query ===");
                run_join_benchmarks(&tree, &hub_probes);
                common::insertable_bench::report_rss("query");
                let peak_kb = peak_rss_kb().unwrap_or(0);
                println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
            }
            other => panic!("unknown phase {other:?}"),
        },
        "file" => match phase.as_str() {
            "build" => {
                let triples = graph_patterns_dataset(num_hubs, fanout, 2026);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}) ({} triples), structure=file, phase=build ===", triples.len());
                let hub_probes: Vec<u32> = (0..num_hubs).step_by((num_hubs as usize / NUM_HUB_PROBES).max(1)).collect();
                save_hub_probes(&probes_path, &hub_probes);
                let t0 = std::time::Instant::now();
                let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, block_capacity);
                let build_time = t0.elapsed();
                let bytes = tree.serialize();
                std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
                println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={} height={} blocks={}", human_bytes(tree.mem_size_bytes()), human_bytes(bytes.len()), tree.height(), tree.block_count());
                common::insertable_bench::report_rss("build");
            }
            "query" => {
                let hub_probes = load_hub_probes(&probes_path);
                let process_baseline_kb = current_rss_kb().unwrap_or(0);
                let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize_from_path(&tree_path);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}), structure=file, phase=query ===");

                // DIAGNOSTIC: isolate stream_row (bound dims [S,P], matches
                // this tree's dim_order) from stream_col (bound dims [P,O],
                // mismatched -- S, the schedule's dominant early-split
                // dimension, is left free) to see whether the join
                // benchmark's slowdown traces to the bound-dims shape
                // itself, not join overhead in general.
                {
                    use common::radixtree_dfs_interleaved_varskip_unrolled_file::diag_pread_count;
                    let before = diag_pread_count();
                    let t0 = std::time::Instant::now();
                    let mut total = 0usize;
                    for &hub in &hub_probes {
                        total += tree.query_two_bound([0, 1], [hub, common::rdf_gen::GP_TRI_1], 2).len();
                    }
                    let row_avg = t0.elapsed() / hub_probes.len() as u32;
                    let row_preads = diag_pread_count() - before;
                    let before = diag_pread_count();
                    let t0 = std::time::Instant::now();
                    for &hub in &hub_probes {
                        total += tree.query_two_bound([1, 2], [common::rdf_gen::GP_TRI_3, hub], 0).len();
                    }
                    let col_avg = t0.elapsed() / hub_probes.len() as u32;
                    let col_preads = diag_pread_count() - before;
                    std::hint::black_box(total);
                    println!(
                        "  DIAGNOSTIC: stream_row-shape([S,P] bound)={row_avg:>10.2?} ({:.2} preads/call)  stream_col-shape([P,O] bound)={col_avg:>10.2?} ({:.2} preads/call)  ratio={:.1}x",
                        row_preads as f64 / hub_probes.len() as f64,
                        col_preads as f64 / hub_probes.len() as f64,
                        col_avg.as_secs_f64() / row_avg.as_secs_f64()
                    );
                }

                {
                    use common::radixtree_dfs_interleaved_varskip_unrolled_file::diag_pread_count;
                    let before = diag_pread_count();
                    for &hub in &hub_probes {
                        ltj_run(&triangle_query(hub), &tree);
                    }
                    let triangle_preads = diag_pread_count() - before;
                    let before = diag_pread_count();
                    for &hub in &hub_probes {
                        ltj_run(&four_cycle_query(hub), &tree);
                    }
                    let cycle_preads = diag_pread_count() - before;
                    println!(
                        "  DIAGNOSTIC: triangle preads/call={:.2}  four_cycle preads/call={:.2}",
                        triangle_preads as f64 / hub_probes.len() as f64,
                        cycle_preads as f64 / hub_probes.len() as f64
                    );
                }

                run_join_benchmarks(&tree, &hub_probes);
                common::insertable_bench::report_rss("query");
                let peak_kb = peak_rss_kb().unwrap_or(0);
                println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
            }
            other => panic!("unknown phase {other:?}"),
        },
        "compressed" => match phase.as_str() {
            "build" => {
                let triples = graph_patterns_dataset(num_hubs, fanout, 2026);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}) ({} triples), structure=compressed, phase=build ===", triples.len());
                let hub_probes: Vec<u32> = (0..num_hubs).step_by((num_hubs as usize / NUM_HUB_PROBES).max(1)).collect();
                save_hub_probes(&probes_path, &hub_probes);
                let t0 = std::time::Instant::now();
                let tree = GenericCompressed::build_dynamic(triples, block_capacity);
                let build_time = t0.elapsed();
                let bytes = tree.serialize();
                std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
                println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={} height={} blocks={}", human_bytes(tree.mem_size_bytes()), human_bytes(bytes.len()), tree.height(), tree.block_count());
                common::insertable_bench::report_rss("build");
            }
            "query" => {
                let hub_probes = load_hub_probes(&probes_path);
                let process_baseline_kb = current_rss_kb().unwrap_or(0);
                let tree = GenericCompressed::deserialize_from_path(&tree_path);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}), structure=compressed, phase=query ===");
                run_join_benchmarks(&tree, &hub_probes);
                common::insertable_bench::report_rss("query");
                let peak_kb = peak_rss_kb().unwrap_or(0);
                println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
            }
            other => panic!("unknown phase {other:?}"),
        },
        "mk2" => match phase.as_str() {
            "build" => {
                let triples = graph_patterns_dataset(num_hubs, fanout, 2026);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}) ({} triples), structure=mk2, phase=build ===", triples.len());
                let hub_probes: Vec<u32> = (0..num_hubs).step_by((num_hubs as usize / NUM_HUB_PROBES).max(1)).collect();
                save_hub_probes(&probes_path, &hub_probes);
                let t0 = std::time::Instant::now();
                let forest = Mk2Forest::build(&triples);
                let build_time = t0.elapsed();
                let bytes = forest.serialize();
                std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
                println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={}", human_bytes(forest.mem_size_bytes()), human_bytes(bytes.len()));
                common::insertable_bench::report_rss("build");
            }
            "query" => {
                let hub_probes = load_hub_probes(&probes_path);
                let process_baseline_kb = current_rss_kb().unwrap_or(0);
                let forest = Mk2Forest::deserialize_from_path(&tree_path);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}), structure=mk2, phase=query ===");
                run_join_benchmarks(&forest, &hub_probes);
                common::insertable_bench::report_rss("query");
                let peak_kb = peak_rss_kb().unwrap_or(0);
                println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
            }
            other => panic!("unknown phase {other:?}"),
        },
        "ik2" => match phase.as_str() {
            "build" => {
                let triples = graph_patterns_dataset(num_hubs, fanout, 2026);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}) ({} triples), structure=ik2, phase=build ===", triples.len());
                let hub_probes: Vec<u32> = (0..num_hubs).step_by((num_hubs as usize / NUM_HUB_PROBES).max(1)).collect();
                save_hub_probes(&probes_path, &hub_probes);
                let t0 = std::time::Instant::now();
                let tree = Ik2Tree::build(8, &triples);
                let build_time = t0.elapsed();
                let bytes = tree.serialize();
                std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));
                println!("  build={build_time:>10.2?} mem_size_bytes={} serialized={}", human_bytes(tree.mem_size_bytes()), human_bytes(bytes.len()));
                common::insertable_bench::report_rss("build");
            }
            "query" => {
                let hub_probes = load_hub_probes(&probes_path);
                let process_baseline_kb = current_rss_kb().unwrap_or(0);
                let tree = Ik2Tree::deserialize_from_path(&tree_path);
                println!("=== graph_patterns(hubs={num_hubs}, fanout={fanout}), structure=ik2, phase=query ===");
                run_join_benchmarks(&tree, &hub_probes);
                common::insertable_bench::report_rss("query");
                let peak_kb = peak_rss_kb().unwrap_or(0);
                println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
            }
            other => panic!("unknown phase {other:?}"),
        },
        other => panic!("unknown structure {other:?}; expected \"heap\", \"file\", \"compressed\", \"mk2\", or \"ik2\""),
    }
}
