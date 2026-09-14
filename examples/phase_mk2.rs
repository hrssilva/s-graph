//! Per-phase peak-RSS measurement for `Mk2Forest`: build and query each
//! run in their *own* process, matching the
//! `radixtree_dfs_interleaved_varskip_unrolled*` family's `phase_*.rs`
//! binaries (see `phase_heap.rs`'s own doc comment for the full
//! rationale) -- now that `Mk2Forest` has `serialize`/`deserialize`
//! (see `common::mk2tree`'s own doc comments), it no longer needs the
//! single combined-process measurement `phase_mk2.rs` used before.
//!
//! No `insert` phase: `Mk2Forest` has no insert operation (or any other
//! structure in this k²-tree family) -- it's build-once, unlike the
//! `radixtree_dfs_interleaved_varskip_unrolled*` family.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_mk2 build large_scale_pred_skew_only
//! cargo run --release --example phase_mk2 query large_scale_pred_skew_only
//! ```

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, load_probe_data, measure_contains_and_row, measure_predicates_for, phase_paths, report_rss, save_probe_data, ProbeData, LARGE_SCALE_SCENARIO_NAMES};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::mk2tree::Mk2Forest;

const STRUCTURE: &str = "mk2";

fn main() {
    let mut args = std::env::args().skip(1);
    let phase = args.next().unwrap_or_else(|| "build".to_string());
    let scenario = args.next().unwrap_or_else(|| LARGE_SCALE_SCENARIO_NAMES[0].to_string());
    let (tree_path, probes_path) = phase_paths(STRUCTURE, &scenario);

    match phase.as_str() {
        "build" => {
            let triples = large_scale_scenario(&scenario);
            println!("=== {scenario} ({} triples), structure={STRUCTURE}, phase=build ===", triples.len());
            let (contains_probes, row_probes) = contains_and_row_probes(&triples);
            let (max_s, max_p, max_o) = domain_bounds(&triples);
            save_probe_data(&probes_path, &ProbeData { max_s, max_p, max_o, contains_probes, row_probes });

            let t0 = std::time::Instant::now();
            let forest = Mk2Forest::build(&triples);
            let build_time = t0.elapsed();
            let mem = forest.mem_size_bytes();
            let bytes = forest.serialize();
            std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));

            println!("  build={build_time:>10.2?} mem_size_bytes={:>10} serialized={:>10} predicates={}", human_bytes(mem), human_bytes(bytes.len()), forest.num_predicates());
            report_rss("build");
        }
        "query" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=query ===");
            let probes = load_probe_data(&probes_path);
            let process_baseline_kb = current_rss_kb().unwrap_or(0);

            let t0 = std::time::Instant::now();
            let forest = Mk2Forest::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| forest.contains(s, p, o), |s, p| forest.row(p, s).len(), &probes.contains_probes, &probes.row_probes);

            // Unbound-predicate query: this is what
            // `large_scale_many_predicates_*` scenarios exist to stress
            // -- `Mk2Forest` must scan every one of its `num_predicates`
            // independent trees to answer this, unlike a bound-predicate
            // `contains`/`row`. Both the sequential and `rayon`-parallel
            // variants are timed, since this is the one query shape
            // where `Mk2Forest`'s independent-trees layout gives it a
            // genuine parallel option `RadixTree`/`Ik2Tree` don't have in
            // the same embarrassingly-parallel way.
            let so_probes: Vec<(u32, u32)> = probes.contains_probes.iter().map(|&(s, _, o)| (s, o)).collect();
            let predicates_for_seq_avg = measure_predicates_for(|s, o| forest.predicates_for_seq(s, o).len(), &so_probes);
            let predicates_for_parallel_avg = measure_predicates_for(|s, o| forest.predicates_for_parallel(s, o).len(), &so_probes);

            println!(
                "  deserialize={deserialize_time:>10.2?} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} predicates_for_seq_avg={predicates_for_seq_avg:>10.2?} predicates_for_parallel_avg={predicates_for_parallel_avg:>10.2?} predicates={}",
                forest.num_predicates()
            );
            report_rss("query");
            let peak_kb = peak_rss_kb().unwrap_or(0);
            println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
        }
        other => panic!("unknown phase {other:?}; expected \"build\" or \"query\""),
    }
}
