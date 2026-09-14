//! Per-phase peak-RSS measurement for `Ik2Tree`: build and query each
//! run in their *own* process. See `phase_mk2.rs`'s own doc comment for
//! the full rationale and for why there's no `insert` phase.
//!
//! `k = 8` (64-way branching), matching this project's other `Ik2Tree`
//! usages -- "RadixTree's node granularity" per `common::ik2tree`'s own
//! demo.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_ik2 build large_scale_pred_skew_only
//! cargo run --release --example phase_ik2 query large_scale_pred_skew_only
//! ```

#[path = "common/mod.rs"]
mod common;

use common::ik2tree::Ik2Tree;
use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, load_probe_data, measure_contains_and_row, measure_predicates_for, phase_paths, report_rss, save_probe_data, ProbeData, LARGE_SCALE_SCENARIO_NAMES};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};

const STRUCTURE: &str = "ik2";
const K: u32 = 8;

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
            let tree = Ik2Tree::build(K, &triples);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let bytes = tree.serialize();
            std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));

            println!(
                "  build={build_time:>10.2?} mem_size_bytes={:>10} serialized={:>10} predicates={} height={}",
                human_bytes(mem),
                human_bytes(bytes.len()),
                tree.num_predicates(),
                tree.height()
            );
            report_rss("build");
        }
        "query" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=query ===");
            let probes = load_probe_data(&probes_path);
            let process_baseline_kb = current_rss_kb().unwrap_or(0);

            let t0 = std::time::Instant::now();
            let tree = Ik2Tree::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.locate(p, s, o), |s, p| tree.row_bound(p, s).len(), &probes.contains_probes, &probes.row_probes);

            // Unbound-predicate query, same shape as `phase_mk2.rs`'s --
            // interleaving is supposed to make this O(depth), not
            // O(predicates), unlike `Mk2Forest`'s independent-trees scan.
            let so_probes: Vec<(u32, u32)> = probes.contains_probes.iter().map(|&(s, _, o)| (s, o)).collect();
            let predicates_for_avg = measure_predicates_for(|s, o| tree.predicates_for(s, o).len(), &so_probes);

            println!(
                "  deserialize={deserialize_time:>10.2?} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} predicates_for_avg={predicates_for_avg:>10.2?} predicates={} height={}",
                tree.num_predicates(),
                tree.height()
            );
            report_rss("query");
            let peak_kb = peak_rss_kb().unwrap_or(0);
            println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
        }
        other => panic!("unknown phase {other:?}; expected \"build\" or \"query\""),
    }
}
