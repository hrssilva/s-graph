//! Per-phase peak-RSS measurement for `RadixTreeDfsInterleavedVarSkipUnrolled`
//! (heap-backed blocks): build, query, and insert each run in their
//! *own* process, so each phase's peak RSS (`VmHWM`) reflects only that
//! phase -- not, as in `rss_heap.rs`, one process's peak across build
//! *and* query *and* insert together, which (per this project's own
//! measurements) is dominated almost entirely by build-time transients
//! (the raw input dataset, counting-sort scratch buffers) common to
//! every phase, making it impossible to see what query or insert
//! *alone* actually cost in memory.
//!
//! The `build` phase serializes the finished structure (see
//! `RadixTreeDfsInterleavedVarSkipUnrolled::serialize`, and
//! `crate::common::tree_serialize`'s own doc comment for the format) to
//! a temp file, plus a small sidecar of query probes and domain bounds
//! (`crate::common::insertable_bench::save_probe_data`) -- the `query`
//! and `insert` phases load *those*, not the original ~20M-triple
//! dataset, so their own peak RSS isn't polluted by a dataset they
//! genuinely don't need to touch.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_heap build large_scale_pred_skew_only
//! cargo run --release --example phase_heap query large_scale_pred_skew_only
//! cargo run --release --example phase_heap insert large_scale_pred_skew_only
//! ```
//! (`query`/`insert` must run after `build` has written the serialized
//! structure for that scenario; scenario defaults to
//! `large_scale_pred_skew_only` if omitted.)

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, load_probe_data, measure_contains_and_row, measure_predicates_for, new_triple, phase_paths, report_rss, save_probe_data, ProbeData, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::rdf_gen::SplitMix64;

const STRUCTURE: &str = "heap";

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
            let tree = RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let bytes = tree.serialize();
            std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));

            println!("  build={build_time:>10.2?} mem_size_bytes={:>10} serialized={:>10} blocks={}", human_bytes(mem), human_bytes(bytes.len()), tree.block_count());
            report_rss("build");
        }
        "query" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=query ===");
            let probes = load_probe_data(&probes_path);
            let process_baseline_kb = common::metrics::current_rss_kb().unwrap_or(0);

            let t0 = std::time::Instant::now();
            let tree = RadixTreeDfsInterleavedVarSkipUnrolled::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &probes.contains_probes, &probes.row_probes);

            let so_probes: Vec<(u32, u32)> = probes.contains_probes.iter().map(|&(s, _, o)| (s, o)).collect();
            let predicates_for_avg = measure_predicates_for(|s, o| tree.predicates_for(s, o).len(), &so_probes);

            println!("  deserialize={deserialize_time:>10.2?} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} predicates_for_avg={predicates_for_avg:>10.2?} blocks={}", tree.block_count());
            report_rss("query");
            let peak_kb = common::metrics::peak_rss_kb().unwrap_or(0);
            println!("  process_baseline={} peak_minus_baseline={}", human_bytes(process_baseline_kb as usize * 1024), human_bytes(peak_kb.saturating_sub(process_baseline_kb) as usize * 1024));
        }
        "insert" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=insert ===");
            let probes = load_probe_data(&probes_path);

            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolled::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let mut rng = SplitMix64::new(0xC0FFEE ^ scenario_seed(&scenario));
            let t0 = std::time::Instant::now();
            let mut new_count = 0u64;
            for _ in 0..NUM_INSERTS {
                let (s, p, o) = new_triple(&mut rng, probes.max_s, probes.max_p, probes.max_o);
                if tree.insert(s, p, o) {
                    new_count += 1;
                }
            }
            let insert_avg = t0.elapsed() / NUM_INSERTS as u32;

            println!("  deserialize={deserialize_time:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}", tree.block_count());
            report_rss("insert");
        }
        other => panic!("unknown phase {other:?}; expected \"build\", \"query\", or \"insert\""),
    }
}

fn scenario_seed(scenario: &str) -> u64 {
    scenario.bytes().fold(0xF00D_u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}
