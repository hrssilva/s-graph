//! Per-phase peak-RSS measurement for
//! `RadixTreeDfsInterleavedVarSkipUnrolledMmap` (anonymous-`mmap`-backed
//! blocks). See `phase_heap.rs`'s doc comment for why build/query/insert
//! each run in their own process and how the serialized-structure +
//! probe-sidecar handoff between them works.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_anon_mmap build large_scale_pred_skew_only
//! cargo run --release --example phase_anon_mmap query large_scale_pred_skew_only
//! cargo run --release --example phase_anon_mmap insert large_scale_pred_skew_only
//! ```

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, load_probe_data, measure_contains_and_row, new_triple, phase_paths, report_rss, save_probe_data, ProbeData, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap;
use common::rdf_gen::SplitMix64;

const STRUCTURE: &str = "anon_mmap";

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
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledMmap::build_dynamic(&triples, BLOCK_CAPACITY);
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

            let t0 = std::time::Instant::now();
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledMmap::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &probes.contains_probes, &probes.row_probes);

            println!("  deserialize={deserialize_time:>10.2?} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} blocks={}", tree.block_count());
            report_rss("query");
        }
        "insert" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=insert ===");
            let probes = load_probe_data(&probes_path);

            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledMmap::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            let mut rng = SplitMix64::new(0x33AA55 ^ scenario_seed(&scenario));
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
