//! Per-phase peak-RSS measurement for `RadixTreeDfsInterleavedVarSkipUnrolledFile`
//! (real file, hand-rolled `pread`/`pwrite`). See `phase_heap.rs`'s doc
//! comment for why build/query/insert each run in their own process and
//! how the serialized-structure + probe-sidecar handoff between them
//! works.
//!
//! Usage:
//! ```sh
//! cargo run --release --example phase_file build large_scale_pred_skew_only
//! cargo run --release --example phase_file query large_scale_pred_skew_only
//! cargo run --release --example phase_file insert large_scale_pred_skew_only
//! ```

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, load_probe_data, measure_contains_and_row, measure_predicates_for, new_triple, phase_paths, report_rss, save_probe_data, ProbeData, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
use common::rdf_gen::SplitMix64;

const STRUCTURE: &str = "file";

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
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let disk = tree.disk_bytes();
            let bytes = tree.serialize();
            std::fs::write(&tree_path, &bytes).unwrap_or_else(|e| panic!("failed to write {tree_path:?}: {e}"));

            println!("  build={build_time:>10.2?} mem_size_bytes(RAM)={:>10} disk={:>10} serialized={:>10} blocks={}", human_bytes(mem), human_bytes(disk as usize), human_bytes(bytes.len()), tree.block_count());
            report_rss("build");
        }
        "query" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=query ===");
            let probes = load_probe_data(&probes_path);

            // RSS breakdown bookkeeping -- see the module doc comment's
            // "RSS breakdown" section. `process_baseline_kb` is sampled
            // before the tree exists at all, so it captures everything
            // NOT attributable to this structure (Rust runtime, binary
            // pages, allocator's initial arena, thread stack).
            let process_baseline_kb = common::metrics::current_rss_kb().unwrap_or(0);

            let t0 = std::time::Instant::now();
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            // `block_stack` is empty right after deserialize (no query
            // has touched it yet), so this is exactly the "struct base"
            // component -- everything in `size_breakdown()` except the
            // cache.
            let struct_base_bytes: usize = tree.size_breakdown().iter().map(|&(_, b)| b).sum();

            // Tracks the single largest row-query result `Vec<u32>`'s
            // *capacity* (not length -- capacity is what's actually
            // resident) seen across the whole workload, as an estimate
            // of the "result" component of peak RSS: each row result is
            // freed right after its `.len()` is taken, so it never
            // shows up in any steady-state reading, only (transiently)
            // in the peak.
            let max_row_result_bytes = std::cell::Cell::new(0usize);
            let (contains_avg, row_avg) = measure_contains_and_row(
                |s, p, o| tree.contains(s, p, o),
                |s, p| {
                    let v = tree.query_two_bound([0, 1], [s, p], 2);
                    max_row_result_bytes.set(max_row_result_bytes.get().max(v.capacity() * std::mem::size_of::<u32>()));
                    v.len()
                },
                &probes.contains_probes,
                &probes.row_probes,
            );

            let so_probes: Vec<(u32, u32)> = probes.contains_probes.iter().map(|&(s, _, o)| (s, o)).collect();
            let predicates_for_avg = measure_predicates_for(|s, o| tree.predicates_for(s, o).len(), &so_probes);

            let struct_plus_cache_bytes: usize = tree.size_breakdown().iter().map(|&(_, b)| b).sum();
            let cache_bytes = struct_plus_cache_bytes.saturating_sub(struct_base_bytes);

            println!("  deserialize={deserialize_time:>10.2?} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} predicates_for_avg={predicates_for_avg:>10.2?} blocks={}", tree.block_count());
            report_rss("query");

            let peak_kb = common::metrics::peak_rss_kb().unwrap_or(0);
            let accounted_kb = process_baseline_kb + (struct_base_bytes as u64 / 1024) + (cache_bytes as u64 / 1024) + (max_row_result_bytes.get() as u64 / 1024);
            let transient_kb = peak_kb.saturating_sub(accounted_kb);
            println!(
                "  RSS breakdown: process_baseline={} struct_base={} cache={} max_row_result={} | peak={} accounted={} transient={}",
                human_bytes(process_baseline_kb as usize * 1024),
                human_bytes(struct_base_bytes),
                human_bytes(cache_bytes),
                human_bytes(max_row_result_bytes.get()),
                human_bytes(peak_kb as usize * 1024),
                human_bytes(accounted_kb as usize * 1024),
                human_bytes(transient_kb as usize * 1024),
            );
        }
        "insert" => {
            println!("=== {scenario}, structure={STRUCTURE}, phase=insert ===");
            let probes = load_probe_data(&probes_path);

            let process_baseline_kb = common::metrics::current_rss_kb().unwrap_or(0);

            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize_from_path(&tree_path);
            let deserialize_time = t0.elapsed();

            // Struct and cache are separated by label, not by
            // before/after subtraction like the query phase does --
            // unlike a query, an insert loop genuinely grows the struct
            // itself (new slots/blocks for the inserted records), so a
            // single before/after diff can't tell real struct growth
            // apart from cache growth. See the module doc comment's "RSS
            // breakdown" section.
            let (struct_base_before, _) = struct_and_cache_bytes(&tree);

            let mut rng = SplitMix64::new(0xF17E ^ scenario_seed(&scenario));
            let t0 = std::time::Instant::now();
            let mut new_count = 0u64;
            for _ in 0..NUM_INSERTS {
                let (s, p, o) = new_triple(&mut rng, probes.max_s, probes.max_p, probes.max_o);
                if tree.insert(s, p, o) {
                    new_count += 1;
                }
            }
            let insert_avg = t0.elapsed() / NUM_INSERTS as u32;

            let (struct_base_after, cache_bytes) = struct_and_cache_bytes(&tree);
            let struct_growth_bytes = struct_base_after.saturating_sub(struct_base_before);

            println!("  deserialize={deserialize_time:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}", tree.block_count());
            report_rss("insert");

            let peak_kb = common::metrics::peak_rss_kb().unwrap_or(0);
            let accounted_kb = process_baseline_kb + (struct_base_after as u64 / 1024) + (cache_bytes as u64 / 1024);
            let transient_kb = peak_kb.saturating_sub(accounted_kb);
            println!(
                "  RSS breakdown: process_baseline={} struct_base(final)={} (of which struct_growth={}) cache={} | peak={} accounted={} transient={}",
                human_bytes(process_baseline_kb as usize * 1024),
                human_bytes(struct_base_after),
                human_bytes(struct_growth_bytes),
                human_bytes(cache_bytes),
                human_bytes(peak_kb as usize * 1024),
                human_bytes(accounted_kb as usize * 1024),
                human_bytes(transient_kb as usize * 1024),
            );
        }
        other => panic!("unknown phase {other:?}; expected \"build\", \"query\", or \"insert\""),
    }
}

fn scenario_seed(scenario: &str) -> u64 {
    scenario.bytes().fold(0xF00D_u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

/// Splits [`RadixTreeDfsInterleavedVarSkipUnrolledFile::size_breakdown`]
/// into (struct-only bytes, cache bytes) by label, rather than by
/// before/after subtraction -- see the "insert" phase's own comment for
/// why that matters there.
fn struct_and_cache_bytes(tree: &RadixTreeDfsInterleavedVarSkipUnrolledFile) -> (usize, usize) {
    let breakdown = tree.size_breakdown();
    let cache: usize = breakdown.iter().filter(|(name, _)| name.starts_with("block_stack")).map(|&(_, b)| b).sum();
    let struct_only: usize = breakdown.iter().filter(|(name, _)| !name.starts_with("block_stack")).map(|&(_, b)| b).sum();
    (struct_only, cache)
}
