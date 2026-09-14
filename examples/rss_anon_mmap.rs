//! RSS measurement for `RadixTreeDfsInterleavedVarSkipUnrolledMmap`
//! (anonymous-`mmap`-backed blocks, never evicted), run alone in its own
//! process. See `rss_heap.rs`'s doc comment for why this needs its own
//! process rather than sharing one with the other three structures, and
//! for why it reports both `peak RSS (VmHWM)` and `steady-state RSS
//! (VmRSS, post-build)`.
//!
//! Usage: `cargo run --release --example rss_anon_mmap [scenario_name]`
//! (defaults to `large_scale_pred_skew_only`).

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap;
use common::rdf_gen::SplitMix64;

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| LARGE_SCALE_SCENARIO_NAMES[0].to_string());
    let triples = large_scale_scenario(&scenario);
    println!("=== {scenario} ({} triples), structure=anon_mmap, cap={BLOCK_CAPACITY} ===", triples.len());
    let (contains_probes, row_probes) = contains_and_row_probes(&triples);
    let (max_s, max_p, max_o) = domain_bounds(&triples);

    let triples_len = triples.len();
    let t0 = std::time::Instant::now();
    let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledMmap::build_dynamic(&triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    let mem = tree.mem_size_bytes();

    // `triples` (the raw ~240MB input) is not needed past this point --
    // every probe/bound was already computed from it above -- so drop it
    // before reading steady-state RSS, otherwise this reading would
    // still include the input dataset alongside the built structure.
    drop(triples);
    let steady_rss = current_rss_kb();

    let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

    let mut rng = SplitMix64::new(0x33AA55 ^ (BLOCK_CAPACITY as u64) ^ triples_len as u64);
    let t0 = std::time::Instant::now();
    let mut new_count = 0u64;
    for _ in 0..NUM_INSERTS {
        let (s, p, o) = new_triple(&mut rng, max_s, max_p, max_o);
        if tree.insert(s, p, o) {
            new_count += 1;
        }
    }
    let insert_avg = t0.elapsed() / NUM_INSERTS as u32;

    println!(
        "  build={build_time:>10.2?} mem_size_bytes={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
        human_bytes(mem),
        tree.block_count()
    );
    match steady_rss {
        Some(kb) => println!("  steady-state RSS (VmRSS, post-build) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  steady-state RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
    match peak_rss_kb() {
        Some(kb) => println!("  peak RSS (VmHWM, whole run) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  peak RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
}
