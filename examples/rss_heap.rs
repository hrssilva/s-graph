//! RSS measurement for `RadixTreeDfsInterleavedVarSkipUnrolled`
//! (heap-backed blocks), run alone in its own process.
//!
//! Why a separate binary per structure instead of one shared loop: the
//! kernel's peak-RSS counter (`VmHWM` in `/proc/self/status`, read via
//! `common::metrics::peak_rss_kb`) is a *whole-process* high-water mark
//! that never decreases, even after the structure that caused it is
//! dropped. Building four different structures one after another in the
//! same process (as `examples/bench_insertable.rs` does for its
//! `mem_size_bytes()`-based comparison) would make every reading after
//! the first meaningless -- contaminated by whichever earlier structure
//! used the most memory. One process per structure is the only way to
//! get four honest, independent peak-RSS numbers.
//!
//! Reports two RSS numbers, not one: `peak RSS (VmHWM)` (the historical
//! high-water mark, which captures build-time transients like the raw
//! input dataset and counting-sort scratch buffers -- often much larger
//! than the structure itself, especially for a small final structure)
//! and `steady-state RSS (VmRSS)`, read immediately after build with the
//! raw input dataset (`triples`) explicitly dropped -- this is what a
//! long-running process would actually see resident once those
//! transient buffers are freed and their pages returned to the OS.
//!
//! Usage: `cargo run --release --example rss_heap [scenario_name]`
//! (defaults to `large_scale_pred_skew_only`; pass any name from
//! `common::insertable_bench::LARGE_SCALE_SCENARIO_NAMES` to pick
//! another). Run once per scenario for a full per-scenario matrix, e.g.:
//! ```sh
//! for s in large_scale_pred_skew_only large_scale_entity_skew_1.0 large_scale_entity_skew_1.5 large_scale_entity_skew_2.0; do
//!     cargo run --release --example rss_heap "$s"
//! done
//! ```

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::rdf_gen::SplitMix64;

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| LARGE_SCALE_SCENARIO_NAMES[0].to_string());
    let triples = large_scale_scenario(&scenario);
    println!("=== {scenario} ({} triples), structure=heap, cap={BLOCK_CAPACITY} ===", triples.len());
    let (contains_probes, row_probes) = contains_and_row_probes(&triples);
    let (max_s, max_p, max_o) = domain_bounds(&triples);

    let triples_len = triples.len();
    let t0 = std::time::Instant::now();
    let mut tree = RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(&triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    let mem = tree.mem_size_bytes();

    // `triples` (the raw ~240MB input) is not needed past this point --
    // every probe/bound was already computed from it above -- so drop it
    // before reading steady-state RSS, otherwise this reading would
    // still include the input dataset alongside the built structure.
    drop(triples);
    let steady_rss = current_rss_kb();

    let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

    let mut rng = SplitMix64::new(0xC0FFEE ^ (BLOCK_CAPACITY as u64) ^ triples_len as u64);
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
