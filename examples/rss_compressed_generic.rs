//! RSS measurement for the generic, `BlockStorage`-monomorphized
//! `Compressed` backend (LZ4-compressed in-RAM blocks -- see
//! `CompressedStorage`'s own module comment in
//! `radixtree_dfs_interleaved_varskip_unrolled_generic.rs` for the full
//! design), run alone in its own process. See `rss_heap.rs`'s own doc
//! comment for the own-process/peak-vs-steady-state rationale this
//! shares with every other `rss_*.rs` binary.
//!
//! This is the binary that answers the question `CompressedStorage`
//! exists for: how much RAM does compression actually save relative to
//! `Heap`, and what does it cost in build time and query/insert latency?
//! Compare its output directly against `rss_heap_generic.rs`'s on the
//! same scenario.
//!
//! Usage: `cargo run --release --example rss_compressed_generic [scenario_name]`
//! (defaults to `large_scale_pred_skew_only`).

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_generic::Compressed as GenericCompressed;
use common::rdf_gen::SplitMix64;

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| LARGE_SCALE_SCENARIO_NAMES[0].to_string());
    let triples = large_scale_scenario(&scenario);
    println!("=== {scenario} ({} triples), structure=generic(compressed), cap={BLOCK_CAPACITY} ===", triples.len());
    let (contains_probes, row_probes) = contains_and_row_probes(&triples);
    let (max_s, max_p, max_o) = domain_bounds(&triples);

    let triples_len = triples.len();
    let t0 = std::time::Instant::now();
    let mut tree = GenericCompressed::build_dynamic(triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    let build_peak_rss = peak_rss_kb();
    let mem = tree.mem_size_bytes();
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
    match build_peak_rss {
        Some(kb) => println!("  peak RSS immediately after build (VmHWM) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  peak RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
    match steady_rss {
        Some(kb) => println!("  steady-state RSS (VmRSS, post-build) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  steady-state RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
    match peak_rss_kb() {
        Some(kb) => println!("  peak RSS (VmHWM, whole run) = {} ({kb} KiB)", human_bytes(kb as usize * 1024)),
        None => println!("  peak RSS unavailable (non-Linux or /proc/self/status unreadable)"),
    }
}
