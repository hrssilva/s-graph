//! RSS measurement for the generic, `BlockStorage`-monomorphized `File`
//! backend (`radixtree_dfs_interleaved_varskip_unrolled_generic::File`),
//! run alone in its own process. See `rss_file.rs`'s own doc comment for
//! the full rationale (own-process isolation, peak vs. steady-state RSS).
//!
//! This is specifically the binary that validates the streaming bulk-build
//! path added in `radixtree_dfs_interleaved_varskip_unrolled_generic.rs`
//! (see `build_rec`'s own doc comment): unlike the pre-streaming design,
//! which materialized the *entire* encoded tree as one `Vec<u8>` (plus a
//! `record_starts: Vec<usize>` index of it) before ever touching storage,
//! node records now stream straight into `FileStorage`'s block-chopped
//! slots as they're produced -- so `peak RSS` here should be dominated by
//! the *input* triples/scratch buffers (common to every variant, and
//! freed once sorted/partitioned) and by whatever transient `FlatBuf`s the
//! parallel fan-out path allocates for subtrees currently in flight, not
//! by a second full copy of the whole encoded tree.
//!
//! Usage: `cargo run --release --example rss_file_generic [scenario_name]`
//! (defaults to `large_scale_pred_skew_only`).

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{current_rss_kb, human_bytes, peak_rss_kb, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_generic::File as GenericFile;
use common::rdf_gen::SplitMix64;

fn main() {
    let scenario = std::env::args().nth(1).unwrap_or_else(|| LARGE_SCALE_SCENARIO_NAMES[0].to_string());
    let triples = large_scale_scenario(&scenario);
    println!("=== {scenario} ({} triples), structure=generic(file), cap={BLOCK_CAPACITY} ===", triples.len());
    let (contains_probes, row_probes) = contains_and_row_probes(&triples);
    let (max_s, max_p, max_o) = domain_bounds(&triples);

    let triples_len = triples.len();
    let t0 = std::time::Instant::now();
    // `build_dynamic` takes `triples` by value (see its own doc comment):
    // this moves the harness's own copy straight into the counting sort's
    // working array, rather than lending a reference and paying for an
    // internal clone -- so unlike `rss_file.rs`'s original variant, there
    // is no separate `drop(triples)` needed afterward here, since the
    // input is already gone by the time `build_dynamic` returns.
    let mut tree = GenericFile::build_dynamic(triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    let build_peak_rss = peak_rss_kb();
    let mem = tree.mem_size_bytes();
    let disk = tree.disk_bytes();
    let steady_rss = current_rss_kb();

    let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

    let mut rng = SplitMix64::new(0xF17E ^ (BLOCK_CAPACITY as u64) ^ triples_len as u64);
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
        "  build={build_time:>10.2?} mem_size_bytes(RAM)={:>10} disk={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
        human_bytes(mem),
        human_bytes(disk as usize),
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
