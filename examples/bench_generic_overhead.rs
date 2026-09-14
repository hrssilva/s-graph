//! A/B comparison: the hand-specialized `RadixTreeDfsInterleavedVarSkipUnrolled`
//! (heap) / `RadixTreeDfsInterleavedVarSkipUnrolledFile` (file) vs. the
//! generic, `BlockStorage`-monomorphized
//! `radixtree_dfs_interleaved_varskip_unrolled_generic::{Heap, File}` that
//! replaced their copy-pasted logic with one shared implementation.
//!
//! This exists purely to answer the question the generic version was
//! built to let us answer: does the trait-based storage abstraction cost
//! anything once monomorphized, relative to the original,
//! hand-specialized-per-backend code? Same scenarios, same probes, same
//! process (so `mem_size_bytes()` is the right memory comparison here,
//! not peak RSS -- see `bench_insertable.rs`'s own doc comment for why).
//!
//! Usage: `cargo run --release --example bench_generic_overhead`.

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
use common::radixtree_dfs_interleaved_varskip_unrolled_generic::{File as GenericFile, Heap as GenericHeap};
use common::rdf_gen::SplitMix64;

fn main() {
    for name in LARGE_SCALE_SCENARIO_NAMES {
        let triples = large_scale_scenario(name);
        println!("=== {name} ({} triples) ===", triples.len());
        let (contains_probes, row_probes) = contains_and_row_probes(&triples);
        let (max_s, max_p, max_o) = domain_bounds(&triples);

        // --- original heap ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0xC0FFEE ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
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
                "  original(heap)  cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }

        // --- generic heap ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = GenericHeap::build_dynamic(triples.to_vec(), BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0xC0FFEE ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
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
                "  generic(heap)   cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }

        // --- original file ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let disk = tree.disk_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0xF17E ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
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
                "  original(file)  cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem(RAM)={:>10} disk={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                human_bytes(disk as usize),
                tree.block_count()
            );
            drop(tree);
        }

        // --- generic file ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = GenericFile::build_dynamic(triples.to_vec(), BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let disk = tree.disk_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0xF17E ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
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
                "  generic(file)   cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem(RAM)={:>10} disk={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                human_bytes(disk as usize),
                tree.block_count()
            );
            drop(tree);
        }
    }
}
