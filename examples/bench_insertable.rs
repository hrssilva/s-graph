//! Six-way benchmark: `RadixTreeDfsInterleavedVarSkip` (original,
//! rebuild-only) vs. five block-list storage backends that all support
//! real `insert()` -- heap-backed (`RadixTreeDfsInterleavedVarSkipUnrolled`),
//! anonymous-`mmap`-backed (`RadixTreeDfsInterleavedVarSkipUnrolledMmap`),
//! a safe-Rust arena/slab allocator (`RadixTreeDfsInterleavedVarSkipUnrolledArena`),
//! a safe-Rust file-backed variant with hand-rolled `pread`/`pwrite` I/O
//! (`RadixTreeDfsInterleavedVarSkipUnrolledFile`), and a file-backed
//! `MAP_SHARED` `mmap` variant that actively evicts pages via
//! `madvise(MADV_DONTNEED)` as traversal moves past them
//! (`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap`) -- on the same
//! four large-scale scenarios used elsewhere in this project. Not part
//! of the shared 12-structure comparison harness -- this exists purely
//! to answer "does block-list storage actually make insertion cheap,
//! what does it cost on reads/memory to get that, does exact-sizing the
//! arena/slab allocator recover mmap's memory win without needing
//! `unsafe`, what does moving the node records to disk entirely cost in
//! exchange for near-zero RAM, and does combining mmap-speed access with
//! active eviction bound memory without paying the pure file variant's
//! full per-touch syscall cost."
//!
//! Uses `mem_size_bytes()` (allocated-capacity) for its memory column,
//! not peak RSS -- comparing peak RSS across structures built one after
//! another in the *same* process would be meaningless, since the
//! kernel's `VmHWM` high-water mark never decreases even after a
//! structure is dropped, so a later structure's reading would be
//! contaminated by whichever earlier one used the most memory. For an
//! honest peak-RSS comparison (particularly relevant to the file-backed
//! and file-mmap variants, whose whole point is a RAM/latency tradeoff
//! `mem_size_bytes()` alone can't fully capture), see
//! `examples/rss_heap.rs`, `rss_anon_mmap.rs`, `rss_file.rs`, and
//! `rss_file_mmap.rs` -- one isolated process per structure.
//!
//! Run with `cargo run --release --example bench_insertable`.

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::{contains_and_row_probes, domain_bounds, large_scale_scenario, measure_contains_and_row, new_triple, BLOCK_CAPACITY, LARGE_SCALE_SCENARIO_NAMES, NUM_INSERTS};
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip::RadixTreeDfsInterleavedVarSkip;
use common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
use common::radixtree_dfs_interleaved_varskip_unrolled_arena::RadixTreeDfsInterleavedVarSkipUnrolledArena;
use common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
use common::radixtree_dfs_interleaved_varskip_unrolled_file_mmap::RadixTreeDfsInterleavedVarSkipUnrolledFileMmap;
use common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap;
use common::rdf_gen::SplitMix64;

fn main() {
    for name in LARGE_SCALE_SCENARIO_NAMES {
        let triples = large_scale_scenario(name);
        println!("=== {name} ({} triples) ===", triples.len());
        let (contains_probes, row_probes) = contains_and_row_probes(&triples);
        let (max_s, max_p, max_o) = domain_bounds(&triples);

        // --- Baseline: original varskip. Its only "insert" is a full rebuild. ---
        let t0 = std::time::Instant::now();
        let base = RadixTreeDfsInterleavedVarSkip::build_dynamic(&triples);
        let base_build = t0.elapsed();
        let base_mem = base.mem_size_bytes();
        let (base_contains, base_row) = measure_contains_and_row(|s, p, o| base.contains(s, p, o), |s, p| base.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

        let mut rng = SplitMix64::new(0xBA5E ^ triples.len() as u64);
        let (es, ep, eo) = new_triple(&mut rng, max_s, max_p, max_o);
        let mut with_one = triples.clone();
        with_one.push((es, ep, eo));
        let t0 = std::time::Instant::now();
        let _ = RadixTreeDfsInterleavedVarSkip::build_dynamic(&with_one);
        let base_single_insert = t0.elapsed();
        drop(base);

        println!(
            "  varskip (baseline)     build={base_build:>10.2?} mem={:>10} contains_avg={base_contains:>10.2?} row_avg={base_row:>10.2?} single_insert(full rebuild)={base_single_insert:>10.2?}",
            human_bytes(base_mem)
        );

        // --- Block-list variant: real build once, then real insert() ---
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
            let total_insert = t0.elapsed();
            let insert_avg = total_insert / NUM_INSERTS as u32;

            println!(
                "  unrolled(heap) cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }

        // --- Block-list variant, anonymous-mmap-backed blocks ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledMmap::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0x33AA55 ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
            let t0 = std::time::Instant::now();
            let mut new_count = 0u64;
            for _ in 0..NUM_INSERTS {
                let (s, p, o) = new_triple(&mut rng, max_s, max_p, max_o);
                if tree.insert(s, p, o) {
                    new_count += 1;
                }
            }
            let total_insert = t0.elapsed();
            let insert_avg = total_insert / NUM_INSERTS as u32;

            println!(
                "  unrolled(mmap) cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }

        // --- Block-list variant, safe-Rust arena/slab-backed blocks ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0x7EA5ED ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
            let t0 = std::time::Instant::now();
            let mut new_count = 0u64;
            for _ in 0..NUM_INSERTS {
                let (s, p, o) = new_triple(&mut rng, max_s, max_p, max_o);
                if tree.insert(s, p, o) {
                    new_count += 1;
                }
            }
            let total_insert = t0.elapsed();
            let insert_avg = total_insert / NUM_INSERTS as u32;

            println!(
                "  unrolled(arena) cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }

        // --- Block-list variant, safe-Rust file-backed blocks (hand-rolled pread/pwrite, no mmap) ---
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
            let total_insert = t0.elapsed();
            let insert_avg = total_insert / NUM_INSERTS as u32;

            println!(
                "  unrolled(file) cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem(RAM)={:>10} disk={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                human_bytes(disk as usize),
                tree.block_count()
            );
            drop(tree);
        }

        // --- Block-list variant, file-backed MAP_SHARED mmap + madvise(MADV_DONTNEED) eviction ---
        {
            let t0 = std::time::Instant::now();
            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(&triples, BLOCK_CAPACITY);
            let build_time = t0.elapsed();
            let mem = tree.mem_size_bytes();
            let (contains_avg, row_avg) = measure_contains_and_row(|s, p, o| tree.contains(s, p, o), |s, p| tree.query_two_bound([0, 1], [s, p], 2).len(), &contains_probes, &row_probes);

            let mut rng = SplitMix64::new(0xE71C7 ^ (BLOCK_CAPACITY as u64) ^ triples.len() as u64);
            let t0 = std::time::Instant::now();
            let mut new_count = 0u64;
            for _ in 0..NUM_INSERTS {
                let (s, p, o) = new_triple(&mut rng, max_s, max_p, max_o);
                if tree.insert(s, p, o) {
                    new_count += 1;
                }
            }
            let total_insert = t0.elapsed();
            let insert_avg = total_insert / NUM_INSERTS as u32;

            println!(
                "  unrolled(file_mmap) cap={BLOCK_CAPACITY:<5} build={build_time:>10.2?} mem(mapped)={:>10} contains_avg={contains_avg:>10.2?} row_avg={row_avg:>10.2?} insert_avg={insert_avg:>10.2?} ({new_count}/{NUM_INSERTS} new) blocks={}",
                human_bytes(mem),
                tree.block_count()
            );
            drop(tree);
        }
    }
}
