//! Does today's "bulk" build (materialize the whole triple set, then
//! `build_dynamic`) actually need the RAM it uses, or is a lot of that a
//! one-time construction cost that a *streaming* build could avoid
//! entirely?
//!
//! `build_dynamic` always starts from a fully-materialized `Vec<Triple>`,
//! copies it into an equally large scratch buffer for its level-order
//! counting sort, and assembles the *entire* node-byte-stream into one
//! RAM `Vec<u8>` (`build_interleaved_rec`'s own `combined` buffer) before
//! handing it to the chosen storage backend at all. That last part means
//! even the `File` backend -- whose whole point is near-zero steady-state
//! RAM -- currently pays a full in-RAM peak *during build* identical to
//! `Heap`'s; it only gets its RAM win *after* the structure exists.
//!
//! This experiment asks: what if triples are `insert()`ed one (or a
//! small batch) at a time, as they're generated, into a tree that starts
//! out empty (see
//! `RadixTreeDfsInterleavedVarSkipUnrolledGeneric::build_dynamic_empty`,
//! which only needs the domain's *size*, not any actual data, to set up
//! the correct schedule/height) -- never materializing the full triple
//! set or any O(N) scratch buffer at all?
//!
//! Three modes, each its own process invocation (own peak RSS,
//! uncontaminated by whatever ran before it in the same process -- same
//! rationale as this project's `phase_*.rs`/`rss_*.rs` binaries):
//! - `bulk`: today's approach.
//! - `streaming`: `build_dynamic_empty` + one plain `insert()` per
//!   generated triple, genuinely one at a time -- `batch_size` here only
//!   controls how many triples are buffered in RAM *before* insertion,
//!   not how insertion itself happens; even `batch_size=1` already gets
//!   the O(1)-memory property this mode exists to demonstrate.
//! - `streaming-batch`: `build_dynamic_empty` +
//!   `RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert_batch` per
//!   `batch_size`-sized chunk -- sorts each chunk into the tree's own
//!   DFS-preorder ("Morton code") order first, then inserts it, resetting
//!   the file backend's block-stack cache once per *chunk* instead of
//!   once per *triple*. Still O(batch_size) memory, not O(num_triples) --
//!   this is about spending the same bounded memory budget more
//!   efficiently, not a different memory/time tradeoff class than
//!   `streaming`.
//!
//! All three modes generate triples via the exact same per-triple RNG
//! calls `common::rdf_gen::uniform` itself makes, so for the same seed
//! they build the identical dataset -- `radixtree_dfs_interleaved_varskip_unrolled_generic.rs`'s
//! own test module (`streaming_insert_build_matches_bulk_build_and_naive`,
//! `streaming_batch_insert_matches_per_triple_insert_and_bulk_build`)
//! already cross-checks that sorting into batches changes *only*
//! insertion order and cache-reset frequency -- never the resulting
//! tree's content -- against both per-triple insertion and a bulk build,
//! on small data; this binary is about the RAM/time/packing story at real
//! scale, not correctness.
//!
//! Usage:
//! ```sh
//! cargo run --release --example bench_streaming_insert_build heap bulk [num_triples] [num_s] [num_p] [num_o] [seed]
//! cargo run --release --example bench_streaming_insert_build heap streaming [num_triples] [num_s] [num_p] [num_o] [seed] [batch_size]
//! cargo run --release --example bench_streaming_insert_build heap streaming-batch [num_triples] [num_s] [num_p] [num_o] [seed] [batch_size]
//! cargo run --release --example bench_streaming_insert_build file bulk ...
//! cargo run --release --example bench_streaming_insert_build file streaming ...
//! cargo run --release --example bench_streaming_insert_build file streaming-batch ...
//! ```
//! Defaults: 2,000,000 triples over a 200,000 x 19 x 200,000 domain, seed
//! 2026 -- scaled down from this project's usual 20,000,000-triple
//! large-scale scenarios so a single run finishes in a reasonable time
//! (`file`/`streaming` in particular pays one syscall per field touch,
//! per insert). `batch_size` defaults to 1 for `streaming` (matching its
//! own point -- even the smallest possible chunk already bounds memory)
//! and 5,000 for `streaming-batch` (needs a real chunk to sort/amortize
//! over). Pass explicit arguments for a full-scale run.
//!
//! # Measured at the default scale (not assumed)
//!
//! `file`/`streaming` is the headline result: peak RSS **3.28 MiB**,
//! identical to its own steady-state RSS -- i.e. genuinely flat memory
//! for the entire build, not just after it -- against `file`/`bulk`'s
//! **162.83 MiB** peak (its `mem(RAM)` after build is a tiny 23.92 KiB,
//! same as streaming's 645.20 KiB, but *during* build it transiently
//! holds the whole node-byte-stream in RAM regardless of the target
//! backend, exactly as this file's own top comment predicts). `heap`
//! shows the same shape, smaller in absolute terms since its steady
//! state was never small to begin with: streaming's peak (82.46 MiB) is
//! ~2.6x below bulk's (211.98 MiB).
//!
//! This is not a free win, though -- two real costs, both measured:
//! - **Wall time**: `file`/`streaming` took ~99.5s to insert 2,000,000
//!   triples one at a time (~50µs/insert, all real `pread`/`pwrite`
//!   syscalls) against `file`/`bulk`'s ~1s total (generate + build).
//!   `heap`/`streaming` took ~9.2s (~4.6µs/insert) against `heap`/`bulk`'s
//!   ~1s. Streaming trades peak RAM for build latency, it doesn't get
//!   both for free.
//! - **Final structure size**: streaming builds end up with *more*
//!   blocks than bulk builds of the identical dataset (heap: 1515 vs.
//!   763; file: 3045 vs. 763) and a correspondingly larger footprint
//!   (heap `mem_size`: 111.35 MiB vs. 53.55 MiB; file `disk`: 190.31 MiB
//!   vs. 47.69 MiB) -- bulk's `chop_into_blocks` packs blocks tightly in
//!   one pass over the known-in-advance final byte stream, while
//!   streaming's block-at-a-time growth via repeated single-record
//!   `insert_bytes` calls leaves more partially-filled blocks behind.
//!   `node_words` is identical either way (both builds contain the exact
//!   same triples) -- this is purely a *packing* cost, not a correctness
//!   or content difference.
//!
//! # Does sorting each chunk (`streaming-batch`) help? Backend-dependent
//! -- measured, and the two backends disagree
//!
//! `file`/`streaming-batch` (batch_size 5,000) is a clean, if modest,
//! win over `file`/`streaming` on every axis: ~91.5s vs. ~99.5s wall time
//! (~8% faster), 3,000 vs. 3,045 blocks, 187.50 MiB vs. 190.31 MiB disk.
//! Both effects this file's own `insert_batch` doc comment predicts are
//! real here: sorted-adjacent triples share more of
//! [`Self::ensure_stack_entry`]'s cache, and land in the same
//! already-open block more often than randomly-ordered ones do.
//!
//! `heap`/`streaming-batch` *first exposed a real regression*: before
//! `HeapStorage::insert_bytes` was fixed (see below), a tail-position
//! insert (`pos.offset == block.len()`) was let through *regardless* of
//! `block_capacity` -- a property of the *original* heap variant this
//! generic backend started as a faithful port of
//! (`radixtree_dfs_interleaved_varskip_unrolled.rs`'s own `insert_bytes`
//! has the byte-for-byte identical condition), never a bug in
//! `insert_batch` itself. Random single-triple insertion (the only
//! pattern any benchmark exercised before this one) rarely lands the
//! same block's tail twice in a row, so the bypass was rarely triggered
//! in practice. Morton-sorted insertion order does close to the
//! *opposite* -- consecutive triples are exactly the ones most likely to
//! all extend the *same* block's tail, back to back -- so that one
//! `Vec<u8>` kept growing past its intended capacity via plain `Vec`
//! amortized-doubling instead of ever splitting, for many blocks across
//! the structure at once: `mem_size` measured at **476.21 MiB** across
//! 3,905 blocks, worse than `bulk` (53.55 MiB) *and* worse than plain
//! `streaming` (111.35 MiB), despite being the *fastest* of the three
//! (~6.7s) -- fewer splits, more (unbounded) appends.
//!
//! **Fixed and re-validated** (not merely proposed): `HeapStorage::insert_bytes`
//! now special-cases a too-big tail insert into a *fresh* block instead
//! of widening the current one past `block_capacity` -- matching
//! `FileStorage::insert_bytes`'s own `is_tail` branch, which never had
//! this bypass. Re-measured after the fix, same 2,000,000-triple run:
//! `heap`/`streaming-batch` is now **113.70 MiB across 1,534 blocks**
//! (was 476.21 MiB / 3,905) -- on par with plain `streaming`'s 111.35
//! MiB / 1,515 blocks, the ~1-2% difference well within the packing
//! noise sort order alone would produce -- and it got *faster* in the
//! process: **4.27s**, down from streaming-batch's own pre-fix 6.7s and
//! plain streaming's 9.2s (repeatedly widening one already-huge `Vec` via
//! `splice`/`extend_from_slice` costs more, not less, than keeping every
//! block near its intended `block_capacity` and splitting -- the
//! unbounded-growth path wasn't even the fast path it looked like).
//! `bench_generic_overhead.rs` was re-run in full across all four
//! 20,000,000-triple scenarios after this change: `heap`'s `contains`/row
//! latency, `mem_size`, and build time all stayed within the same noise
//! band against the *original* heap variant as before the fix (that
//! benchmark's own `NUM_INSERTS=1000` are scattered/random, not sorted,
//! so the bypass it used to rely on was never doing useful work there in
//! the first place) -- the fix cost nothing on the workload the "zero
//! overhead" claim was originally validated against, while fixing a real
//! problem on the one this file's own `streaming-batch` mode newly
//! exercises. Confirmed with `#[test]`s too, not just this binary's own
//! numbers -- see `streaming_batch_insert_matches_per_triple_insert_and_bulk_build`
//! and `streaming_insert_build_matches_bulk_build_and_naive`, both still
//! green after the change.
//!
//! # A second, bigger fix: stop growing blocks organically at all
//!
//! Fixing the tail-append bypass closed the *pathological* gap, but a
//! smaller, structural one remained even for plain one-at-a-time
//! `streaming`: `heap`/`streaming`'s `mem_size` (94.75 MiB before a
//! shrink pass, at 2,000,000 triples) was still ~2x `bulk`'s (47.69
//! MiB), because every block, streaming or bulk, was being grown via
//! repeated `Vec::splice`/`extend_from_slice` calls starting from an
//! empty `Vec::new()` -- and `Vec`'s own amortized-doubling growth means
//! its *capacity* can sit up to ~2x its actual *content* at any given
//! moment, for every block, all the time. This is exactly the ~19%-ish
//! overhead this project's own mmap/arena-variant doc comments already
//! named (in that case, comparing against mmap's fixed-size mappings,
//! which don't have this problem at all).
//!
//! The fix (in `HeapStorage::from_chopped`/`insert_bytes`/the new
//! `BlockStorage::shrink_to_fit`): allocate each block at its *full*
//! `block_capacity` budget up front (`Vec::with_capacity`), fill it
//! (whether via one bulk pass or many separate inserts) without ever
//! reallocating along the way, hand it off once the next record wouldn't
//! fit, and reclaim whatever of that reservation went unused with one
//! explicit `shrink_to_fit()` pass once the caller is done inserting for
//! now -- never automatically after every insert or every batch (that
//! would immediately undo the whole point: the next nearby insert would
//! just force a reallocation right back). A bulk `build_dynamic` already
//! has a natural "done" point, so it calls this on itself automatically;
//! `insert`/`insert_batch` do not, and callers doing a streaming build
//! call `Self::shrink_to_fit` explicitly once, at the very end -- exactly
//! what this binary does.
//!
//! **Re-measured a final time, same 2,000,000-triple run, with both
//! fixes in place:**
//!
//! | mode | mem_size (before shrink) | mem_size (after shrink) | wall time |
//! |---|---|---|---|
//! | `bulk` | -- | **47.69 MiB** | ~933ms (generate + build) |
//! | `streaming` | 94.75 MiB | **47.71 MiB** | 9.17s insert + 296µs shrink |
//! | `streaming-batch` | 95.94 MiB | **47.71 MiB** | 4.22s insert + 313µs shrink |
//!
//! `streaming`/`streaming-batch`'s final size is now indistinguishable
//! from `bulk`'s (47.71 MiB vs. 47.69 MiB -- the 0.02 MiB left over is
//! `bulk` itself getting *tighter* too, not streaming falling short: its
//! own blocks are now pre-reserved-then-shrunk the same way).
//! `shrink_to_fit` itself is essentially free (a few hundred
//! microseconds for the whole 2,000,000-triple structure) precisely
//! *because* pre-reservation meant there was only ever one allocation
//! per block to begin with, not a chain of intermediate ones to unwind.
//! Streaming still keeps the property that motivated this whole
//! experiment -- peak RSS during the build never spikes the way `bulk`'s
//! does (`bulk`'s own peak RSS is 213.01 MiB against `streaming`'s
//! 77.15 MiB steady/peak) -- and now *also* ends up exactly as compact
//! as a bulk build, with `streaming-batch` keeping its ~2x wall-time
//! edge over plain `streaming` on top. `file` was unaffected by either
//! fix in any meaningful way (its RAM cost was never in per-block `Vec`
//! capacity slack to begin with -- the small `shrink_to_fit` added there
//! trims only the tiny bookkeeping vectors, 645 KiB -> 342 KiB, noise
//! next to its ~190 MiB disk footprint).

#[path = "common/mod.rs"]
mod common;

use common::insertable_bench::report_rss;
use common::metrics::{human_bytes, MemSize};
use common::radixtree_dfs_interleaved_varskip_unrolled_generic::{File as GenericFile, Heap as GenericHeap};
use common::rdf_gen::{uniform, SplitMix64, Triple};

const BLOCK_CAPACITY: usize = 65536;

fn uniform_triple(rng: &mut SplitMix64, num_s: u32, num_p: u32, num_o: u32) -> Triple {
    (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32, rng.next_range(num_o as u64) as u32)
}

fn run_bulk_heap(num_triples: usize, num_s: u32, num_p: u32, num_o: u32, seed: u64) {
    let t0 = std::time::Instant::now();
    let triples = uniform(num_s, num_p, num_o, num_triples, seed);
    println!("  generated {} triples in {:?}", triples.len(), t0.elapsed());
    report_rss("after dataset generation");

    let t0 = std::time::Instant::now();
    // `build_dynamic` takes `triples` by value now (see its own doc
    // comment), moving straight into the counting sort's working array
    // instead of cloning it -- so `triples` is already gone by the time
    // this returns, no separate `drop` needed.
    let tree = GenericHeap::build_dynamic(triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    println!("  build_dynamic took {build_time:?}, mem_size={}, node_words={}, blocks={}", human_bytes(tree.mem_size_bytes()), tree.node_word_count(), tree.block_count());
    report_rss("after bulk build (dataset dropped)");
}

fn run_streaming_heap(num_triples: usize, num_s: u32, num_p: u32, num_o: u32, seed: u64, batch_size: usize, chunked: bool) {
    let dimension_sizes = [num_s as u64, num_p as u64, num_o as u64];
    let mut tree = GenericHeap::build_dynamic_empty(dimension_sizes, BLOCK_CAPACITY);
    report_rss("after empty-tree construction");

    let mut rng = SplitMix64::new(seed);
    let mut batch: Vec<Triple> = Vec::with_capacity(batch_size);
    let mut inserted = 0u64;
    let t0 = std::time::Instant::now();
    for i in 0..num_triples {
        batch.push(uniform_triple(&mut rng, num_s, num_p, num_o));
        if batch.len() == batch_size || i + 1 == num_triples {
            if chunked {
                inserted += tree.insert_batch(&mut batch);
            } else {
                for &(s, p, o) in &batch {
                    if tree.insert(s, p, o) {
                        inserted += 1;
                    }
                }
            }
            batch.clear();
        }
    }
    let insert_time = t0.elapsed();
    let label = if chunked { "streaming-batch insert" } else { "streaming insert" };
    let mem_before_shrink = tree.mem_size_bytes();

    let t0 = std::time::Instant::now();
    tree.shrink_to_fit();
    let shrink_time = t0.elapsed();

    println!(
        "  {label} of {num_triples} triples took {insert_time:?} ({inserted} new), mem_size before shrink={}, after shrink={} (shrink took {shrink_time:?}), node_words={}, blocks={}",
        human_bytes(mem_before_shrink),
        human_bytes(tree.mem_size_bytes()),
        tree.node_word_count(),
        tree.block_count()
    );
    report_rss(&format!("after {label} build + shrink_to_fit"));
}

fn run_bulk_file(num_triples: usize, num_s: u32, num_p: u32, num_o: u32, seed: u64) {
    let t0 = std::time::Instant::now();
    let triples = uniform(num_s, num_p, num_o, num_triples, seed);
    println!("  generated {} triples in {:?}", triples.len(), t0.elapsed());
    report_rss("after dataset generation");

    let t0 = std::time::Instant::now();
    // See `run_bulk_heap`'s own comment: `triples` moves straight in, no
    // separate `drop` needed afterward.
    let tree = GenericFile::build_dynamic(triples, BLOCK_CAPACITY);
    let build_time = t0.elapsed();
    println!(
        "  build_dynamic took {build_time:?}, mem(RAM)={}, disk={}, node_words={}, blocks={}",
        human_bytes(tree.mem_size_bytes()),
        human_bytes(tree.disk_bytes() as usize),
        tree.node_word_count(),
        tree.block_count()
    );
    report_rss("after bulk build (dataset dropped)");
}

fn run_streaming_file(num_triples: usize, num_s: u32, num_p: u32, num_o: u32, seed: u64, batch_size: usize, chunked: bool) {
    let dimension_sizes = [num_s as u64, num_p as u64, num_o as u64];
    let mut tree = GenericFile::build_dynamic_empty(dimension_sizes, BLOCK_CAPACITY);
    report_rss("after empty-tree construction");

    let mut rng = SplitMix64::new(seed);
    let mut batch: Vec<Triple> = Vec::with_capacity(batch_size);
    let mut inserted = 0u64;
    let t0 = std::time::Instant::now();
    for i in 0..num_triples {
        batch.push(uniform_triple(&mut rng, num_s, num_p, num_o));
        if batch.len() == batch_size || i + 1 == num_triples {
            if chunked {
                inserted += tree.insert_batch(&mut batch);
            } else {
                for &(s, p, o) in &batch {
                    if tree.insert(s, p, o) {
                        inserted += 1;
                    }
                }
            }
            batch.clear();
        }
    }
    let insert_time = t0.elapsed();
    let label = if chunked { "streaming-batch insert" } else { "streaming insert" };
    let mem_before_shrink = tree.mem_size_bytes();

    let t0 = std::time::Instant::now();
    tree.shrink_to_fit();
    let shrink_time = t0.elapsed();

    println!(
        "  {label} of {num_triples} triples took {insert_time:?} ({inserted} new), mem(RAM) before shrink={}, after shrink={} (shrink took {shrink_time:?}), disk={}, node_words={}, blocks={}",
        human_bytes(mem_before_shrink),
        human_bytes(tree.mem_size_bytes()),
        human_bytes(tree.disk_bytes() as usize),
        tree.node_word_count(),
        tree.block_count()
    );
    report_rss(&format!("after {label} build + shrink_to_fit"));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: bench_streaming_insert_build <heap|file> <bulk|streaming|streaming-batch> [num_triples] [num_s] [num_p] [num_o] [seed] [batch_size]");
        std::process::exit(1);
    }
    let structure = args[1].as_str();
    let mode = args[2].as_str();
    let num_triples: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2_000_000);
    let num_s: u32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    let num_p: u32 = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(19);
    let num_o: u32 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    let seed: u64 = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(2026);
    // `streaming` (per-triple, no sort) defaults to batch_size 1 --
    // matching its own point, that even the *smallest possible* chunk
    // already bounds memory at O(1). `streaming-batch` (sorted,
    // amortized cache reset) needs a real batch to have anything to sort
    // or amortize over, so it defaults far higher.
    let default_batch_size = if mode == "streaming-batch" { 5_000 } else { 1 };
    let batch_size: usize = args.get(8).map(|s| s.parse().unwrap()).unwrap_or(default_batch_size).max(1);

    println!("=== structure={structure} mode={mode} num_triples={num_triples} domain=({num_s}x{num_p}x{num_o}) seed={seed} batch_size={batch_size} ===");

    match (structure, mode) {
        ("heap", "bulk") => run_bulk_heap(num_triples, num_s, num_p, num_o, seed),
        ("heap", "streaming") => run_streaming_heap(num_triples, num_s, num_p, num_o, seed, batch_size, false),
        ("heap", "streaming-batch") => run_streaming_heap(num_triples, num_s, num_p, num_o, seed, batch_size, true),
        ("file", "bulk") => run_bulk_file(num_triples, num_s, num_p, num_o, seed),
        ("file", "streaming") => run_streaming_file(num_triples, num_s, num_p, num_o, seed, batch_size, false),
        ("file", "streaming-batch") => run_streaming_file(num_triples, num_s, num_p, num_o, seed, batch_size, true),
        _ => {
            eprintln!("structure must be heap|file, mode must be bulk|streaming|streaming-batch");
            std::process::exit(1);
        }
    }
}
