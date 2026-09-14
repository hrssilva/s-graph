# Prompt: productionize the succinct RDF triple index

You are a systems/Rust specialist being asked to take an exploratory,
benchmark-validated data structure out of this repository's `examples/`
prototype area and turn it into the project's real, published `src/`
library. This document is your brief. Read it fully before touching
code — it contains the architecture, the design principles to preserve,
the measured numbers that justify every major decision so far, and the
open questions you are expected to make (and justify) a call on, not
silently resolve.

## 1. Mission

`examples/common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs`
is this project's answer to "what would you actually ship": a
DFS-preorder, interleaved-skip-pointer radix trie over Morton-coded
`(subject, predicate, object)` triples, generic over a storage backend,
with real O(height) `insert`, a streaming/parallel/RAM-bounded bulk
build, and three working storage backends (in-heap, real file, and
LZ4-compressed in-RAM). It has been benchmarked extensively against
alternatives (Mk²-tree, IK²-tree, Omnitree — see `examples/README.md`
for the full survey and literature grounding) and against itself across
several rounds of measured optimization (see §5).

**Your job**: review this structure and its test suite, and produce a
production-quality version of it as this crate's real public API
(`src/`), replacing the current placeholder stubs (`src/radix64.rs`,
`src/ksquare.rs` — read them, but they are unfinished sketches with no
connection to the `examples/` work; decide explicitly whether to delete,
replace, or evolve them, and say which in your own writeup). This is not
a rewrite-from-scratch: the algorithm and the storage abstraction are
already right, measured, and correct. Your job is hardening, API design,
error handling, documentation, and — where you find real, measured
opportunity — further optimization. Do not change behavior you can't
justify with a measurement, and do not silently drop a design decision
documented below without saying why in your own report.

## 2. What to read, in order

1. `examples/README.md` — the full comparison this structure "won," and
   the literature it's grounded in (Brisaboa/Ladra/Navarro k²-trees,
   Karras' parallel BVH/octree construction, Vitter's external-memory
   model, HDTCat, and more — full citations inline).
2. `examples/common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs`
   — the structure itself. ~3,800 lines. Read the module-level doc
   comments on each major section (`RecordSink`, `BlockStorage`,
   `HeapStorage`, `FileStorage`, `CompressedStorage`, the bounded-build
   section) before the code — they carry load-bearing design rationale,
   including two documented instances of "we tried something clever,
   measured it, and reverted it" that you should not re-attempt without
   re-reading why it failed the first time (`StackCacheEntry`'s doc
   comment: three reverted write-back cache designs and the exact bug
   class each hit; `CompressedStorage`'s module comment: a whole-block
   compression design that measured 300-600x slower than intended,
   root-caused and fixed by chunking).
3. Its test suite — the `#[cfg(test)] mod tests` at the bottom of the
   same file, plus `examples/common/reference.rs` (`NaiveIndex`, the
   brute-force correctness oracle every non-trivial query is checked
   against).
4. Supporting benchmark harnesses you should treat as the acceptance
   test for any future performance claim you make:
   `examples/rss_heap_generic.rs`, `examples/rss_file_generic.rs`,
   `examples/rss_compressed_generic.rs`, `examples/rss_file_bounded_generic.rs`,
   `examples/bench_generic_overhead.rs`, `examples/phase_join.rs`.

## 3. Architecture summary

- **Encoding**: each tree node is an 8-byte existence bitmap (up to
  64-way branching per level) plus, for nodes more than 2 levels from a
  leaf ("far" nodes), an inline variable-width (1/2/4/8-byte) skip
  pointer giving the byte distance to jump past that node's entire
  subtree. Navigation is pure `pos + span` pointer arithmetic — zero
  rank/select operations anywhere, which is what lets the DFS-preorder
  byte stream be genuinely block-chunked and paged/compressed without
  losing O(1)-per-hop navigation.
- **Schedule**: a per-level bit-consumption schedule (`static_schedule`/
  `dynamic_schedule`) computed once from each dimension's domain size,
  before any data exists. A dimension stops contributing digits once its
  domain is exhausted, so the tree degenerates gracefully instead of
  padding — this is what avoids the k²-tree family's documented
  traversal pathology on skewed/concentrated data (see
  `examples/README.md` §5 for the measured ~2000x latency gap this
  structure avoids structurally, not by luck).
- **Storage abstraction**: `trait BlockStorage`, ~18 methods, addressed
  by logical block index + byte offset (`struct Pos`). Three impls:
  - `HeapStorage`: `Vec<Vec<u8>>`, no indirection, fastest, full RAM
    footprint.
  - `FileStorage`: real file, `pread`/`pwrite`, a slot table (logical
    block → physical slot, needed because a file's newly-allocated space
    is physically append-only), and a small write-through, push-on-
    descend/pop-on-backtrack stack cache of page-chunked windows
    (`READ_CHUNK_BYTES = 4096`). Near-zero steady-state RAM.
  - `CompressedStorage`: blocks live in RAM but LZ4-compressed
    (`lz4_flex`, safe-encode/safe-decode — zero unsafe code, chosen
    specifically for decompression speed over ratio; see its own
    `Cargo.toml` comment). Same write-through stack-cache discipline as
    `FileStorage`, but windowed at **compression** granularity
    (`COMPRESS_CHUNK_BYTES`, matching `READ_CHUNK_BYTES`) rather than
    I/O granularity — each block is many independently-compressed 4 KiB
    chunks, not one blob, so a single-field touch decompresses ~4 KiB
    instead of up to the full `block_capacity` (this was a real, measured
    9.4x fix this session — see §5 and §6).
  - All three are monomorphized via generics (`RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S: BlockStorage>`), not boxed/dyn-dispatched — validated as zero-overhead via
    `bench_generic_overhead.rs`'s direct A/B methodology against
    hand-specialized originals.
- **Bulk build**: `build_rec` (streaming, `RecordSink`-generic — see
  its own doc comment) does an O(N·height) counting-sort/radix-partition
  recursion, streaming node records directly into final block-chopped
  storage (no full-tree byte buffer ever materialized), with a
  size-gated `rayon` parallel fan-out across independent subtrees. A
  second path, `build_external`/`BulkBuildBudget` (see the "bounded-
  memory bulk build" section), spills over-budget subtrees to on-disk
  buckets so peak build RAM is `O(budget)`, not `O(N)` — the standard
  external-memory distribution-sort pattern (Vitter's model; the same
  "build small, budget-bounded pieces" idea HDTCat uses for RDF at
  scale), needed because the in-RAM path alone costs ~56 bytes of peak
  RAM per input triple regardless of backend, i.e. tens of GB at
  billion-triple scale.
- **Insertion**: real O(height) `insert`/`insert_batch` (Morton-sorted
  batching for better cache/block locality), unlike every other
  structure surveyed in `examples/README.md`, whose only option is a
  full rebuild.
- **Join/BFS integration**: one `GraphBackend` impl, generic over
  `BlockStorage`, wires all three backends into the shared leapfrog-join
  (`examples/common/join.rs`) and path/BFS (`examples/common/path.rs`)
  drivers — see `examples/common/sorted_neighbors.rs`.
- **Serialization**: a shared, backend-opaque wire format
  (`examples/common/tree_serialize.rs`) that round-trips through
  **uncompressed** bytes regardless of backend, so a tree built via any
  one backend deserializes correctly into any other (verified directly
  by `serialize_cross_backend_round_trip` in the test suite).

## 4. Design principles to preserve (Data-Oriented Design and beyond)

These are not abstract guidelines — every one below is either already
embodied in the code (with a specific location to look at) or was a
concrete, measured fix this session. Preserve the pattern; where you
extend the structure, apply the same discipline and validate it the same
way (a dedicated before/after benchmark, not intuition).

1. **Structure-of-arrays / flat byte buffers, never a pointer-chasing
   node graph.** The entire tree is `Vec<u8>` block streams plus integer
   offsets. Do not introduce a boxed/`Rc`'d node type for "clarity" —
   this is the property that makes the whole storage abstraction
   (swap RAM for disk for compression) possible at all.
2. **Pre-reserve, never let a hot-path `Vec` grow by amortized
   doubling.** `HeapStorage::append_record` and `FileStorage`/
   `CompressedStorage`'s `build_pending` buffers are always
   `Vec::with_capacity`'d to their final budget up front. This was a
   **measured** real bug this project's own history records: an earlier
   version's unconditional tail-append let a hot block balloon via
   ordinary `Vec` doubling, inflating `mem_size_bytes` **9x** (53.55 MiB
   → 476.21 MiB at 2,000,000 triples) once sorted-batch insertion made
   hitting the same block's tail the common case instead of a rare one.
   See `HeapStorage::insert_bytes`'s own doc comment for the full story.
   Treat this as a live warning, not history: any new hot-path buffer
   you add needs the same discipline.
3. **Batch I/O and batch (de)compression; never one syscall/codec-call
   per record.** `FileStorage`/`CompressedStorage`'s `build_pending`
   accumulates a whole block before one `pwrite`/compress call.
   `CompressedStorage`'s windowed cache decompresses one ~4 KiB chunk
   per miss, not one call per 8-byte field.
4. **Touch only the cache lines/pages you actually need — this was the
   single biggest measured win this session.** The first version of
   `CompressedStorage` decompressed a *whole block* (up to 64 KiB) to
   read one 8-byte skip pointer, because `advance_past_children`'s
   existing, backend-agnostic sibling-skip-ahead loop touches every
   earlier active sibling once per level to jump past it — every backend
   already pays this O(active earlier siblings) cost, but only
   `Compressed` felt it acutely (a decompress vs. `Heap`'s ~100ns
   cache-cold memory read or `File`'s ~1µs page-cache-warm `pread`).
   Windowing the compression to `COMPRESS_CHUNK_BYTES` (mirroring
   `FileStorage`'s own `READ_CHUNK_BYTES` `pread` windowing) cut
   `contains()` latency **9.4x**, row queries **4.8x**, inserts **6.7x**
   (§5/§6). Read `CompressedStorage`'s module doc comment in full before
   touching its cache logic — it documents this exact investigation.
5. **Monomorphize, don't dynamically dispatch, for a compile-time-known
   choice.** `S: BlockStorage` is a type parameter, not `Box<dyn
   BlockStorage>`. The one deliberate exception in the whole codebase is
   `sorted_neighbors::Stream`'s `Box<dyn SortedNeighbors>` — justified
   explicitly (wrapping genuinely different concrete iterator types
   behind one interface, judged negligible next to the tree traversal it
   wraps) rather than reached for by default. Follow the same standard:
   dynamic dispatch is allowed where it's the only reasonable way to
   unify genuinely heterogeneous types, not as a default.
6. **Bit-level iteration over set bits, not boxed iterators.**
   `trailing_zeros()`/`x & (x - 1)`/`count_ones()` throughout the query
   and build paths. Preserve this; it's already idiomatic and fast.
7. **Data-parallelism over disjoint slices, not shared mutable state.**
   The bulk build's `rayon` fan-out splits `points`/`scratch` into
   disjoint sub-slices via `split_at_mut` before handing them to
   independent tasks — never a lock, never shared mutable access to one
   `BlockStorage`. If you parallelize anything else, keep this shape.
8. **Every optimization claim in this codebase is backed by a specific,
   reproducible benchmark, not intuition** — see §5 for the full
   measured history, including two cases where a plausible-sounding
   design was implemented, measured, and found to make things *worse*
   (the pre-chunking `CompressedStorage`; three separate reverted
   `FileStorage` write-back cache attempts, each with a real correctness
   bug, documented in `StackCacheEntry`'s doc comment). Before landing
   any new optimization, add or extend a benchmark in the style of
   `rss_*_generic.rs`/`bench_generic_overhead.rs` and report a real
   before/after number in your PR description, not an estimate.
9. **Never optimize for one dataset shape at the expense of another.**
   Every benchmark in this codebase is run across multiple data shapes
   (uniform, Zipfian-predicate-skew, Zipfian-entity-skew at several
   exponents, plus the structured `graph_patterns` star/triangle/4-cycle
   dataset) specifically so a shape-specific win can't hide a
   shape-specific regression. Keep this practice.

**Open DoD/optimization opportunities not yet explored** — real
candidates for you to investigate and measure, not requirements:
- SIMD batch processing of existence-word bitmaps during the bulk
  build's counting-sort partition step (currently scalar, per-point).
- Explicit software prefetching ahead of the windowed decompress path.
- Cache-line alignment/packing review of `CompressedBlock`/
  `CompressedCacheEntry`/`StackCacheEntry` (none currently use
  `#[repr(...)]` tuning).
- `MmapStorage`/`ArenaStorage` `BlockStorage` impls (explicitly named as
  the natural next extension in `examples/README.md`'s own "honest
  weaknesses" section; `examples/common/radixtree_dfs_interleaved_varskip_unrolled_mmap.rs`
  and `..._file_mmap.rs` are the relevant prior art, `unsafe`-scoped).
- Chaining multiple schedule levels' bucketing into one pass during a
  bounded build's "hot spine" case (a single dominant value forces
  `build_external` to re-bucket nearly the same over-budget population
  once per level) — flagged, not fixed, in the bounded-build section's
  own doc comment; `build_single_point_chain`'s existing single-point
  special case is the closest prior art for the shape of fix.
- **Whether to allow scoped, justified `unsafe` in hot paths for
  production** (e.g. unchecked slice indexing in the innermost bit-
  iteration loops) is a real, open tradeoff against this codebase's
  current "no unsafe outside two explicitly scoped mmap files" policy.
  Make an explicit recommendation with numbers, don't decide silently
  either way. One concrete data point already gathered: swapping
  `lz4_flex`'s default (`safe-encode`/`safe-decode`, zero unsafe code)
  for its `default-features = false, features = ["std"]` configuration
  (hand-optimized unsafe pointer paths) on the same 20M-triple
  `Compressed` benchmark measured `contains()` -24.6%, row queries
  -24.0%, inserts -18.0% (build time was *higher* in that single run,
  +18% — inconsistent with encode also being unsafe-accelerated, most
  likely single-run noise rather than a real effect; re-run a few times
  before trusting it either way). Roughly matches `lz4_flex`'s own
  published ~20-27% safe-mode throughput cost. The repository currently
  ships the safe configuration; this is the number to weigh against the
  no-unsafe policy, not a recommendation to switch.

## 5. Measured performance — current state (this session)

All numbers below are real measurements from this repository's own
benchmark binaries, reproducible via the commands cited. Where a number
changed materially over the course of this session's optimization work,
both the "before" and "after" are given so you understand *why* the
current design looks the way it does — do not assume the current numbers
were always true, or that they're the ceiling.

### 5.1 Point/row/insert access patterns — 20M triples

`large_scale_pred_skew_only` scenario (Zipfian predicate skew,
`BLOCK_CAPACITY = 65536`). Reproduce via `cargo run --release --example
rss_heap_generic|rss_file_generic|rss_compressed_generic large_scale_pred_skew_only`.

| | Heap | File | Compressed |
|---|---|---|---|
| Build time | 2.91s | 2.82s | 3.99s |
| Resident size (`mem_size_bytes`) | 555.14 MiB | 277.58 KiB (+555.00 MiB disk) | 282.54 MiB (2.1x vs. Heap) |
| Peak RSS (whole build) | 1.06 GiB | 1.05 GiB | ~1.06-1.08 GiB |
| Steady-state RSS (post-build) | 577.27 MiB | 25.13-28.63 MiB | 300.64 MiB |
| `contains()` (point query) | 7.33-7.61µs | 70.16-81.70µs | 271.57µs |
| row query | 246.49-255.47µs | 1.18-1.42ms | 4.59ms |
| `insert()` | 27.48-28.60µs | 96.55-101.45µs | 483.48µs |

**Before this session's optimization work**, for reference (same
scenario, original hand-specialized, non-parallel, non-streaming
`Heap`/`File` variants): build 11.85s (Heap) / 9.59s (File), peak RSS
2.27 GiB (Heap) / 2.24 GiB (File) — i.e. build is now **~4x faster** and
peak RSS **~53% lower** for both backends, from: (a) streaming node
records directly into final storage instead of materializing the whole
encoded tree in RAM first, (b) a size-gated parallel fan-out across
independent subtrees, (c) eliminating a redundant input clone (`build`
now takes `Vec<Triple>` by value), and (d) fixing `Vec` capacity
overshoot in the parallel fan-out's transient buffers.

**`Compressed`'s own before/after** (same scenario, the whole-block →
chunked-compression fix): `contains()` 2.55ms → 271.57µs (**9.4x**), row
22ms → 4.59ms (**4.8x**), insert 3.22ms → 483.48µs (**6.7x**), at a
small cost (`mem_size_bytes` +8%, build time +9%). See §4 item 4 and
§6 for the root cause and fix.

### 5.2 Query shapes (leapfrog-join patterns) — ~18.9M triples

`graph_patterns_dataset` (700,000 hubs × fanout 20; every hub has a
guaranteed star, triangle, and 4-cycle instance, so result-count
agreement across backends is itself a correctness check). Reproduce via
`cargo run --release --example phase_join heap|file|compressed build`
then `... query`.

| | Heap | File | Compressed |
|---|---|---|---|
| Build time | 3.51s | 2.59s | **845ms** |
| `mem_size_bytes` | 89.31 MiB | 48.76 KiB (+87.04 MiB serialized) | 38.04 MiB (2.35x vs. Heap) |
| star join | 16.75µs (avg 19.5 results) | 219.00µs | 691.71µs |
| triangle join | 36.59µs (avg 1.00) | 631.43µs | 2.82ms |
| 4-cycle join | 40.26µs (avg 1.00) | 673.36µs | 3.00ms |

Result counts agree exactly across all three backends on every query
shape (correctness signal, not just performance). `Compressed` vs.
`File`: ~3.2-4.5x slower here, consistent with the ~4x gap already
measured on plain point/row queries in §5.1 — the chunking fix
generalizes to join-shaped workloads, which internally issue many more
row/col-style touches than one `contains()` call, not just the query
type originally profiled.

Note the **build-time ranking reversed** relative to §5.1: `Compressed`
built *fastest* here (845ms vs. 2.59-3.51s) because this dataset's
structured, only-8-distinct-predicates content produces a much smaller,
shallower tree (height 9, 1,393 blocks) than the unstructured
`large_scale_*` scenarios (height 8, 9,786 blocks, ~30x more resident
bytes) despite a similar triple count. **Do not generalize a build-time
or compression-ratio ranking across datasets** — it is genuinely
shape-dependent, which is why every benchmark in this codebase runs
multiple shapes (§4 item 9).

### 5.3 Bounded-memory bulk build — RAM-vs-N scaling

`examples/rss_file_bounded_generic.rs`, `File` backend, fixed
`BulkBuildBudget::new(2_000_000)`, varying N (streaming-generated input,
never materialized as a whole `Vec` — see that binary's own doc
comment):

| N | Build time | Peak RSS |
|---|---|---|
| 5,000,000 | 1.04s | 205.14 MiB |
| 10,000,000 | 1.96s | 329.84 MiB |
| 20,000,000 | 3.83s | 372.99 MiB |

Clearly sublinear (not the ~2x-per-doubling an `O(N)`-RAM build would
show) — this is the direct empirical validation of the external-
bucketing bounded-build path's central claim: peak RAM scales with the
caller's budget, not with N. At larger N this is what makes billion-
triple builds tractable in bounded RAM at all; see the "bounded-memory
bulk build" section's own module comment for the full `O(budget)`
accounting and its honest limits (what the budget does and does not
bound, disk-space-during-build implications, the file-descriptor-
pressure caveat).

## 6. Known limitations to inherit honestly, not silently paper over

- **`Compressed` is inherently slower for point/row/join queries than
  `Heap`/`File`** — roughly 36-77x vs. `Heap`, 3-4.5x vs. `File`, even
  after the chunking fix. This is the real, disclosed price of
  "decompress on every touch," appropriate when RAM is the scarce
  resource and there's no disk at all — not a bug, not something to
  chase further without a specific, measured reason.
- **Bounded build's "hot spine" case is RAM-bounded but not time-
  bounded**: a long common prefix shared by most triples forces
  `build_external` to re-bucket nearly the same over-budget population
  once per level, real I/O amplification even though RAM stays flat.
- **No file-descriptor limiter** in the bounded build's `SpillBucket`
  management; worst case is roughly `threads × 64` concurrently open
  buckets. Fine under typical thread counts/`ulimit -n`, worth
  documenting as an operational prerequisite rather than silently
  capping.
- **`CompressedStorage`'s `lz4_flex` dependency**: pin `>=0.14`
  (contains the fix for RUSTSEC-2026-0041, an uninitialized-read issue
  in `lz4_flex`'s own unsafe path, which this crate doesn't use via
  `safe-decode`/`safe-encode`, but pin the fixed version regardless).
- **This project's own example-binary architecture duplicates its test
  suite ~26x** (every example binary that pulls in `common/mod.rs`
  compiles and runs the same shared tests independently). Harmless for
  correctness, just means `cargo test --release --tests --examples`
  takes ~10 minutes; `cargo test --release --example compare` alone
  gives full-suite-equivalent coverage in seconds. Decide explicitly
  whether the production crate's own test layout should avoid this
  duplication (it should — this was a deliberate, disclosed tradeoff of
  the *exploratory* `examples/` layout, not something to carry into
  `src/`).
- **Error handling throughout `examples/` is panic/`.expect()`-based**
  (malformed serialized input, I/O failures, corrupt compressed blocks
  all panic). This is acceptable for benchmark/example code; it is
  **not** acceptable for a production library API. Designing a real
  `Result`-based error type (or a documented, deliberate choice to keep
  panicking on invariant violations vs. recoverable I/O errors) is
  squarely your job, not something to inherit unchanged.
- **No `insert()` support for growing the domain established at build
  time** (`debug_assert!`ed, not just documented) — a materially larger
  problem than the block-relative storage addressing this lineage
  solves, out of scope for this handoff unless you're asked for it.

## 7. Production-readiness bar (beyond what `examples/` needed)

Things the exploratory code deliberately didn't need but the production
library does — treat this as your own checklist, not exhaustive:

- A real, documented public API (crate docs, not just internal doc
  comments meant for someone already reading the source).
- `Result`-based error handling at every fallible boundary (I/O,
  deserialization of untrusted/corrupted data, capacity/domain
  violations) instead of `panic!`/`.expect()`.
- A decision on `unsafe` policy for production (§4's open item) with
  numbers behind it either way.
- Feature-gating: `FileStorage` and `CompressedStorage` pull in real
  I/O and a new dependency (`lz4_flex`) respectively that a `Heap`-only
  consumer shouldn't have to accept — consider `cargo` feature flags.
- Versioning/stability story for the serialization format
  (`tree_serialize.rs`'s `MAGIC`/format version byte) — right now it's
  a single hardcoded magic with no forward-compatibility plan.
- A production-appropriate test layout that keeps this session's
  correctness coverage (cross-backend serialize round-trips, the
  `NaiveIndex` brute-force oracle cross-checks, the cache-invalidation
  stress tests, the parallel-fan-out and bounded-build correctness
  tests) without the ~26x duplication described in §6.
- Decide and document the fate of `src/radix64.rs`/`src/ksquare.rs`
  (§1) explicitly in your own writeup.

## 8. Deliverables expected from you

1. A short written assessment of the current design (agree/disagree with
   specific decisions above, with reasoning) before you start moving
   code — flag anything in §4/§6 you intend to change or not carry
   forward, and why.
2. The production `src/` implementation, preserving (or explicitly,
   individually overriding with justification) every principle in §4.
3. Your own before/after numbers for anything you optimize further, in
   the same format as §5, using the existing `rss_*_generic.rs`/
   `bench_generic_overhead.rs`/`phase_join.rs` benchmarks (extended, not
   replaced) as your methodology.
4. A migration/compatibility note: what changes for anyone currently
   using the `examples/common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs`
   API surface directly.
