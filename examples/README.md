# RDF triple-index structures: survey, benchmarks, and a production recommendation

Everything in this directory is an exploratory comparison of compact, in-memory
(and, in one lineage, disk-backed) index structures for RDF triples
`(subject, predicate, object)`. It exists to answer one question empirically
rather than by intuition: **given `src/radix64.rs`'s design goal (a
mixed-alphabet Morton radix trie generalizing k²-trees / k^d-trees), how does
that idea actually perform against the established literature it descends
from, and what does it take to turn the best variant into something
production-usable (real insertion, a storage-backend choice, no `unsafe`)?**

Nothing here is `src/` code. `src/radix64.rs` and `src/ksquare.rs` are the
user's own in-progress stubs and are treated as background reading only —
`examples/common/` never imports them, so this comparison can't silently
entangle itself with unfinished work-in-progress.

## 1. Architecture and engineering conventions

- **Self-contained `examples/common/`.** Every structure is its own module
  with its own copy of small shared-looking helpers (`push_var`, `read_u64`,
  `level_shifts`, the counting-sort build loop, ...). This is deliberate, not
  duplication debt: almost every variant in the RadixTree lineage exists to
  isolate *one* design axis (storage order, rank mechanism, skip encoding,
  ...), and a shared base class would make it easy for a change meant to
  isolate one axis to silently leak into another. The one genuinely shared
  pieces of infrastructure — [`bitvector.rs`](common/bitvector.rs) (rank/select),
  [`dac.rs`](common/dac.rs) (Directly Addressable Codes),
  [`tree_serialize.rs`](common/tree_serialize.rs) (the insertable variants'
  on-disk format), [`metrics.rs`](common/metrics.rs), [`reference.rs`](common/reference.rs),
  [`rdf_gen.rs`](common/rdf_gen.rs), [`sorted_neighbors.rs`](common/sorted_neighbors.rs) /
  [`join.rs`](common/join.rs) / [`path.rs`](common/path.rs) — are shared
  precisely because they're either a load-bearing *contract* (the
  serialization format, the join/BFS driver) or genuinely identical logic
  with no variant-specific axis to protect.
- **Structure of Arrays / Data-Oriented Design throughout.** No structure in
  this comparison is a pointer-chasing node graph. Every one is flat
  `Vec<u64>`/`Vec<u8>`/`BitVector` arrays plus integer offsets — matching
  `radix64.rs`'s own stated layout goal ("each topology word is a 64-bit
  child bitmap") and what makes `MemSize::mem_size_bytes()` (actual
  allocated-capacity bytes, summed per backing array) a meaningful, honest
  memory number rather than an estimate.
- **`rayon` is a `[dev-dependencies]`-only dependency**, used exclusively for
  `Mk2Forest`'s embarrassingly-parallel per-predicate scan and `path.rs`'s
  parallel BFS frontier expansion. `libc` is dev-only too, scoped to exactly
  the two files that need raw `mmap`/`madvise` syscalls (see below). The
  published library (`src/`) gains zero new dependencies from any of this.
- **A hand-rolled `SplitMix64` PRNG**, not `rand` — one more zero-extra-deps
  choice, and it makes every generated dataset exactly reproducible from a
  `u64` seed.
- **No `unsafe`, with two explicit, scoped exceptions.** Every structure here
  is 100% safe Rust except
  [`radixtree_dfs_interleaved_varskip_unrolled_mmap.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_mmap.rs)
  and
  [`radixtree_dfs_interleaved_varskip_unrolled_file_mmap.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_file_mmap.rs),
  which need raw `mmap`/`mremap`/`munmap`/`madvise` — there is no safe-Rust
  equivalent, and even the standard wrapper crate (`memmap2`) exposes its own
  mapping constructor as `unsafe fn`. Every `unsafe` block in those two files
  carries its own safety-invariant comment; nothing else in this project
  inherits the exception.
- **`NaiveIndex` (`reference.rs`) is the correctness oracle for everything.**
  No external reference implementation is available, so every non-trivial
  query, join, and path algorithm in this comparison is cross-checked against
  a brute-force `HashSet`-backed index over the same dataset — either as
  `#[test]`s embedded per module, or as randomized-probe `assert_eq!`s at
  demo-binary run time.
- **RSS is measured honestly, per-process.** Building several structures one
  after another in the same process makes `peak_rss_kb()` (`VmHWM`)
  meaningless after the first, since that counter is a whole-process
  high-water mark that never decreases. The `phase_*.rs`/`rss_*.rs` binaries
  exist specifically to give each structure its own uncontaminated
  peak/steady-state RSS reading, and additionally separate *build* from
  *query*/*insert* into different process invocations (via
  `tree_serialize`'s streamed, non-doubling-memory serialize/deserialize) so
  a phase's RSS reflects only what that phase actually does.
- **Generic monomorphization over runtime dispatch**, wherever the choice
  came up. The production candidate ([§6](#6-recommendation-the-generic-heapfile-unrolled-interleaved-radix))
  picks its storage backend as a type parameter resolved at compile time, not
  an enum matched at run time; the join driver (`ltj_run::<B: GraphBackend>`)
  and the BFS driver (`AdjacencyBackend`) are likewise monomorphized per
  backend rather than boxed, except `sorted_neighbors::Stream`, which
  deliberately trades one vtable call per `advance`/`seek` for being able to
  wrap several genuinely different concrete iterator types behind one
  interface — a cost judged negligible next to the tree traversal it wraps.

## 2. Literature this comparison is grounded in

- Brisaboa, Ladra, Navarro, **"k²-trees for compact web graph
  representation"** (SPIRE 2009) — the base k²-tree, `Direct(n,p,q,x)`
  traversal (Figure 2), and its own stated O(n) worst case / O(m)
  uniform-distribution average case.
- Álvarez-García, Brisaboa, Fernández, Martínez-Prieto, **"Compressed
  k²-triples for full-in-memory RDF engines"** (AMCIS 2011) — Mk²-tree (one
  k²-tree per predicate).
- García, Brisaboa, de Bernardo, Navarro, **"Interleaved K²-tree: Indexing
  and Navigating Ternary Relations"** (DCC 2014) — IK²-tree; this project's
  own implementation was checked directly against the paper's worked
  numeric example (`rank1(T,5)=3`, child region at `3·4+12=24`).
- Brisaboa, Ladra, Navarro, **"DACs: Bringing direct access to
  variable-length codes"** (Information Processing & Management, 2013) —
  Directly Addressable Codes, `dac.rs`.
- Pollinger, Ishii, Domke, **"The Beauty of Anisotropic Mesh Refinement:
  Omnitrees for Efficient Dyadic Discretizations"** (Mathematics and
  Computers in Simulation, arXiv:2508.06316) — Omnitree, the direct ancestor
  of `radix64.rs`'s own per-node-adaptive design goal.
- Carmona, Manzini, **"Depth first representations of k²-trees"** (2025) —
  DFS/preorder storage for cache locality, and its "Enriched Depth First"
  (explicit subtree-size skip) and Balanced Parenthesis representations for
  recovering O(1)-ish navigation without BFS/rank order.
- Munro, Raman (2001) and Navarro, Sadakane, **"Fully-Functional Succinct
  Trees"** (2014) — range min-max trees / `findclose`, behind
  `radixtree_dfs_bp.rs`'s literal balanced-parenthesis navigation.
- The classical **van Emde Boas layout** for cache-oblivious tree storage,
  behind `radixtree_veb.rs`.
- `sdsl-lite`'s own `k2_tree::_neigh` implementation was read directly (not
  just the paper) to confirm this project's row/column traversal is
  algorithmically faithful to a real reference implementation, not just the
  paper's pseudocode.

## 3. The structures

### 3.1 k²-tree family (fundamentally 2D over S,O; predicate selects context)

| Structure | File | Idea | Strengths | Weaknesses |
|---|---|---|---|---|
| **K2Tree** | [`k2tree.rs`](common/k2tree.rs) | Base k²-tree: `n×n` adjacency matrix recursively split `k×k`, one existence bit per submatrix, `T`/`L` bitmaps, `rank1(T,x+1)·k²` child offset. `k=8` here (64-way, one `u64`/node, apples-to-apples with RadixTree's own node granularity). | Textbook-simple, well-understood succinct representation; `seek`-capable row/col iterators support genuine subtree-pruning leapfrog joins. | Fundamentally 2D — predicate isn't a traversal dimension, only a tree selector; unconditional k-way fan-out at every level (see §5) is a real, literature-documented pathology on skewed data. |
| **Mk2Forest** | [`mk2tree.rs`](common/mk2tree.rs) | One independent `K2Tree` per predicate. | O(1) dispatch for any bound-predicate query; predicate trees are fully independent, so unbound-predicate scans are embarrassingly parallel (`scan_all_parallel` via `rayon`). | Unbound-predicate queries (`(s,?,?)`, `predicates_for`) must scan *every* predicate's tree — cost grows linearly with predicate count, the structure's best-known weakness; no native S,O-bound/P-free query at all without that full scan. |
| **Ik2Tree** | [`ik2tree.rs`](common/ik2tree.rs) | A *single* k²-tree over the union (S,O) matrix; each node's children carry a variable-width bitmap over the parent's still-active predicate list, existence = "own w-bit slice popcount > 0". O(height) `region_start_t` bookkeeping (not O(nodes)) locates each depth's variable-width region. | Solves Mk2's weakness directly: eager *and* lazy (select1-based, O(depth) per hit rather than O(depth·m) per visited node) unbound-predicate decode, both implemented and cross-checked against each other and against `Mk2Forest`. | Highest implementation risk in this whole project (variable-width per-node bookkeeping); still fundamentally the same 2D-over-S,O traversal as base k²-tree, so inherits the same skewed-data traversal pathology (§5) — interleaving fixes the *predicate-scan* weakness, not the *unconditional-fanout* one. |

Shared substrate: [`bitvector.rs`](common/bitvector.rs) — a sampled
rank/select bitvector (one `u64` cumulative-popcount sample every 8 words),
used by `K2Tree`'s `T`/`L`, `Ik2Tree`'s variable-width groups, and
`Omnitree`'s leaf-marker bitmap alike. Exclusive-at-`i` `rank1` is documented
as the single most common off-by-one in this whole family, and is unit-tested
against a naive O(n) loop before anything downstream was trusted.

### 3.2 Omnitree (genuinely 3D, per-node adaptive)

| Structure | File | Idea | Strengths | Weaknesses |
|---|---|---|---|---|
| **Omnitree** | [`omnitree.rs`](common/omnitree.rs) | Each *node* picks its own subset of {S,P,O} to split (a d-bit label), rather than a fixed per-depth schedule — the direct ancestor of `radix64.rs`'s own design goal. Greedy-maximal split policy (an explicit implementation choice, not from the paper): split every dimension with >1 distinct remaining value; DFS-preorder storage with `subtree_size` skip pointers (classic succinct-ordinal-tree trick) for O(#preceding siblings) navigation. | Genuinely 3D-symmetric like RadixTree (native `predicates_for(s,o)`); per-node adaptivity can spend zero bits on an already-resolved dimension at exactly the node where it resolves, not just "from this depth on" like RadixTree's coarser per-level schedule. | Adaptivity has a real storage cost: an explicit label + `child_mask` + `subtree_size` per node, and (unlike k²-tree/RadixTree) leaf coordinates can't be recomputed implicitly from depth alone in every case, since the split schedule isn't fixed in advance. Split policy is this project's own invention, not the paper's (the paper's own criterion is externally/adaptively driven, not a rule for a static point set) — a caveat worth remembering when comparing numbers. |

### 3.3 RadixTree lineage (genuinely 3D; fixed-in-advance per-level schedule; up to 64-way branching)

The common thread: a per-dimension bit-consumption schedule computed once
from `dimension_sizes` (static, evenly-split, or **dynamic** — "water-fills"
the full 6-bit-per-level budget across whichever dimensions are still active,
redistributing a soon-to-exhaust dimension's unused share to the others in
the same pass). Because the schedule is fixed before any query, leaf/branch
coordinates never need to be stored — recomputed implicitly from depth, same
as base k²-tree.

**Baseline and pure layout/encoding/navigation variants** (BFS/level-order
node-word storage unless noted; each isolates exactly one axis against the
[`RadixTree` baseline](common/radixtree.rs)):

| Variant | File | Axis isolated | Verdict |
|---|---|---|---|
| `RadixTree` (baseline) | [`radixtree.rs`](common/radixtree.rs) | — | 64-bit-per-node `t`/`l` words, `rank1(x+1)·64` child offset (base k²-tree's own trick, granularity 64 instead of k²). O(1) navigation, but BFS storage means a query's ~height visited nodes can land far apart once the tree outgrows cache. |
| `RadixTreeVeb` | [`radixtree_veb.rs`](common/radixtree_veb.rs) | Physical word placement only | Van Emde Boas recursive layout keeps rank1 navigation byte-for-byte unchanged; pays for it with an explicit permutation table that roughly doubles `t`'s memory (BFS-ordered original, needed for rank1, plus a full VEB-reordered copy). |
| `RadixTreeDfs` | [`radixtree_dfs.rs`](common/radixtree_dfs.rs) | DFS/preorder storage ("Enriched DFS") | Real cache-locality win; gives up O(1) rank1 jump for O(preceding siblings) skip-based navigation via a per-*far*-node `u32` subtree-word-count (leaves/leaf-parents need none — derivable from depth + `count_ones()`), bridged by a rank-augmented `is_far` bitvector. |
| `RadixTreeBp` | [`radixtree_bp.rs`](common/radixtree_bp.rs) | Existence encoding only (not a literal BP tree — see its own doc comment) | Sorted slot-index list per node instead of a flat 64-bit word — cheaper for sparse nodes (`K+5` bytes vs 8), more expensive once `K` grows; existence check becomes O(log K) binary search instead of one bit test. |
| `RadixTreeDfsBp` | [`radixtree_dfs_bp.rs`](common/radixtree_dfs_bp.rs) | Genuine literal balanced-parenthesis navigation | Real `findclose` via a single-level block-decomposed range min-max approximation (not the full textbook multi-level tree) — asymptotically smaller auxiliary structure than DFS's explicit `u32`/node, at real implementation complexity and a measured navigation-time cost from the block scan. |
| `RadixTreeDfsDac` | [`radixtree_dfs_dac.rs`](common/radixtree_dfs_dac.rs) | DAC-encoded subtree-word-count | Swaps DFS's fixed-width `u32` skip field for a variable-width DAC (`dac.rs`) — smaller for the many small subtrees far nodes typically have, at 0-3 extra `rank1` calls per access. |
| `RadixTreeRankDelta` | [`radixtree_rankdelta.rs`](common/radixtree_rankdelta.rs) | Rank-support mechanism only | Two-level checkpoint (coarse `u32`/63 words) + delta (fine `u8`/3 words) scheme, ~0.40 bytes/word vs. the baseline's 1 byte/word, keeping genuine O(1)-ish random access (a documented v2 dead end — a cursor-based scheme betting on query locality — measured **~17,000x slower**, since BFS-order level gaps aren't actually small). |

**The format lineage that leads to the production candidate** (each step adds
exactly one capability the previous step didn't have):

| Step | File | What it adds |
|---|---|---|
| `RadixTreeDfsInterleaved` | [`radixtree_dfs_interleaved.rs`](common/radixtree_dfs_interleaved.rs) | Interleaves a far node's skip value *inline*, immediately after its own word, in one combined byte stream — eliminating the `is_far`/rank1-translation machinery DFS/BP both need, at the cost of navigation now being pure positional-offset arithmetic (`pos + span`) instead of any rank operation at all. This is the structural insight everything downstream builds on. |
| `RadixTreeDfsInterleavedVarSkip` | [`radixtree_dfs_interleaved_varskip.rs`](common/radixtree_dfs_interleaved_varskip.rs) | Per-*level* variable skip width (1/2/4/8 bytes, native sizes only — no 3-byte tier, to keep every read a single aligned load), sized from a schedule-only worst-case bound computable before any data exists — no DAC-style rank apparatus needed, since which width a level uses is a plain array lookup by depth. |
| `RadixTreeDfsInterleavedVarSkipUnrolled` (**heap**) | [`radixtree_dfs_interleaved_varskip_unrolled.rs`](common/radixtree_dfs_interleaved_varskip_unrolled.rs) | Splits the one flat `Vec<u8>` into an unrolled linked list of fixed-capacity `Block`s (`Pos{block,offset}` addressing) — **the first structure in this entire project with real `insert()`**, O(block_capacity + height) instead of every other structure's only option, a full rebuild. |
| …Mmap | [`..._mmap.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_mmap.rs) | Same design, blocks backed by anonymous `mmap` instead of the allocator — isolates whether OS-managed pages (vs. `Vec`'s amortized-doubling growth) matter. `unsafe`, scoped. |
| …Arena | [`..._arena.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_arena.rs) | Answers mmap's actual win (exact-sized allocation, no doubling overshoot) in 100% safe Rust: one shared `arena: Vec<u8>`, fixed-size slots, sized exactly via a dry-run pass before the one bulk allocation. |
| …File | [`..._file.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_file.rs) | Node records live on disk, real `pread`/`pwrite` (`FileExt::read_exact_at`/`write_all_at`), near-zero RAM. A page-chunked (4096B), stack-shaped cache (one entry per block open on the current root-to-here path, root pinned forever) replaces an initial whole-slot cache that was measured 9-17x faster on row queries but 2-6.5x *slower* on `contains`/`insert` (see its own doc comment for the exact numbers). |
| …FileMmap | [`..._file_mmap.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_file_mmap.rs) | One shared `MAP_SHARED` mapping over a real file (not `MAP_ANONYMOUS`, so `madvise(MADV_DONTNEED)` evicts safely without losing writes), actively evicted as traversal moves past pages — mmap-speed access for hot data, bounded resident memory. `unsafe`, scoped; a first per-block-mapping draft was measured never faster than plain heap and was replaced. |
| **`radixtree_dfs_interleaved_varskip_unrolled_generic.rs`** | [see below](#6-recommendation-the-generic-heapfile-unrolled-interleaved-radix) | Heap and File unified behind one `BlockStorage` trait, monomorphized (not enum-dispatched) — **the proposed production structure.** |

### 3.4 Shared join/path/infrastructure

- [`sorted_neighbors.rs`](common/sorted_neighbors.rs) / [`join.rs`](common/join.rs):
  a `SortedNeighbors` trait (`current`/`advance`/`seek`) and a generic
  leapfrog-triejoin driver (`ltj_run`) that runs *unmodified* over Mk2Forest,
  Ik2Tree, and Omnitree — genuinely apples-to-apples worst-case-optimal join
  comparison, not three separately-tuned join implementations.
- [`path.rs`](common/path.rs): sequential and `rayon`-parallel level-synchronous
  BFS over an `AdjacencyBackend` trait, likewise shared across Mk2Forest,
  Ik2Tree, Omnitree, and RadixTree.
- [`rdf_gen.rs`](common/rdf_gen.rs): four synthetic dataset shapes (`uniform`,
  `zipfian_predicates`, `zipfian_entities`, `chain`, `star_triangle`) plus two
  workload-specific ones added mid-investigation — `graph_patterns_dataset`
  (guaranteed star/triangle/4-cycle matches per hub, for join-latency
  benchmarks) and `wide_intersection_dataset`/`_spread_s` (built specifically
  to isolate the k²-tree-family traversal pathology described in §5).

## 4. What the benchmark binaries actually measure

- `compare/` (`main.rs`/`scenarios.rs`/`report.rs`): the comprehensive
  head-to-head — build time, memory, point/row/col/unbound-predicate query
  latency, BFS (seq/parallel), and LTJ join throughput — across Mk2Forest,
  Ik2Tree, Omnitree, and every RadixTree layout variant, on uniform and
  Zipfian datasets.
- `mk2tree.rs`/`ik2tree.rs`/`omnitree.rs`/`radixtree.rs`: thin per-structure
  correctness demos (build + randomized-probe cross-check against
  `NaiveIndex`).
- `bench_insertable.rs`: the six insertable/rebuild-only variants (varskip
  baseline, heap/mmap/arena/file unrolled) side by side on four 20M-triple
  scenarios, `mem_size_bytes()`-based (single-process, so RSS would be
  contaminated — see §1).
- `bench_generic_overhead.rs`: the A/B validation this session added —
  original heap/file vs. the new generic heap/file, same scenarios, to
  confirm the `BlockStorage` trait costs nothing once monomorphized (§5).
- `rss_heap.rs`/`rss_anon_mmap.rs`/`rss_file.rs`/`rss_file_mmap.rs`: one
  structure per *process*, for honest independent peak/steady-state RSS.
- `phase_heap.rs`/`phase_file.rs`/`phase_anon_mmap.rs`/`phase_file_mmap.rs`/`phase_mk2.rs`/`phase_ik2.rs`:
  build and query/insert in separate process invocations (via
  `tree_serialize`), so a phase's own RSS/timing isn't polluted by the other
  phases.
- `phase_join.rs`: star/triangle/4-cycle join latency across
  heap/file/mk2/ik2 on `graph_patterns_dataset`.
- `phase_wide_join.rs`: isolates lazy vs. eager join-stream behavior — does a
  join cost *less* than materializing its widest input stream alone (only
  possible for a genuinely lazy, subtree-pruning `seek`, which only
  `Mk2Forest`'s `K2Tree` row/col iterators have in this project; `Ik2Tree`
  and the heap/file variants all materialize eagerly).
- `export_dataset.rs`: dumps a dataset as raw `u32` triples so an external
  (non-Rust) benchmark — specifically `sdsl-lite`'s own `k2_tree` — can be
  measured against byte-for-byte the same data.

## 5. Empirical findings

**Mk2Forest's predicate-scan weakness and Ik2Tree's interleaving fix are both
real and measured**, not just literature claims: on Zipfian-predicate-skewed
data, `Mk2Forest`'s unbound-predicate queries degrade with predicate count
(every tree must be scanned) while `Ik2Tree`'s lazy decode stays O(depth) per
hit.

**The k²-tree family's traversal pathology on concentrated-subject data is a
real, literature-acknowledged algorithmic defect, not a size/structure
tradeoff** — verified two ways this investigation:
- Direct comparison against `sdsl-lite`'s own `k2_tree::_neigh` and the
  original SPIRE 2009 paper's `Direct` procedure (Figure 2): both use
  unconditional k-way fan-out at *every* level with no select-based
  skip-ahead, exactly like this project's `RowIter`/`query_rec`. The paper's
  own Analysis section states the O(m) average case explicitly assumes
  *uniformly distributed* 1-bits — an assumption a concentrated/packed
  subject domain violates outright, with the worst case being the
  documented O(n).
- A controlled experiment (`wide_intersection_dataset` vs.
  `wide_intersection_dataset_spread_s`, with structure sizes measured
  alongside latency, per this project's own "always measure, never assume"
  standard): Mk2/Ik2 showed **identical tree size** between the concentrated
  and spread configurations but roughly a **~2000x latency gap** — proof the
  slowdown is pure traversal-algorithm behavior, not a structural size cost.
  RadixTree/heap/file's own schedule genuinely *grows* (**~8.8x** memory) to
  handle the spread case — their immunity to this pathology is a real cost
  paid up front (via `total_bits_needed`/`dynamic_schedule`'s independent
  per-dimension bit budgets), not a free lunch, but it is at least an
  up-front, predictable cost instead of a query-time cliff.

**RadixTree avoids the k²-tree-family pathology structurally, for two
independent reasons**, both visible directly in `total_bits_needed`/
`dynamic_schedule`: (1) a dimension's bit budget is allocated independently
and dropped from `active` once its own domain is exhausted, rather than being
forced into a shared joint k×k split across the *whole* height regardless of
how lopsided the actual data is; (2) skip-pointer "far node" records let a
query skip an entire non-matching subtree in **O(1)**, rather than needing to
descend level-by-level just to discover it doesn't match.

**Insertion cost, across every backend in the unrolled lineage, is
consistently microseconds** — `insert_avg` measured in the ~550ns-100µs
range across all four 20M-triple scenarios and both heap and file backends
(566ns/586ns and 98.61µs/100.19µs are the observed low/high, heap and file
respectively) —
against every other structure surveyed here (`RadixTree` baseline and every
one of its pure layout variants, `Mk2Forest`, `Ik2Tree`, `Omnitree`) whose
*only* option is a full rebuild, several **orders of magnitude** slower for a
single new triple.

**The generic `BlockStorage` monomorphization introduces no measurable
overhead** — but this claim broke once, was measured to have broken, and was
fixed by reverting, not patching forward; worth recording in full rather than
just re-asserting the number. `bench_generic_overhead.rs`, most recently:
build time, `contains`/row-query latency, and `insert_avg` for
`generic(heap)`/`generic(file)` are statistically indistinguishable from
`original(heap)`/`original(file)` across all four 20M-triple scenarios —
e.g. file `insert_avg` 98.61µs/100.19µs, 59.93µs/60.76µs, 20.74µs/21.15µs,
10.88µs/10.10µs (original/generic); disk footprint and block counts
byte-for-byte identical (10664/3869/154/18 blocks both ways). That parity
is the *default* state of this structure, not an incidental one: `FileStorage`
is a direct, deliberate port of the original file backend's own page-chunked
(4096B), write-through stack cache (§6's Storage paragraph), chosen
specifically *because* an intermediate redesign — a fixed-capacity,
always-whole-slot, deferred-write cache, tried mid-investigation to make the
block-list layer also double as a write-back buffer — both reintroduced a
real, silent data-corruption bug (a stale cache entry could survive eviction
and later be resurrected as authoritative; found and root-caused at large
scale, not assumed away) and cost real performance (touching one 8-byte
ancestor field measured at up to ~8x a small windowed touch). Two further
attempts to claw back that regression without reverting the redesign each
introduced their own correctness bug in turn. All of it was reverted back to
the original write-through design rather than carrying any of it forward —
see `StackCacheEntry`'s own doc comment in
[`radixtree_dfs_interleaved_varskip_unrolled_generic.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs)
for the full account, kept there specifically so a future redesign doesn't
re-walk the same path.
One thing *did* survive from that whole detour, kept because it's
independent of the caching question entirely: `shrink_to_fit` now rebuilds
a streamed/inserted tree's block layout by walking the live structure
record-by-record (`dfs_records`/`compact_from_records`) rather than merely
regrouping whichever physical blocks streaming happened to leave behind —
closing a real gap the hand-specialized originals never had a fix for.
Heap and file both now land on the *same*, bulk-build-optimal block count
after a streamed 2,000,000-triple build (763 blocks either way, matching
`build_dynamic` exactly) — heap was permanently stuck at 1515 blocks before
(no compaction step existed at all for it), file at ~1041 (block-granular
regrouping could only merge whole pre-existing blocks, never re-chop across
their boundaries). This validates generic monomorphization as the right
tool for consolidating the heap/file duplication without a runtime cost,
and as a real place to add capability *beyond* what either original had,
not just a container for existing logic.

## 6. Recommendation: the generic heap/file unrolled interleaved radix

**[`radixtree_dfs_interleaved_varskip_unrolled_generic.rs`](common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs)**
is this comparison's answer to "what would you actually ship." It is the
heap and file storage backends' entire logic — build, `contains`,
`query_two_bound`/`query_one_bound`, the bulk-listing query optimization,
`insert`/`insert_new_branch`, serialize/deserialize — written exactly once,
generic over a `BlockStorage` trait, monomorphized per backend
(`type Heap = ...<HeapStorage>`, `type File = ...<FileStorage>`) at compile
time. The case for it, by the same four axes this whole survey uses:

**Storage.** It inherits the interleaved lineage's core structural insight —
a far node's skip pointer lives *inline*, immediately after its own existence
word — which means navigation is pure positional-offset arithmetic
(`pos + span`) with **zero rank operations anywhere**. That single choice is
what lets this structure sidestep the entire compensating-machinery problem
the rest of the RadixTree lineage exists to explore (VEB's doubled memory for
a permutation table, DFS's `is_far`-rank-translated subtree-count array, BP's
block-decomposed `findclose`): DFS-order cache locality *and* O(1)-per-hop
navigation, together, for free. On top of that, per-level variable skip
width (1-8 bytes, chosen from a provable schedule-only bound) avoids paying a
fixed 8 bytes/far-node the way the plain DFS/BP variants do, with no DAC-style
auxiliary rank structure needed to recover it. The block-list layer
(`Pos{block,offset}` instead of one flat buffer) is what makes any of this
*insertable* at all (§ below) — and the `BlockStorage` abstraction over that
layer is validated, empirically, as free (§5): a production system gets to
choose RAM-resident (`Heap`, fastest, full footprint) or disk-resident
(`File`, near-zero steady-state RAM, one `pread`/`pwrite` per touch, backed
by a page-chunked stack cache tuned specifically against the measured
9-17x-faster-on-rows-but-2-6.5x-slower-on-points tradeoff of a naive
whole-slot cache) as a **build-time type parameter**, with no code fork and
no runtime dispatch cost between them.

**Query.** Unlike the entire k²-tree family (fundamentally 2D over S,O, with
predicate only ever selecting *which* tree/interleaving context, never a
first-class traversal axis), this structure is genuinely 3D-symmetric:
`predicates_for(s, o)` is exactly `query_two_bound` with a different pair of
bound dimensions, no special-casing. Up to 64-way branching per node (vs. the
k²-tree family's k=2, or even this project's own apples-to-apples k=8
configuration) means a shallower tree and fewer levels touched per query. And
because each dimension's bit budget is allocated independently and a
skip-pointer lets a query skip a whole non-matching subtree in O(1), it
structurally avoids the exact traversal pathology that is provably, not just
empirically, a property of the k²-tree family's own algorithm (§5) — verified
both against `sdsl-lite`'s real implementation and against a ~2000x measured
latency gap on data engineered to trigger it. Measured `contains` latency at
20M triples ranged from 86ns to ~7.5µs depending on dataset shape (heap) and
~1-70µs (file, dominated by real page-fault/syscall cost, not algorithmic
overhead) — and the generic version's numbers are indistinguishable from the
hand-specialized originals (§5).

**Build.** A single O(N) level-order counting-sort pass, no recursion, one
reused scratch buffer across the whole build (not one per node) — measured
build times from under 1s to ~11s across this project's 20M-triple scenarios.
The schedule itself is a build-time choice: static (evenly split, or a
caller-chosen split) or dynamic (water-filled, redistributing a
soon-to-exhaust dimension's unused bit share to the others), the latter
empirically using fewer node-words once any one dimension exhausts early —
exactly the situation real RDF data's typically-small predicate domain
produces.

**Insertion.** This is the structure's single sharpest differentiator versus
*everything else in this survey*, RadixTree's own other layout variants
included: `insert()` costs O(block_capacity + height), measured in the
~550ns-100µs range across every scenario and both backends, generic and
hand-specialized alike (§5) —
not the full-tree rebuild that is the *only* option for `RadixTree` itself,
every one of its VEB/DFS/BP/DAC/rank-delta siblings, `Mk2Forest`, `Ik2Tree`,
and `Omnitree` alike. For any workload that isn't purely build-once-query-forever,
that gap is not a matter of degree — it's the difference between a structure
that can serve as a live index and one that can only ever be a read-only
artifact rebuilt in a batch job.

**Honest weaknesses / open items**, for balance:
- **No compression relative to the DAC-based variants.** This lineage keeps
  a fixed 8-byte existence word (plus an optionally-narrower but still
  fixed-per-level skip) per node; `radixtree_dfs_dac.rs` shows a
  variable-length encoding *can* shrink a comparable field further, but that
  idea hasn't been ported onto the interleaved/block-list format — doing so
  would reintroduce exactly the rank-based indirection the interleaved
  design was built to eliminate, so it isn't a free combination.
- **No select-based skip-*within*-a-node** — `query_rec` still walks a
  node's matching bits one at a time via `trailing_zeros`/clear-lowest-bit.
  Since a node's branching factor is capped at 64, this is already cheap in
  absolute terms, but it's a real, uninvestigated gap relative to what a
  select-augmented bitmap could offer for a very densely-populated node.
- **`insert()` only supports points within the domain established at build
  time** (`debug_assert!`ed, not just documented) — growing the domain
  changes `height` and the whole per-level schedule, a materially larger
  problem than the block-relative storage addressing this lineage solves.
- **The generic file only unifies Heap and File** — `mmap`/arena were an
  explicit, deliberate scoping decision for this pass (see the task history
  behind this file), not an assessment that they're not worth having behind
  the same trait. If OS-mmap's page-fault-driven access speed or the arena's
  exact-sized-allocation memory profile turn out to matter for a specific
  production deployment, extending `BlockStorage` with `MmapStorage`/
  `ArenaStorage` impls is the natural next step, not a redesign.
- **Not yet wired into the shared join/BFS harness.** `sorted_neighbors.rs`/
  `join.rs`/`path.rs` have `GraphBackend`/`AdjacencyBackend` impls for
  Mk2Forest, Ik2Tree, Omnitree, and the plain `RadixTree` baseline, but not
  yet for the interleaved/unrolled/generic lineage — `phase_join.rs`/
  `phase_wide_join.rs` do cover heap/file join latency directly (including
  the eager-vs-lazy-stream distinction), so this is a real but scoped gap,
  not an untested one.
- **Single-threaded.** Unlike `Mk2Forest`'s embarrassingly-parallel
  per-predicate scan, this is one shared byte-stream keyed by a single
  schedule — there's no analogous natural data-parallel decomposition for an
  unbound-predicate-shaped query here.
- **`FileStorage::insert` does real, unbatched disk I/O on every write** —
  the cache is write-through by design (§5), so there's no deferred-write
  path to amortize a burst of inserts across. This was investigated
  directly, twice, this session: a deferred/write-back cache and a
  write-capable ancestor-window cache were both built specifically to close
  this gap, and both were reverted after introducing real correctness bugs
  rather than shipped as an unproven speed/safety tradeoff. A safe write-back
  design for this workload remains open, not ruled out — just not yet found.

None of these are reasons to prefer any other structure surveyed here for a
production role — they're the specific, scoped list of what "generalize
further" would mean next, kept honest rather than glossed over.

## 7. Running things

```sh
# Correctness demos
cargo run --example mk2tree
cargo run --example ik2tree
cargo run --example omnitree
cargo run --example radixtree

# The comprehensive structure/query/join/BFS comparison
cargo run --release --example compare

# Insertable variants: build+insert cost side by side (single process)
cargo run --release --example bench_insertable

# Validate the generic abstraction's overhead against the hand-specialized originals
cargo run --release --example bench_generic_overhead

# Honest, uncontaminated peak/steady-state RSS (one structure per process)
cargo run --release --example rss_heap [scenario]
cargo run --release --example rss_file [scenario]
cargo run --release --example rss_anon_mmap [scenario]
cargo run --release --example rss_file_mmap [scenario]

# Build/query/insert as separate process phases
cargo run --release --example phase_heap build|query|insert
cargo run --release --example phase_file build|query|insert

# Join-pattern (star/triangle/4-cycle) latency
cargo run --release --example phase_join heap|file|mk2|ik2 build|query

# Lazy-vs-eager join-stream isolation
cargo run --release --example phase_wide_join heap|file|mk2|ik2 build|query
```

All `cargo test` runs (per-module `#[cfg(test)]` suites, reached through
whichever example binary pulls that module in via `#[path]`) cross-check
every non-trivial query against `NaiveIndex` — run `cargo test --tests
--examples` from the crate root to exercise the full suite.
