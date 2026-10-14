# Graph algorithms and storage contracts

This crate owns integer graph representation and execution primitives. It does
not own term dictionaries, named-graph interpretation, query-language algebra,
transactions, property values, or database recovery.

## Representation and navigation

The Morton radix trie stores a **set of directed integer triples**. Repeated
triples collapse. A triple is also the edge identity for trail enumeration;
parallel edges need an encoding with distinct identities supplied by a consumer.
Vertices can be managed externally; zero-hop traversal accepts the supplied
start ID, even if it has no indexed edges. IDs are opaque, including predicates.

Coordinate domains remain fixed after construction. Reserve the required domain
with `build_dynamic_empty` for streaming insertion. This does not allocate a
dense vertex table. Domain growth is not implemented.

`remove` deletes a triple with full, immediate physical reclamation: O(height)
plus the cost of shifting whatever content follows the removed record within
its own block, cascading upward through any ancestor node left entirely empty
by the removal (down to, but never including, the root, which survives with
an all-zero bitmap rather than being deleted itself) and decrementing every
surviving ancestor's own cached subtree span to match. `remove_batch` sorts
into Morton order and calls `remove` once per triple; a true per-level batched
cascade (combine every node's bit-clears across the whole batch, compact each
affected block once) was built and measured, not just theorized -- a real
2-3x win when `block_capacity` is small enough that many blocks each hold
several dead records, but slower than the plain loop at this crate's actual
`DEFAULT_BLOCK_CAPACITY` and for any batch without that physical clustering,
so it was reverted rather than kept as a second, situationally-faster code
path. Block-to-block rebalancing (merging now-underfull blocks after heavy
deletion) is not implemented; a block only ever shrinks, never merges with a
neighbor.

Each node keeps a bitmap and, where needed, a variable-width subtree span.
An auxiliary Fenwick directory uses one `u64` per logical block, about 0.2% of
full 4 KiB block payloads. Same-block moves use direct offsets. Cross-block moves
and byte distances take O(log B), rather than walking B block descriptors.
Non-splitting insertion updates the directory in O(log B); block splits rebuild
it in O(B), reusing its allocation. The backend block vectors still shift on
splits. There is no blanket O(height) insertion guarantee.

`contains_batch`/`contains_batch_into` answer many point-existence queries at
once by sorting a copy of them into Morton order and walking once, caching the
previous query's root-to-leaf path: a query sharing its first `k` Morton
digits with the one just before it reuses that query's already-read node at
each of those `k` levels (same physical position, since descent is fully
determined by the digits visited), re-reading from the trie only from the
first level where consecutive digits actually diverge. The caller's own
input order is preserved in the output; sorting happens on an internal copy.
This degenerates to one independent `contains` per triple in the worst case
(no two queries share a prefix) and is never slower; measured 7-13x faster
across scattered, subject-clustered and graph-diff-style batches of 100k-200k
queries against a 200k-triple tree, at `block_capacity` 64, 256 and 4096 --
unlike the reverted batched removal cascade, this one wins broadly because
sorting alone already makes most batches share long common prefixes (the
average gap between consecutive sorted queries is usually far smaller than
the tree's height), not only genuinely clustered ones.

The internal sort key for a tree with `height <= 16` (true for any tree
built with the default/dynamic schedule, since no coordinate domain needs
more than 32 bits and 16 levels already pack bits as densely as a 64-way
node allows) is a `u128` -- one byte per level, left-aligned -- rather
than the `[u8; 64]` the general case needs for a height-unbounded, custom
`split` schedule. This shrinks the sort's own per-element size by more
than half and turns the comparator and the shared-prefix-length
computation into one native integer compare and one `leading_zeros`/8
each, instead of a per-byte loop; profiling attributed 41% of the
unpacked version's CPU time to the `[u8]` slice comparator's `memcmp`
dispatch. Measured ~1.8x faster end to end on a 2M-triple tree with
scattered queries. `height > 16` (only reachable via a deliberately thin
custom `split`) falls back to the general byte-array path.

`insert_batch`/`remove_batch` sort their input into the same Morton order
by the same two keys, but needed a different fix to actually benefit from
it: both used to call `triples.sort_unstable_by_key(|&(s, p, o)|
self.morton_key(s, p, o))`, which re-invokes the key closure on every
comparison rather than once per element, so it recomputed a whole
`height`-level Morton key from scratch `O(n log n)` times. Measured:
under that recompute pattern, swapping in the `u128` packed key was a
slight *regression*, not a win (packing one byte per level into a `u128`
via a shift-and-or chain costs a little more per computation than
writing one byte per level into an array slot -- the opposite of
`contains_batch_into`'s result, because there the comparison cost is
what dominates, not the extraction cost). The fix that actually
generalizes `contains_batch_into`'s win is doing what it does: compute
every key once into a side buffer, sort that, then write triples back in
the resulting order -- paying the `O(n)` extraction cost once instead of
`O(n log n)` times, with the `u128` comparison-side win layered on top.
Measured on a 500k-triple tree taking a 100k-triple batch: sort-only cost
dropped from ~1.2us/triple (recompute, byte-array key) to ~0.12us/triple
(precompute-once, packed key) -- roughly 10x on the sort step alone --
and `insert_batch` end to end measured ~10-15% faster. `remove_batch`
measured within noise of its old cost either way: `remove`'s own
physical cascading-delete work (shifting bytes in storage, collapsing
now-empty ancestors) dominates its total cost far more than `insert`'s
does, so a faster sort step doesn't move its total much. Both still
switched to the precompute-once version, since it's never worse and
`insert_batch` benefits from it.

`count_matching([Option<u32>; 3])` counts matching triples under the same
bound/wildcard convention as `scan`, without decoding or allocating any
matching triple's coordinates. A last-level (leaf) node's contribution is
one `u64::count_ones` on its masked word -- O(1), not one iteration per
matching bit the way listing pays to decode and collect each one. Internal
nodes still have to be visited one at a time to reach each child's byte
position, since no per-node cache of "matches below this point" exists, so
this is a constant-factor win concentrated at the leaf level, not an
asymptotic one. Measured against `scan(..).count()` on a 2M-triple tree:
~1.1-1.5x for a narrow pattern (one or two bound dimensions, few matches,
internal-node-traversal-dominated) and ~1.9-2.9x for an all-wildcard total
count (leaf-popcount-dominated), at `block_capacity` 64, 256 and 4096.

`scan([Option<u32>; 3])` streams matching triples in Morton order, with a fixed
O(maximum trie height) stack and no heap allocation on Heap. File/compressed
backends may allocate their cache buffers. Scans borrow the tree, preventing
concurrent mutation. They do not promise lexicographic projection order.

Existing adjacency methods retain bulk subtree decoding and sorted,
deduplicated outputs. Their `_into` variants retain caller buffer capacity;
sequential legacy BFS now reuses its adjacency and frontier buffers. Streaming
is useful for early exit and high degree; bulk decoding can be faster when the
whole adjacency list is required. Both paths remain available.

## Native prefix joins

`native_join::NativeJoin` borrows the tree and compiles its global subdivision
schedule into flat stage, check and write arrays. Compilation reads only domains
and schedule metadata. There are no copied input relations, sorted projections,
per-predicate indexes or exponential lookup tables over query variables.

Each stage selects a child for one atom. All atoms select compatible children
at a physical level before the traversal descends to the next level. The compiler
knows which variable bits earlier stages establish and emits masks accordingly:

- Constants and leading zeros implied by coordinate domains filter occupancy
  statically. A variable shared across roles inherits the narrowest role's domain.
- Repeated occurrences of the same variable bit within a child are compared in
  the static mask, preserving correlations between coordinate digits.
- Bits established by earlier stages select precomputed occupancy masks at
  runtime. Different coordinate roles and widths map to absolute integer bits.
- Each remaining variable bit has one owning stage. Its assignment is undone by
  clearing that stage's write mask on backtracking; no per-branch known-bit arrays
  or heap allocations are required.

Each stage writes at most 3 variables (one per S/P/O dimension), so those
writes live in a fixed `[Write; 3]` field inline on `Stage` rather than a
`start..end` slice into a separate shared `Vec<Write>`. Profiling attributed
~20% of `NativeJoin::run`'s self-time to iterating that shared-Vec slice once
per candidate child, mostly the cost of a second, separately-fetched buffer
and a bounds-checked range on top of it, on top of `Stage` already being
loaded. Inlining removes that indirection: `stage.writes[..stage.write_count]`
is already resident the moment `stage` is. Measured on the two join-execution
benchmarks with the most candidate children (`native triangle`, 154k
candidates; `native shared predicate path`, 384k candidates): ~7% and ~8-9%
faster execute time respectively, consistent across repeated runs.

`Stage`'s `checks` got the identical fix for the identical reason: a
stage's total bit count across all 3 dimensions is at most 6 (`slots = 1
<< bits.sum() <= 64`), so at most 6 checks, now a `[Check; 6]` field
inline on `Stage` instead of a `start..end` slice into a shared
`Vec<Check>`. Smaller, consistent win: `open()` runs once per stage
*activation*, not once per surviving candidate the way the write loop
does, so there's less of it to save -- measured ~2-5% faster execute
time on the same two benchmarks (A/B against an otherwise-identical
build with only this one field reverted to the shared-`Vec` shape, back
to back, to isolate it from both machine-load noise and the `writes` fix
already measured above). Applied anyway: leaving `checks` as the one
inconsistent field after fixing `writes` for the exact same reason would
be the kind of gap that's easy to forget was ever a question.

Re-profiled after both fixes (`perf record --call-graph fp` on a
200k-triple, 20k-vertex triangle join) to find what's left: `run`'s own
body is ~72% self-time, of which ~46% is `advance_past_children`/
`child_span_end` (28% specifically inside the variable-width skip-field
read, `read_var_at`) and `open` is ~22%, of which ~8% is `prefix_cursor`
(`read_u64`'s own bounds-checked byte-to-array conversion) and ~3%
is the remaining `Check`/`Write` array iteration itself (the loop's
own pointer-increment/bounds-check, not indirection -- that's now as
cheap as a 0-3/0-6-element array scan gets without `unsafe`). None of
that looks like the same class of bug: `read_var_at` reads a
variable-width integer at a position scattered through the tree's own
byte-packed storage, so the cost is a memory-latency stall on real,
necessary data, not a wasted allocation or an avoidable indirection --
the same conclusion [`crate::tree`]'s own skip-pointer design already
reaches for *why* that cost exists, just confirmed freshly instead of
assumed to still hold after this round's changes.

By induction, surviving prefixes agree on all shared known bits. Every complete
binding identifies an occupied leaf in each atom, and incompatible prefixes are
never discarded unless a required equality or constant fails. Since a full
binding determines each atom's triple, results are unique under set semantics.
Empty conjunctions yield one empty binding; a false ground atom suppresses all
results. Duplicate atoms are removed at compilation.

The iterative executor uses O(atoms * height + variables) workspace and emits
borrowed bindings through `ControlFlow`. Compilation metadata has the same
space bound. Child positions use existing DFS subtree spans; skipped subtrees
are not expanded. Ordinary execution compiles out diagnostic counters, while
`visit_with_stats` reports node openings, surviving child choices and results.
Once its workspace has sufficient capacity, the executor itself allocates nothing.
Backend cache allocations remain subject to the storage contract.

This is a native prefix join, not a proven implementation of the Qdag WCO bound.
It can revisit nodes for different compatible prefix combinations, especially
when asymmetric split schedules reveal shared-variable bits at different levels.
Results are unordered. The borrow prevents mutation while a plan remains in use;
recompile after mutation. Native Heap plans can be shared, with separate
workspaces; File/Compressed plans inherit their backends' threading restrictions.
The [literature review](TOPOLOGY_REVIEW.md) describes the relationship to prior work.

## MINUS / anti-join: left to consumers

A `NOT EXISTS`/`MINUS`-style operator over one or more triple patterns was
prototyped twice in this crate (a fused in-descent `NOT EXISTS` atom inside
`NativeJoin`'s compiled stage array, then a `join_run_minus` helper that
collected rows and bitmap-deduped the minus-relevant columns before one
`contains_batch` call) and both were removed again: this crate's trees are
deliberately shallow and wide (at most 6 bits per level, to keep ordinary
descents fast), so a shared variable rarely becomes fully known early
enough in a join's schedule for in-descent pruning to pay for its own
per-candidate `contains` cost, and a fixed collect-dedup-batch pipeline
forces a buffering shape (one `Vec<u32>` per row) that a consumer
composing its own pipeline on top of `NativeJoin::visit` doesn't need and
can do better informed by its own data's actual duplication pattern (e.g.
skipping a probe only when it repeats the immediately preceding one,
which a join's already-ordered output makes cheap and effective without
a bitmap at all). `NativeJoin::visit` (borrowed `&[u32]` rows, no
allocation after workspace warmup) and `contains_batch` (shared-descent
batched existence checks) remain the composable primitives a consumer
builds a MINUS/anti-join layer from; this crate does not provide one
itself.

## Join API and migration

`TriplePattern<P = u32>` preserves constant-predicate syntax; use
`TriplePattern<Term>` for variables in any coordinate. Variables must cover
`0..num_vars`. Invalid queries return `TreeError::InvalidConfig`.
`graph::join_run` compiles and collects a native join, while `try_join_run` also
returns validation errors. Historical `ltj_run`/`try_ltj_run` names are aliases
for these native helpers and do not execute Leapfrog Triejoin. Both return
`JoinRows`, not `Vec<Vec<u32>>` -- see "Allocation convention" below.

The materialized prepared executor, packing layouts, preparation options and
boxed sorted-neighbor adapters have been removed. Callers use `NativeJoin` with
`NativeJoinWorkspace`, and drop borrowed plans before mutating their trees.
Joins now require the native tree over a `BlockStorage`; `PatternBackend` remains
available for custom scan backends used by path and automaton algorithms.
There is currently no join executor with an established WCO bound. Projection
with duplicate elimination, aggregation, optional matches and query-language bag
semantics remain outside this full-binding set operator.

## Allocation convention: borrow by default, let the caller own

Audited every public function returning owned data for the same question
`contains_batch_into`/`query_two_bound_into` already answer: does this
*have* to allocate per call, or can the caller supply/reuse the buffer and
decide for themselves whether to keep a copy? The crate already follows
this almost everywhere, not just where this round of work landed:

- `NativeJoin::visit`/`visit_with_stats` hand each result row to the
  caller's closure as a borrowed `&[u32]`, zero allocation after
  workspace warmup -- the caller copies only if and when it wants to keep
  a row past the callback.
- `contains_batch_into`, `query_two_bound_into`, `query_one_bound_into`,
  `predicates_for_into` take `out: &mut Vec<_>` and retain its capacity
  across calls; `_into`-less siblings are thin "allocate fresh and
  delegate" convenience wrappers over them, not separate implementations.
- `enumerate_paths` reuses one `GraphPath` buffer for the whole DFS
  (push/pop backtracking) and emits it to the caller's closure as `&GraphPath`
  at every qualifying depth -- no per-path allocation, the same shape as
  `NativeJoin::visit`'s per-row emission.
- `bfs_sequential`'s frontier and neighbor-list buffers are reused across
  the whole walk; only the one `dist: Vec<i32>` result (unavoidably
  O(num_nodes), not O(one allocation per emitted anything)) is owned.

`graph::join_run`/`try_join_run` (and their historical aliases
`ltj_run`/`try_ltj_run`) used to be the one exception: collecting into
`Vec<Vec<u32>>` meant one heap allocation per matching row by
construction of that return type. They now collect into `JoinRows`
instead -- every row's bindings pushed into one flat `Vec<u32>` via
`extend_from_slice` during the same `NativeJoin::visit` call these
helpers already wrapped, with a separate `rows: usize` count (not
derived from `data.len() / row_width`, since `row_width` is legitimately
0 for a fully-ground query with no variables -- one matching row with no
bindings is a real, distinct result from zero matching rows).
`JoinRows::rows()` borrows each row as `&[u32]`, the same shape
`NativeJoin::visit` hands its caller; `JoinRows::to_vec_rows()` is the
escape hatch back to one owned `Vec<u32>` per row, for a caller that
genuinely wants that. This is strictly an allocation-count change, not a
semantic one: still collects everything before returning (unlike
`visit`, still no early exit or bounded memory), still the "don't want
to think about a workspace" convenience path its own doc comment points
away from for repeated or large queries.

A second pass looked past the public, per-query/per-row API at the
storage backends underneath every one of them, for the same per-call
"allocates a scratch buffer, uses it once, throws it away" shape:

- `FileStorage::slot_insert_at`/`slot_remove_at` (every `insert`/`remove`
  on the `File` backend shifts a block's tail through exactly this kind
  of scratch buffer) and `CompressedStorage::insert_bytes`/`remove_bytes`
  (every `insert`/`remove` on the `Compressed` backend decompresses a
  whole block into one) both used to allocate that buffer fresh on every
  call. Both now reuse a `scratch: Vec<u8>` field instead, retaining its
  capacity across calls -- the same convention as everywhere else in
  this list, just one level down, in backend code instead of tree code.
- Measured honestly, not assumed: unlike `contains_batch`/`join_run`,
  where the sort/collection step this fixed was *the* dominant cost,
  here it wasn't. `File`'s `insert`/`remove` is dominated by its
  `pread`/`pwrite` syscalls (tens of microseconds each); `Compressed`'s
  by the LZ4 decompress-mutate-recompress CPU work (tens to over a
  hundred microseconds per call, decompressing/recompressing a whole
  block on every single mutation). Both fixes measured within noise of
  the unfixed version on their own backend's `insert`/`remove` -- still
  worth keeping (never worse, strictly fewer allocator calls, consistent
  with the rest of this convention), just not a result to overstate.
- `HeapStorage::insert_bytes`/`remove_bytes` needed no equivalent fix:
  both already shift a block's tail in place via `Vec::splice` on the
  block's own existing allocation, with no separate scratch buffer at
  all -- the in-RAM backend's structure already avoided this pattern.
- Not touched: `CompressedStorage::chunk_and_compress`'s own
  `Vec<Vec<u8>>` of freshly-compressed chunks. That's the backend's
  actual stored representation being rebuilt, not a throwaway
  intermediate -- reusing old chunk buffers would mean changing how
  `lz4_flex` is called (writing into a caller-supplied destination
  instead of returning a fresh one), a bigger, riskier change to a
  backend whose own module docs already record multiple reverted cache
  designs here, not attempted without a concrete, measured case for it.

## Paths

`HopRange` validates inclusive minimum and maximum edge counts. `Step` chooses
outgoing, incoming or both directions and optionally one predicate. Both-direction
traversal emits each self-loop once; distinct opposite-direction triples remain
distinct edges.

- `reachable` emits each endpoint once, at its first qualifying length. Before
  the minimum depth, visitation is layer-local: a shorter visit must not suppress
  a valid longer route. Once the minimum is reached, an earlier qualifying visit
  dominates a later one, so global endpoint visitation safely prunes cycles.
- `has_path` stops as soon as a qualifying target is emitted.
- `shortest_walk` uses the same layered semantics and retains predecessor
  records to reconstruct the shortest qualifying walk. This may require
  O(min_hops * reached vertices) records before the lower bound is satisfied.
- `enumerate_paths` streams walks, trails or simple paths. Trails cannot reuse
  a stored triple; simple paths cannot repeat vertices. The DFS is iterative,
  retaining O(max_hops) path entries and backend cursors. Output can be
  exponential. It emits every qualifying edge-distinct path, not only endpoints.

`PathWorkspace` defaults to sparse visited sets, independent of the largest ID.
For dense ID spaces, `PathWorkspace::dense(n)` allocates bitsets with sparse
overflow; clearing touches only modified words. Frontiers and visited capacity
are reused. Call `release()` to discard retained high-water memory.

`PathAutomaton` accepts an epsilon-free NFA with packed transition ranges by
source state. This covers label sequences, alternatives and repetition without
bundling a query-language parser. The search tracks `(node, automaton state)`
per layer, then globally after the lower hop bound, and deduplicates emitted
endpoints. A caller's compiler eliminates epsilon transitions. Zero-length
acceptance is represented by an accepting start state. This operator has walk
semantics; simple-path regular expressions and weighted shortest paths are not
provided by this API.

## Validation and backend boundaries

Deserialization verifies domains, dimension permutations, schedules and cached
shifts, widths, exact framing, allocation limits, record boundaries, existence
bitmaps, coordinate bounds, topology, node counts and every stored subtree span.
It validates sequentially rather than trusting skip pointers. Version-1 snapshots
remain compatible, including empty logical blocks from older file insertions.
There is no checksum: a corruption that encodes another valid graph cannot be
distinguished from legitimate data.

Default limits permit 1 GiB of total block capacity, 1,000,000 blocks and 16 MiB
per block. Both byte-slice and file loaders offer explicit limit overrides.
Invalid static split configurations return errors instead of failing to advance
the builder. Degenerate `[1,1,1]` domains support insertion and duplicate detection.

File storage remains scratch storage. Its operational I/O failures panic, and
File/Compressed interior caches remain non-`Sync`. Native plans borrow their
backends and inherit these threading restrictions. These changes do not introduce
recovery, locks on every read, or an unsafe `Sync` implementation.

## Verification and measurements

`tests/algorithms.rs` compares joins with exhaustive assignments across atom
orders and paths with an independent enumerator. It covers diagonals, variable
predicates, unanchored triangles, disconnected queries, cancellation/reuse,
borrowed-plan reuse, sparse IDs, cycles, all directions, all path uniqueness modes,
automaton product states, backend parity, truncations and byte mutations.
Existing insertion, bulk-build, compaction and cross-backend tests remain active.

`tests/native_join.rs` additionally checks native joins against exhaustive
assignments and an independent test-only triple enumerator, including asymmetric splits, all coordinate
orders, full-width IDs, shared predicates, 12-variable disconnected patterns,
cancellation, workspace reuse, inserted/restored trees and all storage backends.

Run `cargo bench --bench algorithms` for a deterministic 20,000-vertex,
60,000-triple fixture. It reports compilation separately from execution, plan and workspace
capacity, bounded traversal, bulk versus streaming adjacency, and unanchored
triangle enumeration. These are smoke measurements, not production latency
guarantees; use representative degree distributions, query shapes, cold/warm
caches and tail-latency measurements before choosing a deployment configuration.

A local release run with Rust 1.98.1 produced the following illustrative native
measurements on the 474,870-byte tree (the harness prints current measurements
on your machine):

| Query | Native compile | Native execution | First result, compiled plan | Native plan + workspace capacity |
|---|---:|---:|---:|---:|
| Anchored, 1 result | 1.06 µs | 1.44 µs | 0.31 µs | 2,480 bytes |
| Triangle, 20,000 results | 1.24 µs | 2.61 ms | 0.33 µs | 3,504 bytes |
| Two-edge path sharing a predicate variable, 60,000 results | 1.08 µs | 6.03 ms | 0.19 µs | 2,256 bytes |

Native compile + fresh workspace + full execution took 2.62 µs, 2.67 ms and
6.02 ms respectively. These are repeated measurements with warm process/data
caches, not cold-cache or tail-latency claims. Small differences between summed
and directly measured times reflect measurement variability.

Historical comparison, recorded before removing the materialized executor:
the packed prepared triangle required 2.93 ms preparation and
6.20 ms execution, plus 225,000 bytes of relation payload. The shared-predicate
path required 7.34 ms preparation, 9.28 ms execution and 480,000 payload bytes.
Prepared-plan metadata and workspace are additional to those payload numbers.
Native capacities exclude inline structs and backend caches; native execution
constructs no input or intermediate relations. The removed prepared anchored query
executed faster when reused (0.10 µs). These historical comparisons do not claim
that native execution is always faster and are not rerun by the current harness.
