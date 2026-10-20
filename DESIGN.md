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
several dead records, but slower than the plain loop at `block_capacity`
4096 (what `DEFAULT_BLOCK_CAPACITY` was at the time) and for any batch
without that physical clustering, so it was reverted rather than kept as a
second, situationally-faster code path. Dated: not re-measured since
`DEFAULT_BLOCK_CAPACITY` changed (below), and the direction that change
pushes mutation cost (up, per the measurements below) is the opposite of
where batching would start to help, so this tradeoff is worth
re-checking, not assumed to still hold. Block-to-block rebalancing
(merging now-underfull blocks after heavy
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
whole adjacency list is required. Both paths remain available -- and, since
`restricted_descent_visit`/`predicates_for_visit` (below), early exit no
longer has to mean giving up bulk decoding's own pruning to get it.

An external review (the "graph store harness team") measured what was then
`query_two_bound_into` as 10-45% faster than `scan` for exactly the bound
patterns both support, but noted it has to build the whole result before
returning, so a consumer wanting `LIMIT`/`ASK` semantics was stuck choosing
between `scan`'s early exit and the bulk form's pruning speed. The two don't
actually have to trade off: `query_rec` (the shared descent both
`query_two_bound_into`/`query_one_bound_into` already walked) was already an
explicit-stack iterative loop, not real recursion, so converting its three
`out.push` sites (two in `query_rec` itself, one inside the recursive
`decode_listing` that backs `bulk_list_subtree`'s own fast path once a
subtree becomes fully unconstrained) to `emit(v)?` against a caller-supplied
`impl FnMut(u32) -> ControlFlow<T>` was a mechanical change, not a redesign.
`decode_listing` combines "propagate an early `Break`" with "return the
decode cursor so recursive sibling calls know where to resume" by returning
`ControlFlow<T, usize>` instead of a bare `usize` -- the `Break`/`Continue`
split already matches the two things that function needed to communicate.

Early exit is real, not merely "a smaller returned set": `?` on
`ControlFlow` performs an actual early return out of every level of
`decode_listing`'s recursion and `query_rec`'s stack-resume loop the moment
`emit` returns `Break`, so nothing past that point is read, decoded, or
decoded-into. Measured directly: breaking on the first match of a one-bound
streaming query against a 100,000-entry adjacency list (one subject, 50
predicates, 100,000 objects) is ~2567x faster than materializing the full
list -- an ASK-style query against a high-degree hub now costs roughly the
depth of the tree, not the size of its adjacency.

### `restricted_descent`: one method instead of a method family, plus real ranges

A follow-up question, after the `_visit` work above: `query_two_bound*`/
`query_one_bound*` were two parallel three-method families (bulk/`_into`/
`_visit`) differing only in how many dimensions were pinned to an exact
value, built on the same per-level bitmask idea K²-tree range queries and
the k²-triples/BMatrix RDF-on-K²-tree line of work already use for point
lookups, row/column listings, *and* range queries -- one recursive descent
that only differs in how each level's match mask is computed. The crate's
own version of that mask builder, `bound_mask`, could only express "pinned
to one exact value" or "free"; widening it to "value range" or "small union
of ranges" would get range/`IN (...)`-style queries pruned at native tree
speed instead of falling back to `scan` plus a manual filter, and would let
`query_two_bound`/`query_one_bound` collapse into one method instead of two
near-duplicate families, since both are just different arities of "some
dimensions restricted, one left free."

**This isn't a type change, because of one hard constraint.**
`precompute_level_masks` walks every level once, top-down, *before*
traversal starts, advancing a single `lo` through *the one* sub-index each
bound dimension's exact value falls into at that level -- correct because an
exact value picks exactly one descent path, so there's only one possible
`lo` to precompute every later level's mask from. A range or small set of
values can match *several* sub-indices at a level, each leading to a child
with a *different* `lo` -- so there is no longer one flat `level_masks[64]`
array that's correct for every node at that level; some siblings now need
different masks for their own descendants. `scan`'s own mask setup has the
identical precomputed-array shape and the identical limitation; it just
never needed more than exact/free because that's all `bound_mask` could
express. So a `Range`/`Multi` restriction needs the traversal to compute
each node's mask from its own actual position, not look one up in an array
built before the first node was even visited -- for every dimension that can
carry more than one matching sub-index at a level, not just the one(s) a
caller happens to restrict that way.

**What shipped.** `DimRestriction<'a>` (`Free | Exact(u32) | Range(u32, u32)
| Multi(&'a [(u32, u32)])`, the last borrowing the caller's slice -- no
allocation, same convention as everything below) replaces the old
`[usize; 2]`/`[u32; 2]` bound-dims-and-values pairs. One method,
`restricted_descent_visit` (plus `restricted_descent`/`restricted_descent_into`
bulk siblings), takes a `[DimRestriction; 3]` and an `emit_dim`, and
subsumes `query_two_bound_visit` (two `Exact`s, `emit_dim` left `Free`) and
`query_one_bound_visit` (one `Exact`, `emit_dim` left `Free`, the third
dimension *also* left `Free` but never decoded -- same duplicate semantics
`query_one_bound_visit` already documented, falling out of the same
mechanism unchanged) as special cases. `query_two_bound*`/`query_one_bound*`
are gone; `predicates_for*` survives as a named convenience wrapper over
`restricted_descent*([Exact(s), Free, Exact(o)], 1, ...)` since it reads
better than the general form at every call site and is used across a dozen
`examples/common/*.rs` comparison harnesses.

The new engine, `query_rec_restricted`, mirrors `query_rec`'s explicit
stack/backtrack shape exactly, with two differences: stack frames carry
every dimension's `lo` (like `scan`'s own `ScanFrame`), not just the emitted
dimension's, and the mask is computed per node via `restriction_mask`
(`bound_mask`'s generalization) instead of indexed out of a precomputed
array. Whether a level still needs any mask-based pruning at all -- the
`unconstrained_from` handoff to `bulk_list_subtree`'s fast path -- turned
out to still be purely schedule-based (a dimension stops being able to
constrain anything once its own bits run out at a level, independent of
*which* value or range it's restricted to), so that part needed no new
per-node logic at all, just a `Free`-or-out-of-bits check generalizing the
old exact-value version.

**Correctness**, checked rather than assumed: the whole existing
`cross_check_tree` matrix (uniform/zipfian/skewed/wide-branching/bounded/
parallel/streaming-insert/remove/serialize, 4 block capacities, 3 backends)
now asserts `restricted_descent`/`restricted_descent_visit` with `Exact`/
`Free` restrictions against the same naive-index oracle the old methods
were checked against, and matched exactly everywhere -- the regression
guard for the collapse itself. New tests cover `Range`/`Multi` directly
against a `scan`-plus-manual-filter oracle: in-domain and partially/fully
out-of-domain ranges, overlapping and out-of-order `Multi` ranges, and a
case built specifically so a `Range`-restricted dimension's bits run out
partway down a deep tree, to exercise the bulk handoff with a real `Range`
(not just `Free`) still live elsewhere in the restriction set.

**Performance, measured rather than assumed to be free.** A direct
benchmark (2,000,000 triples, 50,000x200x50,000 domain, 20,000 queries) of
`restricted_descent_visit` running through the new dynamic-mask engine
unconditionally (no fast path at all) against the old static-precompute
`query_two_bound_visit`/`query_one_bound_visit`, same dataset, same queries:
~12% slower for a two-`Exact` query, ~64% slower for a one-`Exact` query (the
one-bound shape spends most of its traversal with two of three dimensions
`Free`, so it pays the per-node mask recomputation's full cost across the
most levels). Real and worth keeping both engines over: `restricted_descent_visit`
dispatches internally -- when every restriction is `Exact`/`Free` (everything
the old methods could express), it still runs the old
`precompute_level_masks`/`query_rec` path; `query_rec_restricted` only runs
when a `Range`/`Multi` is actually present. The *public* surface is one
method either way; only the implementation forks, and only because the
numbers said to, not because it was assumed necessary going in.

`scan`/`count_matching` are unaffected by any of this -- out of scope for
this round. Folding them into the same restricted-descent engine (which
would let a query restrict and emit more than one dimension in the same
call, matching the shape literally) is real future work, blocked on `scan`
first getting an `unconstrained_from`-equivalent bulk fast path of its own
(today it always walks bit-by-bit, even through fully-open levels), not
designed here.

### `perf`-driven follow-up: `BlockIndex::locate`, and a fix that measured worse

`perf record --call-graph dwarf` on the same benchmark shape above
(`restricted_descent_into`, three restriction shapes) found
`BlockIndex::locate` -- the Fenwick-tree binary search `Tree::advance`'s
slow path uses to turn an absolute byte offset back into a `(block,
offset)` -- as the single largest or co-largest self-time bucket in every
shape measured (42% for two `Exact`s, 11% for one, 22% for a `Range`),
comparable to or larger than `child_span_end`'s own already-documented
skip-field-read cost. Root cause: this tree is shallow and wide (height 7
over a 50,000x200x50,000 domain, 4096-byte blocks), so near-root subtree
spans routinely exceed a block by a wide margin, and `advance`'s two O(1)
fast paths (stays in this block; lands exactly at the next one's start)
essentially never fire that high up -- the slow path runs on nearly every
far-node jump near the root, not as a rare case.

The slow path pays for two independent O(log blocks) Fenwick walks every
time -- `BlockIndex::prefix(pos.block)` to turn `pos` into an absolute
byte offset, then `BlockIndex::locate` to turn the target absolute offset
back into a block -- when the first is pure recomputation: whatever
produced `pos` (a previous `advance`/`locate` call) already derived that
exact value and discarded it. Threading it through instead (as a new
`abs: u64` field on `Pos`, set to `target` directly by `locate`'s two
return branches, and to `pos.abs + n` by `advance`'s two fast paths --
verified first, by hand, that every direct `Pos { .. }` construction site
outside `advance`/`locate` either computes it the same way or never has
its `abs` read at all, since it's only ever consumed by `advance`'s own
slow path) removes `prefix` from `advance` entirely, and does the same
for `Tree::byte_distance`'s identical double-`prefix` shape, collapsing
it to a single subtraction.

Measured, not shipped: implemented, full test suite green across every
feature combination (`Pos`'s invariant held everywhere, including every
insert/remove/serialize path that constructs a `Pos` directly), but a
back-to-back A/B (three runs each direction, same process, same
benchmark) showed it **30-41% slower**, not faster, across all three
restriction shapes. `Pos` is `Copy`, passed by value through
`advance`/`child_span_end`/`advance_past_children` on every single step
of every traversal, and stored in every `[Frame; MAX_HEIGHT]` stack --
growing it from 16 to 24 bytes cost more in copying/register pressure
through those hot, heavily-inlined paths than the removed `prefix` walk
saved. Reverted. The `locate` cost itself is real and unaddressed; a fix
that doesn't grow `Pos` itself (e.g. a separate, explicitly-threaded
cursor carried only through the specific hot recursive functions, not a
field every one of `Pos`'s ~20 unrelated construction sites has to carry)
remains a plausible direction, not designed or attempted here.

**The config lever instead: `block_capacity`.** Since `locate`'s cost is
driven by how often a jump crosses a block boundary without landing
exactly on one, and by how many blocks there are to search over, a
bigger `block_capacity` attacks both without touching any code. Measured
on the same 2,000,000-triple, 50,000x200x50,000-domain tree, sweeping
`block_capacity` from 4096 to 4,194,304, for both the read side
(`restricted_descent_into`, 20,000 calls per shape) and the mutation
side (20,000 inserts then 20,000 removes into the same tree):

| `block_capacity` | blocks | two-`Exact` | one-`Exact` | `Range` | insert | remove |
|---|---|---|---|---|---|---|
| 4,096 (old default) | 12,469 | 28,126 ns | 579,136 ns | 77,195 ns | 20,061 ns | 6,706 ns |
| 16,384 | 3,116 | 21,926 ns | 499,129 ns | 66,862 ns | 7,378 ns | 6,094 ns |
| 65,536 (new default) | 779 | 15,152 ns | 461,913 ns | 59,261 ns | 7,553 ns | 7,595 ns |
| 262,144 | 195 | 12,468 ns | 464,351 ns | 56,207 ns | 12,984 ns | 14,760 ns |
| 1,048,576 | 49 | 11,434 ns | 454,343 ns | 55,095 ns | 37,567 ns | 45,436 ns |
| 4,194,304 | 13 | 10,728 ns | 453,397 ns | 55,104 ns | 118,903 ns | 163,953 ns |

Confirmed causally, not just correlated: re-profiling the two-`Exact`
case at `block_capacity` 1,048,576 (49 blocks) shows `BlockIndex::locate`
at 3.8% self-time, down from 42% at 4096 -- `child_span_end`'s own
already-documented skip-field-read cost is now the dominant, legitimate
bottleneck.

The two curves cross in opposite directions: reads keep improving (with
diminishing returns) as blocks grow, while `insert`/`remove` get
dramatically worse past a few hundred KB -- splicing a block's tail on
every mutation scales with how much is in the block, and at 4 MB blocks
that's 18-24x worse than the best case. `16,384`-`65,536` is close to
the minimum for both inserts and removes while already capturing most of
the read-side win (within 1.4-2x of the 4 MB plateau, vs. 2.6x worse at
the old 4096 default); past that, mutation cost explodes for a read gain
of only another 30-40%. `DEFAULT_BLOCK_CAPACITY` changed from 4096 to
65536 on this basis -- a single global value, not the depth-varying
("bigger blocks near the root, smaller near the leaves") scheme
considered first, since the sweet spot for both sides turned out to
overlap enough that the extra complexity (and the DFS-interleaving
wrinkle that makes "near the root" ambiguous at the block level -- a
single block, filled sequentially during a DFS-order bulk build, spans
many levels at once, not one) isn't earning its cost yet.

**Re-checked at 10x scale, not assumed to hold.** Same sweep, same
density (domains and triple count both x10: 500,000x200x500,000,
20,000,000 triples, height 8 vs. 7), one `block_capacity` at a time
under a memory cap to avoid leaning on an already-swap-exhausted
machine. The shape repeats: `insert`/`remove` are both near their
minimum at 65,536 (32us/9.5us) and climb sharply past it (157us/205us
at 4 MB -- 4.9x/21.7x worse), while reads at 65,536 are within
1.08-1.57x of the 4 MB plateau across all three query shapes, not the
best case but close to it. `DEFAULT_BLOCK_CAPACITY` 65536 holds up as a
near-optimal single value at both scales, not something that happens to
work at 2,000,000 triples and needs rethinking past it -- the crossover
point is set by the absolute `block_capacity` (mutation cost) against a
logarithmic block count (read cost), neither of which shifts much with
total graph size.

### `one-Exact` queries: instrumented first, found a bigger lever than planned

A restriction shape with exactly one `Exact` dimension and the rest
`Free` (e.g. "every object connected to this subject via any
predicate") measured far slower than `two-Exact` at every scale tested
-- 10-100x, growing with graph size. The initial theory (two pinned
dimensions reach a "both exhausted" pivot that hands off to the bulk
decode path, one pinned dimension doesn't) turned out to be wrong when
checked against the real schedule: neither shape ever reaches that
pivot in these domains, because the pinned dimension's own bit budget
(spread across every level by the round-robin `dynamic_schedule`) never
hits zero before the last level, regardless of how many *other*
dimensions are also pinned.

Temporary per-level instrumentation (`AtomicU64` histograms around
`child_span_end`'s fast/slow-path outcome and `query_rec`'s per-node
batch size, same throwaway convention as this project's earlier
sibling-skip-chain measurement; built, run at 2M/20M triples, then fully
removed) found the real mechanism: at *deep* levels, each node has only
~1-2 real children (nothing to batch), but `one-Exact` visits 50-200x
more such nodes than `two-Exact` does at the equivalent level (79-320
million vs. 400K-6M, at 20M triples). Because the pinned dimension's
bits are interleaved across every level rather than resolved early, a
node partway down the tree mixes together every subject sharing the
path's bits *so far*, not just the one being queried -- `two-Exact`
narrows out of that mixing faster because its second pinned dimension
has fewer bits and exhausts early; `one-Exact`, with every other
dimension free, stays mixed for most of the tree's height. This is real
navigation through real tree structure, not traversal-code waste -- no
amount of batched-skip or smarter `BlockIndex` searching removes it,
since the expensive levels' batch sizes are already too small to batch.

**The actual lever: the build-time bit schedule, not the query engine.**
`Heap::build`/`build_with_order` already accept an explicit `split`
(bits per dimension per level) instead of `build_dynamic`'s round-robin
default. Front-loading the commonly-pinned dimension resolves its
identity in a handful of levels instead of spreading it across all of
them. Measured on 2,000,000 triples, back-to-back:

| schedule | height | one-Exact | two-Exact | contains |
|---|---|---|---|---|
| `dynamic` (default) | 7 | 1,216,685 ns | 32,572 ns | 1,298 ns |
| `split=[3,1,2]` | 8 | 149,570 ns (8.1x) | 32,233 ns | 1,522 ns (+17%) |
| `split=[4,1,1]` | 16 | 36,446 ns (33x) | 12,476 ns (2.6x) | 2,116 ns (+63%) |

`[3,1,2]` is a clean win (one extra level, a small `contains` cost);
`[4,1,1]` is far more dramatic but makes the tree much taller and
triples `contains`'s cost -- a real tradeoff a caller who knows their
query shape can make, not a free default change (the schedule has no
way to know a workload's query mix at build time).

**The `remaining & mask` restructuring, implemented on top, not instead.**
Separately, `query_rec`/`query_rec_restricted`'s non-last-level branches
(both the initial-descent loop and the backtrack-resume loop) checked
`matched = (mask >> qi) & 1 != 0` one real sibling bit at a time, calling
`child_span_end` for each regardless of match -- the same bit-by-bit
shape the leaf branch already avoided (`word & level_masks[level]`,
computed once). Restructured to match: `candidates = remaining & mask`
finds the first real-and-matching bit in O(1) via `trailing_zeros()`,
and everything in `remaining` below it (`remaining & ((1 << qi) - 1)`,
necessarily non-matching since it precedes the first match) is skipped
in one `advance_past_children` call instead of one `child_span_end` call
per bit. No change to `advance_past_children` itself -- it still loops
internally -- so this doesn't reduce the number of underlying
`BlockIndex::locate` calls; it only removes redundant per-bit branching
and matching work (same `qi` extraction, same backtrack-frame contents,
same descent semantics, verified against the full existing test suite
unmodified, including the `Range`/`Multi` paths through
`query_rec_restricted`, which exercise the multi-match backtrack case
most heavily).

Expected (per the plan this followed) to be roughly neutral to slightly
positive on its own. Measured instead as a large, independent win,
reproduced back-to-back: at 2,000,000 triples, `dynamic`'s `one-Exact`
dropped from 1,216,685 ns to 350,473 ns (3.47x), `two-Exact` from
32,572 ns to 8,009 ns (4.07x), `contains` from 1,298 ns to 818 ns
(1.59x) -- and the win stacks multiplicatively with the schedule
change: `split=[4,1,1]` plus this restructuring brings `one-Exact` to
13,551 ns, an **89.8x** total improvement over the original
`dynamic`+unrestructured baseline. At 20,000,000 triples (`dynamic`
schedule only), the win is real but smaller for `one-Exact`
specifically (2,957,778 ns -> 2,086,113 ns, 1.42x) while `two-Exact`
keeps a strong gain (56,443 ns -> 19,774 ns, 2.85x) -- consistent with
the instrumentation's own finding that `one-Exact`'s dominant cost at
larger scale concentrates at deep levels where batch sizes are already
~1-2, leaving little for this restructuring to batch; the schedule
change is still the larger lever at that scale for `one-Exact`
specifically, with this restructuring an independent, stacking
improvement on every shape at every scale measured.

**Generalizing the schedule fix: weight by size instead of picking a
caller-specified split.** (**Reverted** -- see "Reverted: the
size-weighted schedule" further below. Kept here as the historical
record of what was measured and why it looked like a strict win on the
data tested; it wasn't one in general.) The `[3,1,2]`/`[4,1,1]` experiments above
required a caller to already know which dimension their workload
usually pins alone, and `[4,1,1]`'s dramatic `one-Exact` win came with
a real cost (height 16, `contains` 63% slower) because `static_schedule`
never redistributes a dimension's unused budget once it exhausts.
`dynamic_schedule` (`build_dynamic`'s default, used by every benchmark
in this investigation unless a custom split was passed explicitly)
already redistributes on exhaustion -- it just splits each level's
budget *evenly* across whichever dimensions are still active, regardless
of how many bits each one actually still needs. Changed to split by
each active dimension's own remaining bit count instead (more bits per
level to whichever dimension still has more left, same redistribution
behavior once one exhausts) -- a one-line change to the inner loop's
`share` computation, no new public API, no caller-visible choice to
make. Full existing test suite green unmodified (this changes
`build_dynamic`'s default schedule, exercised by nearly every existing
test).

Measured, back-to-back, at 2,000,000 triples (on top of the
`remaining & mask` restructuring above):

| schedule | height | one-Exact | two-Exact | contains |
|---|---|---|---|---|
| even split (previous default) | 7 | 350,473 ns | 8,009 ns | 818 ns |
| **size-weighted (new default)** | **7** | **53,534 ns (6.5x)** | **2,515 ns (3.2x)** | 917 ns (+12%) |
| static `[3,1,2]` (for comparison) | 8 | 45,394 ns | 8,345 ns | 1,078 ns |
| static `[4,1,1]` (for comparison) | 16 | 14,282 ns | 4,132 ns | 1,698 ns |

Same height as the previous default -- no structural cost at all, unlike
the static splits -- while beating both static splits on `two-Exact`
(2,515 ns vs. 8,345/4,132 ns) and landing close behind `[3,1,2]`'s
`one-Exact` number without requiring a caller to know their query shape
in advance. Re-checked at 20,000,000 triples: `one-Exact` 2,086,113 ns
-> 161,359 ns (12.9x), `two-Exact` 19,774 ns -> 3,486 ns (5.7x), same
height (8) as before. The only cost found anywhere: `insert`/`remove`
~7-15% slower (8,679/8,144 ns vs. 7,553/7,595 ns at 2,000,000 triples)
-- measured, not assumed away, and small next to the query-side gains.

Why this helps both query shapes, not just the one a formal
"dimension fully exhausted" check targets: the real cost driver
(confirmed by the Phase 0 instrumentation above) is how many distinct
real entities share a path's bits *so far* at a given depth, not
literally whether a dimension's schedule allocation has hit zero.
Giving the larger dimensions more bits per level resolves their
identity faster in practice -- fewer distinct values still mixed
together at any given depth -- which shrinks the real sibling count at
every node along the way, independent of whether any dimension's bit
budget has formally reached zero yet. This held up at every scale and
every query shape *tested* -- but, as the revert note below covers,
every one of those tests used symmetric domains, which turned out to
be exactly the blind spot that mattered. The static-split experiments
remain documented above as what was tried first and what they cost,
not as a live recommendation.

### Re-profiling after both fixes, and a smaller-than-hoped win in `restriction_mask`

Re-profiled `two-Exact`/`one-Exact`/`Range` after the `remaining & mask`
restructuring and the size-weighted schedule landed, to see where the
remaining cost sits once the two big fixes stopped dominating it.
Normalized to exclude build-time contamination (query time didn't
fully dominate build time in this run, so `build_rec` and friends still
show up at 20-33% of raw samples): `two-Exact`'s cost is now mostly
`BlockIndex::locate` (52% of query-only time, *up* from 42% before --
everything else shrank more than it did), `one-Exact`'s is mostly
`child_span_end` (51%) plus, newly visible, `restricted_descent_into`'s
own `sort_unstable`/`dedup`/push bookkeeping (29%, not prominent before
because the traversal used to dwarf it), and `Range`'s is mostly
`restriction_mask` (35%, unchanged by either fix -- neither touched it).

`restriction_mask`'s per-dimension expansion loop built a mask by OR-ing
one shifted copy of the running `mask` per allowed sub-index --
`O(run_len)` shift-and-OR operations, up to 64 for a `Free` dimension
at this crate's max per-level branching. Replaced with `expand_run`
(`src/tree.rs`), which builds the same result via repeated doubling
(the standard double-and-add construction used for fast exponentiation)
in `O(log run_len)` -- `Free`/`Range` always produce a single
contiguous run so they take this path directly; `Multi` decomposes into
its separate ranges and takes it once per range. Verified exhaustively,
not just via the existing end-to-end `Range`/`Multi` tests (which may
not reliably exercise every `(start, run_len, shift)` combination,
particularly the full-width edge case at `run_len = 64`): a new test
compares `expand_run` against a naive one-shift-per-copy reference
across every valid combination the real bit-packing can produce
(`start + run_len <= 64 >> shift`, `shift` 0..=6), not a sample of them.

Measured (A/B, same tree, same queries, 20,000,000 triples): `Range`
went from 24,649 ns/call to 22,525 ns/call -- a real but modest 8.6%,
well short of what `restriction_mask`'s 35%-of-self-time share might
suggest. Why the gap: that 35% was measured against real per-level
widths, and with the schedule now splitting its 6-bit budget across up
to three active dimensions, a single dimension's own width at a given
level is usually small (2-3 bits, width 4-8) -- the `O(run_len)` vs.
`O(log run_len)` difference is real at that size but nowhere near the
difference it'd make at the rare width-64 case (a single dimension
alone with the whole budget, only once the others have exhausted).
Kept anyway: the replacement is a strict improvement with no case where
it's slower (same result, fewer operations, verified exhaustively), it
just isn't the large lever its profiled share implied in isolation.

The other two findings -- `locate`'s cost for `two-Exact` and
`restricted_descent_into`'s collection overhead for `one-Exact` -- are
not yet acted on. `locate`'s absolute cost is small now (~1.3us/call
within an already-~2.5us total), putting a batched/shared `BlockIndex`
search deep in diminishing-returns territory for the benefit it would
need to justify touching a function this central again, especially
given the `Pos.abs` attempt's own lesson earlier in this file.
`restricted_descent_into`'s `sort_unstable`/`dedup` cost is real but the
absolute counts involved (tens of entries at this benchmark's density)
are already small -- `_visit` already exists as the escape hatch for a
caller who doesn't need sorted, deduplicated output and wants to skip
this cost entirely.

### Reverted: the size-weighted schedule, after a real-world regression report

A downstream store team reported that upgrading to this crate's
`94b8945` ("optimized child selection and bit spread" -- the size-
weighted schedule plus the `remaining & mask` restructuring, both
above) made subject-bound lookups and subject-pinned joins 2-5x
faster, exactly as measured here, but made free-subject lookups 3-4x
slower and joins that chain through an object (e.g. `?a p1 ?b . ?b
type C0`) 13-16x slower -- 112ms to 7.8s on one of their real queries.
They'd already ruled out `block_capacity` (switching it back to 4096
didn't change the result) and confirmed it was the tree code, not
their own.

The reason every benchmark in this file's `one-Exact` investigation
missed this: all of them used *symmetric* domains -- subject and
object domains of equal size. The store's real data doesn't: far
fewer distinct subjects than objects, the ordinary RDF shape where many
triples share a subject and objects range over both entity references
and literal-like values. Reproduced directly on a 20,000-subject,
2,000,000-object synthetic dataset built the same way as every other
benchmark here: under the size-weighted schedule, `scan(?,?,o)`
(object bound) cost 3.22ms/call against `scan(s,?,?)` (subject bound)
at 704us/call -- a 4.6x gap in the same direction as the report, despite
the object-bound query returning far fewer rows. Reverting the
schedule to even-split reproduced this crate's own pre-change baseline
(`ba9d07c`) on the same data almost exactly (within 1-2%, i.e. no
difference) and made the two directions symmetric again (1.59ms vs.
2.08ms, no large gap either way). The mechanism: weighting by remaining
size means whichever dimension is *larger* still takes more levels to
fully resolve, bigger per-level share or not, since it simply has more
total bits to get through -- so pinning the larger dimension got
*slower* under the weighted schedule, backwards from what the
`one-Exact` investigation's symmetric benchmarks could ever reveal,
since neither dimension was ever the clearly-larger one there.

Reverted `dynamic_schedule` to the even split (`src/tree.rs`'s doc
comment on it carries this same account). The `remaining & mask`
restructuring shipped in the same commit was not implicated and was
kept: it doesn't change which nodes get visited or depend on domain
shape, only how cheaply existing navigation happens, and `scan` itself
(which doesn't go through `query_rec` at all) reproduced the original
baseline exactly once the schedule was reverted, confirming the
restructuring was never the source of this regression. A per-dimension
weighting heuristic that's actually safe across asymmetric domains
remains unsolved -- this crate's default needs to be safe for data it
has never seen, not just the shape it happened to get tested against.

### `advance`'s slow path: two more ideas tried, both rejected

Follow-up to the complexity analysis that came out of the regression
investigation above: `advance_past_children`'s cost (and so every
accessor built on it -- `scan`, `restricted_descent*`, `NativeJoin`'s
`prefix_child`) is dominated by how many real siblings must be walked
past per level, each walk potentially paying for a `BlockIndex::locate`
search. Two ways to cut that cost were tried and both rejected, measured
rather than assumed either way:

- **Resuming a sorted-target Fenwick search.** Resolving several
  siblings' increasing positions by continuing the search from where
  the previous one landed, instead of restarting `locate` from the
  root each time, would in principle cut a run of searches from `O(k
  log n)` to `O(Σ log gap_i)`. Not attempted: `locate`'s binary-lifting
  search greedily builds its result as a sum of powers of two starting
  from block 0; resuming from a nonzero block isn't a parameter change
  to it, it needs the `sums` array to answer range-relative queries it
  doesn't store. That's a different index structure, not a tweak to
  this one, and not something to improvise into a shared, load-bearing
  structure without its own dedicated design.
- **A short linear probe before falling back to `locate`.** Checked a
  handful of blocks forward via plain `block_len` reads (no
  data-dependent chain, unlike `locate`'s search) before paying for the
  full search -- cheap if it lands, cheap-ish if it doesn't. Measured
  (A/B, same benchmarks as the schedule investigation, back-to-back,
  each side reproduced twice to rule out noise): a real, reproducible
  **4-7% regression** on `one-Exact`, `Range`, and both `scan`
  directions -- exactly the cases this was meant to help, because
  their skips are genuinely far (the whole reason they're expensive),
  so the probe almost always fails and just adds 8 wasted reads before
  the same `locate` call as before. Marginal, not-the-point gains on
  `insert`/`remove` (closer targets there) didn't justify it. Reverted;
  working tree confirmed byte-identical to pre-change afterward.

Net conclusion: the near-root "few real siblings, each a large skip"
case identified by Phase 0 doesn't have an easy fix. Cutting it for
real needs either a different index structure for `BlockIndex` or a
different storage layout, not a local change to `advance`'s call
sites -- left as a real, open, unsolved cost, not a wrong guess papered
over with a workaround.

### A real, measured, but modest waste: eager backtrack-frame precompute

`query_rec`/`query_rec_restricted`'s stack-based descent reads a
matched sibling's own skip field immediately at match time, before
even descending into it, to precompute where backtracking should
resume afterward (`next_child_pos`, stored in the pushed frame). That
read is wasted whenever nothing real follows the match at that level
(`remaining` is already empty once the matched bit and everything
below it are cleared) -- the frame gets popped on the very next visit
without its `child_pos` ever being read.

Instrumented before touching anything (same temporary-counter
convention as every other investigation here): tallied, per level,
every eager read against whether it turned out wasted, across
`two-Exact`, `one-Exact`, `Range`, and both directions of the
asymmetric-domain case from the schedule regression. The pattern was
identical and clean across all five: ~0% wasted at levels 0-3 (dense
branching -- a match is rarely the last real sibling there), climbing
to 48-73% at level 4, ~99% at level 5, and 100% at level 6 and beyond
(sparse branching -- often exactly one real child, so a match
essentially always *is* the last one). Overall waste rate across whole
queries: 9.3-14.1% of all eager reads, consistent across every shape
tested.

The catch: this waste concentrates exactly at the levels Phase 0 already
showed are fast-path dominated (near-0% `locate`-triggering rate at
levels 4+). A wasted read there costs roughly a word read and a
comparison on already-resident data, not a `BlockIndex::locate` search
-- so even a clean fix (skip the `child_span_end` call when `remaining
& !consumed_mask == 0`, storing a sentinel `Pos` that's structurally
guaranteed never to be read, since a frame with `remaining == 0` is
popped before `child_pos` is ever touched) would plausibly save on the
order of 1-3% of total query time for `one-Exact`, less for `two-Exact`
-- real, but not in the same class as the schedule or `remaining & mask`
fixes. Not implemented yet: the waste is confirmed and the fix is
simple and low-risk, but the expected payoff is modest enough that it's
being left as a known, quantified, cheap-to-pick-up-later improvement
rather than rushed in for a small number.

### Hoisting child skip values into the parent record

A different idea for the same `advance_past_children`/`child_span_end`
mechanism above: today, a "far" node (`level + 2 < height`, i.e. more
than two levels above a leaf) stores its own skip-to-end-of-subtree
value *inside its own record*. Skipping past K real siblings before a
match means K sequentially dependent steps -- read sibling 0's skip,
`advance()` to sibling 1, read *its* skip, `advance()` to sibling 2, and
so on -- because you cannot know where sibling i+1 starts without first
reading sibling i's own record. Each of those `advance()` calls can
independently fall into `BlockIndex::locate`'s O(log blocks) search. The
proposal: store all of a parent's real children's skip values
contiguously in the *parent's* own record instead, one entry per real
child (same bitmask-driven indexing already used everywhere else in
this engine). Once the parent is read, every child's span is already in
hand -- no chain of dependent reads, no repeated `locate()` -- so
skipping K siblings collapses to one bulk, cache-resident read, O(K)
pure arithmetic to sum, and exactly one final `advance()`/`locate()`
call to the computed target.

This doesn't inflate storage (the same values move, they aren't
duplicated) and doesn't add mutation cost in the common case: `insert`
already walks down once recording every ancestor's position in `trail`
and patches every far ancestor's skip field afterward
(`insert_new_branch`, O(height) direct writes to already-known
positions) -- under the hoist scheme the write target just moves one
level up (into the parent's slot for that child, at an offset
computable from the sibling popcount already in hand), same asymptotic
cost. The real new complexity is the rarer structural case -- a child
going from absent to present (or vice versa) needs a *slot*
inserted/removed in every far-ancestor's record, not just a value
patched, which is new plumbing beyond what `insert`/`remove` do today
(though it likely reuses the existing `insert_bytes`/byte-shifting
primitives those already rely on).

Instrumented before committing to the rewrite: added counters to
`advance_past_children`'s non-leaf loop, tallying per level how many
real siblings get chained past per call (`avg_k`) and, by duplicating
`advance`'s own fast-path predicate inline (without changing its
behavior), what fraction of those individual chained steps would have
hit the slow `locate()` path. Measured across five shapes --
`two-Exact`, `one-Exact`, `Range` on a symmetric 500,000x200x500,000
domain at 10,000,000 triples, and both directions (small-domain-pinned,
large-domain-pinned) of an asymmetric 20,000x200x2,000,000 domain at
3,000,000 triples -- with the pinned value(s) randomized per call rather
than fixed, since an earlier fixed-query pass turned out to badly
undersample the near-root levels most queries barely touch.

The result was a clean, large, and consistent effect in every shape:

```
                total steps   total slow   levels 1-2 steps   levels 1-2 slow (% of all slow)
two-Exact       3,701,744     8.1%         7.6% of steps      91.4%
one-Exact       16,817,020    0.6%         0.4% of steps      65.7%
Range           4,087,377     1.8%         1.6% of steps      84.2%
asym S-bound    9,147,266     0.7%         0.6% of steps      76.1%
asym O-bound    8,722,747     1.0%         0.9% of steps      78.2%
```

At levels 1-2 specifically, average chain length (`avg_k`) ran
3-30 real siblings depending on domain shape and branching, and
80-100% of the individual steps in that chain hit `BlockIndex::locate`
-- essentially every sibling skipped near the root costs a full search
today. By level 3 the slow fraction drops to 1-3%, and by level 4+ it's
under 0.1%, matching the fast-path-dominated picture from Phase 0 and
the eager-precompute investigation above. So: levels 1-2 account for
under 8% (often under 1%) of all sibling-skip steps in a query, but
91%, 84%, 78%, 76%, and 66% (respectively) of *all* the slow,
`locate`-triggering steps -- the same "small fraction of the work, most
of the real cost" concentration this engine keeps finding near the
root, and -- unlike the size-weighted schedule -- a pattern that held in
exactly the same shape across symmetric and asymmetric domains and both
pin directions, not something that flips sign depending on domain
asymmetry.

Hoisting would collapse each of those near-root chains from K
independent (and today, almost-always-slow) `locate()`-class calls down
to exactly one. That's roughly an 80-90%+ cut in the *slow* chain-step
count specifically -- a materially bigger lever than the eager-precompute
waste above, and aimed at the same root-cause (`advance_past_children`'s
sequential dependency) the earlier hardware-counter profiling pass
identified but didn't have a concrete fix for.

**Prototyped the read path to turn that into a real measurement**,
deliberately scoped to stop short of `insert`/`remove` per the plan:
`src/hoist_proto.rs` (`HoistedProto`) builds a tree with the hoisted
layout (reusing the production engine's own schedule computation --
`dynamic_schedule`, `compute_level_skip_bytes`, etc., made `pub(crate)`
for this, not reimplemented, so both layouts get the identical tree
shape) and a from-scratch recursive query engine (no mutation support
at all; build once, query many times). To isolate the hoisting effect
from the prototype's own implementation style, a second type,
`NonHoistedControl`, builds the *production* layout (each far node
stores its own skip field, patched after recursing -- exactly
`build_rec`'s approach) through the *same* recursive, per-node-mask
visitor shape as `HoistedProto`. Three-way, cross-validated (every
query's result set checked against production, sorted, before trusting
any timing) A/B on the same five shapes as the chain-length
measurement:

```
                              pure hoisting      recursive-style        net
                              (control/proto)    overhead (control/prod) (prod/proto)
symmetric two-Exact           4.14x              1.70x                   2.43x
symmetric one-Exact           1.18x              1.31x                   0.90x
symmetric Range               1.58x              1.04x                   1.52x
asymmetric S-bound            1.50x              1.45x                   1.03x
asymmetric O-bound            1.40x              1.08x                   1.30x
(free-subject shape)
```

The middle column is the honest explanation for why "net" doesn't just
equal "pure hoisting": `HoistedProto`/`NonHoistedControl` are both a
plain recursive descent with no backtracking-stack reuse and no
precomputed flat per-level mask array for the pure-Exact/Free case --
production's actual `query_rec` has both. That overhead happened to
roughly cancel the hoisting win for `one-Exact` and `S-bound` (net
≈1x), while `two-Exact`/`Range`/`O-bound` still show a large net win
even against that handicap. The column that actually answers "does
hoisting help": **pure hoisting effect, positive in all five shapes,
1.18x-4.14x, including both directions of domain asymmetry** -- unlike
the size-weighted schedule, this did not flip sign anywhere tested. A
real implementation landing hoisting *inside* `query_rec`/
`query_rec_restricted` (keeping their existing frame-stack and
bulk-listing optimizations) should get close to the "pure hoisting"
column, not the handicapped "net" one.

One correctness wrinkle surfaced and resolved during validation, worth
recording since it looked like a hoisting bug at first: a `[Free, Free,
Exact(o)]`-shaped query with `emit_dim` equal to the *same* dimension
that's `Exact`-restricted is documented as a degenerate existence gate,
not an enumeration -- using it as a stand-in for "enumerate subjects for
a fixed object" (emit_dim should be `0`, not `2`) produced a false
mismatch against production unrelated to either prototype. Separately,
a real but tiny (≤4/500, ~0.8%) and *pre-existing* divergence shows up
in both `HoistedProto` and `NonHoistedControl` identically for one
shape (`S`-bound, two Free dims): production's bulk-listing fast path
collapses one duplicate emission in a rare shared-value-across-
different-free-dim-branches case that this plain recursive visitor
doesn't. Confirmed unrelated to hoisting specifically (identical in
both the hoisted and non-hoisted control), not investigated further
since it's orthogonal to this measurement and pre-dates it.

### Hoisting landed for real: `build_rec`, `insert`, `remove`, parallel build

The prototype's win was real and consistent enough to implement for
real, touching every code path that creates or mutates node records.
The prototype itself (`src/hoist_proto.rs`, `NonHoistedControl`,
`examples/hoist_proto_bench.rs`) is gone now that production has the
real thing -- this section is the record of landing it.

**The layout change.** A node no longer stores anything describing
itself. A far node (`level + 3 < height` evaluated at its *parent*,
equivalently the old `level + 2 < height` evaluated at the node itself)
has its span written into one slot of its *parent's* hoisted array
(existence word, then one `level_skip_bytes`-wide entry per real child,
in ascending `qi`/real-index order) instead of a self-contained skip
field in its own record. `own_record_bytes` becomes word-dependent
(`WORD_RECORD_BYTES + popcount(word) * level_skip_bytes[level + 1]`
when this node's *children* are far, `WORD_RECORD_BYTES` otherwise --
the leaf-level and second-to-last-level cases are unchanged, since
those never had a stored skip field to begin with). A new
`far_span_sum(parent_pos, child_level, start_real_index, count)`
replaces `child_span_end`'s far branch: one bulk read of `count`
contiguous array entries starting at `start_real_index`, summed in
registers -- `advance_past_children` now takes `parent_pos`/
`parent_word` alongside the child position and bitmask, derives
`start_real_index`/`count` from the mask (always a contiguous run of
real-child bits, by construction of every caller), and does the far
case in one `far_span_sum` call plus one `advance` instead of a chain
of `child_span_end` calls. `child_span_end` itself keeps only its two
non-far branches, with a `debug_assert` that it's never called for a
far level (a hard guard against a silently-wrong read, since that
span no longer exists at a far child's own position at all).
`MAX_RECORD_BYTES` (`WORD_RECORD_BYTES + K_SQUARE * MAX_SKIP_RECORD_BYTES`
= 520) replaces the old `WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES`
(16) as the worst-case single-record size, used as a fixed stack
buffer during build (never a per-node heap allocation, same discipline
as before).

**Every caller that read a child's span now reads its parent's array
instead**, with the same bulk-vs-sequential split the chain-length
measurement found: `query_rec`/`query_rec_restricted`'s stack frames
gained a `pos: Pos` field (the far parent `remaining` was read from)
so a resumed backtrack frame can still call `far_span_sum` against the
right node; the "eager backtrack-frame precompute" these two functions
already did (see the section above) now goes through
`eager_child_span_end`, which dispatches to `far_span_sum` for far
levels and unchanged `child_span_end` otherwise. `bulk_list_subtree`
no longer computes its own span (nothing at a far node's position to
compute it from) -- it takes `span_end` from the caller, which already
has it as a side effect of the same eager-precompute call that sets up
`pos`/`lo`/`level` for the node it's about to bulk-read; the one case
with no such precompute (the whole query starting already past
`unconstrained_from`, only possible at the root) uses
`BlockIndex::total()` (added for this) as the one case where a node's
span legitimately has nowhere else to come from. `count_rec`, `Scan`
(a new `pos`/`real_index` pair on `ScanFrame`), and `PrefixCursor`
(`pos`/`word` alongside the existing shrinking `unskipped` mask) all
got the analogous treatment -- `PrefixCursor::prefix_child` in
particular just calls the same generalized `advance_past_children`
unchanged, since its shape already matched.

**Build (`build_rec`, the parallel fan-out path, `build_external`'s
bounded/spilling path, `build_single_point_chain`).** All four
restructured the same way: no more per-node self-patch once a
subtree's total is known. Instead, the parent (which already owns the
hoisted array for its children) patches each child's array slot
immediately after that child's own recursive build call returns with
its total byte count -- `build_children`/`build_children_sequential`'s
parallel and sequential stitch loops, and `build_external`'s
equivalent (`flush_fit_batch`/`flush_taken_batch*`, threading an
`own_pos`/`child_skip_width`/running `real_index` through what used to
be a pure byte-count accumulator), all patch inline as each child
completes rather than returning just a sum for the caller to self-patch
with. `build_single_point_chain` (a fresh single-point path built new
for `insert_new_branch`) writes `sizes[lvl + 1]` (the child's span)
into `lvl`'s record instead of `sizes[lvl]` (its own) -- the same
relocation, just for a chain where every node has exactly one real
child.

**Mutation was the hard part, and genuinely harder than the earlier
complexity estimate.** Three real bugs surfaced during a 3,000-op
randomized insert/remove stress test (the existing
`interleaved_insert_remove_stress_small_capacity` family, `block_capacity`
deliberately tiny to force splits/merges constantly), each caught by
cross-checking every op against a naive reference *and* round-tripping
through `serialize`/`deserialize` (whose `validate_node` independently
recomputes every subtree's span from scratch and checks it against
what's stored -- the single most valuable correctness tool for this
whole change):

1. **Growing a far node's array can't be a partial splice at all, not
   even of just the new entry.** `insert_bytes` never splits the bytes
   it's given, but it always splits (if a block doesn't have room)
   *at the insertion point* -- so splicing anything less than a
   record's *entire* span (header and array together) risks that
   split landing inside the record, separating the header from (part
   of) its own array across two blocks. `far_span_sum` and friends all
   assume a far node's whole record lives in one block; a header in
   block N and its array in block N+1 reads back as silent nonsense,
   not a crash. The only shape immune to this: read the complete old
   record, splice the new entry into a full copy, then swap it in as
   one `remove_bytes` of the old record's exact span followed by one
   `insert_bytes` of the whole new one, both anchored at the record's
   own start. (Two earlier, more "surgical" attempts -- growing in
   place, and removing/reinserting just the array's *tail* past the
   new slot -- each failed this same way under a tight `block_capacity`
   stress run and got replaced, not patched.)
2. **Removing a node whose only child just vanished must never use
   `own_record_bytes` with the stale pre-removal word.** By the time
   `remove_cascade` reaches a node that's becoming fully empty, the
   *previous* (deeper) iteration of the same loop has already removed
   that one child's slot from this node's array -- so this node's
   *current* on-disk size is always just its bare header, never
   `own_record_bytes(level, word)` (which would still count the
   already-gone entry and try to remove more bytes than the block
   actually has).
3. **`block_capacity` has a real, higher floor now.** A far node's own
   record scales with its branching factor (up to 64 children x 8
   bytes = 512, plus the header = `MAX_RECORD_BYTES` = 520) -- previously
   irrelevant, since a node's own record was always at most 16 bytes
   regardless of branching. `validate_config` now rejects anything
   below `MAX_RECORD_BYTES` (was 16); every test and example using a
   deliberately-tiny capacity to stress splitting (`32`, `64`, `128`)
   got bumped to a still-tiny-but-legal value (`MAX_RECORD_BYTES` or
   `1024`), and the one edge case this surfaces -- removing the root's
   *only* record (pre-array, the very first insert into an empty tree)
   empties all of storage down to zero blocks, leaving nothing for
   `locate` to resolve the re-insert against -- falls back to
   `append_record`+`finish_append` used together as a complete,
   immediately-finalized one-record build session (the only
   backend-agnostic way to go from zero blocks to one; `FileStorage`
   in particular stages `append_record` into a buffer that only
   becomes real, addressable storage once `finish_append` flushes it,
   so the two must always be paired).

**A fourth bug was in storage, not in the hoisting logic, and
pre-existed this change.** `CompressedStorage::remove_bytes` and
`FileStorage::remove_bytes` never dropped a block that shrank to zero
content -- unlike `HeapStorage`, which always has. A lingering
zero-length block is invisible to `BlockIndex` (it contributes nothing
to any prefix sum), so `locate` transparently skips over it when
resolving a byte offset -- exactly the right behavior for *reading*,
and exactly wrong for `insert_new_branch`'s remove-then-reinsert-at-
the-same-logical-offset pattern, which silently landed the reinserted
bytes in the *next* physical block instead. Harmless as long as
nothing ever depended on which literal block a position's bytes
physically sat in -- except `Pos::START` (block 0, offset 0) is
hardcoded as where every traversal starts, so the one time this
happened to the root's own record, the root quietly moved to block 1
and every subsequent operation read an empty block 0 instead. Fixed at
the source (both backends now drop an emptied block and renumber/
invalidate their read caches exactly like `HeapStorage` already did),
not worked around in `insert_new_branch`, since the inconsistency was
real and would have bitten anything else that ever tried the same
remove-then-reinsert pattern against these backends.

**The result, measured for real** (not the prototype's handicapped
recursive visitor): same five shapes, same methodology as every other
A/B in this document -- build this checkout, measure, `git stash` back
to the commit before this session's hoisting work, rebuild, measure
again, restore:

```
                                    before        after        speedup
symmetric two-Exact                 78,470 ns     24,556 ns    3.20x
symmetric one-Exact               1,785,809 ns  1,714,014 ns   1.04x
symmetric Range                     366,828 ns    305,416 ns   1.20x
asymmetric S-bound                1,144,324 ns    952,578 ns   1.20x
asymmetric O-bound (free-subject) 1,233,313 ns    976,296 ns   1.26x
```

Every shape improved, none regressed -- `two-Exact` landed close to
the prototype's own net prediction (and better, since the real
`query_rec` keeps its existing frame-stack/bulk-listing optimizations
that the prototype's plain recursion didn't have); `one-Exact` is a
small but real win rather than the roughly-even prototype number,
likely for the same reason. The asymmetric, free-subject-shaped case
-- the one the store team's regression report flagged as highest
priority -- is a real 26% win, and (unlike the size-weighted schedule
that caused that regression in the first place) this result held in
the same direction and a similar magnitude across every domain shape
tested, symmetric and asymmetric, both pin directions.

**Format consequence:** `serialize::FORMAT_VERSION` bumped from 1 to
2. The node-record byte layout changed; a version-1 snapshot would
silently misparse under version-2 code rather than failing loudly, so
old snapshots are now explicitly rejected (`UnsupportedVersion`)
instead of readable. There is no migration path from v1 to v2 -- an
old snapshot must be rebuilt from its source triples.

### Runtime path compression: tried, measured, reverted

The eager-precompute waste noted above (a matched sibling's own span
gets read and a stack frame gets pushed even when `remaining & !consumed_mask
== 0`, i.e. nothing is left to resume from that frame -- common at
sparse, near-leaf levels where `avg_k` drops to ~1.0-1.4 real siblings
per node) suggested skipping the push and the read entirely in that
case, rather than just skipping the read: push nothing, continue
straight into the child, and let the next backtrack find the stack
frame simply isn't there instead of popping an empty one. Implemented
in all four analogous spots (`query_rec`'s and `query_rec_restricted`'s
initial-descent and backtrack-resume branches), guarded so the one
case that still needs the skipped span (`pos_span_end`, read only when
`level + 1 == unconstrained_from`, for the bulk-listing fast path) is
left untouched.

Doing this correctly required adding an explicit `level` field to
`ConstrainedFrame`/`RestrictedFrame`: both backtrack loops used to
derive the top frame's tree level from stack depth alone
(`start_level + stack_len - 1`), which only holds when every descended
level pushes exactly one frame. Once some levels stop pushing a frame,
stack depth and tree level drift apart, and the old formula silently
computed the wrong level for every frame above the first compressed
one -- caught by the full test suite (11 failures, all "missing a
match" or an out-of-bounds slice read), not by reasoning alone. Fixed
by storing each frame's own level explicitly rather than deriving it.

**Measured, A/B, same methodology as every other change in this
document** (symmetric 500,000x200x500,000 at 10,000,000 triples for
two-Exact/one-Exact/Range, asymmetric 20,000x200x2,000,000 at
3,000,000 triples for S-bound/O-bound; `n=5000` randomized queries per
shape, same build held fixed, swapped code and rebuilt between runs,
repeated twice for the "after" case to rule out noise):

```
                      before (ns/call)   after (ns/call)   after (repeat)
two-Exact             24,333.6           24,974.1          24,678.8
one-Exact             1,696,590.2        1,723,991.7       1,717,317.7
Range                 85,871.7           88,259.2          87,436.0
asym S-bound          948,963.9         962,781.9          959,560.3
asym O-bound          964,923.4         983,293.3          982,859.2
```

Every shape came back 1.5-2.8% *slower*, not faster, and the two
"after" runs agree closely with each other -- a small but real and
reproducible regression, not noise. The likely cause: the extra
branch (`frame_remaining == 0 && level + 1 != unconstrained_from`),
plus the wider frame struct (the new `level` field), runs on *every*
matched candidate, not just the ones where compression actually
triggers -- so the cost is paid everywhere while the saved
write-then-immediately-discard is only avoided in the sparse,
near-leaf levels where it was already cheap relative to the rest of
the per-node work. The change was reverted in full (confirmed via a
clean `diff` against the pre-change backup and a green full test
suite afterward); nothing from this attempt shipped.

### Why isn't every query O(height)? Point lookups are; multi-candidate ones aren't

The structure's headline promise is reaching a leaf in O(height). `contains`
(a single real path per level, no masking, no backtracking) measures 356-422ns
on a height=7, 1,000,000-triple tree -- about 54ns/level -- confirming the
promise holds for true point lookups. `restricted_descent_visit`'s multi-
candidate shapes (`two-Exact`, `one-Exact`, etc.) cost far more: hundreds of
nanoseconds to low microseconds per call on the same tree, growing with result
count.

`perf record -F 999 --call-graph dwarf` plus `perf report`/`perf annotate`
against a release build (a two-Exact-dominated workload) gave a flat symbol
breakdown of `restricted_descent` (query_rec, inlined) 45.61%, `far_span_sum`
13.00%, `advance_past_children` 7.35%, `advance` 5.27%, `BlockIndex::locate`
4.61%, `contains_batch` 2.17%, `contains` 1.81% (the rest is tree-build noise
from the same binary). Annotated disassembly isolated one instruction --
a `cmp` immediately after the load of a `Vec<Vec<u8>>`-backed block -- eating
66% of samples alone: a classic load-stall misattribution, not a real hotspot
in itself; the surrounding function (`far_span_sum`/`eager_child_span_end`)
is the real cost center.

To separate "necessary branching work" from "wasted bookkeeping," a
diagnostic-only prototype (never shipped, `#[allow(dead_code)]`, never
correctness-checked beyond proving it was wrong) skipped all eager-span
bookkeeping unconditionally -- always took the first candidate, never
backtracked. It measured 280.4ns, matching `contains`, but undercounted
results by about 98% (5,249 vs. the correct 315,011). The "speedup" was
from skipping real, required work, not from eliminating waste.

Conclusion: the O(height) promise is real but applies to the *descent*, not
to *enumeration*. A query that can match several candidates at a level still
has to visit every one of them and everything reachable under each, and the
cost of that visit is proportional to how much real branching survives before
every dimension either pins down to one path or goes fully free (the
`unconstrained_from` level, where `bulk_list_subtree` takes over). For the
shapes tested on this dataset (two-Exact, one-Exact, S-bound, O-bound variants
on a 1,000,000-triple / 100,000x200x100,000 tree), `unconstrained_from`
equals `height` every time -- the bulk fast path is never reached at all,
so the entire descent runs through the expensive, masked, node-by-node phase.
This is inherent to how bits interleave across all three dimensions per level
in the build schedule (a free dimension keeps real branching alive for
several levels even once another dimension is fully pinned), not an
implementation accident -- confirmed by the next entry.

### Lazy eager-skip, correctly conditioned this time: still no real win

The "Runtime path compression" entry above tried skipping the eager span/stack-frame bookkeeping whenever
`remaining == 0` and measured a real regression, traced to a wrong trigger
condition changing behavior on every candidate instead of only the rare ones
where skipping was actually possible. The correct condition is narrower:
skip only when, after consuming the current candidate, no *matching* sibling
remains (`remaining_after & mask == 0`, not `remaining_after == 0` -- a real
sibling that will never match this query's mask is not a reason to keep
bookkeeping alive) *and* the next level doesn't need the skipped span for the
bulk-listing handoff (`level + 1 < unconstrained_from`).

Implemented as a separate, diagnostic-only prototype (`LazyFrame`, adding the
explicit `level` field the first attempt needed, same structural reason:
stack-depth-equals-tree-level stops holding once some levels stop pushing a
frame) and measured A/B against plain eager on two datasets -- sparse
(1,000,000 triples, 100,000x200x100,000, height=7) and dense (2,000,000
triples, 500x50x500, height=4) -- across two-Exact/one-Exact/P-pinned-emit-S
shapes. Result: no real improvement, noise-level (~1-2%) either direction in
every case. The correct trigger condition is simply rare for these shapes,
not common -- the previous entry's bit-interleaving explanation predicts
exactly this: a free dimension keeps a real matching sibling alive at most
levels, so there's rarely nothing left to skip. This independently confirms
`eager_child_span_end`'s bookkeeping is genuinely necessary almost every level
for these shapes, not wasted work -- consistent with the DFS k²-tree
literature (Carmona & Manzini's "Enriched DF" design stores subtree sizes
eagerly, for exactly the same reason: a DFS/preorder layout has no free
random access the way a BFS/level-order one does, so recovering fast
navigation requires paying this cost up front). Not landed; both prototypes
removed after measuring.

### `query_rec`/`query_rec_restricted` rewritten as real recursion: landed

Both engines used an explicit `[ConstrainedFrame; MAX_HEIGHT]` (or
`RestrictedFrame`) stack array with a manual backtrack loop, predating this
investigation. Restructured into genuine function-call recursion instead: a
local `loop` still handles same-level sibling iteration (preserving the
`skip_mask`/`advance_past_children` O(1) batch-skip exactly as before), but
a match now descends via a direct recursive call instead of pushing a frame
and looping back to the top. `eager_child_span_end` and its necessity are
unchanged by this -- this is a control-flow restructuring, not a new
algorithm. The only structural change needed: `pos_span_end` (a subtree's own
end position, needed only once `level >= unconstrained_from`, for the bulk-
listing handoff) is now a parameter threaded through each recursive call
instead of a loop-local variable, since under hoisting a far node's own span
lives in its *parent's* record -- by the time a call is sitting at `pos`
there's nothing left there to compute it from.

Measured A/B on the real, landed code (not a prototype): built once, measured
the pre-rewrite iterative version, then the post-rewrite recursive version,
both release builds, same tree (height=7, 1,000,000 triples,
100,000x200x100,000), `n=3000` randomized queries per shape:

```
              iterative (ns/call)   recursive (ns/call)   speedup
two-Exact      6,974.7               6,210.2               1.12x
one-Exact      245,474.0             218,716.2             1.12x
S-bound        374,723.3             366,754.1             1.02x
O-bound        371,101.5             365,070.3             1.017x
```

A real, reproducible, correctness-preserving win (confirmed by the full test
suite, including the `Range`/`Multi` tests that exercise
`query_rec_restricted` directly) -- 1-12% depending on shape, with no shape
regressing. `ConstrainedFrame`/`RestrictedFrame` were removed; nothing else
about the descent (masking, skip-batching, the bulk-listing handoff) changed.

### Rayon fan-out + bulk-mask decode combined: a small win in most shapes, not landed

A prototype combined two independent levers tried separately elsewhere in
this document: a rayon fan-out across the first `parallel_depth` tree levels
(collecting a frontier of subtrees, each then searched independently), and a
generalized bulk-decode-with-masking pass per frontier item instead of a
plain `query_rec` call. Measured against plain rayon fan-out (same frontier,
plain `query_rec` per branch) on the sparse dataset, `n=3000` per shape,
repeated twice:

```
              rayon alone (ns/call)   rayon+bulk-mask (ns/call)   speedup
two-Exact      27,497-28,183           29,164-29,278               0.94-0.96x
one-Exact      1,806,355-1,808,237      1,719,164-1,733,097         1.04-1.05x
S-bound        997,517-999,118          840,257-939,418             1.06-1.19x
O-bound        999,118-1,002,708        939,328-939,418             1.06-1.07x
```

(Numbers above are from two independent runs; see git history of this
section for the exact per-run pairing.) A real, small improvement in three of
four shapes (4-19%), but `two-Exact` regresses slightly (4-6% worse) -- its
per-branch work is cheap enough that the extra dispatch for bulk masking
costs more than it saves. Caught one correctness bug during development: the
first version of the frontier collector didn't thread `pos_span_end` through
its own recursion, so a frontier item landing exactly at `unconstrained_from`
on a non-root branch got the whole tree's span instead of its own subtree's
-- fixed by threading it the same way the recursive rewrite above does.
Not landed -- the win is real but modest, and inconsistent across shapes;
revisit if `two-Exact`-shaped queries aren't the common case in practice.

### Rayon fan-out depth sweep, and resolving the "5x vs. 2x" discrepancy

An earlier rayon fan-out measurement in this project reported 5.56x
(one-Exact) and 5.74x (O-bound) speedups at `parallel_depth=2`. A later
re-measurement on what was assumed to be the same setup found only 2.2-3.3x,
and initial hypotheses (parallel_depth itself being wrong, CPU contention)
were tested directly and ruled out: sweeping `parallel_depth` from 1 to 5 on
the smaller/cheaper dataset (height=7, 1,000,000 triples, serial one-Exact
cost 260,957ns/call) gave depth=1: 3.29x, depth=2: 2.44x, depth=3: 1.01x,
depth=4: 0.49x, depth=5: 0.61x -- depth=1 is best, depth=2 is already past
the sweet spot, and depth=3+ collapses as the frontier balloons (2,449+
near-empty tasks at depth=3, dispatch overhead dominating real work). A
second, heavier shape (P-pinned, emit S; serial cost 457,896ns/call) showed
the same shape-of-curve, with depth=4's collapse becoming catastrophic:
one call took 8.47 *seconds* (frontier ballooning past what the thread pool
can usefully dispatch). `nproc`=12 and `/proc/loadavg` 0.12-0.69 ruled out
contention. None of this explained the gap -- both runs used depth=2, and
neither was anywhere near 5x.

Recovered the original benchmark's raw output (same four shapes, same
`parallel_depth=2`, same machine) and compared baselines directly: its
serial one-Exact cost was 1,693,182ns/call -- about 6.5x higher than the
260,957ns/call baseline in the re-measurement. Every shape's absolute serial
cost was correspondingly higher (`S-bound` 946,912ns vs.
later runs' sub-ms costs, `two-Exact` 24,436ns -- still cheap, and tellingly
still only 1.44x at depth=2, consistent with the pattern below), and its
speedups were uniformly higher: `two-Exact` 1.44x, `one-Exact` 5.56x,
`S-bound` 3.83x, `O-bound` 5.74x. The gap was never about depth or
contention: rayon's dispatch overhead per spawned task is roughly fixed, so
the measured speedup ratio depends on how much real per-branch work there is
to amortize it against. The original benchmark's dataset/domain parameters
happened to make every query shape's per-call cost much higher, which is
exactly the regime where a fixed dispatch cost stops mattering and the
parallel speedup approaches the branch count. The cheaper re-measurement
dataset wasn't a bug or a wrong depth -- it was just a smaller, cheaper
workload, where dispatch overhead eats a proportionally larger share of the
gain. `two-Exact`'s persistently low speedup (1.44x even in the "5x" run)
confirms this: it's the cheapest shape in both runs, and the cheapest shape
never gets close to 5x regardless of dataset.

### `one-Exact`'s batched-skip plan, resumed: instrumented, and the premise didn't hold

An earlier plan hypothesized that `one-Exact`'s much higher cost than
`two-Exact` traces to `advance_past_children` making many independent,
expensive `BlockIndex::locate` calls while skipping past non-matching real
siblings, and proposed instrumenting the real distribution (bit count per
skip, fast-vs-slow `advance`, per level) before designing a batched-locate
primitive to fix it.

**Phase 0, instrumented for real** (temporary per-level counters around
`advance_past_children`'s far and near branches, tallying bits skipped per
call and classifying the resulting `advance` as fast or slow; removed after
collecting the data below). Run on a 1,000,000-triple, 100,000x200x100,000
tree, height=7 -- a smaller scale than the plan's original 2,000,000/
20,000,000/100,000,000-triple ask, because this host's available memory
(consistently under 1 GiB free, swap often near full throughout this
session) can't safely build the larger trees:

```
two-Exact (n=3000, total_results=3153)
level=1 far  calls=2699    avg_bits/call=22.91  advance(fast=0,    slow=2699)  avg_slow_span_blocks=202.5
level=2 far  calls=11136   avg_bits/call=25.19  advance(fast=1893, slow=9243)  avg_slow_span_blocks=6.2
level=3 far  calls=36252   avg_bits/call=29.05  advance(fast=32712,slow=3540)  avg_slow_span_blocks=1.0
level=4 far  calls=51856   avg_bits/call=4.36   advance(fast=51757,slow=99)    avg_slow_span_blocks=1.0
level=5 near calls=707     avg_bits/call=1.03   advance(fast=731,  slow=0)

one-Exact (n=3000, total_results=33003)
level=1 far  calls=2029     avg_bits/call=24.45  advance(fast=0,      slow=2029)  avg_slow_span_blocks=204.8
level=2 far  calls=35264    avg_bits/call=21.10  advance(fast=6938,   slow=28326) avg_slow_span_blocks=5.2
level=3 far  calls=376025   avg_bits/call=29.38  advance(fast=338902, slow=37123) avg_slow_span_blocks=1.0
level=4 far  calls=4553127  avg_bits/call=4.08   advance(fast=4546462,slow=6665)  avg_slow_span_blocks=1.0
level=5 near calls=105432   avg_bits/call=1.04   advance(fast=109707, slow=15)
```

This refutes the plan's premise. The only level where a slow `advance` spans
many blocks (level 1, ~200-300 blocks) fires at most once per query on
average (2,029-2,699 times across 3,000 queries) -- there's nothing to
*batch* there, since it's already exactly one `locate` call per occurrence,
not several. Everywhere else, a slow `advance` spans essentially one block
(avg 1.0), meaning `BlockIndex::locate`'s O(log blocks) search is already as
cheap as it can be when it fires at all -- there's no pattern of "many
independently expensive locate calls" for a shared-search primitive to
exploit. What actually separates `one-Exact` from `two-Exact` is sheer call
*volume*: at level 4, `one-Exact` makes 4,553,127 skip-batch calls against
`two-Exact`'s 51,856 -- 88x more, well beyond `one-Exact`'s 10x larger result
count (33,003 vs. 3,153). Every one of those calls is already O(1) amortized
(one `far_span_sum` plus one `advance`, regardless of how many siblings it
skips) -- the gap is in how many such O(1) operations are needed, which is
exactly the "a free dimension keeps real branching alive for several levels"
mechanism identified earlier in this document, now with hard numbers behind
it, not a `BlockIndex`/`advance_past_children` inefficiency to fix.

**Phase 1** (replacing `query_rec`'s bit-by-bit mask check with the
`remaining & mask` batched-candidate form) turned out to already be landed,
predating this investigation -- confirmed by reading the current code, not
assumed. `count_rec` (cardinality estimation) still walks its real siblings
one at a time, but it's off the `one-Exact`/`restricted_descent` path this
plan was chasing, and Phase 0's data gave no reason to believe it matters
enough there to justify changing it; left alone.

**Phase 2, tried anyway on a data-grounded hypothesis.** Since the real
per-call cost couldn't be in `locate`, the next candidate was `far_span_sum`
itself: at the levels with the most calls, it's also doing `count` (avg
4-30) independent `block_read_var` calls, each paying its own
`self.blocks[block]` double-indirection (`Vec<Vec<u8>>`) even though every
entry lives in the same contiguous span. Rewrote it to one `bulk_read` of
the whole span plus a sequential in-memory decode. Measured A/B on the same
dataset, `n=3000`, two runs:

```
              before (ns/call)   after (ns/call)   change
two-Exact      6,210.2            6,545.9           +5.4%  (worse)
one-Exact      218,716.2          248,182.2         +13.5% (worse)
S-bound        366,754.1          398,376.1         +8.6%  (worse)
O-bound        365,070.3          396,465.3         +8.6%  (worse)
```

A real, reproducible regression, not noise -- the same shape of surprise as
the earlier `Pos.abs` attempt this document already records. Likely cause:
`bulk_read`'s `Cow<[u8]>` return (a branching enum, sized for the
cross-block-copy case even though the common case never takes it) and the
generic trait dispatch through `self.storage.bulk_read` cost more, for these
small `count`s, than the handful of redundant indexing operations it was
meant to save. Reverted in full (confirmed via `grep` and a clean full test
suite afterward); nothing from Phase 2 shipped.

**Conclusion: this plan is answered, not abandoned.** The instrumented data
doesn't support the batched-`locate` primitive the plan set out to build,
in either of its two envisioned forms -- the "many small skips" case
already degenerates to O(1)-cost operations, and the "few large jumps" case
already makes exactly one `locate` call per occurrence. The real driver is
call volume from `P` staying fully free, which is inherent to the query
shape (see the O(height) entry above), not a missed batching opportunity.
No further work is planned here unless a future profiling pass surfaces a
different mechanism.

### Subtree deduplication (Carmona & Manzini-style): prototyped, measured, kept

Carmona & Manzini's "Depth First Representations of k²-trees"
(arXiv:2505.11302) and its follow-up "Extended Depth-First Representations
of k²-trees" (arXiv:2607.28136) propose storing a k²-tree in DFS/preorder
specifically because it makes identical subtrees contiguous, equal
substrings -- detectable with a linear-time suffix-array/LCP scan -- so
every repeat beyond the first occurrence can become a pointer instead of a
second copy. This crate's tree is already DFS/preorder-laid-out (that
precondition already holds), and relative, not absolute, skip/span
encoding (`RecordSink::append_subtree`'s own doc comment: "every skip/span
value inside bytes is a relative byte distance... so the bytes are fully
relocatable") means two structurally identical subtrees would, in
principle, already be byte-for-byte identical wherever they occur -- so
the question was purely empirical: how much of this tree's actual record
bytes, for this crate's RDF-triple workloads, are spent on subtrees that
recur elsewhere at the same level?

**Prototype.** `RadixTreeDfsInterleavedVarSkipUnrolledGeneric::subtree_dedup_report`
(`src/tree.rs`) walks the live tree in the same DFS order `dfs_records`
already uses, computing a 64-bit Merkle-style hash per node (own
existence word, folded with each real child's hash, in `qi` order) and
grouping by `(level, hash)` -- never across levels, since this crate's
per-level bit schedule means a level's record encoding depends on
absolute level, not relative depth. A diagnostic companion,
`examples/subtree_dedup_proto.rs`, builds a tree for a given dataset shape
and reports, per level: node count, distinct-hash count, and both an
*upper-bound* saving (if every repeat were costless to point to) and a
*net* saving using a second, more honest estimate -- a back-reference
costs `ceil(log2(distinct_count))` bits, not zero.

**Unlike every other prototype in this document, this one was kept on
request rather than deleted after measuring** -- the user asked for it to
remain in the repository so a follow-up idea can be benchmarked against
the same baseline. `subtree_dedup_report`/`SubtreeDedupReport`/
`SubtreeDedupLevelReport` are real `pub` surface on the tree type, clearly
doc-commented as diagnostic-only and not wired into storage, serialization,
or any query/insert/remove path.

**Measured** (same two domains used throughout this document -- symmetric
500,000x200x500,000 at 10,000,000 triples, asymmetric 20,000x200x2,000,000
at 3,000,000 triples -- plus a `zipfian_entities(1.5, 1.5, 1.5)` high-skew
variant of each, height=8 in every case):

```
                                       upper bound    net-of-pointer-cost
uniform (symmetric)                   74.27% saved    55.52% saved
zipfian predicates (symmetric)        78.56% saved    59.52% saved
zipfian S+P+O (symmetric, skewed)     78.70% saved    59.04% saved
uniform (asymmetric)                  76.03% saved    59.25% saved
zipfian S+P+O (asymmetric, skewed)    82.96% saved    69.88% saved
```

Headline numbers look large, but the per-level breakdown (also printed by
the example; see its output for the full table) shows where they actually
come from, and it's not where the paper's win comes from. For the uniform
symmetric case: levels 0-3 (the tree's "real" structural levels, closest
to the root) show **zero** redundancy -- every node's existence word is
already distinct, `distinct_count == node_count`. Level 4 shows modest
redundancy (24% fewer distinct than total). The dominant contribution to
the headline percentage is levels 5-7 (the three levels closest to the
leaves), which hold 239 MB of the tree's 343 MB total and show 87-100%
"savings" apparently -- but a leaf-level record is already just one
8-byte existence word with very low popcount (often 1-2 bits set out of
64), so with only a few hundred thousand to a few million *possible*
low-popcount words and tens of millions of leaf nodes, most of that
"redundancy" is pigeonhole coincidence between semantically unrelated
parts of the domain, not the kind of real, repeated-structure-from-
real-world-topology the paper measures on web-graph adjacency matrices
(where many pages genuinely share the same small out-link set). This
crate's `zipfian_entities` skew (the closest analog available to "some
nodes look alike because the underlying graph has repeated structure")
moved the needle on levels 4-6 (24-57% net saving there, vs. 0-25% in the
uniform case) but still left the upper levels at zero and didn't change
the overall picture much, suggesting RDF-triple data in these shapes
doesn't have the same kind of large-subtree structural repetition
web-graph adjacency matrices do -- the paper's own technique, applied
faithfully, but to a workload shape it wasn't designed for.

**A second, independent reason not to build this even if the bytes were
real:** every prior win in this document came from *removing* indirection
(hoisting spans into the parent, bigger blocks, fewer `BlockIndex::locate`
calls), and every attempt to *add* a layer of indirection for some other
benefit measured as a regression (`Pos.abs`, the bulk-read `far_span_sum`
rewrite above) specifically because an extra pointer hop costs more, in
this crate's read-dominated, latency-measured-in-nanoseconds workload,
than the bytes it saves are worth. A real subtree-dedup implementation
would need exactly that: an extra dereference (and, on `File`/
`Compressed`, potentially another `BlockIndex` lookup) on every touch of a
deduplicated subtree during descent. This document has not measured that
cost directly -- it wasn't reached, since the byte-saving side of the
tradeoff already looked weak once pointer cost was priced in -- but it is
a real cost this technique would add on top of an already-unfavorable
space story, not a free lunch even in the scenario where the space saving
were larger.

**Status: not built, prototype kept per explicit request** (not reverted,
unlike every other negative result in this document) so a different idea
can be measured against the same `subtree_dedup_report` baseline.

#### Follow-up: does finer (bit-level, not 64-way) granularity find hidden sharing?

A fair objection to the measurement above: `subtree_dedup_report` only ever
merges whole pre-grouped 64-way digits (up to `BASE`=6 bits, across
however many dimensions are active, combined into one node before any
comparison happens). A true bit-level trie (1-bit branching) is a strict
refinement -- two prefixes that disagree somewhere inside one of today's
6-bit digits might still share a run of bits *before* the point of
disagreement, which an all-or-nothing whole-digit comparison can never
see. If real, that would mean the measured 0% redundancy in the upper/
middle levels is an artifact of the chunking, not a fact about the data.

`examples/bitlevel_dedup_proto.rs` (kept, same as `subtree_dedup_proto.rs`)
tests this directly: it duplicates `total_bits_needed`/`dynamic_schedule`'s
exact per-level, per-dimension bit allocation, un-groups it to individual
bits in the same order, and builds a real bit-level trie from the sorted
keys via recursive partitioning, merging states bottom-up by hash exactly
as `subtree_dedup_rec` does -- the *correct* Myhill-Nerode criterion
(two prefixes merge only if their entire sets of remaining continuations
match, not just some element of one overlapping some element of the
other; a first attempt at this file used a cheaper `HashSet`-over-trailing-
bit-values shortcut that implicitly assumes every prefix has at most one
key below it, which is false here and gave a wrong, overstated number --
caught by it disagreeing with the real tree's measured counts, fixed by
building the actual trie instead of shortcutting it).

Result, on the same small symmetric case (`5,000x200x5,000`, 100,000
triples) the full-scale run above also covers: at every real level
boundary, the bit-level merged-state count matches
`subtree_dedup_report`'s `distinct_count` **exactly** -- 50,968, 2,033,
and 30 at levels 3, 4, and 5 respectively, bit for bit. Printing every
individual bit depth in between (not just the boundaries) shows why:
within a level's span, the merged-state count only ever gets *larger*
(less merged) than the boundary value, converging back down to the exact
same minimum by the next boundary -- never lower. There is no hidden
sharing the coarse grouping is discarding.

This isn't a coincidence specific to this dataset -- it follows from
Myhill-Nerode directly: the number of right-language equivalence classes
at a fixed cut depth is a property of the *data*'s remaining bit values
from that point on, not of how many symbols you group together to walk
there. Grouping 6 bits into one 64-way digit and spending 6 separate
1-bit levels on the same 6 bits produce identical equivalence classes at
the point where both have consumed exactly the same bits, because
"two prefixes have the same right-language" doesn't care what alphabet
you used to get there. **This closes the follow-up: the original
per-level numbers already are the true minimal bound, not an
undercount** -- confirmed by direct construction, not just restated
theory. No further work planned on the compression half of this idea.

### Path FST prototype: a real minimized DAG, `contains` only, static build

The [FST literature review](FST_LITERATURE_REVIEW.md) maps the layout,
dedup, and batched-traversal literature to this section and everything
under it; not repeated inline here.

The two follow-ups above left two open objections to actually building
subtree deduplication: (1) a real back-reference needs to be
distinguishable from an ordinary record inline, costing more than the
bare `ceil(log2(distinct))`-bit estimate `subtree_dedup_report` used; (2)
every prior win in this document came from removing indirection, and
every attempt to add it (`Pos.abs`, the bulk-read `far_span_sum` rewrite)
measured as a regression. Both turned out to have real prior art one
repository over: `../s-dict` (a from-scratch byte-packed FST term
dictionary, `PackedAutomaton`) already builds and ships exactly this
kind of structure. Its record format answers (1) directly -- every
transition already needs a child-pointer field (a backward LEB128 delta,
`field_start - target`, with a `NEXT` flag eliding it to zero bytes for
the common adjacent-child case) regardless of dedup; pointing that same
field at an earlier, structurally-identical state costs nothing beyond
a possibly-larger delta value. `s-dict`'s own DESIGN.md also already ran
into (2)'s mutation concern directly (a mutable, register-deduplicated
automaton bloated 20-37x within ~200 inserts on an 800-term base,
because every insert rewrites every ancestor on its root-to-leaf path)
and already has a shipped fix: don't keep dedup exact through every
mutation, just bound the damage and periodically `compact` (full
bottom-up reachable-only re-walk, exact hash-merge) -- a pattern
`s-graph`'s own `shrink_to_fit`/`dfs_records` already has the hook for.

**Built, not just reasoned about.** `src/path_fst.rs` (`PathFst`) is a
real, from-scratch minimized DAG over this crate's own per-level bit
schedule (`total_bits_needed`/`dynamic_schedule`/`compute_qi`/
`advance_lo`, reused directly from `tree`, not reimplemented, so the
comparison against the production `Heap` tree is exact-schedule
apples-to-apples). Scope, deliberately narrowed per explicit instruction:
**static build only, `contains` only** -- no `insert`/`remove` yet, so
the mutation-vs-dedup tradeoff above is set aside for now, not solved.

Build: the same recursive partition-by-digit algorithm as `build_rec`,
postorder, with an exact (not hashed) register per tree level --
`HashMap<(existence_word: u64, child_addrs: Vec<u32>), address>` -- so
two subtrees merge only on true value equality (no birthday-bound false
merges, unlike the measurement-only `subtree_dedup_report`). One register
*per level*, not a single global one: nothing in principle stops two
different levels' schedules from producing an identical `(word,
child_addrs)` tuple by coincidence for some future domain/schedule shape,
and a state built for level `L+1`'s decode rules must never be reused as
if it were level `L'+1`'s -- made impossible by construction instead of
relying on it being empirically unlikely for the domains tested so far.
Record format (first pass, simplified): `[existence_word: 8
bytes][delta: u32]*`, one fixed-width delta per real child, `qi`-
ascending -- no LEB128/NEXT-elision packing yet (a real version would
want it; `subtree_dedup_report`'s own "net" column already estimated
what that saves).

**Correctness**: `examples/path_fst_proto.rs` checks `PathFst::contains`
against both the production `Heap` tree and a naive `HashSet` ground
truth -- every real triple, plus 20,000 random in- and out-of-domain
probes, per dataset shape. Zero mismatches across four shapes (100K and
10M/3M triples, symmetric and asymmetric, uniform and high-skew
`zipfian_entities`) -- 320,000 probes total, fully agreeing with a built-
from-scratch register-per-level merge against two independent oracles.

**Measured, `perf`-naive wall-clock (not yet `perf stat`-rigorous --
a first signal, to be tightened if this goes further), same domains used
throughout this document**:

```
                                    Heap mem    PathFst mem   saved    Heap contains   PathFst contains
uniform, symmetric, 100K            2.17 MB      1.05 MB      51.5%     161.0 ns          55.0 ns  (0.34x)
uniform, symmetric, 10M           343.12 MB    105.64 MB      69.2%     581.4 ns         343.2 ns  (0.59x)
zipfian S+P+O, symmetric, 10M        5.74 MB      3.28 MB      42.9%      90.0 ns          47.5 ns  (0.53x)
uniform, asymmetric, 3M           109.24 MB     30.03 MB      72.5%     429.8 ns         247.1 ns  (0.57x)
```

A real, fixed-4-byte-pointer implementation beats even the earlier
"net-of-pointer-cost" *estimate* (55-60%) on three of four shapes --
69-72% actual savings on the two uniform shapes, somewhat less (42.9%)
on the high-skew case (less raw redundancy to exploit there in the first
place, consistent with `subtree_dedup_report`'s own per-shape spread).
**`contains` is also faster, not just smaller -- 1.7-2.9x, holding at
both the 100K and 10M/3M scales**, the opposite of this document's prior
pattern (every previous attempt to add indirection for some other
benefit measured as a regression). The likely reason this one doesn't
follow that pattern: `PathFst`'s read path is *simpler* per level than
the production tree's, not more complex -- a direct rank-select into a
fixed-width array and one subtraction, with no `BlockIndex::locate`, no
block-boundary check, no `far_span_sum`, because there is no block
structure at all (one flat `Vec<u8>`, a direct consequence of skipping
insert/remove, not of dedup itself) and no forward-skip-past-a-sibling
logic to maintain (a DAG's children are reached by direct address, not
by skipping past an inline predecessor). Some of this win is therefore
"what a read-only, block-free structure costs" rather than "what dedup
buys" -- not yet decomposed into those two pieces, an open question if
this goes further, not a settled attribution.

### Blocking it: isolating "dedup" from "no block structure"

The open question right above -- is the speedup from dedup, or just from
having no block-indirection hop at all -- is directly testable: put
`PathFst` on the exact same block-chopped storage the production tree
uses, and see whether it still wins. Rebuilt `PathFst` on a real
[`HeapStorage`]/[`BlockStorage`] (same `append_record`/`block_capacity`
contract `Heap` itself uses, same default `block_capacity` of 65,536),
with each child reference now a direct `Pos` (block, offset) packed into
one `u64`, instead of the first version's flat-buffer backward
byte-delta -- a delta stops being a single scalar once "backward" can
cross block boundaries in either direction, so this is the natural
representation once blocking is real, not an arbitrary choice. Costs 8
bytes per child reference instead of 4, and needs no `BlockIndex` at
all: a child's `Pos` is already fully resolved (postorder) by the time
its parent is written, so there's nothing to compute a delta *from*.

**Re-measured, same four shapes, same methodology, both structures now
on identical storage:**

```
                                    Heap mem    PathFst mem   saved    Heap blocks  PathFst blocks   Heap contains   PathFst contains
uniform, symmetric, 100K            2.17 MB      1.67 MB      22.9%        34             26            177.4 ns          73.0 ns  (0.41x)
uniform, symmetric, 10M           343.12 MB    167.58 MB      51.2%      5234           2559            588.2 ns         443.9 ns  (0.75x)
zipfian S+P+O, symmetric, 10M        5.74 MB      5.53 MB       3.6%        88             85             92.1 ns          58.3 ns  (0.63x)
uniform, asymmetric, 3M           109.24 MB     48.05 MB      56.0%      1667            734            437.5 ns         324.2 ns  (0.74x)
```

Correctness unchanged: same 320,000-probe check (real triples + random
in/out-of-domain, against both `Heap` and a `HashSet` ground truth),
zero mismatches, now against the blocked version. **`PathFst` still
wins on both memory and speed in every shape, with the exact same kind
of block-indirection hop on both sides of the comparison now.** Memory
savings are smaller than the unblocked version's (expected -- 8-byte
`Pos` references cost twice the first prototype's 4-byte deltas; the
high-skew shape in particular drops to 3.6%, since it had the least raw
redundancy to begin with per `subtree_dedup_report`, and the wider
pointer eats most of what there was). Speed is also closer than the
unblocked version's 0.34-0.59x (now 0.41-0.75x) -- consistent with part
of the earlier win genuinely being "no block structure," now that both
sides pay for one. But the remaining gap doesn't vanish, and both block
*counts* are themselves meaningfully lower for `PathFst` (fewer blocks
to allocate, touch, and -- on `File`/`Compressed` -- page in), not just
raw bytes. Per the user's own framing going into this: this crate has
already spent considerable effort optimizing the DFS tree's blocked
storage path specifically (block capacity tuning, hoisting, the
recursive rewrite, O(1) sibling-skip); a blocked `PathFst` that still
doesn't lose to that already-tuned baseline is reasonably strong
evidence of a real structural advantage, not an artifact of comparing
against something less optimized.

### `perf stat`/`perf record`-rigorous numbers, and a second delta variant that beats everything

Two follow-ups, requested together: get real `perf` numbers instead of
wall-clock timing, and evaluate whether the compact 4-byte delta from
the first prototype could be kept *and* resolved in `O(1)` instead of
via `BlockIndex`'s `O(log blocks)` Fenwick walk. The second one turned
out to resolve the first one's open question too.

**`perf` methodology**: `examples/path_fst_perf.rs` isolates one
structure's `contains()` loop per process invocation (so
`perf record`'s call graph attributes to one structure, not a mix), and
a `[profile.profiling]` (`inherits = "release", debug = true`) was added
to `Cargo.toml` for symbols. Build cost dominates a short run (a 10M-
triple build takes seconds; the loop itself is sub-second at low
iteration counts), so every number below is **baseline-subtracted**:
each structure is run twice, once with the real iteration count and
once with `iters=0` (build only, loop skipped entirely), and the
build-only counters are subtracted out before dividing by call count --
isolating the loop's own cost, not a mix of build and read.

**`perf stat -e cycles,instructions,cache-references,cache-misses`**,
20,000,000 calls, same symmetric 10M-triple domain used throughout this
document, baseline-subtracted:

```
                          cycles/call   instructions/call   cache-refs/call   cache-misses/call
Heap                        2433.9           2740.8              50.6              15.9
PathFst (Pos, 8B)            1901.3            988.4              33.6              15.5
PathFstDelta (4B+BlockIndex) 2297.4           1794.3              38.9              14.1
```

This is the rigorous version of the wall-clock finding already in this
document: `PathFstDelta` (the compact 4-byte version, `BlockIndex::locate`
resolving each delta) has *fewer* cache misses than either alternative
(the smaller footprint genuinely helps) but pays nearly 800 more
instructions/call than the 8-byte `Pos` version -- confirmed directly,
not just inferred, by `perf record -F 999 --call-graph dwarf` +
`perf annotate`: `BlockIndex::prefix`/`locate` are fully inlined into
`contains`, and the disassembly shows the actual Fenwick binary-search
loop (`bsr` for `leading_zeros`, then `while blocks != 0 { sum +=
self.sums[blocks - 1]; blocks &= blocks - 1; }`) running on every
non-leaf hop. This is also *why* the earlier wall-clock numbers showed
`PathFstDelta` losing to `Heap` outright on the high-skew shape (118.0
ns vs 90.0 ns, i.e. 1.31x *slower*) -- the Fenwick walk's fixed per-hop
cost doesn't scale down with the tree, so on a tree small and fast
enough that `Heap` itself isn't cache-miss-bound, the extra instructions
are pure loss with nothing to buy them back.

**The fix: `PathFstDeltaPow2`, a second delta variant that doesn't need
`BlockIndex` at all.** `HeapStorage::append_record` never lets a block
exceed `block_capacity` -- so if `block_capacity` is a power of two, the
numbering `virtual_addr = block * block_capacity + offset` has no
collisions and inverts in one shift and one mask (`block = addr >>
capacity_bits`, `offset = addr & capacity_mask`), genuinely `O(1)`, not
just asymptotically cheaper than the Fenwick walk. The "wasted" tail
bytes a block leaves unused when the next record doesn't fit are priced
into the numeric *value* of a delta, not into any stored byte -- the
underlying `Vec<Vec<u8>>` is unaffected, still holding only real record
bytes. Building can no longer predict its own resulting address before
calling `append_record` (unlike the running-byte-counter trick the
`BlockIndex` variant uses, which only works because `HeapStorage` never
pads): it appends a placeholder record with zeroed deltas, reads back
the real `Pos` `append_record` actually landed on, computes this node's
own virtual address from it, then backfills every child's delta via
`patch_appended` -- the identical mechanism `tree::build_rec` already
uses for its own forward-skip fields, here for the opposite reason (not
knowing this node's own future address, rather than a child's future
size).

**Confirmed in the disassembly, not just by reasoning**: `perf annotate`
on `PathFstDeltaPow2::contains` shows `shr %cl,%rax` (the `block = addr
>> capacity_bits` shift) directly in the hot path, and no Fenwick-walk
loop anywhere in the function -- the mechanism compiled exactly as
intended.

**Baseline-subtracted `perf stat`, same methodology, all four
structures**:

```
                                cycles/call   instructions/call   cache-refs/call   cache-misses/call
Heap                             2433.9           2740.8              50.6              15.9
PathFst (Pos, 8B)                 1901.3            988.4              33.6              15.5
PathFstDelta (4B+BlockIndex)      2297.4           1794.3              38.9              14.1
PathFstDeltaPow2 (4B+shift/mask)  1648.3           1041.1              24.0              13.9
```

`PathFstDeltaPow2` wins on *every counter against every other
structure* -- fewest cycles (even below the 8-byte `Pos` version),
nearly as few instructions as `Pos` (the shift/mask/subtract itself
costs almost nothing), fewest cache references, fewest cache misses.
Memory matches `PathFstDelta` exactly (same 4-byte format); it is not a
tradeoff against the earlier compact-format win, it is a strict
improvement on it.

**Wall-clock confirmation at full scale, all four shapes** (`Heap` /
`PathFst(Pos)` / `PathFstDelta` / `PathFstDeltaPow2`, ratio to `Heap` in
parens):

```
uniform, symmetric, 10M     587.9 / 452.0 (0.77x) / 540.0 (0.92x) / 366.0 (0.62x) ns
zipfian S+P+O, symmetric    90.0  / 58.6  (0.65x) / 118.0 (1.31x) / 55.8  (0.62x) ns
uniform, asymmetric, 3M     427.8 / 313.6 (0.73x) / 400.9 (0.94x) / 271.7 (0.64x) ns
```

`PathFstDeltaPow2` is fastest in every shape, including the one
(high-skew) where the `BlockIndex` delta variant was slower than `Heap`
outright -- the O(1) navigation removes that regression entirely, not
just shrinks it.

**Status: three real variants now kept for comparison** (`PathFst`,
`PathFstDelta`, `PathFstDeltaPow2`, all in `src/path_fst.rs`), plus the
`perf`-measurement driver (`examples/path_fst_perf.rs`) and the
`[profile.profiling]` Cargo profile added to support it.
`PathFstDeltaPow2` is, on this evidence, the one worth carrying forward
if this continues -- it dominates every other variant measured so far
on every axis tested. `insert`/`remove` (and therefore the real
mutation-vs-dedup tradeoff `s-dict` already navigated once) remain
explicitly out of scope for this pass. The LEB128/NEXT-elision
record-format tightening is the natural next step if more memory needs
to come out of the 4-byte-delta format itself, though `block_capacity`
must stay a power of two for `PathFstDeltaPow2`'s `O(1)` navigation to
apply -- a non-power-of-two capacity would need `PathFstDelta`'s
`BlockIndex` (or a real division) instead.

### BGP-style queries on the FST: automaton-walk state reuse, measured honestly

`PathFstDeltaPow2`'s `contains` only ever answers a fully-bound point
query. The next natural question: can the same deduplicated DAG answer
the "two-Exact" (`[Exact(s), Exact(p), Free]`) and "one-Exact"
(`[Exact(s), Free, Free]`) shapes already named, named, and benchmarked
against `Heap::restricted_descent` elsewhere in this project (see
`src/path.rs`'s own use of exactly these two shapes), and can it do so
*better* than `Heap` by exploiting the thing a plain tree structurally
can't: identical subtrees are the same physical node. The framing that
motivated this: an FST's whole appeal over a trie is that an automaton
can revisit the same state from different input prefixes and reuse
whatever was already computed there (the standard NFA/DFA "cache this
state's accepted continuations" trick) -- so a BGP query walking a
deduplicated DAG should, in principle, get that for free.

**The correctness argument, checked before writing any code, not
assumed**: `restriction_mask` (reused directly from `tree`) folds `lo`
into its result only through `Exact` dimensions -- `Free`'s branch
(`expand_run(mask, 0, width, shifts[d])`) never reads `lo[d]` at all --
and `advance_lo`'s per-level increment to any dimension is a pure
function of `(qi, dim_bits, shifts, child_cell_bits)`, level constants,
never of the incoming `lo` value itself. So as long as every restriction
is `Exact`/`Free` (never `Range`/`Multi`, which can admit more than one
sub-index and so more than one possible `lo` for the same bound
dimension at the same level -- the one precondition this whole argument
needs), an `Exact` dimension's `lo` at a given level is a deterministic
function of that level and its bound value alone, never of which
physical address got us there. Two different free-dimension branches
landing on the *same* deduplicated address at the same level are
therefore guaranteed to see an identical mask and recurse into identical
substructure; the only difference between them is a fixed additive
offset to `emit_dim`'s final value. That makes a `(address, level)`
pair's contribution cacheable *once*, relative to its own entry
`lo[emit_dim]`, and reusable by every other branch that reaches it.

**Three variants built to separate what the cache is worth from what
the traversal itself costs** (all on `PathFstDeltaPow2`, the variant
`contains` already established as the one worth carrying forward):

- `restricted_descent`/`restricted_descent_with_stats(..., use_cache:
  bool)`: `collect_rec` returns an owned `Vec<u32>` per recursive call
  (the relative values get cached under `(addr, level)`, the absolute
  ones get merged into the parent's `Vec` and returned up).
- `restricted_descent_visit`: a pure callback (`FnMut(u32) ->
  ControlFlow<T>`) version, structurally the DAG counterpart of
  `Heap::query_rec_restricted` -- no cache, no per-node allocation,
  stops as soon as `emit` returns `Break`.

Measured with `examples/path_fst_bgp_proto.rs` across the same four
domain shapes `path_fst_proto.rs` already uses, both query shapes,
checked for correctness against `Heap::restricted_descent` (ground
truth) on 2,000 two-Exact and 500 one-Exact random probes per shape,
including the streaming variant (sorted + deduped before comparison,
matching `Heap`'s own contract) -- **0 mismatches across every shape,
both query types, all three FST code paths**, confirming the caching
argument above holds in practice, not just on paper.

**The performance result is a genuine, measured negative for the
caching half of the hypothesis, and a mixed, structurally-explained
result for the FST overall:**

| domain | shape | Heap | FST cached | FST no-cache | FST stream | cache hit rate |
|---|---|---|---|---|---|---|
| small uniform | two-Exact | 1.5us | 4.7us (3.07x) | 1.3us (0.84x) | **1.1us (0.71x)** | 0.02% |
| small uniform | one-Exact | 47.1us | 250.8us (5.32x) | 77.5us (1.64x) | 53.4us (1.13x) | 9.05% |
| 10M uniform | two-Exact | 22.6us | 43.7us (1.93x) | 21.9us (0.97x) | **21.0us (0.93x)** | 0.01% |
| 10M uniform | one-Exact | 1432.0us | 6241.9us (4.36x) | 2543.7us (1.78x) | 2147.0us (1.50x) | 1.11% |
| 10M zipfian | two-Exact | 0.3us | 0.6us (1.82x) | 0.2us (0.57x) | **0.1us (0.45x)** | 0.00% |
| 10M zipfian | one-Exact | 6.9us | 27.3us (3.95x) | 9.0us (1.30x) | **6.3us (0.91x)** | 2.52% |
| 3M asymmetric | two-Exact | 15.2us | 26.6us (1.76x) | 14.2us (0.94x) | **12.4us (0.82x)** | 0.04% |
| 3M asymmetric | one-Exact | 812.0us | 3494.5us (4.30x) | 1503.1us (1.85x) | 1250.6us (1.54x) | 1.93% |

Three honest findings, in the order they were actually found:

1. **Caching never pays for itself here.** Hit rates sit at 0.00-9.05%
   across every shape -- a single BGP query only walks the thin,
   already-position-pinned slice of the DAG its own `Exact` dimensions
   force, and the chance that *that specific slice* happens to revisit
   a shared address at a matching level within one query's lifetime is
   rare, even on the same dataset whose overall dedup ratio (measured
   in the `contains`-only section above) is 15-21%. Dataset-wide
   sharing and single-query-path sharing are different quantities; this
   is the first direct measurement of the second one, and it's small.
   `FST(cached)` loses to `FST(no-cache)` in all eight rows -- the
   `HashMap<(u64, usize), Vec<u32>>` bookkeeping costs more than the
   rare hits save, every time.
2. **The `Vec`-returning/merging recursion (`collect_rec`) was itself a
   real cost independent of caching.** `FST(stream)` (same traversal,
   zero allocation, callback-based) beats `FST(no-cache)` in every row,
   confirming the allocation pattern needed to make caching possible at
   all was already the more expensive design choice -- worth knowing
   before reaching for a cache as the fix for a slow recursive query
   path in general.
3. **Once allocation is out of the picture, the FST's real structural
   position is shape-dependent, not uniformly worse.** `FST(stream)`
   *beats* `Heap` outright on two-Exact in 3 of 4 domain shapes (0.71x,
   0.93x, 0.45x -- small/large/skewed symmetric; 0.82x on the asymmetric
   shape too) and on one-Exact under high skew (0.91x), but loses on
   one-Exact in the other three shapes (1.13x-1.54x). The one-Exact
   loss isn't a DAG-traversal weakness -- it's `Heap::query_rec`'s
   `bulk_list_subtree` fast path, a dedicated bulk-decode routine for
   exactly the "many descendants, no further pruning possible" tail of
   a large result set, something this FST prototype has no counterpart
   for at all (every result, however many share a prefix, is produced
   by one more level of node-by-node recursion). That's a known,
   buildable gap, not a fundamental one -- not attempted in this pass.

**Status before the follow-up below: kept, not pursued further.**
`restricted_descent`/`restricted_descent_with_stats`/
`restricted_descent_visit` all live on `PathFstDeltaPow2` in
`src/path_fst.rs`; `examples/path_fst_bgp_proto.rs` is the comparison
driver. The honest takeaway: the automaton-walk state-reuse idea is
*correct* (proven and measured, 0 mismatches) but *not valuable* for
these domain shapes at single-query granularity -- drop the cache, keep
the streaming traversal. The streaming FST is a real, if narrow, win on
two-Exact queries and matches `Heap` under high skew generally; closing
the one-Exact gap in the other shapes would need a bulk-decode fast
path analogous to `bulk_list_subtree`, not caching.

### Scoping the cache to the `will_always_match` region precisely: still no win

Reading the `fst` crate's `Automaton` trait (`can_match`/
`will_always_match`, see its own entry above) named the flaw in the
caching attempt above directly: it cached *every* `(address, level)`
visited, mixing a state that's still genuinely restrictive (an `Exact`
dimension still has unconsumed bits, so the mask is a real subset of
the word, and a single query's own descent revisits the same address
there only by accident -- the measured 0.00-9.05% hit rates) with a
state that's `will_always_match` (every dimension is `Free` or has
already run out of bits -- `tree`'s own `first_fully_open_level_restricted`
condition, reimplemented here as `PathFstDeltaPow2::first_fully_open_level`
since there's no `RadixTree` to call it on). Past that point, "what does
this address emit" no longer depends on the query at all, only on the
DAG's own shared structure -- the same quantity the dedup ratio already
measures. `restricted_descent_fastpath`/`restricted_descent_fastpath_with_stats`
(`src/path_fst.rs`) scope caching to *exactly* that region
(`bulk_collect_rec`, entered only once `level >= unconstrained_from`)
and use the already-established zero-allocation streaming walk
(`visit_rec_fast`, identical to `restricted_descent_visit`'s traversal)
everywhere before it.

The scoping argument is correct -- the cache hit rate confirms it
directly rather than by inference. In the two symmetric domain shapes
(uniform and zipfian, both 500,000x200x500,000), the fully-open region
*never exists*: `S` (the one `Exact` dimension in both query shapes)
has about as many bits as the free dimensions it shares each level's
budget with, so its bit budget never exhausts before the tree bottoms
out, and `unconstrained_from` lands at or past `height` -- 0 hits, 0
misses, every time, confirmed directly rather than assumed. In the
asymmetric shape (20,000x200x2,000,000), `S`'s much smaller bit budget
*does* exhaust well before the tree's full height, opening a real
fully-open region, and the cache lights up exactly as the theory
predicts: 8.47% hit rate on two-Exact, **97.35%** on one-Exact --
essentially every fully-open node this query touches was already
computed by an earlier branch.

And it still doesn't produce a measured win:

| domain | shape | FST stream | FST fastpath | fastpath hit rate |
|---|---|---|---|---|
| asymmetric | two-Exact | 12.5us (0.81x) | 12.8us (0.83x) | 8.47% |
| asymmetric | one-Exact | 1248.1us (1.52x) | 1288.2us (1.57x) | 97.35% |

Fastpath is *slightly slower* than the plain streaming walk even at a
97.35% hit rate -- the `HashMap` lookup, the `.map(...).collect()` on
every hit, and the extra `Vec` allocation per miss cost more than the
redundant re-walk they replace. The nodes being skipped by a cache hit
in the fully-open region are exactly the cheap, shallow, near-leaf ones
dedup concentrates sharing in (per the subtree-dedup measurements
above) -- re-deriving them directly is already fast enough that
memoizing them is a net loss, not a conditional one. In the two
symmetric shapes, fastpath and stream are statistically indistinguishable
(both near-zero cache traffic, differences within wall-clock noise),
as expected when the region the optimization targets doesn't exist for
that domain shape at all.

**Conclusion, stated plainly:** correctly scoping the cache to the
theoretically sound region doesn't rescue the caching idea -- it
confirms the idea is sound (real hits, at a real rate, on the domain
shape built to expose it) and *still* not profitable, at any
granularity tried so far, for this structure. The lever that actually
closes the one-Exact gap remains what it was before this follow-up: a
genuine `Heap`-style bulk-decode fast path (reading several siblings'
records in one shot, no per-node recursion-call or mask overhead), not
memoization of any kind. `restricted_descent_fastpath` is kept in
`src/path_fst.rs` alongside the other variants as a documented negative
result, not pursued further.

### Correcting the one-Exact attribution: it wasn't `bulk_list_subtree`, it was precomputed masks

The conclusion above -- "the one-Exact loss isn't a DAG-traversal
weakness, it's `Heap`'s `bulk_list_subtree` fast path" -- was wrong for
two of the three losing shapes, and a direct question about it (why
would a dedicated DAG with 5-7x fewer physical nodes lose to a plain
tree on a shape that, per the very next section's own measurement,
*never reaches* `bulk_list_subtree` at all in a symmetric domain) is
what surfaced the mistake. Worth separating two different things that
got conflated: the DAG's node *count* (a storage-time property --
sharing collapses identical subtrees once, regardless of how many
queries later walk through them) says nothing about how many traversal
*steps* one query takes. Without caching -- and caching was already
shown not to pay off above -- two different branches reaching the same
shared node still walk it twice, once per branch; dedup never reduces
a single query's own step count, only the bytes stored. So "fewer
nodes" was never a reason to expect a faster one-Exact traversal, and
the real per-step cost needed a real explanation, not an inference from
the earlier (and, for `one-Exact` in a symmetric domain, inapplicable)
`bulk_list_subtree` story.

**What `Heap::restricted_descent_visit` actually does that
`PathFstDeltaPow2::visit_rec` doesn't, found by reading both side by
side (`src/tree.rs:2210-2343`, `src/path_fst.rs`'s `visit_rec`):**
`Heap::exact_bound_dims` (`src/tree.rs:2210`) returns `Some` for *zero,
one, or two* `Exact` dimensions -- not just two -- so **both**
`one-Exact` and `two-Exact` qualify for the same fast path:
`precompute_level_masks` (`src/tree.rs:2358`) walks `lo` forward once,
before any recursion starts, computing one `u64` mask per tree level
from the restrictions alone, and the entire recursive walk (`query_rec`,
`src/tree.rs:2401`) then does only `word & level_masks[level]` -- an
array index and a bitwise AND, nothing else, at every node. `visit_rec`
(and every other traversal variant built on it: `collect_rec`,
`visit_rec_fast`) instead calls `restriction_mask` (`src/tree.rs:231`,
reused directly, not reimplemented) *fresh, from the full restrictions
array, on every single node visited* -- the full per-dimension branch
(checking `Exact`/`Free`/`Range`/`Multi` for all three dimensions,
calling `expand_run` for each `Free` one) repeated at every level of
every call. `Heap` pays that cost once per query; the FST prototype
paid it once per node. For `two-Exact`, with only a few dimensions'
worth of narrowing and correspondingly few nodes touched per query,
that per-node tax is cheap in absolute terms and `FST(stream)`'s other
structural advantages (direct popcount-indexed delta addressing --
`addr - delta`, no `BlockIndex`/`advance_past_children` skip-ahead
machinery at all) still won outright in most shapes. For `one-Exact`,
visiting 50-200x more nodes per level (per the earlier `one-Exact`
batched-skip instrumentation elsewhere in this file), the same per-node
tax is paid 50-200x more often -- which is a real, sufficient
explanation for the loss on its own, present identically whether or not
`bulk_list_subtree` ever fires.

**The fix, mirroring `Heap`'s own trick exactly**:
`PathFstDeltaPow2::precompute_level_masks` (private) walks `lo` forward
once per query, advancing only the `Exact` dimensions' contribution (the
same path-independence argument the caching work above already proved:
`Free`'s branch never reads `lo[d]`, so only `Exact` dimensions need
tracking), computing one `restriction_mask` call per *level* instead of
per *node*. `restricted_descent_precomputed_visit`/
`restricted_descent_precomputed` use it with the same zero-allocation
callback-based walk (`visit_rec_precomputed`) as `restricted_descent_visit`,
differing only in where the mask comes from. Checked for correctness
against `Heap::restricted_descent` the same way as every other variant
(0 mismatches, all four domain shapes, both query shapes, 2,000
two-Exact + 500 one-Exact probes each).

| domain | shape | Heap | FST stream (old) | FST precomputed (new) |
|---|---|---|---|---|
| small uniform | two-Exact | 1.5us | 1.0us (0.69x) | **0.8us (0.52x)** |
| small uniform | one-Exact | 47.1us | 52.9us (1.12x) | **32.8us (0.70x)** |
| 10M uniform | two-Exact | 22.3us | 20.5us (0.92x) | **16.2us (0.72x)** |
| 10M uniform | one-Exact | 1423.0us | 2140.9us (1.50x) | 1528.0us (1.07x) |
| 10M zipfian | two-Exact | 0.4us | 0.2us (0.41x) | 0.2us (0.56x)\* |
| 10M zipfian | one-Exact | 7.0us | 5.7us (0.82x) | **3.7us (0.53x)** |
| 3M asymmetric | two-Exact | 15.0us | 12.3us (0.82x) | **10.1us (0.67x)** |
| 3M asymmetric | one-Exact | 807.9us | 1241.4us (1.54x) | 863.8us (1.07x) |

(\* zipfian two-Exact: precomputed is marginally slower than stream at
this sample size -- both are sub-microsecond and within wall-clock
noise, consistent with the near-zero node counts this particular shape
touches; not a real regression.)

This closes almost the entire one-Exact gap and improves two-Exact
further on top of the already-measured win: one-Exact now sits at
0.70x/1.07x/0.53x/1.07x across the four shapes (down from
1.12x-1.54x), *beating* `Heap` outright in two of them (small uniform,
zipfian) despite `Heap` using the identical precomputed-mask trick --
meaning once the one real asymmetry (mask computed once vs. once per
node) is removed, the FST's cheaper address decode (no
`BlockIndex`/skip-ahead machinery) shows through on `one-Exact` the same
way it already did on `two-Exact`. The remaining two shapes (10M
uniform, 3M asymmetric) land at near parity (1.07x) rather than a clear
win -- consistent with `Heap`'s `bulk_list_subtree` fast path doing real
work there specifically (both are large enough and, for the asymmetric
shape, skewed enough in per-dimension size that `S`'s bits do exhaust
before the tree bottoms out, unlike the symmetric shapes), closing most
but not all of the remaining difference.

**Conclusion, stated plainly:** the earlier "it's `bulk_list_subtree`,
not caching" explanation was itself incomplete -- correct for the
asymmetric shape, wrong for both symmetric ones, and the error was
only caught because the symmetric shapes' own `bulk_list_subtree`
non-firing was already measured and documented one section above,
making the contradiction checkable rather than a matter of opinion. The
actual, now-measured, now-fixed cause was a missing precomputation this
file simply hadn't built yet.

**Follow-up: wired in as the default, API break accepted on
purpose.** `restricted_descent_precomputed`/
`restricted_descent_precomputed_visit` no longer exist as separate
names -- the precomputed-mask traversal *is* now
`restricted_descent`/`restricted_descent_visit`/`visit_rec`, requested
explicitly ("it's okay to break the API, I really like the idea of
just having a `visit_rec` method, even if the semantics include the
precomputed masks") rather than kept as a second parallel API next to
a slower default. The old per-node-`restriction_mask` `visit_rec` is
gone; `Range`/`Multi` restrictions, which can admit more than one
sub-index per level and so break the one-`lo`-per-level precomputation
this relies on, now panic explicitly at the top of
`restricted_descent_visit` instead of silently falling back to a
slower path the way `Heap::restricted_descent_visit` does (`PathFst`
has no second engine to fall back to -- nothing in this project's own
test suite or examples ever called it with `Range`/`Multi`, confirmed
by grep before accepting the break). `examples/path_fst_bgp_proto.rs`
dropped the now-redundant fifth "precomputed" column (identical to
"stream" once merged) and re-ran clean: same numbers as the
standalone measurement above (within noise), 0 mismatches across every
shape and both query types, full test suite and clippy green. The
`(address, level)`-keyed caching experiments
(`restricted_descent_with_stats`/`restricted_descent_fastpath*`) were
left exactly as they were -- still documented negative results, not
part of the default surface, not touched by this merge.

### `perf`-rigorous one-Exact margin: instructions won, cache misses didn't

With the precomputed-mask fix wired in, `one-Exact` still sits at
near-parity (not a clean win) against `Heap` on the two largest domain
shapes: 1.06x-1.07x on 10M-triple symmetric and 3M-triple asymmetric.
Asked where the remaining margin is, rather than guessing from
wall-clock alone -- same baseline-subtraction methodology as the
`contains()`-only `perf` work above (`examples/path_fst_bgp_perf.rs`,
isolating one structure's `restricted_descent_visit` loop per process,
`iters=0` run subtracted out), `perf stat -e
cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses`:

```
one-Exact, 10M symmetric            pow2          Heap        pow2/Heap
cycles/call                      6,085,548      5,637,455       1.08x
instructions/call                6,057,651     11,531,511       0.53x
cache-references/call              146,054         83,744       1.74x
cache-misses/call                   61,703         32,640       1.89x
branch-instructions/call           516,378      1,665,383       0.31x
branch-misses/call                   9,857         14,658       0.67x
IPC (instr/cycles)                   ~1.00          ~2.05

one-Exact, 3M asymmetric             pow2          Heap        pow2/Heap
cycles/call                      3,421,074      3,188,018       1.07x
instructions/call                3,672,195      7,000,774       0.52x
cache-references/call               76,336         58,370       1.31x
cache-misses/call                   26,886         15,018       1.79x
branch-instructions/call           309,212      1,008,408       0.31x
branch-misses/call                   6,477          9,929       0.65x
IPC (instr/cycles)                   ~1.07          ~2.20
```

Same shape in both domains: `PathFstDeltaPow2` already executes about
*half* `Heap`'s instructions and a third of its branches per call --
the dedup win is real and decisive, confirmed at the instruction level,
not just wall-clock. But it pays 1.3x-1.9x more cache references and
misses per call, and its IPC collapses to roughly half of `Heap`'s
(~1.0-1.1 vs ~2.0-2.2) -- it is memory-latency-bound where `Heap` is
not. The two effects roughly cancel, landing at near-parity cycles
despite the large instruction-count advantage. Contrast with
`two-Exact` (already a clean win, measured the same way on the
symmetric shape: 64,810 vs 89,663 cycles/call, 0.72x): there, `pow2`
still has *more* cache-misses per call than `Heap` (671 vs 587, 1.14x)
-- the same qualitative tilt -- but the absolute node count touched is
small enough that the instruction-count savings dominate outright. For
`one-Exact`, visiting 50-200x more nodes (per this file's earlier
batched-skip instrumentation) scales the *absolute* cache-miss cost up
enough to erase that same advantage.

**Where, mechanically**: `perf record -F 999 --call-graph dwarf` (first
pass, `iters=5000`, build cost still ~13% of samples) followed by a
second pass at `iters=60000` to dilute build-time sampling down to
<6% -- 87.4% of cycle samples land inside `visit_rec` itself at that
point, confirming the loop (not `build_rec_pow2`'s one-time hashmap
dedup, the other visible hot spot) is what a user-facing query actually
pays for. `perf annotate` (both the `cycles` and a separate
`cache-misses`-event recording, to cross-check) puts the overwhelming
majority of samples (73-75%) on the instruction immediately after
`storage.read_var`'s delta-byte load (the `mov (%rax,%rdi,1),%r15` that
decodes `delta` right before `child_addr = addr - delta`) -- consistent
with, but not conclusively pinned to, that load being the actual stall:
this hardware has no PEBS-precise cache-miss event available (`perf
list` shows no `mem-loads`/`:pp` variant), so cycle/cache-miss sampling
here is inherently subject to skid (the CPU's out-of-order execution
attributes a stall to whatever instruction is retiring when the counter
fires, often a few instructions after the load that actually caused
it) -- reported as a strong, consistent signal across two independent
sampling events, not a disassembly-certain single line the way the
`PathFstDeltaPow2`/`BlockIndex` shift-vs-Fenwick finding earlier in this
file was.

**What this means for next steps, not yet attempted**: the lever is
memory locality of the deduplicated DAG's physical layout, not
instruction count -- that battle is already won. A node that many
different `one-Exact` query branches share can be referenced from
parent records built at arbitrary, unrelated times during the bulk
build's postorder traversal, so its physical address bears no relation
to how "hot" (widely shared) it is or to where any particular query's
own descent currently sits; `Heap`'s plain (non-deduplicated) tree, by
contrast, keeps every query's own subtree physically contiguous by
construction. Re-ordering storage to place high-fan-in dedup nodes
near each other (e.g. by descending shared-reference count, or by
level-then-frequency) instead of pure build-discovery order is a
concrete, testable next step -- not implemented or measured in this
pass.

### Deeper FST-only perf dive: a real algorithmic cost, but one this project already learned is dangerous to fix blindly

Asked to dig into `PathFstDeltaPow2` alone (not vs. `Heap`) for an
*algorithmic* lever, given that layout fixes for a DAG are
fundamentally limited -- a node with many parents can't be physically
adjacent to all of them at once, postorder or not. Added temporary
debug instrumentation (`restricted_descent_visit_histogram`, mirroring
`visit_rec` but tallying node visits and multi-child fan-out per tree
level; deleted after collecting the data, same throwaway-instrumentation
convention as the earlier batched-skip investigation) and ran it over
1,000 `one-Exact` probes on both the 10M-triple symmetric and 3M-triple
asymmetric domains:

```
sym (height=8): total_nodes=36,853.9/probe, 1,837.7 nodes per actual result
level  dim_bits     nodes/probe   fanout>1
  0    [2,2,2]          1.00       100.00%
  1    [2,2,2]         16.00       100.00%
  2    [2,2,2]        208.00       100.00%
  3    [2,2,2]      3,092.26        98.74%
  4    [3,0,3]     27,729.51         0.80%   <- P (the 3rd column) just exhausted
  5    [3,0,3]      5,086.10         0.01%
  6    [3,0,3]        641.09         0.00%
  7    [2,0,2]         79.96         0.00%   <- S fully exhausted here (last level)

asym (height=8): total_nodes=22,024.8/probe, 146.7 nodes per actual result
level  dim_bits     nodes/probe   fanout>1
  3    [2,2,2]      3,078.91        97.48%
  4    [3,0,3]     15,879.59         0.31%
  5    [3,0,3]      2,391.49         0.00%
  6    [1,0,5]        299.63         0.00%   <- S fully exhausted here
  7    [0,0,2]        150.18         0.00%   <- matches total_results/probe almost exactly
```

**The mechanism, confirmed by the data, not inferred**: `dynamic_schedule`
spreads `S`'s 19 bits (`uniform`'s domain) evenly across *all 8* levels
(2,2,2,2,3,3,3,2) rather than resolving them up front -- so `S` is
still a real, partially-unresolved constraint at every level up to and
including the last one. `P` (the middle dimension, only 8 bits) fully
resolves by level 3. From level 4 on, a node's word jointly encodes a
few more bits of `S` *and* `O` together, `O` fully free, `S` still
pinned to one sub-range. At level 3, each node has ~9 real matching
children on average (`O` is unconstrained there, and the data, while
sparse, still has that much real variety) -- those children are reached
correctly, but at level 4 most of them turn out to encode the *wrong*
`S` sub-range for *this* query (they're real because some *other*
`S` value's path needs them, or simply because the sparse data doesn't
happen to align), so `relevant` comes back empty and the node dead-ends
immediately after one word read. That single word read, on a 100MB+
structure with no relationship between a child's address and its
logical position, is a real cache-probe cost paid for nothing: level 4
alone accounts for 27,729 of the 36,854 node visits per `sym` probe
(75%) and 15,880 of 22,025 per `asym` probe (72%) -- the dominant single
contributor to the cache-miss count measured in the previous section,
and the reason there are ~1,838 (`sym`) / ~147 (`asym`) nodes visited
per actual result returned. This is not a dedup artifact and not a
layout problem -- a non-deduplicated tree with the identical schedule
would pay exactly the same dead-end cost, visiting the identical real
data in the identical order (`Heap` avoids it in practice only because
its `bulk_list_subtree`/`unconstrained_from` mechanism, present in this
asymmetric domain, bypasses node-by-node descent *after* `S` actually
exhausts late in the tree -- a different fix for a related but distinct
problem, not this one). It is squarely the bit-interleaving *schedule*'s
doing: `S`'s bits are resolved too slowly relative to how early `O`'s
fan-out opens up.

**The obvious fix is exactly the one this project already tried and
reverted, for a documented reason.** `dynamic_schedule`
(`src/tree.rs:353`) already carries a comment recording a prior
attempt to weight bit allocation unevenly across dimensions (by
relative domain size, to speed up the smaller dimension's scans) that
measured a **16x regression** on the opposite-direction query (fast
for the dimension it favored, catastrophically slower for a scan bound
on the dimension it starved) -- reverted back to the current even
round-robin split specifically because "a weighting heuristic that's
actually safe across domain shapes remains unsolved." Front-loading
`S`'s bits (so it fully resolves before `O`'s fan-out opens up,
eliminating this exact dead-end-fanout cost for `one-Exact`/`two-Exact`
as benchmarked here) is a *different* axis of asymmetry than the
reverted attempt -- resolution *order*, not relative bit *budget* --
but carries the identical structural risk: this schedule is built once
per dataset and shared by every query shape a caller might issue
against it, including the mirror case this project hasn't benchmarked
in this investigation -- `[Free, Free, Exact(o)]`, binding `O` instead
of `S`. Front-loading `S` would very plausibly reproduce the same kind
of regression on *that* query shape that the original attempt produced,
for the same underlying reason (whichever dimension's fan-out opens up
early pays the dead-end cost that whichever dimension resolves early
avoids). **Not attempted in this pass**: the honest next step, if this
is worth pursuing, is building the alternate schedule and benchmarking
it against *both* query directions (`S`-bound and `O`-bound, not just
the one this project happens to benchmark by habit) across all four
domain shapes before trusting it -- exactly the validation the earlier
attempt skipped the first time, which is why it had to be reverted.

### A full `one-Exact`/`two-Exact` perf sweep, a real allocation fix, and a negative on local sibling prefetching

Three follow-ups, all on `PathFstDeltaPow2` alone (not vs. `Heap`):
a full baseline-subtracted `perf stat` sweep across both query shapes and
all four domain shapes (`examples/path_fst_bgp_perf.rs`, extended with
`small`/`zipf` domains to match `path_fst_bgp_proto.rs`'s four), a real
fix the sweep's own data motivated, and a test of whether exposing
memory-level parallelism locally (prefetching a node's own children
before recursing into any of them) helps the shapes the earlier `perf`
dive found to be memory-latency-bound.

**The sweep** (cycles, instructions, IPC, cache-references, cache-misses,
branch-instructions, all per call):

```
shape/domain      cycles/call   instr/call    IPC    cache-refs  cache-miss  miss%
small, one-Exact    127,656      388,426     3.04       3,961        702    17.7%
small, two-Exact      3,123        9,379     3.00         209         40    19.1%
sym,   one-Exact  6,004,664    6,089,949     1.01     147,796     62,024    42.0%
sym,   two-Exact     65,497       46,777     0.71       1,378        673    48.9%
zipf,  one-Exact     12,242       36,587     2.99         401         86    21.4%
zipf,  two-Exact        932        3,173     3.41          10          1    13.2%
asym,  one-Exact  3,416,287    3,668,028     1.07      77,209     27,073    35.1%
asym,  two-Exact     41,726       32,818     0.79       1,059        460    43.4%
```

The standout: `two-Exact` on the two large domains (`sym`/`asym`) runs
at IPC *below 1* (0.71, 0.79) -- worse than the much bigger `one-Exact`
traversal in the *same* domain, backwards from what raw node count
alone would predict. That pattern -- a fixed per-call cost dominating a
shape that otherwise does very little real traversal work -- pointed
straight at a concrete bug: [`PathFstDeltaPow2::precompute_level_masks`]
returned a heap-allocated `Vec<u64>`, built fresh on *every* query call,
when the method it was explicitly modeled on
(`tree::precompute_level_masks`) already returns a fixed-size
`[u64; MAX_HEIGHT]` array for exactly this reason -- a detail that
didn't get carried over when the trick was first ported into this file.

**The fix**: switched to `[u64; MAX_HEIGHT]` (`MAX_HEIGHT = 64`,
`pub(crate)` in `tree.rs`, reused directly, not reimplemented -- every
real domain's `height` is nowhere close), matching `tree`'s own pattern
exactly. Zero risk, zero behavior change (0 mismatches across every
shape, full test suite green). Measured effect, same methodology,
before -> after:

```
shape, two-Exact    cycles/call before   cycles/call after   change
small                      3,123                3,037         -2.8%
sym                       65,497               63,447         -3.1%
zipf                         932                  868         -6.9%
asym                      41,726               39,444         -5.5%
```

`one-Exact` was unaffected either way (within noise) -- the allocation
is a fixed per-query cost, invisible once amortized over tens of
thousands of node visits, exactly as the fixed-cost signature in the
sweep predicted before the fix was even written.

Checked the other two categories the sweep was run to find, and found
nothing further to change: `DELTA_BYTES = 4` is a power of two, so
`real_index * DELTA_BYTES` already compiles to a shift; the one
non-power-of-two multiply in the hot path (`level * NUM_DIMS`, `NUM_DIMS
= 3`, for indexing the per-level `[u8; 3]` schedule arrays) already
lowers to a single `lea`, as cheap as a shift on this hardware. Branch
misprediction *rates* (not just branch density) computed from the same
sweep sit at 0.7%-3.4% across all eight rows -- already well-predicted;
for the expensive shapes specifically (`one-Exact`, `sym`/`asym`, IPC
~1.0-1.07, 35-42% cache-miss rate), the bottleneck is memory latency,
not branches, so eliminating the `is_last` check's per-node branch
wouldn't move the needle there -- not attempted.

**Local sibling prefetching, tested and rejected.** The idea (prompted
by the observation that `level_masks[level]` is one shared scalar
constant every node at that level ANDs its own word against, a
structurally SIMD-shaped operation): rather than vectorizing the
arithmetic itself (not the bottleneck -- IPC data above already shows
the expensive shapes are latency-, not throughput-bound), use the same
"many nodes, one mask" fact to issue several independent memory
accesses together instead of one at a time, exposing memory-level
parallelism to the CPU's out-of-order execution instead of the fully
serial dependent-load chain plain recursion forces (each child's
address is only computed, and its own word only read, *after* its
address is decoded, which only happens *inside* the recursive call for
the previous child). `_mm_prefetch` isn't available -- this crate is
`#![forbid(unsafe_code)]`, and there's no safe wrapper in `std` -- so
[`PathFstDeltaPow2::restricted_descent_prefetch_visit`]/
`visit_rec_prefetch` use a real, safe, result-discarded `read_u64`
(wrapped in `black_box` so the optimizer can't prove it dead and elide
it) at each of a node's *own* real children's addresses, decoded ahead
of time (cheap -- the deltas are bytes right next to the already-read
`word`, no extra cache miss to get them), before recursing into any of
them for real. Scoped locally (one node's bounded fan-out, no growable
buffer) rather than a full BFS/frontier rewrite across a whole level,
specifically to test the idea cheaply before paying for that bigger,
riskier redesign.

Correctness held (0 mismatches, every shape, both query types, full
test suite green) but the performance result is a clean, consistent
negative for `one-Exact` -- the one shape this was built to help:

```
one-Exact          stream (precomputed masks)   prefetch (local sibling)
small                32.5us (0.68x)               41.0us (0.86x)   worse
sym (10M)           1530.1us (1.07x)             1610.6us (1.13x)  worse
zipf (10M)            3.6us (0.53x)                4.6us (0.68x)   worse
asym (3M)            853.6us (1.05x)              892.8us (1.10x)  worse
```

(`two-Exact` was mixed/within noise across the four shapes -- small
sample, microsecond-scale, not a clear signal either way.)

**Why, worked out after the fact, not before**: this traversal is DFS,
not BFS -- after the "prefetch" pass warms children 2..k's cache lines,
the very next thing that happens is recursing fully into child 1's
*entire* subtree before ever touching child 2 again. At the level this
costs the most, that subtree is not shallow. By the time control
returns to use child 2's supposedly-warmed line, the intervening
traffic from child 1's whole subtree has very plausibly evicted it from
L1/L2 already -- the prefetch-to-use *distance* here is "one sibling's
full subtree," not a few instructions, which is far outside the window
any practical prefetch depth covers. Meanwhile the technique's fixed
cost -- a second decode pass over every child's delta, every single
node, doubling that arithmetic regardless of whether the memory benefit
ever lands -- is paid unconditionally. The fixed cost is real and
guaranteed; the memory benefit depends on a cache-line surviving a gap
that, on this traversal shape, it structurally can't. This argues for,
not against, the pattern the earlier margin-finding entry already
flagged as the bigger remaining lever: a real BFS/frontier
restructuring processes one whole *level* flat before advancing, so the
gap between "prefetch a frontier node" and "use it" is the rest of that
level's frontier, not a full subtree -- a fundamentally shorter, more
survivable distance. That's a bigger rewrite (needs a growable buffer
sized to the level's node count, which this file's current traversal
has none of, trading away the zero-allocation property) and hasn't been
attempted. `restricted_descent_prefetch_visit`/`restricted_descent_prefetch`
are kept in `src/path_fst.rs` as a documented negative result, same
convention as every other prototype here.

### BFS/frontier restructuring: a real win, confirmed by `perf` not just wall-clock -- then a second version that gets the same win without the heap buffers

Built the "bigger rewrite" the previous entry flagged:
`restricted_descent_frontier_visit`/`restricted_descent_frontier` in
`src/path_fst.rs` process one tree level flat (all nodes across all
parents at that level, in one batch) instead of recursing depth-first.
Per level: issue a discarded `read_u64` (same `black_box`-wrapped,
`#![forbid(unsafe_code)]`-respecting pattern as the local-sibling
prototype) for every node currently in the frontier, *then* do the real
reads, decode, and build the next level's frontier. Two `Vec<(u64, [u64;
NUM_DIMS])>` buffers (`current`/`next`), swapped each level instead of
reallocated -- the zero-allocation property every other traversal in
this file has is given up on purpose, in exchange for a prefetch-to-use
gap of "the rest of this level's frontier" instead of "one sibling's
whole subtree."

Wall-clock (`examples/path_fst_bgp_proto.rs`, FST(stream) is the
existing precomputed-mask default, ratio is vs. `Heap`), 0 mismatches
against `Heap` across every shape:

```
one-Exact          stream (precomputed masks)   frontier
small                32.5us (0.68x)               31.1us (0.67x)   slightly better
sym (10M)           1550.5us (1.08x)             1315.4us (0.92x)  beats Heap outright
zipf (10M)            3.6us (0.52x)                3.8us (0.55x)   slightly worse
asym (3M)            873.1us (1.06x)               610.0us (0.74x)  beats Heap outright

two-Exact           stream (precomputed masks)   frontier
small                 0.8us (0.51x)                1.0us (0.68x)   worse
sym (10M)            16.1us (0.71x)                9.9us (0.43x)   much better
zipf (10M)            0.2us (0.62x)                0.3us (0.80x)   worse
asym (3M)            10.8us (0.72x)                6.2us (0.41x)   much better
```

Per the explicit instruction not to trust wall-clock alone here, the
same four biggest-margin combos (sym/asym x one/two-Exact) were measured
with the same baseline-subtracted `perf stat` methodology as every other
entry in this section (`examples/path_fst_bgp_perf.rs`, `("frontier",
...)` arms added to mirror the existing `("pow2", ...)` ones):

```
shape              cycles (stream->frontier)   cache-miss (stream->frontier)   cycles-per-miss (stream->frontier)
sym,  one-Exact    6,034,353 -> 5,220,307 (-14%)   61,528 -> 89,622  (+46%)        98.1 -> 58.3  (-41%)
sym,  two-Exact       63,447 ->   38,219  (-40%)      664 ->    497  (-25%)        95.5 -> 77.0  (-19%)
asym, one-Exact    3,402,359 -> 2,519,417 (-26%)   27,035 -> 65,664  (+143%)      125.9 -> 38.4  (-70%)
asym, two-Exact       39,444 ->   24,181  (-39%)      452 ->    404  (-11%)        87.3 -> 59.9  (-31%)
```

The honest nuance: raw cache-miss *count* went up in 2 of 4 combos (the
discarded-read-then-real-read pattern reads every node's word twice, the
same fixed cost the rejected local-sibling prototype paid). What's
consistent across all four, and what actually explains the cycle drop,
is **cycles per cache-miss** falling 19%-70% -- the misses that do occur
cost far less each, because issuing them as a batch of mutually
independent reads (different parents, no decode dependency between
them) lets the CPU's out-of-order execution service several concurrently
instead of one at a time through a serialized dependent-load chain. This
is genuine memory-level-parallelism overlap, not a reduction in work --
and it's the mechanism the rejected local-sibling prototype never got
access to, because DFS's "whole subtree" gap was too long for the
overlap window to survive, while BFS's "rest of this level" gap isn't.
Wall-clock and `perf`-derived cycle reductions cross-validate well (e.g.
sym one-Exact: ~15% wall-clock vs. 14% perf-cycle; asym one-Exact: ~30%
wall-clock vs. 26% perf-cycle). `small`/`zipf` (tiny per-query work) were
a wash or mildly negative -- the new heap buffers aren't amortized when
there's barely any work to spread their cost over.

**Follow-up, same session: the same win without the heap buffers.**
Asked directly: can the level-flat batching be done without giving up
the zero-allocation property? `restricted_descent_bfs_visit` (also in
`src/path_fst.rs`) answers yes, with a different trade: every node held
in flight lives in a fixed-size `[(u64, [u64; NUM_DIMS]); BFS_WINDOW]`
array on the stack (`BFS_WINDOW = 256`), never the heap. A level wider
than `BFS_WINDOW` is processed in sequential chunks -- warm, then read
real, with any children produced buffered into a *second* fixed array
that gets flushed (one recursive call, one level deeper) as soon as it
fills rather than being allowed to grow. Recursion depth is bounded by
`self.height` (<= `MAX_HEIGHT`), the same bound the plain DFS `visit_rec`
already has -- this is no deeper, it just carries a window-sized array
per frame instead of one node's worth of state.

Measured the same way (0 mismatches, full test suite green). Wall-clock,
with the frontier column repeated for comparison:

```
one-Exact          frontier         bfs (no heap buffers)
small               31.1us (0.67x)   28.5us (0.59x)   better
sym (10M)         1379.4us (0.97x) 1036.6us (0.73x)   much better, widens the win over Heap
zipf (10M)           3.8us (0.55x)    3.5us (0.52x)   better
asym (3M)           629.5us (0.77x)  538.5us (0.66x)  better

two-Exact           frontier          bfs (no heap buffers)
small                1.1us (0.70x)    1.0us (0.66x)   better
sym (10M)            10.2us (0.45x)    9.6us (0.43x)  slightly better
zipf (10M)            0.3us (0.79x)    0.5us (1.39x)  worse -- see below
asym (3M)             6.7us (0.44x)    6.4us (0.42x)  slightly better
```

`perf stat` on the same four biggest-margin combos (baseline-subtracted
per-call deltas, `("bfs", ...)` arms added to
`examples/path_fst_bgp_perf.rs`):

```
shape              cycles (frontier->bfs)       cache-miss (frontier->bfs)     cycles-per-miss (frontier->bfs)
sym,  one-Exact    5,220,307 -> 4,206,173 (-19%)   89,622 -> 50,810 (-43%)       58.3 -> 82.8  (+42%)
sym,  two-Exact       38,219 ->   38,897 (+2%)        497 ->    491 (-1%)        77.0 -> 79.3  (+3%)
asym, one-Exact    2,519,417 -> 2,207,204 (-12%)   65,664 -> 38,568 (-41%)       38.4 -> 57.2  (+49%)
asym, two-Exact       24,181 ->   24,714 (+2%)        404 ->    418 (+3%)        59.9 -> 59.1  (-1%)
```

A genuinely different (and better) story than "same win, smaller
footprint": `bfs`'s cycles-per-miss is *higher* than the `Vec`-based
frontier's on both one-Exact combos (less overlap per miss -- expected,
since a 256-wide window is narrower than an unbounded whole-level
frontier, so fewer independent reads are ever in flight at once), but
its total cycle count is still lower in 3 of 4 combos, because it also
*reduces the cache-miss count itself* relative to frontier (-43%/-41% on
the two combos that matter most) instead of just overlapping it. The
`two-Exact` combos are a wash either way (both cycles and misses within
a couple percent, consistent with the near-noise-level wall-clock gap
there). Net effect: giving up the heap buffers didn't cost
performance -- on the shapes this investigation has been chasing all
along (large-domain one-Exact), it won again. One-Exact on the
asymmetric domain is now at 0.66x of `Heap` (was 1.06x with the original
`stream` default this whole investigation started from).

The one real regression: `two-Exact` on the `zipf` domain (0.79x
frontier -> 1.39x bfs). That shape returns almost nothing (6 total
results across 2000 probes in the correctness run -- most probes
terminate in 1-2 node visits before even reaching a level with real
fanout), so the fixed overhead of `bfs_window_visit`'s extra recursive
function-call boundary (slice bounds, an `is_last` check, a fresh
`next_buf` array initialization) costs more, as a fraction of an already
tiny total, than the frontier's flat per-level loop does. Not
perf-verified independently (the combo's absolute cost is microseconds,
below where `perf stat`'s per-call signal is reliable at these iteration
counts) -- flagged honestly rather than investigated further, consistent
with `small`/`zipf`'s mixed frontier results never getting a perf pass
either.

Both `restricted_descent_frontier_visit` and `restricted_descent_bfs_visit`
are kept in `src/path_fst.rs` as named, documented alternatives to the
default `restricted_descent_visit` -- neither has been wired in as the
default, since both win decisively on the large-domain shapes this
investigation targeted but regress (or are a wash) on `small`/`zipf`'s
tiny-query shapes. Productionizing either would need a dispatch (e.g. by
expected result-set size or domain scale) rather than an unconditional
swap.

### `PathFstDeltaPow2Bfs`: pairing a BFS-ordered *layout* with the BFS *algorithm* -- a real, if different-than-predicted, win

Every traversal above still runs on `PathFstDeltaPow2`'s original
storage order, which is physically post-order: `build_rec_pow2` appends
a node the moment it's first discovered (children before parents, so
delta-encoding's subtraction stays unsigned), and the per-level `register`
dedup means a shared node keeps whichever address its *first* discovering
parent happened to leave it at. That's the structural ceiling the
rejected local-sibling-prefetch entry already named: no reordering pass
can put a node near *every* parent that references it, because distinct
parents want it in different places.

But `restricted_descent_bfs_visit` doesn't care about any one parent's
locality -- it wants every node at the *same tree level* to be close
together, regardless of which parent built it first. `PathFstDeltaPow2Bfs`
(new struct, `src/path_fst.rs`) lays storage out that way directly:
same dedup as `PathFstDeltaPow2` (two DAG nodes with identical
`(existence_word, child identity)` still share one physical record), but
nodes are *addressed* in BFS/level order -- every distinct level-0 node,
then every distinct level-1 node, and so on. Dedup is inherently
bottom-up (can't know two subtrees are identical until both are fully
built); this layout is inherently top-down (level 0 must be written
before level 1). Building it is three passes instead of one:

1. `collect_rec_bfs` -- the same recursive dedup as `build_rec_pow2`,
   but recording each distinct node's `(existence_word, child_indices)`
   into a per-level `Vec` instead of writing storage (`child_indices`
   are positions within the *next* level's own `Vec`, not addresses).
2. A level-by-level top-down loop assigns every node's real storage
   position (deltas zeroed -- a node's own address is known the moment
   it's appended, but its children's addresses don't exist until the
   next loop iteration).
3. A second top-down loop backfills every real delta via
   `patch_appended`, now that every level's addresses are known. The
   sign flips from `PathFstDeltaPow2`'s: a child is always at a *higher*
   address than its parent here (written in a later level), so
   `delta = child_addr - addr`, and every read path *adds* the delta
   instead of subtracting it.

Measured (0 mismatches against `Heap`, full test suite + clippy green),
wall-clock on the two biggest-margin domains, old layout's best variant
repeated for reference:

```
one-Exact          best on old layout (bfs)   new layout, plain DFS   new layout, bfs-window
sym (10M)           1033.1us (0.72x)            1266.7us (0.88x)       805.4us (0.56x)
asym (3M)            537.6us (0.66x)              795.5us (0.98x)       464.7us (0.57x)

two-Exact          best on old layout (bfs)   new layout, plain DFS   new layout, bfs-window
sym (10M)              9.5us (0.42x)              15.0us (0.66x)         8.1us (0.36x)
asym (3M)              6.2us (0.40x)               9.0us (0.59x)         5.3us (0.35x)
```

Two separate things are visible here, and `perf stat` (baseline-subtracted,
same methodology as every other entry) was used to tell them apart
instead of guessing from wall-clock alone:

**The layout change by itself helps even the plain-DFS traversal that
never touches it on purpose** -- a real surprise, since `visit_rec` was
written with zero awareness of layout. Comparing the *same* DFS
algorithm on old vs. new layout (`stream` vs. `layout`, no discard-read
trick on either side, so any miss-count change here is a clean locality
signal, not conflated with overlap):

```
shape              cycles (stream->layout, DFS)   cache-miss (stream->layout, DFS)
sym,  one-Exact    6,034,353 -> 5,000,275 (-17%)    61,528 -> 42,512 (-31%)
sym,  two-Exact       63,447 ->   56,626  (-11%)       664 ->    595  (-10%)
asym, one-Exact    3,402,359 -> 3,093,751  (-9%)     27,035 -> 27,522  (+2%)
asym, two-Exact       39,444 ->   36,082  (-9%)         452 ->    432  (-5%)
```

On the symmetric domain this is a genuine cache-*miss-count* reduction,
not just overlap -- exactly what laying same-level nodes out contiguously
was supposed to buy, and the first result in this whole investigation
where miss count itself (not just its latency) moved in the right
direction. On the asymmetric domain the miss count is roughly flat
(+2%/-5%, noise-level) even though cycles still dropped ~9% -- the win
there isn't coming from fewer misses; what it is coming from (fewer
TLB misses? a more regular stride the hardware prefetcher likes better?)
wasn't isolated further, flagged honestly rather than guessed at. Either
way, cycles and wall-clock cross-validate tightly (e.g. sym one-Exact:
-17.1% cycles vs. -17.1% wall-clock, asym one-Exact: -9.1% cycles vs.
-8.9% wall-clock).

**Pairing the BFS-window traversal back on top of the new layout
(`layoutbfs`) adds the *other*, already-familiar mechanism on top of
that first win**, and it's worth being precise that it's not more of the
same thing:

```
shape              cycles (layout DFS -> layoutbfs)   cache-miss (layout DFS -> layoutbfs)   cycles-per-miss (layout DFS -> layoutbfs)
sym,  one-Exact    5,000,275 -> 3,252,006 (-35%)       42,512 -> 53,994 (+27%)                117.6 -> 60.2  (-49%)
sym,  two-Exact       56,626 ->   34,637  (-39%)          595 ->    634  (+6%)                 95.2 -> 54.7  (-43%)
asym, one-Exact    3,093,751 -> 1,868,704 (-40%)       27,522 -> 37,566 (+37%)                112.4 -> 49.7  (-56%)
asym, two-Exact       36,082 ->   21,455  (-41%)          432 ->    433  (flat)                83.6 -> 49.6  (-41%)
```

Adding the window-BFS traversal on top of the already-better layout
*increases* raw miss count again (same duplicate discard-then-real-read
cost the original `bfs`-on-old-layout entry paid), and gets its win the
same way that entry did: a large drop in cycles-per-miss from batching
mutually-independent reads so the CPU can overlap them. The two
mechanisms are additive but distinct -- the layout's own contribution is
fewer misses (symmetric domain) or some other per-miss-independent
effect (asymmetric domain); the traversal's contribution on top is
cheaper misses via overlap, the identical mechanism `restricted_descent_bfs_visit`
already demonstrated on the old layout. Together they compound to the
best numbers measured in this entire investigation: one-Exact on the
asymmetric domain is now at 0.57x of `Heap`, down from the 1.06x the
very first precomputed-mask fix started this whole line of work at.

`small` and `zipf` show the same pathology every BFS-shaped variant has
shown: a wash or a real regression (`zipf` two-Exact: still ~1.3x) from
fixed per-call overhead that tiny queries can't amortize -- unchanged by
the layout, since it's a traversal-structure cost, not a memory one.
Not perf-verified independently for the same reason as before (too cheap
per call for a reliable signal at these iteration counts).

`PathFstDeltaPow2Bfs` is kept in `src/path_fst.rs` alongside
`PathFstDeltaPow2`, not merged into it or wired in as any kind of
default -- it's a different build (three passes instead of one, and a
different, slightly more complex, delta sign convention) with its own
trade-offs, not a drop-in replacement. Only `restricted_descent_visit`
(plain DFS) and `restricted_descent_bfs_visit` (fixed-window BFS) are
ported to it; `restricted_descent_frontier_visit` (the `Vec`-based
version) was not, since `restricted_descent_bfs_visit` already
dominates it on this layout the same way it did on the old one.

### SIMD-batching the mask-AND over the BFS window: vectorizes exactly as predicted, loses anyway

Natural follow-up question once nodes at a level are physically
contiguous: instead of computing `word & level_masks[level]` one node at
a time, interleaved with reads/branches/decode, why not pull it out into
its own flat pass over the window and let the compiler vectorize it?
Two things had to be checked before trusting that: first, whether the
BFS-ordered *layout* alone lets you skip the frontier altogether (it
doesn't -- a node's level says nothing about whether *this* query's root
actually reaches it; that's still only established by walking the real
parent-to-child deltas, same as every other traversal here, and masking
a whole level blind would both over-scan and produce false matches from
unrelated DAG nodes that happen to share a bit pattern). Second, whether
real vector instructions are even reachable from safe Rust here:
`#![forbid(unsafe_code)]` rules out `std::arch`'s `_mm256_and_si256` and
friends (those need `unsafe`), and portable `std::simd` is nightly-only
(`#![feature(portable_simd)]`) -- so the only lever available is giving
LLVM's own auto-vectorizer a clean enough loop to work with on stable
Rust, not calling SIMD directly.

`restricted_descent_bfs_masked_visit` (`src/path_fst.rs`, on
`PathFstDeltaPow2Bfs`) restructures `bfs_window_visit`'s per-chunk body
into three separated flat passes instead of one interleaved loop: (A)
read every node's word into a plain `[u64; BFS_WINDOW]` array, no
masking or branching; (B) one branch-free `masked[i] = words[i] & mask`
loop over that array and a second plain array; (C) the normal
decode/recurse logic, now just reading from `words`/`masked` instead of
recomputing them inline.

Checked via `objdump` on the release binary, not assumed: phase B really
did vectorize --

```
movdqu 0x128(%rsp,%rdx,1),%xmm0
movdqu 0x138(%rsp,%rdx,1),%xmm1
pand   %xmm2,%xmm0
pand   %xmm2,%xmm1
movdqu %xmm0,0x928(%rsp,%rdx,1)
movdqu %xmm1,0x938(%rsp,%rdx,1)
```

two real SSE2 `pand`s, each ANDing 2 `u64` lanes against the broadcast
mask in one instruction, unrolled by 2. The premise was correct.

It still loses. Wall-clock (`examples/path_fst_bgp_proto.rs`, 0
mismatches, comparing against `restricted_descent_bfs_visit` on the same
`PathFstDeltaPow2Bfs` layout):

```
one-Exact          bfs-window        bfs-window, masked
sym (10M)           803.0us (0.56x)   873.7us (0.61x)   9% slower
asym (3M)           459.7us (0.56x)   507.2us (0.62x)   10% slower

two-Exact           bfs-window        bfs-window, masked
sym (10M)             8.1us (0.36x)     9.2us (0.41x)   14% slower
asym (3M)             5.1us (0.34x)     5.8us (0.38x)   14% slower
```

`perf stat` (baseline-subtracted, same methodology as every other
entry) confirms it's not noise, and is specific about *why*: cache-miss
count is essentially unchanged in all four combos (the restructuring
reads the exact same bytes, just through two extra arrays instead of
directly) -- the entire regression is in cycles:

```
shape              cycles (bfs-window -> masked)   cache-miss (bfs-window -> masked)
sym,  one-Exact    3,292,894 -> 3,935,653 (+20%)    54,028 -> 53,303 (flat)
sym,  two-Exact        31,141 ->   60,492 (+94%)       618 ->    616 (flat)
asym, one-Exact    1,897,512 -> 1,962,785  (+3%)    36,594 -> 36,520 (flat)
asym, two-Exact        21,508 ->   20,918  (-3%, noise)   430 ->    428 (flat)
```

The flat miss counts rule out a memory-access-pattern explanation --
this is a pure instruction-count/fixed-cost regression. Two real costs,
neither paid by the window-BFS this was meant to improve: the two extra
`[u64; BFS_WINDOW]` arrays are zeroed on *every* chunk call regardless
of how many of their 256 slots are actually used (the same fixed-size-
array convention `bfs_window_visit`'s own `next_buf` already uses, just
now two more of them per call), and routing each word through an
intermediate array before the decode loop reads it back breaks what was
previously a tight read-then-immediately-use access, adding a round
trip through L1 that the single interleaved loop didn't have. `two-Exact
sym`'s outsized +94% (worst of the four, on the cheapest query in the
set) is consistent with that: a fixed per-call cost lands hardest on
whichever query has the least real work to amortize it against -- the
same pathology `small`/`zipf` have shown in every BFS-shaped variant so
far, just triggered here by array-zeroing instead of traversal structure.

The lesson isn't "SIMD doesn't work here" -- it's that the scalar AND
was never the bottleneck being paid for. Two cheap, already-well-
predicted-branch bitwise ANDs per iteration were not where this
traversal's cost lives; memory latency and the bit-by-bit
`trailing_zeros`/`count_ones` decode loop are, and this change touches
neither. `restricted_descent_bfs_masked_visit` is kept in
`src/path_fst.rs` as a documented negative result, same convention as
every other rejected prototype in this file.

### The "mask tree + whole-level sweep" idea: measured, and decisively rejected

A follow-up proposal, explicitly modeled on Hyperscan/Teddy's and
simdjson's "cheap bulk filter pass over the whole input, then a slower
confirm pass on survivors" design (see `FST_LITERATURE_REVIEW.md`):
instead of computing `word & level_masks[level]` only for the nodes a
query's real descent actually reaches, sweep an *entire tree level* at
once -- read every distinct node's word and AND it with that level's
one mask, in a single flat pass over the level's whole address range --
then walk the real parent-to-child links as always, just reading the
precomputed `(word, word & mask)` pair instead of recomputing the AND
inline.

Checked for a real blocker before writing any code: is "the data tree"
even a flat, uniformly-strided structure a sweep could walk without
knowing addresses in advance? No -- `PathFstDeltaPow2Bfs`'s records are
variable-length (`WORD_BYTES + popcount(word) * DELTA_BYTES`), and
`HeapStorage::append_record`'s block-rollover decision for record `i+1`
depends on record `i+1`'s own length, which isn't known without already
knowing where it starts -- a real circular dependency, not a detail that
falls out by construction. Worked around by persisting `build`'s own
`addr_of` on the struct (`level_addrs: Vec<Vec<u64>>`, one extra `u64`
per node versus the production struct) instead of re-deriving positions
by walking bytes, and finding a node's index within its level via binary
search (sound because addresses are assigned in strictly increasing
order during `build`'s Phase 2, so each level's `Vec` is already
sorted). `restricted_descent_sweep_visit`/`restricted_descent_sweep` in
`src/path_fst.rs`.

But before writing that code, the real question was checked with data
already in hand from this session's own build: how much bigger is a
level's *distinct* node population than what any single query actually
needs? Instrumented once (`build`, temporary, removed after collecting
the numbers):

```
sym (height 8):   1, 64, 3328, 191170, 5142944, 130344, 2034, 29   distinct nodes per level
asym (height 8):  1, 48, 2080, 122512, 1351566, 25577, 391, 6      distinct nodes per level
```

against the already-measured average total node-visit count for one
`one-Exact` query (from the whole-descent cache's hits+misses, divided
by probe count): **~36,800/query on `sym`, ~21,900/query on `asym`**.
`sym`'s level 4 alone holds **5,142,944** distinct nodes -- well over
*100x* every node a single query touches across its *entire* 8-level
descent, combined. That ratio is the whole story, and it's a direct,
measured consequence of the DAG dedup that makes `PathFst*` smaller than
`Heap` in the first place: the same shared node is stored once but
reachable from many different `S` bindings, so "distinct nodes at a
level" and "nodes one query's root actually reaches" are nowhere close
to the same quantity. Teddy and simdjson sweep every byte of their input
because every byte of a byte stream genuinely needs classifying --
nothing is already known-irrelevant before it's read. A level of this
DAG has no equivalent property: the overwhelming majority of it is
provably unreachable from any one query's root, but *which* nodes those
are isn't recoverable from position or level alone (the same point
raised when "skip the frontier entirely" was considered a few turns
earlier) -- so a position-based bulk sweep necessarily does work on
nodes that were never going to be visited, and the ratio above says
that's the overwhelming majority of the work.

Built and measured anyway, instead of trusting the ratio alone -- 20/20
correctness probes matched `restricted_descent` on every domain and
shape tested, so the implementation itself is sound:

```
shape               best existing variant   stream baseline   sweep (this prototype)
one-Exact sym (10M)    803.0us                1528.8us          31,863.7us   (~40x / ~21x slower)
one-Exact asym (3M)    459.7us                 873.1us          10,424.3us   (~23x / ~12x slower)
two-Exact sym (10M)      8.1us                  16.1us          21,979.9us  (~2700x / ~1370x slower)
two-Exact asym (3M)      5.1us                  10.0us           4,883.0us   (~960x / ~490x slower)
```

`two-Exact`'s regression is far worse than `one-Exact`'s, even though
`two-Exact` visits *fewer* real nodes per query -- consistent with the
ratio argument: the sweep's cost is roughly *fixed per level*
(independent of query shape), so it dominates far more when the useful
work it's paying for is smaller. Measured at reduced iteration counts
(5-20 probes, not the usual 500-2000) given the predicted and confirmed
per-call cost; not worth the wall-clock time to run the full comparison
at the usual scale once the first few probes already showed this.

So: the proposal's engineering *pattern* (separate a cheap bulk
classification pass from a slower confirm pass) is real and well
established in the literature, and the implementation concern it raised
(records aren't a flat fixed-stride array, so "one sweep" needs either
precomputed addresses or a storage redesign) was a genuine, worth-
checking detail, not a non-issue. But the pattern's precondition --
"most of what you're about to classify will actually get used, or at
least the ratio is close enough that batching pays for itself" -- is
false here by a wide, measured margin, for a structural reason specific
to this project's DAG dedup rather than a tuning problem. `restricted_descent_sweep_visit`
is kept in `src/path_fst.rs` as a documented negative result, same
convention as every other rejected prototype in this file; `level_addrs`
is kept on `PathFstDeltaPow2Bfs` since removing it would mean rebuilding
it from scratch to re-run this experiment, at the cost of one extra
`u64`/node this struct already wasn't trying to minimize relative to the
production default.

### A naive rayon parallelization of plain `visit_rec`: wins where the old caution said it wouldn't

This project already has a cautionary tale on file about naive task-level
parallelism, from a different traversal: `Heap`'s query path, "Rayon
fan-out depth sweep" above. There, fanning out across tree levels with no
depth cutoff collapsed catastrophically once the frontier ballooned past
what the thread pool could usefully dispatch (one call took 8.47
*seconds* at `parallel_depth=4` on a cheap dataset) -- the lesson drawn
was that rayon's roughly-fixed per-task dispatch cost needs real
per-branch work to amortize against, and uncapped fan-out finds the point
where it doesn't fast.

Revisited directly on `PathFstDeltaPow2Bfs`'s plain DFS `visit_rec`
instead of assuming the old number transfers:
`restricted_descent_parallel` (`src/path_fst.rs`, gated behind the
existing `parallel` feature, same as every other rayon use in this
crate) decodes one node's accepted children sequentially (cheap: one
`popcount`, one small `read_var` each), then hands the *recursive
descent* into each child to `rayon::join`, splitting the child list in
half and recursing on both halves, all the way down, with no depth or
fanout cutoff at all -- deliberately as naive as the earlier `Heap`
experiment, to see where *this* traversal's version of the collapse
happens rather than assume it's the same place. Returns a `Vec<u32>`
directly instead of streaming through `restricted_descent_visit`'s
`ControlFlow`-based `emit`: early-exit doesn't have an obvious meaning
once branches are running on different threads, so this prototype
sidesteps that question rather than guess at an answer.

`rayon::join` is why this is safe to run at all despite being fully
uncapped: it's only ever a *candidate* for running on another worker
thread (rayon's work-stealing pool falls back to inline/sequential
execution when no thread is free to steal the work), so it degrades
toward plain sequential recursion under load instead of exploding into
one OS thread per call the way a literal `std::thread::spawn` per node
would. That's a real, structural difference from the failure mode the
`Heap` experiment hit.

Measured (0 mismatches against `restricted_descent` on every probe,
reduced correctness-probe count given the cost of a full 500-probe run
-- 5 probes for the two expensive large-domain combos, 20 for the cheap
`small`-domain ones): the result is the opposite of the `Heap`
precedent, not a repeat of it --

```
shape               best existing (single-threaded)   stream baseline   naive parallel
one-Exact sym (10M)    803.0us                           1528.8us          521.96us   (1.54x faster than best, 2.93x faster than baseline)
one-Exact asym (3M)    459.7us                            873.1us          362.58us   (1.27x faster than best, 2.41x faster than baseline)
two-Exact sym (10M)      8.1us                             16.1us           29.90us   (3.69x slower than best, 1.86x slower than baseline)
two-Exact asym (3M)      5.1us                             10.0us           27.81us   (5.45x slower than best, 2.78x slower than baseline)
```

`one-Exact` -- real per-branch work, deep trees, the shape this whole
investigation has targeted -- genuinely wins, beating every
single-threaded variant measured so far, naive uncapped fan-out and all.
`two-Exact` repeats the `Heap`-side lesson exactly: cheap per-branch
work, dispatch overhead dominates, naive fan-out is a clear loss. Same
shape of result as the old caution, just confirming it holds here too
for the cheap shape, while being a genuine surprise for the expensive
one (the old entry's depth-1-is-usually-best finding doesn't transfer
directly -- this traversal's fully uncapped version still wins on
`one-Exact`, not just an early-cutoff version of it).

`perf stat` on the two winning combos (aggregate across all of rayon's
worker threads, not one core -- a materially different metric now that
real OS-level concurrency is involved, not a single-core latency-hiding
trick like every earlier entry in this investigation):

```
shape               aggregate cycles/call (parallel)   single-threaded best's cycles/call   ratio
one-Exact sym          ~8,647,867                         ~3,252,006 (layoutbfs)                2.66x more aggregate cycles
one-Exact asym         ~11,775,172                        ~1,868,704 (layoutbfs)                6.30x more aggregate cycles
```

(`two-Exact`'s aggregate-cycle deltas came out *negative* at these probe
counts -- the real signal, tens of microseconds times 20 calls, is
smaller than the run-to-run noise floor from rayon's own thread-pool
startup jitter across separate process invocations. The wall-clock
result for `two-Exact` is unambiguous regardless; this is a limit of the
baseline-subtraction methodology at this probe count for a
multi-threaded, cheap-per-call workload, not a contradiction of it.)

The honest read: this isn't a free win. `one-Exact` sym spends roughly
2.66x the total CPU-cycles of the best single-threaded variant to buy a
1.54x wall-clock improvement -- call it needing on the order of 4 cores'
worth of concurrent work to net less than 2x latency; asym's ratio is
worse, roughly 8 cores' worth of aggregate work for a 1.27x latency win.
Legitimate if latency is what's being optimized for and spare cores are
genuinely idle, but it is trading throughput for latency, not creating
free throughput -- a real cost, not a rounding error, and one every
earlier entry in this investigation (all single-threaded) didn't have to
account for.

Not wired in as a default: it's `parallel`-feature-gated, uncapped (no
depth/size heuristic yet, matching the deliberately naive brief), and
returns a `Vec` instead of streaming. The obvious next step the `Heap`
precedent suggests -- a depth or fanout cutoff, falling back to
sequential `visit_rec` below some threshold, closer to how `tree.rs`'s
own build-time fan-out already gates on `BUILD_PARALLEL_MIN_POINTS` --
hasn't been tried yet; this entry is the naive baseline it would be
measured against.

### Depth-cutoff follow-up: a real further win on `one-Exact`, and a sharper lesson on `two-Exact`

Built the obvious next step: `restricted_descent_parallel_cutoff`
(`src/path_fst.rs`, same `parallel` feature gate) takes a
`parallel_depth` parameter -- `rayon::join` fan-out is only used for the
first `parallel_depth` *tree levels* below the root; everything deeper
falls back to the plain sequential `visit_rec` (via a new
`visit_sequential_to_vec` helper that just collects `visit_rec`'s own
`ControlFlow` stream into a `Vec`, reusing the already-correct
traversal rather than re-deriving it). `parallel_depth` counts actual
tree-level descents, not bisection splits -- halving one level's sibling
list in `parallel_collect_cutoff` doesn't consume budget on its own,
only a child actually moving to the next level does -- so it's directly
comparable to `Heap`'s own `parallel_depth` from the "Rayon fan-out
depth sweep" entry, not a differently-shaped knob reusing the name.

Swept `parallel_depth` from 0 (fully sequential) to 4, same methodology
as that earlier entry, 0 mismatches against `restricted_descent` at
every depth tested:

```
one-Exact sym (10M)     depth=0: 1151.2us   depth=1: 342.7us   depth=2: 268.2us*   depth=3: 283.0us   depth=4: 591.2us
one-Exact asym (3M)      depth=0: 706.7us   depth=1: 214.5us   depth=2: 185.7us*   depth=3: 241.5us   depth=4: 315.6us
```
(* reproduced on a second run: 275.7us / 175.6us -- consistent with the
first, not a fluke.)

`parallel_depth=2` is the sweet spot on both domains -- past it, the
same collapse the `Heap` precedent described sets back in (depth=4 is
already worse than depth=1). At its sweet spot, this beats *both*
reference points decisively: **~1.9-2.0x faster than the fully naive
version** (268.2us vs. 521.96us on `sym`, 185.7us vs. 362.58us on
`asym`) and **~2.5-3.0x faster than the best single-threaded variant**
(803.0us / 459.7us). The naive version's win over single-threaded code
undersold what was available -- bounding the fan-out depth doesn't just
avoid a regression, it roughly doubles the win again.

`two-Exact` tells a sharper version of the same story, not a different
one:

```
two-Exact sym (10M)      depth=0: 18.9us*   depth=1: 22.7us   depth=2: 23.9us   depth=3: 28.1us
two-Exact asym (3M)      depth=0: 11.9us*   depth=1: 25.2us   depth=2: 23.1us   depth=3: 53.0us
```
(* best at every domain tested.)

Where `Heap`'s old precedent found `parallel_depth=1` usually best with
real (if modest) gains available, this traversal's `two-Exact` has *no*
useful depth at all -- `depth=0` (parallelism fully disabled) wins
outright, and `depth=1` is already a clear loss. Consistent with the
naive version's finding (dispatch overhead dominates a shape this
cheap), just with zero tolerance instead of a small one: there's no
depth at which `two-Exact` has enough real per-branch work to clear
even one `rayon::join`'s coordination cost.

Tried to cross-validate `sym`/`asym` one-Exact's depth=2 win with `perf
stat`'s aggregate-cycle methodology, same as every single-threaded entry
in this investigation -- and it didn't produce a trustworthy number.
Baseline-subtracted aggregate cycles at `depth=2` came out *higher* than
the fully naive version's (~24.8M/call on `sym`, ~4.8M/call on `asym`,
vs. the naive version's ~8.6M/~11.8M), which would require an
implausible >20 cores' worth of simultaneous activity given this
machine's 12 -- not a real result. The likely explanation: rayon's
worker threads stay alive and can spin before parking between calls:
once a persistent thread pool exists, "cycles consumed by the process"
over a measurement window includes idle-but-spinning worker threads'
cycles, not just real traversal work, and that idle-spin cost doesn't
cleanly cancel out of a baseline subtraction across separate process
invocations the way it did for every single-threaded traversal measured
so far. Not independently confirmed (would need `perf stat --per-thread`
or sampling, not attempted) -- flagged honestly as an open methodology
gap rather than trusted at face value. Wall-clock is the metric actually
being optimized for here (latency, not throughput), it's clean and
reproduced across repeat runs, and it's what every number in the tables
above is based on; the aggregate-cycle cross-check this investigation
has relied on throughout simply doesn't transfer cleanly to a
persistent-thread-pool workload, and that's a real limit of the method,
not a hidden problem with the result.

Net effect: `one-Exact` wants `parallel_depth=2`; `two-Exact` wants
`parallel_depth=0`. One fixed constant can't serve both -- whatever
wires this in for real would need to pick a depth from something about
the query (expected result-set size, or simply which BGP shape it is),
the same unresolved dispatch question the frontier/window-BFS variants
already raised for the `small`/`zipf` regression. Still not wired in
anywhere; `restricted_descent_parallel_cutoff` is kept in
`src/path_fst.rs` alongside the naive version, both `parallel`-gated,
both unused outside this file's own measurement drivers.

### Size instead of depth as the cutoff signal: a real idea, measured wrong for this traversal

Natural follow-up question: depth isn't the only plausible cutoff
metric, and `tree.rs`'s own build-time fan-out already gates on a
*size* signal (`BUILD_PARALLEL_MIN_POINTS`, not a depth), so maybe the
right local decision is "only fan out when a node actually has more
than `x` real children to explore," re-checked fresh at every node,
rather than a fixed, globally-spent depth budget.

`restricted_descent_parallel_size_cutoff` (`src/path_fst.rs`, same
`parallel` feature gate) implements exactly that: `rayon::join` is only
used when a node has more than `min_parallel_size` real children; at or
below that, every child in the list is processed by a plain sequential
loop, but each child's *own* fan-out is re-evaluated against the same
threshold when its turn comes (unlike the depth-cutoff version, where
once the budget hits zero nothing below it can parallelize no matter
how wide any single node's own fan-out turns out to be).

Swept `x` from 2 to 64, 0 mismatches against `restricted_descent` at
every value tested:

```
one-Exact sym (10M)   x=2: 967.8us   x=4: 1495.1us   x=6: 926.0us   x=8: 1197.5us   x=12: 794.2us   x=16: 1803.5us   x=24: 1839.4us   x=32: 1827.6us   x=48: 1896.2us   x=64: 1864.7us
one-Exact asym (3M)   x=2: 551.3us   x=4:  674.8us   x=6: 516.4us   x=8:  593.4us   x=12: 613.3us   x=16: 1216.3us   x=24: 1217.7us   x=32: 1238.7us   x=48: 1180.1us   x=64: 1288.0us

two-Exact sym (10M)   x=2:  43.2us   x=4:   24.3us   x=8:   24.7us   x=16:   24.7us   x=32:   24.2us   x=64:   24.5us
two-Exact asym (3M)  (same shape -- every x tested lands worse than depth=0's 11.9us)
```

Worse than the depth cutoff's sweet spot at *every* value tried, on
both shapes. The best size-cutoff result on `one-Exact` (`sym` x=12:
794.2us, `asym` x=6: 516.4us) doesn't even beat the fully naive, uncapped
version from the previous entry (521.96us / 362.58us) -- let alone the
depth-cutoff's 268.2us / 185.7us -- and past `x=16` it gets
*catastrophically* worse (1.2-1.9ms, worse than even `depth=0`'s fully
sequential baseline), because once the threshold exceeds the tree's own
maximum per-node branching factor (`K_SQUARE`, capped at 64, and in
practice usually well under that except right at the one level where a
pinned dimension's bits exhaust), no node anywhere ever clears it and
the traversal is sequential everywhere, with the added fixed cost of
checking the threshold at every level for nothing.

Worked out *why* size doesn't transfer from the build-time case, not
just that it doesn't: at build time, a node's local fan-out (how many
active children groups remain) is a reasonable proxy for how much work
is downstream, because fewer surviving children at a split generally
means fewer total points left to place. In this traversal, that
correlation doesn't hold -- a node's fan-out counts how many *values of
one dimension* are live here, which says nothing about how much
*further exploration* (remaining free dimensions, remaining tree depth)
sits below any one of those children. Two children of the same node can
have identical, large, expensive subtrees below them regardless of
whether there are 2 siblings or 20 -- so gating on local fan-out count
either forks indiscriminately too often (small `x`, not much better than
naive) or, more damagingly, sequentializes real, substantial,
genuinely-parallelizable work whenever it happens to arrive in a small
sibling group (larger `x`) -- actively discarding achievable overlap,
not just failing to find overhead to cut. Depth, by contrast, really
does correlate with downstream work here: near the root, every child's
subtree (whatever its immediate fan-out) still has many levels left to
explore; near the leaves, there's little left regardless of how wide any
one node's fan-out happens to be. That's the real reason depth was the
right metric for *this* traversal and size is the right metric for
*build* -- not an accident of tuning, a difference in what each
traversal's local shape actually predicts about remaining work.

One honest implementation caveat: the "don't fork" path in this
prototype still routes through `visit_parallel_size_rec`'s own
per-level `Vec`-collecting decode, re-evaluating the threshold at every
level, rather than falling through to the cheap, allocation-light
`visit_rec` the way the depth-cutoff version's exhausted-budget path
does via `visit_sequential_to_vec`. That costs a little extra at every
`x` (visible in `two-Exact`: even `x=64`, which can never fire given the
tree's branching cap, still lands at ~24.5us against `depth=0`'s
18.9us) -- but it's nowhere near large enough to explain `one-Exact`'s
2-4x gap against the depth cutoff's sweet spot, which is the structural
effect described above, not an implementation artifact.

`restricted_descent_parallel_size_cutoff` is kept in `src/path_fst.rs`
as a documented negative result, same convention as every other
rejected prototype in this file -- depth, not local fan-out, is this
traversal's real cutoff signal.

### Dead ends and `rayon::join`: a real cost, fixed with `rayon::scope`, smaller than first guessed

A question raised about every `parallel*` variant above: what actually
happens to a branch that hits a dead end? Every one of them, so far,
still pays a real cost for it -- `rayon::join(a, b)` is a
value-returning rendezvous, so even a branch that contributes nothing
(`relevant == 0`, no real children at all) has to run to completion,
return an empty `Vec::new()`, and have that empty `Vec` `extend`ed into
its sibling's result, purely because `join`'s shape requires something
to merge regardless of whether there's anything in it.

A thread genuinely can't be dropped without being waited on *at all* --
the top-level call still needs every branch's contribution before it
can return a complete answer, and there's no way to know in advance
that a branch is a dead end without running it. But the *merge*
mechanism doesn't have to be the thing that does the waiting.
`restricted_descent_parallel_scoped` (`src/path_fst.rs`, same
`parallel` feature gate, same depth-cutoff shape as
`restricted_descent_parallel_cutoff`) replaces `rayon::join`'s "both
sides return a `Vec`, caller merges them" with `rayon::scope` +
`Scope::spawn`: every branch writes *directly* into one shared
`Mutex<Vec<u32>>` if it has anything to write, and the only
synchronization point is the single outer `scope` call waiting for
every spawned task (however deeply nested) to finish. A dead end now
costs exactly one function call that returns immediately after checking
`relevant == 0` -- no allocation, no lock, no merge, nothing to join
back. Only a branch that actually reaches accepted leaves touches the
`Mutex`, and only once per branch (local results are gathered into a
plain `Vec` first, then pushed in one lock acquisition), not once per
emitted value.

Measured carefully this time -- the first same-depth comparison looked
like a decisive 20-26% win from a single sample each, which turned out
to be exactly the kind of thing this project's own discipline exists to
catch. Multiple samples at the shared sweet spot, `parallel_depth=2`:

```
one-Exact sym (10M)    scoped: 198.5, 265.0, 213.2, 278.3, 242.4us (avg 239.5us)   join: 268.2, 275.7, 235.1, 225.9, 240.3us (avg 249.0us)
one-Exact asym (3M)    scoped: 147.7, 203.3, 155.4, 194.2us (avg 175.2us)          join: 185.7, 175.6, 203.4, 185.6us (avg 187.6us)
```

A real, consistently-signed improvement once averaged -- `scoped` beats
`join` by about 3.8% on `sym` and 6.6% on `asym` -- but nowhere near the
one-sample-each 20-26% first glance suggested, and well within the
~20-30% run-to-run noise either mechanism shows at this iteration count
on its own. The direction is trustworthy (both domains agree, across
several independent samples each); the exact magnitude isn't something
to over-read from five samples.

`two-Exact` adds a genuine surprise, not just a repeat of "still too
cheap to parallelize": at `parallel_depth=0` (no real forking at all),
`scoped` is roughly *2x worse* than `join`'s equivalent (`sym`: 36.8-45.1us
vs. 18.0-19.1us, reproduced, not noise). `rayon::scope` itself has a
real, measurable fixed entry/exit cost -- registering a scope with the
thread pool and waiting on its completion latch -- that's paid even when
zero tasks are ever spawned, unlike `rayon::join`'s more local,
cheaper-when-idle cost. For a workload this cheap (tens of microseconds
total), that fixed cost dominates outright. But `scoped` has its *own*
different sweet spot, not zero -- at `parallel_depth=2`, where real
(if modest) forking starts happening, `scoped` recovers to 16.3-20.3us
(avg ~17.8us), matching or very slightly beating `join`'s own best
(`depth=0`, avg ~18.7us). Comparing the two mechanisms at the *same*
fixed depth for `two-Exact` would have been the wrong comparison --
each needs its own depth found independently, because `scope`'s
non-zero floor cost means "no parallelism at all" isn't actually
`scoped`'s cheapest option the way it is for `join`.

Net: the dead-end-avoidance idea is real and validated, not just
plausible-sounding -- it measurably helps on the shape with real work
to parallelize, and at worst (once each mechanism's own best depth is
used, not a shared guess) roughly matches the join-based version even
on the shape that wants no parallelism. It just isn't the dramatic,
free win "don't join dead threads" suggested before measuring
carefully -- `rayon::scope`'s own fixed cost is a real, separate price,
and the join-based version's dead-end cost (one empty `Vec` allocation
and an empty `extend`) was already small enough that removing it only
buys a few percent, not a multiplier, once genuine work dominates.
`restricted_descent_parallel_scoped` is kept in `src/path_fst.rs`
alongside both cutoff variants, same `parallel` gate, still not wired
in anywhere -- a small, real, measured improvement on the shape that
already won, not a new shape of win.

### Literature check, a branching-factor variant, and the real `perf` breakdown this project's rayon work was missing

Looked for published guidance on rayon/fork-join tuning before trying
more variants by feel. Three findings worth recording:

- **`rayon::join`'s "sequential fallback" is not a special mode -- it's
  the normal case.** Per `rayon-core`'s own docs: calling `join(a, b)`
  runs `a` on the current thread and makes `b` available to steal; if
  nobody steals it before `a` finishes, the same thread just runs `b`
  itself, with no cross-thread handoff at all. This follows Cilk's
  "work-first principle" (the MIT Cilk project, 1990s): the cost of
  exposing potential parallelism should fall on a thief, not be paid
  unconditionally by the task itself. `rayon::scope`'s docs are
  explicit about *not* having this property the same way: `Scope::spawn`
  tasks are heap-allocated (`rayon-core`'s own `HeapJob`), where `join`
  stays entirely on the stack -- a real, documented reason to expect
  `scope` to cost more per call, not a hypothesis, which matches what
  was measured two entries up.
- **Grain-size/threshold tuning (what `parallel_depth` already is) is
  the textbook-correct lever, and the textbook also warns it's
  brittle.** Cilk/TBB literature calls this "grain size": too fine loses
  to overhead, too coarse underuses idle cores, and a *fixed* threshold
  "over-fits... to a particular dataset, platform, and calling context"
  (Tzannes et al., *Lazy Binary Splitting*, 2010). LBS's actual proposal
  -- decide whether to split further based on whether a steal is
  actually happening at runtime, not a hard-coded depth -- is exactly
  the kind of mechanism that could resolve this investigation's own
  unresolved "`one-Exact` wants depth=2, `two-Exact` wants depth=0"
  dispatch problem without hand-tuning either number. Not implemented
  here (it's a real scheduler, not a quick prototype), but it's the
  named, published answer to the exact problem this document has been
  manually tuning around since the naive-parallelism entry.

**Branching factor, tried directly.** `restricted_descent_parallel_scoped_chunked`
(`src/path_fst.rs`) caps how many tasks get spawned *from one node* at
`max_branch`, chunking a wide fan-out into groups instead of one
`Scope::spawn` per child (each `spawn` is a heap allocation per the
point above, so fewer, larger spawns should cost less). Swept
`max_branch` from 2 to 256 at the already-good `parallel_depth=2`:
results bounced between 205-280us on `sym` with no consistent trend
once past 8, because at `depth=2` the real per-node fan-out this early
in the tree rarely exceeds single digits anyway -- most tested values of
`max_branch` never actually trigger chunking, so the "differences" are
noise, not signal. More tellingly: tried `max_branch=8` at `depth=4`
(the setting that collapsed catastrophically two entries up) to see if
capping per-node width could rescue a deeper budget -- it didn't
(1072.7us, essentially unchanged from unbounded `scoped`'s own 1129.8us
at the same depth). The collapse is driven by *total spawned-task count
compounding across levels* (even an 8-way cap still multiplies out to
thousands of tasks by 4 levels deep), not by how wide any single node's
own fan-out is -- confirming depth remains the dominant axis, branching
width a minor, mostly-inert refinement on top of it.

**The `perf` breakdown this investigation was missing.** Every earlier
`parallel*` entry measured wall-clock (and one attempt at
baseline-subtracted aggregate cycles that turned out unreliable for a
persistent thread pool) without ever actually looking at *where* the
cycles go. `perf record -F 999` (no call-graph, enough iterations that
the one-time tree build is a small fraction of total samples) on
`restricted_descent_parallel_scoped` gives that directly, symbol by
symbol, at both the good and the collapsed configuration:

```
parallel_depth=2 (wall-clock sweet spot, ~227us/call):
  76.49%  visit_rec (the real sequential-fallback traversal)
   4.07%  visit_scoped_rec (the real parallel-dispatch/decode work)
  ~8.2%   collect_rec_bfs + its hashing (tree build, not the query loop)
   0.79%  crossbeam_deque::Stealer::steal
   0.53%  rayon_core::latch::CountLatch::set
   0.33%  rayon_core::job::HeapJob::execute
  ~4.1%   malloc/free/memset/memmove (Vec growth for children lists, HeapJob boxing)
   0.07%  Mutex::lock_contended  <- essentially zero contention on the shared sink

parallel_depth=4 (the collapsed configuration, ~1052us/call):
  38.30%  visit_scoped_rec
  27.40%  rayon_core::latch::CountLatch::set   <-- was 0.53% at depth=2
   7.81%  collect_rec_bfs (tree build)
   2.97%  crossbeam_deque::Worker::pop
   2.59%  rayon_core::registry::Registry::inject_or_push
   1.87%  visit_rec
   1.82%  rayon_core::scope::ScopeBase::execute_job_closure
   0.78%  rayon_core::job::HeapJob::execute
```

At the good depth, real traversal work (`visit_rec` + `visit_scoped_rec`)
is ~80% of all samples once the one-time build is excluded, and every
individual rayon-machinery symbol is under 1% -- there's no single
dominant overhead source because there isn't much overhead at all. At
the collapsed depth, `CountLatch::set` alone -- the function that does
nothing but decrement a scope's outstanding-task counter and check
whether it's reached zero -- jumps to 27.40%, becoming a *larger* cost
than all of `visit_rec` and `visit_scoped_rec` combined would need to be
competitive. That one symbol, by itself, is the quantified fingerprint
of "too many small tasks": not stealing, not allocation, not lock
contention (both depths show negligible `Mutex` time) -- just the
sheer number of times a completion counter has to be touched once
task count compounds across levels. This is the concrete mechanism
behind every depth-sweep collapse measured in this whole investigation,
not just a label for it.

**Idle-time probing, attempted and blocked.** `perf sched record` (the
standard tool for per-thread wait/run-time histograms, via `perf sched
timehist`) requires read access to the kernel's scheduler tracepoints;
this sandbox doesn't have it (`No permissions to read
/sys/kernel/tracing//events/sched/sched_switch`). Not worked around --
the `perf record` cycle-sampling breakdown above already answers the
question it would have answered (where the non-useful cycles go), just
via sampled instruction pointers instead of scheduler events. Flagged
honestly as attempted rather than silently skipped.

`restricted_descent_parallel_scoped_chunked` is kept in `src/path_fst.rs`
alongside every other `parallel*` variant, `parallel`-gated, not wired
in anywhere -- a real lever (fewer heap-allocated spawns) that doesn't
move the needle much at the depth this traversal already wants, because
that depth's own fan-out rarely gets wide enough to exercise it.

### Lazy Binary Splitting, implemented for real: two bugs found, a clean negative result either way

The previous entry's literature check named Lazy Binary Splitting (LBS,
Tzannes et al. 2010) as the published answer to this investigation's
unresolved dispatch problem ("`one-Exact` wants `parallel_depth=2`,
`two-Exact` wants `parallel_depth=0`, no fixed constant serves both").
Worth actually building, not just citing.

`restricted_descent_parallel_lazy` (`src/path_fst.rs`, `parallel`
feature gate) adapts LBS via `rayon::join_context`: each closure's
`FnContext::migrated()` reports whether it was actually picked up by a
different worker thread, the same signal rayon's own parallel-iterator
bridge (`plumbing::bridge_producer_consumer`) uses internally to decide
whether to keep splitting. The rule, in its first version: a node's
children only get bisected if the call that reached this node was
itself the product of a real migration; otherwise the whole remaining
subtree finishes through the plain sequential `visit_rec`.

**First measurement, and the first real bug.** On `sym` `one-Exact`:
~1.1-1.3ms/call -- indistinguishable from fully sequential (`depth=0`'s
~1.15ms). Essentially zero parallelism was happening. The cause wasn't
subtle once looked at directly: `one-Exact` spends its first several
tree levels on a single live path (the pinned dimension's bits haven't
exhausted yet), so most nodes near the root have exactly one real
child. The first version treated "nothing to split here" identically to
"not migrated" -- permanently abandoning any chance of splitting the
instant it passed through that corridor, typically at level 1, well
before the tree's real fan-out region several levels deeper. Fixed by
recursing straight through a single-child node, carrying `migrated`
forward unchanged instead of treating it as a missed opportunity.

**Second measurement, and the second real bug.** With that fix, instead
of collapsing to sequential, it now collapsed to the *other* known-bad
regime: ~1.1-1.3ms/call again, but for a different reason. Added
temporary atomic counters (removed once the data below was collected)
around every `join_context` call to find out directly, rather than
guess: **75-80% of attempted splits were actually migrating**. With 12
idle cores and a workload this size, there's simply enough spare
capacity that the adaptive signal kept saying "yes, split more" almost
everywhere -- reproducing the exact `CountLatch`-dominated collapse
`parallel_depth=4` hit two entries up, just reached adaptively instead
of via a bad constant. Migration *alone*, with no floor, doesn't
self-limit when idle capacity is abundant -- exactly why rayon's own
internal splitter combines migration status *with* a length-based
floor, not migration alone (confirmed in the literature search, missed
in the first implementation). Fixed by adding `max_depth_from_root`:
splitting now requires both migration *and* being within the first
`max_depth_from_root` tree levels, mirroring `parallel_depth`'s own
depth-from-root semantics exactly (not a depth-remaining-to-leaf floor,
which was tried first and measured to barely constrain anything, since
most of an 8-level tree satisfies "at least 2 levels remain").

**Third measurement, with both bugs fixed, at `max_depth_from_root=2`
(the already-known-good window):**

```
one-Exact sym (10M)    lazy: 670-867us        join-cutoff depth=2: 268.2us    scoped depth=2: ~239.5us (avg)
one-Exact asym (3M)    lazy: 439-523us        join-cutoff depth=2: 185.7us    scoped depth=2: ~175.2us (avg)
two-Exact sym (10M)    lazy: 24.4-24.8us       join-cutoff depth=0: ~18.7us (avg)   scoped depth=2: ~17.8us (avg)
```

Correct (0 mismatches on every probe), and clearly better than either
unfixed version -- but still 2.5-3x worse than the simple, unconditional
depth-cutoff at its own best setting, on the shape that matters most.
Swept `max_depth_from_root` from 1 to 4 to check this wasn't just a bad
window choice: 1 gives 1.38ms (too shallow, barely anything splits -- 15
attempts over 15 queries), 2-3 cluster around 670-870us, 4 collapses
back toward the uncontrolled regime (13,245 attempts, 1.21ms) -- `lazy`
has its own sweet spot, and it's still worse than eager's.

**Why, worked out rather than left as a number.** Within a window
*already known* to be worth splitting unconditionally (the first 2
tree levels, established by the earlier depth sweep), the migration
gate doesn't add anything -- it can only ever decline a split the eager
version would have taken for free. Every instance where `migrated`
happens to read `false` at a node that *was* worth splitting is a real
opportunity the adaptive version gives up and the eager one never
risks. LBS's actual value proposition -- avoiding tuning effort when
the right amount of parallelism is unknown or varies unpredictably --
doesn't apply here, because the right window *isn't* unknown: the
depth sweep already found it, cheaply, once. Adaptivity is solving a
problem (what's the right threshold?) this project had already solved
by a different, simpler method (sweep and measure) before trying LBS at
all -- so here, it can only add noise on top of an already-right
answer, not improve on it. LBS would plausibly earn its keep in a
setting where that threshold genuinely varies at runtime in ways a
fixed sweep can't anticipate; this traversal's `one-Exact`/`two-Exact`
split isn't that kind of problem -- it's a small, enumerable set of
query shapes, each with its own stable best constant, better served by
picking a constant per shape (still unresolved, but a dispatch problem,
not a scheduling one) than by a scheduler re-deciding moment to moment.

`restricted_descent_parallel_lazy` is kept in `src/path_fst.rs` as a
documented negative result, same convention as every other rejected
prototype in this file -- measured honestly against the much simpler
thing this project had already found, and losing to it.

### Re-checked against the paper directly: it isn't actually Lazy Binary Splitting

Asked directly whether the implementation above is a faithful LBS
implementation, not just inspired by the name. It isn't, and the
previous entry's wording overstated that. Worth correcting precisely:

Tzannes et al.'s actual LBS processes a flat iteration range
*sequentially by default* and only splits in direct, synchronous
response to a thief that is *right now* waiting for work -- a live
demand signal the owner thread polls cheaply as it loops. No split
happens until that concrete event occurs; the owner never pre-decides a
split point in advance.

What's built here is structurally different: `rayon::join_context`
still *eagerly* pre-divides a node's children in half and recurses into
both halves immediately, exactly like TBB's eager binary splitting
(which the LBS paper was written to replace) -- the only adaptive part
is `FnContext::migrated()`, a *retrospective* signal reporting whether
the *previous*, already-completed call happened to get stolen, not a
live "is anyone waiting right now" check. This is the mechanism rayon's
own `plumbing::bridge_producer_consumer` uses internally (confirmed by
reading rayon's source, not assumed), and it's a reasonable, real
technique in its own right -- but it's eager splitting with an adaptive
stopping heuristic, not lazy splitting. `rayon`'s stable public API has
no primitive that exposes a live "a thief is waiting on me now" signal
the way genuine LBS needs; reaching that would mean either depending on
`rayon-core` internals not meant for external use, or writing a custom
cooperative scheduler from scratch -- a real undertaking, well beyond
what this investigation's "try implementing lazy binary splitting" was
scoped to cover. What got built and measured is accurately described as
*rayon's own adaptive-eager splitting heuristic, adapted from a flat
range to a tree* -- a legitimate, related technique, not the paper's
algorithm. The measured result (loses to the simple depth cutoff by
2.5-3x) stands either way; the correction is about what to call it, not
about the number.

**Optimization pass, and what `perf` showed.** Re-reading
`visit_lazy_rec` against the file's own other `rayon::join`-based
variants (`parallel_collect`, `parallel_collect_cutoff`) turned up one
real, avoidable cost: before calling `rayon::join_context`, the code
cloned both halves of `children` into fresh, owned `Vec`s
(`left.to_vec()`/`right.to_vec()`) before moving them into the two
closures. That pattern is necessary for `Scope::spawn` (used by
`restricted_descent_parallel_scoped`/`_chunked`), because a spawned
task is asynchronous relative to its caller and can't safely borrow a
stack-local `Vec` that might be dropped first. `join_context` has no
such requirement -- it's synchronous, like plain `join`, so `children`
is guaranteed to outlive the call, and the closures can borrow `left`/
`right` slices directly, exactly like every other `join`-based variant
in this file already does. Fixed by removing the clones.

Checked with `perf record` before and after, not just by reasoning that
it should help:

```
                                    before (.to_vec())   after (borrowed slices)
visit_rec (sequential fallback)         51.38%               51.58%
collect_rec_bfs + hashing (build)      ~16.2%                ~16.6%
visit_lazy_rec                          2.01%                1.98%
crossbeam/rayon coordination           ~3.3%                 ~2.9%
malloc/free/memset/memmove/unlink      ~6.4%                 ~6.4%
```

No measurable change, in either the allocation-related symbols
specifically or the wall-clock (670-880us before and after, same
range). The fix is still correct and worth keeping -- it removes real,
unnecessary work and brings this variant in line with the rest of the
file's own established pattern -- but `perf` says plainly it was never
where this traversal's time goes: only a handful of `join_context`
splits happen per query at `max_depth_from_root=2`'s shallow budget, so
removing their clones touches too little of the total work to move the
number. The dominant cost, unchanged by this fix and confirmed directly
rather than inferred, is structural: 51% of all samples land in the
plain sequential fallback, because most of an 8-level tree falls
outside a 2-level parallel budget by construction. That's the same
conclusion the previous entry reached by comparing wall-clock numbers --
`perf` now confirms it's not secretly an allocation or rayon-coordination
problem hiding underneath, it really is the shallow-budget architecture
itself. No `CountLatch` entry appears in either profile, confirming the
depth floor is doing its job: this version doesn't over-split the way
the unguarded migration-only version did.

### A per-thread slot queue, proposed directly: wrong the first time, then a real bug, then correct

A fourth, structurally different mechanism, specified directly: not one
shared, unbounded pool of work items, but exactly `N` fixed slots, one
per worker thread, each holding `(node, open_mask)` -- which node a
thread currently owns, and a bitmask of that node's accepted
transitions nobody has claimed yet. A thread that runs out of work at
its own slot looks at every *other* slot for an open bit, claims one
(closing it in the *victim's* slot), and continues DFS by publishing
the claimed child into its *own* slot -- the worked example given: a
thread finds another thread at a node with two children open, claims
one (clearing that bit where it found it), and starts exploring from
there by writing it into its own slot with its own first pick already
closed.

**First implementation, built from a misreading.** It used one shared,
growing `Mutex<Vec<Arc<WorkItem>>>` instead of `N` fixed slots --
structurally the same mistake as the original (also-rejected) shared
queue, just with a different claim mechanism underneath. Measured as a
severe regression everywhere (1.65-2.0ms on `one-Exact`, worse than
fully sequential; 94-140us on `two-Exact`, 5-8x worse than the best
mechanism found so far), with `perf` pinning it on real, measured lock
contention (`Mutex::lock_contended` at ~4.9%) and `Arc` allocation
(~10% in malloc-family symbols) -- because, with no depth or size
floor, every multi-child node anywhere in the tree paid full queue
machinery, not just a shallow window.

**Corrected to the actual design, and a real deadlock.** Rebuilt with
`N = rayon::current_num_threads()` fixed `Slot`s and a lock-free
`compare_exchange` loop on an `AtomicU64` mask for claims (no `Arc`, no
growing `Vec` -- a node's children are never collected at all; the raw
accepted-transitions bitmask *is* the published state, and a specific
child's address is decoded only once actually claimed). This version
hung indefinitely on every input, including the trivial `small` domain
with exactly one worker thread (`RAYON_NUM_THREADS=1`) -- a real
deadlock, not a race, reproducible with zero concurrency at all, which
made it easy to isolate. The bug: `dfs_slot_rec` recursed into a
claimed child by unconditionally republishing `slots[my_slot]` for
that child -- but if the node it had just claimed *from* still had
other open siblings sitting in that same slot, republishing silently
discarded them. Nothing pointed at them anymore, nobody (not even the
thread that orphaned them) could ever claim them again, and the global
`pending` units they were added under could never be decremented,
hanging the whole query forever. Fixed by checking whether a thread's
own slot is actually free before publishing into it -- otherwise
finishing that node's entire subtree by plain recursion instead (still
correct, just unshared at that level), re-checking at every level
below so sharing resumes the instant the slot genuinely frees up.

**Still wrong once real concurrency was involved: a second, subtler
race.** With the deadlock fixed, single-threaded runs passed, but
multi-threaded runs produced real, reproducible `index out of bounds`
panics inside `storage.rs` -- memory corruption from a bad address, not
a hang. The cause: `slot_try_claim` split "claim a bit" (a lock-free CAS
on the mask) from "read which node that bit belongs to" (a *separate*,
later-acquired lock on the position). The reasoning that this was safe
-- "position only changes after the mask reaches zero, so a successful
CAS always has a valid position behind it" -- had a real hole: mask
reaching zero only means no *more* bits are claimable, not that every
thread which already claimed an *earlier* bit has finished reading the
position that went with it. A thief could win the CAS, stall for any
reason before reading position, and in that window the slot's owner --
having legitimately observed the mask at zero -- could republish the
slot for a completely different node. The stalled thief then reads the
*new* position paired with a `qi` that was only ever meaningful for the
*old* one, decodes a bogus delta, and computes a garbage address. Fixed
by merging `position` and `open_mask` into one `Mutex<SlotData>` and
performing the whole "check mask, pick a bit, clear it, read position"
sequence as a single locked critical section -- giving up the
lock-free claim path, but removing the gap a stale read could land in
at all, which no amount of reordering the two independent atomics could
have done. Re-verified with repeated multi-threaded runs (5 consecutive
clean runs on every domain/shape combination, not just one) after the
fix, since the bug the first fix introduced only ever showed up under
genuine concurrency.

**Measured, finally correct:**

```
one-Exact sym (10M)    queue: 281-530us (avg ~424us)    join-cutoff depth=2: 268.2us
one-Exact asym (3M)    queue: 281-419us (avg ~343us)    join-cutoff depth=2: 185.7us
two-Exact sym (10M)    queue: 45-69us (avg ~63us)        best mechanism: ~18.7us
two-Exact asym (3M)    queue: 45-58us (avg ~52us)        best mechanism: ~17.8us
```

Correct, and clearly better than the first (misunderstood) attempt's
catastrophic numbers -- but still 1.6-3.4x behind the best hand-tuned
depth-cutoff mechanism, on every shape. `perf` on the corrected version
shows exactly why, and confirms the fixes actually worked: `Mutex::
lock_contended` dropped from ~4.9% (one global lock) to ~2.1% (`N`
independent per-slot locks, each contended by far fewer threads), and
malloc-family allocation dropped from ~10% to ~5.3% (no more `Arc`, no
more per-node `Vec` -- the stated design goal). `dfs_slot_rec` itself is
56.9% of samples -- real traversal work, including the "my slot is
busy, fall back to plain recursion" path inline. The remaining gap to
the hand-tuned cutoff isn't contention or allocation anymore (both
measurably fixed); it's the same lesson every other un-gated mechanism
in this investigation has produced: this one still does real
lock-and-check work at *every* level with multiple children, not just
a shallow window near the root, because nothing here knows in advance
which depth is worth sharing at -- the "only publish when free"
self-limiting rule prevents catastrophic over-sharing, but it's a
*correctness* safeguard, not a performance tuning mechanism, and it
doesn't substitute for one.

This whole entry is a direct demonstration of why this project insists
on measuring instead of reasoning from the design alone: a mechanism
that looked correct on paper (and compiled, and passed its first
single-node review) hid two independent bugs, one of which only
manifests with real concurrency and would never have been caught
without actually running it under load repeatedly. `restricted_descent_parallel_queue`
is kept in `src/path_fst.rs`, `parallel`-gated, correct and measured,
not wired in anywhere -- a legitimately different mechanism from every
other `parallel*` variant, worse than the best of them on this
traversal, but for a well-understood, now-verified reason rather than
an assumed one.

### From one slot per thread to one cell per thread *per tree depth*: an iterative matrix, two steal heuristics compared, and two debugging detours that turned out not to be new bugs

`restricted_descent_parallel_queue`'s one weakness was structural, not a
bug: a thread's own slot can only ever describe *one* node at a time, so
the moment a node has more than one open child and the thread's own slot
is still occupied by an ancestor, the whole rest of that node's subtree
falls back to plain, unshared recursion -- correct, but it means sharing
is only ever available at the single depth a thread happens to be
occupying at any instant.

The fix: stop using one cell per thread and start using one cell per
thread *per tree depth* -- a `threads x height` matrix where
`matrix[t][c]` is thread `t`'s own cell for the `c`-th level of its
*current* descent path (not real tree level `c` -- after a steal, a
thread's own column 0 can hold a node from any real level, which is why
each cell stores its own `real_level` rather than inferring it from the
column index). Every ancestor level along a thread's path is now its
own independently stealable cell, so the "my slot is occupied, fall back
to recursion" escape hatch the single-slot design needed simply has
nothing left to guard against -- an ancestor and its descendant never
compete for the same cell anymore. The recursion flattens out too: a
thread's "stack" is just its own row, walked with an explicit `depth:
Option<usize>` instead of the call stack, backing up one column at a
time on exhaustion and overwriting the vacated column with whatever it
claims from the column below -- exactly the mechanism proposed directly:
root -> 16 -> 32 -> 48, 48 exhausted, back up to 32's remaining open
child 56, write 56 over 48's now-unreachable (and harmless, since it had
nothing left open) slot.

Three refinements came out of reviewing the pseudocode before writing
any code, all adopted directly:

- **Bootstrap moved out of the per-thread loop entirely.** The root is
  a single node -- there's nothing to parallelize about visiting it --
  so a dedicated `matrix_initialize` step runs once, before any thread
  starts: publish the root into `matrix[0][0]`, then immediately claim
  and dispatch one of its own transitions into `matrix[0][1]`. This
  means no thread's *own* active position is ever real tree level 0;
  level 0 only ever exists afterward as an ordinary stealable backlog
  cell, found through the exact same scan every later steal uses,
  whether the root's fan-out is narrower or wider than the thread count.
  No per-thread `if my_id == 0` special case anywhere in the steady-state
  loop.
- **`open_mask` collapses the level mask permanently, at creation,
  once.** Each cell also caches the node's raw `word`, read from storage
  exactly once when the cell is published. `level_masks[level]` is
  consulted in exactly one place in the whole mechanism --
  `dispatch_claimed_bit`, at the instant a *new* child cell is created
  -- never touched again by `matrix_try_claim`, `matrix_peek_level`, or
  the worker loop for that cell's remaining lifetime, and the node's
  word never gets re-read from storage by a later claim against the
  same cell.
- **`debug_assert!` on the "publish never fails" invariant**, not a
  runtime-handled branch -- every publish call site already argues, by
  construction, that its target is free; a violation means that
  argument was wrong somewhere; treating it as a silent possibility
  would hide exactly the kind of bug this file's own history keeps
  finding.

**Two steal-order heuristics, compared as asked, nothing deferred
beyond what was agreed:** `prefer_shallow` picks the shallowest open
column across every other thread's row; its opposite picks the
deepest. Both use the same plain `O(threads x height)` linear
peek-then-claim scan with no per-row "anything open" summary bit --
that optimization was explicitly deferred pending a measurement showing
the scan itself matters, not built from reasoning alone. The
depth-threshold steal-gating idea (steal only past a certain depth) was
deferred too, as agreed, in favor of measuring these two first.

**A bug found immediately, in the bootstrap, not the steal mechanism:**
the very first correctness check (a tiny 8-triple, 4-entity synthetic
tree) came back short -- some siblings under the root's second accepted
child were silently missing. `matrix_initialize` credited `pending`
with `1` for "the root," not with `root_mask.count_ones()` -- one unit
per *bit* of the published mask, the convention every other publish
site in this mechanism (and the slot design before it) already follows.
Whenever the root had more than one accepted child, `pending` could
reach zero while real root-level siblings were still sitting unclaimed
in `matrix[0][0]`, ending the whole query early. Fixed by crediting
`root_mask.count_ones()` instead of `1`; the same tiny synthetic tree,
plus repeated runs against it, came back clean afterward.

**Two further "bugs" chased hard, and both resolved as tooling
mistakes, not mechanism defects -- worth recording exactly because they
looked, at each step, exactly like the two real concurrency bugs the
slot design had:**

1. A full sweep across every domain/shape/steal-mode combination came
   back with `small` timing out under `shallow`, and `small` producing
   *wrong, non-reproducible* results under `deep` -- different missing
   elements on every run, the unmistakable signature of a genuine race.
   Hours of targeted reproduction attempts followed: a hand-rolled
   synthetic domain at the exact triple count, the exact `uniform` seed
   and exact probe sequence the failing run used, fresh processes,
   oversubscribed `RAYON_NUM_THREADS` up to 96 on a 12-core machine,
   artificial background CPU load -- nothing reproduced it. The
   resolution turned out to be mundane: the fix above had been applied
   to `src/path_fst.rs` and verified against a *freshly rebuilt*
   `matrix_debug` scratch binary, but the large sweep reused an
   already-built `target/debug/examples/path_fst_bgp_perf` from
   *before* that fix -- cargo does not rebuild an example just because
   a library source file changed underneath it without another
   explicit build invocation naming that example. Every failure in that
   sweep was pre-fix object code; a clean rebuild of the actual binary
   under test passed every combination, every time, including repeated
   stress runs under heavy artificial contention.
2. A second, independent sweep (this time against a freshly rebuilt
   binary) still timed out -- but only on the `sym` domain (10M
   triples), on *both* query shapes and *both* steal heuristics
   identically, while the much smaller `small` domain and the larger
   -- but less numerous -- `asym` domain (3M triples) both passed
   cleanly. A domain-scale signature across every shape and heuristic,
   not a shape- or heuristic-specific one, pointed away from a logic
   bug: this investigation's own convention has always measured
   `parallel*` variants through the optimized `profiling` build
   (`target/profiling`), never plain `cargo build`'s unoptimized
   `target/debug` -- which is what the sweep had used throughout. Built
   under `--profile profiling` instead, the identical `sym`-domain runs
   that had timed out at 90 seconds in debug mode completed in under
   four seconds, 5/5 correctness probes passing, for every
   shape/heuristic combination.

Neither detour found a new mechanism defect; both are recorded anyway,
because at the moment each one surfaced it looked exactly as alarming
as the deadlock and the TOCTOU race the slot design actually had, and
distinguishing "a real bug under load" from "a stale binary" or "debug
build overhead at 10M triples" from the symptom alone is exactly the
kind of judgment call this investigation keeps having to make honestly
rather than assume either way.

**Measured, correct, on the `profiling` build (20-iteration timed
loop per cell, `iters` separate from the 5 fixed correctness probes):**

```
                  matrix(shallow)   matrix(deep)   parallelqueue
one-Exact small       115.7us         505.2us          63.9us
one-Exact sym         1061.2us        3610.8us         510.9us
one-Exact asym         769.2us        2276.6us         354.1us
two-Exact small         37.7us          49.3us          32.2us
two-Exact sym            62.3us          80.9us          77.8us
two-Exact asym           61.6us          92.0us          63.0us
```

`prefer_shallow` beats its own opposite outright, everywhere tested --
by 4-4.4x on one-Exact, by roughly 1.3-1.5x on two-Exact -- confirming
the reasoning behind it directly instead of leaving it a guess: a
shallower stolen column carries more remaining subtree in expectation,
on this tree. Against `parallelqueue`, the matrix design is still
1.8-2.2x slower on one-Exact, but competitive on two-Exact -- within
noise on `asym`, and actually faster on `sym`. The extra per-level
cells buy correctness simplicity (no occupied-slot fallback path at
all) at the cost of a steal scan that is `height` times more expensive
per attempt than the single-slot design's -- a cost one-Exact's deeper
trees pay more of than two-Exact's shallower, cheaper ones do.
`restricted_descent_parallel_matrix` is kept `parallel`-gated in
`src/path_fst.rs` alongside every other variant, not wired in as the
default, with both steal-order heuristics exposed (`prefer_shallow:
bool`) rather than hardcoding the winner -- the depth-threshold
steal-gating idea deferred above remains the natural next lever if this
mechanism gets revisited, now that a measured baseline exists to tune
against.

### Sweeping only the leaves, bottom-up: a clean prototype, and a decisively negative result for a reason worth stating precisely

A different SIMD-shaped idea, proposed directly: instead of descending
from the root and pruning as every other traversal in this file does,
sweep *only* the leaf level's existence words with one branch-free
elementwise AND (same `#![forbid(unsafe_code)]`-respecting,
auto-vectorization-friendly convention as
`restricted_descent_bfs_masked_visit`), then, for every leaf with a
surviving bit, walk back *up* toward the root to find out which
triple(s) it belongs to, instead of ever reaching it by descending down
to it.

Two real design questions had to be settled before writing anything:
where the sweep happens (scoped to one already-narrowed subtree, or
genuinely global across the whole structure), and how to walk "up" at
all, since this structure only ever stores forward deltas -- a node's
real parent isn't recoverable from anything already in memory. Decided
directly: global sweep, with real parent back-pointers added
(`build_parent_index`, a fresh reverse-edge map built by walking every
non-leaf level's *unrestricted* existence words once). A genuine new
correctness question came with that choice, inherent to this being a
DAG rather than a plain trie: `build`'s dedup means one leaf can be the
real child of more than one parent, so one matched leaf can have more
than one real root-to-leaf path, and critically *not every one of them
need satisfy this query's `Exact` restrictions* -- the leaf-level mask
only ever checked the free dimensions' bits. `paths_to_root` enumerates
every such path (memoized per address for one call, since a shared
ancestor reached from many leaves would otherwise redo the same upward
walk repeatedly), and every reconstructed path gets checked against the
*full* restriction set, independently, after reconstruction -- not
assumed correct just because the leaf matched.

**Correct on every domain and shape tested** (`tiny` synthetic,
`small`/100K, `asym`/3M, `sym`/10M triples, both one-Exact and
two-Exact) -- output matched `restricted_descent` exactly, every time,
no exceptions, no path-count explosion observed on any of these
domains despite the DAG-sharing concern above being a real one in
principle.

**Measured as a clean, decisive negative result anyway, and the first
guess at *why* was wrong -- worth recording precisely instead of just
the headline number.** First hypothesis: the leaf level has too many
*distinct* nodes to sweep cheaply, the same "DAG dedup means a level's
distinct node count vastly exceeds what one query reaches" problem that
already sank the whole-level `restricted_descent_sweep_visit`. Checked
directly instead of assumed: `leaf_level_node_count()` returns 30 for
`small`, 6 for `asym`, 29 for `sym` -- the leaf sweep itself is
essentially free; that was never the bottleneck. `build_parent_index`'s
one-time structural cost also amortizes fine (9.6ms / 775ms / 3.6s for
`small`/`asym`/`sym` respectively, each paid once, not per query).

The real cost is entirely in `paths_to_root`, and it's the mirror image
of the first guess: those 6-30 distinct leaves are each the real child
of an enormous number of distinct parents (`build_parent_index` holds
54K / 1.5M / 5.5M total reverse edges for `small`/`asym`/`sym`), so
enumerating "every root-to-leaf path through this one leaf" explores a
combinatorially large candidate set, the overwhelming majority of which
get thrown away by the full-restriction check *after* being fully
reconstructed. Measured per single query, parent-index cost already
excluded (the amortized, best-case number):

```
                  one-Exact          two-Exact
small               14.4ms             14.3ms
asym                 3.25s              3.25s
sym                  3.56s              3.60s
```

Four to five orders of magnitude worse than every other mechanism in
this file, which run in microseconds to low milliseconds on the same
domains -- and notably *not* worse at `sym` than `asym` in proportion
to their triple counts (10M vs 3M), consistent with the real cost
being driven by how shared the last few levels are, not by raw triple
count, the same structural property that made the whole-level sweep's
own negative result what it was, just manifesting on the opposite end
of the tree.

Kept as `restricted_descent_leaf_sweep`/`restricted_descent_leaf_sweep_with_index`
(the latter taking an already-built `build_parent_index` so a caller
can amortize it, which is how the honest numbers above were actually
measured) in `src/path_fst.rs`, correct and `leafsweep`-wired into
`path_fst_bgp_perf.rs` for reproducibility, with its correctness-probe
and timed-loop counts hard-capped at 3 regardless of `iters` --
`sweep`'s existing 20-probe cap would already take over a minute at
this cost, and `iters`'s normal 50,000 default would take hours.

### Joins and paths as automatons -- Phase 1: the walk/automaton split, validated on the lowest-risk case first

Follow-up to a design discussion (not reproduced here in full -- the
reasoning that led here): `restricted_descent`'s `word & level_masks[level]`
is already a walk fused with an automaton, just one so degenerate (one
live state per level, known at compile time, no branching in state-space)
that the automaton half collapsed into a flat precomputed array.
`NativeJoin`'s `Stage` array is the same split in different clothes --
its "state" is `(depth, which candidate each earlier atom chose)`,
represented implicitly as a backtracking stack rather than an explicit
`State` type, because its automaton happens to be acyclic and
finite-depth. The real gap is `PathAutomaton` (`automaton.rs`): a
genuine NFA, but every edge expansion calls `scan_pattern` fresh,
independently, per (node, transition) pair -- no synchronized descent
at all, unlike `NativeJoin`. `TOPOLOGY_REVIEW.md` already named this:
"these paths prune within each scan but do not share a traversal
across a whole frontier."

Literature grounding for the split itself: BurntSushi's `fst` crate's
`Automaton` trait (`start`/`is_match`/`can_match`/`accept`), used there
to intersect a query automaton (Levenshtein, regex, literal) with a
stored FST in one pass, plus its generic `And`/`Or`/`Not` combinators
(wrapping `(A::State, B::State)` and delegating) for composing them
without a new method per combination. The one place this project's
version has to depart from that shape: this tree's alphabet is an
up-to-64-slot multi-bit group per level, not one byte, so [`LevelAutomaton`]
(`src/level_automaton.rs`) consumes a `qi` per level instead of a byte,
everything else kept parallel on purpose (`can_match` is the identical
dead-state hook, checked before any storage read).

**Phase 1, the agreed lowest-risk ordering: validate the trait against
plain `restricted_descent` before building anything state-dependent
(a join atom, a path hop) on top of it.**
[`PathFstDeltaPow2Bfs::visit_automaton_rec`] is structurally identical
to `visit_rec`, just reading `automaton.mask(state, level)` instead of
indexing a flat array, and carrying `automaton.advance(...)`'s result
forward as `state`. [`PrecomputedMaskAutomaton`] wraps `precompute_level_masks`'s
exact output as the degenerate one-state case -- `restricted_descent_via_automaton`
built on it reproduces `restricted_descent` byte-for-byte, confirmed
across `small`/`sym`/`asym`, both shapes.

**The second half of Phase 1 -- actually proving composability, not just
the walk split -- found a real bug immediately, which is exactly why it
was built before trusting the design further.**
[`DimRestrictionAutomaton`] represents one dimension's own restriction
(`Exact`/`Free`/`Range`/`Multi`) as its own small automaton; three of
them intersected via [`And`] (one per S/P/O) should equal
`tree::restriction_mask`'s hand-written three-dimension fold exactly,
since `shifts` partitions each dimension into disjoint bit ranges
within one level's `qi`. First version returned *empty results on
every single query* (`restricted_descent_via_composed_automaton`
disagreed with `restricted_descent` on all 50/50 correctness probes,
every domain, every shape -- not a subtle mismatch): each dimension's
own `Free` case was computed as `expand_run(1, 0, width, shift)`, which
sets bits only where the *other two dimensions are exactly zero*, not
where they're unconstrained -- intersecting three such masks forces
everything toward zero. Fixed by delegating to `tree::restriction_mask`
directly with the other two dimensions pinned to `Free`, instead of
re-deriving the cross-dimension interaction by hand.

A second, separate bug surfaced once `Range`/`Multi` were cross-checked
against `Heap::restricted_descent` (which already supports them) rather
than only against `PathFstDeltaPow2Bfs::restricted_descent` (which
panics on them, so the comparison above couldn't have caught this one):
122 of 147 `Range`/`Multi` probes disagreed, including *every single*
`Range`-on-the-pinned-dimension case returning zero results. Root
cause: `advance` only updated the per-dimension `lo` state inside an
`if let Exact(_)` guard, copying `precompute_level_masks`'s own
lo-bookkeeping too literally -- that bookkeeping is only valid in
`Heap`'s narrow fast path (`exact_bound_dims`, both non-free dimensions
pinned to a single value, so there's exactly one real `lo` per level to
precompute ahead of time). `Range`/`Multi` can be reached via different
real sub-indices at the same level depending on which earlier bit was
actually taken; `Heap`'s own general path (`query_rec_restricted`)
handles this by recomputing the mask per node from the real descent
`lo`, never from a flat precomputed array -- this automaton's `state`
*is* that real `lo`, so `advance` has to track the real `qi` taken
unconditionally, for every restriction kind, not just `Exact`. Fixed;
0/147 mismatches afterward, cross-checked against `Heap` as the oracle
specifically because `PathFstDeltaPow2Bfs` itself has never supported
`Range`/`Multi` to check against.

**Both bugs are now a genuinely new capability, not just a validated
abstraction**: `restricted_descent_via_composed_automaton` supports
`Range`/`Multi` restrictions on `PathFstDeltaPow2Bfs`, which no other
traversal in this file does -- a direct, free consequence of
`DimRestrictionAutomaton` already handling all four `DimRestriction`
variants instead of panicking on two of them.

**Measured on the `profiling` build, 50-iteration timed loops, same
trees as every other comparison in this file:**

```
                  seq       automatonwalk   composedautomaton
one-Exact small    35.7us      32.2us          152.3us  (4.3x)
one-Exact sym     1247.1us    1263.7us         3374.2us (2.7x)
one-Exact asym     779.3us     787.6us         2125.8us (2.7x)
two-Exact small     2.4us       1.6us            3.8us  (1.6x)
two-Exact sym      17.4us      15.2us           30.5us  (1.8x)
two-Exact asym     11.3us       9.9us            20.3us  (1.8x)
```

`automatonwalk` is within noise of `seq` everywhere -- the generic
trait genuinely monomorphizes to no added cost, confirmed rather than
assumed, which is the actual point of validating the split this way
before building anything real on top of it. `composedautomaton`'s
1.6-4.3x overhead is real and expected, not a surprise: it recomputes
`restriction_mask` fresh at every node (three full three-dimension
folds per node, one per `And` branch) instead of once per query,
because `lo` genuinely has to track the real descent path for
`Range`/`Multi` correctness in general, unlike the degenerate
`Exact`/`Free`-only case this crate's existing traversals special-case.
Not tuned yet -- Phase 1's goal was proving the split and the
composability idea hold, not optimizing the composed case, which has
an obvious next lever (detect a pure `Exact`/`Free` composition and
hoist its mask computation to once-per-query, the same as today's
`precompute_level_masks` already does) left for whenever the join/path
phases actually need it.

Both are kept `src/level_automaton.rs` + `PathFstDeltaPow2Bfs::
restricted_descent_via_automaton`/`restricted_descent_via_composed_automaton`,
wired into `path_fst_bgp_perf.rs` as `automatonwalk`/`composedautomaton`
for reproducibility.

### Closing (most of) the composed-automaton gap: `Range`/`Multi` precomputed via boundary tracking, not raw `lo`

Follow-up question, asked directly: the composed automaton's 1.6-4.3x
overhead was blamed on recomputing `restriction_mask` fresh at every
node for `Range`/`Multi` -- but can those be precomputed per query too,
the way `Exact`/`Free` already are? Yes, and the mechanism is a
standard one, not specific to this crate: `contiguous_range`'s clip of
a query range against a node's window always produces a *contiguous*
run of accepted sub-indices. That means at any point during a descent,
at most the *leftmost* accepted sub-index can still be touching the
range's lower endpoint, and at most the *rightmost* can still be
touching the upper endpoint -- everything strictly between them is
already fully inside the range, with no way to fall back out, i.e.
unconstrained from there on. So the state needed isn't the raw
accumulated `lo` (which can take many different values depending on
which sub-index was actually taken at each level) -- it's two
independent flags, "still tracking the lower boundary" and "still
tracking the upper boundary", each only ever `Tracking` or `Satisfied`.
And since the range's own endpoints come from the query, not the data,
their own bit-prefix at every level is known *before* touching the
tree at all -- so the whole per-level `(state, level) -> mask` table
can be built once per query, same timing class as
`precompute_level_masks`, instead of calling `contiguous_range`/
`restriction_mask` fresh at every node. `Exact(v)` falls out of the
same mechanism as `Range(v, v)` with both trackers coinciding and
neither ever reaching `Satisfied` (any sub-index satisfying one bound
necessarily violates the other, since they're the same value) --
exactly one sub-index survives at every level, the same single-path
behavior `Exact` already had, now a special case instead of a separate
code path. `RangeTable` (`src/level_automaton.rs`) implements this;
`Multi` is one `RangeTable` per range in the union, tracking
independently, per path, which ranges are still alive (`None` once a
range's own mask stops including the real bit taken -- permanently
dead for that path).

**The first version of this regressed `Exact` itself, caught on the
very first correctness check (the same tiny synthetic tree used
earlier in this investigation) before it ever reached the `Heap`
oracle cross-check for `Range`/`Multi` specifically.** Root cause: the
boundary masks were built via `expand_run(1, start, run_len, shift)`
starting from a bare `1` -- exactly the same cross-dimension mistake
already found and fixed once in this same file (see the entry above):
that only sets bits where the *other two dimensions are also zero*,
not where they're unconstrained. It had been masked (literally) by the
`Free`/`Free`/`Free` sanity check passing, since pure-`Free` dimensions
bypass this path entirely via a `u64::MAX` shortcut -- the bug only
showed up the moment any dimension actually used `Range`/`Exact`/
`Multi`, i.e. immediately. Fixed the same way as the first time:
delegate to `restriction_mask` for each boundary mask too, with the
other two dimensions pinned to `Free` -- still built once per *level*
at construction time, not once per node, so the fix doesn't give up
the precomputation property it was for. Re-verified: 0/583 mismatches
across the full `Heap`-oracle sweep (`Range`/`Multi` on every
position, two simultaneous `Range` dimensions, `Exact` on the
non-emitted dimension, plus the direct `restricted_descent` sanity
checks), `cargo test`/`clippy` clean.

**Measured on the `profiling` build, same methodology as before:**

```
                  seq        composedautomaton (before -> after)
one-Exact small     35.9us     152.3us (4.3x)  ->   76.4us (2.1x)
one-Exact sym     1265.7us    3374.2us (2.7x)  -> 2207.5us (1.7x)
one-Exact asym     760.0us    2125.8us (2.7x)  -> 1376.0us (1.8x)
two-Exact small      2.4us       3.8us (1.6x)  ->    3.5us (1.5x)
two-Exact sym       17.7us      30.5us (1.8x)  ->   24.7us (1.4x)
two-Exact asym      11.1us      20.3us (1.8x)  ->   16.0us (1.4x)
```

Roughly halves the overhead, everywhere, but doesn't close it
entirely -- and the remainder is now architectural rather than a bug:
the composed automaton still does three independent per-dimension
lookups and up to three `advance` calls per accepted bit (one per
`And` branch), against one flat array lookup for the degenerate case.
That's the real, now-understood price of "one automaton per position,
intersected" over "one combined mask computed by a single hand-written
fold" -- composability bought back most, not all, of its cost once the
precomputation was done correctly, and what's left is inherent to the
decomposition itself, not a missed optimization.

Next: Phase 2 ports `NativeJoin`'s logic to the FST through this same
trait (one `LevelAutomaton` per atom, mask reading shared bindings
instead of a static array); Phase 3 rewires `PathAutomaton`'s hop
expansion to call this walk instead of `scan_pattern`.

### Phase 2: conjunctive joins natively on the FST, not a literal port, and a real bug in the adaptation

Direct motivation: asked whether a BGP like `?s :p :o1 ; ?p :o2 .` (shared
subject, a *variable predicate* in the second atom) could be answered
with what existed -- yes, already, via `NativeJoin`/`graph::join_run`
with `TriplePattern<Term>` instead of the default `TriplePattern<u32>`
(confirmed live, not just cited: built a `Heap` tree with deliberate
decoy subjects matching only one of the two atoms, and a subject
matching the second atom twice with different predicates, and the
result came back exactly right). The real gap, confirmed at the same
time: that only runs on `Heap`, never on `PathFstDeltaPow2Bfs` --
exactly the "Phase 2" placeholder from the walk/automaton design entry
above.

**Decided against the design that entry proposed** ("one `LevelAutomaton`
per atom") **once the actual algorithm `NativeJoin` runs was reread
carefully.** `NativeJoin`'s `stages` are grouped *by level first, then
by atom* -- every atom picks its own candidate at level 0 before any
atom moves to level 1. That interleaving is what makes it
Leapfrog-Triejoin-shaped (Veldhuizen, ICDT 2014: advance one variable
at a time, sharing what's already known, never materializing a full
per-atom result before starting the next atom) instead of a nested-loop
join. A `LevelAutomaton`-per-atom design -- fully resolve atom 0's own
top-to-bottom walk via the existing generic walk, *then* start atom 1's
-- would have reused more of Phase 1's machinery, but throws away
exactly that interleaved pruning, which this tree's schedule supports
exactly as well as `Heap`'s does (same per-level `dim_bits`/`shifts`
shape for every atom, since every atom queries the same physical tree).
Chose fidelity to the proven algorithm over fitting the existing trait
-- "don't port literally" cuts both ways: it also means not forcing a
different, worse-shaped abstraction onto it for the sake of reuse.

**What's reused versus rewritten, concretely** (`src/fst_join.rs`):
reused wholesale because it's genuinely backend-agnostic query-planning
logic that never touches `PrefixCursor` or any Heap-specific type --
the `Stage`/`Check`/`Write` compiled-plan shape, the atom-reordering
heuristic (favor constants, then connectivity to already-selected
atoms), and `slot_ones` (made `pub(crate)` in `native_join.rs` instead
of duplicated). Rewritten because it's the actual backend-specific
part: `open`/`run`'s node access. `Heap`'s `PrefixCursor` carries a
precomputed "unskipped occupancy" word; `FstJoin` instead reads
`word`/`real_index`/`delta` directly, the same primitives every other
`path_fst.rs` traversal uses. One genuine improvement `NativeJoin` has
no equivalent of: since a stage's own `level` is known at compile time,
a leaf-level stage (`level == height - 1`) skips the delta read and
child-address computation entirely, because no stage for *that* atom
ever reads that child position again -- not because `Heap` couldn't
support the same skip, but because `PrefixCursor`'s abstraction doesn't
expose the leaf/non-leaf distinction as directly as this tree's own
split already does everywhere else in this file.

**A real bug, caught by the simplest possible case -- a single atom, no
join at all.** First version mutated a frame's `addr`/`lo` fields
in place to become the just-chosen child's position. That's wrong the
moment backtracking returns to the *same* frame to try a different
sibling bit: the parent's own address is already gone, overwritten by
the previous candidate's child address, so the next candidate's delta
gets read from the wrong node entirely. Symptom: a single-pattern query
with three matching rows for the same variable (`?s :p :o100`, three
subjects satisfying it) returned only the first one, every time -- not
a join-specific bug, since nothing about joining was involved yet, just
the adapted backtracking loop's own node-navigation getting the
parent/child distinction wrong. Fixed by giving `Frame` a separate
`child_addr`/`child_lo` pair: `addr`/`lo`/`word` are this frame's own
resolved position, set once by `open` and never mutated afterward;
`child_addr`/`child_lo` hold "where the most recently chosen candidate
leads", read by the *next* real level's own `open` call. Re-verified
after the fix: the single-atom case, the original hand-verified
variable-predicate BGP (including the exact decoy/multi-match shape
that caught the original bug class), and a randomized cross-check
against `NativeJoin`/`Heap` on the same data -- shared-subject
variable-predicate joins, shared-object star joins, 3-atom triangle
joins, a ground (variable-free) query, and a repeated-variable-within-
one-atom query (`?x :p ?x`) -- all matching exactly, including a sweep
specifically re-run until it hit a genuine non-empty case for the
shared-subject shape (not just empty-matches-empty).

**Measured on the `profiling` build, 1M triples, 200 shared-object star
join queries (`?s0 p1 ?o . ?s1 p2 ?o .`), both executors run back to
back on data built from the same triples:**

```
NativeJoin (Heap):  40777.8 us/query, 1030000 total rows
FstJoin (PathFst):  35041.1 us/query, 1030000 total rows
ratio (fst/native): 0.86x
```

Row counts match exactly (1,030,000 = 1,030,000 across all 200
queries) and `FstJoin` runs ~14% faster, plausibly from the leaf-level
delta-skip described above -- not purely an FST-vs-Heap effect, since
`PathFstDeltaPow2Bfs`'s own storage is also considerably smaller here
(6.1MB vs 26.0MB for the same triples). Kept as `FstJoin`/
`FstJoinWorkspace`/`FstJoinStats` in `src/fst_join.rs`
(`#[cfg(feature = "heap")]`, alongside `path_fst`), with the same
`visit`/`visit_with_stats` streaming contract as `NativeJoin` -- no
`graph.rs`-style `join_run` convenience wrapper yet, since Phase 2's
job was proving the executor itself is correct and competitive, not
building out the full API surface.

### The automaton-shaped join, actually built and measured against `FstJoin` -- a real trade-off, not a confirmed assumption

`fst_join::FstJoin`'s own design entry argued, from reasoning alone,
that a `LevelAutomaton`-per-atom design (fully resolve one atom's own
top-to-bottom walk, *then* start the next atom's) would lose
`NativeJoin`'s interleaved-by-level pruning and shouldn't be built that
way. Asked directly to build it anyway and measure, side by side,
without touching `FstJoin` -- the right call: the reasoning turns out
to be correct for some query shapes and wrong for others, which
reasoning alone would not have surfaced.

`automaton_join::AutomatonJoin` is that design, for real: one atom at a
time, each a complete walk through [`level_automaton::LevelAutomaton`]
(composing three `DimRestrictionAutomaton`s via `And`, exactly like
`restricted_descent_via_composed_automaton`), reaching every one of
*that* atom's own matches -- plain recursion, nested-loop join -- before
the next atom's walk starts at all. A variable already bound by an
earlier atom becomes `DimRestriction::Exact(bindings[v])`, rebuilt
fresh (a real `RangeTable` construction) every time that atom is
entered, since the bound value can differ across different outer
candidates -- the real cost this design pays that `FstJoin`'s
precompiled `Stage` array does not. Deliberately narrower than
`FstJoin`: a variable repeated *within* one atom (`?x :p ?x`) needs a
same-level bit-equality constraint between two dimensions that `And`'s
per-dimension independence can't express (`FstJoin` handles this via
`NativeJoin`'s own `owners` mechanism, not reused here) --
`AutomatonJoin::new` returns `InvalidConfig` for that shape rather than
silently getting it wrong; every shape in the comparison below avoids
it. One genuine simplification fell out of the recursive shape, not
from optimizing anything away: since a later atom's walk is only ever
entered as a direct call from inside an earlier atom's own leaf (never
resumed later against a flat frame array the way `NativeJoin`/`FstJoin`'s
backtracking is), nothing here needs `FstJoin`'s `clear_binding` --
the next sibling's own write simply overwrites the same slot before
anything reads it again, because every reader of a variable is
guaranteed (by construction, from the atom ordering) to run after that
variable's one writer.

Correctness: cross-checked against *both* `NativeJoin`/`Heap` and
`FstJoin` together (three-way agreement required) across the same
shapes `FstJoin`'s own validation used -- shared-subject
variable-predicate joins (200 random draws, re-run until a genuine
non-empty case was hit, not just empty-matches-empty), shared-object
star joins, 3-atom triangles, and a ground query -- all matching
exactly, plus a direct check that the same-atom-repeated-variable shape
is rejected rather than silently wrong.

**Measured on the `profiling` build, 1M triples, 200 queries per
shape, all three executors run back to back on data built from the
same triples:**

```
star (shared object, 2 atoms), 1030000 total rows:
  NativeJoin (Heap):      40443.9 us/query
  FstJoin (interleaved):  34932.2 us/query
  AutomatonJoin (nested): 58178.3 us/query   -- 1.67x FstJoin, 1.44x NativeJoin

triangle (3 atoms), 23 total rows:
  NativeJoin (Heap):      43446.7 us/query
  FstJoin (interleaved):  30700.6 us/query
  AutomatonJoin (nested): 51782.2 us/query   -- 1.69x FstJoin, 1.19x NativeJoin

shared-subject variable-predicate (2 atoms), guaranteed non-empty
(1 row/call, a real subject with >=2 outgoing edges picked directly
from the generated data so the comparison isn't measuring three
different "definitely empty" fast paths):
  NativeJoin (Heap):      236.22 us/query
  FstJoin (interleaved):  154.39 us/query
  AutomatonJoin (nested):  14.26 us/query    -- 0.09x FstJoin, 0.06x NativeJoin
```

**Star and triangle confirm the original reasoning: the nested design
loses 1.2-1.7x to interleaving**, consistent with Leapfrog-Triejoin's
own case for sharing partial knowledge across every atom at every
level rather than resolving one atom fully before touching the next --
both shapes join on a variable where *every* atom's own exploration is
comparably wide open, so interleaving's early cross-atom pruning has
real work to do.

**Shared-subject variable-predicate inverts it hard: the nested design
wins by roughly an order of magnitude.** The reason isn't an
implementation accident, it's structural: the shared variable (`?s`)
is *also* heavily constrained by the first atom alone (`Exact`
predicate, `Exact` object, only `?s` free) -- cheap and narrow on its
own, the same shape every `two-Exact` measurement in this file has
already shown resolves in microseconds. `?s`'s own bits are spread
across several tree levels, though, so `FstJoin`/`NativeJoin`'s
interleaving doesn't get to treat it as "resolved, then propagate a
single constant" -- it has to track partial knowledge of `?s`
level-by-level *for the second atom too*, applying a `Check` as each
new piece of `?s` arrives, before the first atom has even finished
narrowing it to one value. `AutomatonJoin` instead fully resolves the
first atom first (cheap, since it's narrow), gets `?s` as one concrete
value, and only then starts the second atom with `Exact(?s)` already
fully known from its very first level -- no partial-bit tracking
overhead at all. Classic selectivity-based join ordering (resolve the
selective predicate first, propagate the result as a constant) beating
worst-case-optimal interleaving on a query where one atom is cheap
enough that the "share knowledge progressively" overhead costs more
than it saves.

**Conclusion, stated honestly rather than picking a winner**: neither
design dominates. `FstJoin` wins when every atom's own search space is
comparably wide (star/triangle shapes, matching the WCO literature's
own motivating case). `AutomatonJoin` wins when some atom can cheaply
and narrowly resolve a shared variable on its own, avoiding
interleaving's partial-knowledge bookkeeping entirely. Both are kept,
side by side, neither replacing the other: `fst_join::FstJoin` and
`automaton_join::AutomatonJoin`, both `#[cfg(feature = "heap")]`. A
real query planner choosing between them would need a selectivity
estimate per atom, not a fixed rule -- not built here, since this round
was about proving the trade-off is real and measurable, not picking a
default.

### Why the automaton-shaped join actually lost on star/triangle, found with `perf`, and fixed -- it flips the comparison entirely

The entry above reasoned that `AutomatonJoin`'s 1.2-1.7x loss to
`FstJoin` on star/triangle was the cost of rebuilding each atom's
bound-variable automaton fresh on every entry. Asked to check that
with `perf` instead of leaving it as a plausible guess -- `perf record`
(2000 star-join queries, 1M triples) says that guess was wrong.
`RangeTable::build` (3.72%) + `tree::restriction_mask` (8.69%), the
actual "rebuild" cost, totaled only ~12%. The real cost (~78%) was the
`LevelAutomaton`/`And` abstraction itself: `And::advance` (31.68%),
`DimRestrictionAutomaton::advance` (14.93%), `DimRestrictionAutomaton::mask`
(8.13%), plus the extra indirection it added to `walk_leaf_rec` (23.25%)
-- three independent per-dimension automata, each dispatched through an
enum match, composed through nested tuples, all paid *per accepted bit
at every level*.

That machinery (`RangeTable`'s `Tracking`/`Satisfied` boundary states)
exists to support genuine `Range`/`Multi` restrictions with real
per-level state. A join atom never needs that: at the instant any atom
is walked, every one of its three dimensions is `Const`, `Free`, or
`Exact` (a bound variable's current value) -- never a real multi-state
automaton. That is exactly the degenerate case `precompute_level_masks`
already handles with one flat `[u64; MAX_HEIGHT]` array and zero
per-node dispatch, the same mechanism `restricted_descent` itself
uses. Using the general `And`-composed automaton for a job that never
needed its generality was reaching for the wrong tool, not a deliberate
trade-off -- the Phase 1 entries above built that machinery to prove
`Range`/`Multi` *could* be composed cheaply; nothing said every later
consumer should default to it.

Rewrote `automaton_join.rs` to compute that flat array once per atom
entry (`compute_atom_masks`, a near-verbatim copy of
`precompute_level_masks`) and walk it directly -- no `LevelAutomaton`,
no `And`, no `DimRestrictionAutomaton`, anywhere in this module anymore.
Everything else (the nested recursion, the atom ordering, the
first-occurrence/bound-variable classification, the lack of an undo
step) is unchanged. Re-verified against `NativeJoin`/`Heap` and
`FstJoin` together on every shape, including the hand-verified case and
the repeated-variable-in-atom rejection: still exact.

Re-profiled after the fix (same star-join benchmark): `walk_leaf_rec`
49.32%, `read_word` 27.77%, `read_delta` 16.32%, `restriction_mask` +
`compute_atom_masks` down to ~4.2% combined -- a storage-bound profile,
the recursion and the actual tree reads, not abstraction overhead.

**Full comparison, `profiling` build, 1M triples, 200 queries per
shape, all three executors run back to back on data built from the
same triples (lower is better; the ratio columns are both
`AutomatonJoin ÷ other`, so a value below 1 always means AutomatonJoin
is faster):**

| Shape | Rows/query | NativeJoin (Heap) | FstJoin (FST, interleaved) | AutomatonJoin (FST, nested) | AutomatonJoin ÷ FstJoin | AutomatonJoin ÷ NativeJoin |
|---|---|---|---|---|---|---|
| Star (shared object, 2 atoms) | 5150.0 | 40,624.9 us | 34,960.0 us | **20,534.1 us** | 0.587 | 0.505 |
| Triangle (3 atoms) | 0.1 | 43,264.5 us | 30,756.8 us | **15,853.2 us** | 0.515 | 0.366 |
| Shared-subject, variable predicate (2 atoms) | 1.0 | 213.0 us | 140.2 us | **3.0 us** | 0.021 | 0.014 |

The fix didn't just close the star/triangle gap, it inverted it:
`AutomatonJoin` now wins on *every* shape tested, not only the one it
already won before -- roughly 1.7-1.9x faster than `FstJoin` on
star/triangle (previously 1.2-1.7x *slower*), and ~50x faster on the
shared-subject shape (previously ~11x faster). The qualitative
explanation from the entry above for *why* interleaving should help
star/triangle wasn't wrong as a reason to expect `FstJoin` to be
competitive there -- it was wrong as a reason `AutomatonJoin` had to be
slow: interleaving's cross-atom pruning is a real advantage in
principle, it just wasn't the thing `AutomatonJoin`'s first version was
actually losing to. Once the wrong tool was swapped for the right one,
`AutomatonJoin`'s own advantage -- no `Stage`/`Check`/`Write`
compilation, no flat backtracking array, a plain recursive walk reading
storage about as directly as `restricted_descent` itself does -- won
outright, on this benchmark, on every shape measured so far.

**Not yet a closed case.** All three shapes here share one property:
the query is small (2-3 atoms) and every shared variable is resolved
by some atom narrowly. Star/triangle's own atoms are individually wide
(`Free` on two of three dimensions), which is exactly the case the WCO
literature uses to motivate interleaving -- `AutomatonJoin` won there
anyway once its per-node cost was fixed, but a query where interleaving's
*cross-atom* pruning (not just its per-atom cost) is doing real work --
many atoms, a long chain of shared variables, deep intermediate results
that `AutomatonJoin`'s nested walk would materialize one atom's full
breadth before ever starting the next -- hasn't been measured. `FstJoin`
is kept, unmodified, specifically so that comparison stays possible
once such a shape is identified, not discarded on the strength of
three benchmarks that all turned out to favor the other design.

### Wired as default: `PathFstDeltaPow2Bfs` + the automaton-driven APIs, documented and promoted, with one more perf fix to make the recommendation honest

Asked directly to document and wire the FST layout as the default,
with the automaton-driven query and join methods as the preferred API
-- "always use automaton, unless for debugging or comparison." Before
writing that recommendation anywhere, checked whether it was actually
true yet, the same way every other claim in this file has been
checked: `restricted_descent_via_composed_automaton` (the general
query entry point, supporting `Range`/`Multi`) still carried the
1.6-4.3x overhead measured in its own earlier entry above, for the
plain `Exact`/`Free` case `restricted_descent` already handles for
free. Recommending it as *the* default while it cost more than the
thing it was replacing, for the common case, would have been telling
people to opt into a wash at best.

Applied the exact fix `automaton_join.rs` had just proven: detect
whether any restriction is genuinely `Range`/`Multi`; if not, skip the
`And`-composed `DimRestrictionAutomaton`s entirely and take the same
flat `[u64; MAX_HEIGHT]` `PrecomputedMaskAutomaton` path
`restricted_descent_via_automaton` already uses. Only a real `Range`/
`Multi` restriction pays for the heavier composition now. Re-verified
against the `Heap` oracle on the same `Range`/`Multi`/mixed-dimension
sweep used when this method was first built (494 cases, 0 mismatches)
plus a direct `Exact`/`Free` sanity check against `restricted_descent`
itself.

**Measured on the `profiling` build, same methodology as every
`composedautomaton` number in this file's earlier entries:**

```
                  seq (restricted_descent)   composedautomaton (before -> after)
one-Exact small          36.2us                   various (see above)  ->   32.4us
one-Exact sym          1245.3us                                        -> 1270.1us
one-Exact asym          774.8us                                        ->  757.7us
two-Exact small           2.6us                                        ->    1.6us
two-Exact sym            17.5us                                        ->   15.1us
two-Exact asym           11.3us                                        ->    9.5us
```

At parity with (sometimes faster than) `restricted_descent` on every
domain/shape, for the case that matters for "default" -- the fast path
*is* `restricted_descent_via_automaton`'s own code now, not a
reimplementation of it, so there's no way for the two to drift apart
silently.

**What actually changed, concretely:**

- `PathFstDeltaPow2Bfs::restricted_descent_via_composed_automaton` is
  now genuinely the one recommended entry point for restricted-
  descent-style queries on this struct -- fast for `Exact`/`Free`,
  correct (if slower) for `Range`/`Multi`, instead of a strictly-
  slower alternative to `restricted_descent_via_automaton` that
  happened to also support more restriction kinds.
- `automaton_join::AutomatonJoin` is documented as the preferred join
  executor (it already measured faster on every shape -- see the
  entries above; this round just updated the doc comments and
  `README.md` to actually say so instead of leaving it framed as "a
  comparison experiment" after it stopped being only that).
- `path_fst::PathFstDeltaPow2Bfs`'s own module doc, `fst_join::FstJoin`'s,
  `native_join::NativeJoin`'s, the crate root (`lib.rs`)'s "Which
  structure to use" section, and `README.md`'s new "The FST layout"
  section all now say the same thing in the same direction: prefer
  `PathFstDeltaPow2Bfs` + `restricted_descent_via_composed_automaton`/
  `AutomatonJoin` for static, query-and-join-heavy workloads; every
  other `restricted_descent*` variant, `FstJoin`, and `NativeJoin`
  (when a `PathFstDeltaPow2Bfs` is viable at all) are debugging/
  comparison tools, not defaults.
- **Not silently glossing over the real gap**: `PathFstDeltaPow2Bfs`
  has no `insert`/`remove` (static build only), no `serialize`/
  `deserialize`, no `contains_batch`/`count_matching`, and isn't a
  `join::PatternBackend` (no graph traversal/paths, no regular-path
  automata) -- `Heap`/`File`/`Compressed` remain the only choice for
  any of that. The recommendation is scoped to what was actually
  measured, not asserted as a blanket replacement.

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
`contains_batch_into`/`restricted_descent_into` already answer: does this
*have* to allocate per call, or can the caller supply/reuse the buffer and
decide for themselves whether to keep a copy? The crate already follows
this almost everywhere, not just where this round of work landed:

- `NativeJoin::visit`/`visit_with_stats` hand each result row to the
  caller's closure as a borrowed `&[u32]`, zero allocation after
  workspace warmup -- the caller copies only if and when it wants to keep
  a row past the callback.
- `contains_batch_into`, `restricted_descent_into`,
  `predicates_for_into` take `out: &mut Vec<_>` and retain its capacity
  across calls; `_into`-less siblings are thin "allocate fresh and
  delegate" convenience wrappers over them, not separate implementations.
  This list was wrong about `contains_batch_into` until an external
  review (the "graph store harness team") caught it: `out` was reused,
  but the internal sort buffer (`order: Vec<(u32, u128)>`, one entry per
  probe) was still allocated fresh every call -- 32 bytes/probe, exactly
  the 128 KB/4096-probe-batch they measured. `insert_batch`/`remove_batch`
  had the identical gap through `sort_by_morton_key`'s own `keyed` buffer.
  Fixed by moving both into fields on the tree itself
  (`morton_sort_packed`/`morton_sort_generic`, reused by
  `sort_by_morton_key` now that it takes `&mut self`) for the two
  `&mut self` batch-mutation methods, and into a new caller-supplied
  `ContainsBatchWorkspace` (same "one workspace per concurrent call" rule
  as `NativeJoinWorkspace`) for `contains_batch_into`, which stays `&self`
  for concurrent readers. A/B'd against an otherwise-identical build with
  the scratch forced back to a fresh allocation every call, back to back:
  no measurable difference (~1%, within run-to-run noise) on a
  4096-probe/4096-triple batch against a 2M-triple tree -- the same
  honest result as the `FileStorage`/`CompressedStorage` scratch-buffer
  fixes below: architecturally correct, matches the convention, but the
  allocation itself is dwarfed by the traversal cost next to it. Fixed
  anyway, for the same reason those were: it's the gap the convention
  says shouldn't exist, independent of whether this particular workload
  size makes it measurable.
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
