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
