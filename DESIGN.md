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
caller-specified split.** The `[3,1,2]`/`[4,1,1]` experiments above
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
budget has formally reached zero yet. This is now `build_dynamic`'s
only schedule; the static-split experiments remain documented above as
what was tried first and what they cost, not as a live recommendation.

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
