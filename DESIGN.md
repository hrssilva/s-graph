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
dense vertex table. Domain growth and deletion are not implemented.

Each node keeps a bitmap and, where needed, a variable-width subtree span.
An auxiliary Fenwick directory uses one `u64` per logical block, about 0.2% of
full 4 KiB block payloads. Same-block moves use direct offsets. Cross-block moves
and byte distances take O(log B), rather than walking B block descriptors.
Non-splitting insertion updates the directory in O(log B); block splits rebuild
it in O(B), reusing its allocation. The backend block vectors still shift on
splits. There is no blanket O(height) insertion guarantee.

`scan([Option<u32>; 3])` streams matching triples in Morton order, with a fixed
O(maximum trie height) stack and no heap allocation on Heap. File/compressed
backends may allocate their cache buffers. Scans borrow the tree, preventing
concurrent mutation. They do not promise lexicographic projection order.

Existing adjacency methods retain bulk subtree decoding and sorted,
deduplicated outputs. Their `_into` variants retain caller buffer capacity;
sequential legacy BFS now reuses its adjacency and frontier buffers. Streaming
is useful for early exit and high degree; bulk decoding can be faster when the
whole adjacency list is required. Both paths remain available.

## General prepared joins

`TriplePattern<P = u32>` preserves existing constant-predicate syntax. Use
`TriplePattern<Term>` for variables in any coordinate. `PreparedJoin` supports
unanchored patterns, cycles, disconnected components, ground constraints,
repeated variables within an atom, and arbitrary variable orders. Variables must
be numbered in `0..num_vars` and each must appear in an atom; malformed queries
return `TreeError::InvalidConfig`.

Preparation proceeds as follows:

1. Validate the query and choose a deterministic order favoring variables shared
   by constant-filtered atoms, or use the explicit permutation in `JoinOptions`.
2. Scan constant-filtered atoms first. Propagate singleton variable projections
   into later scans. Apply repeated-variable equalities before projection.
3. Sort and deduplicate each atom's relation in the chosen variable order.
   Encode each column relative to its minimum, using only the bits needed for
   its value range. Constant columns occupy no payload bits. Plain `u32` rows
   are also available through `RelationLayout::Plain`.
4. Compile variable occurrences and parent cursor offsets into flat arrays.

No graph-wide six-permutation index is installed. A prepared join owns its
query-specific relations and represents a **snapshot**. Mutations require a new
preparation to affect results. Preparation costs the actual tree scans plus
O(sum N_i log N_i) sorting; it is not free and can scan the graph once per
unbound atom. Default packed payload uses each relation's row count times the
sum of its column bit widths, rounded to `u64` words. Each width is
`ceil(log2(max - min + 1))`; extraction takes at most two word reads. Plain
payload uses `4 * sum(arity_i * distinct_rows_i)` bytes. Preparation holds
three-column temporary rows; allocator capacity slack, packing output and plan
metadata are additional. Packed rows reduce memory traffic, while plain rows
avoid bit extraction; select using workload measurements.
`JoinOptions::max_rows` limits total matched rows during preparation; it is a row
budget, not an exact allocator-byte limit. `relation_bytes()` reports final
payload. Keep prepared joins only for workloads that justify this memory.

Execution performs leapfrog intersection of trie-level projections represented
by row intervals. `open` narrows to a parent's prefix interval; monotone seeks
use exponential search followed by binary search. Advancing skips a whole group
of equal keys. Parent cursors remain positioned during child evaluation.
Backtracking is iterative. All applicable atoms constrain every variable,
including when no other variable is bound.

This is the [Leapfrog Triejoin](https://arxiv.org/abs/1210.0481) execution model:
for full conjunctive queries, enumeration is worst-case optimal up to logarithmic
factors **over the prepared relations**. Tree scanning and relation construction
are separate costs. Intersecting materialized adjacency lists alone would not
establish that guarantee. Projection with duplicate elimination, aggregation,
optional matches and query-language bag semantics are outside this operator.

`JoinWorkspace` stores O(query size) cursor and binding arrays. Once warmed for
a query size, execution allocates no heap memory of its own. Callbacks borrow
the current binding and can return `ControlFlow::Break` immediately. Prepared
joins are immutable `Send + Sync`; threads use separate workspaces. `ltj_run`
is a compatibility convenience that prepares and collects on every call;
`try_ltj_run` also returns validation errors. Custom backends implement the GAT
`PatternBackend` interface, which avoids boxed dispatch in the new algorithms.

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
File/Compressed interior caches remain non-`Sync`. Prepared snapshots can be
shared after preparation from either backend. These changes do not introduce
recovery, locks on every read, or an unsafe `Sync` implementation.

## Verification and measurements

`tests/algorithms.rs` compares joins with exhaustive assignments across variable
orders and paths with an independent enumerator. It covers diagonals, variable
predicates, unanchored triangles, disconnected queries, cancellation/reuse,
snapshot behavior, sparse IDs, cycles, all directions, all path uniqueness modes,
automaton product states, backend parity, truncations and byte mutations.
Existing insertion, bulk-build, compaction and cross-backend tests remain active.

Run `cargo bench --bench algorithms` for a deterministic 20,000-vertex,
60,000-triple fixture. It reports preparation separately from execution, relation
payload bytes, bounded traversal, bulk versus streaming adjacency, and unanchored
triangle enumeration. These are smoke measurements, not production latency
guarantees; use representative degree distributions, query shapes, cold/warm
caches and tail-latency measurements before choosing a deployment configuration.

A local release run with Rust 1.98.1 produced the following illustrative numbers
(the harness prints current measurements on your machine):

| Operation | Observed mean |
|---|---:|
| Anchored join preparation | 3.7 µs |
| Repeated anchored join, one binding | 0.096 µs |
| Materialized-adjacency anchored baseline | 1.85 µs |
| Bounded endpoint traversal, 0–8 hops | 56 µs |
| First result from a 20,000-edge hub, streamed | 0.26 µs |
| Same hub, materialize then take first | 46 µs |
| Unanchored triangle preparation, packed | 3.0 ms |
| Unanchored triangle execution, 20,000 bindings, packed | 6.2 ms |
| Same triangle execution, plain | 4.2 ms |

Packed triangle relations occupied 225,000 payload bytes versus 480,000 for
plain rows (53% less); the primary tree occupied 474,870 allocated bytes in
both cases. The anchored query's singleton columns need no packed payload,
but its plan and workspace still occupy memory. Full small adjacency consumption
was faster with bulk materialization (2.0 µs) than streaming (2.3 µs). These
comparisons deliberately show the tradeoffs rather than claiming one layout
wins every workload. The baseline uses the existing adjacency operations on the
same tree, not a separate database engine or an old-revision timing.
