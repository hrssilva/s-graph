# FST / automata promotion

The supported entry points are now `s_graph::{Fst, Join, JoinWorkspace}`.
See the [README](../../README.md) for runnable storage, joins, OPTIONAL,
paths and analytics examples. The earlier prototype designs and measurements
remain in this directory and [DESIGN.md](../../DESIGN.md).

## What moved

| Component | Production location |
|---|---|
| Recognizers, boolean composition, masks, reversible actions, transducers/reducers | `src/automata_core.rs` |
| Adaptive register-automaton joins and explicit scheduling policies | `src/scheduled_join.rs`; default façade in `src/join.rs` |
| Partial bindings, BGP, inner/left joins, MINUS, union and existence | `src/query_ops.rs` |
| Path builders, packed/generic product runners, binding adapter | `src/path_ops.rs` |
| Compressed scans, CSR projection, PageRank and label propagation | `src/analytics.rs` |
| Default FST scans, counts, batches, BGP and adjacency integration | `src/fst.rs` |
| Direct DAG persistence and validation | `src/fst_snapshot.rs` |

`Fst` aliases the established `PathFstDeltaPow2Bfs` layout. The existing
`AutomatonJoin`, `FstJoin`, `RecursiveJoin`, and mutable backend APIs remain
available. `graph::join_run`/`try_join_run` accept either family, choosing the
adaptive executor for FST and `NativeJoin` for mutable trees. No conversion or
materialized relation is inserted into those calls.

The default API accepts both constant and variable predicates. Raw joins use a
dense set of declared variables; composable BGP leaves allow unmentioned global
variables to remain unbound. Plans borrow immutable graph views; workspaces can
be reused after cancellation and shared plans use separate workspaces per
thread. Packed `Fst` is immutable; `into_mutable()` enables local path-copying
insertions/removals while preserving suffix minimality, on heap, file and
compressed storage. Default updates use `Sampled(1)` with narrow relative
addresses and 32-bit prefixes. Consuming `freeze()` uses rank-free BFS packing;
borrowed snapshots and mutable `compact()` retain their rebuilding path.
RPQs return distinct endpoint pairs. See [FST_UPDATES.md](FST_UPDATES.md) and
[FST_COMPACT_INDEX.md](FST_COMPACT_INDEX.md) for current update behavior.

Comparison-only `compact_blocks()`, `compact_depths()` and `compact_recursive()`
are behind the opt-in `experimental` feature. The default compactor, its
`compact_streaming()` diagnostics and ordinary storage backends remain available
without it. Measurements below describe the original query promotion; subsequent
update/compaction experiments are indexed in
[the results guide](RESULTS.md).

## Persistence and integration

Snapshots preserve packed blocks, forward deltas and suffix sharing. The
`SGF1` v1 format is distinct from existing mutable-tree `SGZ1` snapshots.
Streaming writes use O(block count + height) auxiliary metadata, without a
second graph buffer or a physical-node address directory. Loading validates
blocks and record boundaries, next-level references, reachability, coordinate
domains, logical counts and resource limits. Validation operates on the DAG,
without expanding all its logical paths. Its temporary memory is additional to
the stored-byte limit. Query plans are rebuilt after loading.

The FST builder now validates inputs and rejects overflowing child deltas. A
zero-bit leaf distinguishes an empty singleton domain from `{(0,0,0)}`;
the previous builder dropped that triple. The retained demand-join test now
checks this corrected behavior. Diagnostic sweep directories allocate lazily
and are included in memory accounting. Normal graph storage retains only
O(height) scan layouts and physical state counts in addition to packed blocks.

Direct FST scans and `PatternIndex` use one generic `PatternIter`. The scan
constructor and `PatternBackend` entry point are inlinable, which removes the
small overhead initially introduced by a separate direct iterator.

## Promotion controls

These are synthetic, warm, prepared-traversal controls, with compilation and
workspace initialization outside join timings. Every engine's full bindings
are validated against the fixed-schedule executor and analytical row counts.
The profiling Cargo profile inherits release optimization and retains debug
information. Results are local observations, not general bounds or cold-cache
claims.

| Shape | Rows | Public default | Explicit adaptive | AutomatonJoin | FstJoin |
|---|---:|---:|---:|---:|---:|
| Uniform star | 4,950 | 20.37 ms | 19.99 ms | 21.05 ms | 35.52 ms |
| Skew | 1,053 | 4.08 ms | 4.00 ms | 115.63 ms | 12.12 ms |
| Mixed triangle | 64 | 39.12 µs | 39.25 µs | 36.64 ms | 40.71 µs |

The parity triangle is also checked, with zero rows. Adaptive preparation
certifies it empty, so its prepared visit is only a terminal check. Its
nanosecond timing in the raw data excludes preparation and should not be read
as an end-to-end query cost; prior scheduling reports include preparation.

On the 15-endpoint `p0/p1+` path fixture:

| Runner | Direct FST | FST via PatternIndex | Mutable Heap |
|---|---:|---:|---:|
| Packed NFA | 5.63 µs | 5.59 µs | 12.69 µs |
| Composed language | 17.29 µs | 17.30 µs | 50.55 µs |

Five additional path trials give comparable results. OPTIONAL retains 5,000
rows in 1.81 ms with parameterized probes versus 272.90 ms with RHS enumeration;
disjoint MINUS takes 0.21 ms versus 301.48 ms. These controls retain the earlier
query semantics and performance shape after promotion.

The smoke benchmark also exposes a remaining tradeoff on a 20,000-vertex
functional successor graph (60,000 triples):

| Query | Default FST prepared visit | Mutable NativeJoin prepared visit |
|---|---:|---:|
| Anchored, 1 result | 0.38 µs | 1.31 µs |
| Triangle, 20,000 results | 5.04 ms | 2.40 ms |
| Shared-predicate path, 60,000 results | 10.54 ms | 5.55 ms |

The FST occupies 1,764 allocated bytes for this very regular fixture versus
474,862 bytes for the mutable tree; plan/workspace memory is separate. The
default is not uniformly fastest: the bounded scheduler can still choose a
schedule that explores more candidates. The triangle opens 342,861 nodes versus
134,295 for the native plan. Keep the explicit alternatives for these workloads;
further adaptive tuning is separate from this promotion. This synthetic memory
ratio is not a claim about real-data compression.

Raw outputs: [joins](prototype/results/promotion-joins.jsonl),
[query operators](prototype/results/promotion-query-ops.csv),
[path trials](prototype/results/promotion-paths.csv),
[smoke benchmark](prototype/results/promotion-smoke-bench.log),
[environment and source hashes](prototype/results/promotion-manifest.json).
See the [harness instructions](prototype/README.md) to reproduce them.

## Verification

Normal crate tests include the promoted query/path/analytics tests, randomized
join oracles, and connected BGPs with 4,096/4,097 atoms on a 128 KiB stack.
`tests/fst.rs` covers snapshots, malformed inputs, logical count overflow,
singleton/empty/full-width domains, default dispatch, cancellation/reuse,
scans/counts/paths and plan sharing. README snippets are executable doctests.

The no-feature build and the file/compressed/parallel combination are checked,
as are all-feature examples/benchmarks and rustdoc links. The generic tree's
write helper is now available without backend features, and the smoke benchmark
uses the current restricted-descent API alongside default FST controls.
