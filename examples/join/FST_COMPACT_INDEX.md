# Compact update indexes: current defaults

Ordinary `packed.into_mutable()` uses `UpdateLookup::Sampled(1)` with the compact
representation. It stores a 32-bit fingerprint prefix for every indexed physical
state and its address as a relative 16/32/64-bit offset. Address width and origin
are chosen separately for each FST level. All three storage backends use this
same representation. Sampling remains an explicit latency/memory choice:

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable(); // default: Sampled(1)
assert!(graph.insert(9, 2, 13).unwrap());
assert!(graph.remove(9, 2, 13).unwrap());
let packed = graph.freeze().unwrap(); // rank-free BFS compaction
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Sampled(8));
graph.prepare_updates(); // optional: otherwise the first effective edit prepares
assert!(graph.insert(9, 2, 13).unwrap());
```

`Sampled(rate)` uses the same representation with one prefix every `rate`
entries. The rate must be nonzero; construction rejects zero by panicking.
`UpdateLookup::default()` is `Sampled(1)`. The redundant `Indexed` variant was
removed; existing callers should use `Sampled(1)` or `Default::default()`.
There is no retained copy of the old full `(u64 fingerprint, u64 address)` index.

## Representation and correctness

Preparation walks physical states (not triples), computes full fingerprints and
sorts `(fingerprint, address)` pairs within each level. It then retains:

- the upper 32 bits of every selected fingerprint;
- one address origin per level;
- each address's offset from that origin, using the smallest of `u16/u32/u64`
  that can represent the whole level's span.

Truncation preserves fingerprint ordering. Binary search includes the complete
range of matching prefixes, including ranges crossing sample boundaries. Every
candidate is checked against its full occupancy word and ordered child identities.
A prefix collision can add candidate reads; it cannot merge different states.
Local edits continue preserving exact live minimality.

The high tags distinguishing packed/sealed cursors are preserved by arithmetic.
A level mixing segments far apart in cursor space can fall back to 64-bit offsets.
There is no restriction that the complete graph fit in 32-bit addresses. Width
selection dispatches once per candidate search; narrow address decoding requires
one addition and no extra graph traversal.

For `n` indexed states at a level with offset width `b` bytes, retained entry
storage is approximately `b*n + 4*ceil(n/rate)`. The default has rate 1 and therefore
uses 6/8/12 bytes per state. Add per-level origins, representation/vector metadata
and actual allocation capacities. The empty graph's zero-word state is omitted
from index entries. `memory_usage().base_register_bytes` includes the full
retained index allocation; `mem_size_bytes()` includes it in total graph RAM.
These measurements exclude allocator bookkeeping and the inline graph object.

Preparation still materializes full temporary pairs and sorts them. Smaller
resident indexes do not imply a smaller construction workspace or eliminate
one-time preparation latency. `prepare_updates()` pays that cost before a write
burst; duplicate insertions and absent removals skip automatic preparation.

## Scan and memoized alternatives

`Memoized` remains explicit and caches exact requested signatures and confirmed
absences across edits. `ScanBase` and `ScanAll` retain their scan semantics.
Their packed lookup uses the start-only depth cache: one 64-bit cursor per level,
built lazily in one pass over non-leaf record headers. Each subsequent scan jumps
directly to the requested depth. It does not allocate a per-state directory.

The occupancy union/intersection and degree filters were removed, together with
the sealed summary cache and `summary_rejections` counter. The broader mixed
fixture showed no pruning benefit. There is no leaf-summary construction pass.
Sealed storage retains its ordinary scan path. Scan-cache allocation capacity is
included in packed-base memory, and Sampled modes do not construct that cache.

Snapshots store graph data, not indexes, memo entries or start caches. Loading
chooses `Sampled(1)`. `compact(&mut self)` clears metadata and preserves the
selected lookup strategy. Consuming `freeze()` releases index memory and uses the
default rank-free BFS compactor. The packed format and read traversal are unchanged.

## Historical comparisons

[FST_SAMPLED_INDEX.md](FST_SAMPLED_INDEX.md) records the original 64-bit sampled
representation. [FST_LEVEL_METADATA.md](FST_LEVEL_METADATA.md) records narrow
sampled indexes and the removed summaries. Those results used the old 16-byte
Indexed control and must not be read as current default memory measurements.

The promotion comparison uses the same [benchmark](../fst_lazy_perf.rs) and
[runner](prototype/fst_level_metadata.py), with a saved pre-promotion binary.
The new default is compared on heap/file/compressed storage, including repeated
edits, inverse edits, snapshot compaction and fresh-region updates. Medium heap
measurements combine three invocations, each with two reversed fresh-graph passes.
Large fixtures contain 1,048,576 triples and run under systemd with verified
1 GiB hard address-space limits. Timings are sequential, unpinned wall time;
retained capacities are not RSS or peak preparation memory.

## Promotion measurements

The promotion matrix was recorded immediately before removing the redundant
Indexed enum spelling. Its after-column Indexed and Sampled(1) share the same
compact structure/matcher, now exposed only as Sampled(1). Historical JSON labels
are retained. The final API cleanup removes the duplicate mode and its selector
branches, while tests verify Default is exactly Sampled(1).

| Mixed fixture | Former default index | Compact default index | Warm insertion before → after |
|---|---:|---:|---:|
| 131,072 triples, heap | 1,163,760 B | 578,032 B | 1.43 → 1.42 µs/edit |
| 1,048,576 triples, heap | 4,172,184 B | 2,082,286 B | 1.75 → 1.74 µs/edit |
| 8,192 triples, file | 40,872 B | 15,602 B | 2.51 → 2.43 µs/edit |
| 8,192 triples, compressed | 40,872 B | 15,602 B | 5.32 → 5.31 µs/edit |

The medium/large cases save about **50%** of retained default index memory.
Both combine three invocations with two reversed fresh-graph passes each; small
backend cases have one invocation. Large fresh-region insertion is 2.18 → 2.29
µs/edit and complete index preparation is 19.47 → 20.28 ms. Timings are unpinned;
small changes should not be treated as decisive speedups or universal guarantees.
Construction still sorts full temporary pairs. Tiny functional FSTs can grow by
a few hundred bytes because per-level encoding metadata dominates entry savings.

An initial single-invocation large comparison had an outlying slow pass. It is
archived separately, and the table uses the subsequent three-invocation matrix.
Every large job completed under verified 1 GiB RLIMIT_AS and systemd success;
cgroup MemoryPeak is unavailable in this environment. There is no RSS claim.
[Raw measurements, aggregate results and manifest](prototype/results/fst-default-index/).

The full library suite, snapshot/update integration tests and seven README
doctests pass. After enum removal, the mutable-FST suite and all six update tests
were rerun, and all targets compile with all features. Tests cover default rate
selection, preparation idempotence, exact prefix collisions, all address widths,
tagged cursor fallback, inverse edits, minimality and backend snapshots.
