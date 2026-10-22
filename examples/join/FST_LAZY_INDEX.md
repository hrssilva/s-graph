# Memoizing suffix lookup across edits

`MutableFst::mem_size_bytes()` includes retained maintenance allocations: the
packed-state index, edited-state lookup tables, arena slots and free list, plus
live edited graph payload. The prototype's memo cache is included too, as
`memory_usage().memo_register_bytes`. Vector capacities are counted; hash-table
overhead is estimated. Allocator bookkeeping, the inline Rust object and
temporary operation workspaces are outside this estimate. File-backed graph
bytes are reported separately by `disk_bytes()`; serialized snapshots contain
graph data without update indexes or edit history.

The existing `Indexed` strategy and its 16-byte entries remain the default.
Select the prototype explicitly:

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};

let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert(9, 2, 13).unwrap();
graph.remove(9, 2, 13).unwrap();
// Repeating the edit reuses suffix lookup results from the previous edit.
graph.insert(9, 2, 13).unwrap();
println!("{:?}", graph.update_lookup_stats());
println!("{:?}", graph.memory_usage());
```

The subsequent [sampled-index and depth-start experiment](FST_SAMPLED_INDEX.md)
adds `UpdateLookup::Sampled(rate)` and makes packed scan misses jump directly
to the requested depth after a one-time depth-start preparation. The results
below predate that scan improvement.

## Algorithm and invariants

The immutable packed base makes cross-edit memoization straightforward. A
signature consists of the occupancy word and ordered child identities. A
per-level exact hash table maps a signature to either a packed address or a
confirmed absence. Hash collisions are resolved by full signature equality.

An edit also remembers the original packed states it traverses. Their addresses
are already known, so no scan is needed to cache those answers. This makes
restoring a traversed suffix cheap, including its first restoration.

```text
edit(triple, operation):
    path = traverse current graph for triple
    if operation would not change the graph: return unchanged

    replacement = EMPTY
    for level, original_state in reverse(path):
        original = decode original_state
        if original_state belongs to packed base:
            memo[level].remember(original.signature, original_state.address)

        changed = apply operation/replacement to original
        if changed is empty:
            replacement = EMPTY
        else if every child of changed belongs to packed base:
            key = changed.signature
            if memo[level] contains key:
                base_match = memo[level][key]  // may be a cached absence
            else:
                base_match = scan packed physical states at this level for key
                memo[level][key] = base_match
            replacement = base_match or intern_exact_edited_state(changed)
        else:
            replacement = intern_exact_edited_state(changed)

    acquire replacement root
    publish replacement root and release old root
```

Only signatures whose children are packed addresses enter the memo. Edited
arena slots can be reclaimed and reused; caching their identities across edits
would need a generation scheme or invalidation. The existing edited-state table
continues to intern and reclaim live edited states exactly. This therefore
shares the edited-state table with `Indexed` and `ScanBase`; it is not literally
`ScanAll` with an additional cache.

Neither an insertion nor a removal changes the packed base. Positive answers
and confirmed absences consequently remain valid even if an edited state is
freed. `compact()` rebuilds packed addresses, clears the memo and counters, and
preserves the selected lookup strategy. Serialization omits the memo; loading
chooses the default strategy. `prepare_updates()` is a no-op for `Memoized`.

The subsequent [in-place freeze prototype](FST_FREEZE_IN_PLACE.md) releases the
memo while sealing only edited states, and supports reopening for the next
batch. Its ordinary SGF1 snapshots still require compaction.

A scan does **not** retain every state it inspects. Only requested signatures
and original packed states on edited paths are retained. Otherwise the first
unsuccessful scan could construct almost a complete index.

## Costs and scope

- Warm cached signatures use exact hash lookup without scanning the base.
- A new signature still pays the scan baseline's cost. A different edit can
  reuse cached suffixes without repeating the same complete triple, but shared
  prefixes alone do not guarantee that all changed suffixes are cached.
- Memory follows the signatures encountered by edits, including negative
  answers. These prototype entries store full signatures and therefore cost
  more per entry than the packed index's 16-byte fingerprint/address pair.
- This prototype has no eviction or automatic promotion to a complete index.
  A sufficiently broad write workload can make its cache exceed the full
  index. Compaction releases the cache. A bounded cache could safely evict
  answers and fall back to scans, preserving minimality.

## Measurement

```sh
cargo run --release --all-features --example fst_lazy_perf -- 4096 32 3 all
cargo run --release --all-features --example fst_lazy_perf -- 65536 64 3 heap
```

The benchmark compares `Indexed`, `ScanAll` and `Memoized`, with two fresh-graph
passes and reversed strategy order. Both fixtures have two original triples per
vertex. It times the first insertion batch including preparation, two repeated
batches after undo, a fresh batch in different subject regions, deletion and
restoration of original triples, and edits after compaction. Output reports
microseconds per changed triple, per-phase packed scan/cache-hit counts, cache
population and retained allocation estimates. Build, accounting and validation
are outside update timing. Wall times are sequential and unpinned.

Every operation checks its change count and changed triple membership. Inverse
batches must restore packed identity with zero live edited states. Each complete
strategy/pass is compared against the original triple set. Unit tests assert
that repeated edits perform no additional packed scans after warm-up, and that
compaction clears memoized answers. The update integration suite also exercises
the prototype on heap, file and compressed backends, including randomized edits,
live minimality checks, automata queries and snapshots.

## Results

The larger heap case contains 131,072 mixed triples / 72,726 physical states,
with 64 changed triples per batch. Times are microseconds per changed triple;
the first insertion includes complete preparation for `Indexed`. First/fresh
rows aggregate two passes; warm rows aggregate four samples across two passes.

| Strategy | First insert | Warm insert | Fresh region insert | Warm delete original | Warm restore original |
|---|---:|---:|---:|---:|---:|
| `Indexed` | 119.27 | 2.36 | 2.70 | 4.02 | 4.20 |
| `ScanAll` | 2,649.95 | 2,638.73 | 2,587.47 | 580.66 | 1,992.12 |
| `Memoized` | 1,935.50 | 2.08 | 1,580.86 | 3.91 | 3.65 |

The warmed memo eliminates packed scans for all four repeated phases. Its
insertion cache contains 178 signatures (70 confirmed absences), occupying
15,803 bytes versus the complete base index's 1,163,760 bytes: 98.6% less
base-lookup memory for this working set. Including the packed graph, edited
payload and other maintenance structures, retained RAM after the warm insertion
is 1,433,769 bytes versus 2,581,726 bytes with `Indexed`.

The fresh-region insertion performs 161 packed scans per 64-edit batch. After
also exercising fresh regions, original-triple deletions and restorations, the
memo grows to 614 entries / 84,006 bytes. It remains much smaller than the
complete index here, but the growth is evidence that memory depends on the
accumulated working set. The first undo of a batch can also discover uncached
intermediate signatures, even though original traversed states were cached.

At 4,096 vertices / 32 edits on the mixed fixture:

| Backend | Indexed warm insert, µs | Memoized warm insert, µs | Memo after warm insert | Complete base index |
|---|---:|---:|---:|---:|
| Heap | 1.79 | 1.76 | 7,146 B | 40,872 B |
| File | 3.91 | 1.84 | 7,146 B | 40,872 B |
| Compressed | 8.86 | 2.18 | 7,146 B | 40,872 B |

These small backend results are consistent with memo hits avoiding packed
candidate reads. They do not remove the cost of reading the actual edit path,
and first-time scans remain expensive on those backends.

The regular functional control is less favorable for memory: at 65,536
vertices its entire base index is only 320 bytes, while the warm memo occupies
1,871 bytes. Warm insert times are similar (1.81 versus 1.83 µs). Memoization is
useful when a small recurring edit working set sits over a much larger physical
base; it does not universally improve on the complete index.

The full experiment produced 768 phase records, all with successful assertions.
Five mutable-FST unit tests and five update/source integration tests passed;
the heap/file/compressed update exercises each include both lookup strategies.
Clippy reports the same eight existing library warnings, with no new warnings
in the changed code.

[Benchmark source](../fst_lazy_perf.rs) ·
[Raw phases, summary and build manifest](prototype/results/fst-lazy/) ·
[Earlier scan comparison](FST_METADATA.md)
