# FST memory across updates

Run the measurement with assertions:

```sh
cargo run --release --all-features --example fst_memory -- 65536
```

The default is 16,384 vertices if no argument is supplied. The run covers two
shapes and all enabled storage modes. It adds 1, 16, 256 and 4,096 triples,
undoes them, repeats a write burst, compacts, deletes original packed triples,
compacts again, and restores them. Every checkpoint validates the triple count.
Repeated bursts must reuse arena slots; undoing all edits must leave no live
edit payload; compaction must discard the base index and edit arena.

`graph.memory_usage()` separates these retained allocations:

- **Packed base:** original graph blocks and their layout/cache allocations.
- **Base register:** `(fingerprint, packed cursor)` pairs used to find an
  existing equivalent suffix. This enables minimality after edits; it is
  constructed on the first effective edit, or by `prepare_updates()`.
- **Edit payload:** actual new graph states—occupancy words and outgoing edges.
- **Edit arena:** state slots, reference counts, level tags and spare capacity.
- **Edit register:** lookup tables used to merge equivalent edited states.
- **Free list:** reusable slot IDs after edit states become unreachable.

The register/index, reference counts and slot-reuse bookkeeping are what the
previous explanation called **maintenance metadata**. Live edit payload is
new graph data. The original packed graph remains allocated until compaction,
even when edits make some of its states unreachable.

For the mixed heap fixture, initially 131,072 triples, all sizes below are
**KiB (1,024 bytes)**:

| Phase | Added | Removed | Total | Packed base | Base index | Live edit payload | Arena/register/free list |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 0 | 0 | 1,372.8 | 1,372.5 | 0.0 | 0.0 | 0.3 |
| prepared | 0 | 0 | 2,509.2 | 1,372.5 | 1,136.5 | 0.0 | 0.3 |
| insert | 1 | 0 | 2,510.1 | 1,372.5 | 1,136.5 | 0.5 | 0.6 |
| insert | 16 | 0 | 2,513.2 | 1,372.5 | 1,136.5 | 2.5 | 1.7 |
| insert | 256 | 0 | 2,532.5 | 1,372.5 | 1,136.5 | 12.4 | 11.1 |
| insert | 4096 | 0 | 2,786.4 | 1,372.5 | 1,136.5 | 165.3 | 112.2 |
| undo_insert | 0 | 0 | 2,636.8 | 1,372.5 | 1,136.5 | 0.0 | 127.8 |
| compact_undo | 0 | 0 | 1,372.8 | 1,372.5 | 0.0 | 0.0 | 0.3 |
| delete_base | 0 | 4096 | 2,518.2 | 1,372.5 | 1,136.5 | 2.0 | 7.2 |
| compact_delete | 0 | 4096 | 1,334.4 | 1,334.1 | 0.0 | 0.0 | 0.3 |

The main finding: undoing all 4,096 insertions frees the live payload but
retains about 1,264 KiB of index and reusable edit scaffolding. `compact()`
returns this fixture to its original footprint. Repeating the same burst
reuses slots, rather than accumulating a history of edits. Deleting original
edges does not shrink packed base bytes until compaction either.

These are graph-owned retained-capacity estimates, not process RSS or peak
compaction memory. HashMap bucket overhead is approximate; vector capacities
and index pairs are counted directly. Counts run before each memory sample,
so backend read-cache buffers are included. Input batches and temporary query
workspaces are excluded. File/compressed modes keep the maintenance structures
in RAM too; their packed-base column includes RAM allocations, not disk bytes.

[All 114 measurements](prototype/results/fst-memory-65536.csv) include both
shapes and all three modes. The memory experiment and existing five update/source
integration tests passed after adding the breakdown API.

The [runtime lookup experiment](FST_METADATA.md) compares insertion/removal
with the suffix indexes retained or replaced by scans.

The [1M/10M/100M scaling experiment](FST_INDEX_SCALE.md) measures the suffix
index in systemd-limited processes and separates construction peaks.
