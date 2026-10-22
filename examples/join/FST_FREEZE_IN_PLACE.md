# Freezing edited FSTs in blocked storage

The prototype separates sealing a graph from compacting all its records.
`freeze_in_place()` consumes the editable graph and returns `FrozenFst<S>`.
It retains the original packed backend, writes only live edited states into
another blocked segment of the same backend type, and releases update indexes,
memo tables, arena slots and the free list. It performs no traversal or
minimization of the original packed graph. The reachable automaton is already
minimal after updates.

This is an append-only blocked prototype. It does not splice changed records
into the original base file or reclaim holes in existing blocks. A file graph
owns its original temporary file plus a separate temporary file for the sealed
segment; compressed graphs keep the new records in compressed blocks. Further
batches append to the existing sealed segment, rather than creating a chain of
overlay objects or one segment per batch.

## API and periodic batches

```rust
use s_graph::{Fst, FreezeOrder, mutable_fst::UpdateLookup};

let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert_batch(&mut [(9, 2, 13), (10, 2, 14)]).unwrap();

let frozen = graph.freeze_in_place().unwrap(); // default: parent-first BFS edit blocks
assert!(frozen.contains(9, 2, 13));
// frozen.search(&machine), scan, counts, paths and source() remain available.
assert_eq!(frozen.memory_usage().base_register_bytes, 0);
assert_eq!(frozen.memory_usage().edit_arena_bytes, 0);

let mut graph = frozen.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.remove(9, 2, 13).unwrap();
let frozen = graph.freeze_in_place_with_order(FreezeOrder::Levels).unwrap();

// Optional later maintenance: reclaim obsolete records into ordinary PackedFst.
let (packed, stats) = frozen.compact_blocks().unwrap();
assert!(!packed.contains(9, 2, 13));
```

`FrozenFst` is an immutable wrapper over the shared source engine. Its immutable
methods use the same automata, query and analytics machinery; editing requires
consuming `into_mutable[_with_lookup]`. The wrapper retains no update-table or
arena allocation. It forwards immutable methods through `Deref` and also
implements the pattern and adjacency backend traits directly.

This prototype leaves `to_fst()` and default lookup choices unchanged.
The current `freeze()` uses [rank-free streaming BFS compaction](FST_RANK_FREE_COMPACTION.md).
The original comparison below predates that promotion. `FrozenFst::compact()`
returns an ordinary packed graph. `serialize`
and its stream/path variants currently produce standard SGF1 snapshots through
full compaction. There is no native segmented snapshot or crash-recovery format
in this prototype. Freezing in place is not a durability boundary.

## What the DFS tree teaches us

The old DFS tree's `insert_new_branch` replaces a complete growing node record
with `remove_bytes` + `insert_bytes`, preventing the header and outgoing array
from splitting across blocks. Its block index translates positions after block
growth/splitting, while ancestor spans describe subtree sizes. File storage adds
physical-slot indirection so local movement can reuse existing storage.

The FST's outgoing addresses and suffix sharing require a different treatment.
A shared state can have incoming edges from many DFS branches. Splicing bytes
or splitting a logical block can invalidate addresses held by parents outside
the edited path. A DFS layout does not make every shared suffix a unique,
contiguous subtree. This prototype borrows atomic blocked-record writes and
stable reuse of existing storage, and avoids moving old records entirely.

## Record layout and algorithm

Original packed states keep their existing 32-bit forward deltas and original
addresses. New sealed records contain:

```text
level:u8 | occupancy_word:u64 | ordered_absolute_child_cursors:u64[]
```

A cursor tag distinguishes sealed records from original packed addresses. A
separate tag identifies live RAM edit slots while a graph is mutable. The sealed
record can therefore refer backward to an older state, forward or backward
within the sealed segment, or into the original base. No per-state forwarding
directory remains after freeze. Only O(height) physical counts are retained for
future lookup preparation. Sealed blocks use at least 1024 bytes so even the
largest 521-byte absolute-link record fits atomically.

```text
freeze_in_place(graph):
    discard suffix indexes, memo tables and free list
    if live edited states exist:
        allocate relocation[edit_arena_slots], initially UNWRITTEN
        bucket live edit slots by level, from root to leaves
        simulate record packing, assigning relocation[slot] for every live edit
            begin a fresh block at each nonempty level
        for each nonempty level, from root to leaves:
            begin a fresh block
            for each live edit slot in this level:
                take its word and children out of the arena
                translate edited child cursors through relocation[]
                append the complete record, then release its RAM payload
        root = relocation[root.slot] if root is edited, otherwise root
        finish writes to sealed blocked storage
    release edit arena
    return immutable graph owning original base + sealed storage + root
```

Each live edited state is written once, regardless of its incoming multiplicity.
Replacing its cursor consistently preserves sharing and minimality. Absolute
child cursors remove the need to calculate addresses for the whole old graph.
The relocation map costs eight bytes per arena slot, including reusable holes;
level buckets use one `usize` per live edited state plus O(height) headers.
Bucketing takes linear time and needs no sort or graph traversal. Preassigned
addresses make forward references possible even while children remain in RAM.
Original-base size does not determine this workspace. Index destruction still costs time proportional to retained
maintenance allocations. Backend finalization may visit existing sealed-block
metadata, so sealing is not claimed to have constant cost independent of the
accumulated sealed segment.

## DFS versus level layout

`FreezeOrder::Levels` is now the default: it writes parent-first BFS levels and
starts each nonempty level in a fresh block. It also starts the first level of
each new batch in a fresh block, so batches never append into a prior level's
tail. All three built-in backends support these explicit append boundaries.
Custom `BlockStorage` implementations must implement `start_new_append_block`
to use BFS sealing; an unsupported backend returns an error.

`FreezeOrder::Dfs` retains the original child-first recursive writer for
comparison. These choices affect newly sealed records only. The original base
keeps its layout, and later batches retain all prior addresses. RAM updates
still copy the affected path and canonicalize bottom-up; BFS describes their
eventual blocked representation, not a change to the update's dependency order.

The BFS change lets block compaction finish each source block within a single
output level. DFS blocks could contain many levels and stay partially live
until late in copying. There is a cost: more block descriptors, unused tail
space in file slots, and potentially worse compression for small levels. Heap
storage trims unused byte-vector capacity. The sealing writer retains an
additional temporary level-bucket vector compared with DFS.

The earlier timings did not establish a material locality benefit for DFS.
The larger regular-fixture difference included the old level writer's sort;
the current BFS implementation removes that sort. Neither layout solves
incoming-link repair for a shared DAG. The earlier
[whole-FST layout comparison](LAYOUT_COMPARISON.md) already found similar query performance on most fixtures,
with BFS ahead on skew. That experiment predates later scheduling improvements.
The new benchmark below isolates the order of the changed segment; it is not a
new comparison of two completely DFS/BFS-organized base graphs.

## Limits and validation

Obsolete original and sealed records remain allocated. Repeated batches can
therefore accumulate dead storage until explicit compaction. Sealed transitions
use eight bytes instead of the original packed representation's four-byte
deltas. Read traversal dispatches on the cursor kind; its performance can differ
from fully compacted storage. `memory_usage().packed_base_bytes` includes both
backends, and file `disk_bytes()` includes both backing files.

The memo cache is discarded between batches. This change improves freezing; it
does not remove first-encounter scan costs during cold memoized updates. The
batch implementation still applies individual edits after sorting.

The update integration test runs four successive batches for each lookup mode,
each freeze order and all enabled storage backends. It checks complete triple
sets, all automaton schedules, minimal reachable-state counts, and SGF1 snapshot
round trips after each freeze. It also exercises a full 64-child record with
512-byte original blocks. Unit checks cover unchanged original base bytes, zero
retained update allocations, inverse edits, empty/singleton/full-width domains,
and a shared DAG representing 2^48 triples.

## Benchmark

```sh
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all
cargo run --release --all-features --example fst_freeze_perf -- 65536 256 3 heap
```

The benchmark builds a fresh graph for each method/trial, uses memoized updates,
and times ownership conversion, cold batch insertion, freezing, whole-graph
scans, prepared automaton queries and freezing without changes separately.
Three trials reverse method order on alternating trials. Accounting and
validation occur outside the timed freeze. Reported allocation sizes are
retained estimates; the relocation-map size is calculated separately. They are
not isolated process-RSS or full temporary-allocation peaks. Runs are sequential,
unpinned local controls, and the query read timings are small samples.

### Historical results: DFS versus the original deepest-first level writer

These measurements precede the parent-first, block-aligned BFS implementation.
In this table, “levels” means the old sorted, deepest-first writer. It does not
describe the current default.

For 131,072 mixed triples / 72,726 original physical states and 256 inserted
triples on heap storage, the batch leaves 148 live edited states:

| Method | Freeze | Freeze without changes | Retained graph RAM after freeze |
|---|---:|---:|---:|
| Historical rebuilding `freeze()` | 45.53 ms | 44.80 ms | 1,408,208 B |
| In place, DFS | 39.80 µs | 0.56 µs | 1,410,328 B |
| In place, levels | 43.16 µs | 0.43 µs | 1,410,328 B |

In-place sealing writes 148 records and uses a 1,216-byte arena relocation map;
it retains the original 1,405,420-byte graph allocation instead of rebuilding
72,726 states. Mutable RAM before freezing is about 1,471,105 bytes in this
memoized case. All retained lookup, payload-arena and free-list allocations are
zero afterward; both storage backends are included in the graph RAM above.

The improvement is specifically freezing. Cold memoized insertion of this batch
still takes about 257–260 ms, dwarfing in-place sealing. This prototype does not
establish fast cold bulk ingestion. Ownership conversion to mutable is about
0.8 µs in the same control.

Prepared whole-graph automaton traversal takes 5.00 ms after full compaction,
5.32 ms after DFS sealing and 5.31 ms after level sealing. The new-triples query
takes 10.00 / 11.29 / 11.34 µs respectively. This small test shows a read cost
for the segmented/tagged representation, and little sensitivity to the order of
its small edited segment.

The regular functional control has only 11 original states and 438 new edited
states. Freeze times are 162.49 µs compacting, 34.06 µs DFS and 48.70 µs levels.
Here DFS avoids the extra sort used by the level writer; the experiment does
not isolate cache-locality effects. It retains 10,242 bytes versus 6,928 bytes
after full compaction because of old records and wider new transitions.

At 4,096 vertices / 32 mixed insertions, DFS in-place freeze takes 5.54 µs on
heap, 55.70 µs on file and 8.31 µs on compressed storage. Compacting freeze takes
1.05 / 1.28 / 1.26 ms respectively. File timings include temporary-segment file
creation, and are particularly sensitive to the small sample size.

The earlier whole-base DFS/BFS results and this edited-segment test do not
justify changing the default base layout. Stable addresses and removal of
global rebuilding provide the major freeze improvement; those benefits do not
depend on DFS. BFS is now preferred for its source-block retirement behavior
during later compaction.

Validation passed for 131 distinct library/integration tests and seven README
doctests. The new repeated-batch test covers all four lookup modes, both orders,
four successive batches and all three storage modes. Clippy reports the same
eight pre-existing library warnings.

[Benchmark source](../fst_freeze_perf.rs) ·
[Raw measurements and build manifest](prototype/results/fst-freeze/)

### Current parent-first BFS control

After replacing the sort with linear level buckets and isolating level blocks,
the 131,072-triple mixed heap fixture above seals its 148 live edited states in
41.76 µs for BFS versus 40.82 µs for DFS. Prepared whole-graph queries take
5.357 versus 5.354 ms. Retained RAM is 1,410,376 versus 1,410,328 bytes; BFS
adds two block descriptors. These three-trial medians show no material heap
time advantage for DFS. In the regular functional control, BFS sealing takes
34.73 µs versus 36.36 µs for DFS, removing the old sort penalty.

Explicit level boundaries can matter more on the paged backends. With 8192
mixed initial triples and 32 insertions, file sealing takes 114.06 µs for BFS
versus 75.49 µs for DFS; compressed sealing takes 9.69 versus 8.57 µs. File
timings include temporary segment creation and remain noisy. BFS costs another
96 retained RAM bytes on file and 174 on compressed storage in this control.
Unused file-slot tails also consume disk capacity and are not counted as RAM.

[Current raw measurements and manifest](prototype/results/fst-block-compact-bfs/)
record both initial storage layout and the next batch's sealing order. For the
timings immediately above, `input_layout` is `packed` and output modes `dfs`
and `levels` select the edited segment's order.

See [block compaction](FST_BLOCK_COMPACTION.md) for the current BFS/DFS memory
comparison, speed measurements, and the remaining relocation workspace.
