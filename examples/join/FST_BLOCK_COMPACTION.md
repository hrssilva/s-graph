# In-memory compaction by retiring source blocks

`compact_blocks()` consumes a mutable or frozen graph, copies its already
minimal live states into normal packed BFS storage, and releases each source
block after copying its last live record. It creates no metadata spill files
and does not build expanded graph copies or repeat minimization.

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert(9, 2, 13).unwrap();
let (packed, stats) = graph.compact_blocks().unwrap();
assert!(packed.contains(9, 2, 13));
println!("{stats:?}");
```

The output is ordinary `PackedFst<S>`, with 32-bit forward deltas and standard
SGF1 snapshots. The consuming method does not provide rollback if a format
limit fails during copying. Existing `freeze`, `to_fst` and `compact` remain
unchanged. All three built-in backends implement `ReclaimingBlockStorage`.

This comparison prototype requires the opt-in `experimental` feature, alongside
any desired backend features. It is excluded from normal library builds.
`--all-features` enables it for the recorded reproduction commands. The default
rank-free `freeze()` and `compact_streaming()` diagnostics do not require it.

| API | Current implementation |
|---|---|
| `MutableFst::freeze()` | Default rank-free streaming BFS compaction |
| `MutableFst::to_fst()` / `compact(&mut self)` | Borrowing physical-state rebuild, preserves the source until success |
| `FrozenFst::compact()` | Calls the default rank-free consuming `freeze()` |
| Mutable/frozen `compact_blocks()` | This incremental source-block retirement prototype |
| Mutable/frozen `compact_recursive()` | Recursive prototype: reserve BFS slots, then build child first |
| Mutable/frozen `compact_depths()` | Depth-scoped prototype: fused discovery/copy, then block fixups |
| Mutable/frozen `compact_streaming()` | Default rank-free BFS algorithm with diagnostics |
| `freeze_in_place()` | Separate prototype: seals only live edits, using BFS by default; retains obsolete stored records |

The [recursive compaction prototype](FST_RECURSIVE_COMPACTION.md) provides an
alternative that reserves BFS block positions before recursing, then builds
child first. The algorithm below remains available as `compact_blocks()`.

[Gated perf profiles at 1M and 4M triples](FST_FREEZE_PROFILE.md) examine this
algorithm's rank-query and backend source-access costs separately from setup.
The [64-bit rank checkpoint experiment](FST_RANK64.md) measures the subsequent
lookup change and discusses limiting relocation data to adjacent depths.
The [depth-scoped prototype](FST_DEPTH_COMPACTION.md) implements that next step
with reusable buffers and no whole-graph liveness pass.
The [rank-free BFS prototype](FST_RANK_FREE_COMPACTION.md) then replaces rank
and dense address arrays with edit-arena metadata and block displacement runs.

## Algorithm

```text
mark_live(state, level):
    if live_bitmap already contains state: return
    mark state in its source's bitmap           # packed / sealed / RAM edit
    if state is in original packed storage:
        extend original_live_address_bounds[level] to include state
    if level is not the leaf level:
        for each outgoing child of state:
            mark_live(child, level + 1)

compact_blocks(graph):
    visits = checked_u64(graph.logical_visits)  # already maintained by edits
    discard suffix indexes, memo tables and free-list allocations
    if root occupancy word is zero:
        discard graph; return an ordinary empty PackedFst

    allocate live bitmaps for packed, sealed and RAM-edit address spaces
    mark_live(root, 0)                         # one DFS of the reachable DAG
    build rank prefixes for those bitmaps      # old cursor -> dense live index

    placement = empty simulated output
    for level = root_level .. leaf_level:      # FIRST planning pass
        remember this level's destination start
        for state in live_states_at(level):
            placement.reserve(packed_record_size(state))
            increment this level's output state count
        remember this level's destination end

    relocation = array[live_state_count]       # u32 if output fits, else u64
    reset placement
    for level = root_level .. leaf_level:      # SECOND planning pass
        for state in live_states_at(level), in the SAME order:
            address = placement.reserve(packed_record_size(state))
            relocation[rank(state)] = address

    remaining[block] = number of marked record starts in each source block
    release source blocks whose remaining count is zero
    output = empty backend of the same storage type

    for level = root_level .. leaf_level:      # ACTUAL copying pass
        for state in live_states_at(level), in the SAME order:
            word, children = read current source record
            parent_address = relocation[rank(state)]
            deltas = []
            for child in children:            # none at the leaf level
                child_address = relocation[rank(child)]
                deltas.append(checked_u32(child_address - parent_address))
            output.append_atomic_record(word, deltas)
                                               # starts a new block if needed
            if state is a RAM edit:
                release its arena payload
            else:
                block = source block containing state
                remaining[block] -= 1
                if remaining[block] == 0:
                    release that source block's payload

    finish output blocks
    discard old graph and remaining source descriptors
    return PackedFst(output, domains, level_state_counts, visits), statistics
```

`placement.reserve` simulates atomic records in fixed-capacity blocks; it writes
no payload. Packed records occupy eight bytes for the occupancy word plus four
bytes per physical child. The leaf occupancy word has no child-address array.
The two planning passes do not traverse child subgraphs; they inspect record
sizes. Copying reads each outgoing reference and translates it through the map,
without recursively copying that child. Shared states are written once.

`live_states_at` yields original packed states in increasing address order,
then marked sealed states of that level, then marked edit slots of that level.
Original states use the level bounds collected while marking. Before copying,
sealed states use their stored level byte and edits use their arena level tag.
During copying, sealed/edit level membership is instead inferred from the
planned destination interval, so enumeration never reads a released source
block or arena payload. The prototype can filter sealed/edit marks once per
level rather than retaining explicit per-level lists.

There is no one-to-one replacement of an old block with a new block. Output
block boundaries can differ; source payload is retired only after all of its
live records have been copied. No state is re-minimized in this method.

Original packed records are four-byte aligned, so one live bit per four virtual
address bytes suffices. Sealed records use one bit per virtual byte; edit slots
use one bit per slot. Rank prefixes every 64 slots (one per bitmap word) map cursors to dense
relocation indices, avoiding old-address keys and hash buckets. Relocations cost
four bytes per state when output addresses fit 32 bits, otherwise eight.
Each rank query reads the word prefix and counts the masked partial word; it
no longer recounts preceding complete words. These prefixes are temporary
compaction workspace, released before returning the packed FST.

The writer uses a 264-byte stack record buffer, and the output backend buffers
its current block. Bitmaps, rank prefixes, relocations, per-block live counters
and O(height) level descriptors remain additional RAM. The prototype therefore
reduces graph copies but does not guarantee one block of total workspace.

Original live records are already level ordered. Their live address ranges and
bitmaps enumerate them without scanning unreachable records. DFS-sealed records
and edits are filtered per level; the prototype can inspect their marked
positions once per level, a cost proportional to the accumulated sealed segment
and arena rather than rescanning the original base per level.

## A useful count maintained from the touched path

The SGF1 header counts visits in the expanded prefix trie. Updates now maintain
this count by incrementing when an empty non-root prefix becomes nonempty, and
decrementing when a nonempty prefix becomes empty. Changing a terminal bit in
an existing leaf creates no state visit. Root occurrence is always retained.
An internal u128 counter preserves overflow information; packing checks the
existing u64 format limit. Tests compare against independent rebuilds.

This removes a global dynamic-programming count from compaction without a
graph-sized maintenance index.

## Why the lazy memo cannot prove all old states dead

The memo describes outgoing signatures, not every incoming reference. If two
prefixes share a suffix, editing the first prefix does not prove that suffix
dead: the second can still reference it without ever entering the memo cache.

The prototype therefore marks the complete reachable physical DAG. It can skip
unreachable old records and reclaim wholly dead blocks before copying, but is
not a touched-states-only collector. Moving states also requires repairing
addresses in unchanged parents; the relocation plan handles those during copy.
Exact incremental reclamation would need initial incoming-reference/liveness
information maintained during updates, or another proof that all references
were removed. Local reclamation with stable addresses differs from rewriting
a tightly packed FST. The lazy memo alone supplies neither guarantee.

## Backend behavior and the one-block bound

Retirement preserves logical block numbers. Heap drops source byte vectors;
compressed storage drops compressed chunks and associated cached payload.
File storage retires slot lengths and cache bytes, but its disk extents remain
until the consumed backing file is dropped: safe std APIs do not expose hole
punching. Its output uses the normal backend file, not a metadata spill file.

Copying a BFS base advances through its old blocks in order. A DFS-sealed block
can contain states from several output levels, so several blocks can remain
partially live together. To remove that avoidable source of overlap,
`freeze_in_place()` now seals edits in parent-first BFS order and starts each
nonempty level in a fresh block. Its two-pass address plan permits forward
references, and linear slot buckets replace the old level sort. Existing DFS
segments remain readable; `FreezeOrder::Dfs` is retained for comparison.

The deterministic 8192-triple mixed test inserts into an empty base, then seals
the same minimal edited DAG using both orders, with 1024-byte sealed blocks:

| Sealed edit order | Input block contents | Peak old + output contents | Extra contents at peak | Temporary metadata | Source blocks released |
|---|---:|---:|---:|---:|---:|
| DFS | 108,787 B | 155,751 B | 46,964 B | 26,900 B | 115 / 115 |
| BFS, separate blocks per level | 108,787 B | 109,166 B | 379 B | 27,908 B | 122 / 122 |

Both outputs contain 2547 physical states and 63,304 packed record bytes.
Relocations take 10,188 bytes in both cases. BFS trades seven more source block
descriptors and slightly larger rank bitmaps for earlier payload retirement.
Its block-content overlap falls by 99.2% in this control. This is not a proof
of a one-block bound for arbitrary existing layouts, different input/output
record widths, multiple storage namespaces, or total RAM.

`workspace_bytes` counts explicit temporary heap allocations. The block-byte
high-water counter counts **used uncompressed contents**, excluding capacities,
backend caches, allocator retention, compressed RAM and live edit-arena payload.
Neither counter measures isolated process RSS or establishes a total-RAM limit.

## Benchmark and validation

The existing benchmark adds a `blocks` method alongside full compacting freeze
and both in-place sealing orders:

```sh
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all
cargo run --release --all-features --example fst_freeze_perf -- 65536 256 3 heap
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all dfs
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all bfs
```

Fresh graphs, three trials and alternating method order are used. Cold updates,
compaction and prepared reads are timed separately. Block rows include workspace
and source-block retirement counters; other methods report null instrumentation.
Runs are sequential, unpinned local controls, with no cold-cache or RSS claim.
The optional final argument selects an initial packed base (default), or a
relation constructed entirely as a DFS/BFS sealed segment. The latter isolates
source edit-block layout during subsequent block compaction; it does not
change the queried relation. Initial construction and sealing are outside the
measurement. The `dfs` and `levels` output modes still choose how the *next*
batch is sealed.

### Current measurements

Three-trial medians, profiling build, 4096-byte blocks. With 131,072 mixed
initial triples constructed entirely in the sealed segment and 256 more
insertions, both source orders start with 2,284,654 bytes of block contents:

| Initial sealed order | Block compaction time | Peak old + output block contents | Temporary metadata | Source blocks released |
|---|---:|---:|---:|---:|
| DFS | 24.25 ms | 3,316,638 B | 604,412 B | 567 / 567 |
| BFS | 24.91 ms | 2,286,381 B | 606,620 B | 571 / 571 |

The extra block-content overlap falls from 1,031,984 bytes to 1727 bytes.
Heap time differs by about 2.7% in this small sample; this does not establish
a speed advantage for either layout. Temporary metadata remains O(physical
states and virtual address space); the memory benefit is earlier source
retirement, not removal of relocation tables.

For a packed initial base, the same larger heap fixture compares full rebuilding
with direct block compaction:

| Method | Time | Retained output graph RAM |
|---|---:|---:|
| Historical rebuilding `freeze()` (before rank-free promotion) | 44.30 ms | 1,408,208 B |
| `compact_blocks()` | 10.99 ms | 1,408,328 B |

The block method uses 341,372 bytes of explicit workspace and peaks at
1,403,748 bytes of old plus output block contents, starting from 1,396,864.
Small differences in output block alignment account for retained size changes;
both representations have the same minimal state count and triples.

The speed result does not generalize to every backend. At 8192 mixed initial
triples and 32 insertions, direct block compaction from a packed base takes
2.85 ms on file and 5.09 ms on compressed storage, versus 1.46 / 1.28 ms for
full rebuilding. Multiple planning passes and backend reads are still costly.
For entirely sealed inputs at this size, switching source order from DFS to
BFS changes block compaction from 2.73 to 3.73 ms on file and 4.31 to 6.42 ms
on compressed storage, while reducing the measured content peak from 168,159
to 110,486 bytes. More independent small blocks can increase I/O and compression
costs. BFS is chosen for predictable retirement, not a universal speed claim.

[Raw measurements, summaries and build manifest](prototype/results/fst-block-compact-bfs/)
include all 288 rows. The earlier
[pre-BFS controls](prototype/results/fst-block-compact/) are retained separately.

Tests compare full triples, minimal state counts, logical visitation counts and
cross-backend snapshots against independent rebuilding. Cases cover all lookup
strategies, both sealed orders, four successive batches, shared suffixes, empty
graphs, a DAG representing 2^48 triples, rank boundaries and wide relocation
entries. The synthetic wide-entry test is not an end-to-end graph above 4 GiB.
The complete all-features check passes 134 library/integration tests and seven
README doctests. Clippy reports the eight existing library warnings, with none
introduced in this prototype.

[Implementation](../../src/fst_block_compact.rs) ·
[Benchmark source](../fst_freeze_perf.rs)
