# Rank-free BFS compaction using edit metadata and relocation runs

This is the default algorithm for consuming `MutableFst::freeze()` and
`FrozenFst::compact()`, on heap, file and compressed storage.
`compact_streaming()` exposes the same implementation with diagnostics.
The other compaction algorithms require the `experimental` feature; none is selected by
`freeze()`. `compact_depths()` is the reference comparator below. Both preserve
root-to-leaf BFS storage, shared suffixes, minimality and SGF1 snapshots.

Consuming freeze requires `ReclaimingBlockStorage` (implemented by all three
built-in backends), including explicit append-block boundaries. Custom storage
implementations must provide those operations before using consuming freeze.
Borrowed `to_fst::<OtherStorage>()`, mutable `compact(&mut self)` and snapshot
serialization keep their existing physical-state rebuild: they preserve the
source, including update metadata, until packing succeeds. They do not retire
source blocks or consume edit forwarding slots. Use consuming `freeze()` for
the bounded-source-retirement batch pipeline:

```rust
use s_graph::Fst;
let mut graph = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096)
    .into_mutable();
graph.insert_batch(&mut [(9, 2, 13), (13, 2, 17)]).unwrap();
let packed = graph.freeze().unwrap();
assert!(packed.contains(9, 2, 13));
```

For diagnostics:

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let base = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = base.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert(9, 2, 13).unwrap();
let (packed, stats) = graph.compact_streaming().unwrap();
assert!(packed.contains(9, 2, 13));
assert_eq!(stats.rank_lookups, 0);
```

## What changed

Sequential state enumeration already supplies the ordinal. The compactor
records addresses as it writes records, without calculating rank to recover
that position. For incoming references, it uses the regular layout of unchanged
packed records and the existing edited-state arena.

### Stored states: compress the address translation into runs

An immutable packed source record and its output record have the same length.
While consecutive live records retain the same relative spacing, their
addresses differ by a constant displacement:

```text
source offsets:   120  136  148  180
new addresses:  4096 4112 4124 4156
                   displacement = +3976 throughout
```

One run represents all those translations. Start a new run when deletion holes
or different block packing change the displacement. Runs are built while
copying each source block in order; a reusable small range table identifies
that block's entries. A single-run block needs one displacement lookup and an
addition. Multiple runs use a binary search over their starting offsets.
There is no bitmap rank directory or dense per-state address array.

A run applies to the live record starts represented by it, including when its
range spans dead source records. Only reachable child identities are queried.
Unsigned wrapping displacement arithmetic handles movement below the old
offset as well as addresses above 4 GiB; final child deltas remain checked u32.

### Edited states: reuse information and storage already maintained

At the start of consuming compaction, every occupied edit-arena slot is live:
reference counting has already reclaimed unused edit states. Every such node
also has its depth and complete outgoing references.

After dropping the signature lookup indexes, the prototype scans arena slots
once and uses their existing `references` fields as per-depth linked lists.
Only O(height) list heads/counts are allocated. There is no additional edit
frontier bitmap, deduplication table, sorting or per-slot relocation array.

When a node is written, its now-unused signature occupancy word temporarily
holds the new address. Its child-vector allocation is released immediately.
Incoming references use the existing edit-slot identity to read that forwarding
address directly. After fixing all parents of the current depth, the forwarded
nodes are released. The public graph is consumed throughout this operation;
no query or subsequent update can observe these repurposed fields. Signature
registers are dropped first, ensuring the arena uniquely owns its signatures.

The full stored-state index and lazy memo are dropped rather than retained as
large compaction indexes. Neither records complete stored-state liveness or
incoming parents. Existing edit metadata removes duplicated edit bookkeeping;
it does not justify assuming old stored records are live or dead.

## Algorithm

```text
drop_update_lookup_indexes()
edit_heads, edit_counts = link_live_arena_slots_by_depth_using_existing_fields()
current_stored = {root, if root belongs to stored data}
next_stored = empty_reusable_bitmap_frontier()

for depth from root to leaves:
    reset_relocation_runs_preserving_capacity()
    begin_fresh_output_block()
    ordinal = 0

    for reached source block, in address order:
        read_block_into_reused_scratch()
        for live source state in offset order:
            append occupancy + provisional child identities
            mark stored children in next_stored
            extend displacement run or start a new run
            ordinal += 1
        remember this source block's run range

    for edited node in edit_heads[depth]:
        append occupancy + provisional child identities
        mark stored children in next_stored
        replace existing signature word with the new address
        release outgoing-vector storage
        ordinal += 1

    assert ordinal == current_stored.count + edit_counts[depth]
    for previous-depth output block:
        read_into_reused_scratch()
        for child reference:
            if edited child:
                target = edit_arena[child.slot].forwarding_word
            else:
                target = child.offset + displacement_run_for(child)
            replace reference with checked_u32(target - parent.address)
        rewrite block once

    release forwarded edit nodes of this depth
    retire finished source blocks
    count/sort next_stored blocks; swap frontiers; clear old frontier
finish output
```

Liveness discovery remains fused with copying. All live stored transitions
are read once from source. Output block fixup remains: this experiment removes
rank lookup and dense relocation indexing, not the requirement to resolve
forward references. Parent references can be arbitrarily ordered and repeated;
shared children do not require sorting the transitions or copying states again.
Source block retirement and large-identity escape streams follow the depth
compactor's existing rules.

## Memory and scope

Stored-state live bitmaps and source-block directories are still necessary;
`frontier_bytes` includes them. `rank_bytes` is zero for this algorithm because
it builds no rank structure. Popcounts remain for record sizes and one-time
frontier counts; they are not per-reference rank queries. `rank_lookups` reports
lookup calls in the baseline depth compactor (one per internal state plus one
per transition, derived from visited counts outside hot loops) and stays zero
here. This avoids adding counter writes to the measured lookup loops.

`relocation_bytes` counts retained run/range capacities. `relocation_runs`
reports the peak number of live run entries; each run occupies 16 bytes on the
64-bit target. Run/range arrays and live-frontier allocations are reused across
depths. The one block buffer is preallocated for the larger of source/output
block capacities, avoiding amortized doubling after an almost-full first block.
No new graph-sized edit mapping is allocated. Existing arena forwarding records
remain until their incoming references are fixed, then are freed.

This representation favors long unchanged packed regions. Dense deletions can
produce more runs; sealed absolute-address records also shrink during copying,
so they may need a run per state. The tests still cover sealed data, but the
benchmark matrix contains packed input only, as requested. It is a full repack:
all reachable stored states are still visited. Forwarding metadata applies to
edits; it does not identify every obsolete packed state on its own.

## Comparison protocol

Use the same exact synthetic mixed 1M/4M triple fixtures and 2,048 final edits
as the previous depth experiment. Three separate perf core-counter trials per
case run under systemd user services with verified inherited address-space
limits. Build, input/output counting, edit preparation and correctness probes
are outside the freeze-only measurement gate. Both methods use the same binary
and portable profiling build. No BFS-sealed case is included.

```sh
CARGO_TARGET_DIR=target/automata-prototype cargo rustc --offline --all-features \
  --profile profiling --jobs 1 --example fst_freeze_profile -- \
  -C link-arg=-Wl,--threads=1
python3 examples/join/prototype/fst_freeze_profile.py --methods depths streaming \
  --stat-only --output examples/join/prototype/results/fst-rank-free
```

The driver checks that every live edit state was forwarded using arena metadata,
all source blocks were retired and no rank structure/lookup was used. Counters
include `rank_lookups`, `relocation_runs` and `forwarded_edit_states` alongside
allocation-capacity, phase and block-content measurements. Workspace excludes
existing source edit-arena storage, backend caches, output descriptors, allocator
bookkeeping and block buffers; scratch is reported separately. It is not RSS.

## Results

Three-run medians; all 36 contained services completed successfully. The
earlier one-trial smoke run is retained separately and is not used here.

| Triples | Backend | Rank-based depth compaction | Rank-free streaming | Speedup |
|---|---|---:|---:|---:|
| 1,000,000 | heap | 27.96 ms | 27.21 ms | 1.03x |
| 1,000,000 | file | 63.96 ms | 62.13 ms | 1.03x |
| 1,000,000 | compressed | 47.37 ms | 47.83 ms | 0.99x |
| 4,000,000 | heap | 208.47 ms | 146.50 ms | 1.42x |
| 4,000,000 | file | 413.90 ms | 337.06 ms | 1.23x |
| 4,000,000 | compressed | 414.49 ms | 343.38 ms | 1.21x |

Metadata capacities are the same for each backend at a given graph size.

| Triples | Workspace before → after | Reduction | Relocation before → after |
|---|---:|---:|---:|
| 1,000,000 | 1,023,944 → 427,232 B | 58.3% | 499,520 → 51,280 B |
| 4,000,000 | 9,302,540 → 2,176,064 B | 76.6% | 6,739,220 → 354,864 B |

| Triples | Baseline rank lookups → streaming | Peak displacement runs | Edit states forwarded in arena |
|---|---:|---:|---:|
| 1,000,000 | 1,497,702 → 0 | 1,637 | 614 |
| 4,000,000 | 7,680,169 → 0 | 10,140 | 807 |

All counters ran without multiplexing. Physical state counts, retained output
RAM, source/output record-byte counts, source block reads, output fixup block
counts and block-content peaks match the baseline. The single scratch buffer
retains 4,096 bytes instead of 8,000 in the baseline; it is reported separately
from workspace. No persistent index or snapshot overhead was introduced.

At 1M, elapsed times differ by only a few percent, including a roughly 1%
compressed slowdown. These runs do not establish a decisive small-graph speed
win. Instruction counts fall about 7–12%, but fewer instructions alone do not
guarantee lower elapsed time.

At 4M, the heap improvement is decisive in these trials. Its fixup phase falls
from 116.14 to 65.42 ms. File/compressed improve by 1.21–1.23x.
The 4M lookup map is much smaller; this is consistent with avoiding the large
per-state relocation working set, but these core-counter runs do not measure
cache misses directly. Heap IPC rises from 2.19 to 2.68.

## Validation

All 16 mutable-FST unit tests, six update integration tests and seven README
doctests passed with all features. The integration matrix includes both sealing
orders, all lookup modes, repeated batches, all query schedules and snapshots
on heap, file and compressed backends. New tests compare byte-identical
serialized output with the rank-based method for live edits with recycled
slots, across all backends and lookup modes. Additional checks cover empty and
full-width domains, nearly 2^48 logical triples in a tiny shared DAG, holes,
negative displacement, backward/repeated lookups, u64 targets, allocation reuse
and escaped edit references. Clippy reports only the eight existing warnings.

[Counters, summaries and manifests](prototype/results/fst-rank-free/) ·
[Comparison CSV](prototype/results/fst-rank-free/comparison.csv) ·
[Implementation](../../src/fst_stream_compact.rs)
