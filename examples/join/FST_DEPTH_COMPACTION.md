# Depth-scoped compaction with fused liveness

The benchmark runner now selects packed inputs only. BFS-sealed measurements
below are historical results from the completed comparison; that fixture is
excluded from subsequent benchmark matrices.

The subsequent [rank-free BFS default compactor](FST_RANK_FREE_COMPACTION.md) reuses
edit-arena metadata and compresses stored-state relocation into block runs.
The rank-based `compact_depths()` described below remains the comparator.

`compact_depths()` is an opt-in prototype on `MutableFst<S>` and `FrozenFst<S>`.
It returns an ordinary packed BFS FST and `BlockCompactionStats`. Existing
`freeze()`, `compact_blocks()` and `compact_recursive()` remain available.

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let base = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = base.into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert(9, 2, 13).unwrap();
let (packed, stats) = graph.compact_depths().unwrap();
assert!(packed.contains(9, 2, 13));
println!("{stats:?}");
```

## Six steps

1. Discard update indexes. Seed the current frontier with the root.
2. Visit current-frontier source blocks in address order. Read each selected
   record, append its occupancy and provisional child references, and mark its
   children in the next frontier **during that same record visit**.
3. Record internal-state addresses for this depth in one reusable relocation
   buffer. Leaf addresses come directly from rank, without a table. Retire
   source blocks whose remaining depths are complete.
4. Read each preceding-depth output block into the reusable block buffer,
   translate its provisional references with the current relocation table,
   and rewrite that block once.
5. Move the rank-prefix allocations to the next frontier and rebuild their
   contents there. Swap frontiers and clear the finished pages for reuse.
6. Repeat through the leaf depth, finish storage, and drop scratch metadata.
   Leaf records have no references to fix up.

```text
current = {root}
next = empty_frontier()
relocation = reusable_vector()
previous_output = empty_range
scratch = reusable_block_buffer()

for depth from root to leaves:
    begin_new_output_block()              # isolates depths for block fixups
    if depth is internal:
        relocation.resize(current.live_count) # preserves allocation capacity
    this_output = empty_range
    for source_block in current.blocks_in_address_order:
        read_into(source_block, scratch)  # paged backends reuse scratch
        for state in current.live_records(source_block):
            word, children = read_record(scratch, state)
            record = [word]
            for child in children:       # one source-edge loop
                next.mark(child)
                record.push(encode_old_child(child))
            address = append(record)
            if depth is internal:
                relocation[current.rank(state)] = address
            extend(this_output, address)
        retire_finished_source_blocks()

    for block in previous_output:
        read_into(block, scratch)
        for parent_record in scratch:
            for old_child_reference in parent_record:
                child = decode(old_child_reference)
                if depth is leaf:
                    target = this_output.start_address + 8 * current.rank(child)
                else:
                    target = relocation[current.rank(child)]
                replace_reference(checked_u32(target - parent_record.address))
        rewrite_committed_block(block, scratch)

    recycle_rank_prefix_allocation(current, next)
    next.finish_rank_and_sort_blocks()
    previous_output = this_output
    swap(current, next)
    next.clear_preserving_capacity()
finish_output()
```

No whole-graph liveness pass, signature graph, incoming-edge index or repeated
minimization is constructed. Every live source transition is consumed once,
while producing provisional output. Output transitions are then visited once
more in sequential block fixups. A source block crossing depths can be loaded
again at the next depth, although its selected records are disjoint. The
one-time root-empty check reads the root word separately.

## Metadata and allocation reuse

There are two live bitmaps (current/next), a **single** rank-prefix allocation
per source namespace that moves between them, and one relocation vector sized
to the widest processed **internal** depth. Leaf records are uniformly eight
bytes and each depth starts at a block boundary, so their addresses are
`leaf_start + 8 * rank(old_leaf)`. Rank follows source-block order, then slot
order, matching output enumeration across base, sealed and edit namespaces.
There is no leaf relocation table. Namespace here means packed base, sealed
records or RAM edit slots. Bitmap pages exist only for source blocks reached
at that depth. Small source-block-to-page directories cover the complete
source block lists; there is no complete-source-address bitmap or complete-DAG
relocation array.

Frontier page arrays, active-block lists, rank prefixes, relocation addresses,
overflow reference streams and the single block scratch buffer retain their
capacities across depths. Clearing resets active entries; it does not allocate
new graph-sized arrays. Rank remains one checkpoint per 64-bit word. The
relocation vector uses u32 addresses until u64 addresses become necessary.
Capacity reuse means retained high-water capacities can exceed the number of
states active at a later, narrower depth. When a whole depth's required size is
already known, relocation and prefix buffers reserve exactly that size rather
than allowing geometric capacity doubling. Incrementally discovered bitmap
pages still grow amortized, then reuse their high-water capacity.

Ordinary provisional references encode the old identity directly in 32 bits,
combining base address/4, sealed byte address and edit slot into disjoint
ranges. They need no per-edge side table. Identities outside that range use an
escape marker and a reusable ordered u64 stream for the corresponding depth.
This preserves correctness for large old address spaces without a global
hash table, but those exceptional edges add depth-sized temporary memory.
The final reference is always a checked u32 forward delta, as in SGF1.

## Block access and retirement

The built-in backends implement reusable whole-block reads and writes through
`ReclaimingBlockStorage`. Heap copies into scratch and updates existing payload.
File reads into the same allocation and writes the resolved block once.
Compressed storage decompresses into reusable scratch and recompresses each
fixed-up output chunk once, reusing its compressed allocation when it fits.
Custom backends inherit correct default methods and can specialize them.

The packed base already has BFS ordering. Advancing to a later live source
block at the current depth proves all earlier blocks are finished. The last
block remains available if it crosses a depth boundary. No global per-state
liveness count is needed for retirement.

Sealed blocks may contain mixed DFS depths and batches. On first reading a
reached sealed block, scan its record headers in the already-loaded buffer to
remember its maximum stored depth. Retire it after that depth. This scans no
child references and needs no preceding liveness walk; unreachable deeper
records can conservatively postpone retirement. Entirely unreached blocks are
never decoded and are released at completion. This retirement rule can retain
more old payload than exact live-per-block counting on some shapes; the block
content peak is measured separately from scratch metadata.

Depth boundaries start fresh output blocks. There can be a few more output
blocks than in `compact_blocks()`, while record sizes, physical state counts,
shared suffixes, logical visits and SGF1 compatibility are preserved. Every
provisional reference is resolved before the graph is returned. A consuming
format-limit failure does not provide rollback.

## Why this follows the s-dict observation

The useful lesson is discovering reachability as records are copied, rather
than first traversing the entire live DAG to build a relocation universe.
Depth frontiers supply duplicate suppression exactly where it is needed.
The input is visited in storage order instead of DFS jumping between depths,
which was costly for paged/compressed source access in the perf profiles.

Our packed output uses BFS order and forward deltas. A parent can be written
before its children have final addresses by leaving provisional references;
the block fixup resolves them once the child depth is complete. That is the
explicit cost of retaining the current output representation. It should be
judged by measurements rather than assuming child-first and BFS compaction
have identical costs. s-dict also compacts small independent lexical automata;
these measurements use one globally shared graph FST.

## Measurement

The comparison uses the same exact 1M/4M synthetic mixed graphs, 2048 final
edits, portable profiling build and FIFO-gated freeze-only perf counters as
[the original investigation](FST_FREEZE_PROFILE.md). Baseline `blocks` includes
the [64-bit rank checkpoints](FST_RANK64.md). All large jobs run in systemd user
services with verified inherited address-space limits. Construction and result
validation are outside timing. Three core-counter trials are used per case. The initial matrix is retained
in `prototype/results/fst-depth-compact/`; a second run in
`prototype/results/fst-depth-compact-reuse/` measures the final exact-reservation
and rank-derived leaf-address refinements. Final comparisons use the first run's unchanged block baseline
and the refined depth implementation. The earlier smoke result is separate
and is not used as a timing baseline.

`workspace_bytes` counts explicit metadata allocation capacities, excluding
backend caches, allocator bookkeeping, output descriptors and block buffers;
`scratch_bytes` reports the reusable block buffer separately. `rank_bytes`
includes the live bits and rank directories; `frontier_bytes` also includes
page directories and block lists. `fixup_ns` is part of `copy_ns`, not an
additional phase to add. Buffer-growth counts cover page words, active-block
lists, rank prefixes and relocations, excluding initial per-block directories.
Block-content peaks are used old-source plus written-output contents, not RSS.

```sh
CARGO_TARGET_DIR=target/automata-prototype cargo rustc --offline --all-features \
  --profile profiling --jobs 1 --example fst_freeze_profile -- \
  -C link-arg=-Wl,--threads=1
python3 examples/join/prototype/fst_freeze_profile.py --methods blocks depths \
  --stat-only --output examples/join/prototype/results/fst-depth-compact
```

## Results

Medians of three runs. The initial and final matrices completed 81 contained
services successfully; the four smoke services are separate.

| Input | Triples | Backend | Block baseline | Depth-scoped | Speedup |
|---|---:|---|---:|---:|---:|
| packed | 1,000,000 | heap | 35.29 ms | 27.37 ms | 1.29x |
| packed | 1,000,000 | file | 282.30 ms | 63.32 ms | 4.46x |
| packed | 1,000,000 | compressed | 337.33 ms | 49.39 ms | 6.83x |
| packed | 4,000,000 | heap | 271.42 ms | 211.59 ms | 1.28x |
| packed | 4,000,000 | file | 2040.95 ms | 405.70 ms | 5.03x |
| packed | 4,000,000 | compressed | 3724.61 ms | 425.91 ms | 8.75x |
| bfs | 1,000,000 | heap | 91.27 ms | 37.09 ms | 2.46x |
| bfs | 1,000,000 | file | 409.53 ms | 75.47 ms | 5.43x |
| bfs | 1,000,000 | compressed | 702.61 ms | 62.31 ms | 11.28x |

Metadata capacities are identical across backends for the same input.

| Input | Workspace before → after | Change | Relocation before → after |
|---|---:|---:|---:|
| 1,000,000 packed | 1,449,056 → 1,023,944 B | -29.3% | 995,512 → 499,520 B |
| 4,000,000 packed | 9,830,172 → 9,302,540 B | -5.4% | 7,360,364 → 6,739,220 B |
| 1,000,000 bfs | 4,129,448 → 4,342,900 B | +5.2% | 995,512 → 499,520 B |

The 1M BFS-sealed case retains more bitmap capacity than the old global
bitmap. Its total metadata rises 5.2% despite the smaller relocation table.
Reuse of high-water capacities prevents repeated allocation, but does not
guarantee less scratch memory on every shape. The 4M widest internal depth
contains 1,684,805 of 1,840,091 live states (91.6%); that table still dominates.
The uniformly sized leaf optimization does not remove this internal table.

| Input | Source blocks / depth reads | Block-content peak before → after | Frontier / relocation buffer growths |
|---|---:|---:|---:|
| 1,000,000 packed | 1,716 / 1,720 | 6,990,504 → 6,994,588 B | 50 / 5 |
| 4,000,000 packed | 9,353 / 9,357 | 38,085,148 → 38,089,216 B | 61 / 5 |
| 1,000,000 bfs | 3,037 / 3,035 | 12,226,966 → 16,439,021 B | 54 / 5 |

Growth counts occur across the complete compaction, not per state or edge.
The reusable block buffer retains 8,000 bytes in these runs (a logical block
is 4,096 bytes; vector capacity growth accounts for the difference). All source
blocks are retired, all physical-state and record-byte counts match baseline,
and counts plus inserted/removed probes pass. Final output RAM can differ
slightly because each depth starts a new block and compressed references differ.

## What the measurements say

The largest gains are on file/compressed input: the new source block reader
avoids the old DFS pattern of repeatedly revisiting chunks across depths.
The packed 4M graph needs 9,357 whole-block reads for 9,353 source blocks; the
few extra reads come from depth crossings. This is an instrumented block-read
count, not an inferred decompression or physical-disk-read count. `mark_ns` is
zero because marking is fused into `copy_ns`, not because marking is free.

Output fixup now occupies 44–57% of heap total time, 34–42% on file, and
44–54% on compressed storage. This is the remaining cost to target when
approaching child-first compaction performance while retaining BFS output.
The data supports removing the separate source liveness traversal; it does
not establish performance parity with s-dict on different workloads.

## Validation and artifacts

The full all-features suite passed 137 library/integration tests and seven
documentation tests. After the capacity/leaf-rank refinements, all 13 mutable
FST unit tests and six update integration tests passed again, including all
backends, lookup strategies, sealing orders, repeated batches, snapshot
validation and query schedules. Tests cover a shared DAG representing almost
2^48 triples, empty/full-width domains, 64-child records, page reordering,
reused allocation capacities, u64 relocation promotion and escaped identities.
Clippy reports only the eight pre-existing warnings.

[Initial matrix](prototype/results/fst-depth-compact/summary.json) ·
[Final matrix](prototype/results/fst-depth-compact-reuse/summary.json) ·
[CSV comparison](prototype/results/fst-depth-compact-reuse/comparison.csv) ·
[Implementation](../../src/fst_depth_compact.rs)
