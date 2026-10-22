# Recursive compaction: reserve BFS levels, fill them child first

This comparison API requires `--features experimental` (also enabled by
`--all-features`). Ordinary freeze and streaming diagnostics are available
without it.

Status: implemented opt-in prototype. Mutable and frozen graphs expose
`compact_recursive()`, returning `(PackedFst<S>, BlockCompactionStats)` on all
three built-in backends. `compact_blocks()` retains
[the previous algorithm](FST_BLOCK_COMPACTION.md) as a control; ordinary
`freeze()` remains unchanged.

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let mut graph = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096)
    .into_mutable_with_lookup(UpdateLookup::Memoized);
graph.insert(9, 2, 13).unwrap();
let (packed, stats) = graph.compact_recursive().unwrap();
assert!(packed.contains(9, 2, 13));
println!("{stats:?}");
```

The useful idea in `s-dict` is that a parent is finalized after its children,
using their final addresses. Its `PackedAutomaton::compact` combines discovery,
memoized relocation and writing in one postorder walk. It can append children
first because records use backward deltas. Its lexical blocks are independent
automata; our blocks belong to one shared DAG.

For our fixed-width forward-delta records, separate **allocation order** from
**completion order**: reserve output block numbers root first, then recursively
finish levels child first. This preserves physical BFS order while writing the
root's payload last. Recursion is over complete levels; merely appending each
parent before recursively appending its children would produce DFS preorder.

## Six steps

1. Mark the live physical DAG from the edited root. Retain live-start bitmaps
   and per-source-block live counts. Record live level ranges per sealed block so
   later enumeration can skip blocks belonging to other levels.
2. Reserve this level's output blocks, without allocating their payload. One
   sizing walk over its live records determines block counts and used lengths.
   Start each level in a fresh block to simplify block completion.
3. Recurse to reserve and compact the next level. Output block numbers are
   therefore reserved in BFS order.
4. On return, rebuild this level's blocks, translating all child references
   through the completed children's relocation entries. Assign each parent's
   own destination address while writing it. Rebuild a parent once for all
   moved children, without an incoming-edge index or re-minimization.
5. Install complete blocks into their reserved logical slots. Release source
   blocks after consuming their last live record. Finish the root level last.
6. Finalize the ordinary packed FST and release compaction metadata. Retain
   the current compactor as the explicit benchmark/control implementation.

## Pseudocode

```text
compact_recursive(graph):
    preserve domains and incrementally maintained logical visit count
    discard update indexes and memo caches
    if graph is empty: release graph; return empty PackedFst

    live, source_blocks = mark_live_DAG(root)
        # Live-start bitmaps, per-block live counts and live level membership.
    build rank indexes for live-start bitmaps
    release wholly dead source blocks

    output = backend supporting reserved logical slots
    moved = UNALLOCATED
    layouts = array[height]
    compact_level(root_level)
    finalize output
    release old storage and metadata
    return ordinary PackedFst(output, level_counts, logical_visit_count)

compact_level(level):
    layouts[level] = reserve_BFS_blocks(live_records_at(level))
        # One sizing walk; reserve descriptors, not individual state addresses.

    if level is not the leaf level:
        compact_level(level + 1)
    else:
        moved = allocate_dense_relocations(live.count, output.address_extent)
            # Final extent is known: choose u32/u64 entries once.

    writer = block_writer(layouts[level])
    for state in live_records_at(level), in the same order as reservation:
        word, children = read current source record
        address = writer.next_record_address(size(word, level))
        deltas = [checked_u32(moved[rank(child)] - address)
                  for child in children]
        writer.append_atomic_record(word, deltas)
        moved[rank(state)] = address

        # A buffered complete record no longer needs its source payload.
        if state is a RAM edit:
            release state payload
        else:
            decrement its source block's remaining live count
            if zero: release source block payload

    writer.finish_and_install_last_block()
```

The block writer installs full blocks as it goes, buffering one block rather
than a whole level. Every record receives its final forward deltas immediately;
completed compressed blocks need no later decompression/patching. Live-record
enumeration must skip released source blocks and freed edit slots; relocation
lookups must continue working after the corresponding child payload is freed.

## What changes

The previous compactor has marking, an output-sizing pass, a second pass assigning
all destination addresses, and a root-first copying pass. This proposal keeps
marking and sizing, but assigns addresses while building records on recursive
return. It removes the separate all-state address-assignment pass.

Block-level live level ranges also avoid filtering every sealed state at
every level when blocks are level aligned. Older DFS-sealed blocks may contain
many levels, so that optimization has a weaker effect on them. It must not add
a heap-allocated child vector or hash entry per state.

Each sealed block gets a temporary two-byte minimum/maximum live level range.
Uniform-level blocks need no per-record level-byte reads. Mixed-level blocks
fall back to the stored level byte and stop immediately if their final live
record causes retirement partway through enumeration. None of this directory
is retained in the returned packed graph.

It preserves global suffix sharing, graph semantics, current minimality and
SGF1 compatibility. It introduces no independent automaton partitions or
permanent incoming-reference index. The initial live marking remains necessary:
the lazy outgoing-signature memo is not a complete liveness index.

## Backend and memory work

`ReservedBlockStorage` adds compaction-builder operations to reserve logical block slots and
install a completed payload into a reserved slot. Existing `append_record`
alone cannot fill root-first logical slots in reverse order.

- Heap: reserve empty vector descriptors, then install finished byte vectors.
- File: reserve block/slot metadata, then write each finished output block once.
- Compressed: reserve empty descriptors, then compress each finished block once.

Slot reservation must allocate metadata only. File output is normal graph
storage, not a spill file; source file extents still live until the old backend
is dropped. Output levels get separate blocks, adding some descriptors and
unused file-slot tails; compressed size may change. The existing format handles
short blocks, but queries and snapshot validation must verify the new layout.

This prototype keeps the current dense 32/64-bit relocation array to isolate
traversal and backend changes. The stack holds O(height) layouts, not
every state's child list. This is not a one-block total-RAM claim: live bitmaps,
relocations, descriptors, old/new payload and backend caches remain. Mixed-level
source blocks can remain partially live while processing several levels.

Once correctness and speed are established, separately compare maps for only
the current parent and child levels. Every edge crosses exactly one level, so
deeper maps can be discarded. Dense/rank lookups should remain cheap: a hash
entry per state could exceed the current full table's RAM. Adjacent levels can
contain most states, so lower workspace is not guaranteed for every graph.

## Validation

The full all-features suite passes 134 library/integration tests and seven
README doctests. The repeated-batch test runs both compactors across all four
lookup strategies, both sealing orders and all three backends, and compares
full triples, minimal state counts, logical visits and SGF1 round trips. The
recursive output also runs through all five automaton schedules. Empty,
singleton, full-width domains and a shared DAG representing 2^48 triples are
covered, along with mixed-level source blocks that retire during enumeration.
Wide relocation entries have a synthetic test, not an end-to-end >4 GiB graph.
Clippy reports the same eight pre-existing library warnings.

## Comparison

The existing driver now includes `recursive` beside `blocks`, full rebuilding
and the two in-place sealing orders:

```sh
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all packed
cargo run --release --all-features --example fst_freeze_perf -- 65536 256 3 all packed
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all dfs
cargo run --release --all-features --example fst_freeze_perf -- 4096 32 3 all bfs
cargo run --release --all-features --example fst_freeze_perf -- 65536 256 3 heap dfs
cargo run --release --all-features --example fst_freeze_perf -- 65536 256 3 heap bfs
```

The measured binary uses the profiling profile (release optimization with debug
symbols), all features and 4096-byte blocks. Three-trial medians follow; method
order reverses on alternating trials, with fresh construction and the same
memoized batch updates for every method. Benchmarks ran sequentially after
tests/builds finished. These are unpinned local controls, not cold-cache or
isolated RSS measurements. `packed` starts with a normal packed base; `dfs`
and `bfs` construct the same relation entirely in an edited segment, seal it,
then apply the timed batch. The table times compaction only, excluding setup
and insertion. The larger fixture has 131,072 initial triples plus 256 edits;
the smaller has 8192 initial triples plus 32 edits.

### Mixed relation

| Initial storage | Vertices | Backend | Previous `blocks` | `recursive` | Speedup |
|---|---:|---|---:|---:|---:|
| Packed base | 65,536 | Heap | 10.92 ms | 10.02 ms | 1.09x |
| Packed base | 65,536 | File | 72.96 ms | 68.13 ms | 1.07x |
| Packed base | 65,536 | Compressed | 156.74 ms | 154.72 ms | 1.01x |
| DFS sealed | 65,536 | Heap | 22.97 ms | 12.90 ms | 1.78x |
| BFS sealed | 65,536 | Heap | 22.73 ms | 12.13 ms | 1.87x |
| BFS sealed | 4096 | Heap | 0.864 ms | 0.531 ms | 1.63x |
| BFS sealed | 4096 | File | 3.855 ms | 2.874 ms | 1.34x |
| BFS sealed | 4096 | Compressed | 6.331 ms | 5.410 ms | 1.17x |

The larger sealed-BFS heap fixture spends 5.19 -> 0.68 ms in planning and
14.69 -> 8.34 ms in copying/enumeration. This combines removal of address
preassignment with block-level filtering and reserved-block backend changes;
the experiment does not isolate traversal direction alone. All 72,873 live
states remain globally shared and all 571 source blocks are retired.

The packed compressed fixture is dominated by marking: 139.73 ms of the new
154.72 ms operation. File marking similarly takes 55.10 of 68.13 ms. Removing
one planning pass cannot fix that cost. Full rebuilding still wins these
packed paged cases: 51.10 ms on file and 48.61 ms on compressed in this run.
These phase timings are wall-clock instrumentation, not hardware-counter or
backend-read/decompression counts. Marking includes some additional bookkeeping
in the recursive path, and final backend trimming is outside the phase counters.

### Strong sharing and regressions

On the 65,536-vertex functional heap control, compaction takes 120.08 -> 76.35
microseconds (1.57x). The small functional file controls regress: from 108.98
to 164.80 microseconds on packed input, 138.45 to 192.60 on DFS-sealed input,
and 180.59 to 203.34 on BFS-sealed input. Block reservation boundaries and file
creation costs matter when physical automata have very few states. The small
functional compressed DFS control is essentially flat/slightly worse, 28.14
to 28.98 microseconds. All raw samples, including these controls, are retained.

### Memory and layout

The larger mixed BFS-sealed heap control has the following accounting:

| Counter | `blocks` | `recursive` |
|---|---:|---:|
| Input block contents | 2,284,654 B | 2,284,654 B |
| Peak old + output block contents | 2,286,381 B | 2,287,244 B |
| Temporary metadata | 606,620 B | 607,808 B |
| Relocation entries, included above | 291,492 B | 291,492 B |
| Output record contents | 1,399,652 B | 1,399,652 B |
| Retained output graph RAM | 1,408,328 B | 1,408,304 B |

The prototype keeps the full relocation map. Temporary metadata grows slightly
for the level layouts and sealed block ranges; it does not yet implement
adjacent-level maps. Heap output layout differences can save a descriptor while
compressed sizes may grow slightly. Peaks are not uniformly better: on packed
input the larger mixed fixture rises from 1,403,748 to 1,409,812 bytes of block
contents. On the 1024-byte sealed-block unit control, DFS input improves from
155,751 to 115,863 bytes, but the larger DFS benchmark slightly worsens from
3,316,638 to 3,326,026 bytes. Direction alone gives no strict overlap guarantee.

`workspace_bytes` excludes the current output block buffer, output descriptors,
backend caches, allocator overhead and recursive stack. Both builders buffer
at most one output block's records; descriptor reservation allocates no graph
payload. The content-peak counter includes the used bytes buffered for output,
but excludes unused capacity, live RAM edits and compression/RSS effects.

The recursive path is useful after periodic sealing, with modest gains on an
already packed base. It remains opt-in because the paged marking bottleneck,
small-file regressions and full relocation workspace still need attention.

[Raw measurements, all 420 samples, summaries and build manifest](prototype/results/fst-recursive-compact/)

[Implementation](../../src/fst_recursive_compact.rs) ·
[Benchmark driver](../fst_freeze_perf.rs)
