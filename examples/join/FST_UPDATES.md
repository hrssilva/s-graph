# Local FST updates and three storage backends

The packed graph is now `PackedFst<S: BlockStorage>`. The existing `Fst` and
`PathFstDeltaPow2Bfs` names still select heap storage, preserving constructor
inference and existing code. `FileFst` and `CompressedFst` select the other two
block backends. All use the same forward-delta layout, search compiler,
specialized traversal, scans, paths, analytics and SGF1 format.

`into_mutable()` moves a packed graph into `EditableFst<S>`. The concrete names
are `MutableFst`, `MutableFileFst`, and `MutableCompressedFst`. Edits affect the
FST language directly; queries see edits immediately, without unioning answer
relations or materializing triples. `Heap`, `File`, and `Compressed` retain their
existing mutable radix-tree representation and now also accept `search` through
`TreeSource`. Their legacy `join_run` still selects `NativeJoin`.

## Literature and the minimality invariant

[Daciuk, Mihov, Watson and Watson (2000)](https://aclanthology.org/J00-1002/)
describe incremental minimal acyclic automata, including unsorted insertion.
Shared states require cloning before modifying their suffix language. A register
identifies equivalent states bottom-up using finality and canonical outgoing
transitions. Our fixed-length triple encoding has implicit final transitions,
so a level's occupancy word and ordered canonical child identities determine
its right language. Registers are separated by storage level.

[Carrasco and Forcada (2002)](https://aclanthology.org/J02-2004/) cover both
insertion and removal, relating updates to composition with a single-string
automaton. This supports one symmetric implementation here: change the selected
terminal transition, remove empty states, and intern the changed path bottom-up.
The implementation specializes to an acyclic graph and does not construct a
general product or compute equivalence over cyclic states.

[Daciuk's 2004 discussion](https://aclanthology.org/J04-2005.pdf) also separates
online maintenance from a compact read-mostly representation and explains the
register's role. We retain the packed read kernel and keep maintenance metadata
in a separate editable form. This is an implementation choice informed by those
papers, not a claim to implement their cyclic or transducer algorithms.

The first edit prototype interned only edited states. That preserved answers
and shared old suffixes, but could duplicate a packed state's right language.
The completed implementation compares against **both** packed and edited
states. Reversing an edit can therefore restore the original packed root and
reclaim all copied states immediately.

The reachable machine is minimal within this fixed-depth slot encoding after
each successful update. Unreachable packed bytes may remain allocated until
compaction; they are not reachable automaton states. This distinction matters:
semantic minimization does not mean rewriting or reclaiming the entire packed
file on every edit.

## Algorithm

```text
update(triple, insert_or_remove):
    validate insertion domains
    walk its encoded slots from the current root
    record (cursor, original_word, selected_slot) per level
    if duplicate insertion or absent removal: return unchanged

    lazily prepare packed-state register if this is the first effective edit
    child = absent
    for each recorded level, from leaf to root:
        copy this node's ordered outgoing children
        at the leaf: add/remove the selected terminal bit
        otherwise: replace/add/remove the selected edge using child
        child = canonical_state(level, changed_word, changed_children)

    acquire a root reference to child
    publish child as the new root
    release the old root and reclaim unreferenced edit states recursively

canonical_state(level, word, children):
    if word is empty: return absent
    if every child is a packed cursor:
        find matching fingerprint in packed register for this level
        compare the complete word and ordered child identities
        if equal: return that packed cursor
    if exact signature exists in edited-state register: return its cursor
    allocate/reuse an edit slot; acquire references to its edited children
    register the exact signature; return its cursor
```

The old root remains live throughout construction. No shared state is changed
in place. Acquiring the new root before releasing the old one protects reused
children. Reference counts include edge multiplicity: two labels can target
the same state. Register entries are removed before slots are recycled, so no
live signature can refer to a reused identity.

For fixed-length acyclic words, bottom-up interning gives an inductive
minimality argument. Leaves merge by identical accepting-transition words.
At each preceding level, already canonical children make equal right languages
equivalent to equal words and child lists. Both stores participate in lookup;
no duplicate live equivalence class survives. Empty branches are removed.
This relies on starting from a minimal packed graph. Builders and compaction
produce one; snapshot validation now rejects duplicate equivalent states,
in addition to checking links, domains, counts and reachability.

## Work and memory

`into_mutable()` moves storage and allocates only O(height) register headers.
Duplicate insertions and absent removals avoid both state allocation and index
preparation. An effective first edit scans the packed **physical** records and
builds a sorted `(fingerprint, cursor)` index per level. `prepare_updates()` lets
callers pay this cost explicitly before a burst. The index retains 16 bytes per
packed state plus vector metadata. Fingerprints filter candidates; full
structural comparison resolves collisions. A forced-collision test covers this.

After preparation, a single edit walks at most 64 levels, copies at most 64
outgoing transitions per changed level, performs binary searches in relevant
packed level indexes and expected constant-time edited-register lookups. It
does not scan the graph or its triples. First preparation takes O(E + sum of
N_level log N_level) work; ordinary updates take O(64H + H log N) expected work,
including signature hashing and outgoing reference maintenance, apart from
pathological hash collisions. Batch updates validate all insertion coordinates
first, sort the input in place, and apply individual edits; they do not yet
share one copied path across a whole batch.

The edit arena and packed-state index are in RAM for **all three modes**. This
is an explicit memory cost of online minimality. File mode keeps base node
records on disk, but does not promise near-zero RAM during large write bursts.
Call `compact()` to put edits into the chosen backend and discard maintenance
metadata. No automatic threshold or background compaction is enabled.

`compact`, `freeze` and editable serialization traverse reachable physical
states, canonicalize bottom-up and pack BFS records. They need O(N + E)
temporary memory and expected work, not a list of root-to-leaf words. A test
edits, compacts and reloads a DAG representing 2^48 triples. Packed addresses
are calculated first, then each complete record is appended once: there is no
per-child file write or compressed-chunk recompression during backfilling.

Read-only heap FST traversal has no edit tag branch or maintenance directory.
Editable cursors use a tag to select packed or RAM states. Compiled searches
borrow their source, preventing mutation while they remain in use; reuse the
query expression and compile a new search after edits.

## Persistence and API boundary

All three storage modes read and write SGF1 v1. A snapshot contains the compact
current language, not edit history, registers or arena addresses. Editable
`serialize` returns `Result<Vec<u8>>` because packing can fail on representational
limits; stream/path variants are also available, as are all limited loaders.
`freeze()` consumes the graph using the default rank-free BFS compactor: it
reuses edit-arena metadata and retires source blocks incrementally. Updates
already preserve minimality, so freezing only packs live states and rewrites
addresses. All three built-in storage backends support this operation through
`ReclaimingBlockStorage`. `compact_streaming()` uses exactly the same algorithm
and also returns diagnostics; see [the algorithm and measurements](FST_RANK_FREE_COMPACTION.md).

`to_fst::<OtherStorage>()` allows an explicit backend change without materializing
triples. This borrowed snapshot path, serialization and mutable `compact()`
retain the existing rebuilding implementation. They preserve the source until
packing succeeds; `compact()` then replaces the editable graph.

File graphs use temporary backing files. `serialize_to_path` makes a persistent
snapshot; loading a file-backed graph copies validated blocks into its live
temporary file. This is not an mmap or crash-recovery API. The existing
`BlockStorage` hot read interface remains infallible; real backend I/O failures
in that interface retain its existing panic behavior. Snapshot I/O returns
`TreeError::Io`. No transaction or durability guarantee was added.

See the [README](../../README.md#updates-and-storage-modes) for executable usage.
`file` and `compressed` now include the shared `heap` API feature, so explicit
automata work with `--no-default-features --features file` or `compressed` too.
This enables types and executors; it does not load file node records into a
heap graph. Raw slot masks depend on each representation's bit layout; BGP
coordinate constraints work across both FST and mutable-tree sources.

## Measurements and validation

The [update benchmark](../fst_updates_perf.rs) compares prepared edit batches
with the existing mutable tree on functional and mixed graphs, in each mode.
It reports preparation separately, prepared packed/edited visits, compaction,
state counts and retained RAM estimates. Builds and query preparation are
outside visit timing. Seven samples are taken, with initial update preparation
outside the update samples. Timings are local microbenchmarks, not a claim
that edits are universally faster than the mutable radix tree. HashMap bucket
overhead in editable memory accounting is approximate.

```sh
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" cargo build --offline --all-features --profile profiling --example fst_updates_perf
TMPDIR="$PWD/target/automata-tmp" target/profiling/examples/fst_updates_perf 65536 1024 7
cargo test --offline --all-features
```

On the mixed fixture (65,536 vertices, 131,072 initial triples, 1,024 edits),
prepared update medians are **microseconds per changed triple**:

| Backend | FST insert | Tree insert | FST remove | Tree remove | FST preparation, ms | FST compaction, ms |
|---|---:|---:|---:|---:|---:|---:|
| heap | 1.63 | 0.54 | 1.53 | 0.32 | 4.50 | 39.70 |
| file | 3.72 | 31.32 | 4.37 | 39.31 | 6.59 | 50.48 |
| compressed | 6.52 | 32.04 | 9.47 | 30.11 | 6.83 | 48.81 |

The heap radix tree remains faster for these updates. File/compressed FST edits
benefit from keeping the small changed frontier in RAM. The cost is explicit:
the 72,726-state base needs about 1.16 MB of register pairs in every mode,
before edit-arena and cache memory. The 1,024 added triples produce 488 edited
states; compacting returns their bytes to the chosen backend.

Prepared predicate-0 searches return 65,536 rows before and after edits (which
use predicate 3). On the mixed fixture, the scoped automaton read cache changes
packed file traversal from about 168 ms to 125 ms, and compressed traversal
from about 373 ms to 308 ms. Heap is about 2.43 ms. The remaining file/decompression
latency is substantial; storage-mode support does not imply heap-like read
performance. Cache scopes retain at most 65 path entries, restore the previous
policy on nested return/cancellation, and leave heap storage with a no-op scope.
Chunk buffers can retain their allocated capacity after a search.

These are sequential, unpinned local runs with seven fixed-size samples, not
paired randomized trials or calibrated steady-state throughput. Preparation
and compaction are single samples. Old-tree initial loading uses `insert_batch`
and is not presented as a comparable bulk-build timing. Queries are compiled
outside timed visits. The benchmark validates changed counts, inverse edits,
query row counts and compacted counts; integration tests check full bindings.
The scoped-cache control predates final iterator-style cleanup; use the final
run for current numbers rather than pooling samples across versions.

[Final raw medians](prototype/results/fst-updates-final/summary.json),
[pre-cache control](prototype/results/fst-updates-before-cache/summary.json),
and [source/result manifest](prototype/results/fst-updates-manifest.json).

Validation: **126 tests and seven README doctests passed** with all features.
File-only and compressed-only feature configurations each passed all four
applicable update/source integration tests; a no-features check also passed.
After the final iterator-style cleanup, the five integration tests and three
mutable-FST unit tests passed again. A subsequent integration extension checks
far-child spans with full-width IDs and a non-default dimension order.
Rustdoc builds successfully; Clippy succeeds with eight pre-existing style
warnings and no new warnings in this implementation.


The [memory lifecycle experiment](FST_MEMORY.md) breaks out maintenance metadata
across insertion, undo, deletion, reuse and compaction on all three modes.

The experimental [in-place freeze](FST_FREEZE_IN_PLACE.md) seals only live edit
states into blocked storage, retaining the packed base and releasing update
tables. It separates becoming immutable from the full compaction described here.

The explicit `UpdateLookup::Sampled(rate)` prototype retains every packed-state
address and every `rate`th sorted fingerprint. See [sampling and depth-start
measurements](FST_SAMPLED_INDEX.md) for usage, memory/latency tradeoffs and the
updated scan-based lookup behavior. The default remains Indexed.
