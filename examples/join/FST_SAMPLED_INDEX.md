# Sampled fingerprints and direct depth starts

The `Indexed` label in historical results refers to the former variant. The
current default is `Sampled(1)`; the redundant `Indexed` API was removed.

This records the initial prototype and its historical measurements. The current
`Sampled(rate)` representation uses narrow relative addresses and 32-bit sampled
fingerprints. They are now also used by default Indexed at rate 1. Scan-based
updates retain depth starts; the exact word summaries were removed. See
[the subsequent per-level metadata experiment](FST_LEVEL_METADATA.md) for the
historical experiment; [FST_COMPACT_INDEX.md](FST_COMPACT_INDEX.md) describes the
current defaults. The construction API is unchanged.

`UpdateLookup::Indexed` remains the default. The sampled prototype is selected
when opening an editable graph; the interval is any nonzero `usize`:

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Sampled(8));
graph.prepare_updates(); // optional: prepare before accepting edits
assert!(graph.insert(9, 2, 13).unwrap());
assert!(graph.remove(9, 2, 13).unwrap());
let packed = graph.freeze().unwrap();
assert!(packed.contains(0, 0, 1));
```

The same selection works with file/compressed storage and when reopening a
`FrozenFst`. A zero interval panics at construction. Rate 1 retains every
fingerprint; rates larger than a depth's population retain its first fingerprint.
Snapshots omit lookup metadata; loading chooses the ordinary Indexed default.
Mutable `compact()` preserves the selected interval. Consuming freeze releases
the sampled index and uses the default rank-free compactor.

## Sampled lookup

The FST's BFS physical order is unrelated to fingerprint order. Sampling whole
index entries would leave no way to locate an unsampled matching state. This
prototype retains every address, in fingerprint order, and samples fingerprints:

```text
prepare(rate):
    for each physical base/sealed state:
        append (fingerprint(signature), address) to its depth's temporary index
    for each depth:
        sort pairs by (fingerprint, address)
        retain fingerprints at positions 0, rate, 2*rate, ...
        retain every address in that sorted order
        release temporary pairs and excess address capacity

find(depth, signature):
    h = fingerprint(signature)
    first = first sampled fingerprint >= h
    past = first sampled fingerprint > h
    begin = just after the sample preceding first, or zero
    end = sample position past, or end of address array
    for address in addresses[begin:end]:
        if stored signature at address equals signature: return address
    return absent
```

Both binary-search boundaries are necessary. Equal fingerprints may span sample
boundaries; checking only one interval could miss a matching state. Exact word
and ordered child-identity comparison resolves collisions and preserves live
minimality. Ordinary lookups examine at most one interval when there is no equal
sample, and up to two when the queried fingerprint equals a sample; collision
runs may require more. Candidate addresses are in hash order, so their physical
reads can be scattered.

Ignoring per-depth rounding and vector metadata, resident index bytes per state
are `8 + 8/rate`, versus 16 for Indexed. Rate 8 approaches a 43.75% reduction;
rate 16 approaches 46.875%. The separate address and sample vectors add 48 bytes
of vector metadata per depth instead of Indexed's 24. Empty levels/rounding are
included in `memory_usage().base_register_bytes` and `mem_size_bytes()`.

Preparation still scans physical states, hashes child lists and sorts complete
pairs, before converting to the retained representation. This prototype reduces
resident memory, not the construction workspace bound. Preparation remains
lazy on the first effective edit; `prepare_updates()` moves the work before the
first edit. Sampling does not remove this one-time construction latency.

## Depth starting addresses

Packed scan strategies (`Memoized`, `ScanBase`, `ScanAll`) now retain one starting
cursor per FST depth in a lazily populated `OnceLock<Vec<u64>>`. The first scan
walks non-leaf physical records once to discover the starts; no leaf decoding
is needed to locate a later depth. Subsequent scans jump directly to the target
level and inspect only that level's known number of physical records.

This is 8 bytes per depth on the measured machine, counted in packed graph
memory. It is not the diagnostic per-state address directory. Immutable searches,
Indexed and Sampled do not populate it. Serialization omits it; rebuilt/loaded
packed graphs start without the cache. Starts remain valid because local edits
never change packed-base bytes. Sealed segments retain their existing scan
path, so the measured improvement applies to packed-base lookup.

A miss still scans the requested depth. Negative answers and unfamiliar large
levels remain expensive; this change removes repeated parsing of earlier
depths, without adding a filter or a full suffix index.

## Measurement protocol

The extended [benchmark](../fst_lazy_perf.rs) covers Indexed, sampled intervals
1/4/8/16/32, ScanAll and Memoized. Each invocation uses two fresh-graph passes
with reversed strategy order and three edit/undo repetitions. First insertions
include index preparation or first scan-cache construction. Warm insertions use
trials 1 and 2; fresh-region insertions have unfamiliar subjects/signatures.
Separate preparation probes time `prepare_updates()` on fresh Indexed/Sampled
graphs, outside edit timing. Every phase verifies change counts and membership;
inverse batches restore base identity with zero edited states. Complete relation
checks and snapshot compaction occur outside edit timing.

The large heap fixture has 131,072 mixed triples / 72,726 states and 64 edits per
batch. Three current-binary invocations were recorded. Two clean pre-change
binary invocations provide the depth-start baseline; the initial pre-change run
overlapped compilation and is excluded from timing comparisons. The small
fixture has 8,192 triples and 32 edits, on heap/file/compressed storage. Both
mixed and highly shared functional shapes are included. Wall times are
sequential and unpinned; these are small synthetic workloads, not RDF datasets.
Retained memory counts capacities and estimated hash-table overhead; it is not
process RSS or peak preparation memory.

Raw phases, binary/source hashes and commands are in
[the recorded results](prototype/results/fst-sampled/).

## Results

Medians over six fresh-graph passes (three current-binary invocations). Large
mixed heap fixture; edit times are microseconds per changed triple, preparation
is milliseconds for the complete index. Cold batch times include preparation
amortized over 64 edits, not an individual insertion latency percentile.

| Strategy | Index bytes | Preparation, ms | First insert, µs/edit | Warm insert, µs/edit | Fresh-region insert, µs/edit |
|---|---:|---:|---:|---:|---:|
| Indexed | 1,163,760 | 4.690 | 74.29 | 1.48 | 1.62 |
| Sampled(1) | 1,163,904 | 4.697 | 75.07 | 1.58 | 1.71 |
| Sampled(4) | 727,568 | 4.703 | 74.67 | 1.57 | 1.77 |
| Sampled(8) | 654,840 | 4.649 | 74.49 | 1.52 | 1.76 |
| Sampled(16) | 618,480 | 4.650 | 74.18 | 1.62 | 1.85 |
| Sampled(32) | 600,296 | 4.651 | 74.92 | 1.74 | 2.22 |

Rate 8 saves **43.7%** of resident base-index memory here, with approximately
2.5% higher warm-insertion and 8.3% higher fresh-insertion time. Sampling does
not make preparation disappear or proportionally accelerate it. The construction
still hashes/sorts all states. These small unpinned measurements do not establish
that slight preparation-time differences are meaningful.

Depth starts, comparing four clean pre-change passes with six current passes:

| Strategy/phase | Before, µs/edit | After, µs/edit | Speedup |
|---|---:|---:|---:|
| Memoized first insert | 1,171.62 | 454.03 | 2.58x |
| Memoized fresh-region insert | 952.47 | 407.29 | 2.34x |
| Memoized warm insert | 1.31 | 1.29 | approximately unchanged |
| ScanAll first insert | 1,608.77 | 453.30 | 3.55x |
| ScanAll warm insert | 1,618.11 | 442.54 | 3.66x |

Memoized misses still cost hundreds of microseconds on this graph. Cached
repetitions avoid scans, while unfamiliar signatures must search their depth.
The depth-start cache in this six-level fixture retains **48 bytes**.

Small mixed fixture, two fresh-graph passes with four warm-insert samples each
backend/strategy (two warm trials per pass):

| Backend | Indexed warm insert, µs/edit | Sampled(8), µs/edit | Sampled(16), µs/edit |
|---|---:|---:|---:|
| Heap | 1.26 | 1.17 | 1.19 |
| File | 2.49 | 7.40 | 13.22 |
| Compressed | 5.37 | 19.04 | 35.24 |

The index shrinks from 40,872 bytes to 23,184 at rate 8, or 21,912 at rate 16,
on all three backends. Extra scattered candidate reads make file/compressed
updates substantially slower. The slight heap advantage in this small case
should not be generalized; the larger heap fixture shows the expected modest
latency cost. Indexed remains the default. Sampled(8) is a promising explicit
heap tradeoff; its suitability for paged/compressed workloads needs different
candidate-access optimizations or acceptance of higher latency.

On the functional control, full suffix sharing makes the entire index only
320 bytes; Sampled(8) occupies 424 bytes because depth-vector metadata dominates.
Sampling is intended for a large physical state population, not tiny FSTs.

## Validation

Nineteen mutable-FST unit tests and thirteen snapshot/update integration tests
pass. Coverage includes rates 1/4/8/16/32, rates exceeding depth populations,
zero-rate rejection, hash collision runs crossing sample boundaries, retained
address capacity, empty graphs, exact depth scans across blocks, live minimality,
recycled edit slots, repeated sealed batches, all storage modes and consuming
freeze. The seven README doctests also pass; all targets compile with all
features. Raw phases and the aggregate [summary](prototype/results/fst-sampled/summary.json)
are retained with the build manifest.
