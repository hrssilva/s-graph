# Narrow sampled indexes and exact per-level summaries

The `Indexed` label in historical results refers to the former variant. The
current default is `Sampled(1)`; the redundant `Indexed` API was removed.

Historical experiment: narrow addresses and 32-bit prefixes have now been promoted
to both default Indexed and Sampled. The occupancy/degree summaries and rejection
counter were removed. Scan modes retain the 8-byte-per-level start-only cache.
See [FST_COMPACT_INDEX.md](FST_COMPACT_INDEX.md) for current behavior. The algorithm
and measurements below describe the prototype before that promotion.

This prototype keeps the existing API and the Indexed default:

```rust
use s_graph::{Fst, mutable_fst::UpdateLookup};
let packed = Fst::build(vec![(0, 0, 1)], [1024, 4, 1024], 4096);
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::Sampled(8));
graph.prepare_updates(); // optional: otherwise first effective edit prepares it
assert!(graph.insert(9, 2, 13).unwrap());
let packed = graph.freeze().unwrap();
assert!(packed.contains(9, 2, 13));
```

## Compact sampled representation

Preparation still sorts complete `(64-bit fingerprint, 64-bit address)` pairs
per level. Only the retained representation changes:

1. Keep the upper 32 bits of every sampled fingerprint. Truncating a sorted hash
   is monotone, so both binary-search boundaries remain valid. Prefix collisions
   enlarge candidate intervals; full word/child comparisons retain exactness.
2. Find the minimum/maximum physical cursor in each level. Retain the minimum
   once, and store every cursor relative to it in `u16`, `u32` or `u64`, selecting
   the smallest width that represents the entire span exactly.
3. Dispatch on the selected offset width once per candidate search. Decode each
   candidate by adding the level origin. Release excess collection capacity,
   including capacity reused from the temporary 16-byte pair allocation.

This keeps every address; it does not reconstruct states by walking from sampled
addresses. No extra FST reads are needed to decode a narrow address. Ignoring
level metadata and rounding, cost is `address_width + 4/rate` bytes per state.
A 32-bit-address level at rate 8 therefore uses 4.5 bytes/state versus the previous
9. Tagged sealed and packed cursors can have a large combined span; such mixed
levels correctly fall back to 64-bit offsets. Separate segment origins could
improve that case in a future experiment.

Preparation still needs complete temporary pairs and sorting. This is a resident
memory improvement, not a bounded-workspace builder. Index bytes include vector
capacities, origins and representation metadata in `base_register_bytes`.

## Exact summaries for scan-based lookup

The existing lazy depth-start cache now builds starting addresses and word
summaries together, in one pass over physical record headers. Unlike the earlier
start-only cache, the final leaf level must also be read to summarize its words.
No child transitions are decoded for these summaries. Each level retains:

- its starting cursor;
- the union of all occupancy words (which bits can occur);
- the intersection of all words (which bits every state requires);
- a 65-bit set of observed occupancy populations, held in a `u128`.

On this target the combined record occupies 48 bytes per level, including
alignment. There is no per-state or per-transition summary allocation. Immutable
queries and Indexed/Sampled updates do not construct this lazy cache. Allocation
capacity is included in packed-base memory. Snapshots omit it.

```text
find_base(level, signature):
    if memo contains signature: return cached answer
    base_possible = base_summary[level].permits(signature.word)
    sealed_possible = sealed_summary[level].permits(signature.word)
    if neither is possible:
        count summary rejection
        return absent  // do not retain a cheaply proven absence in the memo
    scan only source segments whose summary permits the word
    compare complete signatures to determine exact presence/absence
    memoize that answer when using Memoized
```

These conditions are necessary, not sufficient. A passing word can still be
absent, or belong to states with different children. Exact comparison is always
required after passing a summary. All physical records, including obsolete ones,
contribute: dead records can weaken pruning but cannot cause a false negative.
Empty depths reject every word. The full 64-bit occupancy population is supported.

Sealed storage keeps its own per-level summaries (32 bytes per level here).
Appending a sealed record invalidates that cache, so reopening/editing/sealing
multiple batches cannot reuse an incomplete summary. Sealed lookup still scans
its existing record layout; it does not gain direct depth starts in this pass.
`update_lookup_stats().summary_rejections` counts uncached signatures rejected
without a source scan. Surviving scans remain counted by `base_scans`.

## Protocol

The [runner](prototype/fst_level_metadata.py) compares a saved pre-change binary
with the new binary, using the same extended [update benchmark](../fst_lazy_perf.rs).
It tests Indexed, Sampled(1/4/8/16/32), ScanAll and Memoized on mixed and functional
shapes. Small fixtures have 8,192 triples/32 edits on all three backends; medium
heap fixtures have 131,072 triples/64 edits and three independent invocations;
large heap fixtures have 1,048,576 triples/64 edits and one invocation. Every
invocation uses two reversed fresh-graph passes and three insertion/undo trials.

Cold insertion timing includes complete index preparation or first summary-cache
construction, amortized over the batch. Warm medians use trials 1 and 2. Separate
preparation probes do not change the timed graph. Each phase asserts change counts,
membership and memory-accounting consistency; inverse edits restore base identity.
Complete relation validation, snapshots and setup are outside operation timing.

Runs are sequential, unpinned wall time without concurrent builds/tests. Large
cases use the existing systemd runner with verified RLIMIT_AS, at most 1 GiB and
half available RAM (also leaving its required reserve). Its CPUQuota is 100% for
both binaries. Cgroup memory-controller availability is recorded rather than
assumed. Retained allocation estimates are not process RSS or construction peaks.

[Raw phases, commands, containment records and source/binary hashes](prototype/results/fst-level-metadata/)

## Results

The medium mixed fixture has 131,072 triples and 72,726 physical states. Medians
combine six fresh-graph passes (three invocations), with four warm insertion
samples per invocation. Times below are microseconds per changed triple.

| Strategy | Index bytes before → after | Warm insertion before → after | Fresh insertion before → after |
|---|---:|---:|---:|
| Indexed control | 1,163,760 → 1,163,760 | 1.42 → 1.41 | 1.49 → 1.52 |
| Sampled(1) | 1,163,904 → 578,032 | 1.47 → 1.47 | 1.62 → 1.59 |
| Sampled(4) | 727,568 → 359,864 | 1.45 → 1.45 | 1.63 → 1.68 |
| Sampled(8) | 654,840 → 323,500 | 1.46 → 1.49 | 1.72 → 1.74 |
| Sampled(16) | 618,480 → 305,320 | 1.52 → 1.55 | 1.91 → 1.93 |
| Sampled(32) | 600,296 → 296,228 | 1.64 → 1.70 | 2.04 → 2.19 |

Sampled(8) saves **50.6%** versus the preceding sampled representation and
**72.2%** versus the full Indexed register. Preparation remains essentially
unchanged: 4.663 → 4.671 ms for Sampled(8), versus Indexed's 4.637 → 4.609 ms.
These estimates describe retained lookup allocations, not total graph RAM.

The large fixture contains **1,048,576 triples**. Two reversed passes, one
invocation per binary, under verified 1 GiB hard address-space limits:

| Strategy | Index bytes before → after | Warm insertion, µs/edit before → after | Fresh insertion before → after |
|---|---:|---:|---:|
| Indexed control | 4,172,184 → 4,172,184 | 1.69 → 1.80 | 2.21 → 2.20 |
| Sampled(4) | 2,607,856 → 1,300,038 | 1.75 → 1.80 | 2.49 → 2.24 |
| Sampled(8) | 2,347,112 → 1,169,666 | 1.76 → 1.76 | 2.39 → 2.28 |
| Sampled(16) | 2,216,736 → 1,104,478 | 1.96 → 1.90 | 2.96 → 2.72 |
| Sampled(32) | 2,151,552 → 1,071,886 | 2.02 → 2.21 | 2.97 → 3.12 |

Sampled(8) saves **50.2%** versus the earlier sampled representation and
**72.0%** versus Indexed. Preparation is 19.91 → 20.11 ms. Small latency changes,
including the Indexed control's variation, should not be interpreted as decisive
speedups in this single-invocation larger test. The memory reductions are exact
retained-capacity comparisons.

### Summary selectivity

| Mixed heap phase | Before, µs/edit | After, µs/edit |
|---|---:|---:|
| 131K Memoized cold insertion | 453.45 | 448.91 |
| 131K Memoized fresh-region insertion | 405.43 | 394.28 |
| 131K Memoized warm insertion | 1.25 | 1.29 |
| 1M Memoized cold insertion | 1,410.04 | 1,437.60 |
| 1M Memoized fresh-region insertion | 1,411.66 | 1,426.45 |

On the medium case, cold memoized insertion scans drop from 161 to 144 per
64-edit batch, with 17 summary rejections. Metadata grows from 48 to **288 bytes**
for six packed levels. The larger mixed graph records **zero** summary rejections;
its broad occupancy/degree summaries permit every attempted lookup. Scans remain
207 per cold 64-edit batch, and latency stays around 1.4 ms/edit. The initial
summary pass reads leaf words that the older start-only cache could skip.

The functional control rejects 64 cold signatures per batch, but its physical
FST was already tiny; cold memoized insertion changes only from 1.50 to 1.42
µs/edit. These summaries can be useful exact shortcuts, but do not replace an
index within large diverse levels. This pass establishes no general latency
improvement from summaries on the larger mixed workload.

### File/compressed behavior

Small mixed fixture, 8,192 triples, Sampled(8):

| Backend | Index bytes before → after | Warm insertion before → after, µs/edit |
|---|---:|---:|
| Heap | 23,184 → 6,698 | 1.15 → 1.17 |
| File | 23,184 → 6,698 | 7.19 → 7.17 |
| Compressed | 23,184 → 6,698 | 19.50 → 18.98 |

These levels fit 16-bit relative offsets, so the index shrinks by **71.1%**.
Extra candidate reads remain: full Indexed warm insertion is 1.12 / 2.36 / 5.30
µs/edit respectively in this run. Narrow address decoding does not solve the
paged/compressed penalty caused by sampling fingerprints. The full Indexed
strategy remains the default.

On the tiny functional FST, Sampled(8) grows from 424 to 430 bytes because the
new per-level encoding metadata outweighs entry savings. Indexed remains only
320 bytes there. This optimization targets levels with many states.

## Validation and reproduction

115 library tests and 13 snapshot/update integration tests pass. Added tests
cover 16/32-bit width boundaries, wide/tagged cursor fallback, high origins,
retained offset capacity, distinct hashes with equal 32-bit prefixes, full
64-bit occupancy population, exact-summary false positives, memoized negative
answers that summaries cannot reject, and sealed-cache invalidation. Existing
checks cover minimality, inverse edits, snapshots and all backends. All-feature
all-target compilation and seven README doctests also pass.

```sh
CARGO_TARGET_DIR=target/automata-prototype cargo rustc --offline --all-features \
  --profile profiling --jobs 1 --example fst_lazy_perf -- \
  -C link-arg=-Wl,--threads=1
PYTHONDONTWRITEBYTECODE=1 python3 examples/join/prototype/fst_level_metadata.py \
  --before /path/to/saved/pre-change/fst_lazy_perf
```

The pre-change binary must be saved before rebuilding. The manifest records
both binary hashes and the current source hashes. `--large-only` resumes the
contained large cases after small/medium cases; `--skip-large` omits them.
The larger runs completed with systemd success and verified RLIMIT_AS; cgroup
MemoryPeak is unavailable in this environment and is not reported as zero.
[Aggregate results](prototype/results/fst-level-metadata/summary.json).
