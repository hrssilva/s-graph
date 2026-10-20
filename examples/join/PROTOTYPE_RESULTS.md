# Compiled automata and lifted joins: measured prototypes

**Follow-up:** [On-demand automaton product](ON_DEMAND_JOIN.md) removes eager
relation conversion by retaining a factored product during traversal. It includes
updated comparisons with the executors below; this report describes the original
eager prototypes.

The [composition proposal](AUTOMATON_COMPOSITION.md) is implemented experimentally
in [`prototype/`](prototype/). Compiling complex filters is useful with a
preparation-cost threshold. Lifted joins demonstrate structural reuse and fast
prepared execution, but their preparation is too costly for a general replacement.

Production executors and the FST layout remain unchanged. The preparation script
copies current sources and tests to a temporary crate, then injects the prototypes.
Existing modules receive equality/hash derivations for compile-time interning and
allocation-reporting methods; their baseline traversal behavior is unchanged.

## Implementations

[`compiled_filter.rs`](prototype/compiled_filter.rs) clips, sorts, and coalesces
interval unions; minimizes one automaton per dimension; then composes/minimizes
them into a physical-slot automaton. Visits use integer IDs, masks, and dense
transitions, with DEAD/ALL suffix states and final-transition acceptance.
Exact/Free retains the mask-array kernel. A monolithic compiler and prepared
wrappers around the legacy restriction objects are included as controls.

Minimizing components first matters. For uniform three-Multi restrictions, the
initial monolithic compiler built 13,129 intermediate states and took about
85 ms. Component-first compilation creates 939 component/product states and
retains the same 714-state machine in **0.85 ms**. Construction tables are then
discarded. This compiles Exact/Free/Range/Multi, not arbitrary regex/fuzzy syntax.

[`lifted_join.rs`](prototype/lifted_join.rs) converts shared physical FST states
directly into reduced ordered decision DAGs over common variable bits. Per-atom
memoization reuses `(data address, level)` conversions; slot constraints are
combined with child relations and equivalent decision nodes are shared. It
handles constants, repeated variables, asymmetric domains, and non-power-of-two
bounds without enumerating per-atom tuple tables.

Two modes use that adapter:

- **lifted-lazy:** prepare atom DAGs, then traverse their product during visits.
- **lifted-merged:** prepare atom DAGs and their canonical Boolean intersection,
  then enumerate or symbolically count the result.

Both eagerly prepare lifted atom views. The lazy mode does not yet convert the
source FST on demand. The merged mode computes a **compressed answer during
preparation**: its visit is enumeration, not fresh join evaluation. Empty answers
become a false terminal. Preparation and total time must accompany visit times.

## Method and validation

Rust 1.98.1, `profiling` profile, default CPU target, single-threaded, Ryzen 5 PRO
4650U. Graph building is excluded. Timings use a warmup and five repetitions,
with adaptive batches for fast operations. Construction/release is included in
preparation and end-to-end closures. Prepared visits reuse plans and bindings.
Benchmarks run sequentially; these are synthetic warm-process measurements.

There are 21 filter cases across uniform, asymmetric-domain, and Zipfian data,
plus eight join cases. Complete triples are checked against source data and
legacy traversal; projected collectors are checked against the current public
API. Full join bindings are checked against FstJoin, with independent expected
answers for skew, dense-star, and constructed empty-cycle fixtures.

**78 tests pass:** 70 existing tests and eight new prototype tests. New coverage
includes explicit enumeration, 48 randomized queries and reversed atom orders,
both assignment encodings, lazy/merged modes, wide/asymmetric IDs, non-power-of-two
domains, repeated variables, shared predicates, disconnected patterns, counting,
cancellation/reuse, overlapping/empty filters, and construction-budget errors.

Raw data: [`comparison.csv`](prototype/results/comparison.csv), with construction
counts in [`comparison.log`](prototype/results/comparison.log). The CSV combines
the filter run with the final integer-key-hash join run. The initial standard-hash
run is retained as [`comparison-std.csv`](prototype/results/comparison-std.csv).

Diagnostics have different units: lifted source reads/reuses occur during
preparation, while FstJoin/RecursiveJoin reads occur during visits. `work` means
compiled intermediate states for filters, accepted candidates for the existing
prefix joins, and runtime decision steps for lifted joins. Zero diagnostics for
uninstrumented engines mean not collected. These are not interchangeable counters.

## Filter results

Both columns below include construction, visitation, sorting, and deduplication
of the same object list. The baseline is the actual public
`restricted_descent_via_composed_automaton` API. Units are microseconds.

| Case | Current API | Compiled | Current / compiled |
|---|---:|---:|---:|
| Uniform Exact/Free | 0.95 | 1.09 | 0.87× |
| Uniform subject Range + predicate Exact | 120.63 | **76.49** | 1.58× |
| Uniform three-Range | 12,539.05 | **6,229.17** | 2.01× |
| Uniform sparse Multi | **102.34** | 132.18 | 0.77× |
| Uniform overlapping Multi | 3,337.42 | **455.37** | 7.33× |
| Uniform three-Multi | 1,103.67 | **1,021.51** | 1.08× |
| Selective Zipfian three-Range | **15.32** | 73.17 | 0.21× |

Prepared streaming isolates state overhead: uniform three-Multi takes 1,121 µs
with legacy objects versus **125 µs** compiled, about **9× faster**. Compilation
takes 854 µs, so the one-shot gain is modest. Zipfian three-Range improves from
13.82 to 5.14 µs prepared, but 66.68 µs compilation makes one-shot evaluation
about five times slower. Reuse and expected traversal work must guide dispatch.

Normalization helps independently: overlapping-Multi legacy streaming falls
from 3,496 to 801 µs by coalescing ranges; compiled streaming takes 335 µs.
The CSV separates raw legacy, normalized legacy, and compiled execution.

The prototype's Exact/Free whole-triple wrapper retains the array specialization
but has slightly higher instruction cost than the official scalar collector.
Keep the current Exact/Free public path rather than routing it through this wrapper.

| Persistent filter heap capacity | Legacy | Compiled |
|---|---:|---:|
| Uniform subject Range | 406 B | 4,352 B |
| Uniform overlapping Multi | 22,411 B | 4,352 B |
| Uniform three-Multi | 6,940 B | 278,528 B |

Dense transitions buy cheap visits at a potentially substantial space cost.
Packed transitions or a size-budget interpreter fallback remain useful follow-ups.

## Join results

Units are milliseconds. Total time is measured directly, not obtained by adding
independent medians. Uniform star has 4,950 answers; skew has 1,053.

| Case / executor | Preparation | Prepared visit | Total |
|---|---:|---:|---:|
| Uniform star / AutomatonJoin | <0.001 | 20.04 | **20.04** |
| Uniform star / FstJoin | <0.001 | 34.90 | 35.05 |
| Uniform star / lifted-lazy | 171.75 | 2.61 | 174.54 |
| Uniform star / lifted-merged | 202.55 | **2.01** | 203.99 |
| Skew / AutomatonJoin, current order | <0.001 | 112.79 | 113.21 |
| Skew / FstJoin | <0.001 | 12.13 | **12.14** |
| Skew / lifted-lazy | 494.91 | 2.62 | 499.02 |
| Skew / lifted-merged | 516.00 | **0.45** | 528.43 |
| Dense 16,384-row star / AutomatonJoin | <0.001 | **0.081** | **0.081** |
| Dense 16,384-row star / lifted-merged | 0.020 | 0.183 | 0.202 |

Against current AutomatonJoin, the uniform star amortizes preparation after
roughly 10 lazy or 12 merged visits to the same query and static graph. This
does not imply an advantage over application-level answer caching. The skew
fixture still favors the inexpensive planner fix from INVESTIGATION.md: manually
choosing rare-first took about 5.5 ms without symbolic preparation.

The anchored case exposes eager lifting's weakness: AutomatonJoin takes about
**2.46 µs total**, versus **77 ms** for the lifted modes. The prototype prepares
an entire later predicate view despite a sharply restricted earlier variable.
Bound-variable pushdown or demand-driven conversion is necessary for that shape.

For the contradictory-group triangle at group size 128, total times are:

| Executor | Total |
|---|---:|
| AutomatonJoin | 266.43 ms |
| Lifted-lazy | 0.807 ms |
| Lifted-merged | **0.159 ms** |
| FstJoin | **0.00115 ms** |

Lifting reads **nine distinct source states**, reusing them 210 times. The lazy
form retains seven decision states; merging reduces the answer to false. Sharing
does avoid nested enumeration, while FstJoin's early pruning remains much cheaper.

### Assignment order and memory

Interleaving common-variable bits made lazy uniform-star visits take **96.5 ms**
and 4,997,071 decision steps. Shared-variable-first ordering takes **2.6 ms** and
140,226 steps, but costs more preparation because physical-to-logical reordering
creates more temporary nodes. The default ranks variables by atom incidence,
then preference for the first pair's shared variable, then ID, consuming each
variable's bits MSB first. This is a heuristic, not an optimal order.

An interleaved merged-star construction exceeded the product budget. That is a
failed construction, not a missing fast measurement. Controls are in
[`interleaved-control.csv`](prototype/results/interleaved-control.csv) and the
`initial-interleaved-*` files. Initial files predate complete allocation reporting;
some zero allocation fields there mean unavailable.

| Retained plan heap | Capacity |
|---|---:|
| Uniform star / FstJoin | 3,024 B |
| Uniform star / AutomatonJoin | 112 B |
| Uniform star / either lifted mode | about 2.00 MiB |
| Skew / lifted-lazy | about 8.00 MiB |
| Skew / lifted-merged | about 0.50 MiB |

Merged star retains 79,867 decision nodes, after constructing 700,400 nodes and
caching 710,764 product operations. Temporary tables are additional. Budgets
limit construction to two million decision nodes and four million distinct
products; exhaustion returns an error rather than truncating results.

Byte fields count retained allocated capacities, excluding inline structs and
recursive frames. Reported workspace heap excludes call frames, transient range
vectors, and temporary compiler tables. Phase-exclusive peak RSS was not measured;
the small binding-buffer capacity is not the total construction memory cost.

## Perf findings

For prepared uniform three-Multi, legacy self samples put about 77% of cycles in
range masks, state advancement/vector construction, and allocation/deallocation.
Compiled visits spend about 59% in the walker and 41% in word/delta reads. The
per-transition vector work disappears. Summaries:
[`legacy`](prototype/results/perf-legacy.txt),
[`compiled`](prototype/results/perf-compiled.txt).

The initial lifted-star compiler spent about 59% of cycles hashing. A prototype
integer-key hasher, with full equality checks, reduced lazy preparation from
248 to 172 ms and merged preparation from 291 to 203 ms. The final profile shifts
toward `apply` and node interning. Roughly 1.5 million apply calls remain in star
construction, so cheaper hashing does not resolve its startup cost. This is an
isolated compiler choice, not a production collection change. Profiles:
[`initial`](prototype/results/perf-lifted-build-std.txt),
[`final`](prototype/results/perf-lifted-build-fast.txt).

## Recommended direction

Carry forward normalization and component-first filter compilation, with a
dispatch threshold based on expected traversal work or explicit reuse. Preserve
the Exact/Free kernel. Keep the current nested and prefix join paths. For further
composed-join work, prioritize pushing bindings into later views and lifting
source states only when the product requests them, with bounded canonicalization
tables. Account for query-representation bytes alongside index bytes.

## Reproduction

```sh
python3 examples/join/prototype/prepare.py
```

Set `prototype_dir` to the printed directory. Reuse a target directory to avoid
duplicating debug-symbol builds across snapshots:

```sh
CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo test --offline \
  --profile profiling --manifest-path "$prototype_dir/Cargo.toml" --lib --tests

PROTO_REPS=5 CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo run --offline \
  --profile profiling --manifest-path "$prototype_dir/Cargo.toml" \
  --example composition_bench > /tmp/composition.csv
```

`PROTO_SUITE=filters|joins|all` chooses a suite. `PROTO_CASE` filters case names by
substring; `PROTO_ENGINE` selects a CSV engine name. `PROTO_ORDER=interleaved`
changes assignment order; default is shared-first. `PROTO_HASH=std` selects the
standard hash algorithm for lifted compilation; default is the prototype hasher.
Separate query constructions share no cache.

Build `composition_profile` using the same manifest, profile, and target directory
to record one engine/phase per process:

```sh
perf record --delay=1000 -e cycles:u -F 999 -g --call-graph dwarf \
  -o /tmp/composition.perf.data -- \
  /tmp/sgraph-composition-target/profiling/examples/composition_profile \
  compiled visit 40000
```

Other selections include `legacy visit 4000`, `normalized visit 4000`,
`compiled compile 5000`, `monolithic compile 50`, and `lifted-merged compile 30`.
Saved profiles are text summaries. Temporary binary recordings were removed
after summarization to stay within workspace quota.
