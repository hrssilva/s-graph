# The same join automaton over postorder and BFS FSTs

The two layouts are close on most tested queries. BFS retains a modest advantage
on skew. The remaining functional-graph regression is a scheduling problem:
the faster schedule runs equally well on either layout. I would keep BFS as the
default and focus the next optimization on schedule selection.

## What was held constant

Both variants run the production `ScheduledJoin::with_source`, including its
bounded adaptive planner, guards, register effects, macro transitions, workspace
and output callbacks. The BFS adapter is the production `FstSource`; a
[prototype Pow2Source](prototype/layout_source.rs) reads the postorder layout.
Its accessors mirror the BFS accessors and subtract backward child deltas.
The executor itself is reused without a backend-specific algorithm port.

The comparison checks full sorted binding sets against the fixed-level FstJoin
reference, known fixture cardinalities where available, identical planning
statistics/order/cut, identical execution counters, and equal plan allocation.
Physical state counts, build visitation counts and heights also match. A small
independent exhaustive-assignment test covers repeated variables, predicate
variables and multiple block sizes. All checks pass.

Ten fixtures were run with adaptive, nested and level-synchronized schedules.
Each timing phase uses five warm measurements calibrated to about 60 ms each;
adaptive is repeated in reverse layout/build order. The table aggregates the
two adaptive pass medians. Preparation, prepared traversal and fresh-plan total
are measured separately. Input cloning is outside construction timings.

The default capacity is 65,536 bytes. Uniform-star and skew also run at 4,096
bytes. Graphs, reference results and both layouts are resident during a run.
These are synthetic warm controls, with no CPU pinning or cold-cache claim.

## Adaptive prepared traversal, default block capacity

| Query | Rows | PathFstDeltaPow2 | PathFstDeltaPow2Bfs | Interpretation |
|---|---:|---:|---:|---|
| Uniform star | 4,950 | 19.99 ms | 20.50 ms | Small difference; perf run is within 0.4% |
| Uniform triangle | 0 | 15.08 ms | 14.99 ms | Close |
| Anchored uniform path | 0 | 2.19 µs | 2.16 µs | Close; this fixture is an unsuccessful lookup |
| Skew | 1,053 | 4.48 ms | 4.12 ms | BFS takes about 8% less time |
| Mixed triangle | 64 | 38.65 µs | 39.33 µs | Close |
| Functional triangle | 20,000 | 4.98 ms | 4.95 ms | Close |
| Functional shared-predicate path | 60,000 | 9.99 ms | 10.20 ms | Small difference |
| 4,096-atom chain | 16 | 2.08 ms | 2.15 ms | Small postorder advantage here; separate perf run is tied |
| 4,096-atom cycle | 16 | 2.09 ms | 2.18 ms | Small postorder advantage in these timing passes |

Parity-triangle is also checked. Adaptive preparation certifies it empty on both
layouts, leaving no graph reads during the prepared visit. Its roughly 2 ns
terminal check is not a useful layout measurement. Preparation plus a fresh
visit is about 1.5 µs for both.

The 4 KiB control gives 20.34/20.27 ms for uniform-star (postorder/BFS), and
4.70/4.25 ms for skew. The modest skew improvement persists across capacities;
the small uniform difference changes direction.

Changing layout also does not change the adaptive decision. On skew both choose
the rare atom first and finish whole atoms (`order=[1,0]`, `cut=0`). Both visit
216,250 nodes. The unreordered nested control takes about 108/106 ms, much more
than the layout difference. On mixed-triangle both choose a two-level prefix.

## Perf: locality helps some shapes, not every shape

Counters cover only warmed prepared traversal using FIFO enable/disable, with
construction, validation and preparation excluded. Each pair has three runs,
alternating layout order. All four events report **100% running time**.
These generic hardware cache events describe the whole join, including plan
and workspace traffic; they do not identify graph-only or a particular cache
level's misses.

| Query | Layout | Cycles/query | Instructions/query | Cache misses/query |
|---|---|---:|---:|---:|
| Uniform star | Postorder | 79.84 M | 240.30 M | 70,651 |
| Uniform star | BFS | 80.37 M | 239.09 M | 42,820 |
| Skew | Postorder | 15.97 M | 48.70 M | 92,992 |
| Skew | BFS | 15.00 M | 48.51 M | 64,295 |

On skew, BFS has about 31% fewer cache misses and 6% fewer cycles with only
0.4% fewer instructions. This is consistent with better locality in the chosen
traversal. On uniform-star, about 39% fewer cache misses do not yield a speedup;
the layout difference is not the dominant runtime cost on that fixture.

The huge-chain reference graph has just two physical FST states. Its observed
cache misses mostly concern the much larger query plan/workspace, so attributing
its few-percent variation to graph ordering would be unwarranted. Perf measures
2.086/2.083 ms and 8.13/8.23 M cycles for postorder/BFS on that case: essentially
the same traversal cost.

## The remaining regression is in scheduling

On the functional triangle, both adaptive planners select nested traversal:

| Schedule | Postorder | BFS | Node openings | Candidates |
|---|---:|---:|---:|---:|
| Adaptive, choosing nested | 4.98 ms | 4.95 ms | 342,861 | 362,860 |
| Forced levels | 1.95 ms | 1.96 ms | 134,295 | 154,294 |

Perf confirms the same effect: about 81 M instructions under adaptive versus
32 M under levels, on either layout. Hardware cache-miss counts are already negligible
on this 46-state fixture. The approximately 2.5x improvement comes from the
schedule reducing search work. Switching the default storage to postorder would
leave this problem intact.

The shared-predicate functional path similarly improves from roughly 10 ms to
4.7–4.9 ms under levels. The large chain and cycle improve to roughly 1.4–1.5 ms
under levels on both layouts. Bounded planning still needs better discrimination
on these structured relations; the layout comparison does not change that.

## Space and construction

Suffix minimization produces the same number of physical states and packed
record bytes. At the default capacity, uniform has 178,613 states and 94 blocks
on both; skew has 109,878 states and 83 blocks on both.

Reported retained heap allocations are 6,144,522/6,144,802 bytes for uniform and
5,427,898/5,428,178 bytes for skew (postorder/BFS). The 280-byte BFS difference
is its O(height) scan layouts and physical state counts, not a per-node cost.
These counters exclude inline structs and query plan/workspace memory.

Construction is different: postorder writes during bottom-up minimization;
BFS collects identities, assigns addresses and patches forward deltas. Its
constructor also includes the production input and delta checks. The raw data
records one build in each order, approximately 0.69/0.72 s for the uniform-star
invocations and 0.33/0.34 s for skew. These two samples are descriptive setup
measurements, not a controlled constructor study.

The legacy postorder type still lacks the default FST's persistence and direct
query integrations, checked construction, and singleton-domain fix. In
particular its current builder drops `(0,0,0)` when all domains are one; such
graphs are excluded from the timing comparison. A successful adapter benchmark
does not make those APIs interchangeable without that integration work.

## Reproduce

```sh
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
# Use the printed path:
sgraph_snapshot="/printed/disposable/path"
cargo test --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --lib layout_source
cargo build --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example layout_bench
# Supply the resulting binary path (respect CARGO_TARGET_DIR if set):
sgraph_layout_binary="$sgraph_snapshot/target/profiling/examples/layout_bench"
python3 examples/join/prototype/layout_compare.py "$sgraph_layout_binary" --output target/layout-comparison
python3 examples/join/prototype/layout_compare.py "$sgraph_layout_binary" --output target/layout-comparison-4k --cases uniform-star,skew --capacity 4096
python3 examples/join/prototype/layout_perf.py "$sgraph_layout_binary" --output target/layout-perf --cases uniform-star:adaptive,skew:adaptive,functional-triangle:adaptive,functional-triangle:levels,large-chain:adaptive
```

The scripts run timings sequentially. Prototype accessors are injected only
into the disposable crate; the production implementation is unchanged.

Raw data and provenance:

- [Default-capacity summary](prototype/results/layout-comparison/summary.json), [individual runs](prototype/results/layout-comparison/runs.json).
- [4 KiB summary](prototype/results/layout-comparison-4k/summary.json).
- [Perf summary](prototype/results/layout-perf/summary.json), [individual counters](prototype/results/layout-perf/runs.json).
- [Manifest and source hashes](prototype/results/layout-manifest.json).
