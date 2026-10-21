# Automaton analytics and large BGP composition

This extends the disposable prototype with PageRank, synchronous label
propagation, and connected BGPs containing thousands of distinct patterns.
The production storage and executors are unchanged. The prototype now exercises
one typed composition layer with two different execution strategies: register
backtracking for joins, and edge contributions plus reductions for analytics.

## Components and semantics

[`automata_core.rs`](prototype/automata_core.rs) adds `Transducer`, `Project`, and
`Reducer`. `Project` attaches an output function to an ordinary recognizer;
`And`, `Or`, and `Not` retain their existing language semantics. Reduction owns
persistent effects. It is separate from `ReversibleAction`: contributions to a
rank or vote buffer must survive backtracking out of an FST prefix.

[`analytics.rs`](prototype/analytics.rs) supplies:

- `EdgeSource`, with monomorphized compressed and CSR implementations.
- `CompressedEdges<TripleSource>`, an iterative scan with precompiled predicate
  masks and coordinate decoding layouts.
- `fold`, which applies a composed recognizer/transducer to each edge, then
  forwards accepted contributions to a reducer.
- Reusable PageRank and label-propagation workspaces and iteration controllers.
- Independent tuple-based correctness references.

An edge is a one-event input word in this executor. Its recognizer starts afresh
for every edge. Cross-edge state lives in a reducer; cross-round state lives in
vertex arrays. This is an explicit specialization, not an implementation of an
arbitrary multi-event path automaton. There are no trait objects, per-edge hash
lookups, or allocations inside the compressed scan or PageRank edge kernel.
Label propagation sorts reused per-vertex vote slices; it allocates their backing
storage during run initialization, not during edge emission.

The compressed source retains coordinates per traversal occurrence. An FST
address is not a vertex ID, and shared suffixes do not authorize merging
contributions from different prefixes.

```text
fold_edges(source, transducer, readonly_context, reducer):
    initial = transducer.start()
    for each decoded edge in source:
        state = transducer.step(initial, edge, readonly_context)
        if state exists and transducer.accepts(state, readonly_context):
            reducer.push(transducer.output(state, edge, readonly_context))
```

Predicate equality is lowered to prefix-safe bitmap masks before scanning. The
slot decoder retains each level's shifts, bit masks, and remaining bit counts.
Traversal combines decoded slot bits with the current coordinate prefix.
Masks and decoding depend on physical layout; algorithm contributions do not.

## PageRank

The selected triples define a directed multigraph: differently labelled triples
between the same vertices remain separate edges when all predicates are
selected. Identical RDF triples have set semantics. Explicit vertex count
includes isolated vertices. Damping is 0.85 in the measurements; dangling rank
is distributed uniformly. Termination uses L1 change or the round limit.

```text
degree = reduce outgoing-edge counts by source
rank[v] = 1 / vertex_count
repeat, up to max_rounds:
    dangling = sum(rank[v] where degree[v] == 0)
    contribution[v] = damping * rank[v] / degree[v], or zero for dangling v
    next[v] = (1 - damping + damping * dangling) / vertex_count
    fold edges:
        emit (edge.target, contribution[edge.source])
        reduce by addition into next
    residual = sum(abs(next[v] - rank[v]))
    swap(rank, next)
    stop if residual <= tolerance
return ranks, rounds, convergence status, residual
```

The implementation precomputes one contribution per source vertex each round,
so division does not occur per edge. Four O(V) buffers retain degree, rank,
next-rank, and contribution. Degree is rebuilt on each complete algorithm run;
its scan is included in timings.

## Label propagation

This prototype uses a synchronous undirected projection. Each non-loop selected
triple contributes votes in both directions; reciprocal or differently labelled
triples contribute independently. Self-loops are excluded. Ties choose the
smallest label, and isolated vertices retain their labels. This deterministic
algorithm can oscillate, so the result reports whether it converged before the
round limit. It is a baseline for exercising the architecture, not Louvain or
Leiden and not a community-quality evaluation.

```text
count non-loop neighbor contributions per vertex
prefix-sum counts to reserve one reusable vote segment per vertex
label[v] = v
repeat, up to max_rounds:
    reset vote-segment write positions
    fold edges accepted by And(predicate_filter, NonLoop):
        emit (source, label[target]), (target, label[source])
        append into the corresponding vote segments
    for each vertex:
        sort its vote segment
        next[v] = most frequent label; smallest label breaks ties
        if no votes: next[v] = label[v]
    swap(label, next)
    stop if no label changed
return labels, rounds, convergence status
```

The vote buffer costs O(E) memory, in addition to O(V) state. It contains labels,
not a retained edge list; nevertheless it can dominate compressed graph size.
Sorting costs depend on degree distribution. A memory-constrained or highly
skewed implementation should compare different reducers before adopting this
one as a default.

## Large BGPs

[`large_bgp_cases.rs`](prototype/large_bgp_cases.rs) constructs analytical answer
oracles for 4,096-pattern chains, a 1,024-pattern star with distinct predicates,
a 4,096-pattern cycle, and an unsatisfiable 4,097-pattern cycle. Every pattern is
distinct and connected. Tests assert the retained stage count equals
`distinct_patterns * storage_height`, so deduplication cannot make these tiny
queries. Complete bindings are checked, including every variable column.

The prefix-functional fixtures use `target = source XOR 1` for chains/cycles,
and `target = source` for each star predicate. They deliberately bound the
answer count at 16 to isolate composition cost. They do not establish tractable
runtime for arbitrary thousand-atom conjunctive queries.

Each large query checks Nested, Prefix(1), Levels, and Adaptive schedules,
cancellation followed by two complete visits using the same workspace, and
planning's 832-node exploration bound. Execution runs on a 128 KiB thread stack.
A separate repeated-variable test checks 128 distinct self-loop patterns,
duplicates the input patterns, and verifies deduplication and shared-variable
priority updates.

Two changes make large composition practical:

1. Above 64 distinct atoms, the compiled register executor uses flat heap frames
   for the entire schedule. Small joins keep their fast recursive macro kernel.
2. Large-query compilation uses a B-tree set for deduplication and an incidence
   index plus priority heap for constants/connectivity ordering. Each variable
   becomes bound once; only incident atom scores change. With at most three
   terms per atom, those phases cost O(A log A), followed by O(A * height) stage
   generation per candidate schedule. Original atom ordinal breaks score ties.
   Small queries preserve the previous ordering implementation and tie rules.

```text
order_large_bgp(atoms):
    build variable -> incident atom index
    heap = all atoms, ordered by constants, shared variables, original ordinal
    while an unselected atom remains:
        pop the highest priority non-stale entry
        select it
        for each variable newly bound by this atom:
            increment shared-variable scores of its unselected incident atoms
            push their updated priorities
```

A separate `target = (source + 1) mod 16` carry-chain test runs 4,096 patterns
with Nested and Adaptive. Coarse prefixes of this relation permit many partial
assignments that disappear only at lower bits. Synchronizing thousands of such
atoms can have exponential intermediates. The benchmark compares all three
engines at 20 carry patterns and only the whole-atom schedules at 4,096. The
planner's graph-read budget bounds exploration, not final query execution or
compiled plan size. This is a search-order limitation, not a stack limitation.

## Measurement and reproduction

Both analytics backends run the same statically composed algorithm kernels.
The CSR projection is built **from the compressed source**, with two scans, not
from a free uncompressed copy of the triples. Cold CSR timings include that
conversion and a fresh algorithm workspace. Warm timings reuse a prepared
source and workspace but include degree/vote-layout preparation each run.
FST construction is outside both timings because the engine already stores it.

Reported graph bytes distinguish the complete FST (including a noise predicate)
from an additional CSR projection containing only the selected predicate. CSR
is an optional extra allocation while the FST remains resident. Algorithm bytes
are retained vector capacities; add the small compressed scan stack separately.
They are not peak allocation or process RSS measurements.

```bash
mkdir -p target/automata-tmp
export TMPDIR="$PWD/target/automata-tmp"
export CARGO_TARGET_DIR="$PWD/target/automata-prototype"
snapshot="$(python3 examples/join/prototype/prepare.py)"
cargo test --offline --profile profiling --manifest-path "$snapshot/Cargo.toml" --lib --tests
cargo build --offline --profile profiling --manifest-path "$snapshot/Cargo.toml" \
  --example analytics_bench --example large_bgp
"$CARGO_TARGET_DIR/profiling/examples/analytics_bench"
"$CARGO_TARGET_DIR/profiling/examples/large_bgp"
python3 examples/join/prototype/analytics_perf.py \
  "$CARGO_TARGET_DIR/profiling/examples/analytics_bench" \
  --output examples/join/prototype/results/analytics-perf
```

`ANALYTICS_VERTICES` selects a positive multiple of 64 (default 16,384), and
`ANALYTICS_CASE` selects `uniform`, `skew-dangling`, or `communities`. The default
fixtures have roughly 128k–262k selected edges plus a noise predicate. Algorithm
runs use a 50-round cap, PageRank L1 tolerance 1e-12, and five timing samples
following warmup. The profile harness gates counters and sampling around warmed
algorithm runs only, excluding graph construction, projection, and warmup.

## Results and architectural opinion

All **94 tests pass** (78 library, 9 algorithm integration, 7 native join). The new analytics checks compare complete vectors with independent tuple-based implementations, including randomized graphs, predicate projections, dangling and isolated vertices, self-loops, duplicate triples, full-width coordinates, empty graphs, cancellation/reuse for joins, and explicit nonconvergence.

Cold analytics totals below include source preparation, algorithm setup, and all iterations, in milliseconds. The CSR column includes building that projection from the existing FST. These are synthetic fixtures, not a new real-data result.

| Fixture | PageRank FST | PageRank CSR | Labels FST | Labels CSR |
|---|---:|---:|---:|---:|
| uniform | 132.73 | 17.13 | 118.61 | 50.80 |
| skew-dangling | 54.69 | 7.56 | 31.89 | 16.11 |
| communities | 113.55 | 17.17 | 60.90 | 42.69 |

PageRank converges in 18 rounds on uniform and 21 on skew/dangling; both backends reach the 50-round cap without convergence on the community fixture. Label propagation converges in 10, 7, and 12 rounds respectively. A separate two-vertex test explicitly checks oscillation.

Building the CSR projection costs 12.02 ms, 4.35 ms, and 3.91 ms respectively. Its retained storage is 1.18 MB, 0.64 MB, and 1.08 MB **in addition to** the existing FST. The PageRank vectors occupy 0.52 MB; label-propagation vectors/votes occupy 1.42–2.49 MB. Compressed scan workspace is another 200 bytes in these fixtures.

The largest BGP totals, including planning and fresh workspace, are also milliseconds:

| Shape | Distinct atoms | Rows | Scheduled | AutomatonJoin | FstJoin |
|---|---:|---:|---:|---:|---:|
| chain | 4096 | 16 | 7.64 | 55.63 | 42.98 |
| star | 1024 | 16 | 2.25 | 4.77 | 3.42 |
| cycle | 4096 | 16 | 8.24 | 55.55 | 43.00 |
| cycle | 4097 | 0 | 9.02 | 55.96 | 43.04 |
| carry-chain | 4096 | 16 | 8.38 | 56.92 | not run |

The 4,096-chain compiler drops from 46.52 ms in the initial implementation to 5.54 ms after incremental ordering/deduplication. Warm traversal remains 2.04 ms, versus 1.51 ms for FstJoin and 13.77 ms for AutomatonJoin. Thus the total-query win over FstJoin here comes from preparation; FstJoin remains faster when an already prepared query is repeatedly traversed.

The large compiled plan costs 2.69 MB and workspace 0.41 MB for that chain, versus 0.23/0.016 MB for AutomatonJoin and 1.77/0.67 MB for FstJoin. These are not free abstractions in retained plan size. Small-query preparation can still favor AutomatonJoin. The unchanged uniform-star control remains about 19.9 ms for both Scheduled and AutomatonJoin, versus 35.3 ms for FstJoin.

The 20-pattern carry-chain illustrates the opposite scheduling hazard: FstJoin takes about 505 ms total, AutomatonJoin 28 microseconds, and Scheduled 55 microseconds. This is why neither whole-atom nor level-major scheduling should become the universal policy. The large carry version deliberately omits the exponential FstJoin run.

### What perf says

The warmed uniform PageRank run executes about **1.414 billion instructions** over the compressed source versus **67.6 million** over CSR, with the same contribution/reduction kernel. Sampling attributes about 24% of compressed cycles to `read_word`, 14% to `read_delta`, and the remainder to the inlined algorithm/traversal function. There is no trait-object dispatch in either execution path. The comparison indicates that storage traversal/decoding is the dominant additional cost; the broad inlined symbol alone cannot assign every remaining cycle to decoding.

For label propagation, compressed and CSR execute roughly 1.106 billion and 354 million instructions respectively. Sorting accounts for about 46% of sampled CSR cycles but only 16% of compressed cycles. This explains the narrower backend gap when the reducer itself does more work. Counters use FIFO gating, three trials, and report 100% event running time in these profiles.

An experimental 64-slot coordinate table reduced uniform PageRank instructions from 1.414 to 1.219 billion, but increased cycles from about 530 to 547 million and median profiled runtime from roughly 134 to 138 ms. It also increased scan-plan storage from 160 to 3,880 bytes. The arithmetic decoder remains the default. The rejected table variant is reproducible with [this patch](prototype/results/analytics-table.patch) applied inside the disposable crate; its [measurements](prototype/results/analytics-table.csv) and [profiles](prototype/results/analytics-perf-table/summary.json) are retained.

### Recommendation

Keep the common automaton/transducer/reducer model, with specialized execution chosen for the workload. It now demonstrably covers joins and two iterative analytics algorithms while preserving static dispatch. For repeated full-graph analytics, an optional memory-budgeted CSR projection already repays its measured conversion cost during one complete run on these fixtures. Direct compressed execution remains useful where the extra allocation is unacceptable. Neither result establishes a universal analytics policy or supports replacing the compressed store.

Next performance work should target compressed scan locality and block-level processing, and measure reducers on highly skewed degree distributions. Expanding the automaton API alone will not remove the decoding cost. Louvain/Leiden, graph contraction, parallel reductions, sparse frontiers, and automatic adjacency-view selection remain future work.

Raw artifacts: [analytics CSV](prototype/results/analytics-comparison.csv), [large BGP CSV](prototype/results/large-bgp-comparison.csv), [perf counters](prototype/results/analytics-perf/summary.json), [small join control](prototype/results/analytics-small-join-control.csv), and [test log](prototype/results/analytics-tests.log). Independent timing columns are separately measured medians and need not sum to the measured total.
