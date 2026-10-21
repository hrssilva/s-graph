# Large cycles: multiway pruning and prefix granularity

The large cycles strengthen the case for synchronized traversal. On 4,096
vertices, Levels beats adaptive/nested by 4.5–5.3× on prefix-friendly cycles.
On a dense 129-atom cycle with one answer, Levels finishes in 11.72 µs and 772
node openings; adaptive selects nested and exhausts five million openings
without an answer. The planner is missing an important class of wins.

The earlier functional-triangle result remains correct: about 5 ms adaptive
versus 1.95 ms Levels. The uniform triangle is a different dataset, where
Levels is slower. Cyclic query structure motivates multiway pruning, but data
and the granularity of synchronization also matter.

## Connection to WCOJ

Worst-case optimal joins address the large intermediate results that can arise
when evaluating cyclic joins through successive binary joins. The formal
guarantee concerns input cardinalities and maximum possible output size; it
does not promise superiority on every individual dataset.
[Ngo, Porat, Ré and Rudra](https://arxiv.org/abs/1203.1952).

Leapfrog Triejoin intersects candidate values from relevant relations in a
chosen variable order, conditioned on earlier bindings. Our Levels executor
enumerates occupied S/P/O slots atom by atom within each storage level, checking
previously written variable bits. Sharing prefix bindings supports similar
pruning, but does not establish the same optimality guarantee. Also, conventional
WCOJ analysis fixes the query and studies growing data; increasing atom count
tests query-size scalability separately.
[Veldhuizen, sections 3.2–3.4](https://arxiv.org/html/1210.0481v5).

## New fixtures

Every query is a closed cycle with distinct variable names:

```text
R0(x0,x1) AND R1(x1,x2) AND ... AND R[k-1](x[k-1],x0)
```

These are BGP homomorphisms: variable names need not have distinct values.
This matches ordinary SPARQL BGP semantics, without an all-different filter.

The main controls use one predicate, so subject and object digits cover the
same bit ranges at each level:

| Fixture | Data | Query lengths | Complete answers |
|---|---|---|---:|
| `xor-ring` | 4,096 vertices, edges `s → s XOR 1` | 8, 32, 128 | 4,096 |
| `carry-ring` | 4,096 vertices partitioned into directed rings of query length | 8, 32, 128 | 4,096 |
| `shuffled-ring` | Same rings, with a fixed bijective shuffle of vertex IDs | 8, 32, 128 | 4,096 |
| `odd-planted` | Complete bipartite graph, 128 vertices/side, plus an isolated self-loop | 9, 33, 129 | 1 |

The 128-atom carry fixture contains actual 128-vertex simple rings; the XOR
fixture intentionally reuses two-vertex cycles to control prefix correlation.
Each functional fixture has 4,096 edges. The dense fixture has 32,769 edges.
The sole dense answer binds every variable to the isolated self-loop vertex.
The bipartite component cannot satisfy an odd closed walk.

## Prepared traversal

Times below are milliseconds on the default BFS layout. `Budget` means the
five-million-node diagnostic stopped before completion; it is neither a
timeout nor a measured full runtime. Unfinished cases have no timing estimate.

| Data | Atoms | Adaptive | Levels | Adaptive nodes | Levels nodes |
|---|---:|---:|---:|---:|---:|
| XOR | 8 | 1.909 | **0.421** | 115,273 | 33,345 |
| XOR | 32 | 9.361 | **1.753** | 508,489 | 145,665 |
| XOR | 128 | 32.292 | **7.175** | 2,081,353 | 594,945 |
| Carry ring | 8 | 1.930 | **0.416** | 115,273 | 33,345 |
| Carry ring | 32 | 11.379 | Budget, 0 answers | 631,881 | ≥5,000,000 |
| Carry ring | 128 | 49.470 | Budget, 0 answers | 3,114,121 | ≥5,000,000 |
| Shuffled ring | 8 | 21.433 | Budget, 0 answers | 1,437,678 | ≥5,000,000 |
| Shuffled ring | 32 | Budget, 2,887 answers | Budget, 0 answers | ≥5,000,000 | ≥5,000,000 |
| Shuffled ring | 128 | Budget, 690 answers | Budget, 0 answers | ≥5,000,000 | ≥5,000,000 |

Both layouts give the same planning and execution work counts. For the
128-atom XOR cycle, postorder Pow2 measures 32.703 ms adaptive and 7.119 ms
Levels. The cyclic benefit persists across storage layouts; the carry and
shuffled regressions also persist.

## A dense cycle exposes a severe selection error

`odd-frontier` uses exactly the same edges, query and analytical answer as
`odd-planted`, but declares S/O domains of 1,024 rather than 257. This controlled
padding moves the bipartite group bit to the end of the first three-bit digit.
The first-level choices then express only the group constraint, without an
unconstrained within-group bit. Levels rejects the two incompatible group
assignments around the odd cycle before expanding any within-group values.

| Atoms | Adaptive nodes | Levels nodes | Levels visit | Levels fresh-plan total |
|---|---:|---:|---:|---:|
| 9 | 52 | 52 | 0.78 µs | 3.68 µs |
| 33 | ≥5,000,000, 0 answers | **196** | **2.89 µs** | 13.97 µs |
| 129 | ≥5,000,000, 0 answers | **772** | **11.72 µs** | 54.69 µs |

Levels opens exactly `6 × atoms − 2` nodes in these controls. Adaptive keeps
Levels at nine atoms because its initial 64-node probe completes. At 33 and
129 atoms, that probe exhausts, and the planner eventually selects nested.
Its fixed probe budget gives up on a good schedule just because reaching the
closing constraint now requires more atoms. Pow2 gives the same counts and
2.90/11.94 µs Levels visits at 33/129 atoms.

Without padding, `odd-planted` takes about 33 µs at nine atoms under either
policy; both exhaust the node budget at 33 and 129. Group incompatibility is
still present, but enumerating extra bits inside each three-bit tuple digit
multiplies partial assignments before the closing edge rejects them. This
shows a synchronization-granularity issue in addition to planner selection.

## Storage levels are not always shared variable frontiers

A second control preserves the XOR edges and entire query, changing only the
declared predicate domain from one to two (`xor-ring` versus `xor-padding`).
Both layouts use the following coordinate bit allocation:

```text
Domains [4096, 1, 4096]:
  level 0 takes S=3, P=0, O=3; remaining S=9, O=9
  subsequent levels keep S/O aligned

Domains [4096, 2, 4096], although every stored predicate is still zero:
  level 0 takes S=3, P=1, O=2; remaining S=9, O=10
  subsequent full levels retain the one-bit offset
```

At 128 atoms, Levels goes from 594,945 nodes and 7.175 ms to exhausting five
million nodes without an answer. Adaptive still finishes the padded case in
about 41 ms. This is a controlled consequence of digit allocation, not changed
connectivity or predicate selectivity. A same-numbered storage level does not
necessarily refine the same bits in every occurrence of a shared variable.

## Implications for the automaton design

These results support keeping the multiway automaton approach and strengthening
two components:

1. Keep Levels competitive in schedule selection after the initial probe.
   The dense cycle demonstrates that failing to finish in 64 reads says very
   little about its eventual cost. Probes should account for progress through
   a query and the number of participating atoms, while retaining a strict
   work bound and protection against carry-chain expansion.
2. Explore propagation at shared variable-bit boundaries, including support
   intersections from incident atom automata, before branching on unrelated
   bits within a storage digit. Such a component needs projection/lookahead
   when S/O bit ranges differ. Cache compact support masks and measure the
   cost; repeatedly scanning subtrees could outweigh the pruning benefit.

These are proposed next experiments. This change adds fixtures, diagnostics
and comparisons; it does not change the production planner or storage format.
In particular, it does not claim that Levels is already a proven WCOJ.

## Reproduction and validation

```sh
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
sgraph_snapshot="/printed/disposable/path"
export CARGO_TARGET_DIR="$PWD/target/automata-prototype"
export TMPDIR="$PWD/target/automata-tmp"
cargo build --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example cycle_bench
cargo test --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example cycle_bench
python3 examples/join/prototype/cycle_compare.py target/automata-prototype/profiling/examples/cycle_bench --output target/cycle-results
python3 examples/join/prototype/cycle_compare.py target/automata-prototype/profiling/examples/cycle_bench --output target/cycle-frontier-results --cases odd-frontier:128:9,odd-frontier:128:33,odd-frontier:128:129
```

Sequential warm measurements use five trials targeting 50 ms per phase, in
two passes reversing both layout and policy order. Tables aggregate the two
pass medians. Preparation and fresh-plan total are measured separately;
timed traversal reuses a workspace, does not copy result rows, and has no
diagnostic counters or budget checks. Construction, sorting and validation
are outside timing. CPU is Ryzen 5 PRO 4650U, Rust 1.98.1, `profiling` release
profile, capacity 65,536, without CPU pinning.

Every completed diagnostic compares complete sorted rows to an analytical
oracle. Every partial result must belong to that oracle. Exhaustive assignment
tests validate small instances of all eight main fixture families; an
additional test proves both domain controls preserve edges, query and answer
rows. Both tests pass. Work counts and planning decisions match across layouts
and passes. Production hashes remain unchanged from promotion.

The initial two-predicate controls are also retained: first `k−1` predicates
follow a permutation and the final predicate follows its inverse power,
guaranteeing 4,096 answers. They expose the same bit-allocation sensitivity.
Reproduce with `--cases xor:4096:8,xor:4096:32,xor:4096:128,carry:4096:8,carry:4096:32,carry:4096:128,shuffled:4096:8,shuffled:4096:32,shuffled:4096:128,dense-planted:128:8,dense-planted:128:32,dense-planted:128:128`.

Raw runs and summaries: [main controls](prototype/results/large-cycles-aligned/summary.json),
[dense frontier controls](prototype/results/large-cycles-frontier/summary.json),
[initial two-predicate controls](prototype/results/large-cycles/summary.json).
The [manifest](prototype/results/large-cycles-manifest.json) records source
provenance and result hashes.
