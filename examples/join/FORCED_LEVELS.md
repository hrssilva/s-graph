# Forcing level synchronization on every join shape

Levels improves the functional relations and huge XOR BGPs, but regresses on
uniform joins, anchors and skew. Carry chains become exponentially expensive.
Keep adaptive scheduling; improve its selection of Levels. No production
policy or executor was changed for this experiment.

## Comparison

Both policies run the current `ScheduledJoin` executor. `Policy::Levels` visits
every atom at one storage level before descending to the next level.
`Policy::Adaptive` can finish whole atoms, synchronize a prefix, or use Levels.
The experiment forces Levels without planner probes, using its existing
heuristic atom order. This compares the explicit policy against the default,
including any difference in atom ordering: on skew, adaptive chooses `[1,0]`
where Levels keeps `[0,1]`.

The same executor runs over `PathFstDeltaPow2Bfs` through production `FstSource`
and over `PathFstDeltaPow2` through the comparison adapter. Results below are
**prepared traversal times in microseconds**; smaller is better.

| Shape | Rows | BFS adaptive | BFS Levels | Pow2 adaptive | Pow2 Levels |
|---|---:|---:|---:|---:|---:|
| Uniform star | 4,950 | 19,914 | 29,075 | 20,307 | 29,817 |
| Uniform triangle | 0 | 14,836 | 26,136 | 15,182 | 26,468 |
| Anchored uniform path | 0 | 2.17 | 75.92 | 2.20 | 76.95 |
| Skew | 1,053 | 4,077 | 10,274 | 4,419 | 10,445 |
| Dense star | 16,384 | 80.02 | 86.62 | 78.60 | 88.68 |
| Mixed triangle | 64 | 39.35 | 35.18 | 38.70 | 35.50 |
| Parity triangle, 128/group | 0 | 0.002¹ | 0.386 | 0.002¹ | 0.386 |
| Functional triangle | 20,000 | 5,038 | 1,942 | 5,025 | 1,980 |
| Functional shared-predicate path | 60,000 | 9,936 | 4,800 | 10,161 | 4,893 |
| XOR chain, 64 atoms | 16 | 43.84 | 18.58 | 44.16 | 18.60 |
| XOR chain, 1,024 atoms | 16 | 481.45 | 321.74 | 477.20 | 319.42 |
| XOR chain, 4,096 atoms | 16 | 2,040 | 1,399 | 2,100 | 1,392 |
| Star, 1,024 atoms | 16 | 733.65 | 486.15 | 743.15 | 476.47 |
| XOR cycle, 4,096 atoms | 16 | 2,089 | 1,406 | 2,171 | 1,444 |
| XOR cycle, 4,097 atoms | 0 | 2,052 | 1,448 | 2,026 | 1,365 |

¹ Adaptive proves this query empty during preparation. The prepared visit only
checks that certificate; its time is not comparable search work. Fresh-plan
total is essentially tied: 1.51 µs adaptive versus 1.49 µs Levels on BFS.

The schedules have the same win/loss direction on both layouts. On BFS, Levels
is 2.59× faster on the functional triangle, 2.07× on the functional path, and
1.42–2.36× on these large BGPs. It is 1.46× slower on uniform star, 1.76× on
uniform triangle, 2.52× on skew and 35.1× on the anchored unsuccessful lookup.

## Preparation matters on huge BGPs

Times below are microseconds on BFS. Total independently measures fresh plan
construction, workspace creation, traversal and destruction; it is not the sum
of the other medians.

| Shape | Adaptive prepare | Levels prepare | Adaptive total | Levels total |
|---|---:|---:|---:|---:|
| Functional triangle | 25.74 | 1.45 | 5,059 | 1,948 |
| XOR chain, 64 atoms | 99.88 | 18.54 | 146.30 | 37.59 |
| XOR chain, 1,024 atoms | 1,555 | 287 | 2,021 | 608 |
| XOR chain, 4,096 atoms | 7,450 | 1,307 | 9,666 | 2,797 |
| Star, 1,024 atoms | 2,136 | 339 | 2,935 | 827 |
| XOR cycle, 4,096 atoms | 7,413 | 1,334 | 9,744 | 2,800 |
| XOR cycle, 4,097 atoms | 7,410 | 1,298 | 9,500 | 2,763 |

The bounded node budget does not bound compilation cost: adaptive recompiles
candidate orders while probing. Thousands of atoms make that cost visible.
For these large fixtures, choosing Levels directly improves total time roughly
3.3–3.9×, more than the traversal improvement alone.

## Carry chains expose the failure mode

The relation is just 16 edges, `(s,0,(s+1) mod 16)`. The query joins successive
edges into a chain. Every tested chain has exactly 16 complete answers, checked
against an analytical oracle. Changing the relation from XOR to addition makes
many coarse prefixes locally compatible even when they cannot extend to a
complete solution. Level synchronization enumerates their combinations before
later bits reject them. Whole-atom traversal determines the next value early.

| Atoms | Adaptive node openings | Levels node openings | BFS adaptive visit | BFS Levels visit |
|---|---:|---:|---:|---:|
| 4 | 161 | 465 | 11.75 µs | 18.41 µs |
| 8 | 353 | 8,145 | 7.28 µs | 122.54 µs |
| 12 | 545 | 131,025 | 31.26 µs | 4.20 ms |
| 16 | 737 | >1,000,000 | 14.28 µs | 28.13 ms |
| 20 | 929 | >1,000,000 | 57.25 µs | 450.51 ms |
| 32 | 1,505 | >1,000,000 | 30.12 µs | Process exceeded 5 s |
| 4,096 | 196,577 | >1,000,000 | 2.97 ms | Process exceeded 5 s |

The diagnostic run stops at one million node openings. `>1,000,000` means the
budget was exhausted before completion, not a measured full count. At 20,
32 and 4,096 atoms, it emits no answers before reaching that budget. Adaptive
finishes within the budget in every case. For 4/8/12 atoms, the measured Levels
counts grow approximately 16× per four added atoms while adaptive grows linearly.

Separate uninstrumented runs provide the traversal times, including fresh
workspace allocation and copying the answer rows. These are medians of three
separate process launches, with no internal warmup; short adaptive measurements
are noisy and are not comparable to the warm main matrix. The work counts and
large regressions are decisive. Timed-out configurations are attempted once;
the five-second limit covers the whole process, including graph construction
and preparation, and is not a precise traversal lower bound.

Pow2 reproduces the failure: 28.32 ms at 16 atoms, 447.76 ms at 20, and process
timeouts at 32 and 4,096. Its adaptive 4,096-atom traversal takes 3.00 ms.
Storage order does not resolve the schedule explosion.

## What this says about the planner

Node counts explain the larger differences without attributing them to layout:

| Shape | Adaptive node openings | Levels node openings |
|---|---:|---:|
| Uniform star | 1,220,372 | 1,700,957 |
| Uniform triangle | 913,655 | 1,326,494 |
| Anchored path | 175 | 4,709 |
| Skew | 216,250 | 539,301 |
| Functional triangle | 342,861 | 134,295 |
| Functional path | 698,581 | 324,303 |
| XOR chain, 4,096 atoms | 131,049 | 98,289 |

Mixed triangle opens the same 2,475 nodes under both schedules; its smaller
timing difference comes from execution shape and preparation rather than fewer
node openings. Dense star is close and output dominated.

The current planner tries Levels for 64 node openings as a cheap completion or
contradiction check. If that probe does not finish, it evaluates nested orders
and may choose a synchronized prefix; it does not score Levels alongside those
alternatives. That explains why it abandons Levels on the functional fixtures
where Levels eventually wins.

My next change would keep Levels as a candidate under a bounded, comparable
work probe, retain the strong-anchor shortcut and atom-order search, and reuse
compiled stages across candidate schedules where possible. A short probe can
still miss delayed carry-chain expansion; runtime work monitoring with a safe
restart/fallback deserves testing before relying on the probe alone. Restarting
after output requires duplicate suppression or resuming only unvisited work.
Any such mechanism should remain an executor scheduling component, so all
automata sources and combinators can reuse it. These changes are proposals,
not implemented as part of this comparison.

## Reproduction and checks

Run benchmarks sequentially. This experiment uses the `profiling` release
profile, Rust 1.98.1, AMD Ryzen 5 PRO 4650U, block capacity 65,536, with no CPU
pinning. The main matrix uses five calibrated measurements per phase targeting
50 ms, in two passes with reversed layout/build order. The aggregate is the
median of the two pass medians. Graph construction and correctness checks are
outside the reported query timings.

```sh
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
# Use the disposable path printed above:
sgraph_snapshot="/printed/disposable/path"
export CARGO_TARGET_DIR="$PWD/target/automata-prototype"
export TMPDIR="$PWD/target/automata-tmp"
cargo build --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example layout_bench --example schedule_stress
python3 examples/join/prototype/levels_compare.py target/automata-prototype/profiling/examples/layout_bench --output target/levels-results
python3 examples/join/prototype/levels_stress.py target/automata-prototype/profiling/examples/schedule_stress --output target/levels-stress-results
cargo test --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --lib layout_source
```

For all 15 main fixtures, full sorted bindings match FstJoin under both policies
and layouts. Planning and full execution statistics match across layouts for
each policy, as do plan allocations. Carry-chain completions match all 16
analytical answers; budgeted partial rows are checked against that oracle.
The independent small-graph adapter test also passes. The budget method is
injected into the disposable crate only; production source hashes remain
unchanged from promotion.

Raw main [runs](prototype/results/forced-levels/runs.json) and
[summary](prototype/results/forced-levels/summary.json), carry-chain
[runs](prototype/results/forced-levels-stress/runs.json) and
[summary](prototype/results/forced-levels-stress/summary.json), and
[provenance](prototype/results/forced-levels-manifest.json) accompany this report.
