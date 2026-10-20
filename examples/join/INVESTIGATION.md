# Join shape investigation — 2026-10-20

Keep AutomatonJoin. The reported skew regression is fixable through better
planning; replacing its nested walk with recursion over interleaved levels does
not address its cause. Keep prefix interleaving as a second execution strategy:
there are small cyclic queries for which no atom order rescues the nested walk.

The algorithm proposal and pseudocode are in [PROPOSED_JOIN.md](PROPOSED_JOIN.md).

## What was measured

The supplied examples initially could not run: identifiers had line breaks in
them, the shared-module paths were relative to the wrong directory, and Cargo
did not discover the nested examples. Those issues are repaired, with explicit
Cargo example entries. `JOIN_ITERS` controls the uniform example's query count.
The production join implementations are unchanged by this investigation.

The isolated experiment in `lab/` copies the current library into `/tmp`, then
patches only AutomatonJoin to expose node/candidate counters and optionally
preserve input atom order. It does not introduce an alternative traversal.
Node counters compile out of ordinary `visit`; diagnostics run separately.

Measurements below use Rust 1.98.1, the `profiling` profile (release optimization
with debug symbols), default target CPU settings, and an AMD Ryzen 5 PRO 4650U.
The experiment compiles plans outside the timed region, warms each executor,
then reports the median of seven complete visits with reused workspaces.
It compares sorted full binding vectors, not just row counts. Skew cases also
have an independent answer vector generated directly from the source triples;
the constructed empty triangles have an independently known empty answer.
Uniform cases use FstJoin as the comparison oracle.

These are synthetic, warm-process measurements, not a rerun of the team's
external real-data benchmark. Raw timings and counters are in
[`lab/results.csv`](lab/results.csv) and [`lab/parity.csv`](lab/parity.csv).
Engines run sequentially in a fixed order; timing ranges are included in the
CSVs. The work-count differences are more decisive than small timing differences.

## The skew regression is an atom-order problem

The supplied skew fixture has 2,000,000 subjects, 20 classes, and 19,939 rare
edges spread over a 500,000-object domain. Its query is:

```sparql
?x :type :C0 .
?x :rare ?b .
```

All four executors prioritize constant count before connectivity, without any
data estimate. Consequently the two-constant class atom comes first, even if
the caller reverses the patterns. Reversing the input alone in the live library
does **not** change this plan; the experimental ordering bypass is necessary.

The class has 100,043 matching subjects. AutomatonJoin starts the rare-edge
atom from the root once per subject, paying for many unsuccessful descents.
Putting the rare atom first instead scans its 19,939 edges and checks a fully
bound class-membership pattern per edge. Both produce exactly 1,053 bindings.

| Executor / order | Median visit | Tree-node reads | Atom entries |
|---|---:|---:|---:|
| NativeJoin / current | 13.52 ms | not collected | — |
| FstJoin / current | 12.08 ms | 539,301 | — |
| RecursiveJoin / current | 13.39 ms | 539,301 | — |
| AutomatonJoin / class first | 111.75 ms | 8,192,603 | 100,044 |
| AutomatonJoin / rare first | **5.48 ms** | **216,250** | **19,940** |

Changing the order gives about a **20× speedup** over the current automaton
plan and is about **2.2× faster than FstJoin** on this fixture. It requires no
additional graph index or change to the FST representation. An automatic
planner that chooses this order has not been implemented; these timings assume
the chosen order is already available.

The sensitivity sweep confirms this is not a marginal case:

| Class count | Rare-edge fraction | FstJoin | Automaton current | Automaton rare first |
|---|---:|---:|---:|---:|
| 20 | 0.01% | 0.146 ms | 26.31 ms | 0.052 ms |
| 20 | 0.1% | 1.345 ms | 51.77 ms | 0.536 ms |
| 20 | 1% | 12.08 ms | 111.75 ms | 5.48 ms |
| 20 | 10% | 92.67 ms | 381.80 ms | 51.25 ms |
| 2 | 1% | 12.64 ms | 1,049.39 ms | 5.00 ms |
| 200 | 1% | 9.81 ms | 13.46 ms | 6.25 ms |

The last row matters: there are only 9,901 class matches, fewer than the 19,939
rare edges, yet rare-first is still faster. **“Smallest relation first” is
insufficient.** The planner must account for conditional probe cost. Checking a
fully bound triple is cheaper than repeatedly listing an unknown object through
this physical S/P/O schedule. Changing the object domain from 20 to 2,000,000
also changes the schedule and probe cost without changing the 1,053 answers;
both additional cases are in the CSV.

## What perf says

`perf record` sampled user-space cycles at 999 Hz with DWARF call chains. A
one-second recording delay and repeated execution keep construction out of the
main profile. The profiles use only the chosen automaton engine, with 60 visits
for class-first and 1,200 for rare-first. Percentages below are self samples,
not cumulative call-chain percentages.

| Function | Class first | Rare first |
|---|---:|---:|
| `walk_leaf_rec` | 52.26% | 27.79% |
| `read_word` | 20.93% | 30.77% |
| `read_delta` | 16.31% | 12.37% |
| `restriction_mask` | 6.95% | 17.56% |
| `compute_atom_masks` | 2.01% | 7.04% |

The failing plan spends about 90% in traversal and node access, versus 9% in
mask construction. Eliminating mask construction entirely would not explain or
repair a 9× loss. This is also no longer the earlier `LevelAutomaton::And`
dispatch problem documented in DESIGN.md: that abstraction is absent here.
Cycle sampling alone does not establish whether a function is memory-bound.

After correcting order, mask construction becomes a more worthwhile secondary
target (~25% of a much shorter visit): cache constant masks and compile cheap
bound-variable mask updates. The larger percentage does not mean mask work
increased in absolute time. Summaries are in
[`lab/perf-automaton.txt`](lab/perf-automaton.txt) and
[`lab/perf-reversed.txt`](lab/perf-reversed.txt).

## Why RecursiveJoin does not win decisively

RecursiveJoin literally consumes the same `fst_join::compile` plan and visits
the same `(level, atom)` search space as FstJoin. On the skew fixture both read
539,301 nodes and accept 540,353 candidates. The difference is state management:
stack recursion and next-level node arrays versus explicit backtracking frames.
There is no additional pruning algorithm in RecursiveJoin.

The uniform fixtures reproduce AutomatonJoin's earlier advantage:

| Fixed query over 1M uniform triples | FstJoin | RecursiveJoin | AutomatonJoin |
|---|---:|---:|---:|
| Shared-object star, 4,950 rows | 34.81 ms | 42.87 ms | **19.59 ms** |
| Triangle, zero rows | 30.80 ms | 34.44 ms | **15.56 ms** |

The repaired original harness also ran 40 predicate draws per shape, including
plan construction: AutomatonJoin took about 20.78 ms/query for stars and
16.03 ms/query for triangles, versus FstJoin's 35.49 and 30.91 ms/query.
RecursiveJoin was 12–23% slower than FstJoin in that run. Favor FstJoin as the
measured interleaved baseline here; recursion itself provides no demonstrated
advantage. These measurements do not prove one implementation always wins.

## Ordering cannot solve every cyclic query

An additional fixture partitions each variable domain into two groups of `n`
IDs. R and S connect every pair within the same group. T connects every pair
in opposite groups. Evaluate:

```text
R(x,y), S(y,z), T(z,x)
```

Any two atoms have many matching bindings, but all three together are
inconsistent: x must be in both the same and the opposite group as z.
For any atom order, a nested join enters atoms
`1 + 2n² + 2n³` times, despite producing zero rows. This is logical intermediate
enumeration, streamed through callbacks rather than materialized in a table.

| Group size n | Automaton node reads | Automaton atom entries | FstJoin node reads |
|---|---:|---:|---:|
| 32 | 104,969 | 67,585 | 25 |
| 64 | 680,457 | 532,481 | 25 |
| 128 | 4,817,417 | 4,227,073 | 25 |

At n=64 all six atom permutations were measured: each produces the same work
counts and takes roughly 33–34 ms. At n=128 both measured orders take roughly
252–254 ms. FstJoin rejects these clustered fixtures in under a microsecond;
the work counts are the reliable comparison at that scale. The interleaved
executor tests the contradictory group bits before enumerating full bindings.
The FST occupies only 430 bytes in each of these cases: DAG compression does
not automatically prevent a join from repeatedly unfolding equivalent structure.

The first additional empty-cycle fixture (`empty-triangle-*` in results.csv)
has disjoint adjacent domains. Reordering helps it, but the group-constraint
fixture above removes that escape: every adjacent pair joins widely.

This distinction is consistent with the primary literature:
[Leapfrog Triejoin](https://www.openproceedings.org/2014/conf/icdt/Veldhuizen14.pdf)
binds variables while intersecting participating relations, and
[Optimal Joins Using Compressed Quadtrees](https://doi.org/10.1145/3514231)
uses dimension lifting and synchronized intersection. Neither paper's bound
automatically transfers to this implementation's mixed per-dimension schedule;
the existing DESIGN.md correctly says there is no established WCO bound here.

## Recommended direction

1. **Separate planning from the choice of executor.** Let AutomatonJoin accept
   an explicit validated atom order, then replace constant-count ordering with
   estimates of scan work plus partial-result count × conditional probe work.
   Small per-predicate counts and heavy-hitter `(predicate, object)` statistics
   collected during build are candidates; account for their storage overhead.
   Preserve connectivity, and estimate fanout under already-bound variables.
   Do not perform a full cardinality scan per query just to choose an order.
2. **Keep two execution strategies.** Favor nested execution for selective
   scans followed by cheap bound probes; retain prefix interleaving for cyclic
   cores with large pairwise expansions or uncertain estimates. A later hybrid
   can use an interleaved cyclic core and nested selective branches. Cycle
   detection alone is not a sufficient dispatch rule: the uniform triangle
   already favors nested execution.
3. **Improve masks after fixing planning.** Precompute constant contributions
   and the bound-variable update recipe per atom. This targets the remaining
   rare-first profile without hiding the algorithmic failure of a bad plan.
4. **Benchmark work, answers, and storage together.** Include rare-edge sweeps,
   correlated predicates, high-degree hubs, longer chains, cycles with large
   pairwise joins, empty answers, and different ID layouts. Report full-visit
   and first-result costs separately, as well as planning time and statistics
   bytes. Do not treat `atom_entries` and `node_visits` as comparable units.

For SPARQL-shaped BGPs, also retain a correctness route for repeated variables
within an atom (`?x :p ?x`): AutomatonJoin currently rejects them. The experiments
here cover full conjunctive bindings, not projection or other SPARQL operators.
Some module documentation still says AutomatonJoin wins on every tested shape;
that conflicts with both the current crate-root guidance and these measurements.

## Reproduction

Run the repaired original examples from the repository root:

```sh
cargo run --offline --release --example skew_repro3
JOIN_ITERS=40 cargo run --offline --release --example join_perf_four
```

Create an isolated experimental copy; the script prints its absolute directory:

```sh
python3 examples/join/lab/prepare.py
```

Set `join_lab_dir` to that printed directory, then run the experiment:

```sh
JOIN_INPUT_ORDER=1 cargo run --offline --profile profiling \
  --manifest-path "$join_lab_dir/Cargo.toml" --example investigate
```

`JOIN_CASE` filters case names by substring, `JOIN_ENGINE` selects one executor,
`JOIN_REPS` controls repetitions (default seven), and `JOIN_SUBJECTS` changes the
skew fixture size (default two million). `automaton` preserves the supplied
order; `automaton-reversed` reverses it. `JOIN_INPUT_ORDER=1` is required so a
caller cannot accidentally label two heuristic-reordered plans as different.
`parity` selects the group-constraint fixtures, which used five repetitions in
the recorded run. The two CSVs are separate runs, not competing timings run
concurrently.

After building, profile only the original skew case and chosen executor:

```sh
JOIN_INPUT_ORDER=1 JOIN_CASE=skew-c20-r0.01-o500000 \
JOIN_ENGINE=automaton JOIN_REPS=60 \
perf record --delay=1000 -e cycles:u -F 999 -g --call-graph dwarf \
  -o /tmp/sgraph-join-automaton.perf.data -- \
  "$join_lab_dir/target/profiling/examples/investigate"

perf report --stdio --no-children --call-graph none --sort symbol \
  -i /tmp/sgraph-join-automaton.perf.data --percent-limit 0.5
```

Use `JOIN_ENGINE=automaton-reversed JOIN_REPS=1200` and a different output file
to sample the shorter rare-first execution for a comparable duration.

Validation: both repaired original examples completed; every experiment compared
full binding vectors; `cargo test --offline --lib --tests` passed all 70 tests.
Those existing tests primarily cover the mutable/native implementation; they
are not a substitute for the FST comparisons in the experiment.
