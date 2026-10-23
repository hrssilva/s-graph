# TEP3 report investigation and query-engine direction

Follow-up: [connected orders and execution control](EXECUTION_CONTROL.md) are
now implemented. The investigation and baseline measurements below describe
the engine before that follow-up; their raw evidence is preserved unchanged.

Investigation date: 2026-10-23. Production baseline: s-graph
`b1683f2eda7b16877807b3368241bf220bfa78bd`; inspected s-store:
`a79c1c8d97fbb3d930995f611781922af2053763`.

The first optimization should be **better connected atom orders**, alongside
removing s-store's unconditional schedule override and adding work-level
cancellation. A new synthetic ten-pattern chain exposes an avoidable Cartesian
product even under Adaptive: a connected-order candidate reduces node openings
from 434,341 to 2,341 and prepared traversal from 5.21 ms to 28.75 µs.
It uses the existing Nested executor, compiler, guards and compressed graph.
This is evidence for a planner change, without requiring a new storage format.

These are controlled local results, **not a reproduction of TEP3**. The actual
TEP3 queries, materialized graph and reference answers were not available in the
inspected workspace. In particular, E1/E3 shapes and their causes remain unknown.
No production engine, s-store source or existing FST-build work was changed.
The connected-order constructor exists only in the disposable benchmark crate.

## Evidence and ownership

Inputs reviewed: [optimization direction](../../../notes/REPORT_OPTIMIZATION_DIRECTION.md),
[external comparison](../../../notes/sstore-tep3-comparison.md),
[automata review](../../../notes/AUTOMATA_RESEARCH.md), the current executors and
interfaces, and the related investigations linked below. These documents supply
claims and hypotheses; their proposed work is not treated as a separate instruction.
Primary literature was checked independently where it informs the proposal.

| Report observation | Finding | Status / next evidence |
|---|---|---|
| cq02/E1/E3 exceed 600 s | s-store `src/sparql/patterns.rs` sets `SCHEDULE = Schedule::Levels` and compiles each single-partition BGP with it. This bypasses Adaptive entirely. Carry-chain controls reproduce exponential growth in work. | High confidence in the failure mechanism; query-specific attribution remains unverified, especially E1/E3. Obtain each BGP and plan. |
| cq08/E4/cq19 slow | No corresponding query text or operator traces. E4's 133 s batch versus 84 s idle demonstrates contamination, not an isolated engine cost. | Measure in fresh isolated workers; inspect BGPs, correlated invocations, filters and aggregation separately. |
| Abandoned queries continue | s-store's example starts a thread per connection, evaluates `store.query` synchronously, builds the response, then writes it. s-graph public visits check callback cancellation only when a result is emitted. | Confirmed code-level gap across both layers. Socket write failure cannot interrupt an earlier no-output search. |
| SIGINT ineffective | Neither a signal experiment nor the external process environment was available. | Unresolved; do not attribute it to FSTs or promise cancellation alone fixes it. |
| Load footprint 2.25 GB; max RSS 1.29 GB | The external report distinguishes macOS physical footprint from RSS. s-store's team attributes most transient allocation to graph construction. | Attribution is their measurement, not a fresh memory profile here. Separate parsing, dictionary, input/scratch, minimization and packing peaks. |
| `size_bytes()` varies | The store team measured arrival-order-dependent dictionary layout. | Keep separate from traversal planning; compare graph and dictionary bytes separately before changing graph layout. |
| cq24/cq25 unsupported function; cq11 empty | Report identifies function support and dataset literal issues. | No graph-engine optimization follows from these observations. Their semantic diagnoses were not independently rerun. |

The October 15 and 22 datasets differ by 20,463 triples, and later batch queries
competed with abandoned work. Thus the external run supports an observed storage
and load improvement, but is not a controlled attribution of every timing change.
Neither the claimed “hash join in tens of milliseconds” nor a TEP3 speedup is
established by the supplied evidence or the experiments here.

## What already exists

The history matters: [FORCED_LEVELS.md](FORCED_LEVELS.md) and
[LARGE_CYCLES.md](LARGE_CYCLES.md) include proposals subsequently implemented in
[AUTOMATA_FRONTIERS.md](AUTOMATA_FRONTIERS.md). The current source already:

- Compiles constants, repeated-variable equality, register writes and bound-bit
  guards. Whole-atom masks are cached within an atom invocation.
- Compares bounded Levels and Nested probes, including answers per estimated
  work when Levels produces rows. Large BGPs get a query-size-aware budget.
- Has explicit shared-variable-bit support propagation (`Schedule::Frontier`),
  a reversible trail, and advancement across unequal storage levels.
- Pushes existing complete bindings into RHS BGP traversals through
  `visit_matching`; supports streaming callback breaks and workspace reuse.
- Separates typed recognizers, bitmap masks, reversible actions, transducers,
  binding operators and path NFAs. There is no need for another universal
  automaton hierarchy or eager query DFA.

The gaps are in candidate quality, conditional costing, cheap navigation and
execution control. The public `PlanningStats` records only aggregate probe work
and the chosen order/cut/frontier flag, so it cannot explain why each candidate
was rejected. Existing counters also do not separate root reopenings, pruning,
cache hits and time spent in propagation.

## New measurements

[Runner](prototype/report_probe.py), [new fixture](prototype/report_probe.rs),
[summary](prototype/results/report-investigation/summary.json),
[all runs](prototype/results/report-investigation/runs.json), and
[source/binary manifest](prototype/results/report-investigation/manifest.json).

Two sequential passes on an AMD Ryzen 5 PRO 4650U, Rust 1.98.1,
`profiling` profile, default `heap` feature and block capacity 65,536.
Each phase uses five warm calibrated trials targeting 10 ms; tables use the
median of the two pass medians. Policy order is reversed between passes for
the new fixture. CPU is not pinned; short timings are approximate. Work counts
and complete answer comparisons are stronger evidence than small timing gaps.

Each child has a 2 GiB virtual-address limit, 30 CPU seconds and a 40 s parent
timeout. Diagnostic visits stop at one million node openings; Frontier also
uses its existing propagation/decision work cap. Timings are collected only
after a completed diagnostic, with counters and budget checks compiled out.
They are warm RAM measurements, not cold-cache or cross-backend results.

### Endpoint constraints expose a missing candidate

The graph has a directed ring, one label and one start tag per vertex, and a
type on every eighth vertex. The ten atoms are seven link hops plus:

```text
?x7 :type :T .
?x7 :label ?label .
?x0 :tag :A .
```

This is a full homomorphism query, including all intermediate variables.
Each typed endpoint determines one complete answer. A fixed bijection shuffles
vertex IDs in the second control; connectivity, constraints and answer counts
are preserved. This is a stress fixture inspired by the store team's description,
not reconstructed cq02 text.

| Vertices / IDs | Policy | Completed rows | Node openings | Prepared visit | Fresh total |
|---|---|---:|---:|---:|---:|
| 256 / ordered | Adaptive | 32 | 434,341 | 5,212 µs | 5,244 µs |
| 256 / ordered | Nested | 32 | 434,341 | 5,180 µs | 5,113 µs |
| 256 / ordered | Levels | 7; incomplete | 1,000,000 cap | — | — |
| 256 / ordered | Frontier | 32 | 8,278 | 1,437 µs | 1,442 µs |
| 256 / ordered | Connected Nested candidate | 32 | **2,341** | **28.75 µs** | **35.09 µs** |
| 256 / shuffled | Adaptive | 32 | 919,140 | 10,055 µs | 10,269 µs |
| 256 / shuffled | Connected Nested candidate | 32 | **4,730** | **57.64 µs** | **64.39 µs** |
| 1,024 / ordered | Adaptive | 17; incomplete | 1,000,000 cap | — | — |
| 1,024 / ordered | Connected Nested candidate | 128 | **10,149** | **123.76 µs** | **131.27 µs** |
| 1,024 / shuffled | Adaptive | 4; incomplete | 1,000,000 cap | — | — |
| 1,024 / shuffled | Connected Nested candidate | 128 | **37,953** | **485.44 µs** | **491.77 µs** |

The current order is `[7,9,6,5,4,3,2,0,1,8]`: endpoint type, start tag,
then links. The first two atoms share no variable. They introduce a product
of endpoint and start assignments before the connecting chain checks them.
The candidate is `[7,6,5,4,3,2,1,0,9,8]`: type, walk backward through all
links, check the start tag, emit the endpoint label.

In `scheduled_join.rs`, both small and large ordering heuristics rank constant
count before bound-variable overlap. The small-query `orders` function swaps
only the first atom with up to five others; it does not rebuild a connected
suffix from each seed. The good order is absent from those candidates.
The 960 planning reads in this fixture cannot discover an order never tried.
Its Frontier probe also cannot complete within the small probe allowance,
so Adaptive discards that alternative despite its lower full traversal work.

The candidate prioritizes any connected atom, then constants and overlap.
It is an **additional heuristic**, not a universal ordering rule: a small
independent relation can sometimes be worth evaluating first, and truly
disconnected queries require products. Its preparation currently includes a
normal Nested compilation followed by candidate compilation; fresh totals
include that extra cost. No decoded relation or answer cache is constructed.

At 256 vertices the node reduction is 185.5× ordered / 194.3× shuffled;
prepared speedups over Adaptive are about 181× / 174×. These results do not
predict the same factors on TEP3, whose fanout and selectivity are unknown.
The 1,024-vertex incomplete runs have no full-time estimate or speedup ratio.

### Retained controls explain why hybrid planning is necessary

| Control | Adaptive nodes | Levels nodes | Observation |
|---|---:|---:|---|
| Carry chain, 8 atoms, 16 answers | 353 | 8,145 | Completing atoms resolves arithmetic carries early. |
| Carry chain, 12 atoms, 16 answers | 545 | 131,025 | Levels grows about 16× for four extra atoms here. |
| Carry chain, 20 atoms, 16 answers | 929 | 1,000,000 cap; no rows | Tiny stored graph can still cause enormous search. |
| Uniform shared-object star, 4,950 answers | 1,220,372 | 1,700,957 | Adaptive/Nested: 18.13/18.18 ms; Levels: 27.36 ms. This is an object star, not the store's subject-star measurement. |
| Functional triangle, 20,000 answers | 134,295 | 134,295 | Adaptive selects Levels: 1.74 ms versus Nested's 4.28 ms. |
| Dense odd cycle, 33 atoms, one planted answer | 99 | 1,000,000 cap; no rows | Adaptive selects Frontier: about 19.74 µs. |
| XOR cycle, 32 atoms, 4,096 answers | 145,665 | 145,665 | Adaptive selects Levels: about 1.69 ms. |

The XOR control is particularly instructive: explicit Frontier reads only 18,720
nodes but takes 21.47 ms, over 12× Adaptive's time. Opening fewer nodes is useful
but does not price queue updates, bit decisions and rollback. With identical
XOR edges and an unused extra predicate-domain value, Adaptive instead needs
639,561 nodes / 10.35 ms; Levels hits the cap without answers. Physical digit
allocation and ID encoding belong in the cost model alongside logical shape.
The endpoint shuffle independently shows that effect without changing domains.

## Changes to prioritize

| Priority | Concrete change | Acceptance evidence |
|---|---|---|
| P0: reproducibility | Run cq02/E1/E3 separately on an idle worker; capture parsed algebra, each BGP, graph/term-domain sizes, parameters, chosen schedule and row counts. Record first row and completion separately. | Full multiset equals truth; no prior abandoned worker remains active. |
| P0: execution control | Add cancellation, deadline and work budget across planning and traversal, including no-output branches; propagate them from the store's request lifecycle. | A no-answer dense/carry case stops before its budget without invoking the result callback; reuse returns all expected answers. |
| P1: planner candidate | Add connected orders rebuilt from several promising seeds, not just first-atom swaps. Retain existing candidates and bounded probes. | Endpoint controls improve in node counts and fresh total; stars, skew, carry/cycles and large-BGP preparation remain competitive. |
| P1: store integration | Replace unconditional Levels with query-specific selection. Adaptive is the first comparison for chains, retaining measured overrides only for workloads where justified. | Compare explicit Levels/Nested/Frontier/Adaptive on real subject/object stars and chains; no global schedule replacement based on one fixture. |
| P2: conditional costing | Include overlap, estimated fanout, root reopenings, mask/propagation work, bound parameters and probe completion depth. Retain per-candidate diagnostics. | Selection moves toward the best explicit candidate across ID permutations and skew, with bounded preparation cost. |
| P2: acyclic reduction | Prototype selective semijoin propagation and query-local hash/adjacency projections with strict memory accounting. | Compare scan/build + evaluation + cleanup against compressed traversal; include ASK/LIMIT and tiny selective queries. |
| P3: navigation | Add exact conditional-domain/seek capabilities and optional specialized intersections where the source can support them efficiently. | Measure decoded values, physical reads, intersection work and peak scratch; keep bitmap navigation as the base specialization. |

Classify the **query hypergraph**, including predicate variables and repeated
variables, rather than just detecting `object → subject` syntax. Start with
components, shared-variable incidence and acyclicity (e.g. GYO/join-tree
construction). Stars and chains are both acyclic; cyclic cores can coexist with
selective acyclic branches. Shape proposes candidates; data and bound parameters
choose among them. Do not reorder across OPTIONAL, MINUS or other semantic
boundaries as if they were one conjunctive BGP.

For costly acyclic joins, propagate restrictions in both directions along a join
tree before enumeration. A chain endpoint restriction should reach the other
endpoint without enumerating every pair. For small selective requests, direct
connected probes may beat preprocessing. Query-local hash projections are a
valid competing implementation, but their materialization cost and memory must
be charged; the one-index FST representation remains valuable.

## Automata interface proposal

Keep `Automaton`, `BitmapAutomaton`, `BitmapSource`, `TripleSource`, `Compile`
and `Search`/`Plan` as the existing layers. Extend capabilities at the boundary
where query variables meet storage navigation:

| Primitive | Contract / implementation direction |
|---|---|
| `conditional_domain(atom, variable, binding)` | Logical values consistent with that atom's constants, repeated occurrences and currently bound variables. State explicitly whether support is exact or a conservative superset. Current exposed-bit `BitSupport` is local support, not an exact whole-value oracle. |
| `seek_ge(domain, value)` | Least supported logical value at or above the target, with distinct sorted output. Morton slot order is not generally logical S/P/O value order. Do not advertise logarithmic seek using a hidden full scan. |
| `intersect(domains)` | Intersect before branching: one-word Boolean support today; sorted sparse values with merge/galloping, or dense bitmaps when their construction and density justify them. Respect domain encoding and exactness. |
| `propagate(change)` | Update only affected incidences; advance singleton frontiers; trail mutations. An overapproximation may miss pruning but must never exclude an answer. |
| `enumerate` / `exists` / `count` / `project` | Make the requested result mode explicit to planning. Preserve full-binding multiplicities unless the caller requests distinct projection or endpoint-pair semantics. |

This is a design contract, not an implemented API or an assumption that any
backend supports cheap arbitrary-order domains. `PackedFst::select` already
returns a materialized sorted distinct coordinate projection; `scan` streams
triples; `count_matching` memoizes a fixed pattern's suffix counts. Reuse them
as baselines, while accounting for their work. None supplies a general cheap
binding-conditioned LTJ cursor today.

Planning should receive the result demand and the known-bound-variable mask.
s-store compiles variables once and passes values later, so a plan's constant
count does not describe a correlated invocation's actual selectivity. Start
with a few reusable variants keyed by bound positions, not a new plan for every
value. RHS probe caches must include the relevant parameters and graph epoch.

One additional store-specific constraint: `compile_bgp` distinguishes a variable
in predicate position from the same variable in subject/object position because
predicates use dense IDs. Logical equality is enforced when results are mapped
back to store terms. Such equality can therefore prune later than an internal
s-graph equality guard. Any domain interface must make ID translation explicit;
do not intersect dense predicate IDs directly with term IDs. This is a secondary
investigation target, not an explanation of cq02 without its query text.

### Execution control is separate from result callbacks

An additive conceptual interface could look like:

```rust,ignore
visit_controlled(workspace, control, emit)
    -> Completed | Stopped { reason, work, rows_emitted }
// reasons: caller break, cancellation, deadline, work budget
```

Specify stop reasons distinctly; a budgeted incomplete query is never an empty
answer certificate. Poll in scalar open/branch loops, macro descent, Frontier
propagation/decisions and compilation/probe loops. Include composed operators
and paths, using the same request control through every child call. Check
cancellation before compiling and before emitting a ground/empty-BGP result.
A counter-based polling interval can amortize atomic/clock reads; it bounds
work between checks, not wall time during arbitrary blocking I/O or callbacks.

Keep existing `visit` as the uninstrumented specialization. The current
`Stop::{User,Budget}` is crate-private and used for probes; it is a useful
implementation starting point, not a public deadline/cancellation API. Expose
controlled compilation as well as visitation, since a client can abandon work
before execution starts. Independent workspaces must remain reusable after
every stop and nested storage read sessions must unwind.

Runtime replanning also needs an explicit correctness boundary. Restarting a
different plan **before any output escapes** is straightforward. After output,
restarting from the root duplicates answers, including SPARQL bag multiplicities.
Prefer adapting within an unvisited branch or deliberately budgeting a
pre-output buffer; do not add a global distinct set as an implicit repair.

### Exploit compression only under valid residual equivalence

A shared physical suffix state is not a graph vertex and not a unique binding.
Memoization must include all residual query information that affects future
acceptance: source state(s), relevant bound bits/parameters, layout/schedule,
and source identity or epoch. Enumeration must replay accepted suffixes for
each incoming logical prefix. Counting must add each prefix's contribution;
a global visited set would undercount. ASK can use existential certificates.

The existing [composition investigation](AUTOMATON_COMPOSITION.md) already found
that a simple negative cache made skew queries slower (111 → 467 ms) while
saving only about 1.2% of reads. First fix repeated *unnecessary* probes; then
try bounded residual caches at demonstrated reuse points. Do not add a mandatory
hash lookup to every storage transition based only on FST suffix sharing.

## Literature implications and scientific claims

These links are primary papers or implementation documentation, checked during
this investigation. The engineering choices in the right column are our
inferences; the papers do not establish bounds for the current engine.

| Source | Relevant result and consequence |
|---|---|
| [Free Join, Wang–Willsey–Suciu, 2023](https://arxiv.org/pdf/2301.10841) | Unifies multiway and traditional joins, with lazy tries and vectorized execution. It supports investigating mixed granularity and selective materialization, rather than committing all BGPs to one traversal. |
| [Database Theory in Action: Yannakakis' Algorithm, 2026](https://arxiv.org/html/2601.00098v2) | Reviews acyclic join-tree semijoin reduction and its practical overhead, including bitvector filters and selective reduction. Use as the acyclic baseline; measure the reduction cost rather than assuming two extra passes always help. |
| [Worst-Case Optimal Radix Triejoin](https://arxiv.org/html/1912.12747) | Booleanization and interleaving permit a query-independent index. Its RTJ bound includes `2^n` in the number of query variables; variable-order conflicts can be grouped using SCCs. Investigate layout-induced expansion groups with a measured size cap rather than synchronizing every tuple digit. |
| [Optimal Joins Using Compact Data Structures, ICDT 2020](https://doi.org/10.4230/LIPIcs.ICDT.2020.21), [extended compressed-quadtree paper, 2022](https://doi.org/10.1145/3514231) | Demonstrates formal join evaluation over compact quadtrees/qdags and lazy relational composition. Direct compressed evaluation has substantial prior art; suffix sharing alone is not a new join complexity result. |
| [CompactLTJ, 2025](https://users.dcc.uchile.cl/~gnavarro/ps/vldbj25.pdf) | Compact navigation supports LTJ; conditional seeks and variable ordering are central. A trait named `seek` is insufficient: its supported orders, exactness and physical access cost must be established. |
| [OpenFst efficiency documentation](https://raw.githubusercontent.com/google-research/openfst/main/docs/efficiency.md) | Delayed composition, representation specialization, matching and cache policy are distinct choices. Keep the static bitmap fast path; avoid eager determinization and mandatory generic dispatch. |
| [Daciuk et al., 2000](https://aclanthology.org/J00-1002/) | Incremental acyclic minimization explains the storage-sharing foundation. Construction/minimality improvements and join search reductions require separate measurements. |
| [Time- and Space-Efficient RPQs, ICDE 2022](https://arxiv.org/abs/2111.04556) | Bit-parallel automata run against a compact ring index. Relevant to the path engine; its automaton-state sets and graph product differ from BGP register equality. |
| [Output-Sensitive RPQs, 2025](https://doi.org/10.1145/3725242), [acyclic CRPQs, ICDT 2026](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICDT.2026.18) | Distinct endpoint output and query projection matter. These motivate demand-aware path planning, not replacing a fixed chain that returns intermediate bindings with a distinct reachability relation. |

The current engine has no demonstrated AGM/WCO runtime proof. RTJ's
`O(2^n m w N^rho*)` bound concerns its specified algorithm, Booleanized variable
order and index assumptions, with `n` query variables, `m` atoms, `w` encoding
bits and maximum input-relation size `N`. It is not a bound in physical FST
states. Growing query size, data size, and compressed representation size are
different analyses. Grouping physical slots or using support masks does not
automatically transfer that theorem.

The measured connected-order improvement is useful engineering, not a novelty
claim. A stronger research contribution could be a residual-equivalence rule
with a correctness and cost argument, a conditional-domain primitive with a
proven compressed navigation bound, or a robust adaptive strategy with strong
baseline comparisons. Weighted automata, general SMT predicates, grammar
compression and new string codecs are lower priority for the reported joins.
Keep property-graph edge identity and multiplicities for a later explicit
semantic extension; do not silently impose injective subgraph matching now.

## Reproduction and next benchmark gate

```sh
python3 examples/join/prototype/report_probe.py --output target/report-investigation-new
```

The runner requires a fresh output directory, uses cached Cargo dependencies,
copies current production into a disposable crate, injects only diagnostic
constructors, runs independent fixture/oracle tests, and records source and
binary hashes. It builds outside timed phases, runs children sequentially,
checks deterministic work counts across passes, and verifies source hashes at
the end. Raw captures and logs are retained alongside the summary; the build
directory need not be archived. Limits use Linux/POSIX `resource` support.

Validation here: the chain oracle matches exhaustive assignments on small
ordered/shuffled graphs; all five schedules match its full rows there. Every
completed measured case compares complete sorted bindings with its analytical
or existing reference oracle; incomplete rows are checked against that oracle.
The two existing cycle oracle/domain-control tests also pass. No production
feature matrix was rerun because production sources were not modified.

Before promoting a planner or interface change, include real subject/object
stars, long chains, branched trees, cycles, hubs, empty joins, disconnected
components, repeated/predicate variables, bound/unbound RHS variants, unequal
domains and multiple ID permutations. Include 4–4,096 atoms to catch query-size
preparation regressions and all supported storage backends for navigation work.
Measure preparation, first row, full output, physical reads, candidate/propagation
work, decoded values and peak/retained scratch separately. Compare with explicit
schedules, selective index nested loops, a query-local hash plan including build,
and semijoin reduction; use CompactLTJ/Free Join as stronger follow-up baselines
where integration permits. ASK, COUNT and DISTINCT require their own correctness
and multiplicity checks. An interrupted run must remain marked incomplete.
