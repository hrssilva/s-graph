# Connected orders and cooperative query execution

This implements the first query-engine steps from
[the report investigation](REPORT_INVESTIGATION.md). Graph storage, state
encoding, update algorithms and snapshot formats are unchanged. It introduces
neither new persistent indexes nor relation materialization.

## Planning

Adaptive retains the existing Levels/Nested/prefix/Frontier choices. If its
heuristic order introduces a disconnected variable assignment after starting a
component, it also builds a connected order. The existing incidence heap is
reused with a different priority: connected atoms first, then constant count,
bound-variable overlap and deterministic original ordinal. It includes predicate
variables and repeated occurrences without assuming subject/object chains.

All legacy orders remain candidates; at most one new bounded probe is added.
Already-connected orders do not build a second incidence heap or add a probe.
Small-query planning has at most 1,088 node reads; large-query planning at most
98,304. These are probe limits, not wall-clock limits or WCO guarantees. The
existing early completion/emptiness checks, strong-anchor shortcut and
early explosive-probe shortcut remain. Explicit policies are unchanged.

The new candidate avoids the demonstrated endpoint Cartesian product while
retaining the same compiler, guards and executor. It does not automatically
choose Nested for every chain or Levels for every star. The cost model remains
a bounded heuristic; the first few sampled assignments need not represent the
whole graph. No runtime restart or duplicate suppression is introduced.

## API contracts

The public additions are available from `automata`:

| Addition | Purpose |
|---|---|
| `Execution` | Per-request work allowance, optional deadline and borrowed atomic cancellation flag. Shared by reference across preparation and nested execution. |
| `ControlledCompile` | Optional builder extension: `compile_controlled(source, control)`. Returns a normal reusable plan, or `PreparationError::{Invalid, Interrupted}`. |
| `ControlledSearch` | Alias of `query_ops::ControlledPlan`, adding `visit_controlled` and `visit_matching_controlled`. |
| `ControlledCorrelatedPlan` | Fallible correlated probe, distinguishing interruption from no match. |
| `VisitResult<T>` | `ControlFlow<VisitStop<T>>`: completion, callback `User(T)`, or `Interrupted(StopReason)`. |
| `StopReason` | `Cancelled`, `Deadline`, or `WorkBudget`. |

Ordinary `Compile`, `Search`/`Plan`, `Automaton` and storage-source traits have
no new required methods. Custom algorithms remain compatible and can opt in
explicitly. A default callback wrapper would falsely promise interruption of
algorithms that never emit; for that reason controlled search is a separate
capability. Built-in BGP, VALUES and slot-language plans implement it, together
with JOIN, OPTIONAL/conditional OPTIONAL, UNION ALL, MINUS, filters and
EXISTS/NOT EXISTS. `CorrelatedFilter` supports controlled probes as well.
Compositions require controlled support from their constituent searches.

Use `machine.compile_controlled(graph.source(), &request)` for packed or editable
FSTs, or any compatible `TripleSource`. The low-level `ScheduledJoin` exposes
`with_source_controlled`, `visit_controlled` and `visit_matching_controlled`.
`automata_core::visit_language_controlled` runs a custom bitmap recognizer over
any `BitmapSource`, with reusable path scratch.

`Execution` borrows an `AtomicBool`, which may live in an `Arc` for cancellation
from another thread. Its local `Cell` accounting stays on the executing worker;
plans remain shareable with independent workspaces and independent controls.
A stop is sticky and work is cumulative. Allocate a fresh `Execution` for a new
request; reuse the workspace rather than resetting a stopped request object.
`with_deadline(Instant)` accepts a monotonic absolute deadline.

Every checkpoint spends one unit. Attempting another unit after the allowance
is consumed stops before that unit is performed; finishing exactly at the
allowance is still completion. The counter is not a stable estimate of CPU time
or physical reads. Compare `ScheduleStats` separately for traversal diagnostics.
External flags/deadlines are polled at entry, on successful executor return,
and every 256 checkpoints. `check()` polls immediately without spending work.
No checkpoint can preempt an individual blocking source operation, allocation,
recognizer/filter operation or user callback. Budget zero prevents even a
ground/empty-BGP answer because executors checkpoint before starting work.

Compiler validation, ordering, stage construction and adaptive probes share the
request control. A planning probe hitting its own internal node limit merely
ends that trial; hitting the request limit aborts preparation. An interrupted
probe cannot become an empty-result certificate.

Traversal checkpoints cover scalar branches, macro node/candidate traversal,
parameter setup, frontier initialization, propagation, incidence updates,
decisions and trail restoration. Static specialization through a crate-private
no-op checker removes these checks from ordinary calls. There is no boxed
dispatch, cancellation flag, counter or clock stored in a compiled plan or graph.
Public opt-in traits remain small; execution-limit mechanics are separate from
recognition, variable guards and storage layout.

Stopping can leave scratch populated. The next invocation resets it; source
read-session guards unwind on all stops. Callback breaks retain their values.
Interrupted RHS probes propagate immediately: OPTIONAL cannot emit a fallback,
MINUS cannot retain a row, and NOT EXISTS cannot report absence based on partial
work. Existing bag and partial-binding semantics are preserved.

Regular paths and analytics retain their existing callback-based APIs and do
not implement `ControlledSearch` in this change. Their generic backend iterators
can do unbounded work inside `next()`; polling only around emitted edges would
misrepresent that guarantee. A future controlled path capability must include
cooperative backend navigation. Custom algorithms may implement the controlled
traits directly when they can check their own navigation loops.

## Store integration

s-store must opt in for the request to benefit. The current sibling checkout
still pins every BGP to Levels and does not pass a cancellation signal. It was
inspected but not modified by this s-graph change.

The integration should compile the machine with request limits, use controlled
matching when outer bindings exist, and propagate interruption through every
operator to the HTTP response. Do not handle `Interrupted` as empty results or
OPTIONAL/negation fallback. The request owner must set the atomic flag on
disconnect or server timeout; a failed write after synchronous evaluation is
too late. If compilation is cached, apply execution controls on every request;
if compilation occurs during a request, control it too.

Replace the global Levels override with per-query selection. Compare the new
Adaptive candidate on real chains, keeping narrowly measured star overrides
where justified until those workloads have been validated. This change cannot
improve a store BGP that still explicitly requests Levels. Reordering triples
in SPARQL text is also insufficient: the store's graph planner selects its own
atom order.

## Validation and performance

The regression test compares complete answers for ten- and seventy-atom chains,
ordered and bijectively relabeled IDs, and checks work reductions. Controlled
tests cover all schedules, no-output cancellation, planning-probe interruption,
budget/deadline/pre-cancellation, read-session cleanup, parameterized reuse,
partial bindings, bags, correlated operators, custom recognizers/searches, and
packed/editable sources across enabled backends.

The paired benchmark runner accepts a previous example directory:

```sh
python3 examples/join/prototype/report_probe.py \
  --output target/query-controls-comparison \
  --baseline-binaries /path/to/previous/build/profiling/examples
```

It alternates baseline/current process order, preserves individual captures,
and records source/binary hashes. `controlled` uses the same Adaptive plan as
`adaptive` and times traversal with a live cancellation flag plus work counter
and budget. Preparation is ordinary compilation in both modes, to isolate
execution overhead. Timed traversal does not copy rows or gather diagnostics.
Incomplete diagnostic cases receive no full-runtime estimate.

### Paired results

[Summary](prototype/results/query-controls/summary.json),
[individual runs](prototype/results/query-controls/runs.json),
[source/binary manifest](prototype/results/query-controls/manifest.json),
[complete snapshot source hashes](prototype/results/query-controls/snapshot-sources-sha256.json),
and [validation commands/logs](prototype/results/query-controls/validation/commands.json).
The baseline binaries are the preserved pre-change investigation build; their
hashes and original source manifest are embedded in the new manifest.

192 sequential child processes, two alternating passes, Ryzen 5 PRO 4650U,
Rust 1.98.1, profiling profile, heap backend, capacity 65,536; no CPU pinning.
Five warm calibrated trials per phase target 10 ms; reported values are the
median of the two pass medians. Both BFS and Pow2 layouts are compared on the
existing controls; the new endpoint fixture uses BFS. Construction and answer
validation are outside timed phases. The 2 GiB address-space, 30 CPU-second and
40 s process limits apply to each child. No process timed out.

| Chain fixture | Old Adaptive nodes | New Adaptive nodes | Old → new prepared visit | Old → new fresh total |
|---|---:|---:|---:|---:|
| 256 ordered vertices | 434,341 | **2,341** | 5,128 → **29.49 µs** | 5,214 → **66.62 µs** |
| 256 shuffled vertices | 919,140 | **4,730** | 10,108 → **57.81 µs** | 10,143 → **95.48 µs** |
| 1,024 ordered vertices | 1,000,000 cap; incomplete | **10,149** | — → **129.72 µs** | — → **166.00 µs** |
| 1,024 shuffled vertices | 1,000,000 cap; incomplete | **37,953** | — → **481.09 µs** | — → **524.08 µs** |

Every completed run checks full bindings against the fixture oracle. At 256
vertices, prepared traversal improves about 174–175× and fresh total about
78–106×. These are synthetic controls, not TEP3 speedup predictions. For the
larger fixture, the old capped visits are incomplete and have no full-runtime
estimate. New Adaptive selects the connected candidate in all four cases.

Opt-in control traversal takes 34.64/71.83/150.44/587.07 µs on those four cases,
about **16–24%** above ordinary new Adaptive. This measures a live atomic
cancellation flag and work budget, not a deadline clock in every checkpoint.
Its preparation phase remains ordinary compilation to isolate execution cost.
The additional cost is explicit and optional.

For 51 existing non-endpoint control configurations, baseline/current node and
candidate counts, rows, chosen order/cut/frontier, planning reads and retained
plan/workspace allocations are identical. BFS prepared visits include
18.21 → 17.90 ms for the uniform object star, 1.75 → 1.76 ms for the functional
triangle, and 19.86 → 19.92 µs for the dense odd cycle. The 4,096-atom chain and
1,024-atom star fresh totals are 6.65 → 6.84 ms and 1.92 → 1.96 ms in this run.
These small timing differences do not establish a universal no-overhead result.

The short run contains a preparation outlier on the empty 4,097-atom cycle:
one current pass measured 7.90 ms, the other 4.82 ms. A separate
[longer follow-up](prototype/results/query-controls/large-cycle-followup/summary.json)
uses the **same binaries**, three alternating process passes and seven 50-ms
trials per phase, Adaptive only. BFS baseline/current preparation is 7.35/7.37
ms, prepared visit 3.153/3.151 ms and fresh total 10.63/10.51 ms. Absolute times
shifted in the longer run; no frequency control was applied. The isolated
preparation regression does not reproduce. Raw outliers remain preserved.

Validation passed all five Cargo feature checks, the default/all-feature
production suites, eight README doctests in both modes, and package verification.
After refining entry checks for a request reused between compilation and
execution, the feature matrix, ten controlled-execution regressions in default
and all-feature modes, doctests and package verification passed again. Logs
distinguish the full-suite pass from that final focused revalidation.
