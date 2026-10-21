# Binding operators and compositional graph paths

The disposable prototype now implements binding-aware joins, optional matches,
SPARQL-style MINUS, existence filters, multiset union, and regular-path relations
over the compressed FST. Production modules remain unchanged. Run the code via
[`prototype/prepare.py`](prototype/prepare.py), as with the other join prototypes.

## Two complementary interfaces

[`query_ops.rs`](prototype/query_ops.rs) defines `Plan`: a streaming relation of
partial bindings. A binding has a value array and a separate bound-variable
bitmap. Zero is an ordinary ID, and an unbound variable is distinct from zero.
All children of an operator share a global variable namespace and width.

`Bgp` adapts the existing `ScheduledJoin`. It maps the variables actually used
by each BGP to dense local registers, then maps outputs back to the global
schema. A leaf may leave most global variables unbound. The empty BGP emits
one empty mapping. Duplicate input mappings are retained by the result
operators; identical stored triples retain the graph's existing set semantics.
Keep connected inner-join patterns inside a single BGP when possible: it retains
global ordering and prefix-synchronization choices. `Join` composes separate
plans; this prototype does not automatically flatten/reorder arbitrary operator
trees across OPTIONAL or negation boundaries. The benchmark includes a combined
BGP control for this reason.

Language `And`/`Or`/`Not` still recognize the same input word. Binding `Join`,
`UnionAll`, `LeftJoin`, `AntiJoin`, and `Exists` compose relations and preserve
multiplicities. In particular, Boolean `Or` must not replace `UnionAll`.
Recognition and relation execution remain separate responsibilities within the
compositional architecture.

## Operators and pseudocode

`Join` emits every compatible pair of bindings merged together. Compatibility
means equality on variables bound by both mappings. Merging never substitutes
an ID for an unbound value.

`LeftJoin::new(left, right)` implements OPTIONAL without an extra condition.
The `LeftJoin { left, right, condition }` form accepts a Boolean predicate over
the merged mapping. The caller supplies expression evaluation; this integer
engine does not define RDF expression types or SPARQL expression errors.

```text
left_join(left, right, condition):
    for each occurrence of left binding a:
        matched = false
        for each independently evaluated right binding b compatible with a:
            merged = merge(a, b)
            if condition(merged):
                matched = true
                emit merged
        if not matched:
            emit a                         # right-only variables stay unbound
```

The flag is per left-row occurrence. Rejected matches do not set it, and a
later rejected match does not undo an earlier accepted match. Cancellation
propagates immediately and does not produce a fallback after cancellation.

`AntiJoin`, also exported as `Minus`, uses SPARQL's extra domain-overlap rule:

```text
minus(left, right):
    for each occurrence of left binding a:
        if RHS possible-variable metadata proves no overlap with dom(a):
            emit a
            continue
        if no right binding b exists such that
           compatible(a, b) and intersection(dom(a), dom(b)) is nonempty:
            emit a
```

Thus an empty mapping or a mapping with disjoint variables cannot remove a left
mapping. Surviving left multiplicities are unchanged. Metadata is conservative:
unknown schemas use the normal probe, never an unsafe shortcut.

`Exists` and `NotExists` test whether a compatible right result exists, without
the MINUS domain-overlap requirement. They emit only the original left mapping,
and stop the RHS at the first qualifying result. `CorrelatedFilter` explicitly
receives both the outer binding and the RHS binding. Ordinary `Filter` sees only
its input mapping. This prevents an indexing hint from changing filter scope.

`UnionAll` visits the left stream followed by the right stream, preserving every
occurrence. It neither deduplicates results nor forces equal binding domains.

These are execution primitives following the relevant
[SPARQL algebra definitions](https://www.w3.org/TR/sparql11-query/#sparqlAlgebra),
not a SPARQL parser or a claim of complete language conformance.

## Physical probes preserve logical scope

`Plan::visit_matching(hint)` emits a subset of the plan's independent results.
It **does not merge the hint into those results**. Its default implementation
filters the ordinary stream by compatibility. This is always correct, including
for nested optionals and filters, though it can be expensive.

The BGP adapter overrides this with `ScheduledJoin::visit_matching`. It builds
parameter masks over first-writer register bits and intersects those masks during
storage traversal. Existing binding checks, writer ownership, and undo remain
in force. Parameters outside the representable variable width fail early.

```text
parameterized_bgp_visit(parameters):
    for each compiled stage:
        parameter_mask = allowed slots for its externally bound first-writer bits
    reject any parameter with nonzero bits excluded by all its occurrences
    execute the existing schedule:
        candidates = source_bitmap AND static_guard AND binding_guard
                     AND parameter_mask
    emit the BGP's own bindings, without adding outer-only variables
```

Masks and represented-bit buffers are reused after warmup. No query compilation,
materialized RHS relation, or per-node hash table is introduced by these probes.
Unparameterized visits use a separate const-generic specialization with the
parameter-mask operation compiled out.

Do not blindly push all hints through `LeftJoin`: if the RHS actually matches
`y=9`, an outer lookup for `y=8` must not invent an unmatched left fallback. The
nested OPTIONAL test covers exactly this case. Similarly, MINUS RHS filters
retain their independent scope; correlated filters use the explicit existence
interface instead.

## Paths over the FST

[`path_ops.rs`](prototype/path_ops.rs) adds `PatternIndex<TripleSource>`, a
`PatternBackend` implementation with exact subject/predicate/object bounds. It
retains the source's per-level layout and computes lookup masks once per scan.
Its iterator uses fixed stack frames for at most 64 physical levels and does
not allocate a vector for each adjacency lookup. Coordinates belong to traversal
occurrences, so shared FST suffixes do not collapse distinct triples.

`PathExpr` builds sequences, alternatives, optional steps, `star`, `plus`,
inverse edges, wildcard steps, epsilon, and reversal of whole expressions.
Compilation uses Thompson construction and epsilon elimination, producing the
existing packed `PathAutomaton`. Defaults bound preparation to 4,096 states,
1,000,000 emitted transitions, and nesting depth 256; the state and transition
budgets are configurable. There is no regex/SPARQL text parser.

```rust
let expr = PathExpr::label(0)
    .then(PathExpr::inverse(1).or(PathExpr::label(2)).plus());
let machine = expr.compile()?;
let index = PatternIndex::new(FstSource(&tree))?;
let path = PathBindings::new(
    &index, &machine, vertices, Term::Var(0), Term::Var(1), 2,
    HopRange::new(0, u32::MAX)?,
)?;
```

`PathBindings` is a `Plan`, so it can be the RHS of `LeftJoin`, `Minus`, or an
existence test, or participate in joins and unions. An unbound start enumerates
an explicit, deduplicated vertex universe supplied by the caller. A bound-start
lookup restricts that universe. A constant start follows the existing traversal
contract: it is considered valid even when isolated, enabling zero-hop queries.
Repeated endpoint variables require equal source/target values.

The existing packed runner visits `(vertex, NFA state)` pairs, pushes transition
predicates and directions into adjacency scans, and emits distinct endpoints
with their first qualifying length. Visitation before the minimum hop count is
layer-local, so revisits needed to meet a lower bound are preserved.

## Executing general language compositions

`PathAutomaton` also implements the shared typed `Automaton<PathEvent>` contract.
Its language state is a canonical NFA-state set. Automata can be borrowed when
composed, avoiding copies of compiled programs.
Programs with at most 64 states keep that set in an inline bitmap; larger
programs fall back to a sorted sparse vector. This removes state-vector
allocations from small generic path compositions.

`Composed(machine)` supplies a generic product runner for shared `And`, `Or`,
and `Not` compositions. It retains the full product state in visitation keys,
scans outgoing and incoming edges conservatively, and uses explicit final-state
acceptance. Complement's missing-transition sink remains live.

```rust
let prefix = PathExpr::label(0).then(PathExpr::label(1).star()).compile()?;
let one_step = PathExpr::label(0).compile()?;
let machine = Composed(And(&prefix, Not(&one_step)));
// Pass &machine to PathBindings::new just like a packed NFA.
```

This recognizes `p0/p1+` using shared language combinators. For a negated edge
label set, constrain complement to one step, for example
`And(any_one_edge, Not(excluded_one_edge_alternatives))`. Unconstrained language
complement is not a negated predicate set, MINUS, or NOT EXISTS.

Generic composition is correct but costs more: canonical state sets may
allocate, and conservative expansion does not push individual label constraints
into storage. The packed NFA runner remains the preferred specialization where
the expression can already be compiled into it.

**Path results use distinct endpoint-pair reachability semantics.** They do not
enumerate all walks or preserve distinct intermediate witnesses. In particular,
this adapter does not implement every SPARQL property-path multiplicity rule:
fixed-length sequences translated to BGPs can retain duplicate witnesses. Keep
that lowering in the query compiler when those semantics are required.

## Validation and reproduction

Tests cover partial and empty mappings, zero IDs, duplicates and multiplication
of multiplicities, 65/130-variable masks, nested optional probes, optional
conditions, independent versus correlated filters, disjoint MINUS schema
shortcuts, cancellation/reuse, parameterized BGPs under all schedules, full-width
IDs, exact FST scans, path reversal and inverse edges, epsilon/cycles, bounded
walks, repeated endpoints, and actual execution of composed path languages.

The independent relation reference uses Boolean relational composition and
transitive closure; path endpoints are also checked against the existing heap
backend. Binding operators are checked against a separate multiset evaluator.
The existing huge-BGP and analytics tests continue to run.

```bash
mkdir -p target/automata-tmp
export TMPDIR="$PWD/target/automata-tmp"
export CARGO_TARGET_DIR="$PWD/target/automata-prototype"
export PROTO_DIAGNOSTICS=0
snapshot="$(python3 examples/join/prototype/prepare.py)"
cargo test --offline --profile profiling --manifest-path "$snapshot/Cargo.toml" --lib --tests
cargo build --offline --profile profiling --manifest-path "$snapshot/Cargo.toml" \
  --example query_ops_bench --example composition_bench
"$CARGO_TARGET_DIR/profiling/examples/query_ops_bench"
```

The benchmark validates complete result multisets before timing. It compares the
same operators with parameterized BGP probes against the correct default
full-enumeration probe. It reports preparation, warmed execution, and fresh-plan
execution separately, excluding the already-existing graph build. Path rows
report only warmed execution; blank columns mean unmeasured, not zero cost.

The shared FIFO-controlled perf harness also supports this benchmark:

```bash
python3 examples/join/prototype/analytics_perf.py \
  "$CARGO_TARGET_DIR/profiling/examples/query_ops_bench" \
  --backends parameterized,fullscan --algorithms optional,minus-disjoint \
  --seconds 1 --output examples/join/prototype/results/query-ops-perf
```

Graph construction, query preparation, and warmup precede counter enablement;
only complete warmed visits are profiled.

## Measured results

All **110 tests pass**: 94 library tests plus 16 integration tests. This includes the earlier analytics and huge-BGP cases. The tests use independent reference evaluators and validate complete output multisets/vectors rather than only row counts.

The synthetic join fixture stores 100,000 class triples and 991 sparse edge triples. The left pattern selects 5,000 subjects; 50 have a compatible sparse edge. The following totals include fresh BGP preparation and fresh workspace, in milliseconds. The full-scan column uses the same result combinators and the correct default compatibility probe, without parameter pushdown or schema metadata. It is an explicit reference strategy, not a pre-existing production OPTIONAL implementation.

| Operator | Rows | Parameterized / metadata | Full RHS enumeration |
|---|---:|---:|---:|
| join | 50 | 1.814 | 272.450 |
| optional | 5000 | 1.870 | 272.804 |
| minus | 4950 | 1.749 | 294.483 |
| minus-disjoint | 5000 | 0.208 | 296.574 |
| exists | 50 | 1.827 | 283.956 |
| not-exists | 4950 | 1.825 | 284.721 |

The combined BGP control takes 1.443 ms total, compared with 1.814 ms for composing two separate BGP plans. Keep the global BGP compiler in the architecture: operator composition does not replace its scheduling choices. Streaming UnionAll emits 5,991 rows in about 0.258 ms total.

The bounded path fixture has 4,096 vertices and three labelled outgoing edges per vertex. All runners recognize the same `p0/p1+` language with a maximum length of 16 and return the same 15 endpoint/length pairs. These are warmed execution medians in microseconds, excluding path compilation and graph/index construction:

| Runner | FST adapter | Existing heap backend |
|---|---:|---:|
| Packed NFA | 5.19 | 12.92 |
| Generic And/Not composition | 16.27 | 50.77 |

Inline state sets remove small-program state-vector allocations, but the generic runner remains slower than the packed NFA on this fixture. It explores both directions conservatively and hashes composite states. The packed runner retains per-transition label/direction pushdown. These are complementary execution strategies over the same language semantics.

The existing uniform-star control still returns exactly 4,950 bindings with the same storage-read/candidate counts: Scheduled is about 19.9 ms, AutomatonJoin about 20.0 ms, and FstJoin about 34.8 ms. This checks the unparameterized specialization after adding parameter masks.

### Perf interpretation

- `optional`: 24.06 million instructions per visit versus 4053.27 million; 7.39 million cycles versus 1082.68 million.
- `minus-disjoint`: 3.10 million instructions per visit versus 4517.37 million; 0.83 million cycles versus 1202.46 million.

Counter values are medians of three FIFO-gated trials, normalized per complete visit; all four counters report 100% running time. Sampling shows that full enumeration spends its time descending the RHS and reading FST words/deltas. Parameterized OPTIONAL moves compatibility checks into traversal guards, avoiding almost all repeated enumeration. Schema-aware disjoint MINUS removes the RHS traversal altogether; a counter-based unit test also verifies zero RHS visits.

The remaining parameterized OPTIONAL profile includes the parameter-mask setup and ordinary compressed descent; setup is not a hidden precomputed relation. There is no dynamic trait-object dispatcher in the result or path kernels.

Artifacts: [benchmark CSV](prototype/results/query-ops-comparison.csv), [perf summary](prototype/results/query-ops-perf/summary.json), [unparameterized join control](prototype/results/query-ops-join-control.csv), [test log](prototype/results/query-ops-tests.log), and [source manifest](prototype/results/query-ops-manifest.json). Timing columns are separately measured medians and need not sum to the total.
