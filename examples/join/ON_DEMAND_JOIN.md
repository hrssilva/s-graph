# On-demand automaton product

Implemented follow-up: [scheduled compositional automata](SCHEDULED_AUTOMATA.md)
combines whole-atom and prefix execution, bounded planning, and lazy compiled masks.

Follow-up: [paired perf profiles against AutomatonJoin](PERF_SHAPES.md) explain
where each traversal wins and loses, including per-atom work counts.

[`prototype/demand_join.rs`](prototype/demand_join.rs) implements an on-demand,
factored product of the graph FST and a query binding automaton. It keeps
FstJoin's level-by-level pruning, compiles only query metadata, and expands graph
transitions during visitation. It never prepares per-atom lifted relations or a
compressed answer. Production executors and storage remain unchanged.

## What changed from the eager prototype

The previous [lifted prototype](PROTOTYPE_RESULTS.md) converts each matching atom
relation into a common-variable decision DAG before traversal. That conversion
can process large relations even when the join rejects nearly everything.

This prototype keeps each relation implicit in the existing FST. Its product
state consists of:

- A physical FST position for each atom.
- The stage `(level, atom)` currently consuming a transition.
- The variable bits already assigned by earlier transitions.

The event alphabet is `(level, atom, physical slot)`. Each event both advances
one atom and contributes constraints/assignments to the common query variables.
Different atoms can map a shared variable to different physical dimensions.
Their slot numbers consequently need not be equal: their **logical variable
bits** must agree. This avoids translating entire relations into a new alphabet.

The binding machine is a guarded register automaton represented by masks and
bit writes, rather than an explicitly expanded DFA over every possible binding.
Its finite state space remains implicit. Constants, equalities within a slot,
known high zero bits, and cross-stage equality are compiled once. The product
accepts only after every atom consumes its final level.

This is the same search algorithm and atom ordering as FstJoin. It deliberately
uses the existing `fst_join::compile` for shared semantics. The contribution is
a demand-driven automaton execution shape with a smaller representation, not a
new pruning rule or a new query planner. It is also **not** on-demand conversion
into the previous ordered Boolean decision DAG: the product stays factored.

## Pseudocode

```text
compile_query(query, graph_schedule):
    stages = compile_constants_and_variable_bit_ownership(query, graph_schedule)
    # Stages contain constant/local-equality masks, bound-bit checks,
    # owned-bit writes, and whether the physical atom has another level.
    return stages                     # No graph nodes have been read.

open_product_state(depth):
    stage = stages[depth]
    address = graph.root if stage.level == 0
              else stack[depth - atom_count].child_address
    word = graph.read_word(address)
    mask = stage.constant_and_local_equality_mask
    for check in stage.bound_bit_checks:
        mask &= slots_whose_bit_equals(bindings[check.variable] & check.value_bit)
    remaining = word & mask
    return State(address, word, remaining)

visit(query_machine, graph, emit):
    if query is statically impossible: return
    bindings = zeroes(variable_count)
    if query has no atoms: emit(bindings); return
    depth = 0
    stack[depth] = open_product_state(depth)

    loop:
        if stack[depth].remaining is empty:
            if depth == 0: return
            depth -= 1
            undo_bit_writes(stages[depth], bindings)
            continue

        slot = remove_lowest_transition(stack[depth].remaining)
        apply_bit_writes(stages[depth], slot, bindings)

        if this atom has another physical level:
            rank = popcount(stack[depth].word below slot)
            stack[depth].child_address = stack[depth].address
                                      + graph.read_delta(stack[depth].address, rank)

        if depth is the final stage:
            if emit(bindings) requests cancellation: return
            undo_bit_writes(stages[depth], bindings)
        else:
            depth += 1
            stack[depth] = open_product_state(depth)
```

The compiler assigns each variable bit exactly one writer. Later occurrences
check that bit; equal occurrences within the same slot are filtered statically.
Undo clears only the current stage's owned bits, preserving earlier bindings.
For unequal dimension widths, high bits known to be zero are fixed before
ownership assignment. The graph's actual paths enforce non-power-of-two domains.

The stack holds the parent position separately from its chosen child, so sibling
backtracking cannot overwrite the parent's address. Each visit resets bindings
and reached stack entries; cancellation leaves no state that affects the next
visit. Independent workspaces permit independent traversals of a borrowed plan.

## Implementation choices

Each live frame is four 64-bit words: node address, original bitmap, remaining
transitions, and chosen child address. FstJoin's frame additionally carries two
three-coordinate lower-bound arrays. This executor needs only addresses and
logical binding bits, so it does not calculate or carry those physical prefixes.
A frame therefore takes **32 bytes instead of 80 bytes** on this target.

`demand` applies the individual guard masks during traversal. An optional
`demand-table` control precompiles all at-most-64 possible guard masks per stage,
then packs the current checked bits to index the table. Both enumerate exactly
the same product transitions. Dispatch between them and
optional diagnostic counters occurs outside the hot loop using const generics.
There is no hash table, state interning, relation materialization, or answer cache
on either path. A warmed workspace performs no allocation during visitation.

The default uses `O(atoms × height)` transition metadata, plus the shared
compiler's variable metadata, independent of graph cardinality. The optional
tables add at most 64 masks per stage. Traversal workspace is
`O(atoms × height + variables)`. These are representation bounds, not a claim
of worst-case-optimal join complexity. The shared compiler also performs atom
validation, deduplication, and ordering; no preparation work reads graph nodes.

## Validation and measurement

The harness checks complete sorted bindings against FstJoin on every benchmark;
skew, dense-star, and contradictory-cycle fixtures also have independent source
oracles. Both product variants run through the existing brute-force fixtures and
48 randomized queries in both atom orders. Tests verify identical node-read and
candidate counts to FstJoin, cancellation followed by reuse, repeated variables,
shared predicates, disconnected/ground/duplicate/empty queries, unused/invalid
variables, non-power-of-two and unequal domains, 32-bit IDs, and 72 atoms/variables.

The current FST builder represents height-zero domains (`[1, 1, 1]`) as empty even
if supplied `(0, 0, 0)`. This pre-existing storage limitation is not repaired here;
the prototype follows the stored graph. Singleton individual dimensions with
nonzero overall height are covered separately.

Timings use Rust 1.98.1, the `profiling` profile, the default CPU target, one
thread on a Ryzen 5 PRO 4650U, a warmup, and seven repetitions with adaptive
batching. Graph construction is excluded; fresh plan and workspace construction
and destruction are included in total time. Prepared visits reuse only metadata
and workspace: the full traversal executes again. Runs are sequential and use
synthetic data, so these measurements do not establish a universal ranking.

**81 tests pass** (70 existing, eight earlier prototype tests, three new tests).
The demand variants also extend the earlier enumeration tests with complete
binding and pruning-count comparisons.

## Results

These are measured **total times**, including fresh preparation and workspace.
The default on-demand variant evaluates guard masks directly.

Large-case units: **milliseconds**.

| Case | Existing AutomatonJoin | FstJoin | RecursiveJoin | On-demand product |
|---|---:|---:|---:|---:|
| uniform-star | 21.06 | 37.38 | 45.24 | 29.82 |
| uniform-triangle | 16.20 | 31.47 | 35.51 | 27.83 |
| skew | 116.03 | 12.30 | 13.59 | 10.51 |

Small-case units: **microseconds**.

| Case | Existing AutomatonJoin | FstJoin | RecursiveJoin | On-demand product |
|---|---:|---:|---:|---:|
| uniform-anchored | 2.56 | 97.10 | 125.82 | 78.18 |
| parity-triangle-32 | 4,752.31 | 1.17 | 1.20 | 1.31 |
| parity-triangle-64 | 35,979.11 | 1.17 | 1.22 | 1.31 |
| parity-triangle-128 | 279,756.57 | 1.20 | 1.25 | 1.32 |
| dense-star | 79.38 | 83.21 | 134.50 | 78.00 |

Uniform star emits 4,950 bindings, skew 1,053, dense-star 16,384; the other cases
are empty. The on-demand product is about 20% faster than FstJoin on uniform star,
12% on uniform triangle, and 15% on skew in this run. Prepared parity traversal
is faster than FstJoin, but its extra preparation makes total time roughly
0.13–0.15 µs slower. This meets the target of comparable FstJoin behavior while
avoiding AutomatonJoin's regressions.

Whole-atom AutomatonJoin remains faster on the uniform star/triangle and sharply
anchored query. On the anchored case, on-demand eliminates the earlier eager
prototype's ~80 ms conversion, but still takes 78 µs versus AutomatonJoin's
2.56 µs. Synchronizing every atom at every level preserves FstJoin's search;
it does not obtain the whole-atom executor's early complete binding for free.
The earlier manually reordered skew join also remains faster (~5.5 ms; see
[the investigation](INVESTIGATION.md)), so this is not a replacement for planning.

### Preparation and space

For uniform star:

| Executor | Preparation (µs) | Prepared visit (ms) | Total (ms) | Plan heap | Workspace heap |
|---|---:|---:|---:|---:|---:|
| fst | 0.79 | 38.31 | 37.38 | 3,024 B | 1,136 B |
| demand | 1.19 | 29.86 | 29.82 | 3,024 B | 464 B |
| lifted-lazy | 205,431.05 | 2.75 | 198.82 | 2,098,192 B | 16 B |
| lifted-merged | 231,844.62 | 2.16 | 237.69 | 2,098,180 B | 16 B |

On-demand preparation reads no graph nodes. Its workspace is **59% smaller**
than FstJoin's on this case. Reported heap counts are retained vector capacity,
excluding inline objects, allocator overhead, compiler temporaries, and the
borrowed graph. They are not peak process memory.

The default and FstJoin perform exactly the same work on all eight cases:

| Case | Node reads per visit | Accepted slot candidates |
|---|---:|---:|
| uniform-star | 1,700,957 | 1,705,906 |
| uniform-triangle | 1,326,494 | 1,326,493 |
| uniform-anchored | 4,709 | 4,708 |
| skew | 539,301 | 540,353 |
| parity-triangle-128 | 25 | 24 |
| dense-star | 2,321 | 18,704 |

Counters are taken in separate instrumented visits; ordinary timing compiles
counter updates out. In the shared CSV, demand `retained_states` denotes compiled
control stages, not expanded product states. Lifted source reads are preparation
work; demand/FstJoin source reads are traversal work. Uninstrumented zeros mean
unavailable rather than no work.

### Mask-table control and perf

The optional table variant takes 29.95 ms on star, 26.69 ms on uniform triangle,
and 10.24 ms on skew, versus direct guards' 29.82/27.83/10.51 ms. It also takes
2.05 µs on parity-128 versus 1.32 µs direct. The
[preceding comparison](prototype/results/demand-ablation.csv) shows similarly
small/noisy large-case differences and the same preparation penalty on tiny
cases. These results favor direct guards as the simpler default; they do not
establish a useful general mask-table speedup.

`perf record` captured 300 prepared uniform-star visits per executor. FIFO
`enable`/`disable` commands exclude graph building and teardown. The sampled
cycle estimate is about **34.7 billion** for direct on-demand versus **42.9
billion** for FstJoin, about 19% less. Direct on-demand self samples split into
54.3% traversal, 14.6% opening/filtering states, 15.7% `read_word`, and 15.3%
`read_delta`. No relation-conversion or hash-consing phase appears on this path.

These measurements and the table control support the smaller product-frame
explanation for the gain; they are not evidence of a new asymptotic advantage.
Text profiles: [on-demand](prototype/results/perf-demand.txt),
[FstJoin](prototype/results/perf-fst.txt), and
[table control](prototype/results/perf-demand-table.txt).

## Recommendation

Use this factored product as the next automaton execution core. It keeps query
compilation cheap, streams immediately, supports repeated-variable patterns,
and retains the pruning behavior that protects skew and cyclic joins. The main
architectural change is expressing equality checks and bit ownership as the
query machine's transitions, composed with independently positioned graph FSTs.

Keep the existing whole-atom traversal available. The measurements justify
choosing the product for shapes vulnerable to large intermediate bindings;
they also justify keeping whole-atom execution for selective anchored lookups.
Automatic selection or switching is not implemented in this prototype. Neither
is a generic arbitrary-regex/fuzzy-query API or on-demand BDD minimization.

## Reproduction

From the repository root:

```sh
join_snapshot=$(python3 examples/join/prototype/prepare.py)
CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo test --offline --profile profiling --manifest-path "$join_snapshot/Cargo.toml" --lib --tests
CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo build --offline --profile profiling --manifest-path "$join_snapshot/Cargo.toml" --example composition_bench --example composition_profile
PROTO_SUITE=joins PROTO_REPS=7 PROTO_ENGINE=fst,demand,demand-table,recursive,automaton,lifted-lazy,lifted-merged /tmp/sgraph-composition-target/profiling/examples/composition_bench
```

`PROTO_CASE` selects a case by substring. `PROTO_ENGINE` accepts comma-separated
engine names. No production module registration is required; the preparation
script injects the prototypes into its disposable crate.

For a graph-build-excluded profile, create two FIFOs and pass their paths to
both the profiler and the example:

```sh
join_perf_dir=$(mktemp -d /tmp/sgraph-demand-perf-XXXXXX)
mkfifo "$join_perf_dir/control" "$join_perf_dir/ack"
PROTO_PERF_CONTROL="$join_perf_dir/control" PROTO_PERF_ACK="$join_perf_dir/ack" perf record --no-buildid-cache --delay=-1 --control "fifo:$join_perf_dir/control,$join_perf_dir/ack" -e cycles:u -F 999 --call-graph dwarf,4096 -o "$join_perf_dir/perf.data" -- /tmp/sgraph-composition-target/profiling/examples/composition_profile demand visit 300
perf report --stdio --no-children --percent-limit 0.5 -i "$join_perf_dir/perf.data"
```

Raw data: [final comparison](prototype/results/demand-comparison.csv),
[construction diagnostics](prototype/results/demand-comparison.log),
[source manifest and profile label mapping](prototype/results/demand-manifest.txt).
The original eager report remains in [PROTOTYPE_RESULTS.md](PROTOTYPE_RESULTS.md).
