# Explicit automata, shared-bit propagation and bounded schedule selection

The public construction path is now `automata::Bgp` / composed recognizers,
followed by `Fst::search`. Adaptive execution compares synchronized and nested
traversals and can select a new frontier propagator for cyclic constraints.
Existing `JoinQuery`, `Join`, BGP operators and storage alternatives remain
available. The graph representation and snapshot format are unchanged.

## Research that informed the change

These are relevant approaches reviewed on 21 October 2026, rather than a claim
to exhaustively survey every graph-matching system:

| Source | Relevant idea and decision |
|---|---|
| [Rust `fst` automaton API](https://github.com/BurntSushi/fst/blob/master/src/automaton/mod.rs), [search example](https://github.com/BurntSushi/fst#example) | Construct the query machine first, then search immutable storage. Keep state transitions, acceptance, conservative pruning and static combinators separate. We use typed alphabets and binding-producing register machines alongside ordinary recognizers. |
| [Worst-Case Optimal Radix Triejoin](https://arxiv.org/html/1912.12747) | Booleanized variable refinement and bit-interleaved representations are directly relevant to K³ storage. Its analysis distinguishes query-size factors from data complexity; a same-level tuple walk does not automatically inherit its bound. |
| [Optimal Joins using Compact Data Structures](https://arxiv.org/abs/1908.01812) | Shared subtrees and lazy composition of relational operations support doing query work directly over compressed states. We keep the index immutable and build query-sized control structures. |
| [CompactLTJ, 2025](https://users.dcc.uchile.cl/~gnavarro/ps/vldbj25.pdf) | Compact indexes can support practical variable-oriented multiway intersections. We borrow the support-intersection principle while retaining our single graph index; this is not a port of its permutation-trie implementation or a benchmark against that system. |
| [FSST](https://github.com/cwida/fsst), [OptFSST, 2026](https://arxiv.org/abs/2607.11271) | FSST is **Fast Static Symbol Table**, a string codec, rather than a graph join algorithm. Static tables and fine-grained compressed access fit the architecture. OptFSST improves symbol selection/encoding; it is a possible future term-dictionary tool, not a replacement for this integer graph index. No string codec or graph-wide re-encoding was introduced. |

The implementation remains a heuristic adaptive join engine. Neither the
planner nor frontier propagation has a proven AGM/WCO runtime bound. Frontier
support is local to currently exposed FST nodes; hidden suffix correlations
can still require substantial search.

## Explicit construction and composition

```rust
use s_graph::automata::{Bgp, Query, Search, Term::{Const as C, Var as V}};

// Construct languages without binding them to an index.
let people = Bgp::new(2).triple([C(0), C(0), V(0)]);
let detail = Bgp::new(2).triple([V(0), C(1), V(1)]);
let machine = people.optional(detail);

// Compile against the chosen immutable graph; reuse execution memory.
let search = graph.search(&machine)?;
let mut workspace = Default::default();
let _: std::ops::ControlFlow<()> = search.visit(&mut workspace, |row| {
    consume(row.get(0), row.get(1));
    std::ops::ControlFlow::Continue(())
});
```

This illustrative fragment assumes `graph` and `consume`; complete executable
examples are in the [README](../../README.md) and are tested as doctests.

- `.triple` and `Bgp::and` fuse conjunctive constraints into one register
  automaton before choosing a schedule. `Bgp::and` keeps the left schedule;
  call `.schedule(...)` after merging to override it.
- `Query` supplies `.join`, `.optional`, `.optional_if`, `.minus`, `.union_all`,
  `.exists`, `.not_exists` and `.filter`. Compilation validates global widths.
  These operators retain their existing partial-binding and bag semantics.
- `Values` can also be compiled/composed. Compiled expressions own query
  control data, while graph references and deliberately borrowed constituent
  recognizers determine lifetimes. Filter conditions are cloned at compilation.
- `AutomatonExt` builds `.intersection`, `.union` and `.complement` for typed
  recognizers. `Language(machine)` adapts a packed-slot recognizer to
  `graph.search`, returning S/P/O bindings. `Unfiltered` provides the safe
  path for ordinary recognizers and complements without a mask oracle.
- Paths still use explicit `PathExpr` / `PathAutomaton` construction and graph
  product traversal. `PathBindings` composes with compiled BGP searches through
  the existing binding operators. Edge transducers and reducers support
  analytics through the same underlying automaton traits.

The `automata` module is a public facade over the existing execution traits,
not another incompatible automaton hierarchy. `.search` compiles expression
types statically: there is no boxed dispatch per edge, result or transition.
The `Compile<Source>` trait also supports alternative triple sources. A BGP
that binds its entire global schema forwards the engine's borrowed row directly
to its callback; sparse schemas retain the remapping workspace. This avoids
adding a full-row copy to the primary explicit-automaton API.

## Frontier propagation

The [large-cycle investigation](LARGE_CYCLES.md) found two distinct problems:
an entire tuple digit can contain unrelated free bits before a cycle closes,
and S/O occurrences of a variable can reach different bit positions at the
same storage level. Merely interleaving atoms by level misses both opportunities.

`Policy::Frontier` maintains one live FST node and a 64-bit surviving-transition
mask per atom. Constants, repeated-variable equality and out-of-domain zero
bits are compiled into local masks. For every variable bit, compact incidence
lists identify constraints that can observe it.

The reusable `BitSupport` primitive projects an allowed-transition bitmap onto
the Boolean values of one bit. Intersecting such supports can force a value or
prove a contradiction. On a Boolean domain, propagating every singleton to all
incident constraints reaches the same support fixed point without repeatedly
constructing larger projected relations.

```text
compile(patterns, source_layout):
    compile constants/equality masks for each (atom, storage_level)
    index occurrences by (variable, value_bit)

search(workspace, optional_pinned_values):
    reset live cells, bindings, known bits, ready-bit index, queue and undo trail
    open every atom's root; enqueue its cell
    propagate_to_fixed_point()
    loop:
        if consistent and every atom has accepted:
            emit bindings; honor cancellation
        else if consistent:
            bit = first exposed unknown bit in variable/MSB order
            save trail checkpoint and the untried value 1
            bind bit = 0; propagate_to_fixed_point(); continue

        restore the latest checkpoint with an untried value
        if none exists: finish
        bind that bit = 1; propagate_to_fixed_point()

propagate_to_fixed_point():
    while a changed cell is queued:
        intersect its transition mask with already known variable bits
        if no transition remains: reject this branch
        for each still-unknown bit visible in this cell:
            if its transition support is a singleton:
                bind that value and enqueue every currently incident cell
        if exactly one transition remains:
            follow it immediately, or mark the atom accepting at a leaf
            expose and enqueue its next cell
```

Immediate advancement matters: one atom can expose a bit at its next level
while another occurrence of that bit is still at an earlier level. The shared
binding connects them without requiring a global level barrier. It does not
split or expand stored FST states.

The implementation uses a two-level ready-bit bitmap, deduplicated work queue,
flat decision stack and reversible trail. Rollback restores masks, cursors,
bit availability and acceptance counts. Callback cancellation and budget
exhaustion can leave scratch populated; the next run resets it. Pinned values
constrain this same machine and cannot become additional result bindings.
Workspace and plan memory scale with query bits/stages, not graph relations.

Correctness follows from conservative restriction: every valid answer supports
its bit values in every live cell, so propagation cannot remove it. Advancing
a singleton preserves its only possible path. Accepted cells across all atoms
witness every pattern under one binding. Binary decisions partition assignments
and rollback restores the exact parent state. This is a correctness argument,
not a worst-case running-time proof.

## Adaptive selection and preparation cost

Small unanchored BGPs retain a 64-node Levels probe and up to six 128-node
nested-order probes. Unlike the previous planner, Levels stays competitive:
when it emits answers, its answers-per-work score is compared with the best
nested candidate. Work is estimated as `nodes + candidates / 8`, with a 5%
margin to avoid selecting Levels on a near tie. Nested order search retains
the previous terminal-check progress heuristic when no answers are sampled.

For more than 16 atoms, each schedule receives
`clamp(4 × atoms × height, 64, 32768)` node reads. Compare Levels and nested
on the incidence-based heuristic order, rather than recompiling six large
permutations. The best compiled candidate is retained instead of compiled
again. The strong anchored-lookup shortcut still avoids probing.

When neither cheap schedule emits a row, a bounded frontier probe is attempted
for at most 1,024 atom/level specifications. Automatic selection requires that
probe to finish completely; a completed zero-row run becomes an exact empty
certificate. This compilation cap protects preparation cost on huge queries
with weak evidence. Explicit `Schedule::Frontier` has no such query-size cap.
Carry chains keep nested traversal when it makes useful progress.

`PlanningStats.node_budget` sums the hard limits of attempted probes; actual
reads are in `nodes`. Small-query planning is bounded by 960 node reads;
the largest automatic budget is 65,536. These limits bound node openings,
not compilation time, every CPU operation or query runtime. Frontier probes
additionally cap propagation updates plus bit decisions at eight times the
node limit: cached leaf words must not permit an unbounded Cartesian-product
enumeration without further reads. A chosen schedule
can still be poor on data not represented by its early probe.

## Measurements

The baseline is the production algorithm before this change, rebuilt from the
snapshot used for the preceding cycle investigation. Both versions use the same
fixture definitions and analytical/reference checks. Tables aggregate two pass
medians on BFS. **All times in the first table are microseconds.**

| Shape | Baseline visit | Current visit | Baseline fresh total | Current fresh total |
|---|---:|---:|---:|---:|
| Uniform star | 19,925 | 20,133 | 19,951 | 20,163 |
| Uniform triangle | 14,968 | 15,207 | 15,026 | 15,185 |
| Anchored path | 2.161 | 2.176 | 3.413 | 3.451 |
| Skew | 4,066 | 4,344 | 4,074 | 4,278 |
| Dense star | 78.94 | 78.93 | 79.93 | 79.57 |
| Mixed triangle | 39.11 | 35.39 | 45.27 | 41.96 |
| Parity triangle, empty certificate | 0.002 | 0.003 | 1.504 | 1.492 |
| Functional triangle | 4,979 | 1,962 | 4,996 | 1,976 |
| Functional shared-predicate path | 9,999 | 4,790 | 9,942 | 4,926 |
| XOR chain, 64 atoms | 43.75 | 18.62 | 145 | 65.52 |
| XOR chain, 1,024 atoms | 489 | 327 | 2,044 | 1,342 |
| XOR chain, 4,096 atoms | 2,122 | 1,585 | 9,858 | 7,190 |
| Star, 1,024 atoms | 747 | 487 | 2,912 | 2,060 |
| XOR cycle, 4,096 atoms | 2,082 | 1,415 | 9,769 | 7,334 |
| Empty XOR cycle, 4,097 atoms | 2,129 | 2,096 | 10,115 | 7,245 |

Adaptive now selects Levels for both functional fixtures and the nonempty large
XOR BGPs. Mixed triangle also selects Levels with the same 2,475 node openings
as its former prefix schedule. Uniform/anchored/skew choices retain exactly the
same traversal node/candidate counts as baseline. Some prepared timings differ
between runs; no improvement is claimed for those shapes. Skew prepared visit
is 6.8% slower in this paired timing run, while the separate perf control below
does not reproduce that slowdown. Skew preparation does increase from about
9.5 to 38.2 µs because the fallback probe is attempted
and discarded. Parity's prepared visit is just a certificate check; its total
remains about 1.5 µs.

Preparation remains material on huge BGPs. The 4,096-atom chain improves from
7.55 to 5.44 ms of preparation, so total improves less than simply forcing
Levels would. The empty 4,097-atom cycle still chooses nested traversal; its
gain comes from compiling fewer candidate orders. These are remaining costs,
not hidden in the traversal numbers.

The more demanding cycle controls show why both executors are retained:

| Cycle on BFS | Baseline prepared visit | Current prepared visit | Current fresh total | Current node openings |
|---|---:|---:|---:|---:|
| XOR, 4,096 vertices, 32 atoms | 9.252 ms | **1.769 ms** | 1.816 ms | 145,665 |
| XOR, 4,096 vertices, 128 atoms | 32.292 ms | **7.155 ms** | 7.297 ms | 594,945 |
| Carry ring, 4,096 vertices, 128 atoms | 49.535 ms | 49.621 ms | 49.742 ms | 3,114,121 |
| Dense odd cycle, 33 atoms | Budget, no answer | **19.64 µs** | 87.99 µs | **99** |
| Dense odd cycle, 129 atoms | Budget, no answer | **77.90 µs** | 299.37 µs | **387** |
| Dense odd cycle, aligned group frontier, 129 atoms | Budget, no answer | **11.38 µs** | 66.28 µs | **772** |
| XOR with padded predicate domain, 128 atoms | 40.732 ms | 40.766 ms | 41.108 ms | 2,605,641 |

`Budget` means five million diagnostic node openings without completion; it is
not a measured full time. Both versions still exceed that budget on the
128-atom shuffled ring. The unpadded dense odd cases select frontier propagation;
the aligned dense case selects Levels. Carry and padded-XOR retain nested
traversal. The 20-, 32- and 4,096-atom carry-chain diagnostics also retain nested
and finish all 16 answers with identical work to baseline. Both FST layouts
agree on answers, planning decisions and execution counters for each version.

Explicit frontier propagation helps the previously explosive dense cases, but
is not a universal fast path: an initial prototype run took about 83 ms on the
128-atom XOR cycle versus about 7 ms with Levels, and about 450 ms on its carry
ring versus about 50 ms nested. Those early measurements use shorter three-trial
10-ms targets and are retained as diagnostics, not merged into the final paired
table. They motivated conservative automatic fallback selection. The current
default dense 129-atom frontier plan/workspace retain about 153/125 KB, compared
with the unfinished baseline's 123/19 KB. The better pruning costs more query
memory; graph storage bytes remain identical.

### Perf: scheduling gains, unchanged skew kernel

Gated hardware counters cover warmed prepared traversal only: three trials per
layout/version, targeting 0.5 s, construction/validation outside the gate. All
four events run 100% of the measured interval.

| BFS case | Baseline instructions/query | Current instructions/query | Baseline cycles/query | Current cycles/query |
|---|---:|---:|---:|---:|
| Skew | 48.506 M | 48.507 M | 16.138 M | 15.730 M |
| Functional triangle | 81.255 M | **32.406 M** | 19.700 M | **7.807 M** |

Skew perf medians are 4.093/3.995 ms, so the paired wall-clock
difference is not a repeatable kernel slowdown in this control. Its extra
preparation is real and excluded from these counters. The functional triangle
instruction reduction tracks fewer node openings (342,861 → 134,295), rather
than a graph-layout change. Pow2 shows the same instruction reduction.

Raw final results: [joins](prototype/results/automata-final-joins/summary.json),
[cycles](prototype/results/automata-final-cycles/summary.json),
[carry diagnostics](prototype/results/automata-final-carry/summary.json),
[baseline perf](prototype/results/automata-evolution-perf-baseline/summary.json),
[current perf](prototype/results/automata-final-perf-current/summary.json),
and the [source/result manifest](prototype/results/automata-evolution-manifest.json).

## Reproduction

The comparison harness copies production sources. Build the baseline and current
snapshots with **separate target directories**, then keep their executables
stable throughout measurements. Reusing a target directory across same-name
snapshot crates can leave a stale unqualified example executable even when
Cargo reports the other snapshot fresh. Binary hashes and a dense-cycle
behavior check were verified before the recorded paired experiment.

```sh
# Build a disposable current snapshot:
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
sgraph_snapshot="/printed/disposable/path"
CARGO_TARGET_DIR="$PWD/target/current-automata" cargo build --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example layout_bench --example cycle_bench --example schedule_stress

# Prepare a second disposable snapshot; record its printed path:
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
sgraph_baseline="/printed/second/disposable/path"
# Apply the recorded reverse patch inside that disposable snapshot only.
GIT_CEILING_DIRECTORIES="$(dirname "$sgraph_baseline")" git -C "$sgraph_baseline" apply "$PWD/examples/join/prototype/pre_frontier.patch"
CARGO_TARGET_DIR="$PWD/target/baseline-automata" cargo build --offline --profile profiling --manifest-path "$sgraph_baseline/Cargo.toml" --example layout_bench --example cycle_bench --example schedule_stress

python3 examples/join/prototype/evolution_compare.py target/baseline-automata/profiling/examples/layout_bench target/current-automata/profiling/examples/layout_bench --output target/evolution-joins
python3 examples/join/prototype/evolution_compare.py target/baseline-automata/profiling/examples/cycle_bench target/current-automata/profiling/examples/cycle_bench --cycles --output target/evolution-cycles
python3 examples/join/prototype/evolution_compare.py target/baseline-automata/profiling/examples/schedule_stress target/current-automata/profiling/examples/schedule_stress --carry --output target/evolution-carry
```

The [baseline reverse patch](prototype/pre_frontier.patch) restores the measured
pre-change engine and benchmark entry points in a prepared snapshot. Build its
example executables only: newly added public-API tests remain in that disposable
snapshot and refer to the new API.

The before/after harness alternates version and layout order, uses five warm
trials calibrated to 50 ms per phase, and measures preparation, prepared visit
and fresh-plan total independently. Default block capacity is 65,536. Runs
are sequential without competing builds/tests; CPU is not pinned. Diagnostic
counter runs are separate from timed visits. Cyclic cases stop diagnostics at
five million node openings and do not report a partial traversal as a query
time. The timing harness calls `ScheduledJoin` directly, excluding explicit-builder
construction and binding-remapping overhead. Main cases check complete sorted
bindings against FstJoin; cyclic cases
use independently tested analytical answers.

Validation includes exhaustive small assignments, all explicit schedules,
repeated/predicate variables, unequal and full-width domains, parameter hints,
partial/bag operator semantics, cancellation/reuse, and thousands of atoms on
a 128 KiB stack. Storage and serialization compatibility tests remain active.
The all-feature suite passed 117 tests and six README doctests. The subsequent
full-schema row forwarding change was checked again with query-operator and
public automaton tests. Both frontier tests also passed after adding the
cached-leaf work cap, including a Cartesian-product regression test. The
standalone `automata` example and no-default-features check also passed.
