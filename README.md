# s-graph

A compact graph engine over integer `(subject, predicate, object)` triples.
Start with **`Fst`**, an immutable index that shares identical suffixes of the
K³ radix tree. Build an automaton, compose its constraints, then apply it with
`graph.search(&machine)`. Queries run directly over compressed states with
reusable execution memory.

The crate provides graph-pattern joins, OPTIONAL, MINUS, existence filters,
regular paths, and building blocks for graph analytics. Applications supply
term dictionaries, parsing, named-graph semantics and transactions.

Run `cargo run --example automata` for a complete construction/composition example.

## Install

```toml
[dependencies]
s-graph = { path = "../s-graph" }
```

The default `heap` feature enables FST storage and the composed query/analytics
modules, without additional runtime dependencies. `file` and `compressed` add
FST and mutable-tree storage backends and include `heap` automatically. `parallel`
adds Rayon bulk building and parallel BFS for compatible backends. The optional
`experimental` feature enables comparison-only compaction algorithms; ordinary
updates, snapshots and rank-free `freeze()` do not require it.

## Build, read and persist an FST

```rust
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};

let graph = Fst::try_build_dynamic(vec![
    (0, 0, 1), (0, 0, 3), (1, 1, 2),
], DEFAULT_BLOCK_CAPACITY)?;

assert!(graph.contains(0, 0, 1));
let mut neighbors: Vec<_> = graph.scan([Some(0), Some(0), None])
    .map(|(_, _, object)| object).collect();
neighbors.sort_unstable();
assert_eq!(neighbors, [1, 3]);
assert_eq!(graph.count_matching([Some(0), None, None]), 2);
assert_eq!(graph.contains_batch(&[(0, 0, 1), (9, 9, 9)]), [true, false]);

let bytes = graph.serialize();
let restored = Fst::deserialize(&bytes)?;
assert!(restored.contains(1, 1, 2));
# Ok::<(), s_graph::TreeError>(())
```

`try_build_dynamic` infers each domain as its largest ID plus one. Use
`Fst::try_build(triples, [subjects, predicates, objects], block_capacity)` to
supply explicit domain **sizes**, each in `1..=2^32`. Duplicate triples collapse
to one edge. Empty graphs and `(0, 0, 0)` are supported. Block capacity must be
a power of two of at least 512 bytes; use `DEFAULT_BLOCK_CAPACITY` normally.
`build`/`build_dynamic` are convenience constructors that panic on invalid input;
the `try_` forms return errors, including unrepresentable 32-bit child deltas.
Build input is currently limited to `u32::MAX` triples.

An opt-in [parallel construction prototype](examples/join/FST_PARALLEL_BUILD.md)
provides `Fst::try_build_parallel` with `parallel` and `experimental` enabled.
It preserves global suffix sharing and uses the current Rayon pool.

`scan` accepts exact values or `None` wildcards and allocates no heap memory.
`select([DimRestriction; 3], output_dimension)` projects sorted, distinct values
with exact, inclusive range, union-of-ranges or free restrictions (`tree::DimRestriction`).
`count_matching` counts logical matches while memoizing shared suffixes.
Batch existence checks currently perform independent point lookups.

For files, use `graph.serialize_to_path("graph.sgf")?` and
`Fst::deserialize_from_path("graph.sgf")?`. `serialize_into`/`deserialize_from`
stream through `Write`/`Read`. Loading checks the version, resource limits,
record boundaries, DAG links, reachability and coordinate domains before
exposing the graph. `serialize::DeserializeLimits` and the `_with_limits`
loaders control maximum block count, block size and stored bytes (default 1 GiB);
validation also needs temporary memory proportional to physical states/edges.
FST snapshots use their own format, separate from mutable-tree snapshots.
Persist graphs; rebuild query plans against the loaded graph.

## Updates and storage modes

Packed FSTs support heap, real-file and LZ4-compressed block storage. Their
editable forms support local insertion/removal with the same automata:

| Storage | Packed graph | Editable graph | Feature |
|---|---|---|---|
| RAM | `Fst` | `MutableFst` | `heap` (default) |
| Temporary backing file | `FileFst` | `MutableFileFst` | `file` |
| LZ4 blocks in RAM | `CompressedFst` | `MutableCompressedFst` | `compressed` |

Enable `file` or `compressed` in Cargo features; both include the shared `heap`
API. This does not copy file-backed data into a heap graph. `PackedFst<S>` and
`EditableFst<S>` are the generic types behind these aliases.

```rust
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{Bgp, Search, Term::{Const as C, Var as V}};
use std::ops::ControlFlow;

// Declare room for future IDs; updates keep these domains fixed.
let mut graph = Fst::try_build(vec![(0, 0, 1)], [16, 4, 16], DEFAULT_BLOCK_CAPACITY)?
    .into_mutable(); // moves the packed graph; no graph copy
assert!(graph.insert(1, 0, 2)?);
assert!(!graph.insert(1, 0, 2)?); // already present
assert_eq!(graph.insert_batch(&mut [(2, 0, 3), (3, 0, 4)])?, 2);
assert!(graph.remove(0, 0, 1)?);
assert!(!graph.remove(15, 0, 15)?); // absent

let machine = Bgp::new(1).triple([C(1), C(0), V(0)]);
{
    let search = graph.search(&machine)?;
    let _: ControlFlow<()> = search.visit(&mut Default::default(), |row| {
        assert_eq!(row.get(0), Some(2));
        ControlFlow::Continue(())
    });
} // searches borrow the graph; finish them before editing
assert!(graph.insert(1, 0, 3)?);
// Reuse the machine; compile a fresh search after edits.

let bytes = graph.serialize()?; // compact SGF1 snapshot, no edit history
let packed = graph.freeze()?; // stream live states into a compact packed FST
assert!(packed.contains(1, 0, 3));
let restored = Fst::deserialize(&bytes)?;
assert!(restored.contains(1, 0, 3));
# Ok::<(), s_graph::TreeError>(())
```

`MutableFst::try_build` builds an editable graph directly. The file/compressed
aliases have the same builders, `search`, scans, counts, updates and snapshot
loaders. SGF1 snapshots load into **any** of the three modes, for example
`FileFst::deserialize_from_path("graph.sgf")?`. A live file graph uses a temporary
backing file; use `serialize_to_path` for persistence. `disk_bytes` reports its
backing-file size.

Edits copy the affected path and merge suffixes against both packed and edited
states, preserving minimality of the reachable machine. The default `Sampled(1)`
lookup stores a 32-bit fingerprint prefix and a 16/32/64-bit relative address
per physical state, choosing address width per level. The first effective edit
prepares this index. Call `prepare_updates()` before a burst to pay that cost
explicitly. Duplicate/absent updates skip preparation. The edit arena lives in RAM in all modes; its obsolete nodes are
reclaimed immediately. Packed base bytes stay intact until `compact()` or
`freeze()`. Call `compact()` between write bursts to reclaim dead base bytes
and move edits into the chosen block storage. Consuming `freeze()` uses rank-free
BFS compaction: it reuses edit metadata and releases source blocks as it copies
live states. It preserves the minimality maintained by updates. Use
`compact_streaming()` for the same algorithm with timing and memory statistics.
Borrowed `to_fst()`, `compact()` and serialization preserve the source until
packing succeeds and currently use the separate physical-state rebuild path.
See [compaction details and measurements](examples/join/FST_RANK_FREE_COMPACTION.md).
For a smaller update index, select
`packed.into_mutable_with_lookup(UpdateLookup::Sampled(8))` (import
`s_graph::mutable_fst::UpdateLookup`). It keeps every state address as a
per-level 16/32/64-bit relative offset, and one 32-bit fingerprint per eight
states. This trades extra candidate reads for a smaller resident index. The rate must be nonzero. `prepare_updates()` builds it
before the first edit; otherwise the first effective edit pays that cost.
See [compact update indexes](examples/join/FST_COMPACT_INDEX.md).
There is no automatic compaction
or background I/O. `memory_usage()` reports base storage, suffix indexes,
live edits and reusable arena/table capacity separately; see the
[memory experiment](examples/join/FST_MEMORY.md). Compaction works over physical DAG states, without listing
all triples, and uses temporary memory proportional to physical states/edges.
Editable `serialize` returns `Result<Vec<u8>>` because it performs this packing;
`serialize_into` and `serialize_to_path` also compact into a temporary packed graph.

For periodic batches, the experimental `UpdateLookup::Memoized` strategy caches
suffix lookups as edits encounter them. `freeze_in_place()` returns an immutable
`FrozenFst`: it keeps the original blocks, packs live edited states into blocked
storage and releases update indexes. Consume `into_mutable_with_lookup(...)`
to apply the next batch. Obsolete records remain until explicit compaction;
ordinary SGF1 serialization still repacks. See the
[in-place freeze prototype](examples/join/FST_FREEZE_IN_PLACE.md) for usage
and BFS edit blocks. The experimental
[block compactor](examples/join/FST_BLOCK_COMPACTION.md) releases old blocks
as their live records are copied.
The [recursive variant](examples/join/FST_RECURSIVE_COMPACTION.md),
`compact_recursive()`, reserves BFS positions and builds child first. Comparison
methods `compact_blocks()`, `compact_depths()` and `compact_recursive()` require
`--features experimental`; `compact_streaming()` provides default freeze
diagnostics without that feature.

Insertions outside declared domains fail; `insert_batch` validates the whole
batch before changing the graph. Duplicate updates are no-ops. Removal outside
the domains returns `false`. Batch methods return the number of changed triples.

The existing mutable `Heap`, `File` and `Compressed` radix trees also expose
`search(&machine)` and `source()`, alongside their existing updates, scans,
paths and SGZ1 serialization. They retain `NativeJoin` for `join_run`; explicit
`search` uses the shared adaptive automaton engine. `source()` also feeds the
compressed-edge analytics API on either representation.

## Build a graph-pattern automaton

Variables are numbered from zero; predicates can be constants or variables.
Build the query language independently of the graph. Compilation binds it to
an index and chooses an execution schedule. Reuse the compiled search and its
workspace, and consume borrowed result rows inside the callback.

```rust
use std::ops::ControlFlow;
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{Bgp, Search, Term::{Const as C, Var as V}};

let graph = Fst::build_dynamic(vec![
    (0, 0, 1), (0, 0, 3), (1, 1, 2),
], DEFAULT_BLOCK_CAPACITY);
let machine = Bgp::new(2)
    .triple([C(0), C(0), V(0)])
    .triple([V(0), C(1), V(1)]);
let search = graph.search(&machine)?;
let mut workspace = Default::default();
let mut rows = Vec::new();
let _: ControlFlow<()> = search.visit(&mut workspace, |row| {
    rows.push(row.values.to_vec()); // copy only if retaining the row
    ControlFlow::Continue(())
});
assert_eq!(rows, [vec![1, 2]]);

// ASK / LIMIT: Break stops traversal, and the workspace can be reused.
assert_eq!(search.visit(&mut workspace, |_| ControlFlow::Break(true)),
           ControlFlow::Break(true));
# Ok::<(), s_graph::TreeError>(())
```

`Bgp::new(width)` declares a global binding namespace; unused slots remain
unbound. `.triple` adds a pattern and `.and(other_bgp)` fuses two BGPs before
compilation, so their constraints can prune together. The builder can be
searched against multiple graphs; compiled searches borrow their graph.
Repeated variables, predicate variables, cycles, disconnected patterns, ground
constraints and empty BGPs are supported. Output order is unspecified.

Preparation compiles guards, register writes and traversal schedules. Adaptive
selection compares bounded probes; larger patterns receive enough work to
reach across more atoms, capped at 32,768 node reads per probe. A BGP search's
`.planning()` reports actual reads, its `node_budget`, chosen schedule and
whether shared-bit frontier propagation was selected. No graph relations or
graph-sized query DFA are constructed. Set `.schedule(Schedule::Frontier)`
or another `automata::Schedule` on a BGP for explicit execution control.

`Join::new`, `Fst::join`, `Fst::bgp`, and `graph::try_join_run` remain available
for callers using `JoinQuery`. They use the same engine. Low-level
`scheduled_join::ScheduledJoin` also exposes traversal counters and memory
diagnostics.

## Cooperative query limits

`ControlledCompile` and `ControlledSearch` add cancellation, deadlines and a
shared work budget to BGPs, slot recognizers, VALUES and their binding operators.
Checks run inside unsuccessful searches as well as when results are produced.
Ordinary `search`/`visit` keep their existing statically specialized fast path.

```rust
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{Bgp, ControlledCompile, ControlledSearch, Execution,
    PreparationError, StopReason, VisitResult, VisitStop, Term::{Const as C, Var as V}};
use std::{ops::ControlFlow, sync::atomic::{AtomicBool, Ordering}};

let graph = Fst::build_dynamic(vec![(0, 0, 1), (1, 1, 2)], DEFAULT_BLOCK_CAPACITY);
let cancelled = AtomicBool::new(false); // may also live inside an Arc
let request = Execution::new().with_budget(50_000).with_cancellation(&cancelled);
let machine = Bgp::new(2).triple([C(0), C(0), V(0)]).triple([V(0), C(1), V(1)]);
let search = machine.compile_controlled(graph.source(), &request)?;
let mut workspace = Default::default();
let outcome: VisitResult<()> = search.visit_controlled(&mut workspace, &request, |row| {
    assert_eq!(row.get(1), Some(2));
    ControlFlow::Continue(())
});
assert_eq!(outcome, ControlFlow::Continue(()));
cancelled.store(true, Ordering::Relaxed);
assert_eq!(search.visit_controlled(&mut workspace, &request, |_| ControlFlow::Break(())),
           ControlFlow::Break(VisitStop::Interrupted(StopReason::Cancelled)));
# Ok::<(), PreparationError>(())
```

`VisitStop::User(value)` preserves a callback break; `Interrupted(reason)` means
incomplete output. Work units are executor checkpoints, including compiler and
propagation work, rather than node reads. Use a fresh `Execution` per request;
work is cumulative and stops are sticky. Custom searches opt in by implementing
`ControlledSearch` and checking their own loops. Paths and analytics retain
their existing callback control. See [contracts and integration](examples/join/EXECUTION_CONTROL.md).

## Compose queries: OPTIONAL, MINUS and more

All leaves in a composed query use the **same global variable width**.
`Binding::get(variable)` returns `Option<u32>`: an OPTIONAL result may leave a
variable unbound. Operators stream rows and preserve duplicate solution rows.

```rust
use std::ops::ControlFlow;
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{Bgp, Query, Search, Term::{Const as C, Var as V}};

let graph = Fst::build_dynamic(vec![
    (0, 0, 1), (0, 0, 3), (1, 1, 2),
], DEFAULT_BLOCK_CAPACITY);
let people = Bgp::new(2).triple([C(0), C(0), V(0)]);
let details = Bgp::new(2).triple([V(0), C(1), V(1)]);
let machine = people.optional(details);
let optional = graph.search(&machine)?;
let mut workspace = Default::default();
let mut rows = Vec::new();
let _: ControlFlow<()> = optional.visit(&mut workspace, |row| {
    rows.push(row.to_options());
    ControlFlow::Continue(())
});
rows.sort_unstable();
assert_eq!(rows, [vec![Some(1), Some(2)], vec![Some(3), None]]);
# Ok::<(), s_graph::TreeError>(())
```

Build these expressions first, then pass them to `graph.search`. Compiled
operators share the `automata::Search::visit` interface:

| Operator | Construction / meaning |
|---|---|
| Inner join | `left.join(right)`; merges compatible partial bindings |
| Union | `left.union_all(right)`; retains duplicates |
| MINUS | `left.minus(right)`; a matching RHS must share a bound variable |
| Existence | `left.exists(right)` / `left.not_exists(right)` |
| Filter | `input.filter(\|row: Binding<'_>\| ...)` |
| Conditional OPTIONAL | `left.optional_if(right, \|merged: Binding<'_>\| ...)` |
| Explicit values | `Values::new(width, rows)?`; `Values::unit(width)` is one empty mapping |

Existing compiled plans and path relations also compose through `query_ops`
using its public operator structs. Schema mismatches in a
builder expression return a compilation error. Filter closures are cloned
into the compiled expression.

Put connected inner-join patterns into a single BGP to let the scheduler prune
them together. RHS BGP/path probes push existing bindings into traversal.
Ordinary RHS filters see their own relation; use `CorrelatedFilter` for a
predicate that explicitly inspects both the outer binding and an RHS row.
`MINUS` and `NOT EXISTS` differ on disjoint/empty mappings. OPTIONAL emits the
original left row only if no right match passes its condition. These are
binding operators, with semantics beyond boolean language composition.

## What “automaton” means here

The graph is a finite language of encoded triples. Identical suffix languages
share one physical FST state. A query supplies another machine: constants,
variable-equality guards and reversible binding writes constrain which storage
transitions can be followed. A join is a register automaton over those choices;
its schedule decides when to bind full atoms and when to synchronize prefixes.

`automata::Automaton<Event, Context>` separates state transitions,
acceptance and pruning. Implement it for your own event type and compose typed
machines with `And`, `Or`, `Not` or `MapInput`. Composition is generic Rust, with
no boxed virtual calls in the traversal loop. `BitmapAutomaton` provides fast
candidate masks; `GraphAutomaton`, `FstSource` and `visit_language` connect
recognizers to compressed storage. `ReversibleAction` models branch-local
updates; `Project`/`Transducer` and `Reducer` model lasting output contributions.

`AutomatonExt` adds `.intersection`, `.union` and `.complement` constructors.
`And` intersects languages on the **same event sequence**; `Or` unites them.
`Not` complements a language, including a dead-state sink. It cannot safely
prune by simply inverting a prefix mask: use `Unfiltered` for arbitrary
complements in a bitmap walk. Relational `Join`, OPTIONAL and MINUS use the
binding operators above, because they combine mappings rather than recognize
the same word. Wrap a slot recognizer in `Language` to apply it through the
same `graph.search` entry point; its three output slots are S/P/O:

```rust
use std::ops::ControlFlow;
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{AutomatonExt, Language, Mask, Search};

let graph = Fst::build(vec![(0,0,0), (0,0,1), (1,0,1)], [2,1,2], DEFAULT_BLOCK_CAPACITY);
let machine = Language(Mask(3).intersection(Mask(u64::MAX)));
let search = graph.search(machine)?;
let mut count = 0;
let _: ControlFlow<()> = search.visit(&mut Default::default(), |_| {
    count += 1; ControlFlow::Continue(())
});
assert_eq!(count, 2);
# Ok::<(), s_graph::TreeError>(())
```

`Mask` constrains **encoded slot numbers**, which depend on the storage layout;
use `Bgp` for coordinate/variable constraints. Custom slot recognizers implement
`Automaton<Slot>`; add `BitmapAutomaton` for fast masks or use `Unfiltered`.
Frontier joins propagate supported variable bits across live atom automata,
including across storage levels, and keep reversible effects in a workspace.

## Regular graph paths

`PathExpr` builds label languages with `.then`, `.or`, `.optional`, `.star`,
`.plus`, inverse labels and `.reversed()`. Compilation produces a packed NFA;
execution traverses the product of graph vertices and automaton states.

```rust
use std::ops::ControlFlow;
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{PathExpr, PathBindings};
use s_graph::traversal::HopRange;
use s_graph::graph::Term::{Const as C, Var as V};
use s_graph::query_ops::Plan;

let graph = Fst::build_dynamic(vec![(0, 0, 1), (1, 1, 2), (2, 1, 3)], DEFAULT_BLOCK_CAPACITY);
let language = PathExpr::label(0).then(PathExpr::label(1).plus()).compile()?;
let paths = PathBindings::new(&graph, &language, vec![], C(0), V(0), 1,
                              HopRange::new(1, 8)?)?;
let mut endpoints = Vec::new();
let _: ControlFlow<()> = paths.visit(&mut Default::default(), |row| {
    endpoints.push(row.get(0).unwrap()); ControlFlow::Continue(())
});
endpoints.sort_unstable();
assert_eq!(endpoints, [2, 3]);
# Ok::<(), s_graph::TreeError>(())
```

Use `PathBindings` anywhere a `Plan` is accepted, including OPTIONAL and MINUS.
For an unbound subject, pass the vertex universe (including isolated vertices)
as the third constructor argument; it is deduplicated once. A constant start
can be supplied without that universe and is valid for zero-hop queries.
`PathAutomaton::reachable` also streams `(endpoint, shortest_qualifying_hops)`
directly, using `AutomatonWorkspace`.

Path languages implement the shared automaton trait too. For example,
`Composed(And(&allowed, Not(&excluded)))` is a path runner and can be supplied
to `PathBindings`; both operands are compiled `PathAutomaton`s. Its generic
product runner preserves language intersection/complement semantics; a plain
packed NFA normally runs faster by pushing label bounds directly into scans.

Paths return **distinct endpoint pairs**, rather than every witness walk.
Use BGPs for fixed-length sequences whose intermediate bindings or multiplicity
matter. `traversal` separately exposes bounded reachability, shortest-walk
reconstruction, and walk/trail/simple-path enumeration. Cancellation is via
`ControlFlow`; enumeration can have exponential output.

## Transducers and graph analytics

The same recognizers can filter edges, project contributions and feed a
reducer. Here two automata select non-loop edges with predicate 0:

```rust
use s_graph::{Fst, DEFAULT_BLOCK_CAPACITY};
use s_graph::automata::{And, FstSource, Project};
use s_graph::analytics::{CompressedEdges, Edge, NonLoop, Predicate, fold, PageRank, LabelPropagation};

let graph = Fst::build_dynamic(vec![(0, 0, 0), (0, 0, 1), (1, 1, 2)], DEFAULT_BLOCK_CAPACITY);
let edges = CompressedEdges::new(FstSource(&graph), 3, None)?;
let machine = Project(And(Predicate(Some(0)), NonLoop), |edge: &Edge, _: &()| edge.target);
let mut targets = Vec::new();
fold(&edges, &mut Default::default(), &machine, &(), &mut |target| targets.push(target));
assert_eq!(targets, [1]);

let mut rank = PageRank::default();
let (_, converged, _) = rank.run(&edges, 0.85, 1e-9, 100);
assert!(converged);
assert_eq!(rank.rank.len(), 3);
let mut communities = LabelPropagation::default();
let _ = communities.run(&edges, 20);
assert_eq!(communities.labels.len(), 3);
# Ok::<(), s_graph::TreeError>(())
```

Graph algorithms keep per-vertex state in their workspaces/reducers; shared FST
states are storage states, not graph vertices. PageRank is directed and handles
dangling vertices. Label propagation is synchronous and undirected, ignores
self-loops, chooses the smallest tied label, and can oscillate; check its
convergence flag. Both currently use dense vertex arrays. For repeated full
scans, `Csr::from_source(&edges, predicate)` trades an explicit two-pass
projection and extra memory for faster iteration. Composability does not make
all graph algorithms finite-state or remove their algorithm-specific state.

## Further reading

- [DESIGN.md](DESIGN.md): layouts, algorithms, alternatives and measurements.
- [Automata and frontier propagation](examples/join/AUTOMATA_FRONTIERS.md): current API, algorithm, research and performance comparisons.
- [Promotion notes](examples/join/PROMOTION.md): migration, compatibility and performance controls.
- [Query operators and paths](examples/join/QUERY_OPERATORS_AND_PATHS.md): semantics and benchmark details.
- [Automata scheduling](examples/join/SCHEDULED_AUTOMATA.md): the adaptive executor.
- [Analytics and large BGPs](examples/join/ANALYTICS_AND_LARGE_BGP.md): full-scan tradeoffs and thousands-of-pattern tests.
- [Historical backend guide](examples/join/LEGACY_BACKENDS.md): mutable storage APIs and earlier defaults.
- [Comparison harness](examples/join/prototype/prepare.py): reproducible alternatives; promoted modules come from `src/`.

README Rust examples are compiled and executed by `cargo test --doc`.

Build/test and packaging checks are documented in [CONTRIBUTING.md](CONTRIBUTING.md).
Benchmark reports link to a [results index](examples/join/RESULTS.md);
raw measurements remain in repository checkouts and are excluded from Cargo packages.
