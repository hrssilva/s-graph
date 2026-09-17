# s-graph

A compact graph representation and algorithm library over integer triples
`(subject, predicate, object)`: a DFS-preorder, interleaved-skip-pointer radix trie over
Morton-coded triples, generic over three storage backends (in-RAM,
real-file, LZ4-compressed in-RAM), with local insertion, native prefix
conjunctive joins, bounded paths, and regular-path automata. Dictionaries,
named-graph semantics, query languages and transactions belong to consumers.

It grew out of an exploratory comparison against the k²-tree family
(Mk²-tree, IK²-tree) and Omnitree, kept in `examples/` along with the
full survey and literature grounding (see `examples/README.md`). This
crate (`src/`) is the productionized result: the same structure, with
`Result`-based error handling, feature-gated backends, and a real public
API instead of a benchmark harness.

## Why

Compact RDF indexes (k²-tree-family structures especially) are usually
build-once, query-forever: adding one triple means rebuilding the whole
structure. `s-graph` inserts locally, with O(height log B + block bytes)
work for ordinary insertions and O(B) block-directory work on splits,
where B is the block count, while keeping the same
DFS-preorder byte layout that makes the structure page/compress
naturally. Query latency and memory footprint don't degrade on skewed or
concentrated data the way the k²-tree family's unconditional per-level
fan-out does (see `examples/README.md` §5 for the measured ~2000x gap
this design avoids structurally).

## Install

```toml
[dependencies]
s-graph = { path = "../s-graph" }  # or a git dependency, once published
```

Feature flags (`heap` is the only one on by default):

| Feature | Adds | For |
|---|---|---|
| `heap` | `Heap` backend | In-RAM, fastest, no extra dependencies |
| `file` | `File` backend | Real backing file, `pread`/`pwrite`, near-zero steady-state RAM |
| `compressed` | `Compressed` backend | LZ4-compressed in-RAM blocks, for RAM-constrained/no-disk deployments |
| `parallel` | `rayon`-based bulk-build fan-out | ~4x faster bulk build at large N (measured); off by default so a single-threaded consumer isn't forced to accept the dependency |

## Quick start

```rust
use s_graph::Heap;

// Build from a batch of (subject, predicate, object) u32 triples.
// A real application dictionary-encodes its own IRIs/literals to u32s
// (see "Getting real data in" below) -- this crate indexes ids, it
// doesn't do term interning itself.
let triples = vec![(0, 0, 1), (0, 1, 2), (1, 0, 2)];
let mut tree = Heap::build_dynamic(triples, /* block_capacity */ 4096);

assert!(tree.contains(0, 0, 1));
assert!(!tree.contains(2, 2, 2));

// Row/column-style bound queries.
let objects_of_0_via_pred_0 = tree.query_two_bound([0, 1], [0, 0], 2); // subject=0, predicate=0, object=?
let predicates_between = tree.predicates_for(0, 1); // subject=0, object=1, predicate=?

// Insert within the declared coordinate domains.
assert!(tree.insert(1, 1, 1).unwrap()); // true: newly inserted
assert!(!tree.insert(0, 0, 1).unwrap()); // false: already present

// An out-of-domain coordinate is rejected, not silently accepted into
// the wrong bit of an unrelated triple -- subject 2 is out of range
// here since only subjects 0 and 1 appeared in the triples this tree
// was built from. Check `tree.in_bounds(s, p, o)` up front if you'd
// rather filter than handle the error.
assert!(tree.insert(2, 0, 0).is_err());

// Persist and reload -- the format is backend-agnostic: bytes from a
// `Heap` deserialize correctly into a `File` or `Compressed` tree too.
let bytes = tree.serialize();
let restored = Heap::deserialize(&bytes).unwrap();
assert!(restored.contains(0, 0, 1));
```

Swap `Heap` for `s_graph::File` or `s_graph::Compressed` (behind their
features) to change *only* the storage backend — every method above is
identical across all three; see `src/storage_file.rs`/
`src/storage_compressed.rs` for what each one actually trades off.

### Joins and graph traversal

```rust
use s_graph::graph::{JoinQuery, Term, TriplePattern, join_run};
use s_graph::path::bfs_sequential;
use s_graph::Heap;

let tree = Heap::build_dynamic(vec![(0, 0, 1), (1, 1, 2), (0, 0, 3)], 4096);

// Collect a native prefix-join result: "subject 0 --p0--> ?a . ?a --p1--> ?b"
let query = JoinQuery {
    num_vars: 2,
    patterns: vec![
        TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
        TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
    ],
};
let bindings = join_run(&query, &tree); // Vec<Vec<u32>>, one Vec per match, indexed by variable

// BFS / shortest hop-count reachability.
let dist = bfs_sequential(&tree, /* start */ 0, /* predicate filter */ None);
```

For joins directly over the S/P/O topology, compile the global prefix schedule
and stream borrowed bindings. Predicates remain structural, including a predicate
variable shared across atoms. Compilation does not scan or copy graph relations:

```rust
use std::ops::ControlFlow;
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::native_join::{NativeJoin, NativeJoinWorkspace};
use s_graph::path::{HopRange, Step, PathWorkspace, has_path};

let tree = s_graph::Heap::build_dynamic(vec![(0, 0, 1), (1, 0, 2)], 4096);
let query = JoinQuery {
    num_vars: 4,
    patterns: vec![
        TriplePattern { subject: Term::Var(0), predicate: Term::Var(1), object: Term::Var(2) },
        TriplePattern { subject: Term::Var(2), predicate: Term::Var(1), object: Term::Var(3) },
    ],
};
let plan = NativeJoin::new(&query, &tree).unwrap();
let mut workspace = NativeJoinWorkspace::default();
let _: ControlFlow<()> = plan.visit(&mut workspace, |binding| {
    // Consume binding immediately; Break stops enumeration.
    assert_eq!(binding, &[0, 0, 1, 2]);
    ControlFlow::Continue(())
});
let mut paths = PathWorkspace::default(); // sparse visitation, independent of max ID
assert!(has_path(&tree, 0, 2, HopRange::new(2, 2).unwrap(), Step::default(), &mut paths));
```

`NativeJoin` uses O(atoms × tree height + variables) metadata and workspace,
independent of the number of triples. It supports constants, repeated variables,
cycles, ground constraints and disconnected patterns. The plan borrows its tree;
drop it before mutation and compile a new plan afterwards. Output order is
unspecified. There is no established WCO bound for this native traversal, and
uneven split schedules can delay pruning. `visit_with_stats` reports node
openings and surviving child choices. `allocated_bytes` reports retained vector
capacity for the plan or workspace; backend cache memory is separate.

`graph::join_run` and `graph::try_join_run` compile and collect native joins.
The historical names `ltj_run` and `try_ltj_run` are aliases for those helpers;
they no longer execute Leapfrog Triejoin. Reuse `NativeJoin` and its workspace
for repeated queries. Native joins accept the tree's `BlockStorage` backends;
`join::PatternBackend` remains the generic scan interface for path algorithms.

The materialized `PreparedJoin`, its `JoinOptions`, `RelationLayout` and
`JoinWorkspace`, and the boxed `GraphBackend`/sorted-neighbor adapters have been
removed. Migrate prepared queries to `NativeJoin`/`NativeJoinWorkspace`; plans
now borrow the tree rather than own snapshots. No proven WCO executor is currently
provided; proving or improving the native traversal's bound remains open work.

- `tree.scan([s, p, o])`: allocation-free filtered triples in Morton order.
- `query_one_bound_into` / `query_two_bound_into`: sorted adjacency in reusable buffers.
- `path::reachable`: distinct endpoints reachable by walks within min/max hops.
- `path::shortest_walk`: shortest qualifying walk with nodes and edges reconstructed.
- `path::enumerate_paths`: streamed walks, trails or simple paths, with optional target.
- `automaton::PathAutomaton`: epsilon-free NFA evaluation for label sequences,
  alternatives and repetition, with hop bounds and distinct endpoint output.

Traversal supports outgoing, incoming or both directions. Identical triples are
one edge; edge multiplicity must be encoded by the consumer. Supplied start IDs
are considered valid for zero-hop paths, including externally managed isolated
vertices. Path enumeration can have exponential output; callbacks can stop early.

All serial algorithms support every storage backend. `bfs_parallel` requires
a `Sync` backend: `Heap` qualifies, while `File` and
`Compressed` retain single-threaded interior caches. Native plans borrow the
backend and inherit its threading constraints; Heap plans can be shared with
separate workspaces.

See [DESIGN.md](DESIGN.md) for complexity, memory, semantics and tradeoffs.

### Error handling

`insert`/`insert_batch`, `build`/`build_with_order`, and
`deserialize*` return `Result<_, TreeError>`. Read methods
(`contains`, `query_two_bound`, `query_one_bound`, `predicates_for`)
return false/empty results for out-of-domain coordinates, which cannot match
anything. File I/O failures during traversal still panic.
Snapshot loading validates metadata, topology, record boundaries and skip spans.
Default limits are 1 GiB of block capacity, 1,000,000 blocks and 16 MiB per block;
use `deserialize_with_limits` / `deserialize_from_path_with_limits` to change them.
Configuration-taking infallible builders panic on invalid configuration; `build`
and `build_with_order` return errors. See crate docs for the remaining panic contract.

## Getting real data in

`s-graph` indexes `(u32, u32, u32)` triples; it doesn't parse RDF syntax
or intern terms itself. For real IRIs/literals, parse with a real RDF
parser and dictionary-encode the terms you get back into `u32` ids
(one shared id space for subjects/objects, since the same IRI can appear
in both positions; blank nodes scoped per source document, since their
identity is only meaningful within it).

## Benchmarks

[`s-bench`](https://github.com/hrssilva/s-bench) is a separate repository
that builds and queries this crate (`Heap`/`File`/`Compressed`, all three)
against the real [W3C RDF 1.1 test suite](https://github.com/w3c/rdf-tests),
parsed with [`s-parser`](https://github.com/hrssilva/s-parser) — real
IRIs/literals dictionary-encoded into triples, not this project's own
synthetic dataset generator. It's a separate repository specifically so
neither it nor its private dependencies ever become something
`cargo build`/`cargo test` on this crate needs to resolve. See its README
for current numbers (build time, memory, `contains` latency, all three
backends, at both the real corpus's own small scale and amplified to
~1.5M triples) and exactly how to reproduce them.

## Project layout

- `src/` — the published library (this crate).
- `examples/` — the exploratory comparison this design was measured
  against (Mk²-tree, IK²-tree, Omnitree, and every intermediate
  RadixTree layout variant), kept for its literature grounding and
  measured history. Not part of the public API.

## Testing

```sh
cargo test                 # default features (heap)
cargo test --all-features  # + file, compressed, parallel
cargo bench --bench algorithms  # preparation, repeated execution, scans and paths
```

## License

MIT — see `LICENSE`.
