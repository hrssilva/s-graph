# s-graph

A succinct, insertable index for RDF triples `(subject, predicate,
object)`: a DFS-preorder, interleaved-skip-pointer radix trie over
Morton-coded triples, generic over three storage backends (in-RAM,
real-file, LZ4-compressed in-RAM), with real `O(height)` insertion — not
the full-rebuild-only limitation most compact triple indexes have.

It grew out of an exploratory comparison against the k²-tree family
(Mk²-tree, IK²-tree) and Omnitree, kept in `examples/` along with the
full survey and literature grounding (see `examples/README.md`). This
crate (`src/`) is the productionized result: the same structure, with
`Result`-based error handling, feature-gated backends, and a real public
API instead of a benchmark harness. `PRODUCTION_ASSESSMENT.md` has the
full design rationale and the decisions made turning it into this crate.

## Why

Compact RDF indexes (k²-tree-family structures especially) are usually
build-once, query-forever: adding one triple means rebuilding the whole
structure. `s-graph`'s `insert`/`insert_batch` cost `O(block_capacity +
height)` — microseconds, not a rebuild — while keeping the same
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

// Real O(height) insertion, not a rebuild.
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
use s_graph::graph::{JoinQuery, Term, TriplePattern, ltj_run};
use s_graph::path::bfs_sequential;
use s_graph::Heap;

let tree = Heap::build_dynamic(vec![(0, 0, 1), (1, 1, 2), (0, 0, 3)], 4096);

// Worst-case-optimal (leapfrog triejoin) multiway join: "subject 0 --p0--> ?a . ?a --p1--> ?b"
let query = JoinQuery {
    num_vars: 2,
    patterns: vec![
        TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
        TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
    ],
};
let bindings = ltj_run(&query, &tree); // Vec<Vec<u32>>, one Vec per match, indexed by variable

// BFS / shortest hop-count reachability.
let dist = bfs_sequential(&tree, /* start */ 0, /* predicate filter */ None);
```

`GraphBackend`/`AdjacencyBackend` are implemented for every `BlockStorage`
backend, so joins and BFS work identically on `Heap`, `File`, and
`Compressed`. Enable the `parallel` feature for `path::bfs_parallel`
(level-synchronous, `rayon`-based).

### Error handling

`insert`/`insert_batch`, `build`/`build_with_order`, and
`deserialize*` return `Result<_, TreeError>`. Everything else
(`contains`, `query_two_bound`, `query_one_bound`, `predicates_for`)
can't fail — an out-of-domain query coordinate simply can't match
anything, so they return an empty/`false` result instead of erroring.
See `src/lib.rs`'s crate docs for exactly which internal failures stay
panics (mid-traversal I/O against an already-built `File`/`Compressed`
tree, mainly) and why.

## Getting real data in

`s-graph` indexes `(u32, u32, u32)` triples; it doesn't parse RDF syntax
or intern terms itself. For real IRIs/literals, parse with a real RDF
parser and dictionary-encode the terms you get back into `u32` ids
(one shared id space for subjects/objects, since the same IRI can appear
in both positions; blank nodes scoped per source document, since their
identity is only meaningful within it).

The `w3c-demo/` directory in this repository is a complete, runnable
example of exactly that pipeline, using
[`s-parser`](https://github.com/hrssilva/s-parser) (a separate,
from-scratch N-Triples parser) instead of `examples/`'s own synthetic
data generator — see below.

## Demonstration: real W3C RDF data, not synthetic

Every number below comes from actually parsing RDF — the [W3C RDF 1.1
test suite](https://github.com/w3c/rdf-tests) (the same corpus
`s-parser` validates its own conformance against), read with
`s-parser`'s `NtTokenizer`, dictionary-encoded, and built into each
`s-graph` backend. Reproduce it yourself:

```sh
cd w3c-demo
./fetch_w3c_data.sh          # shallow-clones w3c/rdf-tests
cargo run --release -- ./rdf-tests [replicas]
```

**Pure real data, zero amplification** (328 fixture files, one pass —
`replicas` defaults to 1):

```
809 records parsed (1 file skipped, see stderr), 758 distinct triples, 751 distinct entities, 117 distinct predicates, in 5.8ms

Heap        build= 459.974µs mem=      9137 bytes (12.05 bytes/triple) height=5 blocks=3  contains_avg=  292ns (758/758 probes hit)
File        build= 550.926µs mem=       155 bytes (0.20 bytes/triple) height=5 blocks=3  contains_avg=1.528µs (758/758 probes hit)
Compressed  build= 644.311µs mem=     12497 bytes (16.49 bytes/triple) height=5 blocks=3  contains_avg=3.794µs (758/758 probes hit)

`rdf:type` used as a predicate for 37 distinct object(s) in this corpus (query_one_bound)
```

(The one skipped file is a character-boundary fixture that deliberately
encodes a raw UTF-16 surrogate half — not a valid Unicode scalar value.
`s-parser` correctly rejects it rather than silently accepting malformed
input; the demo logs it and moves on to the other 327 files.)

**Amplified to demonstration scale** (`replicas = 2000`): each replica
gets its own subject namespace (so triples stay genuinely distinct, not
duplicate rows collapsing in the set), while every predicate and object
stays on the real, bounded W3C vocabulary — see `w3c-demo/src/main.rs`'s
`replicate_key` for exactly what this does and doesn't change. Parsing
1.6M records this way (re-reading the same small files 2000 times) takes
~7.3s, dominated by that repetition, not by `s-parser`'s own throughput —
see [`s-parser`'s own benchmarks](https://github.com/hrssilva/s-parser)
for real parsing-speed numbers on non-repeated input.

```
1618000 records parsed, 1516000 distinct triples, 796353 distinct entities, 117 distinct predicates, in 7.3s

Heap        build=163.015954ms mem=  22061077 bytes (14.55 bytes/triple) height=7 blocks=5359 contains_avg=  755ns (2000/2000 probes hit)
File        build=179.141994ms mem=    171561 bytes (0.11 bytes/triple) height=7 blocks=5359 contains_avg= 8.32µs (2000/2000 probes hit)
Compressed  build=217.618461ms mem=   7416542 bytes (4.89 bytes/triple) height=7 blocks=5359 contains_avg=23.775µs (2000/2000 probes hit)
```

`File`'s `mem_size_bytes` (RAM only, not disk — the tree content itself
lives in the backing file, not in this number) staying two to three
orders of magnitude below `Heap`/`Compressed` at both scales, and
`Compressed` landing at roughly a third of `Heap`'s footprint, are the
same backend tradeoffs `examples/README.md`'s own large-scale synthetic
benchmarks measured — this just confirms they hold on real parsed RDF,
not only generated data.

Measured on the machine this repository was developed on; re-run the
commands above for your own hardware.

## Project layout

- `src/` — the published library (this crate).
- `examples/` — the exploratory comparison this design was measured
  against (Mk²-tree, IK²-tree, Omnitree, and every intermediate
  RadixTree layout variant), kept for its literature grounding and
  measured history. Not part of the public API.
- `w3c-demo/` — the real-data demonstration above. A separate Cargo
  project (not a workspace member) because it depends on the private
  `s-parser` repository over SSH; `cargo build`/`cargo test` from this
  repository's root never touches it.
- `PRODUCTION_ASSESSMENT.md` — the design assessment and the decisions
  made turning the `examples/` prototype into `src/` (error-handling
  boundaries, feature gating, what got dropped and why, migration notes).

## Testing

```sh
cargo test                 # default features (heap)
cargo test --all-features  # + file, compressed, parallel
```

## License

MIT — see `LICENSE`.
