//! Compact, insertable integer graph storage and algorithms over
//! `(subject, predicate, object)` triples, generic over disk/RAM/compression.
//! Terms, named graphs, query languages and transactions belong to consumers.
//!
//! # What this is
//!
//! A DFS-preorder, interleaved-skip-pointer radix trie over Morton-coded
//! triples: each node is an 8-byte existence bitmap (up to 64-way
//! branching per level) plus, for nodes more than 2 levels from a leaf,
//! an inline variable-width (1/2/4/8-byte) skip pointer giving the byte
//! distance to jump past that node's entire subtree. Same-block navigation
//! uses direct offsets; a compact Fenwick directory resolves cross-block spans
//! in O(log blocks), without rank/select over node records.
//! A per-dimension bit-consumption schedule is computed once from each
//! dimension's domain size, before any data exists; a dimension stops
//! contributing digits once its own domain is exhausted, so the tree
//! degenerates gracefully on skewed/concentrated data instead of paying
//! the k²-tree family's unconditional-fan-out traversal cost.
//!
//! Three storage backends share one generic implementation via
//! [`storage::BlockStorage`], monomorphized (not boxed/dyn-dispatched)
//! per backend:
//! - [`Heap`] (feature `heap`, on by default): in-RAM, no slot
//!   indirection, fastest, full RAM footprint.
//! - [`File`] (feature `file`): a real backing file, `pread`/`pwrite`,
//!   near-zero steady-state RAM.
//! - [`Compressed`] (feature `compressed`): LZ4-compressed in-RAM blocks,
//!   for RAM-constrained deployments with no disk available at all.
//!
//! Local [`Heap::insert`]/[`Heap::insert_batch`] is supported
//! on every backend, unlike the from-scratch-rebuild-only structures this
//! design is descended from. A streaming/parallel/RAM-bounded bulk build
//! ([`Heap::build_dynamic`], [`Heap::build_dynamic_bounded`]) is also
//! shared across backends. [`Heap::remove`]/[`Heap::remove_batch`] delete
//! a triple with full, immediate cascading reclamation of any node left
//! entirely empty by the removal -- not a leaf-only bit flip deferred to
//! a separate compaction step. [`Heap::contains_batch`] answers many
//! point-existence queries (filters, graph diff) by sharing descent work
//! across queries that land on the same trie node, instead of
//! independently re-descending from the root for each one.
//! [`Heap::count_matching`] counts a pattern's matches directly, without
//! decoding or collecting them the way [`Heap::scan`] must to list them.
//!
//! This crate is the productionized form of an exploratory comparison
//! (see the repository's `examples/` directory and its own `README.md`
//! for the full survey and literature grounding this design was measured
//! against). Ancestry lineage: k²-trees (Brisaboa/Ladra/Navarro), the
//! k^d-tree-as-trie observation (Arroyuelo et al.), Omnitree's per-node
//! adaptive dimension selection (Pollinger/Ishii/Domke), and the UB-tree
//! literature's precedent for a Z-order encoding that stops interleaving
//! an exhausted dimension (Bayer/Markl). This structure's own point of
//! departure: the subdivision schedule is fixed in advance from each
//! dimension's domain size (not adaptively chosen per node), and a
//! dimension may contribute multiple coordinate bits per radix step
//! (mixed-radix, not strictly dyadic).
//!
//! # Which structure to use
//!
//! For a **static, build-once, query-and-join-heavy** workload, prefer
//! [`path_fst::PathFstDeltaPow2Bfs`] (feature `heap`, same default as
//! [`Heap`]) over `Heap` itself: smaller on disk/in RAM for the same
//! triples (the whole point of its DAG deduplication). A real
//! benchmark against production-shaped (power-law) data found this
//! isn't a strict win on every query shape, though -- see `DESIGN.md`'s
//! "wired as default" entry and its "a benchmark report, and a
//! corrected recommendation" follow-up for the honest numbers both
//! directions, not just the favorable ones.
//!
//! For restricted-descent-style pattern queries (`Exact`/`Free`/
//! `Range`/`Multi` on any position), use
//! [`path_fst::PathFstDeltaPow2Bfs::restricted_descent_via_composed_automaton`] --
//! it's at parity with `restricted_descent` for the common case and
//! strictly more capable, on every query shape tried. One real caveat:
//! a wide, `Free`-heavy query under a high-fanout predicate (`?s p ?o`
//! with `p` bound, `s`/`o` free) measured ~2.6-2.7x slower on this
//! structure than the equivalent `Heap` query on skewed real-world-shaped
//! data -- a genuine cost of this structure's DAG deduplication for that
//! access pattern, not fixed here. Benchmark that shape specifically if
//! it matters for your workload.
//!
//! For conjunctive joins, **default to [`fst_join::FstJoin`] or
//! [`recursive_join::RecursiveJoin`]** (same algorithm, same measured
//! behavior -- interleaved by level like [`native_join::NativeJoin`],
//! robust across every shape tried including adversarial skew).
//! [`automaton_join::AutomatonJoin`] measures 1.7-50x *faster* when
//! every atom's own search space is comparably selective, but measured
//! 8-16x *slower* on a realistic shape (one atom structurally
//! constant-heavy but not actually low-cardinality on the real data,
//! like `?x rdf:type :SomeClass` where that class covers a large
//! fraction of all subjects, joined with a much rarer predicate) --
//! confirmed on a real benchmark report, not a synthetic worst case.
//! Use it only when you've confirmed your own join atoms are
//! comparably selective; default to `FstJoin`/`RecursiveJoin`
//! otherwise, the same way you'd default to `NativeJoin` on `Heap`.
//!
//! **Use [`Heap`]/[`File`]/[`Compressed`] instead** for anything
//! `PathFstDeltaPow2Bfs` doesn't support: local
//! [`Heap::insert`]/[`Heap::remove`] after the initial build,
//! [`Heap::serialize`]/[`Heap::deserialize`], [`Heap::contains_batch`]/
//! [`Heap::count_matching`], or graph traversal/paths
//! ([`traversal`]) and regular-path automata ([`automaton`]), none of
//! which `PathFstDeltaPow2Bfs` implements (it isn't a
//! [`join::PatternBackend`]). `PathFstDeltaPow2Bfs` is a query/join
//! index over a structure built once; the mutable tree family below is
//! everything else.
//!
//! # Error handling: what returns `Result`, what panics
//!
//! [`error::TreeError`] covers three kinds of fallible boundary:
//! - **Out-of-domain coordinates** ([`Heap::insert`]/
//!   [`Heap::insert_batch`], and [`Heap::build_with_order`]'s `split`
//!   parameter). Check [`Heap::in_bounds`] up front if you'd rather
//!   filter than handle the error. Read methods (`contains`,
//!   `restricted_descent`) never need this: an
//!   out-of-domain query coordinate simply can't match anything, so they
//!   already return an empty/`false` result rather than erroring.
//!   [`Heap::remove`]/[`Heap::remove_batch`] also return `Result` (for
//!   signature symmetry with `insert`/`insert_batch`) but follow the
//!   read-method rule here, not insert's: `Ok(false)` for an out-of-domain
//!   coordinate, never `Err`, since removal never changes the domain
//!   either way.
//! - **Corrupt or truncated serialized input** ([`Heap::deserialize`]
//!   and friends) -- this is genuinely untrusted external data (a
//!   truncated file, a bit-flipped transfer), so it returns `Result`
//!   rather than panicking.
//! - **I/O during deserialization** (`file`/`compressed` backends
//!   reconstructing their backing storage from a byte stream).
//!
//! Everything else -- a mid-traversal disk read/write failure against an
//! already-built `File`/`Compressed` tree, or an internal invariant the
//! build path itself already guarantees (e.g. a node record never
//! exceeding `block_capacity`) -- stays a panic. Operational I/O errors are
//! not yet exposed as recoverable traversal errors; consumers must account for
//! that backend contract. Deserialization validates structure and enforces
//! configurable resource limits before exposing a tree.
//!
//! Native Morton-prefix joins live in [`join`] and [`native_join`];
//! bounded walks, path reconstruction
//! and enumeration in [`traversal`]; regular-path automata in [`automaton`].
//!
//! # Feature flags
//!
//! `heap` (default, no extra dependencies), `file`, `compressed` (pulls
//! in `lz4_flex`), `parallel` (pulls in `rayon`, enables a measured ~4x
//! bulk-build speedup at large N via a size-gated fan-out across
//! independent subtrees -- off by default so a single-threaded consumer
//! isn't forced to accept the dependency).

#![forbid(unsafe_code)]

pub mod error;
mod block_index;
mod level_automaton;
pub mod automaton;
pub mod graph;
pub mod join;
pub mod native_join;
pub mod path;
#[cfg(feature = "heap")]
pub mod automaton_join;
#[cfg(feature = "heap")]
pub mod fst_join;
#[cfg(feature = "heap")]
pub mod recursive_join;
#[cfg(feature = "heap")]
pub mod path_fst;
pub mod serialize;
pub mod storage;
#[cfg(feature = "compressed")]
pub mod storage_compressed;
#[cfg(feature = "file")]
pub mod storage_file;
pub mod tree;
pub mod traversal;

#[cfg(all(test, feature = "heap"))]
mod test_support;

pub use error::{OutOfDomain, Result, TreeError};
pub use storage::BlockStorage;
pub use tree::{BulkBuildBudget, Triple, DEFAULT_BLOCK_CAPACITY};

#[cfg(feature = "compressed")]
pub use tree::Compressed;
#[cfg(feature = "file")]
pub use tree::File;
#[cfg(feature = "heap")]
pub use tree::Heap;

/// Actual allocated-capacity bytes of a `Vec`, not just its logical
/// length -- what actually shows up in the allocator, used throughout
/// this crate's `mem_size_bytes`/`size_breakdown` introspection methods.
pub(crate) fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}
