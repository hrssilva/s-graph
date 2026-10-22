//! Compact integer graph storage with composable automata.
//!
//! With the default `heap` feature, start with `Fst`: a packed, suffix-sharing index over `(u32, u32, u32)`
//! triples. Build `automata::Bgp` or a composed recognizer and apply it through
//! `Fst::search`. `Join` also executes adaptive joins directly on the index;
//! `query_ops` composes BGPs, OPTIONAL, MINUS, unions and existence filters.
//! `path_ops` builds regular path languages, while `automata_core` supplies
//! reusable recognizers, boolean composition, transducers and reducers.
//!
//! `into_mutable` enables local minimality-preserving updates. File and compressed
//! FST modes share the same automata and validated snapshot format. Mutable radix trees are
//! available separately as [`tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric`]
//! with heap, file and compressed backends. See the repository README for
//! runnable examples and DESIGN.md for algorithms and measured tradeoffs.

#![forbid(unsafe_code)]

pub mod automaton;
#[cfg(feature = "heap")]
pub mod automaton_join;
mod block_index;
pub mod error;
#[cfg(feature = "heap")]
pub mod fst_join;
pub mod graph;
pub mod join;
#[cfg(feature = "heap")]
mod level_automaton;
pub mod native_join;
pub mod path;
#[cfg(feature = "heap")]
pub mod path_fst;
#[cfg(feature = "heap")]
pub mod recursive_join;
pub mod serialize;
pub mod storage;
#[cfg(feature = "compressed")]
pub mod storage_compressed;
#[cfg(feature = "file")]
pub mod storage_file;
pub mod traversal;
pub mod tree;

#[cfg(all(test, feature = "heap"))]
mod test_support;

pub use error::{OutOfDomain, Result, TreeError};
pub use storage::BlockStorage;
pub use tree::{BulkBuildBudget, DEFAULT_BLOCK_CAPACITY, Triple};

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

#[cfg(feature = "heap")]
pub mod analytics;
#[cfg(feature = "heap")]
pub mod automata;
#[cfg(feature = "heap")]
pub mod automata_core;
#[cfg(feature = "heap")]
pub mod fst;
#[cfg(feature = "heap")]
pub mod mutable_fst;
#[cfg(feature = "heap")]
pub mod tree_source;
#[cfg(feature = "heap")]
pub use mutable_fst::MutableFst as EditableFst;
#[cfg(feature = "heap")]
pub use mutable_fst::{BlockCompactionStats, FreezeOrder, FrozenFst};
#[cfg(feature = "heap")]
pub type MutableFst = EditableFst<storage::HeapStorage>;
#[cfg(feature = "file")]
pub type MutableFileFst = EditableFst<storage_file::FileStorage>;
#[cfg(feature = "compressed")]
pub type MutableCompressedFst = EditableFst<storage_compressed::CompressedStorage>;
#[cfg(feature = "heap")]
mod frontier_join;
#[cfg(all(test, feature = "heap"))]
mod large_bgp_cases;
#[cfg(feature = "heap")]
pub mod path_ops;
#[cfg(feature = "heap")]
pub mod query_ops;
#[cfg(feature = "heap")]
pub mod scheduled_join;
#[cfg(all(feature = "heap", feature = "compressed"))]
pub use fst::CompressedFst;
#[cfg(all(feature = "heap", feature = "file"))]
pub use fst::FileFst;
#[cfg(feature = "heap")]
pub use fst::Fst;
#[cfg(feature = "heap")]
pub use join::{Join, JoinWorkspace};
#[cfg(feature = "heap")]
pub use path_fst::PackedFst;

#[cfg(all(test, feature = "heap"))]
mod join_oracle_tests;

/// Runnable examples from the repository README.
#[cfg(feature = "heap")]
#[doc = include_str!("../README.md")]
pub mod guide {}
