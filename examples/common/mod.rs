//! Shared code for the Mk²-tree / IK²-tree / Omnitree comparison examples.
//! Included by each example binary via `#[path = "common/mod.rs"] mod common;`
//! (or `../common/mod.rs` from the `compare/` directory example), since
//! Cargo examples are independent binaries that cannot depend on one
//! another directly.
//!
//! `dead_code` is silenced module-wide: this is a shared library used by
//! several independent binaries (three thin per-structure demos plus the
//! full `compare` harness), each of which only exercises the subset of the
//! API it needs -- unlike an actual unused-code smell, that's expected here.
#![allow(dead_code)]

pub mod bitvector;
pub mod dac;
pub mod ik2tree;
pub mod insertable_bench;
pub mod join;
pub mod k2_serialize;
pub mod k2tree;
pub mod metrics;
pub mod mk2tree;
pub mod omnitree;
pub mod path;
pub mod radixtree;
pub mod radixtree_bp;
pub mod radixtree_dfs;
pub mod radixtree_dfs_bp;
pub mod radixtree_dfs_dac;
pub mod radixtree_dfs_interleaved;
pub mod radixtree_dfs_interleaved_varskip;
pub mod radixtree_dfs_interleaved_varskip_unrolled;
pub mod radixtree_dfs_interleaved_varskip_unrolled_arena;
pub mod radixtree_dfs_interleaved_varskip_unrolled_file;
pub mod radixtree_dfs_interleaved_varskip_unrolled_file_mmap;
pub mod radixtree_dfs_interleaved_varskip_unrolled_generic;
pub mod radixtree_dfs_interleaved_varskip_unrolled_mmap;
pub mod radixtree_rankdelta;
pub mod radixtree_veb;
pub mod rdf_gen;
pub mod reference;
pub mod sorted_neighbors;
pub mod tree_serialize;
