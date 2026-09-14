//! A shared, structure-agnostic interface over "sorted neighbor" queries,
//! letting the same join ([`crate::common::join`]) and path
//! ([`crate::common::path`]) algorithms run unmodified over Mk²-tree,
//! IK²-tree, Omnitree, and the [`RadixTree`] prototype.
//!
//! The demo join/path patterns (2-hop paths, triangles, stars, BFS) always
//! keep the predicate bound as a query-level constant and only join on
//! subject/object -- exactly the shape a k²-tree-family row/col query
//! already answers. That lets [`GraphBackend`] stay to just two methods
//! (row: objects for a bound subject; col: subjects for a bound object)
//! instead of needing to support an unbound predicate as a join variable.
//!
//! Streams are boxed (`Box<dyn SortedNeighbors>`) rather than a
//! per-backend closed enum: `K2Tree`'s row/col iterators, `Ik2Tree`'s and
//! `Omnitree`'s materialized-`Vec` results wrap into genuinely different
//! concrete types, and a trait object keeps `join`/`path` simple at the
//! cost of one vtable call per `advance`/`seek` -- negligible next to the
//! tree traversal work it wraps.

use crate::common::ik2tree::Ik2Tree;
use crate::common::k2tree::{ColIter, RowIter};
use crate::common::mk2tree::Mk2Forest;
use crate::common::omnitree::Omnitree;
use crate::common::radixtree::RadixTree;

/// A sorted, deduplicated stream of `u32` ids with the two primitives a
/// leapfrog join needs: read the current value, and seek forward.
pub trait SortedNeighbors {
    fn current(&self) -> Option<u32>;
    fn advance(&mut self);
    fn seek(&mut self, target: u32);
}

/// An `Iterator<Item = u32>` that also supports skipping whole subtrees via
/// a `seek`, implemented by [`K2Tree`]'s row/col iterators.
pub trait SeekableIterator: Iterator<Item = u32> {
    fn seek_to(&mut self, target: u32);
}

impl SeekableIterator for RowIter<'_> {
    fn seek_to(&mut self, target: u32) {
        self.seek(target);
    }
}
impl SeekableIterator for ColIter<'_> {
    fn seek_to(&mut self, target: u32) {
        self.seek(target);
    }
}

/// Wraps any [`SeekableIterator`] into a [`SortedNeighbors`], caching the
/// current value so `current()` can be read without consuming it.
pub struct Stream<I: SeekableIterator> {
    iter: I,
    current: Option<u32>,
}

impl<I: SeekableIterator> Stream<I> {
    pub fn new(mut iter: I) -> Self {
        let current = iter.next();
        Stream { iter, current }
    }
}

impl<I: SeekableIterator> SortedNeighbors for Stream<I> {
    fn current(&self) -> Option<u32> {
        self.current
    }
    fn advance(&mut self) {
        self.current = self.iter.next();
    }
    fn seek(&mut self, target: u32) {
        if self.current.is_some_and(|c| c >= target) {
            return;
        }
        self.iter.seek_to(target);
        self.current = self.iter.next();
    }
}

/// Wraps an already-sorted, already-materialized `Vec<u32>` (what
/// [`Ik2Tree`] and [`Omnitree`]'s query methods return) into a
/// [`SortedNeighbors`], with `seek` implemented as a binary search over the
/// remaining slice. Not as cheap as [`K2Tree`]'s true subtree-skipping
/// `seek` (the whole result is materialized up front), but correct and
/// still O(log n) per seek -- a deliberate simplification given these two
/// structures don't yet have lazy row/col iterators.
pub struct SortedVecIter {
    data: Vec<u32>,
    pos: usize,
}

impl SortedVecIter {
    pub fn new(data: Vec<u32>) -> Self {
        SortedVecIter { data, pos: 0 }
    }
}

impl SortedNeighbors for SortedVecIter {
    fn current(&self) -> Option<u32> {
        self.data.get(self.pos).copied()
    }
    fn advance(&mut self) {
        self.pos += 1;
    }
    fn seek(&mut self, target: u32) {
        let offset = self.data[self.pos..].partition_point(|&v| v < target);
        self.pos += offset;
    }
}

/// Row/col-style bound-predicate queries, uniform across all three
/// structures, used as the shared basis for both joins and path
/// algorithms.
pub trait GraphBackend {
    /// Ascending objects `o` such that `(subject, predicate, o)` is present.
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a>;
    /// Ascending subjects `s` such that `(s, predicate, object)` is present.
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a>;
}

impl GraphBackend for Mk2Forest {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        match self.tree_for(predicate) {
            Some(t) => Box::new(Stream::new(t.row_iter(subject))),
            None => Box::new(SortedVecIter::new(Vec::new())),
        }
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        match self.tree_for(predicate) {
            Some(t) => Box::new(Stream::new(t.col_iter(object))),
            None => Box::new(SortedVecIter::new(Vec::new())),
        }
    }
}

impl GraphBackend for Ik2Tree {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.row_bound(predicate, subject)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.col_bound(predicate, object)))
    }
}

impl GraphBackend for Omnitree {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for RadixTree {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

/// Same body as the baseline's impl three times over -- see the matching
/// comment in `path.rs`'s `AdjacencyBackend` impls for why this isn't one
/// blanket `impl<T: RadixTreeApi>`.
impl GraphBackend for crate::common::radixtree_veb::RadixTreeVeb {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_dfs::RadixTreeDfs {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_bp::RadixTreeBp {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_dfs_bp::RadixTreeDfsBp {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_rankdelta::RadixTreeRankDelta {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

/// Same materialize-then-wrap pattern as every other `RadixTree` variant
/// above (none of them have a lazy row/col iterator) -- added later than
/// the rest, since `RadixTreeDfsInterleavedVarSkipUnrolled` is this
/// project's newest storage-comparison structure (see
/// `crate::common::insertable_bench`'s `heap`/`file`/`mk2`/`ik2`
/// benchmarks), not present when this trait was first written.
impl GraphBackend for crate::common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

impl GraphBackend for crate::common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::uniform;
    use crate::common::reference::NaiveIndex;

    #[test]
    fn streams_agree_with_naive_and_are_sorted() {
        let triples = uniform(12, 4, 12, 200, 999);
        let naive = NaiveIndex::build(&triples);
        let mk2 = Mk2Forest::build(&triples);
        let ik2 = Ik2Tree::build(2, &triples);
        let omni = Omnitree::build(&triples);
        let radix = RadixTree::build(&triples, RadixTree::default_split());

        fn drain(mut s: Box<dyn SortedNeighbors + '_>) -> Vec<u32> {
            let mut out = Vec::new();
            while let Some(v) = s.current() {
                out.push(v);
                s.advance();
            }
            out
        }

        for p in 0..4u32 {
            for s in 0..12u32 {
                let expect = naive.row(p, s);
                assert_eq!(drain(mk2.stream_row(p, s)), expect, "mk2 row p={p} s={s}");
                assert_eq!(drain(ik2.stream_row(p, s)), expect, "ik2 row p={p} s={s}");
                assert_eq!(drain(omni.stream_row(p, s)), expect, "omni row p={p} s={s}");
                assert_eq!(drain(radix.stream_row(p, s)), expect, "radix row p={p} s={s}");
            }
            for o in 0..12u32 {
                let expect = naive.col(p, o);
                assert_eq!(drain(mk2.stream_col(p, o)), expect, "mk2 col p={p} o={o}");
                assert_eq!(drain(ik2.stream_col(p, o)), expect, "ik2 col p={p} o={o}");
                assert_eq!(drain(omni.stream_col(p, o)), expect, "omni col p={p} o={o}");
                assert_eq!(drain(radix.stream_col(p, o)), expect, "radix col p={p} o={o}");
            }
        }
    }

    #[test]
    fn seek_matches_linear_skip_across_backends() {
        let triples = uniform(20, 3, 20, 300, 555);
        let mk2 = Mk2Forest::build(&triples);
        let ik2 = Ik2Tree::build(2, &triples);
        let omni = Omnitree::build(&triples);
        let radix = RadixTree::build(&triples, RadixTree::default_split());

        for p in 0..3u32 {
            for s in 0..20u32 {
                let full: Vec<u32> = {
                    let mut v = Vec::new();
                    let mut it = mk2.stream_row(p, s);
                    while let Some(x) = it.current() {
                        v.push(x);
                        it.advance();
                    }
                    v
                };
                for &target in &[0u32, 5, 10, 15, 20] {
                    let expect: Vec<u32> = full.iter().copied().filter(|&v| v >= target).collect();
                    for backend_stream in [mk2.stream_row(p, s), ik2.stream_row(p, s), omni.stream_row(p, s), radix.stream_row(p, s)] {
                        let mut it = backend_stream;
                        it.seek(target);
                        let mut got = Vec::new();
                        while let Some(x) = it.current() {
                            got.push(x);
                            it.advance();
                        }
                        assert_eq!(got, expect, "seek({target}) p={p} s={s}");
                    }
                }
            }
        }
    }
}
