//! Graph patterns and compatibility adjacency adapters.
//! General joins use [`crate::join::PreparedJoin`]: packed ordered relations
//! prepared once, with allocation-free leapfrog execution using a workspace.

use crate::storage::BlockStorage;
use crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric;

/// A sorted, deduplicated stream of `u32` ids with the two primitives a
/// leapfrog join needs: read the current value, and seek forward.
pub trait SortedNeighbors {
    fn current(&self) -> Option<u32>;
    fn advance(&mut self);
    fn seek(&mut self, target: u32);
}

/// Wraps an already-sorted, already-materialized `Vec<u32>` (what
/// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::query_two_bound`]
/// returns -- this structure has no lazy row/col iterator, unlike a
/// k²-tree's subtree-pruning one) into a [`SortedNeighbors`], with `seek`
/// implemented as a binary search over the remaining slice.
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
        self.pos = (self.pos + 1).min(self.data.len());
    }
    fn seek(&mut self, target: u32) {
        let offset = self.data[self.pos..].partition_point(|&v| v < target);
        self.pos += offset;
    }
}

/// Compatibility row/column adapters. General joins and bounded paths use
/// [`crate::join::PatternBackend`] for scans in all three dimensions.
pub trait GraphBackend {
    /// Ascending objects `o` such that `(subject, predicate, o)` is present.
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a>;
    /// Ascending subjects `s` such that `(s, predicate, object)` is present.
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a>;
}

impl<S: BlockStorage> GraphBackend for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    fn stream_row<'a>(&'a self, predicate: u32, subject: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([0, 1], [subject, predicate], 2)))
    }
    fn stream_col<'a>(&'a self, predicate: u32, object: u32) -> Box<dyn SortedNeighbors + 'a> {
        Box::new(SortedVecIter::new(self.query_two_bound([1, 2], [predicate, object], 0)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    Var(usize),
    Const(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct TriplePattern<P = u32> {
    pub subject: Term,
    pub predicate: P,
    pub object: Term,
}

pub struct JoinQuery<P = u32> {
    pub num_vars: usize,
    pub patterns: Vec<TriplePattern<P>>,
}

impl From<u32> for Term {
    fn from(value: u32) -> Self { Self::Const(value) }
}

/// Convenience collection API. For repeated execution or bounded output memory,
/// use `PreparedJoin` and `JoinWorkspace` directly. Invalid queries panic;
/// `try_ltj_run` returns a validation error instead.
pub fn ltj_run<B: crate::join::PatternBackend, P: Copy + Into<Term>>(query: &JoinQuery<P>, backend: &B) -> Vec<Vec<u32>> {
    try_ltj_run(query, backend).expect("invalid join query")
}

pub fn try_ltj_run<B: crate::join::PatternBackend, P: Copy + Into<Term>>(query: &JoinQuery<P>, backend: &B) -> crate::Result<Vec<Vec<u32>>> {
    let prepared = crate::join::PreparedJoin::new(query, backend)?;
    let mut workspace = crate::join::JoinWorkspace::default();
    let mut out = Vec::new();
    let _: std::ops::ControlFlow<()> = prepared.visit(&mut workspace, |row| {
        out.push(row.to_vec());
        std::ops::ControlFlow::Continue(())
    });
    Ok(out)
}

#[cfg(all(test, feature = "heap"))]
mod tests {
    use super::*;
    use crate::test_support::{uniform, NaiveIndex};
    use crate::Triple;

    fn naive_join(query: &JoinQuery, naive: &NaiveIndex, all_nodes: &[u32]) -> Vec<Vec<u32>> {
        fn rec(query: &JoinQuery, naive: &NaiveIndex, all_nodes: &[u32], var_idx: usize, bindings: &mut Vec<Option<u32>>, out: &mut Vec<Vec<u32>>) {
            if var_idx == query.num_vars {
                let candidate: Vec<u32> = bindings.iter().map(|b| b.unwrap()).collect();
                let ok = query.patterns.iter().all(|pat| {
                    let s = match pat.subject {
                        Term::Const(c) => c,
                        Term::Var(v) => candidate[v],
                    };
                    let o = match pat.object {
                        Term::Const(c) => c,
                        Term::Var(v) => candidate[v],
                    };
                    naive.contains(s, pat.predicate, o)
                });
                if ok {
                    out.push(candidate);
                }
                return;
            }
            for &node in all_nodes {
                bindings[var_idx] = Some(node);
                rec(query, naive, all_nodes, var_idx + 1, bindings, out);
            }
            bindings[var_idx] = None;
        }
        let mut out = Vec::new();
        let mut bindings = vec![None; query.num_vars];
        rec(query, naive, all_nodes, 0, &mut bindings, &mut out);
        out
    }

    fn sorted(mut v: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        v.sort_unstable();
        v
    }

    fn check_heap(triples: &[Triple], query: &JoinQuery, all_nodes: &[u32]) {
        let naive = NaiveIndex::build(triples);
        let expected = sorted(naive_join(query, &naive, all_nodes));
        assert!(!expected.is_empty(), "test dataset/query produced zero matches -- not exercising real join logic");

        let heap = crate::Heap::build_dynamic(triples.to_vec(), 4096);
        assert_eq!(sorted(ltj_run(query, &heap)), expected, "Heap join mismatch");
    }

    // Anchored compatibility cases; general patterns are tested in join.rs.

    #[test]
    fn two_hop_path_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (0, 0, 3), (1, 1, 10), (1, 1, 11), (2, 2, 20)];
        let all_nodes: Vec<u32> = (0..30).collect();
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }, TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) }] };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn triangle_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 1, 4), (3, 2, 0)];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
                TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Const(0) },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn four_cycle_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 5), (1, 1, 2), (5, 1, 6), (2, 2, 3), (6, 2, 7), (3, 3, 0)];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 3,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
                TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Var(2) },
                TriplePattern { subject: Term::Var(2), predicate: 3, object: Term::Const(0) },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn star_pattern_with_bound_center() {
        let triples = uniform(1, 3, 12, 30, 1717);
        let mut triples: Vec<Triple> = triples.into_iter().map(|(_, p, o)| (0, p, o)).collect();
        triples.push((0, 0, 99));
        let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0);
        let all_nodes: Vec<u32> = (0..=max_node).collect();
        let query = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }] };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn streams_agree_with_naive_and_are_sorted() {
        let triples = uniform(12, 4, 12, 200, 999);
        let naive = NaiveIndex::build(&triples);
        let heap = crate::Heap::build_dynamic(triples, 4096);

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
                assert_eq!(drain(heap.stream_row(p, s)), naive.row(p, s), "row p={p} s={s}");
            }
            for o in 0..12u32 {
                assert_eq!(drain(heap.stream_col(p, o)), naive.col(p, o), "col p={p} o={o}");
            }
        }
    }

    #[test]
    fn seek_matches_linear_skip() {
        let triples = uniform(20, 3, 20, 300, 555);
        let heap = crate::Heap::build_dynamic(triples, 4096);

        for p in 0..3u32 {
            for s in 0..20u32 {
                let full: Vec<u32> = {
                    let mut v = Vec::new();
                    let mut it = heap.stream_row(p, s);
                    while let Some(x) = it.current() {
                        v.push(x);
                        it.advance();
                    }
                    v
                };
                for &target in &[0u32, 5, 10, 15, 20] {
                    let expect: Vec<u32> = full.iter().copied().filter(|&v| v >= target).collect();
                    let mut it = heap.stream_row(p, s);
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
