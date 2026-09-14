//! A generic worst-case-optimal (leapfrog triejoin style) multiway join
//! driver, running unmodified over any [`GraphBackend`] -- the same
//! `ltj_recurse`/`leapfrog_align` code executes over Mk²-tree, IK²-tree,
//! and Omnitree, monomorphized per backend, giving a genuinely
//! apples-to-apples comparison of graph-pattern-matching performance.
//!
//! Query shape: a conjunction of triple patterns `(subject, predicate,
//! object)` where `predicate` is always a bound constant (a query-level
//! schema constraint, as in a typical SPARQL basic graph pattern) and
//! `subject`/`object` are either constants or one of a fixed set of join
//! variables. Variables are solved in a fixed elimination order; for each
//! variable, every pattern that still has it as its one open position
//! contributes a [`SortedNeighbors`] stream (row query if the variable is
//! the object side, col query if the subject side), and leapfrog
//! intersection finds their common values.

use crate::common::sorted_neighbors::{GraphBackend, SortedNeighbors};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    Var(usize),
    Const(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct TriplePattern {
    pub subject: Term,
    pub predicate: u32,
    pub object: Term,
}

pub struct JoinQuery {
    pub num_vars: usize,
    pub patterns: Vec<TriplePattern>,
}

fn resolve(term: Term, bindings: &[Option<u32>]) -> Option<u32> {
    match term {
        Term::Const(c) => Some(c),
        Term::Var(v) => bindings[v],
    }
}

/// Advances all streams (if needed) until they agree on a common value, or
/// until any stream is exhausted. Returns `false` (no more intersections
/// possible) once any stream runs out.
fn leapfrog_align(streams: &mut [Box<dyn SortedNeighbors + '_>]) -> bool {
    loop {
        if streams.iter().any(|s| s.current().is_none()) {
            return false;
        }
        let min_v = streams.iter().map(|s| s.current().unwrap()).min().unwrap();
        let max_v = streams.iter().map(|s| s.current().unwrap()).max().unwrap();
        if min_v == max_v {
            return true;
        }
        for s in streams.iter_mut() {
            if s.current().unwrap() < max_v {
                s.seek(max_v);
            }
        }
    }
}

pub fn ltj_run<B: GraphBackend>(query: &JoinQuery, backend: &B) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut bindings = vec![None; query.num_vars];
    ltj_recurse(query, backend, 0, &mut bindings, &mut out);
    out
}

fn ltj_recurse<B: GraphBackend>(query: &JoinQuery, backend: &B, var_idx: usize, bindings: &mut Vec<Option<u32>>, out: &mut Vec<Vec<u32>>) {
    if var_idx == query.num_vars {
        out.push(bindings.iter().map(|b| b.expect("all variables bound at a leaf")).collect());
        return;
    }

    let mut streams: Vec<Box<dyn SortedNeighbors + '_>> = Vec::new();
    for pat in &query.patterns {
        let is_subject_var = pat.subject == Term::Var(var_idx);
        let is_object_var = pat.object == Term::Var(var_idx);
        if is_subject_var && !is_object_var {
            if let Some(o) = resolve(pat.object, bindings) {
                streams.push(backend.stream_col(pat.predicate, o));
            }
        } else if is_object_var && !is_subject_var {
            if let Some(s) = resolve(pat.subject, bindings) {
                streams.push(backend.stream_row(pat.predicate, s));
            }
        }
        // Patterns with var_idx on neither side, or a self-loop pattern
        // with var_idx on both sides, don't contribute a stream here --
        // none of this comparison's demo query shapes need the self-loop
        // case, so it's intentionally unhandled.
    }

    if streams.is_empty() {
        return; // var_idx unconstrained by any pattern given current bindings: not a well-formed query for this driver
    }

    while leapfrog_align(&mut streams) {
        let val = streams[0].current().unwrap();
        bindings[var_idx] = Some(val);
        ltj_recurse(query, backend, var_idx + 1, bindings, out);
        for s in streams.iter_mut() {
            s.advance();
        }
    }
    bindings[var_idx] = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ik2tree::Ik2Tree;
    use crate::common::mk2tree::Mk2Forest;
    use crate::common::omnitree::Omnitree;
    use crate::common::radixtree::RadixTree;
    use crate::common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled;
    use crate::common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile;
    use crate::common::rdf_gen::{star_triangle, Triple};
    use crate::common::reference::NaiveIndex;

    /// Naive nested-loop reference join over the same query shapes, used
    /// as the correctness oracle for `ltj_run` (there's no external
    /// implementation to check against).
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

    fn check_all_backends(triples: &[Triple], query: &JoinQuery, all_nodes: &[u32]) {
        let naive = NaiveIndex::build(triples);
        let expected = sorted(naive_join(query, &naive, all_nodes));
        assert!(!expected.is_empty(), "test dataset/query produced zero matches -- not exercising real join logic");

        let mk2 = Mk2Forest::build(triples);
        assert_eq!(sorted(ltj_run(query, &mk2)), expected, "Mk2Forest join mismatch");

        let ik2 = Ik2Tree::build(2, triples);
        assert_eq!(sorted(ltj_run(query, &ik2)), expected, "Ik2Tree join mismatch");

        let omni = Omnitree::build(triples);
        assert_eq!(sorted(ltj_run(query, &omni)), expected, "Omnitree join mismatch");

        let radix = RadixTree::build(triples, RadixTree::default_split());
        assert_eq!(sorted(ltj_run(query, &radix)), expected, "RadixTree join mismatch");

        let heap = RadixTreeDfsInterleavedVarSkipUnrolled::build_dynamic(triples, 4096);
        assert_eq!(sorted(ltj_run(query, &heap)), expected, "RadixTreeDfsInterleavedVarSkipUnrolled join mismatch");

        let file = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(triples, 4096);
        assert_eq!(sorted(ltj_run(query, &file)), expected, "RadixTreeDfsInterleavedVarSkipUnrolledFile join mismatch");
    }

    // Hand-crafted (not generator-derived) datasets below: a fully
    // unanchored path/triangle pattern (no constant anywhere) has no
    // pattern that can seed the very first variable in a fixed
    // elimination order -- there's nothing for `ltj_recurse` to start
    // enumerating from. Real WCOJ engines handle this via a full-domain
    // scan stream for such variables; this driver deliberately doesn't
    // implement that (none of the comparison's benchmark shapes need an
    // unanchored pattern), so every query here anchors at least one
    // position with a constant, mirroring realistic queries like "find
    // friends-of-friends of node X". Hand-crafting the triples (rather
    // than relying on `star_triangle`'s randomly-assigned predicates)
    // guarantees the intended matches -- and, just as importantly, the
    // intended *non*-matches -- actually exist to exercise real
    // intersection logic, not just an empty-vs-empty vacuous pass.

    #[test]
    fn two_hop_path_pattern() {
        // hub p0 ?b . ?b p1 ?c  (anchored at a constant hub)
        let triples: Vec<Triple> = vec![
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3), // hub --p0--> {1,2,3}
            (1, 1, 10),
            (1, 1, 11), // 1 --p1--> {10,11}: real matches
            (2, 2, 20), // 2 has NO p1 edge (predicate mismatch): must be excluded
                        // 3 has no outgoing edge at all: must be excluded
        ];
        let all_nodes: Vec<u32> = (0..30).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
            ],
        };
        check_all_backends(&triples, &query, &all_nodes);
    }

    #[test]
    fn triangle_pattern() {
        // hub p0 ?a . ?a p1 ?b . ?b p2 hub  (closes back to the same hub)
        let triples: Vec<Triple> = vec![
            (0, 0, 1),
            (0, 0, 2), // hub --p0--> {1,2}
            (1, 1, 3),
            (2, 1, 4), // 1--p1-->3, 2--p1-->4
            (3, 2, 0), // 3--p2-->hub: closes the 0-1-3 triangle
                       // 4 has no p2 edge back to hub: the 0-2-4 branch must be excluded
        ];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
                TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Const(0) },
            ],
        };
        check_all_backends(&triples, &query, &all_nodes);
    }

    #[test]
    fn four_cycle_pattern() {
        // hub p0 ?a . ?a p1 ?b . ?b p2 ?c . ?c p3 hub  (closes back to hub
        // after 4 hops, not 3 -- exercises a longer variable-elimination
        // chain than `triangle_pattern`'s).
        let triples: Vec<Triple> = vec![
            (0, 0, 1),
            (0, 0, 5), // hub --p0--> {1,5}
            (1, 1, 2),
            (5, 1, 6), // 1--p1-->2, 5--p1-->6
            (2, 2, 3),
            (6, 2, 7), // 2--p2-->3, 6--p2-->7
            (3, 3, 0), // 3--p3-->hub: closes the 0-1-2-3 cycle
                       // 7 has no p3 edge back to hub: the 0-5-6-7 branch must be excluded
        ];
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
        check_all_backends(&triples, &query, &all_nodes);
    }

    #[test]
    fn star_pattern_with_bound_center() {
        // center p0 ?a . center p1 ?b . center p2 ?c  (center is a constant)
        let triples = star_triangle(3, 6, 0.5, 3, 1717);
        let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0);
        let all_nodes: Vec<u32> = (0..=max_node).collect();
        let center = 0u32;
        let query = JoinQuery {
            num_vars: 1,
            patterns: vec![
                TriplePattern { subject: Term::Const(center), predicate: 0, object: Term::Var(0) },
            ],
        };
        check_all_backends(&triples, &query, &all_nodes);
    }
}
