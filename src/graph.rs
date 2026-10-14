//! Integer graph patterns and native prefix-join collection helpers.
//! For streaming results and reusable execution memory, use
//! [`crate::join::NativeJoin`].

use crate::storage::BlockStorage;
use crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric;

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

/// Compile and collect a native prefix join. For repeated execution or bounded
/// output memory, use [`crate::join::NativeJoin`] and its workspace directly.
/// Invalid queries panic; [`try_join_run`] returns validation errors instead.
/// Bindings are indexed by variable ID; result order is unspecified.
pub fn join_run<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> Vec<Vec<u32>> {
    try_join_run(query, tree).expect("invalid join query")
}

/// Fallible native prefix-join collection helper.
pub fn try_join_run<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> crate::Result<Vec<Vec<u32>>> {
    let plan = crate::join::NativeJoin::new(query, tree)?;
    let mut workspace = crate::join::NativeJoinWorkspace::default();
    let mut out = Vec::new();
    let _: std::ops::ControlFlow<()> = plan.visit(&mut workspace, |row| {
        out.push(row.to_vec());
        std::ops::ControlFlow::Continue(())
    });
    Ok(out)
}

/// Historical name for [`join_run`]; executes the native prefix algorithm.
pub use join_run as ltj_run;
/// Historical name for [`try_join_run`]; executes the native prefix algorithm.
pub use try_join_run as try_ltj_run;

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

    // Anchored compatibility cases; general patterns are tested in tests/native_join.rs.

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

}
