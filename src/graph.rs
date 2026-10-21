//! Integer graph patterns and join collection helpers. FST inputs use the
//! default adaptive automaton; mutable inputs retain their native executor.

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
    fn from(value: u32) -> Self {
        Self::Const(value)
    }
}

/// Rows collected by [`join_run`]/[`try_join_run`]: every row's bindings
/// live back to back in one flat buffer, each row a `row_width()`-wide
/// borrowed slice -- one allocation for the whole result set rather than
/// one `Vec<u32>` per row (what this type replaced). Bindings are indexed
/// by variable ID; result order is unspecified, matching `NativeJoin`'s
/// own contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JoinRows {
    data: Vec<u32>,
    row_width: usize,
    rows: usize,
}

impl JoinRows {
    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Bindings per row -- `query.num_vars` for the query this collected.
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn get(&self, index: usize) -> Option<&[u32]> {
        if index >= self.rows {
            return None;
        }
        Some(if self.row_width == 0 {
            &[]
        } else {
            &self.data[index * self.row_width..(index + 1) * self.row_width]
        })
    }

    /// Borrowed rows in collection order.
    pub fn rows(&self) -> Rows<'_> {
        Rows {
            data: &self.data,
            row_width: self.row_width,
            remaining: self.rows,
        }
    }

    /// Copies every row into its own owned `Vec<u32>` -- the shape this
    /// type replaces. An escape hatch for a caller that genuinely wants
    /// per-row ownership, not the default.
    pub fn to_vec_rows(&self) -> Vec<Vec<u32>> {
        self.rows().map(<[u32]>::to_vec).collect()
    }
}

impl std::ops::Index<usize> for JoinRows {
    type Output = [u32];
    fn index(&self, index: usize) -> &[u32] {
        self.get(index).expect("row index out of bounds")
    }
}

impl<'a> IntoIterator for &'a JoinRows {
    type Item = &'a [u32];
    type IntoIter = Rows<'a>;
    fn into_iter(self) -> Rows<'a> {
        self.rows()
    }
}

/// Borrowing iterator over a [`JoinRows`]'s rows; see [`JoinRows::rows`].
pub struct Rows<'a> {
    data: &'a [u32],
    row_width: usize,
    remaining: usize,
}

impl<'a> Iterator for Rows<'a> {
    type Item = &'a [u32];
    fn next(&mut self) -> Option<&'a [u32]> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        if self.row_width == 0 {
            return Some(&[]);
        }
        let (row, rest) = self.data.split_at(self.row_width);
        self.data = rest;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Rows<'_> {}

/// Compile and collect using the backend’s join executor: adaptive automata for
/// FST, native prefix traversal for mutable backends. Reuse plans/workspaces
/// directly for repeated execution or bounded output memory.
/// Invalid queries panic; [`try_join_run`] returns validation errors instead.
/// Bindings are indexed by variable ID; result order is unspecified.
pub fn join_run<B: JoinBackend, P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &B) -> JoinRows {
    try_join_run(query, tree).expect("invalid join query")
}

/// Fallible join collection with static backend dispatch.
pub fn try_join_run<B: JoinBackend, P: Copy + Into<Term>>(
    query: &JoinQuery<P>,
    tree: &B,
) -> crate::Result<JoinRows> {
    tree.collect_join(query)
}

/// Backend selection for collection helpers; no dynamic dispatch is involved.
pub trait JoinBackend {
    fn collect_join<P: Copy + Into<Term>>(&self, query: &JoinQuery<P>) -> crate::Result<JoinRows>;
}

impl<S: BlockStorage> JoinBackend for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    fn collect_join<P: Copy + Into<Term>>(&self, query: &JoinQuery<P>) -> crate::Result<JoinRows> {
        let tree = self;
        let plan = crate::join::NativeJoin::new(query, tree)?;
        let mut workspace = crate::join::NativeJoinWorkspace::default();
        let mut data = Vec::new();
        let mut rows = 0usize;
        let _: std::ops::ControlFlow<()> = plan.visit(&mut workspace, |row| {
            data.extend_from_slice(row);
            rows += 1;
            std::ops::ControlFlow::Continue(())
        });
        Ok(JoinRows {
            data,
            row_width: query.num_vars,
            rows,
        })
    }
}

#[cfg(feature = "heap")]
impl<S: BlockStorage> JoinBackend for crate::path_fst::PackedFst<S> {
    fn collect_join<P: Copy + Into<Term>>(&self, query: &JoinQuery<P>) -> crate::Result<JoinRows> {
        let plan = crate::Join::new(query, self)?;
        let mut workspace = crate::JoinWorkspace::default();
        let mut data = Vec::new();
        let mut rows = 0;
        let _: std::ops::ControlFlow<()> = plan.visit(&mut workspace, |row| {
            data.extend_from_slice(row);
            rows += 1;
            std::ops::ControlFlow::Continue(())
        });
        Ok(JoinRows {
            data,
            row_width: query.num_vars,
            rows,
        })
    }
}

#[cfg(feature = "heap")]
impl<S: BlockStorage> JoinBackend for crate::EditableFst<S> {
    fn collect_join<P: Copy + Into<Term>>(&self, query: &JoinQuery<P>) -> crate::Result<JoinRows> {
        let plan = self.join(query)?;
        let mut data = Vec::new();
        let mut rows = 0;
        let _: std::ops::ControlFlow<()> = plan.visit(&mut Default::default(), |row| {
            data.extend_from_slice(row);
            rows += 1;
            std::ops::ControlFlow::Continue(())
        });
        Ok(JoinRows {
            data,
            row_width: query.num_vars,
            rows,
        })
    }
}

/// Historical name for [`join_run`]; uses the same backend dispatch.
pub use join_run as ltj_run;
/// Historical name for [`try_join_run`]; uses the same backend dispatch.
pub use try_join_run as try_ltj_run;

#[cfg(all(test, feature = "heap"))]
mod tests {
    use super::*;
    use crate::Triple;
    use crate::test_support::{NaiveIndex, uniform};

    fn naive_join(query: &JoinQuery, naive: &NaiveIndex, all_nodes: &[u32]) -> Vec<Vec<u32>> {
        fn rec(
            query: &JoinQuery,
            naive: &NaiveIndex,
            all_nodes: &[u32],
            var_idx: usize,
            bindings: &mut Vec<Option<u32>>,
            out: &mut Vec<Vec<u32>>,
        ) {
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
        assert!(
            !expected.is_empty(),
            "test dataset/query produced zero matches -- not exercising real join logic"
        );

        let heap = crate::Heap::build_dynamic(triples.to_vec(), 4096);
        assert_eq!(
            sorted(ltj_run(query, &heap).to_vec_rows()),
            expected,
            "Heap join mismatch"
        );
    }

    // Anchored compatibility cases; general patterns are tested in tests/native_join.rs.

    #[test]
    fn two_hop_path_pattern() {
        let triples: Vec<Triple> = vec![
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (1, 1, 10),
            (1, 1, 11),
            (2, 2, 20),
        ];
        let all_nodes: Vec<u32> = (0..30).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern {
                    subject: Term::Const(0),
                    predicate: 0,
                    object: Term::Var(0),
                },
                TriplePattern {
                    subject: Term::Var(0),
                    predicate: 1,
                    object: Term::Var(1),
                },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn triangle_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 1, 4), (3, 2, 0)];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern {
                    subject: Term::Const(0),
                    predicate: 0,
                    object: Term::Var(0),
                },
                TriplePattern {
                    subject: Term::Var(0),
                    predicate: 1,
                    object: Term::Var(1),
                },
                TriplePattern {
                    subject: Term::Var(1),
                    predicate: 2,
                    object: Term::Const(0),
                },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn four_cycle_pattern() {
        let triples: Vec<Triple> = vec![
            (0, 0, 1),
            (0, 0, 5),
            (1, 1, 2),
            (5, 1, 6),
            (2, 2, 3),
            (6, 2, 7),
            (3, 3, 0),
        ];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 3,
            patterns: vec![
                TriplePattern {
                    subject: Term::Const(0),
                    predicate: 0,
                    object: Term::Var(0),
                },
                TriplePattern {
                    subject: Term::Var(0),
                    predicate: 1,
                    object: Term::Var(1),
                },
                TriplePattern {
                    subject: Term::Var(1),
                    predicate: 2,
                    object: Term::Var(2),
                },
                TriplePattern {
                    subject: Term::Var(2),
                    predicate: 3,
                    object: Term::Const(0),
                },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn star_pattern_with_bound_center() {
        let triples = uniform(1, 3, 12, 30, 1717);
        let mut triples: Vec<Triple> = triples.into_iter().map(|(_, p, o)| (0, p, o)).collect();
        triples.push((0, 0, 99));
        let max_node = triples
            .iter()
            .flat_map(|&(s, _, o)| [s, o])
            .max()
            .unwrap_or(0);
        let all_nodes: Vec<u32> = (0..=max_node).collect();
        let query = JoinQuery {
            num_vars: 1,
            patterns: vec![TriplePattern {
                subject: Term::Const(0),
                predicate: 0,
                object: Term::Var(0),
            }],
        };
        check_heap(&triples, &query, &all_nodes);
    }
}
