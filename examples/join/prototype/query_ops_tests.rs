use super::*;
use crate::{DEFAULT_BLOCK_CAPACITY, automata_core::FstSource, path_fst::PathFstDeltaPow2Bfs};
use std::cell::Cell;
type R = Vec<Option<u32>>;
fn collect<P: Plan>(p: &P) -> Vec<R> {
    let mut ws = P::Workspace::default();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = p.visit(&mut ws, |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    rows.sort();
    if !rows.is_empty() {
        assert_eq!(
            p.visit(&mut ws, |_| ControlFlow::Break(17)),
            ControlFlow::Break(17)
        );
    }
    let mut again = Vec::new();
    let _: ControlFlow<()> = p.visit(&mut ws, |r| {
        again.push(r.to_options());
        ControlFlow::Continue(())
    });
    again.sort();
    assert_eq!(rows, again);
    rows
}
fn compatible(a: &R, b: &R) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| a.is_none() || b.is_none() || a == b)
}
fn merged(a: &R, b: &R) -> R {
    a.iter().zip(b).map(|(&a, &b)| a.or(b)).collect()
}
fn reference(a: &[R], b: &[R], kind: usize) -> Vec<R> {
    let mut out = Vec::new();
    if kind == 2 {
        out.extend_from_slice(a);
        out.extend_from_slice(b);
    } else {
        for x in a {
            let matches: Vec<_> = b.iter().filter(|y| compatible(x, y)).collect();
            match kind {
                0 => {
                    for y in matches {
                        out.push(merged(x, y));
                    }
                }
                1 => {
                    if matches.is_empty() {
                        out.push(x.clone());
                    } else {
                        for y in matches {
                            out.push(merged(x, y));
                        }
                    }
                }
                3 => {
                    if !matches.iter().any(|y| {
                        x.iter()
                            .zip(y.iter())
                            .any(|(a, b)| a.is_some() && b.is_some())
                    }) {
                        out.push(x.clone());
                    }
                }
                4 => {
                    if !matches.is_empty() {
                        out.push(x.clone());
                    }
                }
                5 => {
                    if matches.is_empty() {
                        out.push(x.clone());
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    out.sort();
    out
}
#[test]
fn randomized_bags_partial_domains_zero_and_wide_masks() {
    let mut seed = 128u64;
    for width in [0, 1, 2, 5, 65, 130] {
        for _ in 0..24 {
            let mut table = || {
                (0..6)
                    .map(|_| {
                        (0..width)
                            .map(|_| {
                                seed ^= seed << 13;
                                seed ^= seed >> 7;
                                seed ^= seed << 17;
                                if seed % 3 == 0 {
                                    None
                                } else {
                                    Some((seed % 3) as u32)
                                }
                            })
                            .collect::<R>()
                    })
                    .collect::<Vec<_>>()
            };
            let mut a = table();
            let mut b = table();
            a.push(a[0].clone());
            b.push(b[0].clone());
            let left = Values::new(width, a.clone()).unwrap();
            let right = Values::new(width, b.clone()).unwrap();
            assert_eq!(
                collect(&Join {
                    left: &left,
                    right: &right
                }),
                reference(&a, &b, 0)
            );
            assert_eq!(collect(&LeftJoin::new(&left, &right)), reference(&a, &b, 1));
            assert_eq!(
                collect(&UnionAll {
                    left: &left,
                    right: &right
                }),
                reference(&a, &b, 2)
            );
            assert_eq!(
                collect(&AntiJoin {
                    left: &left,
                    right: &right
                }),
                reference(&a, &b, 3)
            );
            assert_eq!(
                collect(&Exists {
                    left: &left,
                    right: &right
                }),
                reference(&a, &b, 4)
            );
            assert_eq!(
                collect(&NotExists {
                    left: &left,
                    right: &right
                }),
                reference(&a, &b, 5)
            );
        }
    }
}
#[test]
fn minus_disjoint_empty_mapping_and_optional_multiplicity() {
    let a = Values::new(2, vec![vec![Some(0), None]; 2]).unwrap();
    let b = Values::new(2, vec![vec![None, Some(7)]; 3]).unwrap();
    assert_eq!(
        collect(&LeftJoin::new(&a, &b)),
        vec![vec![Some(0), Some(7)]; 6]
    );
    assert_eq!(
        collect(&Minus {
            left: &a,
            right: &b
        }),
        vec![vec![Some(0), None]; 2]
    );
    assert!(
        collect(&NotExists {
            left: &a,
            right: &b
        })
        .is_empty()
    );
    let unit = Values::unit(2);
    assert_eq!(
        collect(&Minus {
            left: &a,
            right: &unit
        }),
        collect(&a)
    );
    assert!(
        collect(&NotExists {
            left: &a,
            right: &unit
        })
        .is_empty()
    );
    let reject = LeftJoin {
        left: &a,
        right: &b,
        condition: |r: Binding<'_>| r.get(1) == Some(999),
    };
    assert_eq!(collect(&reject), collect(&a));
}
#[test]
fn optional_nested_composition_and_probe_fallback_scope() {
    let a = Values::new(2, vec![vec![Some(0), None], vec![Some(1), None]]).unwrap();
    let b = Values::new(2, vec![vec![Some(0), Some(9)]]).unwrap();
    let optional = LeftJoin::new(&a, &b);
    let hint = Values::new(2, vec![vec![None, Some(8)]]).unwrap();
    // x=0 has a right match y=9. An outer probe y=8 must not invent an
    // unmatched fallback x=0. x=1 genuinely has no match and can bind y=8.
    assert_eq!(
        collect(&Join {
            left: &hint,
            right: &optional
        }),
        vec![vec![Some(1), Some(8)]]
    );
    let nested = LeftJoin::new(&optional, &hint);
    assert_eq!(
        collect(&nested),
        vec![vec![Some(0), Some(9)], vec![Some(1), Some(8)]]
    );
    assert_eq!(
        collect(&UnionAll {
            left: &optional,
            right: &optional
        })
        .len(),
        4
    );
}
#[test]
fn independent_minus_filters_and_explicit_correlated_filters() {
    let left = Values::new(2, vec![vec![Some(0), None]]).unwrap();
    let right = Values::new(2, vec![vec![None, Some(3)]]).unwrap();
    let independent = Filter {
        input: &right,
        condition: |r: Binding<'_>| r.get(0).is_some(),
    };
    assert_eq!(
        collect(&Minus {
            left: &left,
            right: independent
        }),
        collect(&left)
    );
    let correlated = CorrelatedFilter {
        input: &right,
        condition: |outer: Binding<'_>, rhs: Binding<'_>| {
            outer.get(0) == Some(0) && rhs.get(1) == Some(3)
        },
    };
    assert!(
        collect(&NotExists {
            left: &left,
            right: correlated
        })
        .is_empty()
    );
}
struct Count<'a> {
    rows: &'a Values,
    reads: &'a Cell<usize>,
}
impl Plan for Count<'_> {
    type Workspace = ();
    fn width(&self) -> usize {
        self.rows.width
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        self.rows.possible_variables()
    }
    fn visit<T>(
        &self,
        ws: &mut (),
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.rows.visit(ws, |r| {
            self.reads.set(self.reads.get() + 1);
            emit(r)
        })
    }
}
#[test]
fn existence_short_circuits_and_union_cancellation_skips_rhs() {
    let left = Values::new(1, vec![vec![Some(0)]; 2]).unwrap();
    let right = Values::new(1, vec![vec![Some(0)]; 100]).unwrap();
    let reads = Cell::new(0);
    let p = Exists {
        left: &left,
        right: Count {
            rows: &right,
            reads: &reads,
        },
    };
    let mut emitted = 0;
    let _: ControlFlow<()> = p.visit(&mut Default::default(), |_| {
        emitted += 1;
        ControlFlow::Continue(())
    });
    assert_eq!(emitted, 2);
    assert_eq!(reads.get(), 2);
    reads.set(0);
    let p = Minus {
        left: &left,
        right: Count {
            rows: &right,
            reads: &reads,
        },
    };
    let _: ControlFlow<()> = p.visit(&mut Default::default(), |_| panic!("matched MINUS row"));
    assert_eq!(reads.get(), 2);
    reads.set(0);
    let p = UnionAll {
        left: &left,
        right: Count {
            rows: &right,
            reads: &reads,
        },
    };
    assert_eq!(
        p.visit(&mut Default::default(), |_| ControlFlow::Break(9)),
        ControlFlow::Break(9)
    );
    assert_eq!(reads.get(), 0);
    let left = Values::new(2, vec![vec![Some(0), None]; 2]).unwrap();
    let right = Values::new(2, vec![vec![None, Some(7)]; 100]).unwrap();
    let minus = Minus {
        left: &left,
        right: Count {
            rows: &right,
            reads: &reads,
        },
    };
    let mut rows = 0;
    let _: ControlFlow<()> = minus.visit(&mut Default::default(), |_| {
        rows += 1;
        ControlFlow::Continue(())
    });
    assert_eq!(rows, 2);
    assert_eq!(reads.get(), 0);
    let exists = Exists {
        left: &left,
        right: Count {
            rows: &right,
            reads: &reads,
        },
    };
    let _: ControlFlow<()> = exists.visit(&mut Default::default(), |_| ControlFlow::Continue(()));
    assert_eq!(reads.get(), 2);
}
#[test]
fn compiled_bgp_subset_schemas_and_optional_filters() {
    use Term::{Const as C, Var as V};
    let tree = PathFstDeltaPow2Bfs::build(
        vec![(0, 0, 0), (1, 0, 2), (1, 1, 3), (4, 0, 4)],
        [5, 2, 5],
        DEFAULT_BLOCK_CAPACITY,
    );
    let query = |p, o| JoinQuery {
        num_vars: 3,
        patterns: vec![TriplePattern {
            subject: V(0),
            predicate: C(p),
            object: V(o),
        }],
    };
    let a = Bgp::new(&query(0, 1), FstSource(&tree), Policy::Adaptive).unwrap();
    let b = Bgp::new(&query(1, 2), FstSource(&tree), Policy::Adaptive).unwrap();
    let mut combined = query(0, 1);
    combined.patterns.extend(query(1, 2).patterns);
    let combined = Bgp::new(&combined, FstSource(&tree), Policy::Adaptive).unwrap();
    assert_eq!(
        collect(&combined),
        collect(&Join {
            left: &a,
            right: &b
        })
    );
    assert_eq!(
        collect(&LeftJoin::new(&a, &b)),
        vec![
            vec![Some(0), Some(0), None],
            vec![Some(1), Some(2), Some(3)],
            vec![Some(4), Some(4), None]
        ]
    );
    assert_eq!(
        collect(&Minus {
            left: &a,
            right: &b
        }),
        vec![vec![Some(0), Some(0), None], vec![Some(4), Some(4), None]]
    );
    assert_eq!(
        collect(&NotExists {
            left: &a,
            right: &b
        }),
        collect(&Minus {
            left: &a,
            right: &b
        })
    );
    let unit = Bgp::new(
        &JoinQuery {
            num_vars: 3,
            patterns: vec![],
        },
        FstSource(&tree),
        Policy::Adaptive,
    )
    .unwrap();
    assert_eq!(collect(&unit), vec![vec![None; 3]]);
}
#[test]
fn parameterized_register_machine_matches_filtering_all_schedules() {
    use Term::{Const as C, Var as V};
    let triples = (0..17)
        .flat_map(|s| {
            (0..3).flat_map(move |p| {
                (0..33)
                    .filter(move |o| (s * 7 + p * 13 + o) % 11 == 0)
                    .map(move |o| (s, p, o))
            })
        })
        .collect();
    let tree = PathFstDeltaPow2Bfs::build(triples, [17, 3, 33], DEFAULT_BLOCK_CAPACITY);
    for patterns in [
        vec![(V(0), V(1), V(2)), (V(2), C(1), V(0))],
        vec![(V(0), V(1), V(0))],
    ] {
        let variables = if patterns.len() == 1 { 2 } else { 3 };
        let q = JoinQuery {
            num_vars: variables,
            patterns: patterns
                .into_iter()
                .map(|(subject, predicate, object)| TriplePattern {
                    subject,
                    predicate,
                    object,
                })
                .collect(),
        };
        for policy in [
            Policy::Nested,
            Policy::Prefix(1),
            Policy::Levels,
            Policy::Adaptive,
        ] {
            let p = ScheduledJoin::with_source(&q, FstSource(&tree), policy).unwrap();
            let mut ws = ScheduleWorkspace::default();
            let mut all = Vec::new();
            let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                all.push(r.to_vec());
                ControlFlow::Continue(())
            });
            for v in 0..variables {
                for value in [0, 1, 2, 16, 32, u32::MAX] {
                    let mut hint = vec![None; variables];
                    hint[v] = Some(value);
                    let mut expected: Vec<_> =
                        all.iter().filter(|r| r[v] == value).cloned().collect();
                    expected.sort();
                    let mut rows = Vec::new();
                    let _: ControlFlow<()> = p.visit_matching(&hint, &mut ws, |r| {
                        rows.push(r.to_vec());
                        ControlFlow::Continue(())
                    });
                    rows.sort();
                    assert_eq!(rows, expected);
                    if !rows.is_empty() {
                        assert_eq!(
                            p.visit_matching(&hint, &mut ws, |_| ControlFlow::Break(5)),
                            ControlFlow::Break(5)
                        );
                    }
                }
            }
            for row in all.iter().take(8) {
                for mask in 0..(1usize << variables) {
                    let hint: Vec<_> = row
                        .iter()
                        .enumerate()
                        .map(|(v, &x)| ((mask >> v) & 1 != 0).then_some(x))
                        .collect();
                    let mut expected: Vec<_> = all
                        .iter()
                        .filter(|r| {
                            hint.iter()
                                .enumerate()
                                .all(|(v, x)| x.is_none_or(|x| r[v] == x))
                        })
                        .cloned()
                        .collect();
                    expected.sort();
                    let mut actual = Vec::new();
                    let _: ControlFlow<()> = p.visit_matching(&hint, &mut ws, |r| {
                        actual.push(r.to_vec());
                        ControlFlow::Continue(())
                    });
                    actual.sort();
                    assert_eq!(actual, expected);
                }
            }
            let mut count = 0;
            let _: ControlFlow<()> = p.visit(&mut ws, |_| {
                count += 1;
                ControlFlow::Continue(())
            });
            assert_eq!(count, all.len());
        }
    }
}
#[test]
fn optional_condition_accepts_some_matches_without_fallback() {
    let left = Values::new(2, vec![vec![Some(0), None]; 2]).unwrap();
    for rhs in [
        vec![vec![Some(0), Some(7)], vec![Some(0), Some(8)]],
        vec![vec![Some(0), Some(8)], vec![Some(0), Some(7)]],
    ] {
        let right = Values::new(2, rhs).unwrap();
        let optional = LeftJoin {
            left: &left,
            right,
            condition: |r: Binding<'_>| r.get(1) == Some(8),
        };
        assert_eq!(collect(&optional), vec![vec![Some(0), Some(8)]; 2]);
    }
}

#[test]
fn full_width_parameter_values_and_sparse_global_variable_ids() {
    use Term::{Const as C, Var as V};
    let tree = PathFstDeltaPow2Bfs::build(
        vec![(0, 0, 0), (u32::MAX, 0, 0)],
        [1u64 << 32, 1, 1],
        DEFAULT_BLOCK_CAPACITY,
    );
    let q = JoinQuery {
        num_vars: 70,
        patterns: vec![TriplePattern {
            subject: V(69),
            predicate: C(0),
            object: C(0),
        }],
    };
    let p = Bgp::new(&q, FstSource(&tree), Policy::Nested).unwrap();
    let mut ws = Default::default();
    for value in [0, u32::MAX] {
        let mut hint = Row::empty(70);
        hint.set(69, value);
        hint.set(68, 17);
        let mut count = 0;
        let _: ControlFlow<()> = p.visit_matching(&mut ws, hint.view(), |r| {
            assert_eq!(r.get(69), Some(value));
            assert_eq!(r.get(68), None);
            count += 1;
            ControlFlow::Continue(())
        });
        assert_eq!(count, 1);
    }
}
