use super::*;
use crate::{
    DEFAULT_BLOCK_CAPACITY, Heap,
    automata_core::{And, FstSource, Not},
    graph::{JoinQuery, TriplePattern},
    path_fst::PathFstDeltaPow2Bfs,
    query_ops::{Bgp, LeftJoin, Values},
    scheduled_join::Policy,
};
fn relation(expr: &PathExpr, triples: &[Triple], n: usize) -> Vec<Vec<bool>> {
    let identity = || {
        (0..n)
            .map(|a| (0..n).map(|b| a == b).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let join = |a: &Vec<Vec<bool>>, b: &Vec<Vec<bool>>| {
        (0..n)
            .map(|s| {
                (0..n)
                    .map(|t| (0..n).any(|m| a[s][m] && b[m][t]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let closure = |mut a: Vec<Vec<bool>>| {
        for k in 0..n {
            for s in 0..n {
                for t in 0..n {
                    a[s][t] |= a[s][k] && a[k][t];
                }
            }
        }
        a
    };
    match expr {
        PathExpr::Epsilon => identity(),
        PathExpr::Edge(step) => {
            let mut a = vec![vec![false; n]; n];
            for &(s, p, o) in triples {
                if step.predicate.is_none_or(|x| x == p) {
                    if step.direction != Direction::Incoming {
                        a[s as usize][o as usize] = true;
                    }
                    if step.direction != Direction::Outgoing {
                        a[o as usize][s as usize] = true;
                    }
                }
            }
            a
        }
        PathExpr::Sequence(parts) => parts
            .iter()
            .fold(identity(), |a, p| join(&a, &relation(p, triples, n))),
        PathExpr::Alternative(parts) => parts.iter().fold(vec![vec![false; n]; n], |mut a, p| {
            let b = relation(p, triples, n);
            for s in 0..n {
                for t in 0..n {
                    a[s][t] |= b[s][t];
                }
            }
            a
        }),
        PathExpr::Optional(p) => {
            let mut a = relation(p, triples, n);
            for s in 0..n {
                a[s][s] = true;
            }
            a
        }
        PathExpr::Star(p) => {
            let mut a = relation(p, triples, n);
            for s in 0..n {
                a[s][s] = true;
            }
            closure(a)
        }
        PathExpr::Plus(p) => closure(relation(p, triples, n)),
    }
}
fn endpoints<G: PatternBackend>(
    machine: &PathAutomaton,
    graph: &G,
    start: u32,
    hops: HopRange,
) -> Vec<(u32, u32)> {
    let mut ws = AutomatonWorkspace::default();
    let mut out = Vec::new();
    let _: ControlFlow<()> = machine.reachable(graph, start, hops, &mut ws, |n, d| {
        out.push((n, d));
        ControlFlow::Continue(())
    });
    out.sort();
    if !out.is_empty() {
        assert_eq!(
            machine.reachable(graph, start, hops, &mut ws, |_, _| ControlFlow::Break(31)),
            ControlFlow::Break(31)
        );
    }
    let mut again = Vec::new();
    let _: ControlFlow<()> = machine.reachable(graph, start, hops, &mut ws, |n, d| {
        again.push((n, d));
        ControlFlow::Continue(())
    });
    again.sort();
    assert_eq!(again, out);
    out
}
#[test]
fn fst_pattern_adapter_exact_bounds_and_shared_suffix_multiplicity() {
    let triples: Vec<_> = (0..17)
        .flat_map(|s| {
            (0..3).flat_map(move |p| {
                (0..33)
                    .filter(move |o| (s * 7 + p * 13 + o) % 11 == 0)
                    .map(move |o| (s, p, o))
            })
        })
        .collect();
    let tree = PathFstDeltaPow2Bfs::build(triples.clone(), [17, 3, 33], DEFAULT_BLOCK_CAPACITY);
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    for s in [None, Some(0), Some(16), Some(17), Some(u32::MAX)] {
        for p in [None, Some(0), Some(2), Some(3)] {
            for o in [None, Some(0), Some(32), Some(33)] {
                let mut expected: Vec<_> = triples
                    .iter()
                    .copied()
                    .filter(|&(a, b, c)| {
                        s.is_none_or(|x| x == a)
                            && p.is_none_or(|x| x == b)
                            && o.is_none_or(|x| x == c)
                    })
                    .collect();
                expected.sort();
                let mut actual: Vec<_> = index.scan_pattern([s, p, o]).collect();
                actual.sort();
                assert_eq!(actual, expected);
            }
        }
    }
    let triples = vec![(0, 0, u32::MAX), (u32::MAX, 1, 0)];
    let t = PathFstDeltaPow2Bfs::build(
        triples.clone(),
        [1u64 << 32, 2, 1u64 << 32],
        DEFAULT_BLOCK_CAPACITY,
    );
    let index = PatternIndex::new(FstSource(&t)).unwrap();
    let mut actual: Vec<_> = index.scan_pattern([None, None, None]).collect();
    actual.sort();
    assert_eq!(actual, triples);
}
#[test]
fn regular_path_builders_match_independent_relational_closure() {
    let n = 7;
    let triples = vec![
        (0, 0, 1),
        (0, 0, 2),
        (1, 1, 3),
        (2, 1, 3),
        (3, 1, 0),
        (3, 2, 3),
        (4, 0, 4),
        (1, 2, 5),
    ];
    let tree = PathFstDeltaPow2Bfs::build(
        triples.clone(),
        [n as u64, 3, n as u64],
        DEFAULT_BLOCK_CAPACITY,
    );
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    let heap = Heap::build_dynamic(triples.clone(), DEFAULT_BLOCK_CAPACITY);
    let expressions = vec![
        PathExpr::Epsilon,
        PathExpr::label(0),
        PathExpr::inverse(0),
        PathExpr::label(0).then(PathExpr::label(1)),
        PathExpr::label(0).or(PathExpr::label(1)),
        PathExpr::label(0).optional(),
        PathExpr::label(1).star(),
        PathExpr::label(1).plus(),
        PathExpr::label(0).then(PathExpr::label(1).star()),
        PathExpr::label(0).or(PathExpr::inverse(2)).star(),
        PathExpr::Epsilon.star().then(PathExpr::label(0).optional()),
        PathExpr::any(Direction::Both).plus(),
        PathExpr::Alternative(vec![]),
        PathExpr::Alternative(vec![]).optional(),
        PathExpr::Sequence(vec![]),
    ];
    for expression in expressions {
        let machine = expression.compile().unwrap();
        let expected = relation(&expression, &triples, n);
        for start in 0..n {
            let actual = endpoints(
                &machine,
                &index,
                start as u32,
                HopRange::new(0, u32::MAX).unwrap(),
            );
            assert_eq!(
                actual,
                endpoints(
                    &machine,
                    &heap,
                    start as u32,
                    HopRange::new(0, u32::MAX).unwrap()
                )
            );
            let nodes: Vec<_> = actual.iter().map(|&(v, _)| v as usize).collect();
            assert_eq!(
                nodes,
                (0..n).filter(|&v| expected[start][v]).collect::<Vec<_>>()
            );
        }
    }
}
fn accepts<A: Automaton<PathEvent>>(a: &A, word: &[(u32, Direction)]) -> bool {
    let mut state = a.start();
    for &(predicate, direction) in word {
        let Some(next) = a.step(
            &state,
            &PathEvent {
                predicate,
                direction,
            },
            &(),
        ) else {
            return false;
        };
        state = next;
    }
    a.is_accepting(&state, &())
}
#[test]
fn path_language_uses_shared_and_not_algebra_and_compile_budgets() {
    let a = PathExpr::label(0)
        .then(PathExpr::label(1).optional())
        .compile()
        .unwrap();
    let b = PathExpr::label(0).compile().unwrap();
    let product = And(&a, Not(&b));
    assert!(!accepts(&product, &[(0, Direction::Outgoing)]));
    assert!(accepts(
        &product,
        &[(0, Direction::Outgoing), (1, Direction::Outgoing)]
    ));
    assert!(!accepts(
        &product,
        &[(0, Direction::Incoming), (1, Direction::Outgoing)]
    ));
    assert!(PathExpr::label(0).compile_with_limits(1, 100).is_err());
    assert!(PathExpr::label(0).compile_with_limits(100, 0).is_err());
    let large = PathExpr::Sequence(vec![PathExpr::label(1); 40])
        .compile()
        .unwrap();
    assert!(matches!(large.start(), NfaStateSet::Large(_)));
    assert!(accepts(&large, &vec![(1, Direction::Outgoing); 40]));
    assert!(!accepts(&large, &vec![(1, Direction::Outgoing); 39]));
}
#[test]
fn reverse_sequences_and_negated_one_edge_languages() {
    let triples = vec![(0, 0, 1), (1, 1, 2), (3, 2, 0)];
    let tree = PathFstDeltaPow2Bfs::build(triples, [4, 3, 4], DEFAULT_BLOCK_CAPACITY);
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    let reverse = PathExpr::label(0)
        .then(PathExpr::label(1))
        .reversed()
        .compile()
        .unwrap();
    assert_eq!(
        endpoints(&reverse, &index, 2, HopRange::new(0, 4).unwrap()),
        vec![(0, 2)]
    );
    // !(p0 | ^p2), constrained to exactly one edge; this is label-set
    // exclusion, whereas unconstrained Not complements arbitrary words.
    let any = PathExpr::any(Direction::Both).compile().unwrap();
    let excluded = PathExpr::label(0)
        .or(PathExpr::inverse(2))
        .compile()
        .unwrap();
    let machine = Composed(And(&any, Not(&excluded)));
    let mut rows = Vec::new();
    let _: ControlFlow<()> = machine.reachable(
        &index,
        0,
        HopRange::new(0, 4).unwrap(),
        &mut Default::default(),
        |n, d| {
            rows.push((n, d));
            ControlFlow::Continue(())
        },
    );
    assert!(rows.is_empty());
    let mut rows = Vec::new();
    let _: ControlFlow<()> = machine.reachable(
        &index,
        1,
        HopRange::new(0, 4).unwrap(),
        &mut Default::default(),
        |n, d| {
            rows.push((n, d));
            ControlFlow::Continue(())
        },
    );
    rows.sort();
    assert_eq!(rows, vec![(0, 1), (2, 1)]);
    let mut nested = PathExpr::Epsilon;
    for _ in 0..300 {
        nested = nested.optional();
    }
    assert!(nested.compile().is_err());
}
#[test]
fn path_bindings_join_optional_repeated_variable_and_zero_hops() {
    use Term::{Const as C, Var as V};
    let tree = PathFstDeltaPow2Bfs::build(
        vec![(0, 0, 1), (1, 1, 2), (2, 1, 1), (3, 0, 4)],
        [6, 2, 6],
        DEFAULT_BLOCK_CAPACITY,
    );
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    let machine = PathExpr::label(1).plus().compile().unwrap();
    let q = JoinQuery {
        num_vars: 3,
        patterns: vec![TriplePattern {
            subject: V(0),
            predicate: C(0),
            object: V(1),
        }],
    };
    let bgp = Bgp::new(&q, FstSource(&tree), Policy::Adaptive).unwrap();
    let path = PathBindings::new(
        &index,
        &machine,
        vec![0, 1, 2, 3, 4, 5],
        V(1),
        V(2),
        3,
        HopRange::new(1, u32::MAX).unwrap(),
    )
    .unwrap();
    let optional = LeftJoin::new(bgp, path);
    let mut ws = Default::default();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = optional.visit(&mut ws, |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    rows.sort();
    assert_eq!(
        rows,
        vec![
            vec![Some(0), Some(1), Some(1)],
            vec![Some(0), Some(1), Some(2)],
            vec![Some(3), Some(4), None]
        ]
    );
    assert_eq!(
        optional.visit(&mut ws, |_| ControlFlow::Break(8)),
        ControlFlow::Break(8)
    );
    let same = PathBindings::new(
        &index,
        &machine,
        vec![1, 1, 2],
        V(0),
        V(0),
        1,
        HopRange::new(1, u32::MAX).unwrap(),
    )
    .unwrap();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = same.visit(&mut Default::default(), |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    rows.sort();
    assert_eq!(rows, vec![vec![Some(1)], vec![Some(2)]]);
    let epsilon = PathExpr::Epsilon.compile().unwrap();
    let isolated = PathBindings::new(
        &index,
        &epsilon,
        vec![],
        C(99),
        V(0),
        1,
        HopRange::new(0, 0).unwrap(),
    )
    .unwrap();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = isolated.visit(&mut Default::default(), |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    assert_eq!(rows, vec![vec![Some(99)]]);
    let hints = Values::new(1, vec![vec![Some(2)]]).unwrap();
    let joined = crate::query_ops::Join {
        left: hints,
        right: same,
    };
    let mut rows = Vec::new();
    let _: ControlFlow<()> = joined.visit(&mut Default::default(), |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    assert_eq!(rows, vec![vec![Some(2)]]);
}
#[test]
fn path_minimum_hops_retains_revisits_before_lower_bound() {
    let tree = PathFstDeltaPow2Bfs::build(
        vec![(0, 0, 1), (1, 1, 2), (2, 1, 1)],
        [3, 2, 3],
        DEFAULT_BLOCK_CAPACITY,
    );
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    let machine = PathExpr::label(0)
        .then(PathExpr::label(1).star())
        .compile()
        .unwrap();
    assert_eq!(
        endpoints(&machine, &index, 0, HopRange::new(3, 4).unwrap()),
        vec![(1, 3), (2, 4)]
    );
}
#[test]
fn composed_path_products_execute_as_binding_relations() {
    use Term::{Const as C, Var as V};
    let tree = PathFstDeltaPow2Bfs::build(
        vec![(0, 0, 1), (1, 1, 2), (2, 1, 1)],
        [3, 2, 3],
        DEFAULT_BLOCK_CAPACITY,
    );
    let index = PatternIndex::new(FstSource(&tree)).unwrap();
    let composed = Composed(And(
        PathExpr::label(0)
            .then(PathExpr::label(1).star())
            .compile()
            .unwrap(),
        Not(PathExpr::label(0).compile().unwrap()),
    ));
    let path = PathBindings::new(
        &index,
        &composed,
        vec![],
        C(0),
        V(0),
        1,
        HopRange::new(1, 4).unwrap(),
    )
    .unwrap();
    let mut ws = Default::default();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = path.visit(&mut ws, |r| {
        rows.push(r.to_options());
        ControlFlow::Continue(())
    });
    rows.sort();
    assert_eq!(rows, vec![vec![Some(1)], vec![Some(2)]]);
    assert_eq!(
        path.visit(&mut ws, |_| ControlFlow::Break(8)),
        ControlFlow::Break(8)
    );
    let mut again = Vec::new();
    let _: ControlFlow<()> = path.visit(&mut ws, |r| {
        again.push(r.to_options());
        ControlFlow::Continue(())
    });
    again.sort();
    assert_eq!(again, rows);
    let machine = PathExpr::Epsilon.compile().unwrap();
    let restricted = PathBindings::new(
        &index,
        &machine,
        vec![1],
        V(0),
        V(0),
        1,
        HopRange::new(0, 0).unwrap(),
    )
    .unwrap();
    let mut hint = Row::empty(1);
    hint.set(0, 2);
    let mut count = 0;
    let _: ControlFlow<()> =
        restricted.visit_matching(&mut Default::default(), hint.view(), |_| {
            count += 1;
            ControlFlow::Continue(())
        });
    assert_eq!(count, 0);
}
