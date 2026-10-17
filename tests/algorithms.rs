#![cfg(feature = "heap")]

use s_graph::graph::{JoinQuery, Term, TriplePattern, join_run, try_join_run};
use s_graph::join::{NativeJoin, NativeJoinWorkspace};
use s_graph::path::*;
use s_graph::{Heap, Triple};
use std::ops::ControlFlow;

fn brute_join(query: &JoinQuery<Term>, triples: &[Triple], domain: u32) -> Vec<Vec<u32>> {
    let mut result = Vec::new();
    for mut code in 0..domain.pow(query.num_vars as u32) {
        let mut binding = vec![0; query.num_vars];
        for v in &mut binding {
            *v = code % domain;
            code /= domain;
        }
        let resolve = |t| match t {
            Term::Const(c) => c,
            Term::Var(v) => binding[v],
        };
        if query.patterns.iter().all(|p| triples.contains(&(resolve(p.subject), resolve(p.predicate), resolve(p.object)))) {
            result.push(binding);
        }
    }
    result.sort_unstable();
    result
}

fn collect<S: s_graph::BlockStorage>(plan: &NativeJoin<'_, S>, ws: &mut NativeJoinWorkspace) -> Vec<Vec<u32>> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = plan.visit(ws, |r| {
        rows.push(r.to_vec());
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}

#[test]
fn general_join_matches_exhaustive_assignments() {
    use Term::{Const as C, Var as V};
    let mut seed = 81u64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u32
    };
    for _ in 0..30 {
        let triples: Vec<_> = (0..18).map(|_| (rand() % 3, rand() % 3, rand() % 3)).collect();
        let tree = Heap::build_dynamic(triples.clone(), 1024);
        let term = |x| if x < 3 { V(x as usize) } else { C(x - 3) };
        let mut patterns = vec![TriplePattern { subject: V(0), predicate: V(1), object: V(2) }];
        for _ in 0..3 {
            patterns.push(TriplePattern { subject: term(rand() % 6), predicate: term(rand() % 6), object: term(rand() % 6) });
        }
        let mut query = JoinQuery { num_vars: 3, patterns };
        let expected = brute_join(&query, &triples, 3);
        for _ in 0..query.patterns.len() {
            let plan = NativeJoin::new(&query, &tree).unwrap();
            let mut ws = NativeJoinWorkspace::default();
            assert_eq!(collect(&plan, &mut ws), expected);
            // Cancellation must not leak cursor state into the next execution.
            let _: ControlFlow<()> = plan.visit(&mut ws, |_| ControlFlow::Break(()));
            assert_eq!(collect(&plan, &mut ws), expected);
            query.patterns.rotate_left(1);
        }
    }
}

#[test]
fn join_regressions_and_validation() {
    use Term::{Const as C, Var as V};
    let tree = Heap::build_dynamic(vec![(0, 0, 1), (1, 0, 2), (2, 0, 0), (2, 1, 2)], 1024);
    let unanchored = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: V(0), predicate: 0, object: V(1) }] };
    assert_eq!(join_run(&unanchored, &tree).len(), 3);
    let constants = JoinQuery { num_vars: 0, patterns: vec![TriplePattern { subject: C(0), predicate: 0, object: C(2) }] };
    assert!(join_run(&constants, &tree).is_empty());
    let diagonal = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: V(0), predicate: 1, object: V(0) }] };
    assert_eq!(join_run(&diagonal, &tree).to_vec_rows(), vec![vec![2]]);
    let disconnected = JoinQuery {
        num_vars: 2,
        patterns: vec![TriplePattern { subject: C(0), predicate: 0, object: V(1) }, TriplePattern { subject: V(0), predicate: 1, object: V(0) }],
    };
    assert_eq!(join_run(&disconnected, &tree).to_vec_rows(), vec![vec![2, 1]]);
    let invalid = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: V(1), predicate: 0, object: C(0) }] };
    assert!(try_join_run(&invalid, &tree).is_err());
    assert_eq!(s_graph::graph::ltj_run(&unanchored, &tree), join_run(&unanchored, &tree));
    assert!(s_graph::graph::try_ltj_run(&invalid, &tree).is_err());
    let empty: JoinQuery = JoinQuery { num_vars: 0, patterns: vec![] };
    assert_eq!(join_run(&empty, &tree).to_vec_rows(), vec![Vec::<u32>::new()]);
    let max = Heap::build_dynamic(vec![(u32::MAX, 0, u32::MAX)], 1024);
    assert_eq!(join_run(&unanchored, &max).to_vec_rows(), vec![vec![u32::MAX, u32::MAX]]);
}

fn check_scan<S: s_graph::BlockStorage>(tree: &s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>, triples: &[Triple]) {
    for s in [None, Some(0), Some(3), Some(u32::MAX)] {
        for p in [None, Some(0), Some(1), Some(u32::MAX)] {
            for o in [None, Some(0), Some(5), Some(u32::MAX)] {
                let mut expected: Vec<_> =
                    triples.iter().copied().filter(|t| s.is_none_or(|v| t.0 == v) && p.is_none_or(|v| t.1 == v) && o.is_none_or(|v| t.2 == v)).collect();
                expected.sort_unstable();
                expected.dedup();
                let mut got: Vec<_> = tree.scan([s, p, o]).collect();
                got.sort_unstable();
                assert_eq!(got, expected);
            }
        }
    }
}

#[test]
fn scans_across_backends_and_dimension_orders() {
    let triples = vec![(0, 0, 0), (3, 1, 5), (0, 1, 5), (u32::MAX, 1, 0), (3, 1, 5)];
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        check_scan(&Heap::build_dynamic_with_order(triples.clone(), order, 1024), &triples);
        #[cfg(feature = "file")]
        check_scan(&s_graph::File::build_dynamic_with_order(triples.clone(), order, 1024), &triples);
        #[cfg(feature = "compressed")]
        check_scan(&s_graph::Compressed::build_dynamic_with_order(triples.clone(), order, 1024), &triples);
    }
    check_scan(&Heap::build_dynamic(vec![], 1024), &[]);
    check_scan(&Heap::build_dynamic(vec![(0, 0, 0)], 1024), &[(0, 0, 0)]);
}

fn brute_paths(triples: &[Triple], start: u32, options: PathOptions) -> Vec<GraphPath> {
    fn visit(triples: &[Triple], options: PathOptions, path: &mut GraphPath, out: &mut Vec<GraphPath>) {
        let node = *path.nodes.last().unwrap();
        let depth = path.edges.len() as u32;
        if depth >= options.hops.min() && options.target.is_none_or(|v| v == node) {
            out.push(path.clone());
        }
        if depth == options.hops.max() {
            return;
        }
        for &edge @ (s, p, o) in triples {
            if options.step.predicate.is_some_and(|v| v != p) {
                continue;
            }
            let next = if s == node && options.step.direction != Direction::Incoming {
                Some(o)
            } else if o == node && options.step.direction != Direction::Outgoing {
                Some(s)
            } else {
                None
            };
            let Some(next) = next else {
                continue;
            };
            if options.semantics == PathSemantics::Simple && path.nodes.contains(&next) {
                continue;
            }
            if options.semantics == PathSemantics::Trail && path.edges.contains(&edge) {
                continue;
            }
            path.nodes.push(next);
            path.edges.push(edge);
            visit(triples, options, path, out);
            path.nodes.pop();
            path.edges.pop();
        }
    }
    let mut out = Vec::new();
    visit(triples, options, &mut GraphPath { nodes: vec![start], edges: vec![] }, &mut out);
    out
}

#[test]
fn bounded_paths_match_independent_enumeration() {
    let triples = vec![(0, 0, 1), (0, 1, 1), (1, 0, 2), (0, 0, 2), (2, 0, 0), (1, 1, 1)];
    let tree = Heap::build_dynamic(triples.clone(), 1024);
    for direction in [Direction::Outgoing, Direction::Incoming, Direction::Both] {
        for predicate in [None, Some(0), Some(1)] {
            for min in 0..=3 {
                let hops = HopRange::new(min, 3).unwrap();
                let step = Step { direction, predicate };
                for semantics in [PathSemantics::Walk, PathSemantics::Trail, PathSemantics::Simple] {
                    let options = PathOptions { hops, step, semantics, target: None };
                    let mut expected = brute_paths(&triples, 0, options);
                    let mut actual = Vec::new();
                    let _: ControlFlow<()> = enumerate_paths(&tree, 0, options, |p| {
                        actual.push(p.clone());
                        ControlFlow::Continue(())
                    });
                    let key = |p: &GraphPath| (p.nodes.clone(), p.edges.clone());
                    actual.sort_unstable_by_key(key);
                    expected.sort_unstable_by_key(key);
                    assert_eq!(actual, expected);
                    if semantics != PathSemantics::Walk {
                        continue;
                    }
                    for mut ws in [PathWorkspace::default(), PathWorkspace::dense(2)] {
                        let mut got = Vec::new();
                        let _: ControlFlow<()> = reachable(&tree, 0, hops, step, &mut ws, |v, _| {
                            got.push(v);
                            ControlFlow::Continue(())
                        });
                        got.sort_unstable();
                        let mut ends: Vec<_> = expected.iter().map(|p| *p.nodes.last().unwrap()).collect();
                        ends.sort_unstable();
                        ends.dedup();
                        assert_eq!(got, ends);
                        for target in 0..4 {
                            let candidates: Vec<_> = expected.iter().filter(|p| p.nodes.last() == Some(&target)).collect();
                            assert_eq!(has_path(&tree, 0, target, hops, step, &mut ws), !candidates.is_empty());
                            let path = shortest_walk(&tree, 0, target, hops, step, &mut ws);
                            assert_eq!(path.as_ref().map(|p| p.edges.len()), candidates.iter().map(|p| p.edges.len()).min());
                            if let Some(path) = path {
                                assert!(candidates.contains(&&path));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_ids_zero_hops_cancellation_and_backends() {
    let triples = vec![(0, 0, u32::MAX), (u32::MAX, 0, 0)];
    fn check<B: s_graph::join::PatternBackend>(tree: &B) {
        let mut ws = PathWorkspace::default();
        assert!(has_path(tree, 0, u32::MAX, HopRange::new(3, 3).unwrap(), Step::default(), &mut ws));
        assert!(has_path(tree, 42, 42, HopRange::new(0, 0).unwrap(), Step::default(), &mut ws));
        assert!(!has_path(tree, 0, u32::MAX, HopRange::new(2, 2).unwrap(), Step::default(), &mut ws));
        let result = reachable(tree, 0, HopRange::new(0, u32::MAX).unwrap(), Step::default(), &mut ws, |_, _| ControlFlow::Break(9));
        assert_eq!(result, ControlFlow::Break(9));
    }
    check(&Heap::build_dynamic(triples.clone(), 1024));
    #[cfg(feature = "file")]
    check(&s_graph::File::build_dynamic(triples.clone(), 1024));
    #[cfg(feature = "compressed")]
    check(&s_graph::Compressed::build_dynamic(triples, 1024));
    assert!(HopRange::new(3, 2).is_err());
}

#[test]
fn degenerate_insertion_and_snapshot_validation() {
    let mut tree = Heap::build_dynamic_empty([1, 1, 1], 1024);
    assert!(tree.insert(0, 0, 0).unwrap());
    assert!(!tree.insert(0, 0, 0).unwrap());
    assert_eq!(tree.insert_batch(&mut [(0, 0, 0)]).unwrap(), 0);
    assert!(Heap::deserialize(&tree.serialize()).unwrap().contains(0, 0, 0));
    assert!(Heap::build(vec![(2, 0, 1)], [0, 0, 1], 1024).is_err());
    let bytes = Heap::build_dynamic(vec![(0, 0, 1), (99, 3, 100)], 1024).serialize();
    for offset in 0..bytes.len() {
        for mask in [1, 0x80, 0xff] {
            let mut bad = bytes.clone();
            bad[offset] ^= mask;
            // Some mutations describe a different valid graph, which is allowed.
            if let Ok(tree) = Heap::deserialize(&bad) {
                let _ = tree.scan([None; 3]).count();
            }
        }
    }
    let mut corrupt = bytes.clone();
    corrupt[5..9].copy_from_slice(&100u32.to_le_bytes());
    assert!(Heap::deserialize(&corrupt).is_err());
    for len in 0..bytes.len() {
        assert!(Heap::deserialize(&bytes[..len]).is_err());
    }
    let limits = s_graph::serialize::DeserializeLimits { max_bytes: 1, ..Default::default() };
    assert!(Heap::deserialize_with_limits(&bytes, limits).is_err());
}

#[test]
fn regular_paths_preserve_automaton_state_and_hop_bounds() {
    use s_graph::automaton::{AutomatonWorkspace, PathAutomaton, Transition};
    let step = |predicate| Step { predicate: Some(predicate), ..Default::default() };
    // Language p0 p1*: accepting after p0, then any number of p1 edges.
    let automaton = PathAutomaton::new(2, 0, &[1], vec![Transition { from: 0, to: 1, step: step(0) }, Transition { from: 1, to: 1, step: step(1) }]).unwrap();
    let tree = Heap::build_dynamic(vec![(0, 0, 1), (1, 1, 2), (2, 1, 1), (0, 1, 3), (2, 0, 3)], 1024);
    let mut ws = AutomatonWorkspace::default();
    let mut rows = Vec::new();
    let _: ControlFlow<()> = automaton.reachable(&tree, 0, HopRange::new(3, 4).unwrap(), &mut ws, |n, d| {
        rows.push((n, d));
        ControlFlow::Continue(())
    });
    assert_eq!(rows, vec![(1, 3), (2, 4)]);
    rows.clear();
    let _: ControlFlow<()> = automaton.reachable(&tree, 0, HopRange::new(0, u32::MAX).unwrap(), &mut ws, |n, d| {
        rows.push((n, d));
        ControlFlow::Continue(())
    });
    assert_eq!(rows, vec![(1, 1), (2, 2)]);
    // Same graph vertex reached in different states must not be collapsed.
    let automaton = PathAutomaton::new(3, 0, &[2], vec![Transition { from: 0, to: 1, step: step(0) }, Transition { from: 0, to: 2, step: step(0) }]).unwrap();
    rows.clear();
    let _: ControlFlow<()> = automaton.reachable(&tree, 0, HopRange::new(1, 1).unwrap(), &mut ws, |n, d| {
        rows.push((n, d));
        ControlFlow::Continue(())
    });
    assert_eq!(rows, vec![(1, 1)]);
    assert!(PathAutomaton::new(0, 0, &[], vec![]).is_err());
}

#[test]
fn native_joins_borrow_the_tree_and_are_shareable() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<NativeJoin<'static, s_graph::storage::HeapStorage>>();
    use Term::Var as V;
    let triples = vec![(0, 0, 1), (1, 1, 2), (2, 2, 0), (0, 0, 2), (2, 1, 1), (1, 2, 0)];
    let tree = Heap::build_dynamic(triples.clone(), 1024);
    let triangle = JoinQuery {
        num_vars: 3,
        patterns: vec![
            TriplePattern { subject: V(0), predicate: Term::Const(0), object: V(1) },
            TriplePattern { subject: V(1), predicate: Term::Const(1), object: V(2) },
            TriplePattern { subject: V(2), predicate: Term::Const(2), object: V(0) },
        ],
    };
    let plan = NativeJoin::new(&triangle, &tree).unwrap();
    assert_eq!(collect(&plan, &mut NativeJoinWorkspace::default()), brute_join(&triangle, &triples, 3));
    #[cfg(feature = "file")]
    assert_eq!(
        collect(&NativeJoin::new(&triangle, &s_graph::File::build_dynamic(triples.clone(), 1024)).unwrap(), &mut NativeJoinWorkspace::default()),
        brute_join(&triangle, &triples, 3)
    );
    #[cfg(feature = "compressed")]
    assert_eq!(collect(&NativeJoin::new(&triangle, &s_graph::Compressed::build_dynamic(triples, 1024)).unwrap(), &mut NativeJoinWorkspace::default()).len(), 2);
    let mut tree = Heap::build_dynamic_empty([3, 1, 3], 1024);
    tree.insert(0, 0, 1).unwrap();
    let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: V(0), predicate: 0, object: V(1) }] };
    {
        let plan = NativeJoin::new(&query, &tree).unwrap();
        std::thread::scope(|scope| {
            let first = scope.spawn(|| collect(&plan, &mut NativeJoinWorkspace::default()));
            let second = scope.spawn(|| collect(&plan, &mut NativeJoinWorkspace::default()));
            assert_eq!(first.join().unwrap(), vec![vec![0, 1]]);
            assert_eq!(second.join().unwrap(), vec![vec![0, 1]]);
        });
    }
    tree.insert(1, 0, 2).unwrap();
    assert_eq!(join_run(&query, &tree).len(), 2);
}

#[test]
fn snapshots_after_block_splits_preserve_scans_and_insertability() {
    fn check<S: s_graph::BlockStorage>() {
        use s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric as Tree;
        let mut tree = Tree::<S>::build_dynamic_empty([64,3,64],1024);
        let triples: Vec<_> = (0..200u32).map(|i| ((i*37)%64,i%3,(i*17+5)%64)).collect();
        for &(s,p,o) in &triples { tree.insert(s,p,o).unwrap(); }
        let mut expected = triples.clone(); expected.sort_unstable(); expected.dedup();
        let bytes = tree.serialize();
        let mut restored = Tree::<S>::deserialize(&bytes).unwrap();
        let mut got: Vec<_> = restored.scan([None;3]).collect(); got.sort_unstable();
        assert_eq!(got,expected);
        assert!(restored.insert(63,2,63).unwrap());
        assert!(restored.contains(63,2,63));
        let heap = Heap::deserialize(&restored.serialize()).unwrap();
        assert!(heap.contains(63,2,63));
        for t in expected { assert!(heap.contains(t.0,t.1,t.2)); }
    }
    check::<s_graph::storage::HeapStorage>();
    #[cfg(feature="file")]
    check::<s_graph::storage_file::FileStorage>();
    #[cfg(feature="compressed")]
    check::<s_graph::storage_compressed::CompressedStorage>();
}
