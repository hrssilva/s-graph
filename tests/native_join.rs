#![cfg(feature = "heap")]

use Term::{Const as C, Var as V};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::native_join::{NativeJoin, NativeJoinStats, NativeJoinWorkspace};
use s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric as Tree;
use s_graph::{BlockStorage, Heap, Triple};
use std::ops::ControlFlow;

fn pattern(s: Term, p: Term, o: Term) -> TriplePattern<Term> {
    TriplePattern { subject: s, predicate: p, object: o }
}

fn collect<S: BlockStorage>(plan: &NativeJoin<'_, S>, ws: &mut NativeJoinWorkspace) -> Vec<Vec<u32>> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = plan.visit(ws, |r| {
        rows.push(r.to_vec());
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}

fn reference(query: &JoinQuery<Term>, triples: &[Triple], domain: u32) -> Vec<Vec<u32>> {
    let mut result = Vec::new();
    for mut code in 0..domain.pow(query.num_vars as u32) {
        let mut row = vec![0; query.num_vars];
        for v in &mut row {
            *v = code % domain;
            code /= domain;
        }
        let resolve = |t| match t {
            C(c) => c,
            V(v) => row[v],
        };
        if query.patterns.iter().all(|p| triples.contains(&(resolve(p.subject), resolve(p.predicate), resolve(p.object)))) {
            result.push(row);
        }
    }
    result.sort_unstable();
    result
}

fn compare<S: BlockStorage>(tree: &Tree<S>, query: &JoinQuery<Term>) {
    // Independent, deliberately simple test oracle: enumerate compatible
    // triples atom by atom, without using Morton prefixes or compiled masks.
    fn enumerate(query: &JoinQuery<Term>, triples: &[Triple], atom: usize, row: &mut [Option<u32>], out: &mut Vec<Vec<u32>>) {
        if atom == query.patterns.len() {
            out.push(row.iter().map(|v| v.unwrap()).collect());
            return;
        }
        let p = &query.patterns[atom];
        for &(s, pred, o) in triples {
            let mut assigned = [0; 3];
            let mut count = 0;
            let mut compatible = true;
            for (term, value) in [(p.subject, s), (p.predicate, pred), (p.object, o)] {
                match term {
                    C(c) => compatible &= c == value,
                    V(v) => match row[v] {
                        Some(existing) => compatible &= existing == value,
                        None => {
                            row[v] = Some(value);
                            assigned[count] = v;
                            count += 1;
                        }
                    },
                }
            }
            if compatible {
                enumerate(query, triples, atom + 1, row, out);
            }
            for &v in &assigned[..count] {
                row[v] = None;
            }
        }
    }
    let triples: Vec<_> = tree.scan([None; 3]).collect();
    let mut expected = Vec::new();
    enumerate(query, &triples, 0, &mut vec![None; query.num_vars], &mut expected);
    expected.sort_unstable();
    let plan = NativeJoin::new(query, tree).unwrap();
    let mut ws = NativeJoinWorkspace::default();
    assert_eq!(collect(&plan, &mut ws), expected);
    let _: ControlFlow<()> = plan.visit(&mut ws, |_| ControlFlow::Break(()));
    assert_eq!(collect(&plan, &mut ws), expected);
    let mut stats = NativeJoinStats::default();
    let _: ControlFlow<()> = plan.visit_with_stats(&mut ws, &mut stats, |_| ControlFlow::Continue(()));
    assert_eq!(stats.results as usize, expected.len());
}

#[test]
fn randomized_queries_match_exhaustive_assignments() {
    let mut seed = 719u64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u32
    };
    for _ in 0..60 {
        let triples: Vec<_> = (0..35).map(|_| (rand() % 4, rand() % 3, rand() % 4)).collect();
        let term = |n| if n < 3 { V(n as usize) } else { C(n - 3) };
        let mut patterns = vec![pattern(V(0), V(1), V(2))];
        for _ in 0..3 {
            patterns.push(pattern(term(rand() % 7), term(rand() % 7), term(rand() % 7)));
        }
        let query = JoinQuery { num_vars: 3, patterns };
        let expected = reference(&query, &triples, 4);
        for (split, order) in [([1, 1, 1], [0, 1, 2]), ([3, 1, 2], [2, 0, 1]), ([1, 4, 1], [1, 2, 0])] {
            let tree = Heap::build_with_order(triples.clone(), split, order, 1024).unwrap();
            let plan = NativeJoin::new(&query, &tree).unwrap();
            assert_eq!(collect(&plan, &mut NativeJoinWorkspace::default()), expected);
        }
    }
}

#[test]
fn shared_predicates_cycles_diagonals_and_disconnected_patterns() {
    let triples = vec![(0, 0, 1), (1, 0, 2), (2, 0, 0), (2, 1, 2), (0, 1, 2), (1, 1, 1), (0, 0, 0)];
    let tree = Heap::build_dynamic(triples.clone(), 1024);
    let queries = [
        JoinQuery { num_vars: 4, patterns: vec![pattern(V(0), V(1), V(2)), pattern(V(2), V(1), V(3))] },
        JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), C(0), V(1)), pattern(V(1), C(0), V(2)), pattern(V(2), C(0), V(0))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(V(0), V(1), V(0)), pattern(V(1), V(1), V(1))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(C(0), C(0), V(0)), pattern(V(1), C(1), C(2))] },
        JoinQuery { num_vars: 1, patterns: vec![pattern(V(0), C(0), C(1)), pattern(V(0), C(0), C(1))] },
    ];
    let mut ws = NativeJoinWorkspace::default();
    for query in queries {
        let plan = NativeJoin::new(&query, &tree).unwrap();
        assert_eq!(collect(&plan, &mut ws), reference(&query, &triples, 3));
        assert!(ws.allocated_bytes() > 0);
    }
    ws.release();
    assert_eq!(ws.allocated_bytes(), 0);
}

#[test]
fn unequal_domains_full_width_ids_and_all_coordinate_roles() {
    let triples = vec![(u32::MAX, 0, 3), (3, 1, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), (3, 3, 3), (1 << 31, 3, 1)];
    let queries = [
        JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), V(1), V(2))] },
        JoinQuery { num_vars: 1, patterns: vec![pattern(V(0), V(0), V(0))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(V(0), V(1), V(1)), pattern(V(1), V(1), V(1))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(V(0), C(0), V(1)), pattern(V(1), C(1), C(2))] },
    ];
    for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let tree = Heap::build_dynamic_with_order(triples.clone(), order, 1024);
        for query in &queries {
            compare(&tree, query);
        }
        let tree = Heap::build_with_order(triples.clone(), [1, 1, 4], order, 1024).unwrap();
        for query in &queries {
            compare(&tree, query);
        }
        #[cfg(feature = "file")]
        for query in &queries {
            compare(&s_graph::File::build_dynamic_with_order(triples.clone(), order, 1024), query);
        }
        #[cfg(feature = "compressed")]
        for query in &queries {
            compare(&s_graph::Compressed::build_dynamic_with_order(triples.clone(), order, 1024), query);
        }
    }
}

#[test]
fn empty_ground_degenerate_queries_and_validation() {
    for triples in [vec![], vec![(0, 0, 0)], vec![(2, 1, 3)]] {
        let tree = Heap::build_dynamic(triples.clone(), 1024);
        for query in [
            JoinQuery { num_vars: 0, patterns: vec![] },
            JoinQuery { num_vars: 0, patterns: vec![pattern(C(0), C(0), C(0))] },
            JoinQuery { num_vars: 0, patterns: vec![pattern(C(2), C(1), C(3))] },
            JoinQuery { num_vars: 1, patterns: vec![pattern(V(0), C(0), V(0))] },
            JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), V(1), V(2))] },
            JoinQuery { num_vars: 1, patterns: vec![pattern(V(0), C(u32::MAX), C(0))] },
        ] {
            compare(&tree, &query);
        }
    }
    let tree = Heap::build_dynamic_empty([1u64 << 32, 1, 2], 1024);
    compare(&tree, &JoinQuery { num_vars: 1, patterns: vec![pattern(V(0), C(0), C(0))] });
    for query in [
        JoinQuery { num_vars: usize::MAX, patterns: vec![] },
        JoinQuery { num_vars: 1, patterns: vec![pattern(V(1), C(0), C(0))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(V(0), C(0), C(0))] },
    ] {
        assert!(NativeJoin::new(&query, &tree).is_err());
    }
}

#[test]
fn cancellation_reuse_mutation_and_serialized_tree() {
    let mut tree = Heap::build_dynamic_empty([256, 4, 256], 1024);
    for v in 0..200 {
        tree.insert(v, v % 4, (v + 1) % 200).unwrap();
    }
    let query = JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), V(1), V(2))] };
    {
        let plan = NativeJoin::new(&query, &tree).unwrap();
        let mut ws = NativeJoinWorkspace::default();
        let mut stats = NativeJoinStats::default();
        assert_eq!(plan.visit_with_stats(&mut ws, &mut stats, |_| ControlFlow::Break(42)), ControlFlow::Break(42));
        assert_eq!(stats.results, 1);
        assert!(stats.node_visits < 200);
        assert_eq!(collect(&plan, &mut ws).len(), 200);
        let capacity = ws.allocated_bytes();
        assert_eq!(collect(&plan, &mut ws).len(), 200);
        assert_eq!(ws.allocated_bytes(), capacity);
    }
    tree.insert(200, 0, 201).unwrap();
    compare(&tree, &query);
    let restored = Heap::deserialize(&tree.serialize()).unwrap();
    compare(&restored, &query);
}

#[test]
fn many_variables_do_not_require_exponential_masks() {
    let tree = Heap::build_dynamic(vec![(0, 0, 0), (1, 0, 1)], 1024);
    let query = JoinQuery { num_vars: 12, patterns: (0..12).map(|v| pattern(V(v), C(0), V(v))).collect() };
    let plan = NativeJoin::new(&query, &tree).unwrap();
    let rows = collect(&plan, &mut NativeJoinWorkspace::default());
    assert_eq!(rows.len(), 1 << 12);
    assert!(rows.windows(2).all(|w| w[0] != w[1]));
    assert!(plan.allocated_bytes() < 64 * 1024);
}

#[test]
fn multilevel_prefixes_match_independent_enumeration_under_skew() {
    let queries = [
        JoinQuery { num_vars: 4, patterns: vec![pattern(V(0), V(1), V(2)), pattern(V(2), V(1), V(3))] },
        JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), C(0), V(1)), pattern(V(1), C(1), V(2)), pattern(V(2), C(2), V(0))] },
        JoinQuery { num_vars: 3, patterns: vec![pattern(V(0), V(1), V(2)), pattern(V(1), V(2), V(0))] },
        JoinQuery { num_vars: 2, patterns: vec![pattern(V(0), V(1), V(0))] },
    ];
    for offset in [0, 128, 1 << 30] {
        let triples: Vec<_> = (0..64)
            .flat_map(|v| {
                [(v + offset, 0, (v + 1) % 64 + offset), (v + offset, 1, (v + 3) % 64 + offset), (v + offset, 2, (v + 60) % 64 + offset), (v, v % 3, v)]
            })
            .collect();
        for split in [[1, 1, 1], [4, 1, 1], [1, 1, 4], [2, 2, 2]] {
            let tree = Heap::build_with_order(triples.clone(), split, [2, 1, 0], 1024).unwrap();
            for query in &queries {
                compare(&tree, query);
            }
        }
    }
}
