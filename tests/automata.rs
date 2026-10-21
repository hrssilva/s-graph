#![cfg(feature = "heap")]
use s_graph::automata::Term::{Const as C, Var as V};
use s_graph::automata::{
    AutomatonExt, Bgp, Language, Mask, Query, Schedule, Search, Unfiltered, Values,
};
use s_graph::{DEFAULT_BLOCK_CAPACITY, Fst};
use std::ops::ControlFlow;

fn rows<P: Search>(p: &P) -> Vec<Vec<Option<u32>>> {
    let mut out = Vec::new();
    let _: ControlFlow<()> = p.visit(&mut Default::default(), |r| {
        out.push(r.to_options());
        ControlFlow::Continue(())
    });
    out.sort_unstable();
    out
}
#[test]
fn explicit_automata_fuse_compose_and_rebind_to_another_graph() {
    let graph = Fst::build_dynamic(
        vec![(0, 0, 1), (0, 0, 3), (1, 1, 2)],
        DEFAULT_BLOCK_CAPACITY,
    );
    let people = Bgp::new(2).triple([C(0), C(0), V(0)]);
    let detail = Bgp::new(2).triple([V(0), C(1), V(1)]);
    let required = people
        .clone()
        .and(detail.clone())
        .schedule(Schedule::Frontier);
    assert_eq!(
        rows(&graph.search(&required).unwrap()),
        vec![vec![Some(1), Some(2)]]
    );
    let optional = people.clone().optional(detail.clone());
    assert_eq!(
        rows(&graph.search(&optional).unwrap()),
        vec![vec![Some(1), Some(2)], vec![Some(3), None]]
    );
    let absent = people.clone().minus(detail.clone());
    assert_eq!(
        rows(&graph.search(absent).unwrap()),
        vec![vec![Some(3), None]]
    );
    let present = people.clone().exists(detail.clone());
    assert_eq!(
        rows(&graph.search(present).unwrap()),
        vec![vec![Some(1), None]]
    );
    let absent = people.clone().not_exists(detail);
    assert_eq!(
        rows(&graph.search(absent).unwrap()),
        vec![vec![Some(3), None]]
    );
    let twice = people.clone().union_all(people);
    assert_eq!(rows(&graph.search(twice).unwrap()).len(), 4);
    let values = Values::new(2, vec![vec![Some(1), None], vec![Some(1), None]]).unwrap();
    assert_eq!(
        rows(
            &graph
                .search(Bgp::new(2).triple([C(0), C(0), V(0)]).join(values))
                .unwrap()
        )
        .len(),
        2
    );
    let other = Fst::build_dynamic(vec![(0, 0, 4), (4, 1, 7)], DEFAULT_BLOCK_CAPACITY);
    assert_eq!(
        rows(&other.search(&optional).unwrap()),
        vec![vec![Some(4), Some(7)]]
    );
    let p = {
        let local = required;
        graph.search(&local).unwrap()
    };
    let mut ws = Default::default();
    assert_eq!(
        p.visit(&mut ws, |_| ControlFlow::Break(17)),
        ControlFlow::Break(17)
    );
    let mut count = 0;
    let _: ControlFlow<()> = p.visit(&mut ws, |_| {
        count += 1;
        ControlFlow::Continue(())
    });
    assert_eq!(count, 1);
}
#[test]
fn recognizers_search_triples_and_complement_respects_final_acceptance() {
    let graph = Fst::build(
        (0..2)
            .flat_map(|s| (0..2).flat_map(move |p| (0..2).map(move |o| (s, p, o))))
            .collect(),
        [2; 3],
        DEFAULT_BLOCK_CAPACITY,
    );
    let language = Language(Mask(3).intersection(Mask(5)));
    assert_eq!(
        rows(&graph.search(language).unwrap()),
        vec![vec![Some(0), Some(0), Some(0)]]
    );
    let inverse = Language(Unfiltered(Mask(1).complement()));
    assert_eq!(rows(&graph.search(inverse).unwrap()).len(), 7);
    let union = Language(Mask(1).union(Mask(2)));
    assert_eq!(rows(&graph.search(union).unwrap()).len(), 2);
    let graph = Fst::build(
        vec![(u32::MAX, 1, 0), (0, 0, u32::MAX)],
        [1u64 << 32, 2, 1u64 << 32],
        DEFAULT_BLOCK_CAPACITY,
    );
    assert_eq!(
        rows(&graph.search(Language(Mask(u64::MAX))).unwrap()),
        vec![
            vec![Some(0), Some(0), Some(u32::MAX)],
            vec![Some(u32::MAX), Some(1), Some(0)]
        ]
    );
}
#[test]
fn schema_mismatches_are_compilation_errors() {
    let graph = Fst::build_dynamic(vec![(0, 0, 0)], DEFAULT_BLOCK_CAPACITY);
    assert!(graph.search(Bgp::new(1).optional(Bgp::new(2))).is_err());
    assert!(graph.search(Bgp::new(1).and(Bgp::new(2))).is_err());
    assert!(
        graph
            .search(Bgp::new(1).triple([V(9), C(0), C(0)]))
            .is_err()
    );
}
