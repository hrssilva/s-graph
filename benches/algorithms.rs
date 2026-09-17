//! Reproducible smoke benchmarks, not a replacement for representative datasets.
use s_graph::Heap;
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::join::{JoinOptions, JoinWorkspace, PreparedJoin, RelationLayout};
use s_graph::path::{HopRange, PathWorkspace, Step, reachable};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

fn measure(mut work: impl FnMut(), iterations: u32) -> f64 {
    for _ in 0..10 {
        work();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        work();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    let nodes = 20_000u32;
    let triples: Vec<_> = (0..nodes).flat_map(|v| [(v, 0, (v + 1) % nodes), (v, 1, (v + 2) % nodes), (v, 2, (v + 3) % nodes)]).collect();
    let tree = Heap::build_dynamic(triples, 4096);
    println!("dataset: {nodes} vertices, {} triples; tree: {} allocated bytes", nodes * 3, tree.mem_size_bytes());
    use Term::{Const as C, Var as V};
    let query = JoinQuery {
        num_vars: 2,
        patterns: vec![
            TriplePattern { subject: C(0), predicate: 0, object: V(0) },
            TriplePattern { subject: V(0), predicate: 1, object: V(1) },
            TriplePattern { subject: C(0), predicate: 2, object: V(1) },
        ],
    };
    println!(
        "join preparation: {:.0} ns",
        measure(
            || {
                black_box(PreparedJoin::new(&query, &tree).unwrap());
            },
            20
        )
    );
    let prepared = PreparedJoin::new(&query, &tree).unwrap();
    println!("prepared relation payload: {} bytes", prepared.relation_bytes());
    let mut ws = JoinWorkspace::default();
    println!(
        "prepared join execution: {:.0} ns",
        measure(
            || {
                let mut count = 0;
                let _: ControlFlow<()> = prepared.visit(&mut ws, |row| {
                    black_box(row);
                    count += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(count, 1);
            },
            100_000
        )
    );
    // The previous anchored algorithm's adjacency/materialization strategy.
    println!(
        "anchored materialized-adjacency baseline: {:.0} ns",
        measure(
            || {
                let mut count = 0;
                for a in tree.query_two_bound([0, 1], [0, 0], 2) {
                    for b in tree.query_two_bound([0, 1], [a, 1], 2) {
                        if tree.contains(0, 2, b) {
                            count += 1;
                        }
                    }
                }
                assert_eq!(count, 1);
            },
            20_000
        )
    );
    let mut path_ws = PathWorkspace::dense(nodes as usize);
    println!(
        "bounded reachability (0..8): {:.0} ns",
        measure(
            || {
                let mut count = 0;
                let _: ControlFlow<()> = reachable(&tree, 0, HopRange::new(0, 8).unwrap(), Step::default(), &mut path_ws, |_, _| {
                    count += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(count, 25);
            },
            5_000
        )
    );
    println!(
        "single-node allocated adjacency: {:.0} ns",
        measure(
            || {
                black_box(tree.query_one_bound(0, 100, 2));
            },
            20_000
        )
    );
    let mut scratch = Vec::new();
    println!(
        "single-node reused adjacency: {:.0} ns",
        measure(
            || {
                tree.query_one_bound_into(0, 100, 2, &mut scratch);
                black_box(&scratch);
            },
            20_000
        )
    );
    println!(
        "single-node streamed edges: {:.0} ns",
        measure(
            || {
                black_box(tree.scan([Some(100), None, None]).count());
            },
            20_000
        )
    );
    let hub = Heap::build_dynamic((0..nodes).map(|v| (0, 0, v)).collect(), 4096);
    println!(
        "hub materialization, first result: {:.0} ns",
        measure(
            || {
                black_box(hub.query_two_bound([0, 1], [0, 0], 2)[0]);
            },
            100
        )
    );
    println!(
        "hub scan, first result: {:.0} ns",
        measure(
            || {
                black_box(hub.scan([Some(0), Some(0), None]).next());
            },
            20_000
        )
    );
    let triangle = JoinQuery {
        num_vars: 3,
        patterns: vec![
            TriplePattern { subject: V(0), predicate: 0, object: V(1) },
            TriplePattern { subject: V(1), predicate: 1, object: V(2) },
            TriplePattern { subject: V(0), predicate: 2, object: V(2) },
        ],
    };
    let start = Instant::now();
    let prepared = PreparedJoin::new(&triangle, &tree).unwrap();
    println!("unanchored triangle preparation: {} ns; payload: {} bytes", start.elapsed().as_nanos(), prepared.relation_bytes());
    println!(
        "unanchored triangle, {nodes} results: {:.0} ns",
        measure(
            || {
                let mut count = 0;
                let _: ControlFlow<()> = prepared.visit(&mut ws, |row| {
                    black_box(row);
                    count += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(count, nodes);
            },
            20
        )
    );
    let plain = PreparedJoin::with_options(&triangle, &tree, JoinOptions { layout: RelationLayout::Plain, ..Default::default() }).unwrap();
    println!("plain triangle payload: {} bytes", plain.relation_bytes());
    println!(
        "plain triangle, {nodes} results: {:.0} ns",
        measure(
            || {
                let mut count = 0;
                let _: ControlFlow<()> = plain.visit(&mut ws, |row| {
                    black_box(row);
                    count += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(count, nodes);
            },
            20
        )
    );
}
