//! Reproducible smoke benchmarks, not a replacement for representative datasets.
use s_graph::Heap;
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::native_join::{NativeJoin, NativeJoinStats, NativeJoinWorkspace};
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

fn native_bench<P: Copy + Into<Term>>(label: &str, query: &JoinQuery<P>, tree: &Heap, expected: u64, iterations: u32) {
    let compile = measure(
        || {
            black_box(NativeJoin::new(query, tree).unwrap());
        },
        1000,
    );
    let plan = NativeJoin::new(query, tree).unwrap();
    let mut ws = NativeJoinWorkspace::default();
    let execute = measure(
        || {
            let mut count = 0;
            let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                black_box(r);
                count += 1;
                ControlFlow::Continue(())
            });
            assert_eq!(count, expected);
        },
        iterations,
    );
    let first = measure(
        || {
            let _ = black_box(plan.visit(&mut ws, |r| {
                black_box(r);
                ControlFlow::Break(())
            }));
        },
        1000,
    );
    let total = measure(
        || {
            let plan = NativeJoin::new(query, tree).unwrap();
            let mut local = NativeJoinWorkspace::default();
            let mut count = 0;
            let _: ControlFlow<()> = plan.visit(&mut local, |row| {
                black_box(row);
                count += 1;
                ControlFlow::Continue(())
            });
            assert_eq!(count, expected);
        },
        iterations,
    );
    let mut stats = NativeJoinStats::default();
    let _: ControlFlow<()> = plan.visit_with_stats(&mut ws, &mut stats, |_| ControlFlow::Continue(()));
    println!("native {label}: compile {compile:.0} ns; execute {execute:.0} ns; first {first:.0} ns; compile + fresh workspace + execute {total:.0} ns");
    println!(
        "native {label}: plan {} bytes; workspace {} bytes; {} node openings; {} candidate children; {} results",
        plan.allocated_bytes(),
        ws.allocated_bytes(),
        stats.node_visits,
        stats.candidates,
        stats.results
    );
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
    native_bench("anchored", &query, &tree, 1, 10_000);
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
    native_bench("triangle", &triangle, &tree, nodes as u64, 20);
    let labels = JoinQuery {
        num_vars: 4,
        patterns: vec![TriplePattern { subject: V(0), predicate: V(1), object: V(2) }, TriplePattern { subject: V(2), predicate: V(1), object: V(3) }],
    };
    native_bench("shared predicate path", &labels, &tree, nodes as u64 * 3, 20);
}
