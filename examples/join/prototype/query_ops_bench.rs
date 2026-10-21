use s_graph::{
    DEFAULT_BLOCK_CAPACITY, Heap,
    automata_core::{And, FstSource, Not},
    graph::{JoinQuery, Term, TriplePattern},
    path_fst::PathFstDeltaPow2Bfs,
    path_ops::{Composed, PathExpr, PathRunner, PatternIndex},
    query_ops::{AntiJoin, Bgp, Binding, Exists, Join, LeftJoin, NotExists, Plan, UnionAll},
    scheduled_join::Policy,
    traversal::HopRange,
};
use std::{hint::black_box, ops::ControlFlow, time::Instant};
fn control(command: &str) {
    use std::io::{BufRead, Write};
    if let Ok(path) = std::env::var("PROTO_PERF_CONTROL") {
        writeln!(
            std::fs::OpenOptions::new().write(true).open(path).unwrap(),
            "{command}"
        )
        .unwrap();
        if let Ok(path) = std::env::var("PROTO_PERF_ACK") {
            let mut response = String::new();
            std::io::BufReader::new(std::fs::File::open(path).unwrap())
                .read_line(&mut response)
                .unwrap();
            assert_eq!(response.trim(), "ack");
        }
    }
}
fn profile<P: Plan>(plan: P, algorithm: &str, repeats: usize) {
    let mut ws = P::Workspace::default();
    let mut run = || {
        let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
            black_box(r.values);
            black_box(r.bound);
            ControlFlow::Continue(())
        });
    };
    run();
    control("enable");
    let t = Instant::now();
    for _ in 0..repeats {
        run();
    }
    let elapsed = t.elapsed().as_secs_f64();
    control("disable");
    println!("{{\"algorithm\":\"{algorithm}\",\"repeats\":{repeats},\"elapsed_s\":{elapsed}}}");
}
struct FullScan<P>(P);
impl<P: Plan> Plan for FullScan<P> {
    type Workspace = P::Workspace;
    fn width(&self) -> usize {
        self.0.width()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.0.visit(ws, emit)
    }
    // Inherit the correct default: enumerate then test compatibility.
}
fn measure(mut f: impl FnMut()) -> f64 {
    f();
    let mut batch = 1;
    loop {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        if t.elapsed().as_millis() >= 5 || batch >= 8192 {
            break;
        }
        batch *= 2;
    }
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        times.push(t.elapsed().as_secs_f64() * 1e6 / batch as f64);
    }
    times.sort_by(f64::total_cmp);
    times[2]
}
fn collect<P: Plan>(p: &P) -> Vec<Vec<Option<u32>>> {
    let mut out = Vec::new();
    let _: ControlFlow<()> = p.visit(&mut Default::default(), |r| {
        out.push(r.to_options());
        ControlFlow::Continue(())
    });
    out.sort();
    out
}
fn bench<P: Plan>(case: &str, engine: &str, build: impl Fn() -> P, expected: &[Vec<Option<u32>>]) {
    let p = build();
    assert_eq!(collect(&p), expected);
    let mut ws = P::Workspace::default();
    let mut count = 0;
    let prep = measure(|| {
        black_box(build());
    });
    let visit = measure(|| {
        count = 0;
        let _: ControlFlow<()> = p.visit(&mut ws, |r| {
            black_box(r.values);
            black_box(r.bound);
            count += 1;
            ControlFlow::Continue(())
        });
    });
    assert_eq!(count, expected.len());
    let total = measure(|| {
        let p = build();
        let _: ControlFlow<()> = p.visit(&mut Default::default(), |r| {
            black_box(r.values);
            black_box(r.bound);
            ControlFlow::Continue(())
        });
    });
    println!("{case},{engine},{count},{prep:.3},{visit:.3},{total:.3}");
}
fn path_bench<G: s_graph::join::PatternBackend, M: PathRunner<G>>(
    name: &str,
    graph: &G,
    machine: &M,
) {
    let hops = HopRange::new(1, 16).unwrap();
    let mut ws = M::Workspace::default();
    let mut out = Vec::new();
    let _: ControlFlow<()> = machine.reachable(graph, 0, hops, &mut ws, |n, d| {
        out.push((n, d));
        ControlFlow::Continue(())
    });
    out.sort();
    let expected: Vec<_> = (1..16).map(|i| ((1 + 17 * i) % 4096, (i + 1))).collect();
    assert_eq!(out, expected);
    let time = measure(|| {
        let _: ControlFlow<()> = machine.reachable(graph, 0, hops, &mut ws, |n, d| {
            black_box((n, d));
            ControlFlow::Continue(())
        });
    });
    println!("path-p0-p1plus,{name},{},,{time:.3},", out.len());
}
fn main() {
    use Term::{Const as C, Var as V};
    let n = 100_000;
    let mut triples = Vec::new();
    for s in 0..n {
        triples.push((s, 0, s % 20));
        if s % 101 == 0 {
            triples.push((s, 1, (s * 17 + 13) % n));
        }
    }
    let tree = PathFstDeltaPow2Bfs::build(triples, [n as u64, 2, n as u64], DEFAULT_BLOCK_CAPACITY);
    let query = |right| JoinQuery {
        num_vars: 3,
        patterns: vec![TriplePattern {
            subject: V(0),
            predicate: C(if right { 1 } else { 0 }),
            object: if right { V(2) } else { C(0) },
        }],
    };
    let left = || Bgp::new(&query(false), FstSource(&tree), Policy::Nested).unwrap();
    let right = || Bgp::new(&query(true), FstSource(&tree), Policy::Nested).unwrap();
    let disjoint = || {
        Bgp::new(
            &JoinQuery {
                num_vars: 3,
                patterns: vec![TriplePattern {
                    subject: V(1),
                    predicate: C(1),
                    object: V(2),
                }],
            },
            FstSource(&tree),
            Policy::Nested,
        )
        .unwrap()
    };
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 1 {
        assert_eq!(args.len(), 5);
        assert_eq!(args[1], "--profile");
        let repeats = args[4].parse().unwrap();
        match (args[2].as_str(), args[3].as_str()) {
            ("parameterized", "optional") => {
                profile(LeftJoin::new(left(), right()), &args[3], repeats)
            }
            ("fullscan", "optional") => {
                profile(LeftJoin::new(left(), FullScan(right())), &args[3], repeats)
            }
            ("parameterized", "minus-disjoint") => profile(
                AntiJoin {
                    left: left(),
                    right: disjoint(),
                },
                &args[3],
                repeats,
            ),
            ("fullscan", "minus-disjoint") => profile(
                AntiJoin {
                    left: left(),
                    right: FullScan(disjoint()),
                },
                &args[3],
                repeats,
            ),
            _ => panic!("unknown profile"),
        }
        return;
    }
    println!("case,engine,rows,prepare_us,visit_us,total_us");
    let expected = collect(&Join {
        left: left(),
        right: FullScan(right()),
    });
    bench(
        "join",
        "parameterized",
        || Join {
            left: left(),
            right: right(),
        },
        &expected,
    );
    bench(
        "join",
        "fullscan",
        || Join {
            left: left(),
            right: FullScan(right()),
        },
        &expected,
    );
    bench(
        "join",
        "combined-bgp",
        || {
            let mut q = query(false);
            q.patterns.extend(query(true).patterns);
            Bgp::new(&q, FstSource(&tree), Policy::Adaptive).unwrap()
        },
        &expected,
    );
    let expected = collect(&LeftJoin::new(left(), FullScan(right())));
    bench(
        "optional",
        "parameterized",
        || LeftJoin::new(left(), right()),
        &expected,
    );
    bench(
        "optional",
        "fullscan",
        || LeftJoin::new(left(), FullScan(right())),
        &expected,
    );
    let expected = collect(&AntiJoin {
        left: left(),
        right: FullScan(right()),
    });
    bench(
        "minus",
        "parameterized",
        || AntiJoin {
            left: left(),
            right: right(),
        },
        &expected,
    );
    bench(
        "minus",
        "fullscan",
        || AntiJoin {
            left: left(),
            right: FullScan(right()),
        },
        &expected,
    );
    let expected = collect(&AntiJoin {
        left: left(),
        right: FullScan(disjoint()),
    });
    bench(
        "minus-disjoint",
        "parameterized",
        || AntiJoin {
            left: left(),
            right: disjoint(),
        },
        &expected,
    );
    bench(
        "minus-disjoint",
        "fullscan",
        || AntiJoin {
            left: left(),
            right: FullScan(disjoint()),
        },
        &expected,
    );
    let expected = collect(&Exists {
        left: left(),
        right: FullScan(right()),
    });
    bench(
        "exists",
        "parameterized",
        || Exists {
            left: left(),
            right: right(),
        },
        &expected,
    );
    bench(
        "exists",
        "fullscan",
        || Exists {
            left: left(),
            right: FullScan(right()),
        },
        &expected,
    );
    let expected = collect(&NotExists {
        left: left(),
        right: FullScan(right()),
    });
    bench(
        "not-exists",
        "parameterized",
        || NotExists {
            left: left(),
            right: right(),
        },
        &expected,
    );
    bench(
        "not-exists",
        "fullscan",
        || NotExists {
            left: left(),
            right: FullScan(right()),
        },
        &expected,
    );
    let expected = collect(&UnionAll {
        left: left(),
        right: right(),
    });
    bench(
        "union-all",
        "streaming",
        || UnionAll {
            left: left(),
            right: right(),
        },
        &expected,
    );
    let triples: Vec<_> = (0..4096)
        .flat_map(|s| {
            [
                (s, 0, (s + 1) % 4096),
                (s, 1, (s + 17) % 4096),
                (s, 2, (s + 31) % 4096),
            ]
        })
        .collect();
    let fst = PathFstDeltaPow2Bfs::build(triples.clone(), [4096, 3, 4096], DEFAULT_BLOCK_CAPACITY);
    let index = PatternIndex::new(FstSource(&fst)).unwrap();
    let heap = Heap::build_dynamic(triples, DEFAULT_BLOCK_CAPACITY);
    let expr = PathExpr::label(0).then(PathExpr::label(1).plus());
    let machine = expr.compile().unwrap();
    path_bench("fst-nfa", &index, &machine);
    path_bench("heap-nfa", &heap, &machine);
    let composed = Composed(And(
        PathExpr::label(0)
            .then(PathExpr::label(1).star())
            .compile()
            .unwrap(),
        Not(PathExpr::label(0).compile().unwrap()),
    ));
    path_bench("fst-composed", &index, &composed);
    path_bench("heap-composed", &heap, &composed);
}
