//! Paired join profiles: validated bindings; only prepared traversal is gated in.
use s_graph::automaton_join::AutomatonJoin;
use s_graph::demand_join::{DemandJoin, DemandStats, DemandWorkspace};
use s_graph::fst_join::{FstJoin, FstJoinStats, FstJoinWorkspace};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::path_fst::PathFstDeltaPow2Bfs;
use s_graph::scheduled_join::{HybridJoin, Policy, ScheduleStats, ScheduleWorkspace};
use std::{hint::black_box, ops::ControlFlow, time::Instant};
#[path = "rdf_gen.rs"]
#[allow(dead_code)]
mod rdf_gen;
fn atom(s: Term, p: u32, o: Term) -> TriplePattern<Term> {
    TriplePattern {
        subject: s,
        predicate: Term::Const(p),
        object: o,
    }
}
fn fixture(name: &str) -> (Vec<s_graph::Triple>, [u64; 3], JoinQuery<Term>) {
    use Term::{Const as C, Var as V};
    if name == "skew" {
        let n = 2_000_000;
        let mut rng = rdf_gen::SplitMix64::new(42);
        let mut triples = Vec::new();
        for s in 0..n {
            triples.push((s, 0, rng.next_range(20) as u32));
        }
        for s in 0..n {
            if rng.next_f64() < 0.01 {
                triples.push((s, 1, rng.next_range(500_000) as u32));
            }
        }
        return (
            triples,
            [n as u64, 2, 500_000],
            JoinQuery {
                num_vars: 2,
                patterns: vec![atom(V(0), 0, C(0)), atom(V(0), 1, V(1))],
            },
        );
    }
    if name == "parity-triangle-128" {
        let n = 128;
        let mut triples = Vec::new();
        for g in 0..2 {
            for a in 0..n {
                for b in 0..n {
                    triples.extend([
                        (g * n + a, 0, g * n + b),
                        (g * n + a, 1, g * n + b),
                        (g * n + a, 2, (1 - g) * n + b),
                    ]);
                }
            }
        }
        return (
            triples,
            [(2 * n) as u64, 3, (2 * n) as u64],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(V(0), 0, V(1)),
                    atom(V(1), 1, V(2)),
                    atom(V(2), 2, V(0)),
                ],
            },
        );
    }
    if name == "mixed-triangle" {
        let n = 64;
        let mut triples = Vec::new();
        for g in 0..2 {
            for a in 0..n {
                for b in 0..n {
                    triples.extend([(g * n + a, 0, g * n + b), (g * n + a, 1, g * n + b)]);
                }
            }
        }
        for a in 0..n {
            for b in 0..n {
                triples.push((a, 2, n + b));
            }
        }
        triples.push((n, 2, n));
        return (
            triples,
            [(2 * n) as u64, 3, (2 * n) as u64],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(V(0), 0, V(1)),
                    atom(V(1), 1, V(2)),
                    atom(V(2), 2, V(0)),
                ],
            },
        );
    }
    let triples = rdf_gen::uniform(20_000, 100, 20_000, 1_000_000, 42);
    let row = triples[1234];
    let (num_vars, patterns) = match name {
        "uniform-star" => (3, vec![atom(V(0), 3, V(2)), atom(V(1), 17, V(2))]),
        "uniform-triangle" => (
            3,
            vec![
                atom(V(0), 3, V(1)),
                atom(V(1), 17, V(2)),
                atom(V(2), 41, V(0)),
            ],
        ),
        "uniform-anchored" => (2, vec![atom(C(row.0), row.1, V(0)), atom(V(0), 17, V(1))]),
        _ => panic!("unknown fixture {name}"),
    };
    (
        triples,
        [20_000, 100, 20_000],
        JoinQuery { num_vars, patterns },
    )
}
fn control(command: &str) {
    use std::io::{BufRead, Write};
    if let Ok(path) = std::env::var("PROTO_PERF_CONTROL") {
        let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        writeln!(file, "{command}").unwrap();
        if let Ok(path) = std::env::var("PROTO_PERF_ACK") {
            let mut response = String::new();
            std::io::BufReader::new(std::fs::File::open(path).unwrap())
                .read_line(&mut response)
                .unwrap();
            assert_eq!(response.trim(), "ack");
        }
    }
}
fn measure(engine: &str, case: &str, repeats: usize, rows: usize, mut visit: impl FnMut()) {
    visit(); // Warm code/workspace/data before enabling perf.
    control("enable");
    let start = Instant::now();
    for _ in 0..repeats {
        visit();
    }
    let elapsed = start.elapsed().as_secs_f64();
    control("disable");
    println!(
        "{{\"engine\":\"{engine}\",\"case\":\"{case}\",\"repeats\":{repeats},\"rows\":{rows},\"elapsed_s\":{elapsed},\"visit_us\":{}}}",
        elapsed * 1e6 / repeats as f64
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let engine = args.get(1).map(String::as_str).unwrap_or("automaton");
    let case = args.get(2).map(String::as_str).unwrap_or("uniform-star");
    let phase = args.get(3).map(String::as_str).unwrap_or("visit");
    let repeats: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(200);
    assert!(repeats > 0);
    let (triples, dims, mut query) = fixture(case);
    let tree = PathFstDeltaPow2Bfs::build(triples, dims, s_graph::DEFAULT_BLOCK_CAPACITY);
    if engine == "automaton-rare-first" {
        assert_eq!(case, "skew");
        assert!(
            s_graph::PROTOTYPE_DIAGNOSTICS,
            "rare-first requires the temporary diagnostic patch"
        );
        assert!(std::env::var_os("JOIN_INPUT_ORDER").is_some());
        query.patterns.reverse();
    }
    let baseline = FstJoin::new(&query, &tree).unwrap();
    let mut expected = Vec::new();
    let _: ControlFlow<()> = baseline.visit(&mut FstJoinWorkspace::default(), |r| {
        expected.push(r.to_vec());
        ControlFlow::Continue(())
    });
    expected.sort_unstable();
    let expected_count = match case {
        "uniform-star" => 4950,
        "skew" => 1053,
        "mixed-triangle" => 64,
        _ => 0,
    };
    assert_eq!(expected.len(), expected_count);
    let sink = |r: &[u32]| {
        black_box(r);
        ControlFlow::<()>::Continue(())
    };
    match engine {
        "automaton" | "automaton-rare-first" => {
            let plan = AutomatonJoin::new(&query, &tree).unwrap();
            let mut ws = Vec::new();
            let mut actual = Vec::new();
            let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                actual.push(r.to_vec());
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, expected);
            if phase == "stats" {
                println!("{}", plan.prototype_profile_stats(&mut ws));
            } else {
                assert_eq!(phase, "visit");
                measure(engine, case, repeats, expected.len(), || {
                    let _ = plan.visit(&mut ws, sink);
                });
            }
        }
        "scheduled-adaptive" | "scheduled-nested" | "scheduled-prefix1" | "scheduled-prefix2"
        | "scheduled-levels" => {
            let policy = match engine {
                "scheduled-nested" => Policy::Nested,
                "scheduled-prefix1" => Policy::Prefix(1),
                "scheduled-prefix2" => Policy::Prefix(2),
                "scheduled-levels" => Policy::Levels,
                _ => Policy::Adaptive,
            };
            let plan = HybridJoin::new(&query, &tree, policy).unwrap();
            let mut ws = ScheduleWorkspace::default();
            let mut actual = Vec::new();
            let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                actual.push(r.to_vec());
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, expected);
            if phase == "stats" {
                let mut st = ScheduleStats::default();
                let _ = plan.visit_with_stats(&mut ws, &mut st, sink);
                println!(
                    "{{\"nodes\":{},\"candidates\":{},\"rows\":{},\"mask_updates\":{},\"planning_nodes\":{},\"cut\":{}}}",
                    st.nodes,
                    st.candidates,
                    st.rows,
                    st.mask_updates,
                    plan.planning.nodes,
                    plan.planning.selected_cut
                );
            } else {
                assert_eq!(phase, "visit");
                measure(engine, case, repeats, expected.len(), || {
                    let _ = plan.visit(&mut ws, sink);
                });
            }
        }
        "demand" => {
            let plan = DemandJoin::new(&query, &tree).unwrap();
            let mut ws = DemandWorkspace::default();
            let mut actual = Vec::new();
            let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                actual.push(r.to_vec());
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, expected);
            if phase == "stats" {
                let mut stats = DemandStats::default();
                let _ = plan.visit_with_stats(&mut ws, &mut stats, sink);
                println!(
                    "{{\"nodes\":{},\"candidates\":{},\"rows\":{}}}",
                    stats.source_reads, stats.transitions, stats.rows
                );
            } else {
                assert_eq!(phase, "visit");
                measure(engine, case, repeats, expected.len(), || {
                    let _ = plan.visit(&mut ws, sink);
                });
            }
        }
        "fst" => {
            let mut ws = FstJoinWorkspace::default();
            if phase == "stats" {
                let mut stats = FstJoinStats::default();
                let _ = baseline.visit_with_stats(&mut ws, &mut stats, sink);
                println!(
                    "{{\"nodes\":{},\"candidates\":{},\"rows\":{}}}",
                    stats.node_visits, stats.candidates, stats.results
                );
            } else {
                assert_eq!(phase, "visit");
                measure(engine, case, repeats, expected.len(), || {
                    let _ = baseline.visit(&mut ws, sink);
                });
            }
        }
        _ => panic!("unknown engine {engine}"),
    }
}
