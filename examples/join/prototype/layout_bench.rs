//! Same register-automaton executor, two physical FST layouts.
use s_graph::automata_core::{FstSource, TripleSource};
use s_graph::fst_join::{FstJoin, FstJoinWorkspace};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::layout_source::Pow2Source;
use s_graph::path_fst::{PathFstDeltaPow2, PathFstDeltaPow2Bfs};
use s_graph::scheduled_join::{Policy, ScheduleStats, ScheduleWorkspace, ScheduledJoin};
use s_graph::{DEFAULT_BLOCK_CAPACITY, Triple};
use std::{hint::black_box, ops::ControlFlow, time::Instant};
#[path = "join_profile.rs"]
#[allow(dead_code)]
mod fixtures;

fn fixture(case: &str) -> (Vec<Triple>, [u64; 3], JoinQuery<Term>) {
    use Term::{Const as C, Var as V};
    if case.starts_with("large-") {
        let shape = case.trim_start_matches("large-");
        let (kind, atoms) = match shape.rsplit_once('-') {
            Some((kind, n)) if n.parse::<usize>().is_ok() => (kind, n.parse().unwrap()),
            _ => (shape, 4096),
        };
        let (graph, query, _) = s_graph::large_bgp_cases::fixture(kind, atoms);
        return (
            graph.scan([None; 3]).collect(),
            graph.dimension_sizes(),
            query,
        );
    }
    if case == "dense-star" {
        let n = 128;
        return (
            (0..n).flat_map(|i| [(0, 0, i), (0, 1, n + i)]).collect(),
            [1, 2, 2 * n as u64],
            JoinQuery {
                num_vars: 2,
                patterns: vec![
                    TriplePattern {
                        subject: C(0),
                        predicate: C(0),
                        object: V(0),
                    },
                    TriplePattern {
                        subject: C(0),
                        predicate: C(1),
                        object: V(1),
                    },
                ],
            },
        );
    }
    if case.starts_with("functional-") {
        let n = 20_000;
        let triples = (0..n)
            .flat_map(|v| {
                [
                    (v, 0, (v + 1) % n),
                    (v, 1, (v + 2) % n),
                    (v, 2, (v + 3) % n),
                ]
            })
            .collect();
        let patterns = if case == "functional-triangle" {
            vec![[V(0), C(0), V(1)], [V(1), C(1), V(2)], [V(0), C(2), V(2)]]
        } else {
            vec![[V(0), V(1), V(2)], [V(2), V(1), V(3)]]
        };
        return (
            triples,
            [n as u64, 3, n as u64],
            JoinQuery {
                num_vars: if case == "functional-triangle" { 3 } else { 4 },
                patterns: patterns
                    .into_iter()
                    .map(|a| TriplePattern {
                        subject: a[0],
                        predicate: a[1],
                        object: a[2],
                    })
                    .collect(),
            },
        );
    }
    fixtures::fixture(case)
}
fn policy(name: &str) -> Policy {
    match name {
        "adaptive" => Policy::Adaptive,
        "nested" => Policy::Nested,
        "levels" => Policy::Levels,
        "frontier" => Policy::Frontier,
        "prefix2" => Policy::Prefix(2),
        _ => panic!("unknown policy {name}"),
    }
}
fn sink(row: &[u32]) -> ControlFlow<()> {
    black_box(row);
    ControlFlow::Continue(())
}

fn rows<S: TripleSource>(plan: &ScheduledJoin<S>) -> Vec<Vec<u32>> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = plan.visit(&mut ScheduleWorkspace::default(), |r| {
        rows.push(r.to_vec());
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}
fn measure(trials: usize, target_us: f64, mut work: impl FnMut()) -> (f64, f64, f64) {
    work();
    let start = Instant::now();
    for _ in 0..3 {
        work();
    }
    let single = start.elapsed().as_secs_f64() * 1e6 / 3.0;
    let repeats = (target_us / single.max(0.005)).clamp(1.0, 5_000_000.0) as usize;
    let mut values = Vec::new();
    for _ in 0..trials {
        let start = Instant::now();
        for _ in 0..repeats {
            work();
        }
        values.push(start.elapsed().as_secs_f64() * 1e6 / repeats as f64);
    }
    values.sort_by(f64::total_cmp);
    (
        values[values.len() / 2],
        values[0],
        values[values.len() - 1],
    )
}
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
fn profile<S: TripleSource>(plan: ScheduledJoin<S>, repeats: usize, count: usize) {
    let mut ws = ScheduleWorkspace::default();
    let _ = plan.visit(&mut ws, sink);
    control("enable");
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = black_box(plan.visit(&mut ws, sink));
    }
    let seconds = start.elapsed().as_secs_f64();
    control("disable");
    println!(
        "{{\"elapsed_s\":{seconds},\"rows\":{count},\"repeats\":{repeats},\"visit_us\":{}}}",
        seconds * 1e6 / repeats as f64
    );
}

fn report<S: TripleSource + Copy>(
    case: &str,
    layout: &str,
    policy_name: &str,
    query: &JoinQuery<Term>,
    source: S,
    plan: &ScheduledJoin<S>,
    stats: &ScheduleStats,
    graph_bytes: usize,
    build_us: f64,
    blocks: usize,
    nodes: u64,
    count: usize,
    trials: usize,
    target_us: f64,
) {
    let prep = measure(trials, target_us, || {
        black_box(
            ScheduledJoin::with_source(black_box(query), source, policy(policy_name)).unwrap(),
        );
    });
    let mut ws = ScheduleWorkspace::default();
    let visit = measure(trials, target_us, || {
        let _ = black_box(plan.visit(&mut ws, sink));
    });
    let total = measure(trials, target_us, || {
        let plan =
            ScheduledJoin::with_source(black_box(query), source, policy(policy_name)).unwrap();
        let _ = black_box(plan.visit(&mut ScheduleWorkspace::default(), sink));
    });
    println!(
        concat!(
            "{{\"case\":\"{}\",\"layout\":\"{}\",\"policy\":\"{}\",\"rows\":{},",
            "\"prepare_us\":{},\"visit_us\":{},\"visit_min_us\":{},\"visit_max_us\":{},\"total_us\":{},",
            "\"build_us\":{},\"graph_bytes\":{},\"blocks\":{},\"physical_nodes\":{},\"plan_bytes\":{},\"workspace_bytes\":{},",
            "\"nodes\":{},\"candidates\":{},\"mask_updates\":{},\"planning_nodes\":{},\"cut\":{},\"order\":{:?},\"certified_empty\":{}}}"
        ),
        case,
        layout,
        policy_name,
        count,
        prep.0,
        visit.0,
        visit.1,
        visit.2,
        total.0,
        build_us,
        graph_bytes,
        blocks,
        nodes,
        plan.allocated_bytes(),
        ws.allocated_bytes(),
        stats.nodes,
        stats.candidates,
        stats.mask_updates,
        plan.planning.nodes,
        plan.planning.selected_cut,
        plan.planning.selected_order,
        plan.planning.certified_empty
    );
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let profiling = args.get(1).is_some_and(|x| x == "--profile");
    let case = if profiling {
        args[3].split(':').next().unwrap()
    } else {
        args.get(1).map(String::as_str).unwrap_or("uniform-star")
    };
    let (triples, domains, query) = fixture(case);
    let capacity = std::env::var("LAYOUT_BLOCK_CAPACITY")
        .map(|x| x.parse().unwrap())
        .unwrap_or(DEFAULT_BLOCK_CAPACITY);
    let trials = std::env::var("LAYOUT_TRIALS")
        .map(|x| x.parse().unwrap())
        .unwrap_or(5);
    let target_us = std::env::var("LAYOUT_TARGET_MS")
        .map(|x| x.parse::<f64>().unwrap() * 1000.0)
        .unwrap_or(60_000.0);
    let a_input = triples.clone();
    let b_input = triples.clone();
    let (a, b, a_build, b_build) = if std::env::var_os("LAYOUT_BFS_FIRST").is_some() {
        let start = Instant::now();
        let b = PathFstDeltaPow2Bfs::build(b_input, domains, capacity);
        let tb = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        let a = PathFstDeltaPow2::build(a_input, domains, capacity);
        let ta = start.elapsed().as_secs_f64() * 1e6;
        (a, b, ta, tb)
    } else {
        let start = Instant::now();
        let a = PathFstDeltaPow2::build(a_input, domains, capacity);
        let ta = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        let b = PathFstDeltaPow2Bfs::build(b_input, domains, capacity);
        let tb = start.elapsed().as_secs_f64() * 1e6;
        (a, b, ta, tb)
    };
    assert_eq!(a.height(), b.height());
    assert_eq!(a.node_count(), b.node_count());
    assert_eq!(a.visited_count(), b.visited_count());
    for &(s, p, o) in triples.iter().step_by((triples.len() / 128).max(1)) {
        assert!(a.contains(s, p, o) && b.contains(s, p, o));
    }
    let baseline = FstJoin::new(&query, &b).unwrap();
    let mut expected = Vec::new();
    let _: ControlFlow<()> = baseline.visit(&mut FstJoinWorkspace::default(), |r| {
        expected.push(r.to_vec());
        ControlFlow::Continue(())
    });
    expected.sort_unstable();
    if let Some(count) = match case {
        "uniform-star" => Some(4950),
        "dense-star" => Some(16384),
        "skew" => Some(1053),
        "mixed-triangle" => Some(64),
        "parity-triangle-128" => Some(0),
        "functional-triangle" => Some(20000),
        "functional-shared" => Some(60000),
        "large-chain" | "large-cycle" => Some(16),
        _ => None,
    } {
        assert_eq!(expected.len(), count);
    }
    let requested = std::env::var("LAYOUT_POLICIES").unwrap_or("adaptive,nested,levels".into());
    let names = if profiling {
        vec![args[3].split(':').nth(1).unwrap()]
    } else {
        requested.split(',').collect()
    };
    for name in names {
        let pa = ScheduledJoin::with_source(&query, Pow2Source(&a), policy(name)).unwrap();
        let pb = ScheduledJoin::with_source(&query, FstSource(&b), policy(name)).unwrap();
        assert_eq!(rows(&pa), expected);
        assert_eq!(rows(&pb), expected);
        assert_eq!(format!("{:?}", pa.planning), format!("{:?}", pb.planning));
        let (mut sa, mut sb) = (ScheduleStats::default(), ScheduleStats::default());
        let _ = pa.visit_with_stats(&mut ScheduleWorkspace::default(), &mut sa, sink);
        let _ = pb.visit_with_stats(&mut ScheduleWorkspace::default(), &mut sb, sink);
        assert_eq!(format!("{sa:?}"), format!("{sb:?}"));
        assert_eq!(pa.allocated_bytes(), pb.allocated_bytes());
        if profiling {
            let repeats = args[4].parse().unwrap();
            match args[2].as_str() {
                "pow2" => profile(pa, repeats, expected.len()),
                "bfs" => profile(pb, repeats, expected.len()),
                _ => panic!("unknown layout"),
            }
            return;
        }
        // Alternate which layout is measured first between invocations.
        if std::env::var_os("LAYOUT_BFS_FIRST").is_some() {
            report(
                case,
                "bfs",
                name,
                &query,
                FstSource(&b),
                &pb,
                &sb,
                b.mem_size_bytes(),
                b_build,
                b.block_count(),
                b.node_count(),
                expected.len(),
                trials,
                target_us,
            );
            report(
                case,
                "pow2",
                name,
                &query,
                Pow2Source(&a),
                &pa,
                &sa,
                a.mem_size_bytes(),
                a_build,
                a.block_count(),
                a.node_count(),
                expected.len(),
                trials,
                target_us,
            );
        } else {
            report(
                case,
                "pow2",
                name,
                &query,
                Pow2Source(&a),
                &pa,
                &sa,
                a.mem_size_bytes(),
                a_build,
                a.block_count(),
                a.node_count(),
                expected.len(),
                trials,
                target_us,
            );
            report(
                case,
                "bfs",
                name,
                &query,
                FstSource(&b),
                &pb,
                &sb,
                b.mem_size_bytes(),
                b_build,
                b.block_count(),
                b.node_count(),
                expected.len(),
                trials,
                target_us,
            );
        }
    }
}
