use s_graph::automaton_join::AutomatonJoin;
use s_graph::compiled_filter::{CompiledFilter, LegacyFilter};
use s_graph::demand_join::{DemandJoin, DemandStats, DemandWorkspace};
use s_graph::fst_join::{FstJoin, FstJoinStats, FstJoinWorkspace};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::lifted_join::LiftedJoin;
use s_graph::native_join::{NativeJoin, NativeJoinWorkspace};
use s_graph::path_fst::PathFstDeltaPow2Bfs;
use s_graph::recursive_join::{RecursiveJoin, RecursiveJoinStats};
use s_graph::scheduled_join::{HybridJoin, Policy, ScheduleStats, ScheduleWorkspace};
use s_graph::tree::DimRestriction;
use s_graph::{DEFAULT_BLOCK_CAPACITY, Heap, Triple};
use std::{hint::black_box, ops::ControlFlow, time::Instant};
#[path = "rdf_gen.rs"]
#[allow(dead_code)]
mod rdf_gen;
use rdf_gen::{SplitMix64, uniform, zipfian_entities};
fn selected(name: &str) -> bool {
    std::env::var("PROTO_CASE").map_or(true, |s| name.contains(&s))
}
fn engine(name: &str) -> bool {
    std::env::var("PROTO_ENGINE").map_or(true, |s| s.split(',').any(|x| x == name))
}
fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .map(|s| s.parse().ok().unwrap())
        .unwrap_or(default)
}
#[derive(Clone, Copy)]
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}
fn measure(mut f: impl FnMut()) -> Timing {
    let reps = env("PROTO_REPS", 7usize);
    assert!(reps > 0);
    f();
    let mut batch = 1;
    loop {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        if t.elapsed().as_millis() >= 2 || batch >= 8192 {
            break;
        }
        batch *= 2;
    }
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        times.push(t.elapsed().as_secs_f64() * 1e6 / batch as f64);
    }
    times.sort_by(f64::total_cmp);
    Timing {
        median: times[times.len() / 2],
        min: times[0],
        max: *times.last().unwrap(),
    }
}
fn output(
    kind: &str,
    case: &str,
    name: &str,
    rows: usize,
    compile: Timing,
    visit: Timing,
    e2e: Timing,
    bytes: usize,
    states: usize,
    reads: u64,
    reuses: u64,
    work: u64,
    workspace: usize,
) {
    println!(
        "{kind},{case},{name},{rows},{:.6},{:.6},{:.6},{:.6},{:.6},{bytes},{states},{reads},{reuses},{work},{workspace}",
        compile.median, visit.median, e2e.median, e2e.min, e2e.max
    );
}
fn matches(r: DimRestriction<'_>, v: u32) -> bool {
    match r {
        DimRestriction::Free => true,
        DimRestriction::Exact(x) => v == x,
        DimRestriction::Range(a, b) => a <= v && v <= b,
        DimRestriction::Multi(xs) => xs.iter().any(|&(a, b)| a <= v && v <= b),
    }
}
fn collect_legacy(p: &LegacyFilter<'_>) -> Vec<[u32; 3]> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = p.visit(|t| {
        rows.push(t);
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}
fn collect_compiled(p: &CompiledFilter<'_>) -> Vec<[u32; 3]> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = p.visit(|t| {
        rows.push(t);
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}
fn filter_case(
    name: &str,
    tree: &PathFstDeltaPow2Bfs,
    triples: &[Triple],
    r: [DimRestriction<'_>; 3],
) {
    if !selected(name) {
        return;
    }
    let mut expected: Vec<_> = triples
        .iter()
        .filter(|&&(s, p, o)| matches(r[0], s) && matches(r[1], p) && matches(r[2], o))
        .map(|&(s, p, o)| [s, p, o])
        .collect();
    expected.sort_unstable();
    expected.dedup();
    let legacy = LegacyFilter::new(tree, r, false);
    assert_eq!(collect_legacy(&legacy), expected);
    let normalized = LegacyFilter::new(tree, r, true);
    assert_eq!(collect_legacy(&normalized), expected);
    let compiled = CompiledFilter::new(tree, r).unwrap();
    assert_eq!(collect_compiled(&compiled), expected);
    let sink = |t: [u32; 3]| {
        black_box(t);
        ControlFlow::<()>::Continue(())
    };
    for (label, normal, p) in [
        ("legacy", false, &legacy),
        ("legacy-normalized", true, &normalized),
    ] {
        if !engine(label) {
            continue;
        }
        let build = measure(|| {
            black_box(LegacyFilter::new(tree, r, normal));
        });
        let visit = measure(|| {
            let _ = p.visit(sink);
        });
        let e2e = measure(|| {
            let p = LegacyFilter::new(tree, r, normal);
            let _ = p.visit(sink);
        });
        output(
            "filter-stream",
            name,
            label,
            expected.len(),
            build,
            visit,
            e2e,
            p.allocated_bytes(),
            0,
            0,
            0,
            0,
            0,
        );
    }
    if engine("compiled") {
        let build = measure(|| {
            black_box(CompiledFilter::new(tree, r).unwrap());
        });
        let visit = measure(|| {
            let _ = compiled.visit(sink);
        });
        let e2e = measure(|| {
            let p = CompiledFilter::new(tree, r).unwrap();
            let _ = p.visit(sink);
        });
        output(
            "filter-stream",
            name,
            "compiled",
            expected.len(),
            build,
            visit,
            e2e,
            compiled.allocated_bytes(),
            compiled.states(),
            0,
            0,
            compiled.raw_states as u64,
            0,
        );
    }
    let mut objects: Vec<_> = expected.iter().map(|t| t[2]).collect();
    objects.sort_unstable();
    objects.dedup();
    assert_eq!(
        tree.restricted_descent_via_composed_automaton(r, 2),
        objects
    );
    let zero = Timing {
        median: 0.0,
        min: 0.0,
        max: 0.0,
    };
    if engine("official-current") {
        let e2e = measure(|| {
            black_box(tree.restricted_descent_via_composed_automaton(r, 2));
        });
        output(
            "filter-collect",
            name,
            "official-current",
            objects.len(),
            zero,
            zero,
            e2e,
            legacy.allocated_bytes(),
            0,
            0,
            0,
            0,
            0,
        );
    }
    if engine("compiled-collect") {
        let e2e = measure(|| {
            let p = CompiledFilter::new(tree, r).unwrap();
            let mut out = Vec::new();
            let _: ControlFlow<()> = p.visit(|t| {
                out.push(t[2]);
                ControlFlow::Continue(())
            });
            out.sort_unstable();
            out.dedup();
            black_box(out);
        });
        output(
            "filter-collect",
            name,
            "compiled-collect",
            objects.len(),
            zero,
            zero,
            e2e,
            compiled.allocated_bytes(),
            compiled.states(),
            0,
            0,
            compiled.raw_states as u64,
            0,
        );
    }
}
fn filters() {
    for (label, dims, count, skew) in [
        ("uniform", [20_000u32, 100, 20_000], 1_000_000, false),
        ("asymmetric", [2048, 16, 131_072], 300_000, false),
        ("zipf", [16_384, 16, 16_384], 300_000, true),
    ] {
        let names = [
            "exact",
            "range",
            "range-three",
            "multi-sparse",
            "multi-overlap",
            "multi-three",
            "universal-range",
        ];
        if !names.iter().any(|n| selected(&format!("{label}-{n}"))) {
            continue;
        }
        let triples = if skew {
            zipfian_entities(dims[0], dims[1], dims[2], count, 1.3, 1.3, 1.3, 42)
        } else {
            uniform(dims[0], dims[1], dims[2], count, 42)
        };
        let tree = PathFstDeltaPow2Bfs::build(
            triples.clone(),
            dims.map(|d| d as u64),
            DEFAULT_BLOCK_CAPACITY,
        );
        eprintln!("filter {label}: FST={} bytes", tree.mem_size_bytes());
        let sparse: Vec<_> = (0..16)
            .map(|i| {
                let a = (dims[0] / 20) * i;
                (a, (a + 3).min(dims[0] - 1))
            })
            .collect();
        let overlap: Vec<_> = (0..64)
            .map(|i| {
                let a = (dims[0] / 100) * i;
                (a, (a + dims[0] / 10).min(dims[0] - 1))
            })
            .collect();
        let subjects: Vec<_> = (0..8)
            .map(|i| {
                let a = (dims[0] / 10) * i;
                (a, (a + dims[0] / 50).min(dims[0] - 1))
            })
            .collect();
        let objects: Vec<_> = (0..8)
            .map(|i| {
                let a = (dims[2] / 10) * i;
                (a, (a + dims[2] / 50).min(dims[2] - 1))
            })
            .collect();
        let predicates = [(0, 0), (2, 3), (6, 7), (12, 13)];
        let row = triples[1234];
        let cases = [
            (
                "exact",
                [
                    DimRestriction::Exact(row.0),
                    DimRestriction::Exact(row.1),
                    DimRestriction::Free,
                ],
            ),
            (
                "range",
                [
                    DimRestriction::Range(dims[0] / 10, dims[0] / 5),
                    DimRestriction::Exact(3),
                    DimRestriction::Free,
                ],
            ),
            (
                "range-three",
                [
                    DimRestriction::Range(dims[0] / 10, dims[0] * 3 / 5),
                    DimRestriction::Range(2, dims[1] / 2),
                    DimRestriction::Range(dims[2] / 5, dims[2] * 3 / 5),
                ],
            ),
            (
                "multi-sparse",
                [
                    DimRestriction::Multi(&sparse),
                    DimRestriction::Exact(3),
                    DimRestriction::Free,
                ],
            ),
            (
                "multi-overlap",
                [
                    DimRestriction::Multi(&overlap),
                    DimRestriction::Exact(3),
                    DimRestriction::Free,
                ],
            ),
            (
                "multi-three",
                [
                    DimRestriction::Multi(&subjects),
                    DimRestriction::Multi(&predicates),
                    DimRestriction::Multi(&objects),
                ],
            ),
            (
                "universal-range",
                [
                    DimRestriction::Range(0, dims[0] - 1),
                    DimRestriction::Range(0, dims[1] - 1),
                    DimRestriction::Range(0, dims[2] - 1),
                ],
            ),
        ];
        for (name, r) in cases {
            filter_case(&format!("{label}-{name}"), &tree, &triples, r);
        }
    }
}
fn atom(s: Term, p: u32, o: Term) -> TriplePattern<Term> {
    TriplePattern {
        subject: s,
        predicate: Term::Const(p),
        object: o,
    }
}
fn join_case(
    name: &str,
    triples: Vec<Triple>,
    dims: [u64; 3],
    query: JoinQuery<Term>,
    oracle: Option<Vec<Vec<u32>>>,
) {
    if !selected(name) {
        return;
    }
    let heap = Heap::build_dynamic(triples.clone(), DEFAULT_BLOCK_CAPACITY);
    let tree = PathFstDeltaPow2Bfs::build(triples, dims, DEFAULT_BLOCK_CAPACITY);
    eprintln!(
        "join {name}: FST={} heap={}",
        tree.mem_size_bytes(),
        heap.mem_size_bytes()
    );
    let fp = FstJoin::new(&query, &tree).unwrap();
    let mut fw = FstJoinWorkspace::default();
    let mut expected = Vec::new();
    let _: ControlFlow<()> = fp.visit(&mut fw, |r| {
        expected.push(r.to_vec());
        ControlFlow::Continue(())
    });
    expected.sort_unstable();
    if let Some(mut oracle) = oracle {
        oracle.sort_unstable();
        assert_eq!(expected, oracle);
    }
    let sink = |r: &[u32]| {
        black_box(r);
        ControlFlow::<()>::Continue(())
    };
    if engine("native") {
        let p = NativeJoin::new(&query, &heap).unwrap();
        let mut ws = NativeJoinWorkspace::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit(&mut ws, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected);
        let build = measure(|| {
            black_box(NativeJoin::new(&query, &heap).unwrap());
        });
        let visit = measure(|| {
            let _ = p.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = NativeJoin::new(&query, &heap).unwrap();
            let _ = p.visit(&mut NativeJoinWorkspace::default(), sink);
        });
        output(
            "join",
            name,
            "native",
            expected.len(),
            build,
            visit,
            e2e,
            p.allocated_bytes(),
            0,
            0,
            0,
            0,
            ws.allocated_bytes(),
        );
    }
    if engine("fst") {
        let mut st = FstJoinStats::default();
        let _: ControlFlow<()> =
            fp.visit_with_stats(&mut fw, &mut st, |_| ControlFlow::Continue(()));
        let build = measure(|| {
            black_box(FstJoin::new(&query, &tree).unwrap());
        });
        let visit = measure(|| {
            let _ = fp.visit(&mut fw, sink);
        });
        let e2e = measure(|| {
            let p = FstJoin::new(&query, &tree).unwrap();
            let _ = p.visit(&mut FstJoinWorkspace::default(), sink);
        });
        output(
            "join",
            name,
            "fst",
            expected.len(),
            build,
            visit,
            e2e,
            fp.allocated_bytes(),
            0,
            st.node_visits,
            0,
            st.candidates,
            fw.allocated_bytes(),
        );
    }
    for label in ["demand", "demand-table"] {
        if !engine(label) {
            continue;
        }
        let tables = label == "demand-table";
        let p = DemandJoin::with_tables(&query, &tree, tables).unwrap();
        let mut ws = DemandWorkspace::default();
        let mut st = DemandStats::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut st, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected, "demand on {name}");
        let build = measure(|| {
            black_box(DemandJoin::with_tables(&query, &tree, tables).unwrap());
        });
        let visit = measure(|| {
            let _ = p.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = DemandJoin::with_tables(&query, &tree, tables).unwrap();
            let _ = p.visit(&mut DemandWorkspace::default(), sink);
        });
        output(
            "join",
            name,
            label,
            expected.len(),
            build,
            visit,
            e2e,
            p.allocated_bytes(),
            p.states(),
            st.source_reads,
            0,
            st.transitions,
            ws.allocated_bytes(),
        );
    }
    for (label, policy) in [
        ("scheduled-nested", Policy::Nested),
        ("scheduled-prefix1", Policy::Prefix(1)),
        ("scheduled-prefix2", Policy::Prefix(2)),
        ("scheduled-levels", Policy::Levels),
        ("scheduled-adaptive", Policy::Adaptive),
    ] {
        if !engine(label) {
            continue;
        }
        let p = HybridJoin::new(&query, &tree, policy).unwrap();
        let mut ws = ScheduleWorkspace::default();
        let mut st = ScheduleStats::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut st, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected, "{label} on {name}");
        let build = measure(|| {
            black_box(HybridJoin::new(&query, &tree, policy).unwrap());
        });
        let visit = measure(|| {
            let _ = p.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = HybridJoin::new(&query, &tree, policy).unwrap();
            let _ = p.visit(&mut ScheduleWorkspace::default(), sink);
        });
        output(
            "join",
            name,
            label,
            expected.len(),
            build,
            visit,
            e2e,
            p.allocated_bytes(),
            p.states(),
            st.nodes,
            0,
            st.candidates,
            ws.allocated_bytes(),
        );
        eprintln!("{name} {label}: planning={:?} visit={st:?}", p.planning);
    }
    if engine("recursive") {
        let p = RecursiveJoin::new(&query, &tree).unwrap();
        let mut ws = Vec::new();
        let mut stats = RecursiveJoinStats::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut stats, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected);
        let build = measure(|| {
            black_box(RecursiveJoin::new(&query, &tree).unwrap());
        });
        let visit = measure(|| {
            let _ = p.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = RecursiveJoin::new(&query, &tree).unwrap();
            let _ = p.visit(&mut Vec::new(), sink);
        });
        output(
            "join",
            name,
            "recursive",
            expected.len(),
            build,
            visit,
            e2e,
            p.allocated_bytes(),
            0,
            stats.node_visits,
            0,
            stats.candidates,
            ws.capacity() * 4,
        );
    }
    if engine("automaton") {
        let p = AutomatonJoin::new(&query, &tree).unwrap();
        let mut ws = Vec::new();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit(&mut ws, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected);
        let build = measure(|| {
            black_box(AutomatonJoin::new(&query, &tree).unwrap());
        });
        let visit = measure(|| {
            let _ = p.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = AutomatonJoin::new(&query, &tree).unwrap();
            let _ = p.visit(&mut Vec::new(), sink);
        });
        output(
            "join",
            name,
            "automaton",
            expected.len(),
            build,
            visit,
            e2e,
            p.prototype_allocated_bytes(),
            0,
            0,
            0,
            0,
            ws.capacity() * 4,
        );
    }
    for merged in [false, true] {
        let label = if merged {
            "lifted-merged"
        } else {
            "lifted-lazy"
        };
        if !engine(label) {
            continue;
        }
        let plan = match lifted(&query, &tree, merged) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("BUDGET {name} {label}: {e}");
                continue;
            }
        };
        let mut ws = Vec::new();
        let mut actual = Vec::new();
        let mut stats = s_graph::lifted_join::VisitStats::default();
        let _: ControlFlow<()> = plan.visit_with_stats(&mut ws, &mut stats, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected, "{label} on {name}");
        if merged {
            assert_eq!(plan.count().unwrap(), expected.len() as u128);
        }
        let build = measure(|| {
            black_box(lifted(&query, &tree, merged).unwrap());
        });
        let visit = measure(|| {
            let _ = plan.visit(&mut ws, sink);
        });
        let e2e = measure(|| {
            let p = lifted(&query, &tree, merged).unwrap();
            let _ = p.visit(&mut Vec::new(), sink);
        });
        output(
            "join",
            name,
            label,
            expected.len(),
            build,
            visit,
            e2e,
            plan.allocated_bytes(),
            plan.nodes(),
            plan.stats.source_reads,
            plan.stats.source_reuses,
            stats.decisions,
            ws.capacity() * 4,
        );
        eprintln!(
            "{name} {label}: build_nodes={} cached_products={} apply_calls={} apply_hits={}",
            plan.stats.nodes_built,
            plan.stats.cached_products,
            plan.stats.apply_calls,
            plan.stats.apply_hits
        );
    }
}
fn lifted(
    query: &JoinQuery<Term>,
    tree: &PathFstDeltaPow2Bfs,
    merged: bool,
) -> Result<LiftedJoin, String> {
    let shared_first = std::env::var("PROTO_ORDER")
        .as_deref()
        .unwrap_or("shared-first")
        != "interleaved";
    LiftedJoin::with_order_and_hash(
        query,
        tree,
        merged,
        shared_first,
        std::env::var("PROTO_HASH").as_deref().unwrap_or("fast") != "std",
    )
}
fn joins() {
    if selected("uniform-star") || selected("uniform-triangle") || selected("uniform-anchored") {
        let triples = uniform(20_000, 100, 20_000, 1_000_000, 42);
        join_case(
            "uniform-star",
            triples.clone(),
            [20_000, 100, 20_000],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 3, Term::Var(2)),
                    atom(Term::Var(1), 17, Term::Var(2)),
                ],
            },
            None,
        );
        join_case(
            "uniform-triangle",
            triples.clone(),
            [20_000, 100, 20_000],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 3, Term::Var(1)),
                    atom(Term::Var(1), 17, Term::Var(2)),
                    atom(Term::Var(2), 41, Term::Var(0)),
                ],
            },
            None,
        );
        let first = triples[1234];
        join_case(
            "uniform-anchored",
            triples,
            [20_000, 100, 20_000],
            JoinQuery {
                num_vars: 2,
                patterns: vec![
                    atom(Term::Const(first.0), first.1, Term::Var(0)),
                    atom(Term::Var(0), 17, Term::Var(1)),
                ],
            },
            None,
        );
    }
    if selected("skew") {
        let n = 2_000_000;
        let mut rng = SplitMix64::new(42);
        let mut triples = Vec::new();
        let mut classes = Vec::new();
        for s in 0..n {
            let c = rng.next_range(20) as u32;
            classes.push(c);
            triples.push((s, 0, c));
        }
        let mut oracle = Vec::new();
        for s in 0..n {
            if rng.next_f64() < 0.01 {
                let o = rng.next_range(500_000) as u32;
                triples.push((s, 1, o));
                if classes[s as usize] == 0 {
                    oracle.push(vec![s, o]);
                }
            }
        }
        join_case(
            "skew",
            triples,
            [n as u64, 2, 500_000],
            JoinQuery {
                num_vars: 2,
                patterns: vec![
                    atom(Term::Var(0), 0, Term::Const(0)),
                    atom(Term::Var(0), 1, Term::Var(1)),
                ],
            },
            Some(oracle),
        );
    }
    for n in [32, 64, 128] {
        let name = format!("parity-triangle-{n}");
        if !selected(&name) {
            continue;
        }
        let mut triples = Vec::new();
        for g in 0..2 {
            for a in 0..n {
                for b in 0..n {
                    triples.push((g * n + a, 0, g * n + b));
                    triples.push((g * n + a, 1, g * n + b));
                    triples.push((g * n + a, 2, (1 - g) * n + b));
                }
            }
        }
        join_case(
            &name,
            triples,
            [(2 * n) as u64, 3, (2 * n) as u64],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 0, Term::Var(1)),
                    atom(Term::Var(1), 1, Term::Var(2)),
                    atom(Term::Var(2), 2, Term::Var(0)),
                ],
            },
            Some(vec![]),
        );
    }
    if selected("mixed-triangle") {
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
        let oracle = (n..2 * n).map(|y| vec![n, y, n]).collect();
        join_case(
            "mixed-triangle",
            triples,
            [(2 * n) as u64, 3, (2 * n) as u64],
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 0, Term::Var(1)),
                    atom(Term::Var(1), 1, Term::Var(2)),
                    atom(Term::Var(2), 2, Term::Var(0)),
                ],
            },
            Some(oracle),
        );
    }
    if selected("dense-star") {
        let n = 128;
        let triples = (0..n).flat_map(|i| [(0, 0, i), (0, 1, n + i)]).collect();
        let oracle = (0..n)
            .flat_map(|a| (0..n).map(move |b| vec![a, n + b]))
            .collect();
        join_case(
            "dense-star",
            triples,
            [1, 2, 2 * n as u64],
            JoinQuery {
                num_vars: 2,
                patterns: vec![
                    atom(Term::Const(0), 0, Term::Var(0)),
                    atom(Term::Const(0), 1, Term::Var(1)),
                ],
            },
            Some(oracle),
        );
    }
}
fn main() {
    eprintln!(
        "query hash: {}",
        std::env::var("PROTO_HASH").as_deref().unwrap_or("fast")
    );
    eprintln!(
        "assignment order: {}",
        std::env::var("PROTO_ORDER")
            .as_deref()
            .unwrap_or("shared-first")
    );
    println!(
        "kind,case,engine,rows,compile_us,visit_us,end_to_end_us,end_to_end_min_us,end_to_end_max_us,plan_heap_bytes,retained_states,source_reads,source_reuses,work,workspace_heap_bytes"
    );
    match std::env::var("PROTO_SUITE").as_deref().unwrap_or("all") {
        "filters" => filters(),
        "joins" => joins(),
        "all" => {
            filters();
            joins();
        }
        x => panic!("unknown suite {x}"),
    }
}
