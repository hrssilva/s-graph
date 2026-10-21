//! Cyclic BGP controls: prefix correlation, ID ordering and dense rejection.
use s_graph::{
    Triple,
    automata_core::{FstSource, TripleSource},
    graph::{JoinQuery, Term, TriplePattern},
    layout_source::Pow2Source,
    path_fst::{PathFstDeltaPow2, PathFstDeltaPow2Bfs},
    scheduled_join::{Policy, ScheduleStats, ScheduleWorkspace, ScheduledJoin},
};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

struct Fixture {
    triples: Vec<Triple>,
    domains: [u64; 3],
    query: JoinQuery<Term>,
    expected: Vec<Vec<u32>>,
}
fn fixture(kind: &str, n: usize, atoms: usize) -> Fixture {
    assert!(n >= 2 && atoms >= 3);
    let mut triples = Vec::new();
    let mut expected = Vec::new();
    let domains;
    let single =
        kind.ends_with("-ring") || matches!(kind, "odd-planted" | "odd-frontier" | "xor-padding");
    if matches!(kind, "odd-planted" | "odd-frontier") {
        assert_eq!(atoms % 2, 1);
        for a in 0..n as u32 {
            for b in 0..n as u32 {
                triples.extend([(a, 0, n as u32 + b), (n as u32 + b, 0, a)]);
            }
        }
        let planted = 2 * n as u32;
        triples.push((planted, 0, planted));
        expected.push(vec![planted; atoms]);
        let domain = if kind == "odd-frontier" {
            assert!(n.is_power_of_two());
            // For n=128, first digits end on the group bit (bit 7),
            // instead of including an unconstrained within-group bit.
            8 * n as u64
        } else {
            planted as u64 + 1
        };
        domains = [domain, 1, domain];
    } else if kind == "dense-planted" {
        for g in 0..2 {
            for a in 0..n as u32 {
                for b in 0..n as u32 {
                    triples.push((g * n as u32 + a, 0, g * n as u32 + b));
                    triples.push((g * n as u32 + a, 1, (1 - g) * n as u32 + b));
                }
            }
        }
        let planted = 2 * n as u32;
        triples.extend([(planted, 0, planted), (planted, 1, planted)]);
        expected.push(vec![planted; atoms]);
        domains = [planted as u64 + 1, 2, planted as u64 + 1];
    } else {
        let padded = kind == "xor-padding";
        let kind = if padded {
            "xor"
        } else {
            kind.strip_suffix("-ring").unwrap_or(kind)
        };
        assert!(matches!(kind, "xor" | "carry" | "shuffled"));
        assert!(n.is_power_of_two());
        let mut ids = (0..n as u32).collect::<Vec<_>>();
        if kind == "shuffled" {
            // Fixed SplitMix64/Fisher-Yates relabeling of the identical ring.
            let mut seed = 42u64;
            for i in (1..n).rev() {
                seed = seed.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                ids.swap(i, (z % (i as u64 + 1)) as usize);
            }
        }
        let period = (1usize << atoms.ilog2()).min(n);
        for s in 0..n {
            if single {
                let at = |distance: usize| {
                    ids[if kind == "xor" {
                        s ^ (distance % 2)
                    } else {
                        (s / period) * period + (s % period + distance) % period
                    }]
                };
                triples.push((ids[s], 0, at(1)));
                if atoms % (if kind == "xor" { 2 } else { period }) == 0 {
                    expected.push((0..atoms).map(at).collect());
                }
                continue;
            }
            let next = if kind == "xor" { s ^ 1 } else { (s + 1) % n };
            let close = if kind == "xor" {
                s ^ ((atoms - 1) % 2)
            } else {
                (s + n - (atoms - 1) % n) % n
            };
            triples.extend([(ids[s], 0, ids[next]), (ids[s], 1, ids[close])]);
            expected.push(
                (0..atoms)
                    .map(|i| {
                        ids[if kind == "xor" {
                            s ^ (i % 2)
                        } else {
                            (s + i) % n
                        }]
                    })
                    .collect(),
            );
        }
        domains = [n as u64, if single && !padded { 1 } else { 2 }, n as u64];
    }
    expected.sort_unstable();
    Fixture {
        triples,
        domains,
        query: JoinQuery {
            num_vars: atoms,
            patterns: (0..atoms)
                .map(|i| TriplePattern {
                    subject: Term::Var(i),
                    predicate: Term::Const(if !single && i + 1 == atoms { 1 } else { 0 }),
                    object: Term::Var((i + 1) % atoms),
                })
                .collect(),
        },
        expected,
    }
}
fn measure(trials: usize, target_us: f64, mut work: impl FnMut()) -> f64 {
    work();
    let start = Instant::now();
    work();
    let repeats = (target_us / (start.elapsed().as_secs_f64() * 1e6).max(0.01))
        .clamp(1., 1_000_000.) as usize;
    let mut times = Vec::new();
    for _ in 0..trials {
        let start = Instant::now();
        for _ in 0..repeats {
            work();
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / repeats as f64);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}
fn sink(row: &[u32]) -> ControlFlow<()> {
    black_box(row);
    ControlFlow::Continue(())
}
fn run<S: TripleSource + Copy>(
    source: S,
    f: &Fixture,
    policy_name: &str,
    budget: u64,
    trials: usize,
    target_us: f64,
) {
    let policy = match policy_name {
        "adaptive" => Policy::Adaptive,
        "levels" => Policy::Levels,
        "nested" => Policy::Nested,
        "frontier" => Policy::Frontier,
        _ => panic!("unknown policy"),
    };
    let plan = ScheduledJoin::with_source(&f.query, source, policy).unwrap();
    let mut ws = ScheduleWorkspace::default();
    let mut stats = ScheduleStats::default();
    let mut actual = Vec::new();
    let finished =
        plan.prototype_visit_budget(&mut ws, &mut stats, budget, |row| actual.push(row.to_vec()));
    actual.sort_unstable();
    if finished {
        assert_eq!(actual, f.expected);
    } else {
        assert!(
            actual
                .iter()
                .all(|row| f.expected.binary_search(row).is_ok())
        );
    }
    // A bounded diagnostic precedes uninstrumented measurements. An exhausted
    // budget is reported explicitly; no partial time is called a query time.
    let (mut prepare, mut visit, mut total) = (None, None, None);
    if finished {
        prepare = Some(measure(trials, target_us, || {
            black_box(ScheduledJoin::with_source(black_box(&f.query), source, policy).unwrap());
        }));
        visit = Some(measure(trials, target_us, || {
            let _ = black_box(plan.visit(&mut ws, sink));
        }));
        total = Some(measure(trials, target_us, || {
            let p = ScheduledJoin::with_source(&f.query, source, policy).unwrap();
            let _ = black_box(p.visit(&mut ScheduleWorkspace::default(), sink));
        }));
    }
    let number = |x: Option<f64>| x.map(|x| x.to_string()).unwrap_or("null".into());
    println!(
        concat!(
            "{{\"finished\":{},\"rows\":{},\"expected_rows\":{},\"nodes\":{},\"candidates\":{},",
            "\"planning_nodes\":{},\"planning_budget\":{},\"frontier\":{},\"cut\":{},\"order\":{:?},\"certified_empty\":{},",
            "\"prepare_us\":{},\"visit_us\":{},\"total_us\":{},\"plan_bytes\":{},",
            "\"workspace_bytes\":{},\"height\":{},\"triples\":{},\"level_layouts\":{:?}}}"
        ),
        finished,
        actual.len(),
        f.expected.len(),
        stats.nodes,
        stats.candidates,
        plan.planning.nodes,
        plan.planning.node_budget,
        plan.planning.selected_frontier,
        plan.planning.selected_cut,
        plan.planning.selected_order,
        plan.planning.certified_empty,
        number(prepare),
        number(visit),
        number(total),
        plan.allocated_bytes(),
        ws.allocated_bytes(),
        source.height(),
        f.triples.len(),
        (0..source.height())
            .map(|level| {
                let (bits, shifts, remaining) = source.layout(level);
                [bits, shifts, remaining]
            })
            .collect::<Vec<_>>()
    );
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let f = fixture(&args[1], args[2].parse().unwrap(), args[3].parse().unwrap());
    let budget = args.get(6).map(|x| x.parse().unwrap()).unwrap_or(5_000_000);
    let trials = std::env::var("CYCLE_TRIALS")
        .map(|x| x.parse().unwrap())
        .unwrap_or(5);
    let target_us = std::env::var("CYCLE_TARGET_MS")
        .map(|x| x.parse::<f64>().unwrap() * 1000.)
        .unwrap_or(50_000.);
    match args[4].as_str() {
        "bfs" => {
            let graph = PathFstDeltaPow2Bfs::build(f.triples.clone(), f.domains, 65536);
            run(FstSource(&graph), &f, &args[5], budget, trials, target_us);
        }
        "pow2" => {
            let graph = PathFstDeltaPow2::build(f.triples.clone(), f.domains, 65536);
            run(Pow2Source(&graph), &f, &args[5], budget, trials, target_us);
        }
        _ => panic!("unknown layout"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_controls_preserve_edges_query_and_answers() {
        for (a, b, n, atoms) in [
            ("xor-ring", "xor-padding", 4096, 128),
            ("odd-planted", "odd-frontier", 128, 129),
        ] {
            let (a, b) = (fixture(a, n, atoms), fixture(b, n, atoms));
            assert_eq!(a.triples, b.triples);
            assert_eq!(a.expected, b.expected);
            assert_eq!(a.query.num_vars, b.query.num_vars);
            assert_eq!(
                format!("{:?}", a.query.patterns),
                format!("{:?}", b.query.patterns)
            );
            assert_ne!(a.domains, b.domains);
        }
    }
    #[test]
    fn analytic_cycle_answers_match_exhaustive_assignments() {
        // Enumerating every assignment checks the independent oracle, including
        // the closing edge and the single planted answer in dense data.
        for kind in [
            "xor",
            "carry",
            "shuffled",
            "dense-planted",
            "xor-ring",
            "carry-ring",
            "shuffled-ring",
            "odd-planted",
        ] {
            for atoms in [3, 4, 5] {
                if kind == "odd-planted" && atoms % 2 == 0 {
                    continue;
                }
                let f = fixture(
                    kind,
                    if matches!(kind, "dense-planted" | "odd-planted") {
                        2
                    } else {
                        4
                    },
                    atoms,
                );
                let n = f.domains[0] as usize;
                let edges = f
                    .triples
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>();
                let mut expected = Vec::new();
                for mut code in 0..n.pow(atoms as u32) {
                    let mut row = vec![0; atoms];
                    for v in &mut row {
                        *v = (code % n) as u32;
                        code /= n;
                    }
                    if (0..atoms).all(|i| {
                        let Term::Const(predicate) = f.query.patterns[i].predicate else {
                            unreachable!()
                        };
                        edges.contains(&(row[i], predicate, row[(i + 1) % atoms]))
                    }) {
                        expected.push(row);
                    }
                }
                expected.sort_unstable();
                assert_eq!(expected, f.expected, "{kind}/{atoms}");
            }
        }
    }
}
