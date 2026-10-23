//! Controlled ten-atom chain; run through report_probe.py in a disposable crate.
//! No TEP3 data or query is assumed by this synthetic diagnostic.
use s_graph::{
    DEFAULT_BLOCK_CAPACITY, Fst, Triple,
    automata_core::FstSource,
    execution::Execution,
    graph::{JoinQuery, Term, TriplePattern},
    scheduled_join::{Policy, ScheduleStats, ScheduleWorkspace, ScheduledJoin},
};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

struct Fixture {
    graph: Fst,
    query: JoinQuery<Term>,
    expected: Vec<Vec<u32>>,
}

fn fixture(n: u32, hops: usize, shuffled: bool) -> Fixture {
    use Term::{Const as C, Var as V};
    assert!(n >= 4 && n.is_power_of_two() && hops > 0);
    let mut ids: Vec<_> = (0..n).collect();
    if shuffled {
        let mut seed = 42u64;
        for i in (1..ids.len()).rev() {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ids.swap(i, (z % (i as u64 + 1)) as usize);
        }
    }
    let id = |v: u32| ids[(v % n) as usize];
    let mut triples: Vec<Triple> = Vec::new();
    let mut expected = Vec::new();
    for v in 0..n {
        triples.extend([(id(v), 0, id(v + 1)), (id(v), 2, n + 1), (id(v), 3, n + 2)]);
        if v % 8 == 0 {
            triples.push((id(v), 1, n));
            let start = (v + n - (hops as u32 % n)) % n;
            let mut row: Vec<_> = (0..=hops).map(|i| id(start + i as u32)).collect();
            row.push(n + 1);
            expected.push(row);
        }
    }
    expected.sort_unstable();
    let mut atoms: Vec<_> = (0..hops).map(|i| [V(i), C(0), V(i + 1)]).collect();
    atoms.extend([
        [V(hops), C(1), C(n)],
        [V(hops), C(2), V(hops + 1)],
        [V(0), C(3), C(n + 2)],
    ]);
    Fixture {
        graph: Fst::build(triples, [n as u64, 4, n as u64 + 3], DEFAULT_BLOCK_CAPACITY),
        query: JoinQuery {
            num_vars: hops + 2,
            patterns: atoms
                .into_iter()
                .map(|a| TriplePattern {
                    subject: a[0],
                    predicate: a[1],
                    object: a[2],
                })
                .collect(),
        },
        expected,
    }
}

fn plan<'a>(f: &'a Fixture, name: &str) -> ScheduledJoin<FstSource<'a>> {
    if name == "connected" {
        // Only the disposable crate gains this experimental constructor.
        return ScheduledJoin::prototype_connected(&f.query, FstSource(&f.graph)).unwrap();
    }
    ScheduledJoin::new(
        &f.query,
        &f.graph,
        match name {
            "adaptive" | "controlled" => Policy::Adaptive,
            "nested" => Policy::Nested,
            "levels" => Policy::Levels,
            "frontier" => Policy::Frontier,
            _ => panic!("unknown policy {name}"),
        },
    )
    .unwrap()
}

fn median_us(trials: usize, mut work: impl FnMut()) -> f64 {
    work();
    let start = Instant::now();
    work();
    let repeats =
        (10_000. / (start.elapsed().as_secs_f64() * 1e6).max(0.01)).clamp(1., 100_000.) as usize;
    let mut samples = Vec::new();
    for _ in 0..trials {
        let start = Instant::now();
        for _ in 0..repeats {
            work();
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / repeats as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn sink(row: &[u32]) -> ControlFlow<()> {
    black_box(row);
    ControlFlow::Continue(())
}

fn visit(p: &ScheduledJoin<FstSource<'_>>, ws: &mut ScheduleWorkspace, controlled: bool) {
    if controlled {
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let control = Execution::new()
            .with_budget(1_000_000_000)
            .with_cancellation(&cancelled);
        let _ = black_box(p.visit_controlled(ws, black_box(&control), sink));
        black_box(control.work());
    } else {
        let _ = black_box(p.visit(ws, sink));
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: u32 = args[1].parse().unwrap();
    let shuffled = args[2] == "shuffled";
    assert!(shuffled || args[2] == "ordered");
    let name = &args[3];
    let budget: u64 = args[4].parse().unwrap();
    let f = fixture(n, 7, shuffled);
    let p = plan(&f, name);
    let mut ws = ScheduleWorkspace::default();
    let mut stats = ScheduleStats::default();
    let mut actual = Vec::new();
    let finished =
        p.prototype_visit_budget(&mut ws, &mut stats, budget, |row| actual.push(row.to_vec()));
    actual.sort_unstable();
    assert!(
        actual.windows(2).all(|w| w[0] != w[1]),
        "duplicate full bindings"
    );
    if finished {
        assert_eq!(actual, f.expected);
    } else {
        assert!(actual.iter().all(|r| f.expected.binary_search(r).is_ok()));
    }
    let (prep, visit, total) = if finished {
        let prep = median_us(5, || {
            black_box(plan(&f, name));
        });
        let visit = median_us(5, || {
            visit(&p, &mut ws, name == "controlled");
        });
        let total = median_us(5, || {
            let p = plan(&f, name);
            self::visit(&p, &mut ScheduleWorkspace::default(), name == "controlled");
        });
        (format!("{prep}"), format!("{visit}"), format!("{total}"))
    } else {
        ("null".into(), "null".into(), "null".into())
    };
    println!(
        concat!(
            "{{\"n\":{},\"ids\":\"{}\",\"policy\":\"{}\",\"finished\":{},\"rows\":{},\"expected_rows\":{},",
            "\"nodes\":{},\"candidates\":{},\"outer_rows\":{},\"terminal_checks\":{},",
            "\"planning_nodes\":{},\"order\":{:?},\"cut\":{},\"frontier\":{},",
            "\"prepare_us\":{},\"visit_us\":{},\"total_us\":{},\"plan_bytes\":{},\"workspace_bytes\":{}}}"
        ),
        n,
        args[2],
        name,
        finished,
        actual.len(),
        f.expected.len(),
        stats.nodes,
        stats.candidates,
        stats.outer_rows,
        stats.terminal_checks,
        p.planning.nodes,
        p.planning.selected_order,
        p.planning.selected_cut,
        p.planning.selected_frontier,
        prep,
        visit,
        total,
        p.allocated_bytes(),
        ws.allocated_bytes()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_answers_match_exhaustive_assignments() {
        for shuffled in [false, true] {
            let f = fixture(4, 2, shuffled);
            let edges: std::collections::HashSet<_> = f.graph.scan([None; 3]).collect();
            let mut actual = Vec::new();
            for mut code in 0..7u32.pow(f.query.num_vars as u32) {
                let mut row = vec![0; f.query.num_vars];
                for v in &mut row {
                    *v = code % 7;
                    code /= 7;
                }
                let value = |t| match t {
                    Term::Const(c) => c,
                    Term::Var(v) => row[v],
                };
                if f.query.patterns.iter().all(|p| {
                    edges.contains(&(value(p.subject), value(p.predicate), value(p.object)))
                }) {
                    actual.push(row);
                }
            }
            actual.sort_unstable();
            assert_eq!(actual, f.expected);
            for name in ["nested", "adaptive", "levels", "frontier", "connected"] {
                let p = plan(&f, name);
                let mut actual = Vec::new();
                let _: ControlFlow<()> = p.visit(&mut Default::default(), |r| {
                    actual.push(r.to_vec());
                    ControlFlow::Continue(())
                });
                actual.sort_unstable();
                assert_eq!(actual, f.expected, "{name}, shuffled={shuffled}");
            }
        }
    }
}
