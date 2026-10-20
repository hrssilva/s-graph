use s_graph::automaton_join::{AutomatonJoin, AutomatonJoinStats};
use s_graph::fst_join::{FstJoin, FstJoinStats, FstJoinWorkspace};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::native_join::{NativeJoin, NativeJoinWorkspace};
use s_graph::path_fst::PathFstDeltaPow2Bfs;
use s_graph::recursive_join::{RecursiveJoin, RecursiveJoinStats};
use s_graph::{DEFAULT_BLOCK_CAPACITY, Heap, Triple};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn fraction(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn atom(s: Term, p: u32, o: Term) -> TriplePattern<Term> {
    TriplePattern {
        subject: s,
        predicate: Term::Const(p),
        object: o,
    }
}
fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .map(|s| s.parse().ok().unwrap())
        .unwrap_or(default)
}
fn selected(name: &str) -> bool {
    std::env::var("JOIN_CASE")
        .map(|s| name.contains(&s))
        .unwrap_or(true)
}
fn engine(name: &str) -> bool {
    std::env::var("JOIN_ENGINE")
        .map(|s| name == s)
        .unwrap_or(true)
}
fn measure(mut f: impl FnMut(), reps: usize) -> (f64, f64, f64) {
    f();
    let mut ts = Vec::new();
    for _ in 0..reps {
        let start = Instant::now();
        f();
        ts.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    ts.sort_by(f64::total_cmp);
    (ts[ts.len() / 2], ts[0], *ts.last().unwrap())
}
fn print(
    name: &str,
    mode: &str,
    time: (f64, f64, f64),
    rows: usize,
    nodes: u64,
    candidates: u64,
    entries: u64,
) {
    println!(
        "{name},{mode},{rows},{:.6},{:.6},{:.6},{nodes},{candidates},{entries}",
        time.0, time.1, time.2
    );
}
fn run(
    name: &str,
    triples: Vec<Triple>,
    dims: [u64; 3],
    q: JoinQuery<Term>,
    oracle: Option<Vec<Vec<u32>>>,
) {
    if !selected(name) {
        return;
    }
    let heap = Heap::build_dynamic(triples.clone(), DEFAULT_BLOCK_CAPACITY);
    let fst = PathFstDeltaPow2Bfs::build(triples, dims, DEFAULT_BLOCK_CAPACITY);
    eprintln!(
        "{name}: heap={} FST={} height={}",
        heap.mem_size_bytes(),
        fst.mem_size_bytes(),
        fst.height()
    );
    let reps = env("JOIN_REPS", 7);
    let mut expected = Vec::new();
    let fp = FstJoin::new(&q, &fst).unwrap();
    let mut fw = FstJoinWorkspace::default();
    let _: ControlFlow<()> = fp.visit(&mut fw, |row| {
        expected.push(row.to_vec());
        ControlFlow::Continue(())
    });
    expected.sort_unstable();
    if let Some(mut oracle) = oracle {
        oracle.sort_unstable();
        assert_eq!(expected, oracle, "oracle on {name}");
    }
    if engine("native") {
        let p = NativeJoin::new(&q, &heap).unwrap();
        let mut ws = NativeJoinWorkspace::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit(&mut ws, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected);
        let t = measure(
            || {
                let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            },
            reps,
        );
        print(name, "native", t, expected.len(), 0, 0, 0);
    }
    if engine("fst") {
        let mut stats = FstJoinStats::default();
        let _: ControlFlow<()> =
            fp.visit_with_stats(&mut fw, &mut stats, |_| ControlFlow::Continue(()));
        let t = measure(
            || {
                let _: ControlFlow<()> = fp.visit(&mut fw, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            },
            reps,
        );
        print(
            name,
            "fst",
            t,
            expected.len(),
            stats.node_visits,
            stats.candidates,
            0,
        );
    }
    if engine("recursive") {
        let p = RecursiveJoin::new(&q, &fst).unwrap();
        let mut ws = Vec::new();
        let mut stats = RecursiveJoinStats::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut stats, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected);
        let t = measure(
            || {
                let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            },
            reps,
        );
        print(
            name,
            "recursive",
            t,
            expected.len(),
            stats.node_visits,
            stats.candidates,
            0,
        );
    }
    for (mode, query) in [
        (
            "automaton",
            JoinQuery {
                num_vars: q.num_vars,
                patterns: q.patterns.clone(),
            },
        ),
        (
            "automaton-reversed",
            JoinQuery {
                num_vars: q.num_vars,
                patterns: q.patterns.iter().cloned().rev().collect(),
            },
        ),
    ] {
        if !engine(mode) {
            continue;
        }
        // The experimental library bypasses its heuristic only when this variable is set by the caller.
        let p = AutomatonJoin::new(&query, &fst).unwrap();
        let mut ws = Vec::new();
        let mut stats = AutomatonJoinStats::default();
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut stats, |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, expected, "{mode} on {name}");
        let t = measure(
            || {
                let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            },
            reps,
        );
        print(
            name,
            mode,
            t,
            expected.len(),
            stats.node_visits,
            stats.candidates,
            stats.atom_entries,
        );
    }
}
fn main() {
    assert!(
        std::env::var_os("JOIN_INPUT_ORDER").is_some(),
        "set JOIN_INPUT_ORDER=1 to enable the order experiment"
    );
    println!("case,engine,rows,median_ms,min_ms,max_ms,node_visits,candidates,atom_entries");
    let n = env("JOIN_SUBJECTS", 2_000_000u32);
    for (classes, rare, objects) in [
        (20, 0.0001, 500_000),
        (20, 0.001, 500_000),
        (20, 0.01, 500_000),
        (20, 0.1, 500_000),
        (2, 0.01, 500_000),
        (200, 0.01, 500_000),
        (20, 0.01, 20),
        (20, 0.01, 2_000_000),
    ] {
        let name = format!("skew-c{classes}-r{rare}-o{objects}");
        if !selected(&name) {
            continue;
        }
        let mut rng = Rng(42);
        let mut triples = Vec::new();
        let mut class_values = Vec::new();
        for s in 0..n {
            let c = rng.range(classes);
            class_values.push(c);
            triples.push((s, 0, c));
        }
        let mut oracle = Vec::new();
        for s in 0..n {
            if rng.fraction() < rare {
                let o = rng.range(objects);
                triples.push((s, 1, o));
                if class_values[s as usize] == 0 {
                    oracle.push(vec![s, o]);
                }
            }
        }
        let q = JoinQuery {
            num_vars: 2,
            patterns: vec![
                atom(Term::Var(0), 0, Term::Const(0)),
                atom(Term::Var(0), 1, Term::Var(1)),
            ],
        };
        run(
            &name,
            triples,
            [n as u64, 2, objects.max(classes) as u64],
            q,
            Some(oracle),
        );
    }
    for size in [32u32, 64, 128] {
        let name = format!("empty-triangle-n{size}");
        if !selected(&name) {
            continue;
        }
        // R = A x B; S = B x C; T = D x A, with C disjoint from D.
        let mut triples = Vec::new();
        for a in 0..size {
            for b in 0..size {
                triples.push((a, 0, size + b));
                triples.push((size + a, 1, 2 * size + b));
                triples.push((3 * size + a, 2, b));
            }
        }
        let q = JoinQuery {
            num_vars: 3,
            patterns: vec![
                atom(Term::Var(0), 0, Term::Var(1)),
                atom(Term::Var(1), 1, Term::Var(2)),
                atom(Term::Var(2), 2, Term::Var(0)),
            ],
        };
        run(
            &name,
            triples,
            [(4 * size) as u64, 3, (4 * size) as u64],
            q,
            Some(vec![]),
        );
    }
    for size in [32u32, 64, 128] {
        for rotation in 0..if size == 64 { 3 } else { 1 } {
            let name = format!("parity-triangle-n{size}-rot{rotation}");
            if !selected(&name) {
                continue;
            }
            // All adjacent pairs join widely; the three high-bit constraints are inconsistent.
            let mut triples = Vec::new();
            for g in 0..2 {
                for a in 0..size {
                    for b in 0..size {
                        triples.push((g * size + a, 0, g * size + b));
                        triples.push((g * size + a, 1, g * size + b));
                        triples.push((g * size + a, 2, (1 - g) * size + b));
                    }
                }
            }
            let mut patterns = vec![
                atom(Term::Var(0), 0, Term::Var(1)),
                atom(Term::Var(1), 1, Term::Var(2)),
                atom(Term::Var(2), 2, Term::Var(0)),
            ];
            patterns.rotate_left(rotation);
            run(
                &name,
                triples,
                [(2 * size) as u64, 3, (2 * size) as u64],
                JoinQuery {
                    num_vars: 3,
                    patterns,
                },
                Some(vec![]),
            );
        }
    }
    if !selected("uniform-star") && !selected("uniform-triangle") {
        return;
    }
    let mut rng = Rng(42);
    let triples: Vec<_> = (0..1_000_000)
        .map(|_| (rng.range(20_000), rng.range(100), rng.range(20_000)))
        .collect();
    for (name, q) in [
        (
            "uniform-star",
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 3, Term::Var(2)),
                    atom(Term::Var(1), 17, Term::Var(2)),
                ],
            },
        ),
        (
            "uniform-triangle",
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    atom(Term::Var(0), 3, Term::Var(1)),
                    atom(Term::Var(1), 17, Term::Var(2)),
                    atom(Term::Var(2), 41, Term::Var(0)),
                ],
            },
        ),
    ] {
        run(name, triples.clone(), [20_000, 100, 20_000], q, None);
    }
}
