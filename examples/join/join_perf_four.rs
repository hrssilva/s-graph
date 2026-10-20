use s_graph::automaton_join::AutomatonJoin;
use s_graph::fst_join::{FstJoin, FstJoinWorkspace};
use s_graph::graph::{Term, TriplePattern};
use s_graph::native_join::{NativeJoin, NativeJoinWorkspace};
use s_graph::path_fst::PathFstDeltaPow2Bfs;
use s_graph::recursive_join::RecursiveJoin;
use s_graph::{Heap, Triple, graph::JoinQuery};

#[path = "../common/rdf_gen.rs"]
#[allow(dead_code)]
mod rdf_gen;
use rdf_gen::{SplitMix64, uniform};

fn bench(
    name: &str,
    iters: usize,
    queries: &[JoinQuery<Term>],
    heap: &Heap,
    fst: &PathFstDeltaPow2Bfs,
) {
    let mut native_ws = NativeJoinWorkspace::default();
    let mut native_rows = 0u64;
    let t0 = std::time::Instant::now();
    for q in queries {
        let plan = NativeJoin::new(q, heap).unwrap();
        let _: std::ops::ControlFlow<()> = plan.visit(&mut native_ws, |_row| {
            native_rows += 1;
            std::ops::ControlFlow::Continue(())
        });
    }
    let native_elapsed = t0.elapsed();

    let mut fst_ws = FstJoinWorkspace::default();
    let mut fst_rows = 0u64;
    let t0 = std::time::Instant::now();
    for q in queries {
        let plan = FstJoin::new(q, fst).unwrap();
        let _: std::ops::ControlFlow<()> = plan.visit(&mut fst_ws, |_row| {
            fst_rows += 1;
            std::ops::ControlFlow::Continue(())
        });
    }
    let fst_elapsed = t0.elapsed();

    let mut rbindings = Vec::new();
    let mut recursive_rows = 0u64;
    let t0 = std::time::Instant::now();
    for q in queries {
        let plan = RecursiveJoin::new(q, fst).unwrap();
        let _: std::ops::ControlFlow<()> = plan.visit(&mut rbindings, |_row| {
            recursive_rows += 1;
            std::ops::ControlFlow::Continue(())
        });
    }
    let recursive_elapsed = t0.elapsed();

    let mut bindings = Vec::new();
    let mut automaton_rows = 0u64;
    let t0 = std::time::Instant::now();
    for q in queries {
        let plan = AutomatonJoin::new(q, fst).unwrap();
        let _: std::ops::ControlFlow<()> = plan.visit(&mut bindings, |_row| {
            automaton_rows += 1;
            std::ops::ControlFlow::Continue(())
        });
    }
    let automaton_elapsed = t0.elapsed();

    assert_eq!(native_rows, fst_rows, "fst disagrees with native on {name}");
    assert_eq!(
        native_rows, recursive_rows,
        "recursive disagrees with native on {name}"
    );
    assert_eq!(
        native_rows, automaton_rows,
        "automaton disagrees with native on {name}"
    );

    let native_us = native_elapsed.as_micros() as f64 / iters as f64;
    let fst_us = fst_elapsed.as_micros() as f64 / iters as f64;
    let recursive_us = recursive_elapsed.as_micros() as f64 / iters as f64;
    let automaton_us = automaton_elapsed.as_micros() as f64 / iters as f64;
    println!(
        "{name}: rows/query={:.1}",
        native_rows as f64 / iters as f64
    );
    println!("  NativeJoin (Heap):      {native_us:10.1} us/query");
    println!("  FstJoin (flat array):   {fst_us:10.1} us/query");
    println!(
        "  RecursiveJoin (recur):  {recursive_us:10.1} us/query   {:.3}x FstJoin",
        recursive_us / fst_us
    );
    println!(
        "  AutomatonJoin (nested): {automaton_us:10.1} us/query   {:.3}x FstJoin, {:.3}x RecursiveJoin",
        automaton_us / fst_us,
        automaton_us / recursive_us
    );
}

fn main() {
    let num_s: u32 = 20_000;
    let num_p: u32 = 100;
    let num_o: u32 = 20_000;
    let num_triples = 1_000_000usize;
    let triples: Vec<Triple> = uniform(num_s, num_p, num_o, num_triples, 42);

    let t0 = std::time::Instant::now();
    let heap = Heap::build_dynamic(triples.clone(), s_graph::DEFAULT_BLOCK_CAPACITY);
    eprintln!("built Heap in {:?}", t0.elapsed());
    let t0 = std::time::Instant::now();
    let fst = PathFstDeltaPow2Bfs::build(
        triples,
        [num_s as u64, num_p as u64, num_o as u64],
        s_graph::DEFAULT_BLOCK_CAPACITY,
    );
    eprintln!("built PathFstDeltaPow2Bfs in {:?}", t0.elapsed());

    let mut rng = SplitMix64::new(777);
    let iters = std::env::var("JOIN_ITERS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(200);

    let star: Vec<JoinQuery<Term>> = (0..iters)
        .map(|_| {
            let p1 = rng.next_range(num_p as u64) as u32;
            let p2 = rng.next_range(num_p as u64) as u32;
            JoinQuery::<Term> {
                num_vars: 3,
                patterns: vec![
                    TriplePattern {
                        subject: Term::Var(0),
                        predicate: Term::Const(p1),
                        object: Term::Var(2),
                    },
                    TriplePattern {
                        subject: Term::Var(1),
                        predicate: Term::Const(p2),
                        object: Term::Var(2),
                    },
                ],
            }
        })
        .collect();
    bench("star (shared object, 2 atoms)", iters, &star, &heap, &fst);

    let triangle: Vec<JoinQuery<Term>> = (0..iters)
        .map(|_| {
            let p1 = rng.next_range(num_p as u64) as u32;
            let p2 = rng.next_range(num_p as u64) as u32;
            let p3 = rng.next_range(num_p as u64) as u32;
            JoinQuery::<Term> {
                num_vars: 3,
                patterns: vec![
                    TriplePattern {
                        subject: Term::Var(0),
                        predicate: Term::Const(p1),
                        object: Term::Var(1),
                    },
                    TriplePattern {
                        subject: Term::Var(1),
                        predicate: Term::Const(p2),
                        object: Term::Var(2),
                    },
                    TriplePattern {
                        subject: Term::Var(2),
                        predicate: Term::Const(p3),
                        object: Term::Var(0),
                    },
                ],
            }
        })
        .collect();
    bench("triangle (3 atoms)", iters, &triangle, &heap, &fst);
}
