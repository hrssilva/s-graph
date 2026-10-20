//! One engine/phase per process for perf sampling, after correctness was checked by the main harness.
use s_graph::compiled_filter::{CompiledFilter, LegacyFilter};
use s_graph::demand_join::{DemandJoin, DemandWorkspace};
use s_graph::fst_join::{FstJoin, FstJoinWorkspace};
use s_graph::graph::{JoinQuery, Term, TriplePattern};
use s_graph::lifted_join::LiftedJoin;
use s_graph::path_fst::PathFstDeltaPow2Bfs;
use s_graph::tree::DimRestriction;
use std::{hint::black_box, ops::ControlFlow};
#[path = "rdf_gen.rs"]
#[allow(dead_code)]
mod rdf_gen;
// Optional perf FIFO control excludes graph construction and teardown entirely.
fn perf_control(command: &str) {
    use std::io::{BufRead, Write};
    if let Ok(path) = std::env::var("PROTO_PERF_CONTROL") {
        let mut control = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        writeln!(control, "{command}").unwrap();
        if let Ok(path) = std::env::var("PROTO_PERF_ACK") {
            let mut response = String::new();
            std::io::BufReader::new(std::fs::File::open(path).unwrap())
                .read_line(&mut response)
                .unwrap();
            assert_eq!(response.trim(), "ack");
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let engine = args.get(1).map(String::as_str).unwrap_or("compiled");
    let phase = args.get(2).map(String::as_str).unwrap_or("visit");
    let repeats: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5000);
    let tree = PathFstDeltaPow2Bfs::build(
        rdf_gen::uniform(20_000, 100, 20_000, 1_000_000, 42),
        [20_000, 100, 20_000],
        s_graph::DEFAULT_BLOCK_CAPACITY,
    );
    let subjects: Vec<_> = (0..8).map(|i| (2000 * i, 2000 * i + 400)).collect();
    let predicates = [(0, 0), (2, 3), (6, 7), (12, 13)];
    let r = [
        DimRestriction::Multi(&subjects),
        DimRestriction::Multi(&predicates),
        DimRestriction::Multi(&subjects),
    ];
    let query = JoinQuery {
        num_vars: 3,
        patterns: vec![
            TriplePattern {
                subject: Term::Var(0),
                predicate: Term::Const(3),
                object: Term::Var(2),
            },
            TriplePattern {
                subject: Term::Var(1),
                predicate: Term::Const(17),
                object: Term::Var(2),
            },
        ],
    };
    let sink = |t: [u32; 3]| {
        black_box(t);
        ControlFlow::<()>::Continue(())
    };
    perf_control("enable");
    match (engine, phase) {
        ("demand", "visit") | ("demand-table", "visit") => {
            let plan = DemandJoin::with_tables(&query, &tree, engine == "demand-table").unwrap();
            let mut ws = DemandWorkspace::default();
            for _ in 0..repeats {
                let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            }
        }
        ("fst", "visit") => {
            let plan = FstJoin::new(&query, &tree).unwrap();
            let mut ws = FstJoinWorkspace::default();
            for _ in 0..repeats {
                let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                    black_box(r);
                    ControlFlow::Continue(())
                });
            }
        }
        ("legacy", "visit") | ("normalized", "visit") => {
            let plan = LegacyFilter::new(&tree, r, engine == "normalized");
            for _ in 0..repeats {
                let _ = plan.visit(sink);
            }
        }
        ("compiled", "visit") => {
            let plan = CompiledFilter::new(&tree, r).unwrap();
            for _ in 0..repeats {
                let _ = plan.visit(sink);
            }
        }
        ("compiled", "compile") => {
            for _ in 0..repeats {
                black_box(CompiledFilter::new(&tree, r).unwrap());
            }
        }
        ("monolithic", "compile") => {
            for _ in 0..repeats {
                black_box(CompiledFilter::new_monolithic(&tree, r).unwrap());
            }
        }
        ("lifted-lazy", "compile") | ("lifted-merged", "compile") => {
            for _ in 0..repeats {
                black_box(
                    LiftedJoin::with_order_and_hash(
                        &query,
                        &tree,
                        engine == "lifted-merged",
                        true,
                        std::env::var("PROTO_HASH").as_deref().unwrap_or("fast") != "std",
                    )
                    .unwrap(),
                );
            }
        }
        _ => panic!("unsupported engine/phase"),
    }
    perf_control("disable");
    eprintln!("{engine} {phase}: {repeats} repetitions");
}
