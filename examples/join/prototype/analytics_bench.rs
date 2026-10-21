use s_graph::{
    DEFAULT_BLOCK_CAPACITY, Triple,
    analytics::{
        self, CompressedEdges, Csr, EdgeSource, LabelPropagation, PageRank, ScanWorkspace,
    },
    automata_core::FstSource,
    path_fst::PathFstDeltaPow2Bfs,
};
use std::{hint::black_box, time::Instant};
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
fn profile<S: EdgeSource>(source: &S, algorithm: &str, repeats: usize) {
    let mut pr = PageRank::default();
    let mut lp = LabelPropagation::default();
    let mut run = || match algorithm {
        "pagerank" => {
            black_box(pr.run(source, 0.85, 1e-12, 50));
            black_box(&pr.rank);
        }
        "label-propagation" => {
            black_box(lp.run(source, 50));
            black_box(&lp.labels);
        }
        _ => panic!("unknown algorithm"),
    };
    run();
    control("enable");
    let start = Instant::now();
    for _ in 0..repeats {
        run();
    }
    let elapsed = start.elapsed().as_secs_f64();
    control("disable");
    println!("{{\"algorithm\":\"{algorithm}\",\"repeats\":{repeats},\"elapsed_s\":{elapsed}}}");
}
fn measure(mut f: impl FnMut()) -> f64 {
    f();
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    times[2]
}
fn data(kind: &str, n: u32) -> Vec<Triple> {
    let mut seed = 713u64;
    let mut triples = Vec::new();
    for s in 0..n {
        for j in 0..16 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let o = match kind {
                "communities" if j < 15 => (s / 64) * 64 + (seed as u32 % 64),
                "skew-dangling" => {
                    if j < 10 {
                        seed as u32 % 128
                    } else {
                        seed as u32 % n
                    }
                }
                _ => seed as u32 % n,
            };
            if kind != "skew-dangling" || s < n / 2 {
                triples.push((s, 0, o));
            }
            if j % 4 == 0 {
                triples.push((s, 1, (o + 37) % n));
            }
        }
    }
    triples.sort_unstable();
    triples.dedup();
    triples
}
fn run_pr<S: EdgeSource>(source: &S) -> (f64, usize, bool, usize, Vec<f64>) {
    let mut p = PageRank::default();
    let status = p.run(source, 0.85, 1e-12, 50);
    let ms = measure(|| {
        black_box(p.run(source, 0.85, 1e-12, 50));
        black_box(&p.rank);
    });
    (ms, status.0, status.1, p.allocated_bytes(), p.rank)
}
fn run_lp<S: EdgeSource>(source: &S) -> (f64, usize, bool, usize, Vec<u32>) {
    let mut p = LabelPropagation::default();
    let status = p.run(source, 50);
    let ms = measure(|| {
        black_box(p.run(source, 50));
        black_box(&p.labels);
    });
    (ms, status.0, status.1, p.allocated_bytes(), p.labels)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = std::env::var("ANALYTICS_VERTICES")
        .ok()
        .map(|x| x.parse().unwrap())
        .unwrap_or(16384);
    if args.len() == 1 {
        println!(
            "case,algorithm,backend,vertices,selected_edges,rounds,converged,prepare_ms,run_ms,cold_total_ms,graph_bytes,plan_bytes,algorithm_bytes"
        );
    }
    assert!(
        n > 0 && n % 64 == 0,
        "benchmark vertex count must be a positive multiple of 64"
    );
    for kind in ["uniform", "skew-dangling", "communities"] {
        if std::env::var("ANALYTICS_CASE").is_ok_and(|c| c != kind) {
            continue;
        }
        let triples = data(kind, n);
        let selected: Vec<_> = triples.iter().copied().filter(|t| t.1 == 0).collect();
        let build = Instant::now();
        let tree = PathFstDeltaPow2Bfs::build(
            triples.clone(),
            [n as u64, 2, n as u64],
            DEFAULT_BLOCK_CAPACITY,
        );
        eprintln!(
            "{kind}: {} selected / {} stored triples; FST build {:.3}ms, {} bytes",
            selected.len(),
            triples.len(),
            build.elapsed().as_secs_f64() * 1e3,
            tree.mem_size_bytes()
        );
        let compressed = CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap();
        let csr = Csr::from_source(&compressed, Some(0));
        // --profile fst|csr pagerank|label-propagation REPEATS. Warmup/setup
        // precede the perf enable handshake; each run includes degree setup.
        if args.len() > 1 {
            assert_eq!(args.len(), 5);
            assert_eq!(args[1], "--profile");
            match args[2].as_str() {
                "fst" => profile(&compressed, &args[3], args[4].parse().unwrap()),
                "csr" => profile(&csr, &args[3], args[4].parse().unwrap()),
                _ => panic!("unknown backend"),
            }
            return;
        }
        let prepare_fst = measure(|| {
            black_box(CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap());
        });
        let prepare_csr = measure(|| {
            black_box(Csr::from_source(&compressed, Some(0)));
        });
        let reference = analytics::reference_pagerank(&selected, n as usize, 0.85, 1e-12, 50);
        let reference_lp = analytics::reference_labels(&selected, n as usize, 50);
        let pr_fst = run_pr(&compressed);
        let pr_csr = run_pr(&csr);
        let lp_fst = run_lp(&compressed);
        let lp_csr = run_lp(&csr);
        for run in [&pr_fst, &pr_csr] {
            assert_eq!((run.1, run.2), (reference.rounds, reference.converged));
            assert!(
                run.4
                    .iter()
                    .zip(&reference.values)
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
            assert!((run.4.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        }
        for run in [&lp_fst, &lp_csr] {
            assert_eq!(run.4, reference_lp.values);
            assert_eq!(
                (run.1, run.2),
                (reference_lp.rounds, reference_lp.converged)
            );
        }
        let cold_pr_fst = measure(|| {
            let s = CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap();
            let mut p = PageRank::default();
            black_box(p.run(&s, 0.85, 1e-12, 50));
            black_box(p.rank);
        });
        let cold_pr_csr = measure(|| {
            let plan = CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap();
            let s = Csr::from_source(&plan, Some(0));
            let mut p = PageRank::default();
            black_box(p.run(&s, 0.85, 1e-12, 50));
            black_box(p.rank);
        });
        let cold_lp_fst = measure(|| {
            let s = CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap();
            let mut p = LabelPropagation::default();
            black_box(p.run(&s, 50));
            black_box(p.labels);
        });
        let cold_lp_csr = measure(|| {
            let plan = CompressedEdges::new(FstSource(&tree), n as usize, Some(0)).unwrap();
            let s = Csr::from_source(&plan, Some(0));
            let mut p = LabelPropagation::default();
            black_box(p.run(&s, 50));
            black_box(p.labels);
        });
        for (backend, prep, graph, plan, pr, lp, cold_pr, cold_lp) in [
            (
                "fst",
                prepare_fst,
                tree.mem_size_bytes(),
                compressed.allocated_bytes(),
                pr_fst,
                lp_fst,
                cold_pr_fst,
                cold_lp_fst,
            ),
            (
                "csr",
                prepare_csr,
                csr.allocated_bytes(),
                0,
                pr_csr,
                lp_csr,
                cold_pr_csr,
                cold_lp_csr,
            ),
        ] {
            println!(
                "{kind},pagerank,{backend},{n},{},{},{},{prep},{},{cold_pr},{graph},{plan},{}",
                selected.len(),
                pr.1,
                pr.2,
                pr.0,
                pr.3
            );
            println!(
                "{kind},label-propagation,{backend},{n},{},{},{},{prep},{},{cold_lp},{graph},{plan},{}",
                selected.len(),
                lp.1,
                lp.2,
                lp.0,
                lp.3
            );
        }
        // Independent scan measurement isolates compressed decoding from reducers.
        let mut ws = ScanWorkspace::default();
        let scan = measure(|| {
            compressed.edges(&mut ws, |e| {
                black_box(e);
            });
        });
        let csr_scan = measure(|| {
            csr.edges(&mut (), |e| {
                black_box(e);
            });
        });
        eprintln!(
            "{kind}: raw scan fst={scan:.3}ms csr={csr_scan:.3}ms, scan workspace={} bytes",
            ws.allocated_bytes()
        );
    }
}
