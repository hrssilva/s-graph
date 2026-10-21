use s_graph::{
    automaton_join::AutomatonJoin,
    fst_join::{FstJoin, FstJoinWorkspace},
    large_bgp_cases::fixture,
    scheduled_join::{HybridJoin, Policy, ScheduleWorkspace},
};
use std::{hint::black_box, ops::ControlFlow, time::Instant};
fn time(mut f: impl FnMut()) -> f64 {
    f();
    let mut batch = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..batch {
            f();
        }
        if start.elapsed().as_millis() >= 2 || batch >= 8192 {
            break;
        }
        batch *= 2;
    }
    let mut values = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..batch {
            f();
        }
        values.push(t.elapsed().as_secs_f64() * 1e6 / batch as f64);
    }
    values.sort_by(f64::total_cmp);
    values[2]
}
fn run() {
    println!(
        "shape,atoms,variables,rows,engine,prepare_us,visit_us,total_us,plan_bytes,workspace_bytes"
    );
    for (kind, n) in [
        ("chain", 64),
        ("chain", 1024),
        ("chain", 4096),
        ("star", 1024),
        ("cycle", 4096),
        ("cycle", 4097),
        ("carry-chain", 20),
        ("carry-chain", 4096),
    ] {
        let (tree, q, expected) = fixture(kind, n);
        let p = HybridJoin::new(&q, &tree, Policy::Adaptive).unwrap();
        let mut ws = ScheduleWorkspace::default();
        let mut rows = Vec::new();
        let _: ControlFlow<()> = p.visit(&mut ws, |r| {
            rows.push(r.to_vec());
            ControlFlow::Continue(())
        });
        rows.sort_unstable();
        assert_eq!(rows, expected);
        let prep = time(|| {
            black_box(HybridJoin::new(&q, &tree, Policy::Adaptive).unwrap());
        });
        let visit = time(|| {
            let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        let total = time(|| {
            let p = HybridJoin::new(&q, &tree, Policy::Adaptive).unwrap();
            let mut ws = ScheduleWorkspace::default();
            let _: ControlFlow<()> = p.visit(&mut ws, |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        println!(
            "{kind},{n},{},{},scheduled,{prep},{visit},{total},{},{}",
            q.num_vars,
            rows.len(),
            p.allocated_bytes(),
            ws.allocated_bytes()
        );
        let a = AutomatonJoin::new(&q, &tree).unwrap();
        let mut bindings = Vec::new();
        rows.clear();
        let _: ControlFlow<()> = a.visit(&mut bindings, |r| {
            rows.push(r.to_vec());
            ControlFlow::Continue(())
        });
        rows.sort_unstable();
        assert_eq!(rows, expected);
        let prep = time(|| {
            black_box(AutomatonJoin::new(&q, &tree).unwrap());
        });
        let visit = time(|| {
            let _: ControlFlow<()> = a.visit(&mut bindings, |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        let total = time(|| {
            let a = AutomatonJoin::new(&q, &tree).unwrap();
            let _: ControlFlow<()> = a.visit(&mut Vec::new(), |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        println!(
            "{kind},{n},{},{},automaton,{prep},{visit},{total},{},{}",
            q.num_vars,
            rows.len(),
            a.prototype_allocated_bytes(),
            bindings.capacity() * 4
        );
        // Huge carry chains have exponential coarse-prefix intermediates
        // under level-major execution. Keep that comparison bounded at 20.
        if kind == "carry-chain" && n > 64 {
            continue;
        }
        let f = FstJoin::new(&q, &tree).unwrap();
        let mut fw = FstJoinWorkspace::default();
        rows.clear();
        let _: ControlFlow<()> = f.visit(&mut fw, |r| {
            rows.push(r.to_vec());
            ControlFlow::Continue(())
        });
        rows.sort_unstable();
        assert_eq!(rows, expected);
        let prep = time(|| {
            black_box(FstJoin::new(&q, &tree).unwrap());
        });
        let visit = time(|| {
            let _: ControlFlow<()> = f.visit(&mut fw, |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        let total = time(|| {
            let f = FstJoin::new(&q, &tree).unwrap();
            let _: ControlFlow<()> = f.visit(&mut FstJoinWorkspace::default(), |r| {
                black_box(r);
                ControlFlow::Continue(())
            });
        });
        println!(
            "{kind},{n},{},{},fst,{prep},{visit},{total},{},{}",
            q.num_vars,
            rows.len(),
            f.allocated_bytes(),
            fw.allocated_bytes()
        );
    }
}
fn main() {
    // Baseline AutomatonJoin has atom-depth recursion. Give the comparison
    // thread room for it; scheduled_join is also tested on a 128 KiB stack.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
