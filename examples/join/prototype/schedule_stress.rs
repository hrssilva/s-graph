//! Force schedules on carry chains with analytical answers and bounded diagnostics.
use s_graph::automata_core::{FstSource, TripleSource};
use s_graph::graph::{JoinQuery, Term};
use s_graph::layout_source::Pow2Source;
use s_graph::path_fst::PathFstDeltaPow2;
use s_graph::scheduled_join::{Policy, ScheduleStats, ScheduleWorkspace, ScheduledJoin};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

fn run<S: TripleSource>(
    mode: &str,
    source: S,
    query: &JoinQuery<Term>,
    expected: &[Vec<u32>],
    policy_name: &str,
    atoms: usize,
    layout: &str,
    budget: u64,
) {
    let policy = match policy_name {
        "adaptive" => Policy::Adaptive,
        "levels" => Policy::Levels,
        "frontier" => Policy::Frontier,
        _ => panic!("unknown policy"),
    };
    let start = Instant::now();
    let plan = ScheduledJoin::with_source(query, source, policy).unwrap();
    let prepare_us = start.elapsed().as_secs_f64() * 1e6;
    let mut ws = ScheduleWorkspace::default();
    let mut actual = Vec::new();
    let mut stats = ScheduleStats::default();
    let start = Instant::now();
    let finished = if mode == "budget" {
        plan.prototype_visit_budget(&mut ws, &mut stats, budget, |row| actual.push(row.to_vec()))
    } else {
        assert_eq!(mode, "full");
        let _: ControlFlow<()> = plan.visit(&mut ws, |row| {
            black_box(row);
            actual.push(row.to_vec());
            ControlFlow::Continue(())
        });
        true
    };
    let visit_us = start.elapsed().as_secs_f64() * 1e6;
    actual.sort_unstable();
    if finished {
        assert_eq!(actual, expected);
    } else {
        assert!(actual.iter().all(|r| expected.contains(r)));
    }
    println!(
        concat!(
            "{{\"mode\":\"{}\",\"layout\":\"{}\",\"policy\":\"{}\",\"atoms\":{},\"finished\":{},",
            "\"rows\":{},\"expected_rows\":{},\"prepare_us\":{},\"visit_us\":{},\"nodes\":{},\"candidates\":{},",
            "\"planning_nodes\":{},\"cut\":{},\"plan_bytes\":{},\"workspace_bytes\":{}}}"
        ),
        mode,
        layout,
        policy_name,
        atoms,
        finished,
        actual.len(),
        expected.len(),
        prepare_us,
        visit_us,
        stats.nodes,
        stats.candidates,
        plan.planning.nodes,
        plan.planning.selected_cut,
        plan.allocated_bytes(),
        ws.allocated_bytes()
    );
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mode = &args[1];
    let layout = &args[2];
    let policy = &args[3];
    let atoms = args[4].parse().unwrap();
    let budget = args.get(5).map(|x| x.parse().unwrap()).unwrap_or(1_000_000);
    let (bfs, query, expected) = s_graph::large_bgp_cases::fixture("carry-chain", atoms);
    match layout.as_str() {
        "bfs" => run(
            mode,
            FstSource(&bfs),
            &query,
            &expected,
            policy,
            atoms,
            layout,
            budget,
        ),
        "pow2" => {
            let pow2 = PathFstDeltaPow2::build(
                bfs.scan([None; 3]).collect(),
                bfs.dimension_sizes(),
                bfs.block_capacity(),
            );
            run(
                mode,
                Pow2Source(&pow2),
                &query,
                &expected,
                policy,
                atoms,
                layout,
                budget,
            );
        }
        _ => panic!("unknown layout"),
    }
}
