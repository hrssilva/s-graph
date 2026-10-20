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
use rdf_gen::SplitMix64;

const P_DOMINANT: u32 = 0;
const P_RARE: u32 = 1;

fn main() {
    let num_s: u32 = 2_000_000;
    let num_classes: u32 = 20;
    let num_o: u32 = 500_000;
    let rare_fraction = 0.01;

    let mut rng = SplitMix64::new(42);
    let mut triples: Vec<Triple> =
        Vec::with_capacity(num_s as usize + (num_s as f64 * rare_fraction) as usize);
    for s in 0..num_s {
        let class = rng.next_range(num_classes as u64) as u32;
        triples.push((s, P_DOMINANT, class));
    }
    let mut rare_count = 0u64;
    for s in 0..num_s {
        if rng.next_f64() < rare_fraction {
            let o = rng.next_range(num_o as u64) as u32;
            triples.push((s, P_RARE, o));
            rare_count += 1;
        }
    }
    eprintln!(
        "{} triples total, {rare_count} with the rare predicate ({:.2}%)",
        triples.len(),
        100.0 * rare_count as f64 / num_s as f64
    );

    let num_p = 2u32;
    let dims = [num_s as u64, num_p as u64, num_o.max(num_classes) as u64];

    let t0 = std::time::Instant::now();
    let heap = Heap::build_dynamic(triples.clone(), s_graph::DEFAULT_BLOCK_CAPACITY);
    eprintln!("built Heap in {:?}", t0.elapsed());
    let t0 = std::time::Instant::now();
    let fst = PathFstDeltaPow2Bfs::build(triples, dims, s_graph::DEFAULT_BLOCK_CAPACITY);
    eprintln!("built PathFstDeltaPow2Bfs in {:?}", t0.elapsed());

    let c1 = 0u32;
    let query = JoinQuery::<Term> {
        num_vars: 2,
        patterns: vec![
            TriplePattern {
                subject: Term::Var(0),
                predicate: Term::Const(P_DOMINANT),
                object: Term::Const(c1),
            },
            TriplePattern {
                subject: Term::Var(0),
                predicate: Term::Const(P_RARE),
                object: Term::Var(1),
            },
        ],
    };

    let mut native_ws = NativeJoinWorkspace::default();
    let mut native_rows = 0u64;
    let t0 = std::time::Instant::now();
    let plan = NativeJoin::new(&query, &heap).unwrap();
    let _: std::ops::ControlFlow<()> = plan.visit(&mut native_ws, |_row| {
        native_rows += 1;
        std::ops::ControlFlow::Continue(())
    });
    let native_elapsed = t0.elapsed();

    let mut fst_ws = FstJoinWorkspace::default();
    let mut fstjoin_rows = 0u64;
    let t0 = std::time::Instant::now();
    let plan = FstJoin::new(&query, &fst).unwrap();
    let _: std::ops::ControlFlow<()> = plan.visit(&mut fst_ws, |_row| {
        fstjoin_rows += 1;
        std::ops::ControlFlow::Continue(())
    });
    let fstjoin_elapsed = t0.elapsed();

    let mut bindings = Vec::new();
    let mut recursive_rows = 0u64;
    let t0 = std::time::Instant::now();
    let plan = RecursiveJoin::new(&query, &fst).unwrap();
    let mut rstats = s_graph::recursive_join::RecursiveJoinStats::default();
    let _: std::ops::ControlFlow<()> = plan.visit_with_stats(&mut bindings, &mut rstats, |_row| {
        recursive_rows += 1;
        std::ops::ControlFlow::Continue(())
    });
    let recursive_elapsed = t0.elapsed();

    let mut bindings2 = Vec::new();
    let mut automaton_rows = 0u64;
    let t0 = std::time::Instant::now();
    let plan = AutomatonJoin::new(&query, &fst).unwrap();
    let mut astats = s_graph::automaton_join::AutomatonJoinStats::default();
    let _: std::ops::ControlFlow<()> = plan.visit_with_stats(&mut bindings2, &mut astats, |_row| {
        automaton_rows += 1;
        std::ops::ControlFlow::Continue(())
    });
    let automaton_elapsed = t0.elapsed();

    println!("?x type C{c1} . ?x P_rare ?b:");
    println!("  NativeJoin (Heap):       {native_rows} rows in {native_elapsed:?}");
    println!("  FstJoin (flat array):    {fstjoin_rows} rows in {fstjoin_elapsed:?}");
    println!(
        "  RecursiveJoin (recur):   {recursive_rows} rows in {recursive_elapsed:?} (node_visits={})",
        rstats.node_visits
    );
    println!(
        "  AutomatonJoin (nested):  {automaton_rows} rows in {automaton_elapsed:?} (atom_entries={})",
        astats.atom_entries
    );
    assert_eq!(native_rows, fstjoin_rows);
    assert_eq!(native_rows, recursive_rows);
    assert_eq!(native_rows, automaton_rows);
    println!();
    println!(
        "  ratio RecursiveJoin/FstJoin:     {:.3}x",
        recursive_elapsed.as_secs_f64() / fstjoin_elapsed.as_secs_f64()
    );
    println!(
        "  ratio RecursiveJoin/NativeJoin:  {:.3}x",
        recursive_elapsed.as_secs_f64() / native_elapsed.as_secs_f64()
    );
    println!(
        "  ratio AutomatonJoin/RecursiveJoin: {:.3}x",
        automaton_elapsed.as_secs_f64() / recursive_elapsed.as_secs_f64()
    );
}
