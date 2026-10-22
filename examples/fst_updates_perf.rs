//! Deterministic update/storage comparison. Run in release/profiling mode.
use s_graph::automata::{
    Bgp, Search,
    Term::{Const as C, Var as V},
};
use s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric as Tree;
use s_graph::{PackedFst, Triple};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_secs_f64() * 1e6)
}
fn visit<P: Search>(plan: &P, ws: &mut P::Workspace) -> usize {
    let mut rows = 0;
    let _: ControlFlow<()> = plan.visit(ws, |row| {
        black_box(row);
        rows += 1;
        ControlFlow::Continue(())
    });
    rows
}
fn run<S: s_graph::storage::ReclaimingBlockStorage>(
    name: &str,
    n: u32,
    edits: usize,
    trials: usize,
    shape: &str,
) -> s_graph::Result<()> {
    let initial: Vec<Triple> = (0..n)
        .flat_map(|s| {
            (0..2).map(move |p| {
                (
                    s,
                    p,
                    if shape == "functional" {
                        s ^ (p + 1)
                    } else {
                        mix(s ^ (p * 0x1234)) % n
                    },
                )
            })
        })
        .collect();
    let mut batch: Vec<_> = (0..edits as u32)
        .map(|s| (s, 3, mix(s.wrapping_add(456)) % n))
        .collect();
    let (base, build_us) = timed(|| {
        PackedFst::<S>::try_build(initial.clone(), [n as u64, 4, n as u64], 4096).unwrap()
    });
    let base_bytes = base.mem_size_bytes();
    let base_nodes = base.node_count();
    let machine = Bgp::new(2).triple([V(0), C(0), V(1)]);
    let mut packed_visits = Vec::new();
    let packed_search = base.search(&machine)?;
    let mut ws = Default::default();
    assert_eq!(visit(&packed_search, &mut ws), n as usize);
    for _ in 0..trials {
        packed_visits.push(timed(|| assert_eq!(visit(&packed_search, &mut ws), n as usize)).1);
    }
    drop(packed_search);
    let mut graph = base.into_mutable();
    let (_, prepare_us) = timed(|| graph.prepare_updates());
    let prepared_bytes = graph.mem_size_bytes();
    let mut insert = Vec::new();
    let mut remove = Vec::new();
    let mut edited_visit = Vec::new();
    let mut edited_bytes = 0;
    let mut edited_nodes = 0;
    for _ in 0..trials {
        insert.push(
            timed(|| assert_eq!(graph.insert_batch(&mut batch).unwrap(), edits as u64)).1
                / edits as f64,
        );
        edited_bytes = graph.mem_size_bytes();
        edited_nodes = graph.edited_node_count();
        let search = graph.search(&machine)?;
        let mut ws = Default::default();
        assert_eq!(visit(&search, &mut ws), n as usize);
        edited_visit.push(timed(|| assert_eq!(visit(&search, &mut ws), n as usize)).1);
        drop(search);
        remove.push(
            timed(|| assert_eq!(graph.remove_batch(&mut batch).unwrap(), edits as u64)).1
                / edits as f64,
        );
        assert_eq!(
            graph.edited_node_count(),
            0,
            "inverse batch failed to reuse packed base"
        );
    }
    graph.insert_batch(&mut batch)?;
    let (packed, compact_us) = timed(|| graph.freeze().unwrap());
    assert_eq!(
        packed.count_matching([None; 3]),
        2 * n as u64 + edits as u64
    );
    println!(
        "{{\"backend\":\"{name}\",\"engine\":\"fst\",\"shape\":\"{shape}\",\"vertices\":{n},\"edits\":{edits},\"trials\":{trials},\"build_us\":{build_us},\"prepare_us\":{prepare_us},\"insert_us_per_triple\":{},\"remove_us_per_triple\":{},\"packed_visit_us\":{},\"edited_visit_us\":{},\"compact_us\":{compact_us},\"base_nodes\":{base_nodes},\"edited_nodes\":{edited_nodes},\"base_ram_bytes\":{base_bytes},\"prepared_ram_bytes\":{prepared_bytes},\"edited_ram_bytes\":{edited_bytes},\"compacted_ram_bytes\":{}}}",
        median(insert),
        median(remove),
        median(packed_visits),
        median(edited_visit),
        packed.mem_size_bytes()
    );
    let (mut tree, build_us) =
        timed(|| Tree::<S>::build_dynamic_empty([n as u64, 4, n as u64], 4096));
    let mut initial = initial;
    let (_, bulk_insert_us) = timed(|| tree.insert_batch(&mut initial).unwrap());
    let base_bytes = tree.mem_size_bytes();
    let mut insert = Vec::new();
    let mut remove = Vec::new();
    for _ in 0..trials {
        insert.push(
            timed(|| assert_eq!(tree.insert_batch(&mut batch).unwrap(), edits as u64)).1
                / edits as f64,
        );
        remove.push(
            timed(|| assert_eq!(tree.remove_batch(&mut batch).unwrap(), edits as u64)).1
                / edits as f64,
        );
    }
    assert_eq!(tree.count_matching([None; 3]), 2 * n as u64);
    println!(
        "{{\"backend\":\"{name}\",\"engine\":\"tree\",\"shape\":\"{shape}\",\"vertices\":{n},\"edits\":{edits},\"trials\":{trials},\"empty_build_us\":{build_us},\"initial_insert_us\":{bulk_insert_us},\"insert_us_per_triple\":{},\"remove_us_per_triple\":{},\"base_ram_bytes\":{base_bytes}}}",
        median(insert),
        median(remove)
    );
    Ok(())
}
fn main() -> s_graph::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(16384, |s| s.parse::<u32>().unwrap());
    let edits = args.get(2).map_or(256, |s| s.parse::<usize>().unwrap());
    let trials = args.get(3).map_or(5, |s| s.parse::<usize>().unwrap());
    assert!(n.is_power_of_two() && n >= 4 && edits > 0 && edits <= n as usize && trials > 0);
    for shape in ["functional", "mixed"] {
        run::<s_graph::storage::HeapStorage>("heap", n, edits, trials, shape)?;
        #[cfg(feature = "file")]
        run::<s_graph::storage_file::FileStorage>("file", n, edits, trials, shape)?;
        #[cfg(feature = "compressed")]
        run::<s_graph::storage_compressed::CompressedStorage>(
            "compressed",
            n,
            edits,
            trials,
            shape,
        )?;
    }
    Ok(())
}
