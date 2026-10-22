//! Freeze cost and read locality: full repack versus blocked in-place sealing.
use s_graph::automata::{
    Bgp, Search,
    Term::{Const as C, Var as V},
};
use s_graph::mutable_fst::UpdateLookup;
use s_graph::storage::ReservedBlockStorage;
use s_graph::{FreezeOrder, PackedFst, Triple};
use std::{hint::black_box, ops::ControlFlow, time::Instant};

fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn timed<T>(action: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = action();
    (value, start.elapsed().as_secs_f64() * 1e6)
}
fn visit<P: Search>(plan: &P, workspace: &mut P::Workspace) -> usize {
    let mut n = 0;
    let _: ControlFlow<()> = plan.visit(workspace, |row| {
        black_box(row);
        n += 1;
        ControlFlow::Continue(())
    });
    n
}
fn run<S: ReservedBlockStorage>(
    backend: &str,
    shape: &str,
    n: u32,
    edits: usize,
    trials: usize,
    input_layout: &str,
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
    let pattern = Bgp::new(2).triple([V(0), C(3), V(1)]);
    let all = Bgp::new(3).triple([V(0), V(1), V(2)]);
    for trial in 0..trials {
        for mode in if trial % 2 == 0 {
            ["compact", "blocks", "recursive", "dfs", "levels"]
        } else {
            ["levels", "dfs", "recursive", "blocks", "compact"]
        } {
            let packed = PackedFst::<S>::try_build(initial.clone(), [n as u64, 4, n as u64], 4096)?;
            let base_nodes = packed.node_count();
            let (mut graph, base_bytes, initial_sealed_nodes, into_mutable_us) =
                if input_layout == "packed" {
                    let base_bytes = packed.mem_size_bytes();
                    let (graph, elapsed) =
                        timed(|| packed.into_mutable_with_lookup(UpdateLookup::Memoized));
                    (graph, base_bytes, 0, elapsed)
                } else {
                    drop(packed);
                    // Construct the same relation entirely in the edited segment.
                    // All setup, including this first sealing, is outside timing.
                    let mut graph =
                        PackedFst::<S>::try_build(Vec::new(), [n as u64, 4, n as u64], 4096)?
                            .into_mutable();
                    graph.insert_batch(&mut initial.clone())?;
                    let frozen = graph.freeze_in_place_with_order(if input_layout == "dfs" {
                        FreezeOrder::Dfs
                    } else {
                        FreezeOrder::Levels
                    })?;
                    let base_bytes = frozen.mem_size_bytes();
                    let sealed_nodes = frozen.sealed_node_count();
                    let (graph, elapsed) =
                        timed(|| frozen.into_mutable_with_lookup(UpdateLookup::Memoized));
                    (graph, base_bytes, sealed_nodes, elapsed)
                };
            let mut updates: Vec<_> = (0..edits as u32)
                .map(|s| (s, 3, mix(s.wrapping_add(456)) % n))
                .collect();
            let (changed, update_us) = timed(|| graph.insert_batch(&mut updates));
            assert_eq!(changed?, edits as u64);
            let live_nodes = graph.edited_node_count();
            let usage = graph.memory_usage();
            let before_bytes = usage.total_ram_bytes();
            let mut block_stats = None;
            let (freeze_us, after_bytes, scan_us, search_us, full_search_us, unchanged_freeze_us) =
                if matches!(mode, "compact" | "blocks" | "recursive") {
                    let (packed, freeze_us) = timed(|| {
                        if mode == "blocks" || mode == "recursive" {
                            let (packed, stats) = if mode == "recursive" {
                                graph.compact_recursive().unwrap()
                            } else {
                                graph.compact_blocks().unwrap()
                            };
                            block_stats = Some(stats);
                            packed
                        } else {
                            graph.freeze().unwrap()
                        }
                    });
                    let after_bytes = packed.mem_size_bytes();
                    assert_eq!(
                        packed.count_matching([None; 3]),
                        2 * n as u64 + edits as u64
                    );
                    for &t in &updates {
                        assert!(packed.contains(t.0, t.1, t.2));
                    }
                    let (_, scan_us) = timed(|| {
                        assert_eq!(packed.scan([None; 3]).count(), 2 * n as usize + edits)
                    });
                    let plan = packed.search(&pattern)?;
                    let mut ws = Default::default();
                    assert_eq!(visit(&plan, &mut ws), edits);
                    let (_, search_us) = timed(|| assert_eq!(visit(&plan, &mut ws), edits));
                    drop(plan);
                    let plan = packed.search(&all)?;
                    let mut ws = Default::default();
                    assert_eq!(visit(&plan, &mut ws), 2 * n as usize + edits);
                    let (_, full_search_us) =
                        timed(|| assert_eq!(visit(&plan, &mut ws), 2 * n as usize + edits));
                    drop(plan);
                    let graph = packed.into_mutable_with_lookup(UpdateLookup::Memoized);
                    let (_, unchanged_freeze_us) = timed(|| {
                        if mode == "recursive" {
                            graph.compact_recursive().unwrap().0
                        } else if mode == "blocks" {
                            graph.compact_blocks().unwrap().0
                        } else {
                            graph.freeze().unwrap()
                        }
                    });
                    (
                        freeze_us,
                        after_bytes,
                        scan_us,
                        search_us,
                        full_search_us,
                        unchanged_freeze_us,
                    )
                } else {
                    let order = if mode == "dfs" {
                        FreezeOrder::Dfs
                    } else {
                        FreezeOrder::Levels
                    };
                    let (frozen, freeze_us) =
                        timed(|| graph.freeze_in_place_with_order(order).unwrap());
                    assert_eq!(
                        frozen.sealed_node_count(),
                        initial_sealed_nodes + live_nodes
                    );
                    let after_bytes = frozen.mem_size_bytes();
                    assert_eq!(frozen.memory_usage().packed_base_bytes, after_bytes);
                    assert_eq!(
                        frozen.count_matching([None; 3]),
                        2 * n as u64 + edits as u64
                    );
                    for &t in &updates {
                        assert!(frozen.contains(t.0, t.1, t.2));
                    }
                    let (_, scan_us) = timed(|| {
                        assert_eq!(frozen.scan([None; 3]).count(), 2 * n as usize + edits)
                    });
                    let plan = frozen.search(&pattern)?;
                    let mut ws = Default::default();
                    assert_eq!(visit(&plan, &mut ws), edits);
                    let (_, search_us) = timed(|| assert_eq!(visit(&plan, &mut ws), edits));
                    drop(plan);
                    let plan = frozen.search(&all)?;
                    let mut ws = Default::default();
                    assert_eq!(visit(&plan, &mut ws), 2 * n as usize + edits);
                    let (_, full_search_us) =
                        timed(|| assert_eq!(visit(&plan, &mut ws), 2 * n as usize + edits));
                    drop(plan);
                    let graph = frozen.into_mutable_with_lookup(UpdateLookup::Memoized);
                    let (frozen, unchanged_freeze_us) =
                        timed(|| graph.freeze_in_place_with_order(order).unwrap());
                    assert_eq!(
                        frozen.sealed_node_count(),
                        initial_sealed_nodes + live_nodes
                    );
                    (
                        freeze_us,
                        after_bytes,
                        scan_us,
                        search_us,
                        full_search_us,
                        unchanged_freeze_us,
                    )
                };
            let stats_json = block_stats.map_or_else(|| "null".to_owned(), |s| format!(
                "{{\"reachable_states\":{},\"source_blocks\":{},\"released_blocks\":{},\"relocation_bytes\":{},\"rank_bytes\":{},\"workspace_bytes\":{},\"input_uncompressed_block_bytes\":{},\"peak_uncompressed_block_bytes\":{},\"output_record_bytes\":{},\"mark_ns\":{},\"planning_ns\":{},\"copy_ns\":{}}}",
                s.reachable_states, s.source_blocks, s.released_blocks, s.relocation_bytes, s.rank_bytes, s.workspace_bytes,
                s.input_uncompressed_block_bytes, s.peak_uncompressed_block_bytes, s.output_record_bytes,
                s.mark_ns, s.planning_ns, s.copy_ns));
            println!(
                "{{\"backend\":\"{backend}\",\"shape\":\"{shape}\",\"input_layout\":\"{input_layout}\",\"vertices\":{n},\"edits\":{edits},\"trial\":{trial},\"mode\":\"{mode}\",\"base_nodes\":{base_nodes},\"edited_nodes\":{live_nodes},\"base_bytes\":{base_bytes},\"before_freeze_bytes\":{before_bytes},\"after_freeze_bytes\":{after_bytes},\"relocation_bytes\":{},\"into_mutable_us\":{into_mutable_us},\"cold_update_us\":{update_us},\"freeze_us\":{freeze_us},\"scan_us\":{scan_us},\"search_us\":{search_us},\"full_search_us\":{full_search_us},\"unchanged_freeze_us\":{unchanged_freeze_us},\"block_stats\":{stats_json}}}",
                if let Some(stats) = block_stats {
                    stats.relocation_bytes
                } else if mode == "compact" {
                    0
                } else {
                    usage.arena_slots * 8
                }
            );
        }
    }
    Ok(())
}
fn main() -> s_graph::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(4096, |s| s.parse::<u32>().unwrap());
    let edits = args.get(2).map_or(32, |s| s.parse::<usize>().unwrap());
    let trials = args.get(3).map_or(3, |s| s.parse::<usize>().unwrap());
    let backend = args.get(4).map_or("all", String::as_str);
    let input_layout = args.get(5).map_or("packed", String::as_str);
    assert!(n >= 4 && n.is_power_of_two() && edits > 0 && edits <= n as usize && trials > 0);
    assert!(["packed", "dfs", "bfs"].contains(&input_layout));
    for shape in ["functional", "mixed"] {
        if backend == "all" || backend == "heap" {
            run::<s_graph::storage::HeapStorage>("heap", shape, n, edits, trials, input_layout)?;
        }
        #[cfg(feature = "file")]
        if backend == "all" || backend == "file" {
            run::<s_graph::storage_file::FileStorage>(
                "file",
                shape,
                n,
                edits,
                trials,
                input_layout,
            )?;
        }
        #[cfg(feature = "compressed")]
        if backend == "all" || backend == "compressed" {
            run::<s_graph::storage_compressed::CompressedStorage>(
                "compressed",
                shape,
                n,
                edits,
                trials,
                input_layout,
            )?;
        }
    }
    Ok(())
}
