//! Cross-edit memoization: first edits, repeated edits, and fresh regions.
//! cargo run --release --all-features --example fst_lazy_perf -- 65536 64 3 heap
use s_graph::mutable_fst::UpdateLookup;
use s_graph::{BlockStorage, EditableFst, PackedFst, Triple};
use std::{collections::BTreeSet, time::Instant};

fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn object(s: u32, p: u32, n: u32, shape: &str) -> u32 {
    if shape == "functional" {
        s ^ (p + 1)
    } else {
        mix(s ^ (p * 0x1234)) % n
    }
}
struct Context<'a> {
    backend: &'a str,
    shape: &'a str,
    mode: UpdateLookup,
    vertices: u32,
    pass: usize,
}
fn phase<S: BlockStorage>(
    graph: &mut EditableFst<S>,
    context: &Context<'_>,
    label: &str,
    trial: usize,
    triples: &mut [Triple],
    insert: bool,
) -> s_graph::Result<()> {
    let before = graph.update_lookup_stats();
    let start = Instant::now();
    let changed = if insert {
        graph.insert_batch(triples)?
    } else {
        graph.remove_batch(triples)?
    };
    let us = start.elapsed().as_secs_f64() * 1e6 / triples.len() as f64;
    assert_eq!(changed as usize, triples.len());
    let after = graph.update_lookup_stats();
    let memory = graph.memory_usage();
    assert_eq!(memory.total_ram_bytes(), graph.mem_size_bytes());
    if context.mode != UpdateLookup::Indexed {
        assert_eq!(memory.base_register_bytes, 0);
    }
    if context.mode != UpdateLookup::Memoized {
        assert_eq!(memory.memo_register_bytes, 0);
    }
    for &(s, p, o) in triples.iter() {
        assert_eq!(graph.contains(s, p, o), insert);
    }
    let Context {
        backend,
        shape,
        mode,
        vertices,
        pass,
    } = context;
    println!(
        "{{\"backend\":\"{backend}\",\"shape\":\"{shape}\",\"mode\":\"{mode:?}\",\"vertices\":{vertices},\"pass\":{pass},\"phase\":\"{label}\",\"trial\":{trial},\"edits\":{changed},\"us_per_edit\":{us},\"base_scans\":{},\"memo_hits\":{},\"memo_entries\":{},\"cached_absences\":{},\"total_ram_bytes\":{},\"packed_base_bytes\":{},\"base_index_bytes\":{},\"memo_bytes\":{},\"edit_index_bytes\":{},\"bookkeeping_bytes\":{}}}",
        after.base_scans - before.base_scans,
        after.memo_hits - before.memo_hits,
        after.memo_entries,
        after.cached_absences,
        memory.total_ram_bytes(),
        memory.packed_base_bytes,
        memory.base_register_bytes,
        memory.memo_register_bytes,
        memory.edit_register_bytes,
        memory.edit_arena_bytes + memory.free_list_bytes,
    );
    Ok(())
}
fn run<S: BlockStorage>(
    backend: &str,
    shape: &str,
    n: u32,
    edits: usize,
    trials: usize,
) -> s_graph::Result<()> {
    let initial: Vec<_> = (0..n)
        .flat_map(|s| (0..2).map(move |p| (s, p, object(s, p, n, shape))))
        .collect();
    let modes = [
        UpdateLookup::Indexed,
        UpdateLookup::ScanAll,
        UpdateLookup::Memoized,
    ];
    for pass in 0..2 {
        for i in if pass == 0 {
            vec![0, 1, 2]
        } else {
            vec![2, 1, 0]
        } {
            let mode = modes[i];
            let mut graph =
                PackedFst::<S>::try_build(initial.clone(), [n as u64, 4, n as u64], 4096)?
                    .into_mutable_with_lookup(mode);
            let context = Context {
                backend,
                shape,
                mode,
                vertices: n,
                pass,
            };
            let mut updates: Vec<_> = (0..edits as u32)
                .map(|s| (s, 3, mix(s.wrapping_add(456)) % n))
                .collect();
            for trial in 0..trials {
                // Trial zero includes any complete index preparation. Later
                // trials reuse the same signatures after undoing the batch.
                phase(&mut graph, &context, "insert", trial, &mut updates, true)?;
                phase(&mut graph, &context, "undo", trial, &mut updates, false)?;
                assert_eq!(graph.edited_node_count(), 0);
            }
            // Different subjects and endpoints: a cache cannot answer new
            // signatures merely because the earlier batch is warm.
            let mut fresh: Vec<_> = (0..edits as u32)
                .map(|s| {
                    let subject = n / 2 + s.wrapping_mul(197) % (n / 2);
                    (subject, 3, mix(s.wrapping_add(98765)) % n)
                })
                .collect();
            phase(&mut graph, &context, "fresh_insert", 0, &mut fresh, true)?;
            phase(&mut graph, &context, "fresh_undo", 0, &mut fresh, false)?;
            assert_eq!(graph.edited_node_count(), 0);
            let mut original: Vec<_> = (0..edits as u32)
                .map(|s| {
                    let subject = s.wrapping_mul(197) % n;
                    (subject, 0, object(subject, 0, n, shape))
                })
                .collect();
            for trial in 0..trials {
                phase(
                    &mut graph,
                    &context,
                    "delete_base",
                    trial,
                    &mut original,
                    false,
                )?;
                phase(
                    &mut graph,
                    &context,
                    "restore_base",
                    trial,
                    &mut original,
                    true,
                )?;
                assert_eq!(graph.edited_node_count(), 0);
            }
            assert_eq!(graph.count_matching([None; 3]), initial.len() as u64);
            assert_eq!(
                graph.scan([None; 3]).collect::<BTreeSet<_>>(),
                initial.iter().copied().collect()
            );
            graph.compact()?;
            assert_eq!(graph.update_lookup_stats(), Default::default());
            phase(
                &mut graph,
                &context,
                "after_compact_insert",
                0,
                &mut updates,
                true,
            )?;
            phase(
                &mut graph,
                &context,
                "after_compact_undo",
                0,
                &mut updates,
                false,
            )?;
            assert_eq!(graph.edited_node_count(), 0);
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
    assert!(n >= 4 && n.is_power_of_two() && edits > 0 && edits <= n as usize / 2 && trials >= 2);
    assert!(["heap", "file", "compressed", "all"].contains(&backend));
    for shape in ["functional", "mixed"] {
        if backend == "heap" || backend == "all" {
            run::<s_graph::storage::HeapStorage>("heap", shape, n, edits, trials)?;
        }
        #[cfg(feature = "file")]
        if backend == "file" || backend == "all" {
            run::<s_graph::storage_file::FileStorage>("file", shape, n, edits, trials)?;
        }
        #[cfg(feature = "compressed")]
        if backend == "compressed" || backend == "all" {
            run::<s_graph::storage_compressed::CompressedStorage>(
                "compressed",
                shape,
                n,
                edits,
                trials,
            )?;
        }
    }
    Ok(())
}
