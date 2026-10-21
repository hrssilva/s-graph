//! Compare indexed suffix matching with exact runtime scans.
//! cargo run --release --all-features --example fst_metadata_perf -- 4096 32 3
use s_graph::mutable_fst::UpdateLookup;
use s_graph::{BlockStorage, PackedFst, Triple};
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
fn timed<T>(action: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = action();
    (result, start.elapsed().as_secs_f64() * 1e6)
}
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}
#[derive(Default)]
struct ResultRow {
    cold: Vec<f64>,
    insert: Vec<f64>,
    undo: Vec<f64>,
    delete: Vec<f64>,
    restore: Vec<f64>,
    total: usize,
    base_index: usize,
    edit_index: usize,
    bookkeeping: usize,
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
    let mut updates: Vec<Triple> = (0..edits as u32)
        .map(|s| (s, 3, mix(s.wrapping_add(456)) % n))
        .collect();
    let mut original: Vec<Triple> = (0..edits as u32)
        .map(|s| (s, 0, object(s, 0, n, shape)))
        .collect();
    let modes = [
        UpdateLookup::Indexed,
        UpdateLookup::ScanBase,
        UpdateLookup::ScanAll,
    ];
    let mut results: [ResultRow; 3] = std::array::from_fn(|_| ResultRow::default());
    for pass in 0..2 {
        for i in if pass == 0 {
            vec![0, 1, 2]
        } else {
            vec![2, 1, 0]
        } {
            let mut graph =
                PackedFst::<S>::try_build(initial.clone(), [n as u64, 4, n as u64], 4096)?
                    .into_mutable_with_lookup(modes[i]);
            let r = &mut results[i];
            // First real insertion includes lazy index construction, if any.
            let t = updates[0];
            let (changed, us) = timed(|| graph.insert(t.0, t.1, t.2));
            assert!(changed?);
            r.cold.push(us);
            assert!(graph.remove(t.0, t.1, t.2)?);
            assert_eq!(graph.edited_node_count(), 0);
            for _ in 0..trials {
                let (count, us) = timed(|| graph.insert_batch(&mut updates));
                assert_eq!(count?, edits as u64);
                r.insert.push(us / edits as f64);
                let memory = graph.memory_usage();
                r.total = memory.total_ram_bytes();
                r.base_index = memory.base_register_bytes;
                r.edit_index = memory.edit_register_bytes;
                r.bookkeeping = memory.edit_arena_bytes + memory.free_list_bytes;
                if modes[i] != UpdateLookup::Indexed {
                    assert_eq!(memory.base_register_bytes, 0);
                }
                if modes[i] == UpdateLookup::ScanAll {
                    assert_eq!(memory.edit_register_bytes, 0);
                }
                let (count, us) = timed(|| graph.remove_batch(&mut updates));
                assert_eq!(count?, edits as u64);
                r.undo.push(us / edits as f64);
                assert_eq!(graph.edited_node_count(), 0);
                let (count, us) = timed(|| graph.remove_batch(&mut original));
                assert_eq!(count?, edits as u64);
                r.delete.push(us / edits as f64);
                let (count, us) = timed(|| graph.insert_batch(&mut original));
                assert_eq!(count?, edits as u64);
                r.restore.push(us / edits as f64);
                assert_eq!(
                    graph.edited_node_count(),
                    0,
                    "inverse edits lost canonical packed identity"
                );
            }
            assert_eq!(graph.count_matching([None; 3]), initial.len() as u64);
            assert_eq!(
                graph.scan([None; 3]).collect::<BTreeSet<_>>(),
                initial.iter().copied().collect()
            );
            graph.compact()?;
            graph.insert_batch(&mut updates)?; // selected mode survives compaction
            assert_eq!(
                graph.memory_usage().base_register_bytes == 0,
                modes[i] != UpdateLookup::Indexed
            );
        }
    }
    for (mode, r) in modes.into_iter().zip(results) {
        println!(
            "{{\"backend\":\"{backend}\",\"shape\":\"{shape}\",\"vertices\":{n},\"edits\":{edits},\"mode\":\"{mode:?}\",\"cold_insert_us\":{},\"insert_us\":{},\"undo_insert_us\":{},\"delete_base_us\":{},\"restore_base_us\":{},\"total_ram_bytes\":{},\"base_index_bytes\":{},\"edit_index_bytes\":{},\"bookkeeping_bytes\":{},\"insert_samples_us\":{:?},\"undo_samples_us\":{:?},\"delete_samples_us\":{:?},\"restore_samples_us\":{:?},\"cold_samples_us\":{:?}}}",
            median(&r.cold),
            median(&r.insert),
            median(&r.undo),
            median(&r.delete),
            median(&r.restore),
            r.total,
            r.base_index,
            r.edit_index,
            r.bookkeeping,
            r.insert,
            r.undo,
            r.delete,
            r.restore,
            r.cold
        );
    }
    Ok(())
}
fn main() -> s_graph::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let n = a.get(1).map_or(4096, |s| s.parse::<u32>().unwrap());
    let edits = a.get(2).map_or(32, |s| s.parse::<usize>().unwrap());
    let trials = a.get(3).map_or(3, |s| s.parse::<usize>().unwrap());
    let backend = a.get(4).map_or("all", String::as_str);
    assert!(n >= 4 && n.is_power_of_two() && edits > 0 && edits <= n as usize && trials > 0);
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
