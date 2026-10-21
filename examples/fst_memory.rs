//! Quick retained-memory experiment, with assertions for update/reclamation.
//! cargo run --release --all-features --example fst_memory -- 65536
use s_graph::{BlockStorage, EditableFst, PackedFst, Triple};

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
fn report<S: BlockStorage>(
    graph: &EditableFst<S>,
    backend: &str,
    shape: &str,
    phase: &str,
    added: usize,
    removed: usize,
    expected: u64,
) {
    assert_eq!(graph.count_matching([None; 3]), expected);
    let m = graph.memory_usage();
    assert_eq!(m.total_ram_bytes(), graph.mem_size_bytes());
    println!(
        "{backend},{shape},{phase},{added},{removed},{expected},{},{},{},{},{},{},{},{},{},{}",
        m.total_ram_bytes(),
        m.packed_base_bytes,
        m.base_register_bytes,
        m.edit_payload_bytes,
        m.edit_arena_bytes,
        m.edit_register_bytes,
        m.free_list_bytes,
        m.live_edit_states,
        m.arena_slots,
        m.free_slots
    );
}
fn run<S: BlockStorage>(backend: &str, shape: &str, n: u32) -> s_graph::Result<()> {
    let initial: Vec<_> = (0..n)
        .flat_map(|s| (0..2).map(move |p| (s, p, object(s, p, n, shape))))
        .collect();
    let mut graph =
        PackedFst::<S>::try_build(initial, [n as u64, 4, n as u64], 4096)?.into_mutable();
    let original = 2 * n as u64;
    report(&graph, backend, shape, "before", 0, 0, original);
    assert!(!graph.insert(0, 0, object(0, 0, n, shape))?);
    assert!(!graph.remove(0, 3, 0)?);
    assert_eq!(graph.memory_usage().base_register_bytes, 0);
    graph.prepare_updates();
    report(&graph, backend, shape, "prepared", 0, 0, original);

    let max = n.min(4096) as usize;
    let mut checkpoints: Vec<_> = [1, 16, 256, max]
        .into_iter()
        .filter(|&x| x <= max)
        .collect();
    checkpoints.sort_unstable();
    checkpoints.dedup();
    let mut updates: Vec<Triple> = (0..max as u32)
        .map(|s| (s, 3, mix(s.wrapping_add(456)) % n))
        .collect();
    let mut current = 0;
    for &target in &checkpoints {
        assert_eq!(
            graph.insert_batch(&mut updates[current..target])?,
            (target - current) as u64
        );
        current = target;
        report(
            &graph,
            backend,
            shape,
            "insert",
            current,
            0,
            original + current as u64,
        );
    }
    for target in checkpoints
        .iter()
        .rev()
        .copied()
        .skip(1)
        .chain(std::iter::once(0))
    {
        assert_eq!(
            graph.remove_batch(&mut updates[target..current])?,
            (current - target) as u64
        );
        current = target;
        report(
            &graph,
            backend,
            shape,
            "undo_insert",
            current,
            0,
            original + current as u64,
        );
    }
    assert_eq!(graph.edited_node_count(), 0);
    assert_eq!(graph.memory_usage().edit_payload_bytes, 0);
    // Repeating the same burst must reuse slots rather than accumulate history.
    // Warm this exact removal order once: different orders can have different
    // temporary live-state peaks even when the final graph is identical.
    graph.insert_batch(&mut updates)?;
    graph.remove_batch(&mut updates)?;
    let slots = graph.memory_usage().arena_slots;
    for _ in 0..3 {
        assert_eq!(graph.insert_batch(&mut updates)?, max as u64);
        assert_eq!(graph.remove_batch(&mut updates)?, max as u64);
        assert_eq!(graph.edited_node_count(), 0);
        assert_eq!(graph.memory_usage().arena_slots, slots);
    }
    report(&graph, backend, shape, "churn_x3", 0, 0, original);
    graph.compact()?;
    let m = graph.memory_usage();
    assert_eq!(
        (
            m.base_register_bytes,
            m.edit_arena_bytes,
            m.free_list_bytes,
            m.arena_slots
        ),
        (0, 0, 0, 0)
    );
    report(&graph, backend, shape, "compact_undo", 0, 0, original);

    // Also delete original packed edges, then compact and restore them.
    let mut deleted: Vec<_> = (0..max as u32)
        .map(|s| (s, 0, object(s, 0, n, shape)))
        .collect();
    let mut removed = 0;
    for &target in &checkpoints {
        assert_eq!(
            graph.remove_batch(&mut deleted[removed..target])?,
            (target - removed) as u64
        );
        removed = target;
        report(
            &graph,
            backend,
            shape,
            "delete_base",
            0,
            removed,
            original - removed as u64,
        );
    }
    graph.compact()?;
    report(
        &graph,
        backend,
        shape,
        "compact_delete",
        0,
        removed,
        original - removed as u64,
    );
    assert_eq!(graph.insert_batch(&mut deleted)?, removed as u64);
    report(&graph, backend, shape, "restore_base", 0, 0, original);
    graph.compact()?;
    report(&graph, backend, shape, "compact_restore", 0, 0, original);
    Ok(())
}
fn main() -> s_graph::Result<()> {
    let n = std::env::args()
        .nth(1)
        .map_or(16384, |s| s.parse::<u32>().expect("vertex count"));
    assert!(n >= 4 && n.is_power_of_two());
    println!(
        "backend,shape,phase,added,removed,triples,total_ram_bytes,packed_base_bytes,base_register_bytes,edit_payload_bytes,edit_arena_bytes,edit_register_bytes,free_list_bytes,live_edit_states,arena_slots,free_slots"
    );
    for shape in ["functional", "mixed"] {
        run::<s_graph::storage::HeapStorage>("heap", shape, n)?;
        #[cfg(feature = "file")]
        run::<s_graph::storage_file::FileStorage>("file", shape, n)?;
        #[cfg(feature = "compressed")]
        run::<s_graph::storage_compressed::CompressedStorage>("compressed", shape, n)?;
    }
    Ok(())
}
