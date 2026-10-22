#![cfg(feature = "heap")]
use s_graph::automata::{
    Bgp, Language, Mask, Query, Schedule, Search,
    Term::{Const as C, Var as V},
};
use s_graph::path_fst::PackedFst;
use s_graph::{BlockStorage, EditableFst, MutableFst, Triple};
use std::collections::{BTreeSet, HashSet};
use std::ops::ControlFlow;

fn rows<P: Search>(plan: P) -> Vec<Vec<Option<u32>>> {
    let mut rows = Vec::new();
    let _: ControlFlow<()> = plan.visit(&mut Default::default(), |row| {
        rows.push(row.to_options());
        ControlFlow::Continue(())
    });
    rows.sort_unstable();
    rows
}
fn check<S: BlockStorage>(graph: &EditableFst<S>, expected: &BTreeSet<Triple>) {
    assert_eq!(graph.scan([None; 3]).collect::<BTreeSet<_>>(), *expected);
    assert_eq!(graph.count_matching([None; 3]), expected.len() as u64);
    for s in 0..32 {
        let bounds = [Some(s), None, Some(s ^ 1)];
        let n = expected
            .iter()
            .filter(|&&(a, _, b)| a == s && b == s ^ 1)
            .count();
        assert_eq!(graph.count_matching(bounds), n as u64);
    }
    for &(s, p, o) in expected {
        assert!(graph.contains(s, p, o));
    }
    let pattern = Bgp::new(3).triple([V(0), V(1), V(2)]);
    let oracle: Vec<_> = expected
        .iter()
        .map(|&(s, p, o)| vec![Some(s), Some(p), Some(o)])
        .collect();
    for schedule in [
        Schedule::Adaptive,
        Schedule::Nested,
        Schedule::Levels,
        Schedule::Frontier,
        Schedule::Prefix(1),
    ] {
        assert_eq!(
            rows(graph.search(pattern.clone().schedule(schedule)).unwrap()),
            oracle
        );
    }
    let scan = rows(graph.search(Language(Mask(u64::MAX))).unwrap());
    assert_eq!(scan, oracle);
    // Compare the current reachable machine, before compaction, with an
    // independent full rebuild. Equal counts prove no redundant live states.
    use s_graph::automata::BitmapSource;
    let source = graph.source();
    let mut pending = HashSet::from([source.root().unwrap()]);
    let mut live = 0;
    for level in 0..source.height() {
        let mut next = HashSet::new();
        live += pending.len();
        for cursor in pending {
            let word = source.word(cursor);
            let mut todo = word;
            if level + 1 < source.height() {
                while todo != 0 {
                    let slot = todo.trailing_zeros();
                    todo &= todo - 1;
                    next.insert(source.follow(cursor, word, slot));
                }
            }
        }
        pending = next;
    }
    let rebuilt = s_graph::Fst::build(
        expected.iter().copied().collect(),
        graph.dimension_sizes(),
        512,
    );
    assert_eq!(
        live as u64,
        rebuilt.node_count(),
        "live FST lost minimality before compaction"
    );
}

fn exercise<S: s_graph::storage::ReclaimingBlockStorage>() {
    for lookup in [
        s_graph::mutable_fst::UpdateLookup::Sampled(1),
        s_graph::mutable_fst::UpdateLookup::Sampled(8),
        s_graph::mutable_fst::UpdateLookup::Memoized,
    ] {
        exercise_lookup::<S>(lookup);
    }
}

fn exercise_lookup<S: s_graph::storage::ReclaimingBlockStorage>(
    lookup: s_graph::mutable_fst::UpdateLookup,
) {
    let initial: Vec<_> = (0..32)
        .flat_map(|s| (0..32).map(move |o| (s, 0, o)))
        .collect();
    let packed = PackedFst::<S>::try_build(initial.clone(), [32, 4, 32], 512).unwrap();
    let bytes = packed.serialize();
    let heap = s_graph::Fst::deserialize(&bytes).unwrap();
    assert_eq!(bytes, heap.serialize());
    assert_eq!(packed.node_count(), heap.node_count());
    assert_eq!(
        packed.scan([None; 3]).collect::<BTreeSet<_>>(),
        initial.iter().copied().collect()
    );
    let packed_machine = Bgp::new(2).triple([V(0), C(0), V(1)]);
    assert_eq!(rows(packed.search(&packed_machine).unwrap()).len(), 1024);
    let cross_loaded = PackedFst::<S>::deserialize(&heap.serialize()).unwrap();
    assert_eq!(
        rows(cross_loaded.search(&packed_machine).unwrap()).len(),
        1024
    );
    let mut graph = packed.into_mutable_with_lookup(lookup);
    let mut expected: BTreeSet<_> = initial.into_iter().collect();
    assert_eq!(graph.edited_node_count(), 0);
    assert!(graph.remove(0, 0, 0).unwrap());
    expected.remove(&(0, 0, 0));
    assert!(
        graph.contains(1, 0, 0),
        "a different prefix sharing the old suffix changed"
    );
    assert!(graph.contains(0, 0, 1));
    assert!(!graph.remove(0, 0, 0).unwrap());
    assert!(!graph.remove(u32::MAX, 0, 0).unwrap());
    assert!(graph.insert(0, 0, 0).unwrap());
    expected.insert((0, 0, 0));
    assert_eq!(
        graph.edited_node_count(),
        0,
        "inverse edit did not merge back into the packed base"
    );
    assert!(!graph.insert(0, 0, 0).unwrap());
    check(&graph, &expected);
    let before = graph.scan([None; 3]).collect::<BTreeSet<_>>();
    assert!(graph.insert_batch(&mut [(0, 1, 0), (32, 1, 0)]).is_err());
    assert_eq!(graph.scan([None; 3]).collect::<BTreeSet<_>>(), before);
    assert_eq!(
        graph
            .insert_batch(&mut [(0, 1, 0), (0, 1, 0), (31, 3, 31)])
            .unwrap(),
        2
    );
    expected.extend([(0, 1, 0), (31, 3, 31)]);
    let machine = Bgp::new(2)
        .triple([C(0), C(0), V(0)])
        .optional(Bgp::new(2).triple([V(0), C(1), V(1)]));
    let result = rows(graph.search(&machine).unwrap());
    assert!(result.contains(&vec![Some(0), Some(0)]));
    assert!(result.contains(&vec![Some(1), None]));
    let mut seed = 7u64;
    for i in 0..600 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t = (
            (seed >> 12) as u32 % 32,
            (seed >> 32) as u32 % 4,
            (seed >> 46) as u32 % 32,
        );
        let changed = if seed & 1 == 0 {
            graph.insert(t.0, t.1, t.2).unwrap()
        } else {
            graph.remove(t.0, t.1, t.2).unwrap()
        };
        let wanted = if seed & 1 == 0 {
            expected.insert(t)
        } else {
            expected.remove(&t)
        };
        assert_eq!(changed, wanted);
        if i % 100 == 0 {
            check(&graph, &expected);
        }
    }
    check(&graph, &expected);
    let path = s_graph::automata::PathExpr::label(0).compile().unwrap();
    let relation = s_graph::automata::PathBindings::new(
        &graph,
        &path,
        vec![0],
        C(0),
        V(0),
        1,
        s_graph::traversal::HopRange::new(1, 1).unwrap(),
    )
    .unwrap();
    let mut wanted: Vec<_> = expected
        .iter()
        .filter(|&&(s, p, _)| s == 0 && p == 0)
        .map(|&(_, _, o)| vec![Some(o)])
        .collect();
    wanted.sort_unstable();
    assert_eq!(rows(relation), wanted);
    use s_graph::analytics::{CompressedEdges, EdgeSource, PageRank};
    let edges = CompressedEdges::new(graph.source(), 32, None).unwrap();
    let mut scanned = BTreeSet::new();
    edges.edges(&mut Default::default(), |e| {
        scanned.insert((e.source, e.predicate, e.target));
    });
    assert_eq!(scanned, expected);
    let mut rank = PageRank::default();
    rank.run(&edges, 0.85, 1e-12, 100);
    let oracle_rank = s_graph::analytics::reference_pagerank(
        &expected.iter().copied().collect::<Vec<_>>(),
        32,
        0.85,
        1e-12,
        100,
    );
    for (a, b) in rank.rank.iter().zip(oracle_rank.values) {
        assert!((a - b).abs() < 1e-12);
    }
    let snapshot = graph.serialize().unwrap();
    let restored = EditableFst::<S>::deserialize(&snapshot).unwrap();
    check(&restored, &expected);
    let heap = s_graph::Fst::deserialize(&snapshot).unwrap();
    assert_eq!(heap.scan([None; 3]).collect::<BTreeSet<_>>(), expected);
    let rebuilt = s_graph::Fst::build(expected.iter().copied().collect(), [32, 4, 32], 512);
    assert_eq!(
        heap.node_count(),
        rebuilt.node_count(),
        "freeze failed to minimize physical states"
    );
    graph.compact().unwrap();
    assert_eq!(graph.edited_node_count(), 0);
    assert_eq!(graph.update_lookup_stats(), Default::default());
    check(&graph, &expected);
    let mut all: Vec<_> = expected.iter().copied().collect();
    assert_eq!(graph.remove_batch(&mut all).unwrap(), expected.len() as u64);
    assert_eq!(graph.edited_node_count(), 0);
    assert_eq!(graph.count_matching([None; 3]), 0);
    let empty = graph.freeze().unwrap();
    assert_eq!(empty.node_count(), 1);
    assert!(empty.scan([None; 3]).next().is_none());
    assert!(PackedFst::<S>::deserialize(&empty.serialize()).is_ok());
}

#[test]
fn heap_fst_updates_and_queries() {
    exercise::<s_graph::storage::HeapStorage>();
}
#[cfg(feature = "file")]
#[test]
fn file_fst_updates_and_queries() {
    exercise::<s_graph::storage_file::FileStorage>();
}
#[cfg(feature = "compressed")]
#[test]
fn compressed_fst_updates_and_queries() {
    exercise::<s_graph::storage_compressed::CompressedStorage>();
}

#[cfg(not(feature = "experimental"))]
use s_graph::storage::ReclaimingBlockStorage as CompactionStorage;
#[cfg(feature = "experimental")]
use s_graph::storage::ReservedBlockStorage as CompactionStorage;

fn in_place<S: CompactionStorage>() {
    use s_graph::FreezeOrder;
    use s_graph::mutable_fst::UpdateLookup;
    let methods: &[u8] = if cfg!(feature = "experimental") {
        &[0, 1, 2, 3]
    } else {
        &[3]
    };
    for &method in methods {
        for order in [FreezeOrder::Dfs, FreezeOrder::Levels] {
            for lookup in [
                UpdateLookup::Sampled(1),
                UpdateLookup::Sampled(8),
                UpdateLookup::ScanBase,
                UpdateLookup::ScanAll,
                UpdateLookup::Memoized,
            ] {
                let initial: Vec<_> = (0..32)
                    .flat_map(|s| (0..32).map(move |o| (s, 0, o)))
                    .collect();
                let mut expected: BTreeSet<_> = initial.iter().copied().collect();
                let mut graph = PackedFst::<S>::build(initial, [32, 4, 32], 512)
                    .into_mutable_with_lookup(lookup);
                for cycle in 0..4 {
                    for i in 0..32 {
                        let added = (i, 1 + cycle % 3, (i * 7 + cycle) % 32);
                        assert_eq!(
                            graph.insert(added.0, added.1, added.2).unwrap(),
                            expected.insert(added)
                        );
                        let removed = (i, cycle % 4, (i * 11 + cycle) % 32);
                        assert_eq!(
                            graph.remove(removed.0, removed.1, removed.2).unwrap(),
                            expected.remove(&removed)
                        );
                    }
                    check(&graph, &expected);
                    let frozen = graph.freeze_in_place_with_order(order).unwrap();
                    check(&frozen, &expected);
                    let memory = frozen.memory_usage();
                    assert_eq!(
                        memory.edit_arena_bytes
                            + memory.edit_payload_bytes
                            + memory.edit_register_bytes
                            + memory.base_register_bytes
                            + memory.memo_register_bytes
                            + memory.free_list_bytes,
                        0
                    );
                    assert_eq!(frozen.mem_size_bytes(), memory.packed_base_bytes);
                    assert!(frozen.sealed_node_count() > 0);
                    let snapshot = frozen.serialize().unwrap();
                    let loaded = s_graph::Fst::deserialize(&snapshot).unwrap();
                    assert_eq!(loaded.scan([None; 3]).collect::<BTreeSet<_>>(), expected);
                    graph = frozen.into_mutable_with_lookup(lookup);
                    assert_eq!(graph.update_lookup_stats(), Default::default());
                }
                let frozen = graph.freeze_in_place().unwrap();
                let oracle = frozen.to_fst::<s_graph::storage::HeapStorage>().unwrap();
                let (packed, stats) = match method {
                    #[cfg(feature = "experimental")]
                    0 => frozen.compact_blocks().unwrap(),
                    #[cfg(feature = "experimental")]
                    1 => frozen.compact_recursive().unwrap(),
                    #[cfg(feature = "experimental")]
                    2 => frozen.compact_depths().unwrap(),
                    _ => frozen.compact_streaming().unwrap(),
                };
                assert_eq!(packed.scan([None; 3]).collect::<BTreeSet<_>>(), expected);
                assert_eq!(packed.visited_count(), oracle.visited_count());
                assert_eq!(packed.node_count(), oracle.node_count());
                assert_eq!(stats.released_blocks, stats.source_blocks);
                assert_eq!(stats.reachable_states as u64, packed.node_count());
                let snapshot = packed.serialize();
                let loaded = s_graph::Fst::deserialize(&snapshot).unwrap();
                assert_eq!(loaded.scan([None; 3]).collect::<BTreeSet<_>>(), expected);
                if method != 0 {
                    check(&packed.into_mutable(), &expected);
                }
            }
        }
    }
    for domains in [[1, 1, 1], [1 << 32, 1, 1 << 32]] {
        let graph = PackedFst::<S>::build(Vec::new(), domains, 512).into_mutable();
        let (packed, stats) = graph.compact_streaming().unwrap();
        assert_eq!(packed.node_count(), 1);
        assert_eq!(packed.count_matching([None; 3]), 0);
        assert_eq!(stats.released_blocks, stats.source_blocks);
        let t = ((domains[0] - 1) as u32, 0, (domains[2] - 1) as u32);
        let mut graph = packed.into_mutable();
        graph.insert(t.0, t.1, t.2).unwrap();
        let (packed, stats) = graph.compact_streaming().unwrap();
        assert_eq!(stats.released_blocks, stats.source_blocks);
        assert_eq!(packed.scan([None; 3]).collect::<Vec<_>>(), vec![t]);
        let loaded = s_graph::Fst::deserialize(&packed.serialize()).unwrap();
        assert!(loaded.contains(t.0, t.1, t.2));
        let mut graph = packed.into_mutable();
        graph.remove(t.0, t.1, t.2).unwrap();
        assert_eq!(
            graph
                .compact_streaming()
                .unwrap()
                .0
                .count_matching([None; 3]),
            0
        );
    }
    // A full 64-child edited root has a 521-byte absolute-link record. Its
    // sealed block budget must accommodate this even with 512-byte base blocks.
    let initial: Vec<_> = (0..4)
        .flat_map(|s| (0..4).flat_map(move |p| (0..4).map(move |o| (s * 4, p * 4, o * 4))))
        .collect();
    let mut expected: BTreeSet<_> = initial.iter().copied().collect();
    let mut graph = PackedFst::<S>::build(initial, [16, 16, 16], 512).into_mutable();
    graph.insert(1, 1, 1).unwrap();
    expected.insert((1, 1, 1));
    let frozen = graph.freeze_in_place().unwrap();
    assert_eq!(frozen.scan([None; 3]).collect::<BTreeSet<_>>(), expected);
    let mut graph = frozen.into_mutable();
    graph.remove(1, 1, 1).unwrap();
    assert_eq!(
        graph.edited_node_count(),
        0,
        "inverse edit must reuse original packed root"
    );
    let frozen = graph.freeze_in_place().unwrap();
    assert!(!frozen.contains(1, 1, 1));
}

#[test]
fn freeze_in_place_across_storage_modes_and_repeated_batches() {
    in_place::<s_graph::storage::HeapStorage>();
    #[cfg(feature = "file")]
    in_place::<s_graph::storage_file::FileStorage>();
    #[cfg(feature = "compressed")]
    in_place::<s_graph::storage_compressed::CompressedStorage>();
}

#[test]
fn full_width_zero_bit_dimensions_and_reused_slots() {
    let mut graph = MutableFst::build(vec![], [1u64 << 32, 1, 1u64 << 32], 512);
    for _ in 0..100 {
        assert!(graph.insert(u32::MAX, 0, u32::MAX).unwrap());
        assert!(graph.contains(u32::MAX, 0, u32::MAX));
        assert_eq!(graph.count_matching([None; 3]), 1);
        assert!(graph.remove(u32::MAX, 0, u32::MAX).unwrap());
        assert_eq!(graph.edited_node_count(), 0);
    }
    let mut graph = MutableFst::build(vec![], [1, 1, 1], 512);
    assert!(graph.insert(0, 0, 0).unwrap());
    assert_eq!(graph.scan([None; 3]).collect::<Vec<_>>(), vec![(0, 0, 0)]);
    assert!(graph.remove(0, 0, 0).unwrap());
    assert_eq!(graph.freeze().unwrap().count_matching([None; 3]), 0);
}

fn legacy<S: BlockStorage>() {
    let mut tree =
        s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric::<S>::build_dynamic_empty(
            [32, 4, 32],
            1024,
        );
    let machine = Bgp::new(2).triple([V(0), C(0), V(1)]);
    assert!(rows(tree.search(&machine).unwrap()).is_empty());
    tree.insert(1, 0, 2).unwrap();
    assert_eq!(
        rows(tree.search(&machine).unwrap()),
        vec![vec![Some(1), Some(2)]]
    );
    tree.remove(1, 0, 2).unwrap();
    assert!(rows(tree.search(&machine).unwrap()).is_empty());
    let tree = s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric::<S>::build_dynamic(
        vec![(0, 0, 0)],
        1024,
    );
    assert_eq!(
        rows(tree.search(Language(Mask(1))).unwrap()),
        vec![vec![Some(0); 3]]
    );
    // Exercise far-child span navigation, non-default dimension order and
    // full-width IDs through the shared automaton source on every backend.
    let mut tree = s_graph::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric::<S>::build_dynamic_empty_with_order(
        [1u64 << 32, 3, 1u64 << 32], [2, 0, 1], 1024);
    let mut triples: Vec<_> = (0u32..100)
        .map(|i| {
            (
                i.wrapping_mul(0x9e3779b9),
                i % 3,
                i.wrapping_mul(0x7feb352d),
            )
        })
        .collect();
    tree.insert_batch(&mut triples).unwrap();
    let mut expected: Vec<_> = triples
        .iter()
        .map(|&(s, p, o)| vec![Some(s), Some(p), Some(o)])
        .collect();
    expected.sort_unstable();
    for schedule in [
        Schedule::Adaptive,
        Schedule::Nested,
        Schedule::Levels,
        Schedule::Frontier,
    ] {
        let machine = Bgp::new(3).triple([V(0), V(1), V(2)]).schedule(schedule);
        assert_eq!(rows(tree.search(machine).unwrap()), expected);
    }
}
#[test]
fn mutable_tree_automata() {
    legacy::<s_graph::storage::HeapStorage>();
    #[cfg(feature = "file")]
    legacy::<s_graph::storage_file::FileStorage>();
    #[cfg(feature = "compressed")]
    legacy::<s_graph::storage_compressed::CompressedStorage>();
}
