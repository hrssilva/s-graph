#![cfg(feature = "heap")]
use s_graph::graph::{JoinQuery, Term, TriplePattern, join_run};
use s_graph::serialize::DeserializeLimits;
use s_graph::{DEFAULT_BLOCK_CAPACITY, Fst, Heap, JoinWorkspace};
use std::ops::ControlFlow;

fn sorted(mut triples: Vec<(u32, u32, u32)>) -> Vec<(u32, u32, u32)> {
    triples.sort_unstable();
    triples.dedup();
    triples
}

#[test]
fn snapshots_preserve_scans_counts_joins_paths_and_storage() {
    for (triples, domains) in [
        (vec![], [1, 1, 1]),
        (vec![(0, 0, 0), (0, 0, 0)], [1, 1, 1]),
        (vec![], [33, 5, 129]),
        (
            (0..33)
                .flat_map(|s| (0..5).map(move |p| (s, p, (s * 3 + p) % 129)))
                .collect(),
            [33, 5, 129],
        ),
        (
            vec![(u32::MAX, 0, 1), (1, 1, 2), (2, 0, u32::MAX), (0, 0, 0)],
            [1u64 << 32, 2, 1u64 << 32],
        ),
        (
            (0..128)
                .flat_map(|s| (0..64).map(move |o| (s, 0, o)))
                .collect(),
            [128, 1, 64],
        ),
    ] {
        let graph = Fst::try_build(triples.clone(), domains, 512).unwrap();
        let before = graph.mem_size_bytes();
        let bytes = graph.serialize();
        assert_eq!(
            before,
            graph.mem_size_bytes(),
            "serialization retained an address directory"
        );
        let restored = Fst::deserialize(&bytes).unwrap();
        assert_eq!(
            bytes,
            restored.serialize(),
            "snapshot is not canonical on round trip"
        );
        assert_eq!(graph.node_count(), restored.node_count());
        assert_eq!(graph.visited_count(), restored.visited_count());
        assert_eq!(graph.dimension_sizes(), restored.dimension_sizes());
        assert_eq!(
            sorted(restored.scan([None; 3]).collect()),
            sorted(triples.clone())
        );
        assert_eq!(
            restored.count_matching([None; 3]),
            sorted(triples.clone()).len() as u64
        );
        for &(s, p, o) in &triples {
            assert!(restored.contains(s, p, o));
            for mask in 0..8 {
                let bounds =
                    std::array::from_fn(|d| (mask & (1 << d) != 0).then_some([s, p, o][d]));
                let expected = sorted(
                    triples
                        .iter()
                        .copied()
                        .filter(|t| {
                            [t.0, t.1, t.2]
                                .into_iter()
                                .zip(bounds)
                                .all(|(x, b)| b.is_none_or(|b| x == b))
                        })
                        .collect(),
                );
                assert_eq!(sorted(restored.scan(bounds).collect()), expected);
                assert_eq!(restored.count_matching(bounds), expected.len() as u64);
            }
            // A representative subset suffices for the dense fixture.
            if s > 2 {
                break;
            }
        }
        let query = JoinQuery {
            num_vars: 3,
            patterns: vec![TriplePattern {
                subject: Term::Var(0),
                predicate: Term::Var(1),
                object: Term::Var(2),
            }],
        };
        let mut rows = join_run(&query, &restored).to_vec_rows();
        rows.sort_unstable();
        assert_eq!(
            rows,
            sorted(triples.clone())
                .iter()
                .map(|&(s, p, o)| vec![s, p, o])
                .collect::<Vec<_>>()
        );
        let machine = s_graph::path_ops::PathExpr::any(s_graph::traversal::Direction::Outgoing)
            .plus()
            .compile()
            .unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let hops = s_graph::traversal::HopRange::new(1, 3).unwrap();
        let _: ControlFlow<()> =
            machine.reachable(&graph, 0, hops, &mut Default::default(), |n, h| {
                a.push((n, h));
                ControlFlow::Continue(())
            });
        let _: ControlFlow<()> =
            machine.reachable(&restored, 0, hops, &mut Default::default(), |n, h| {
                b.push((n, h));
                ControlFlow::Continue(())
            });
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}

#[test]
fn default_api_matches_mutable_backend_and_reuses_workspaces() {
    let triples = vec![(0, 0, 1), (0, 0, 3), (1, 1, 2), (3, 1, 2)];
    let fst = Fst::build_dynamic(triples.clone(), DEFAULT_BLOCK_CAPACITY);
    let heap = Heap::build_dynamic(triples, DEFAULT_BLOCK_CAPACITY);
    let query = JoinQuery {
        num_vars: 2,
        patterns: vec![
            TriplePattern {
                subject: Term::Const(0),
                predicate: 0,
                object: Term::Var(0),
            },
            TriplePattern {
                subject: Term::Var(0),
                predicate: 1,
                object: Term::Var(1),
            },
        ],
    };
    let mut a = join_run(&query, &fst).to_vec_rows();
    a.sort_unstable();
    let mut b = join_run(&query, &heap).to_vec_rows();
    b.sort_unstable();
    assert_eq!(a, b);
    let plan = fst.join(&query).unwrap();
    let mut ws = JoinWorkspace::default();
    assert_eq!(
        plan.visit(&mut ws, |_| ControlFlow::Break(9)),
        ControlFlow::Break(9)
    );
    let mut count = 0;
    let _: ControlFlow<()> = plan.visit(&mut ws, |_| {
        count += 1;
        ControlFlow::Continue(())
    });
    assert_eq!(count, 2);
    assert!(plan.planning.nodes <= plan.planning.node_budget);
    assert_eq!(
        s_graph::path::bfs_sequential(&fst, 0, None),
        s_graph::path::bfs_sequential(&heap, 0, None)
    );
    let mut batch = Vec::new();
    fst.contains_batch_into(&[(0, 0, 1), (9, 9, 9)], &mut batch);
    assert_eq!(batch, vec![true, false]);
    assert_eq!(batch, fst.contains_batch(&[(0, 0, 1), (9, 9, 9)]));
}

#[test]
fn snapshots_reject_corruption_truncation_and_limits_without_panicking() {
    let graph = Fst::build(vec![(0, 0, 1), (1, 0, 2), (2, 0, 0)], [33, 1, 33], 512);
    let bytes = graph.serialize();
    for len in 0..bytes.len() {
        assert!(
            Fst::deserialize(&bytes[..len]).is_err(),
            "accepted truncation at {len}"
        );
    }
    let mut bad = bytes.clone();
    bad[0] ^= 1;
    assert!(Fst::deserialize(&bad).is_err());
    let mut bad = bytes.clone();
    bad[4] = 255;
    assert!(matches!(
        Fst::deserialize(&bad),
        Err(s_graph::TreeError::UnsupportedVersion(255))
    ));
    for limits in [
        DeserializeLimits {
            max_blocks: 0,
            ..Default::default()
        },
        DeserializeLimits {
            max_block_bytes: 1,
            ..Default::default()
        },
        DeserializeLimits {
            max_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(Fst::deserialize_with_limits(&bytes, limits).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Fst::deserialize(&bad).is_err());
    // Mutate each wire byte with several extreme values, exercising headers,
    // occupancy, deltas, block lengths and counts under small allocation limits.
    let limits = DeserializeLimits {
        max_blocks: 64,
        max_block_bytes: 512,
        max_bytes: 4096,
    };
    for i in 0..bytes.len() {
        for value in [0, 1, 127, 255] {
            let mut bad = bytes.clone();
            bad[i] = value;
            if let Ok(graph) = Fst::deserialize_with_limits(&bad, limits) {
                let triples = graph.scan([None; 3]).collect::<Vec<_>>();
                assert_eq!(graph.count_matching([None; 3]), triples.len() as u64);
                for (s, p, o) in triples {
                    assert!(graph.in_bounds(s, p, o) && graph.contains(s, p, o));
                }
            }
        }
    }
    // A root link pointing to itself must be rejected before any traversal.
    let height = u32::from_le_bytes(bytes[53..57].try_into().unwrap()) as usize;
    let table = 57 + height * 8;
    let blocks = u32::from_le_bytes(bytes[table..table + 4].try_into().unwrap()) as usize;
    let payload = table + 4 + blocks * 8;
    let mut bad = bytes.clone();
    bad[payload + 8..payload + 12].fill(0);
    assert!(Fst::deserialize(&bad).is_err());
    // Smaller non-power-of-two domain with unchanged schedule: only the
    // bottom-up coordinate proof catches the out-of-domain endpoint 2.
    let graph = Fst::build(vec![(5, 0, 0)], [6, 1, 1], 512);
    let mut bad = graph.serialize();
    bad[5..13].copy_from_slice(&5u64.to_le_bytes());
    assert!(Fst::deserialize(&bad).is_err());
}

#[test]
fn builders_validate_domains_and_single_point_graphs() {
    for domains in [[0, 1, 1], [1u64 << 33, 1, 1]] {
        assert!(Fst::try_build(vec![], domains, 512).is_err());
    }
    for capacity in [0, 8, 511, 513] {
        assert!(Fst::try_build(vec![], [1; 3], capacity).is_err());
    }
    assert!(matches!(
        Fst::try_build(vec![(1, 0, 0)], [1; 3], 512),
        Err(s_graph::TreeError::OutOfDomain(_))
    ));
    for present in [false, true] {
        let triples = if present { vec![(0, 0, 0)] } else { vec![] };
        let graph = Fst::build_dynamic(triples, 512);
        assert_eq!(graph.contains(0, 0, 0), present);
        let q = JoinQuery {
            num_vars: 1,
            patterns: vec![TriplePattern {
                subject: Term::Var(0),
                predicate: 0,
                object: Term::Var(0),
            }],
        };
        assert_eq!(join_run(&q, &graph).len(), usize::from(present));
        assert_eq!(graph.count_matching([None; 3]), u64::from(present));
        assert!(!graph.contains(1, 0, 0));
    }
}

#[test]
fn snapshot_streams_files_and_writer_errors() {
    let graph = Fst::build_dynamic(vec![(0, 0, 1), (1, 0, 2)], 512);
    let mut stream = graph.serialize();
    stream.extend(graph.serialize());
    let mut input = std::io::Cursor::new(stream);
    for _ in 0..2 {
        assert!(Fst::deserialize_from(&mut input).unwrap().contains(1, 0, 2));
    }
    struct Fail;
    impl std::io::Write for Fail {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("test writer failure"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert!(graph.serialize_into(&mut Fail).is_err());
    let path = std::env::temp_dir().join(format!("sgraph-fst-{}.sgf", std::process::id()));
    graph.serialize_to_path(&path).unwrap();
    assert!(Fst::deserialize_from_path(&path).unwrap().contains(0, 0, 1));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn compact_dags_cannot_overflow_logical_counts() {
    // Eleven shared full 64-way levels describe 2^66 triples using only
    // eleven records. Logical node visits still fit u64, so validating only
    // physical size or visit counts would miss the public count overflow.
    let height = 11u32;
    let mut bytes = b"SGF1\x01".to_vec();
    for n in [1u64 << 22; 3] {
        bytes.extend(n.to_le_bytes());
    }
    let visits: u64 = (0..height).map(|l| 64u64.pow(l)).sum();
    for n in [512u64, visits, height as u64] {
        bytes.extend(n.to_le_bytes());
    }
    bytes.extend(height.to_le_bytes());
    for _ in 0..height {
        bytes.extend(1u64.to_le_bytes());
    }
    bytes.extend(height.to_le_bytes());
    for l in 0..height {
        let len = if l + 1 == height { 8u32 } else { 264 };
        bytes.extend(len.to_le_bytes());
        bytes.extend(len.to_le_bytes());
    }
    for l in 0..height {
        bytes.extend(u64::MAX.to_le_bytes());
        if l + 1 < height {
            for _ in 0..64 {
                bytes.extend(512u32.to_le_bytes());
            }
        }
    }
    match Fst::deserialize(&bytes) {
        Err(s_graph::TreeError::CorruptData(message)) => {
            assert!(message.contains("triple count overflow"), "{message}")
        }
        _ => panic!("accepted an overflowing logical graph"),
    }
}

#[test]
fn default_fst_plans_can_be_shared_with_separate_workspaces() {
    let graph = Fst::build_dynamic(vec![(0, 0, 1), (1, 0, 2)], 512);
    let query = JoinQuery {
        num_vars: 2,
        patterns: vec![TriplePattern {
            subject: Term::Var(0),
            predicate: 0,
            object: Term::Var(1),
        }],
    };
    let plan = graph.join(&query).unwrap();
    std::thread::scope(|scope| {
        let workers = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut count = 0;
                    let _: ControlFlow<()> = plan.visit(&mut JoinWorkspace::default(), |_| {
                        count += 1;
                        ControlFlow::Continue(())
                    });
                    count
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), 2);
        }
    });
}
