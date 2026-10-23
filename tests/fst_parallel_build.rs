#![cfg(all(feature = "heap", feature = "parallel", feature = "experimental"))]
use s_graph::{Fst, PackedFst, Triple};

fn compare<S: s_graph::storage::ReclaimingBlockStorage + Send>(
    triples: Vec<Triple>,
    domains: [u64; 3],
    threads: usize,
) {
    let expected = Fst::try_build(triples.clone(), domains, 512).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let actual = pool
        .install(|| PackedFst::<S>::try_build_parallel(triples, domains, 512))
        .unwrap();
    assert_eq!(
        actual.node_count(),
        expected.node_count(),
        "lost global suffix sharing"
    );
    assert_eq!(actual.visited_count(), expected.visited_count());
    let mut a: Vec<_> = actual.scan([None; 3]).collect();
    let mut b: Vec<_> = expected.scan([None; 3]).collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
    for s in [0, 15, 777, 65_535] {
        let bounds = [Some(s), None, None];
        let mut a: Vec<_> = actual.scan(bounds).collect();
        let mut b: Vec<_> = expected.scan(bounds).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(
            actual.count_matching(bounds),
            expected.count_matching(bounds)
        );
    }
    let bytes = actual.serialize();
    let restored = Fst::deserialize(&bytes).unwrap();
    assert_eq!(restored.node_count(), expected.node_count());
    assert_eq!(restored.count_matching([None; 3]), a.len() as u64);
    let mut mutable = actual.into_mutable();
    let t = (0, 0, 0);
    mutable.insert(t.0, t.1, t.2).unwrap();
    assert!(mutable.contains(t.0, t.1, t.2));
    mutable.remove(t.0, t.1, t.2).unwrap();
    assert!(!mutable.freeze().unwrap().contains(t.0, t.1, t.2));
}

fn fixtures() -> Vec<(Vec<Triple>, [u64; 3])> {
    vec![
        (vec![], [1; 3]),
        (vec![(0, 0, 0); 70_000], [1; 3]),
        (
            (0..70_000u32).map(|x| (x, x % 4, x ^ 7)).collect(),
            [131_072, 4, 131_072],
        ),
        (
            (0..80_000u32)
                .map(|x| {
                    (
                        x,
                        x.wrapping_mul(17) % 32,
                        x.wrapping_mul(0x9e3779b9).rotate_left(7) % 131_072,
                    )
                })
                .collect(),
            [131_072, 32, 131_072],
        ),
        // Identical suffixes occur in different prefix partitions.
        (
            (0..1024u32)
                .flat_map(|x| (0..128).map(move |o| (x, 0, o)))
                .collect(),
            [1024, 1, 128],
        ),
        // Long unary prefix and uneven branching, with full-width domains.
        (
            (0..70_000u32)
                .map(|x| (x % 19, x % 3, x.wrapping_mul(0x9e3779b9) % 131_072))
                .collect(),
            [1 << 32, 3, 1 << 32],
        ),
    ]
}

#[test]
fn parallel_preserves_global_minimality_and_queries() {
    for threads in [1, 2, 4] {
        for (triples, domains) in fixtures() {
            compare::<s_graph::storage::HeapStorage>(triples, domains, threads);
        }
    }
}

#[test]
fn parallel_validates_before_partitioning() {
    assert!(Fst::try_build_parallel(vec![], [0, 1, 1], 512).is_err());
    assert!(Fst::try_build_parallel(vec![], [1; 3], 513).is_err());
    assert!(Fst::try_build_parallel(vec![(1, 0, 0)], [1; 3], 512).is_err());
}

#[cfg(feature = "file")]
#[test]
fn parallel_file() {
    let (triples, domains) = fixtures().remove(2);
    compare::<s_graph::storage_file::FileStorage>(triples, domains, 4);
}

#[cfg(feature = "compressed")]
#[test]
fn parallel_compressed() {
    let (triples, domains) = fixtures().remove(2);
    compare::<s_graph::storage_compressed::CompressedStorage>(triples, domains, 4);
}
