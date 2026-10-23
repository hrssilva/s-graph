//! One isolated construction per process; use /usr/bin/time for peak RSS.
use s_graph::{BlockStorage, PackedFst, Triple};
use std::time::Instant;

fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn run<S: BlockStorage + Send>(mode: &str, shape: &str, n: u32, threads: usize) {
    let mut triples: Vec<Triple> = (0..n)
        .map(|i| match shape {
            "functional" => (i, i % 4, i ^ 7),
            "shared" => (i / 1024, 0, i % 1024),
            "irregular" => (i, mix(i) % 32, mix(i ^ 12345) % n),
            "skewed" => (i % 19, i % 3, i),
            _ => panic!("unknown shape"),
        })
        .collect();
    let mut domains = [1; 3];
    for &(s, p, o) in &triples {
        for (d, v) in [s, p, o].into_iter().enumerate() {
            domains[d] = domains[d].max(v as u64 + 1);
        }
    }
    // Same shuffled input for every mode; generation/shuffle excluded from time.
    for i in (1..triples.len()).rev() {
        let j = mix(i as u32) as usize % (i + 1);
        triples.swap(i, j);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let start = Instant::now();
    let graph = pool
        .install(|| match mode {
            "sequential" => PackedFst::<S>::try_build(triples, domains, 4096),
            "parallel" => PackedFst::<S>::try_build_parallel(triples, domains, 4096),
            _ => panic!("unknown mode"),
        })
        .unwrap();
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(graph.count_matching([None; 3]), n as u64);
    println!(
        "mode={mode},shape={shape},triples={n},threads={threads},build_ms={ms:.3},retained_bytes={},nodes={},visited={}",
        graph.mem_size_bytes(),
        graph.node_count(),
        graph.visited_count()
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("parallel");
    let shape = args.get(2).map(String::as_str).unwrap_or("irregular");
    let n = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let threads = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    match args.get(5).map(String::as_str).unwrap_or("heap") {
        "heap" => run::<s_graph::storage::HeapStorage>(mode, shape, n, threads),
        #[cfg(feature = "file")]
        "file" => run::<s_graph::storage_file::FileStorage>(mode, shape, n, threads),
        #[cfg(feature = "compressed")]
        "compressed" => {
            run::<s_graph::storage_compressed::CompressedStorage>(mode, shape, n, threads)
        }
        _ => panic!("backend unavailable"),
    }
}
