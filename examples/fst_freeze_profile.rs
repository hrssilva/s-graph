//! Gated profiling of compact_blocks / ordinary freeze on exact-count graphs.
//! Use examples/join/prototype/fst_freeze_profile.py for perf and memory limits.
use s_graph::{BlockCompactionStats, PackedFst, Triple, storage::ReclaimingBlockStorage};
use std::{
    hint::black_box,
    io::{BufRead, Write},
    time::Instant,
};

fn control(command: &str) {
    if let Ok(path) = std::env::var("PROTO_PERF_CONTROL") {
        let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        writeln!(file, "{command}").unwrap();
        let path = std::env::var("PROTO_PERF_ACK").unwrap();
        let mut response = String::new();
        std::io::BufReader::new(std::fs::File::open(path).unwrap())
            .read_line(&mut response)
            .unwrap();
        assert_eq!(response.trim(), "ack");
    }
}
fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn input(triples: usize, shape: &str) -> Vec<Triple> {
    let vertices = (triples / 2) as u32;
    let mut values = Vec::with_capacity(triples);
    values.extend((0..vertices).flat_map(|s| {
        (0..2).map(move |p| {
            (
                s,
                p,
                if shape == "mixed" {
                    mix(s ^ (p * 0x1234)) % vertices
                } else {
                    (s ^ (p + 1)) % vertices
                },
            )
        })
    }));
    values
}
fn emit(value: String) {
    println!("{value}");
    std::io::stdout().flush().unwrap();
}
fn run<S: ReclaimingBlockStorage>(
    backend: &str,
    triples: usize,
    layout: &str,
    mode: &str,
    repeats: usize,
    shape: &str,
) -> s_graph::Result<()> {
    let vertices = (triples / 2) as u32;
    let domains = [vertices as u64, 4, vertices as u64];
    let setup = Instant::now();
    let snapshot = if layout == "packed" {
        Some(PackedFst::<S>::try_build(input(triples, shape), domains, 4096)?.serialize())
    } else {
        None
    };
    emit(format!(
        "{{\"phase\":\"fixture_ready\",\"triples\":{triples},\"backend\":\"{backend}\",\"layout\":\"{layout}\",\"mode\":\"{mode}\",\"shape\":\"{shape}\",\"setup_s\":{}}}",
        setup.elapsed().as_secs_f64()
    ));
    for repeat in 0..repeats {
        let setup = Instant::now();
        let mut graph = if let Some(snapshot) = &snapshot {
            PackedFst::<S>::deserialize(snapshot)?.into_mutable()
        } else {
            let mut graph = PackedFst::<S>::try_build(Vec::new(), domains, 4096)?.into_mutable();
            graph.insert_batch(&mut input(triples, shape))?;
            graph.freeze_in_place()?.into_mutable()
        };
        // Default Sampled(1) preparation keeps cold lookup scans out of setup cost at
        // scale. Index destruction is part of consuming freeze and is profiled.
        graph.prepare_updates();
        let edits = 1024usize.min(vertices as usize);
        for s in 0..edits as u32 {
            let old_o = if shape == "mixed" {
                mix(s) % vertices
            } else {
                (s ^ 1) % vertices
            };
            assert!(graph.remove(s, 0, old_o)?);
            assert!(graph.insert(s, 3, mix(s.wrapping_add(456)) % vertices)?);
        }
        assert_eq!(graph.count_matching([None; 3]), triples as u64);
        emit(format!(
            "{{\"phase\":\"prepared\",\"repeat\":{repeat},\"edits\":{},\"live_edit_states\":{},\"before_ram_bytes\":{},\"prepare_s\":{}}}",
            2 * edits,
            graph.edited_node_count(),
            graph.mem_size_bytes(),
            setup.elapsed().as_secs_f64()
        ));

        let live_edit_states = graph.edited_node_count();
        control("enable");
        let start = Instant::now();
        let (packed, stats) = if mode == "blocks" {
            graph.compact_blocks()?
        } else if mode == "depths" {
            graph.compact_depths()?
        } else if mode == "streaming" {
            graph.compact_streaming()?
        } else {
            (graph.freeze()?, BlockCompactionStats::default())
        };
        black_box(&packed);
        let elapsed_s = start.elapsed().as_secs_f64();
        control("disable");

        assert_eq!(packed.count_matching([None; 3]), triples as u64);
        for s in [0, (edits / 2) as u32, edits as u32 - 1] {
            assert!(packed.contains(s, 3, mix(s.wrapping_add(456)) % vertices));
            let old_o = if shape == "mixed" {
                mix(s) % vertices
            } else {
                (s ^ 1) % vertices
            };
            assert!(!packed.contains(s, 0, old_o));
        }
        if mode != "freeze" {
            assert_eq!(stats.released_blocks, stats.source_blocks);
        }
        if mode == "streaming" {
            assert_eq!(stats.rank_lookups, 0);
            assert_eq!(stats.rank_bytes, 0);
            assert_eq!(stats.forwarded_edit_states, live_edit_states);
        }
        emit(format!(
            "{{\"phase\":\"freeze\",\"repeat\":{repeat},\"triples\":{triples},\"backend\":\"{backend}\",\"layout\":\"{layout}\",\"mode\":\"{mode}\",\"shape\":\"{shape}\",\"elapsed_s\":{elapsed_s},\"physical_states\":{},\"after_ram_bytes\":{},\"mark_ns\":{},\"planning_ns\":{},\"copy_ns\":{},\"workspace_bytes\":{},\"rank_bytes\":{},\"relocation_bytes\":{},\"source_blocks\":{},\"released_blocks\":{},\"input_block_bytes\":{},\"peak_block_bytes\":{},\"output_record_bytes\":{},\"frontier_bytes\":{},\"scratch_bytes\":{},\"peak_frontier_states\":{},\"peak_relocation_states\":{},\"source_block_reads\":{},\"output_fixup_blocks\":{},\"frontier_buffer_growths\":{},\"relocation_buffer_growths\":{},\"fixup_ns\":{},\"rank_lookups\":{},\"relocation_runs\":{},\"forwarded_edit_states\":{}}}",
            packed.node_count(),
            packed.mem_size_bytes(),
            stats.mark_ns,
            stats.planning_ns,
            stats.copy_ns,
            stats.workspace_bytes,
            stats.rank_bytes,
            stats.relocation_bytes,
            stats.source_blocks,
            stats.released_blocks,
            stats.input_uncompressed_block_bytes,
            stats.peak_uncompressed_block_bytes,
            stats.output_record_bytes,
            stats.frontier_bytes,
            stats.scratch_bytes,
            stats.peak_frontier_states,
            stats.peak_relocation_states,
            stats.source_block_reads,
            stats.output_fixup_blocks,
            stats.frontier_buffer_growths,
            stats.relocation_buffer_growths,
            stats.fixup_ns,
            stats.rank_lookups,
            stats.relocation_runs,
            stats.forwarded_edit_states
        ));
    }
    Ok(())
}
fn main() -> s_graph::Result<()> {
    if let Ok(expected) = std::env::var("SGRAPH_EXPECTED_AS_BYTES") {
        let limits = std::fs::read_to_string("/proc/self/limits").unwrap();
        let values: Vec<_> = limits
            .lines()
            .find_map(|l| l.strip_prefix("Max address space"))
            .unwrap()
            .split_whitespace()
            .collect();
        assert_eq!(values[0], expected);
        assert_eq!(values[1], expected);
    }
    let args: Vec<_> = std::env::args().collect();
    let triples = args
        .get(1)
        .map_or(1_000_000, |s| s.parse::<usize>().unwrap());
    let backend = args.get(2).map_or("heap", String::as_str);
    let layout = args.get(3).map_or("packed", String::as_str);
    let mode = args.get(4).map_or("streaming", String::as_str);
    let repeats = args.get(5).map_or(1, |s| s.parse::<usize>().unwrap());
    let shape = args.get(6).map_or("mixed", String::as_str);
    assert!(triples >= 1_000_000 && triples.is_multiple_of(2) && triples <= u32::MAX as usize);
    assert!(
        ["packed", "bfs"].contains(&layout)
            && ["blocks", "depths", "streaming", "freeze"].contains(&mode)
    );
    assert!(["mixed", "functional"].contains(&shape) && repeats > 0);
    match backend {
        "heap" => {
            run::<s_graph::storage::HeapStorage>(backend, triples, layout, mode, repeats, shape)
        }
        #[cfg(feature = "file")]
        "file" => run::<s_graph::storage_file::FileStorage>(
            backend, triples, layout, mode, repeats, shape,
        ),
        #[cfg(feature = "compressed")]
        "compressed" => run::<s_graph::storage_compressed::CompressedStorage>(
            backend, triples, layout, mode, repeats, shape,
        ),
        _ => panic!("unknown or disabled backend"),
    }
}
