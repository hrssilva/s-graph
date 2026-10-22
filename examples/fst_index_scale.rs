//! Exact-triple-count index scaling. Run via prototype/fst_index_scale.py.
use s_graph::Fst;
use std::{io::Write, time::Instant};

fn mix(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn process_memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let value = |key: &str| -> u64 {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            * 1024
    };
    format!(
        "\"rss_bytes\":{},\"peak_rss_bytes\":{},\"virtual_bytes\":{}",
        value("VmRSS:"),
        value("VmHWM:"),
        value("VmSize:")
    )
}
fn emit(row: &str) {
    let row = row.strip_suffix('}').unwrap();
    println!("{row},{} }}", process_memory());
    std::io::stdout().flush().unwrap();
}
fn main() -> s_graph::Result<()> {
    let cap: u64 = std::env::var("SGRAPH_EXPECTED_AS_BYTES")
        .expect("run via the systemd driver")
        .parse()
        .unwrap();
    let limits = std::fs::read_to_string("/proc/self/limits").unwrap();
    let address_space: Vec<_> = limits
        .lines()
        .find_map(|l| l.strip_prefix("Max address space"))
        .unwrap()
        .split_whitespace()
        .collect();
    assert_eq!(address_space[0].parse::<u64>().unwrap(), cap);
    assert_eq!(address_space[1].parse::<u64>().unwrap(), cap);
    let args: Vec<_> = std::env::args().collect();
    let triples: usize = args.get(1).expect("triple count").parse().unwrap();
    let shape = args.get(2).expect("mixed or functional");
    assert!(triples >= 8 && triples.is_multiple_of(2) && triples <= u32::MAX as usize);
    assert!(shape == "mixed" || shape == "functional");
    let vertices = (triples / 2) as u32;
    let started = Instant::now();
    emit(&format!(
        "{{\"phase\":\"generating\",\"triples\":{triples},\"shape\":\"{shape}\",\"input_bytes\":{}}}",
        triples * 12
    ));
    // Every (subject,predicate) pair is unique: these are distinct triples.
    let mut input = Vec::with_capacity(triples);
    input.extend((0..vertices).flat_map(|s| {
        (0..2).map(move |p| {
            let o = if shape == "functional" {
                (s ^ (p + 1)) % vertices
            } else {
                mix(s ^ (p * 0x1234)) % vertices
            };
            (s, p, o)
        })
    }));
    let generated_s = started.elapsed().as_secs_f64();
    emit(&format!(
        "{{\"phase\":\"building\",\"triples\":{triples},\"shape\":\"{shape}\",\"generated_s\":{generated_s}}}"
    ));
    let graph = Fst::try_build(input, [vertices as u64, 4, vertices as u64], 4096)?;
    let built_s = started.elapsed().as_secs_f64();
    let nodes = graph.node_count();
    let height = graph.height();
    let packed = graph.mem_size_bytes();
    let levels = graph.physical_states_per_level().to_vec();
    emit(&format!(
        "{{\"phase\":\"built\",\"triples\":{triples},\"shape\":\"{shape}\",\"physical_states\":{nodes},\"packed_ram_bytes\":{packed},\"built_s\":{built_s}}}"
    ));
    let mut graph = graph.into_mutable();
    let before = graph.memory_usage();
    let prepare = Instant::now();
    graph.prepare_updates();
    let prepare_s = prepare.elapsed().as_secs_f64();
    let after = graph.memory_usage();
    // Each entry has at most an 8-byte offset and a 4-byte prefix. Include
    // a conservative bound for per-level representation/vector metadata.
    assert!(after.base_register_bytes <= nodes as usize * 12 + height * 128);
    assert_eq!(after.edit_payload_bytes, 0);
    emit(&format!(
        "{{\"phase\":\"indexed\",\"triples\":{triples},\"shape\":\"{shape}\",\"physical_states\":{nodes},\"height\":{height},\"states_per_level\":{levels:?},\"packed_ram_bytes\":{packed},\"before_ram_bytes\":{},\"indexed_ram_bytes\":{},\"index_bytes\":{},\"prepare_s\":{prepare_s}}}",
        before.total_ram_bytes(),
        after.total_ram_bytes(),
        after.base_register_bytes
    ));
    // Counting visits physical states with memoization. Include its transient
    // peak in /proc/self/status, but outside the retained index figures.
    assert_eq!(graph.count_matching([None; 3]), triples as u64);
    assert!(graph.insert(0, 3, 0)?);
    assert!(graph.remove(0, 3, 0)?);
    assert_eq!(graph.edited_node_count(), 0);
    emit(&format!(
        "{{\"phase\":\"complete\",\"triples\":{triples},\"shape\":\"{shape}\",\"elapsed_s\":{},\"count_and_inverse_edit_checks\":true}}",
        started.elapsed().as_secs_f64()
    ));
    Ok(())
}
