//! One-off tool: generate a synthetic RDF triple dataset and dump it as raw
//! little-endian u32 triples to a binary file, so an external (non-Rust)
//! benchmark program can consume byte-for-byte the same dataset used by
//! `examples/compare`. Not part of the tracked comparison suite itself --
//! used to feed the sdsl-lite k2_tree benchmark.
//!
//! File format: 4x u32 header (num_s, num_p, num_o, num_triples), then
//! num_triples x 3x u32 (s, p, o), all little-endian.
//!
//! Usage: cargo run --release --example export_dataset -- <scenario> <out_path> [num_triples]
//!   scenario: "entity_skew_1.0" | "entity_skew_1.5" | "entity_skew_2.0" | "pred_skew_only"

#[path = "common/mod.rs"]
mod common;

use common::rdf_gen::{zipfian_entities, zipfian_predicates};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: export_dataset <scenario> <out_path> [num_triples]");
        std::process::exit(1);
    }
    let scenario = &args[1];
    let out_path = &args[2];
    let num_triples: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(20_000_000);

    const S: u32 = 2_000_000;
    const P: u32 = 19;
    const O: u32 = 2_000_000;

    let triples = match scenario.as_str() {
        "pred_skew_only" => zipfian_predicates(S, P, O, num_triples, 1.2, 2026),
        "entity_skew_1.0" => zipfian_entities(S, P, O, num_triples, 1.0, 1.2, 1.0, 2026),
        "entity_skew_1.5" => zipfian_entities(S, P, O, num_triples, 1.5, 1.2, 1.5, 2026),
        "entity_skew_2.0" => zipfian_entities(S, P, O, num_triples, 2.0, 1.2, 2.0, 2026),
        other => {
            eprintln!("Unknown scenario: {other}");
            std::process::exit(1);
        }
    };

    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    let header = [S, P, O, triples.len() as u32];
    for h in header {
        f.write_all(&h.to_le_bytes()).unwrap();
    }
    for &(s, p, o) in &triples {
        f.write_all(&s.to_le_bytes()).unwrap();
        f.write_all(&p.to_le_bytes()).unwrap();
        f.write_all(&o.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    eprintln!("wrote {} triples to {out_path}", triples.len());
}
