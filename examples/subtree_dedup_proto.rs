//! Diagnostic prototype -- see `DESIGN.md`'s "subtree deduplication"
//! entry and `tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric::
//! subtree_dedup_report`'s own doc comment. Measures, on a handful of
//! dataset shapes already used throughout this project's investigation,
//! how much of each tree's node-record bytes are spent on subtrees that
//! are structurally identical to another subtree at the same level --
//! the precondition a Carmona & Manzini-style ("Depth First
//! Representations of k²-trees", arXiv:2505.11302/2607.28136) dedup
//! scheme would need to pay off.
//!
//! Deliberately kept in the repository (not deleted after measuring,
//! unlike this project's usual throwaway-instrumentation convention) so
//! a follow-up idea can be measured against the same baseline. Run with:
//!   cargo run --release --example subtree_dedup_proto --features heap

use s_graph::tree::{Heap, SubtreeDedupReport, Triple};

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::{uniform, zipfian_entities, zipfian_predicates};

fn human_bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.2} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2} MB", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.2} KB", n / 1e3)
    } else {
        format!("{n:.0} B")
    }
}

/// Bytes needed for a back-reference into a group of `distinct` subtrees --
/// `ceil(log2(distinct))` bits, rounded up to whole bytes, minimum 1 when
/// there is more than one distinct subtree to point among. Still an
/// optimistic stand-in (a real pointer format would need to be
/// distinguishable from an ordinary record inline, which costs more than
/// this), but a far more honest number than the raw "bytes not stored"
/// figure, which assumes a free, zero-cost indirection.
fn pointer_bytes(distinct: u64) -> u64 {
    if distinct <= 1 {
        0
    } else {
        ((64 - (distinct - 1).leading_zeros()) as u64).div_ceil(8).max(1)
    }
}

fn report_shape(name: &str, triples: Vec<Triple>, num_s: u32, num_p: u32, num_o: u32) {
    let tree = Heap::build_dynamic(triples, s_graph::DEFAULT_BLOCK_CAPACITY);
    let report: SubtreeDedupReport = tree.subtree_dedup_report();
    println!("=== {name} (height={}, domain={num_s}x{num_p}x{num_o}) ===", tree.height());
    println!(
        "  {:>6} {:>12} {:>12} {:>8} {:>14} {:>14} {:>8} {:>8}",
        "level", "nodes", "distinct", "ratio", "bytes", "distinct_bytes", "saved%", "net%"
    );
    let mut total_net_bytes = 0u64;
    for lvl in &report.per_level {
        let ratio = lvl.distinct_count as f64 / lvl.node_count.max(1) as f64;
        let saved_pct = 100.0 * (1.0 - lvl.distinct_bytes as f64 / lvl.total_bytes.max(1) as f64);
        let ptr = pointer_bytes(lvl.distinct_count);
        let net_bytes = lvl.distinct_bytes + (lvl.node_count - lvl.distinct_count) * ptr;
        total_net_bytes += net_bytes;
        let net_pct = 100.0 * (1.0 - net_bytes as f64 / lvl.total_bytes.max(1) as f64);
        println!(
            "  {:>6} {:>12} {:>12} {:>7.3} {:>14} {:>14} {:>7.2}% {:>7.2}%",
            lvl.level,
            lvl.node_count,
            lvl.distinct_count,
            ratio,
            human_bytes(lvl.total_bytes),
            human_bytes(lvl.distinct_bytes),
            saved_pct,
            net_pct
        );
    }
    let total_bytes = report.total_bytes();
    let distinct_bytes = report.total_distinct_bytes();
    let saved_pct = 100.0 * (1.0 - distinct_bytes as f64 / total_bytes.max(1) as f64);
    let net_pct = 100.0 * (1.0 - total_net_bytes as f64 / total_bytes.max(1) as f64);
    println!(
        "  TOTAL: {} nodes, {} distinct ({:.3} ratio), {} -> {} upper bound ({:.2}% saved) / {} net-of-pointer-cost ({:.2}% saved)",
        report.total_nodes(),
        report.total_distinct(),
        report.total_distinct() as f64 / report.total_nodes().max(1) as f64,
        human_bytes(total_bytes),
        human_bytes(distinct_bytes),
        saved_pct,
        human_bytes(total_net_bytes),
        net_pct
    );
    let real_mem = tree.mem_size_bytes() as u64;
    let real_saved_pct = 100.0 * (1.0 - total_net_bytes as f64 / real_mem.max(1) as f64);
    println!(
        "  REAL TREE: mem_size_bytes() = {} (node records are {} of that, {:.1}%); FST-net estimate would be {} ({:.2}% of real total mem, not just node records)",
        human_bytes(real_mem),
        human_bytes(total_bytes),
        100.0 * total_bytes as f64 / real_mem.max(1) as f64,
        human_bytes(total_net_bytes),
        real_saved_pct
    );
    println!();
}

fn main() {
    // Same symmetric domain used throughout DESIGN.md's query benchmarks.
    const NUM_S: u32 = 500_000;
    const NUM_P: u32 = 200;
    const NUM_O: u32 = 500_000;
    const NUM_TRIPLES: usize = 10_000_000;

    report_shape("uniform (symmetric)", uniform(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 42), NUM_S, NUM_P, NUM_O);
    report_shape(
        "zipfian predicates (symmetric)",
        zipfian_predicates(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 1.2, 42),
        NUM_S,
        NUM_P,
        NUM_O,
    );
    report_shape(
        "zipfian S+P+O (symmetric, high skew)",
        zipfian_entities(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 1.5, 1.5, 1.5, 42),
        NUM_S,
        NUM_P,
        NUM_O,
    );

    // Asymmetric domain used for the S-bound/O-bound shapes in DESIGN.md.
    const NUM_S2: u32 = 20_000;
    const NUM_P2: u32 = 200;
    const NUM_O2: u32 = 2_000_000;
    const NUM_TRIPLES2: usize = 3_000_000;

    report_shape("uniform (asymmetric)", uniform(NUM_S2, NUM_P2, NUM_O2, NUM_TRIPLES2, 42), NUM_S2, NUM_P2, NUM_O2);
    report_shape(
        "zipfian S+P+O (asymmetric, high skew)",
        zipfian_entities(NUM_S2, NUM_P2, NUM_O2, NUM_TRIPLES2, 1.5, 1.5, 1.5, 42),
        NUM_S2,
        NUM_P2,
        NUM_O2,
    );
}
