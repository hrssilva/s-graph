//! Comprehensive head-to-head comparison of Mk²-tree, IK²-tree, Omnitree,
//! and the `radix64.rs`-inspired RadixTree prototype: build time, memory
//! footprint (actual allocated bytes), process peak RSS, point/row/col
//! query latency, unbound-predicate query strategies, BFS path-algorithm
//! throughput (sequential vs. parallel), and worst-case-optimal
//! multiway-join throughput (2-hop/triangle/star patterns), all run over
//! identical synthetic RDF datasets.
//!
//! Run with `cargo run --release --example compare` -- release mode
//! matters, these are meant to be meaningful timing/memory numbers, not
//! debug-build noise.

#[path = "../common/mod.rs"]
mod common;
mod report;
mod scenarios;

use common::radixtree_bp::RadixTreeBp;
use common::radixtree_dfs::RadixTreeDfs;
use common::radixtree_veb::RadixTreeVeb;
use common::rdf_gen::{chain, star_triangle, uniform, zipfian_entities, zipfian_predicates};
use report::Report;

fn main() {
    let mut report = Report::new();

    println!("Building datasets and running benchmarks (release mode recommended: `cargo run --release --example compare`)...\n");

    // --- Uniform and Zipfian-predicate-skew datasets: build/size, point,
    // row/col, and unbound-predicate query benchmarks. Zipfian mirrors real
    // RDF's heavy predicate skew (e.g. DBpedia) -- the shape that
    // specifically stresses Mk2-tree's scan-every-predicate-tree weakness
    // and IK2-tree's interleaving strength on unbound-predicate queries.
    const NUM_S: u32 = 300;
    const NUM_P: u32 = 25;
    const NUM_O: u32 = 300;
    const NUM_TRIPLES: usize = 5000;

    for (name, triples) in [
        ("uniform", uniform(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 42)),
        ("zipfian(1.2)", zipfian_predicates(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 1.2, 42)),
    ] {
        println!("=== {name} ({} triples) ===", triples.len());
        let (mk2, ik2, omni, radix_static, radix_dynamic) = scenarios::bench_build_and_size(name, &triples, &mut report);
        scenarios::bench_point_queries(name, &triples, NUM_S, NUM_P, NUM_O, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_fixed_predicate_query(name, &triples, 0, &mk2, &ik2, &mut report);
        scenarios::bench_row_col_queries(name, &triples, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_unbound_predicate(name, NUM_S, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_predicate_lookup(name, &triples, &mk2, &ik2, &mut report);
        scenarios::bench_predicate_lookup_omni(name, &triples, &omni, &mut report);
        for (radix, label) in [(&radix_static, "RadixTree(static)"), (&radix_dynamic, "RadixTree(dynamic)")] {
            scenarios::bench_point_queries_radix(name, &triples, NUM_S, NUM_P, NUM_O, radix, label, &mut report);
            scenarios::bench_fixed_predicate_query_radix(name, &triples, 0, radix, label, &mut report);
            scenarios::bench_row_col_queries_radix(name, &triples, radix, label, &mut report);
            scenarios::bench_unbound_predicate_radix(name, NUM_S, radix, label, &mut report);
            scenarios::bench_predicate_lookup_radix(name, &triples, radix, label, &mut report);
        }

        // Layout/encoding-only variants (dynamic split only -- static vs.
        // dynamic isn't the axis these investigate, so one schedule keeps
        // the report from ballooning). See radixtree_veb/_dfs/_bp's own
        // doc comments for what each isolates.
        let veb = scenarios::bench_build_and_size_variant(name, "RadixTree(veb)", &triples, RadixTreeVeb::build_dynamic, &mut report);
        scenarios::bench_point_queries_variant(name, &triples, NUM_S, NUM_P, NUM_O, &veb, "RadixTree(veb)", &mut report);
        scenarios::bench_row_col_queries_variant(name, &triples, &veb, "RadixTree(veb)", &mut report);
        scenarios::bench_unbound_predicate_variant(name, NUM_S, &veb, "RadixTree(veb)", &mut report);
        scenarios::bench_predicate_lookup_variant(name, &triples, &veb, "RadixTree(veb)", &mut report);

        let dfs = scenarios::bench_build_and_size_variant(name, "RadixTree(dfs)", &triples, RadixTreeDfs::build_dynamic, &mut report);
        scenarios::bench_point_queries_variant(name, &triples, NUM_S, NUM_P, NUM_O, &dfs, "RadixTree(dfs)", &mut report);
        scenarios::bench_row_col_queries_variant(name, &triples, &dfs, "RadixTree(dfs)", &mut report);
        scenarios::bench_unbound_predicate_variant(name, NUM_S, &dfs, "RadixTree(dfs)", &mut report);
        scenarios::bench_predicate_lookup_variant(name, &triples, &dfs, "RadixTree(dfs)", &mut report);

        let bp = scenarios::bench_build_and_size_variant(name, "RadixTree(bp)", &triples, RadixTreeBp::build_dynamic, &mut report);
        scenarios::bench_point_queries_variant(name, &triples, NUM_S, NUM_P, NUM_O, &bp, "RadixTree(bp)", &mut report);
        scenarios::bench_row_col_queries_variant(name, &triples, &bp, "RadixTree(bp)", &mut report);
        scenarios::bench_unbound_predicate_variant(name, NUM_S, &bp, "RadixTree(bp)", &mut report);
        scenarios::bench_predicate_lookup_variant(name, &triples, &bp, "RadixTree(bp)", &mut report);
    }

    // --- Sanity check: a single predicate. This removes any reason for
    // predicate-handling differences to matter at all -- IK2Tree's active-
    // list width is always 1 (no compaction possible or needed), Omnitree's
    // P dimension never has >1 distinct value so it's never split,
    // RadixTree's P dimension needs 0 bits (domain size 1), and Mk2Forest
    // is just one K2Tree. If the four structures *don't* converge to
    // roughly the same shape here, something other than predicate handling
    // is driving the differences seen elsewhere.
    {
        let name = "single_predicate";
        let (s_n, p_n, o_n) = (300u32, 1u32, 300u32);
        let triples = uniform(s_n, p_n, o_n, NUM_TRIPLES, 77);
        println!("=== {name} ({} triples) ===", triples.len());
        let (mk2, ik2, omni, radix_static, radix_dynamic) = scenarios::bench_build_and_size(name, &triples, &mut report);
        scenarios::bench_point_queries(name, &triples, s_n, p_n, o_n, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_row_col_queries(name, &triples, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_unbound_predicate(name, s_n, &mk2, &ik2, &omni, &mut report);
        for (radix, label) in [(&radix_static, "RadixTree(static)"), (&radix_dynamic, "RadixTree(dynamic)")] {
            scenarios::bench_point_queries_radix(name, &triples, s_n, p_n, o_n, radix, label, &mut report);
            scenarios::bench_row_col_queries_radix(name, &triples, radix, label, &mut report);
            scenarios::bench_unbound_predicate_radix(name, s_n, radix, label, &mut report);
        }
    }

    // --- Opposite ratio: predicate cardinality roughly equal to
    // subject/object cardinality (unlike the main loop's 25-vs-300 skew).
    // RadixTree's height is driven by S+P+O bits combined, while IK2Tree's
    // is driven by S,O alone (predicate lives in per-node width, not tree
    // depth) -- when P needed far fewer bits than S,O, RadixTree paid for
    // an "extra level" of combined depth that bought it nothing once P
    // exhausted, while IK2Tree never had that tax. With P's bit-width now
    // comparable to S,O's, that asymmetry should shrink or disappear,
    // testing whether RadixTree's predicate_lookup deficit was really just
    // that same structural tax showing up again, or something else.
    {
        let name = "high_predicate_cardinality";
        let (s_n, p_n, o_n) = (300u32, 300u32, 300u32);
        let triples = uniform(s_n, p_n, o_n, NUM_TRIPLES, 314);
        println!("=== {name} ({} triples) ===", triples.len());
        let (mk2, ik2, omni, radix_static, radix_dynamic) = scenarios::bench_build_and_size(name, &triples, &mut report);
        scenarios::bench_point_queries(name, &triples, s_n, p_n, o_n, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_row_col_queries(name, &triples, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_predicate_lookup(name, &triples, &mk2, &ik2, &mut report);
        scenarios::bench_predicate_lookup_omni(name, &triples, &omni, &mut report);
        for (radix, label) in [(&radix_static, "RadixTree(static)"), (&radix_dynamic, "RadixTree(dynamic)")] {
            scenarios::bench_point_queries_radix(name, &triples, s_n, p_n, o_n, radix, label, &mut report);
            scenarios::bench_row_col_queries_radix(name, &triples, radix, label, &mut report);
            scenarios::bench_predicate_lookup_radix(name, &triples, radix, label, &mut report);
        }
    }

    // --- Large-entity-count / few-predicates dataset: this is the shape
    // the dynamic split schedule is meant for -- predicate exhausts (few
    // bits needed) very early while subject/object (many more distinct
    // values) continue for many more levels, so a static split wastes bits
    // on every one of those remaining levels while dynamic keeps
    // redistributing predicate's freed budget to them.
    {
        let name = "large_domain_few_predicates";
        let triples = uniform(20_000, 10, 20_000, 50_000, 99);
        println!("=== {name} ({} triples) ===", triples.len());
        scenarios::bench_radix_split_comparison(name, &triples, &mut report);

        // radixtree_bp's whole hypothesis needs low actual per-node
        // branching to pay off (see its own doc comment) -- this dataset's
        // huge S/O domains relative to its 50k triples is the closest
        // thing in this harness to that regime, unlike the dense
        // 300/300/300 uniform loop above.
        scenarios::bench_build_and_size_variant(name, "RadixTree(bp)", &triples, RadixTreeBp::build_dynamic, &mut report);
    }

    // --- Large-scale, realistic-skew datasets (all eight structures):
    // this is where the "our uniform benchmarks look nothing like a real
    // graph" investigation landed. 20M triples, a 2M-entity universe, and
    // a progression from predicate-only skew (still uniform S/O, like
    // `zipfian_predicates` above) through increasingly strong entity
    // (subject/object) skew. Entity skew turned out to dominate: with real
    // Zipfian skew over a fixed domain, the vast majority of the domain's
    // long tail is never actually sampled in 20M draws, so the *effective*
    // distinct-value count collapses far below the nominal 2M -- directly
    // shrinking `total_bits_needed` (fewer bits -> shorter tree) and
    // sharply increasing node-sharing (many triples converging on the same
    // small set of hot entities). Uniform S/O with only predicate skew
    // (LUBM has ~19 predicates) is close to worst-case entropy for every
    // structure here, not a RadixTree-specific weakness -- this isn't
    // about which structure wins, it's about how unrealistic the earlier
    // uniform-S/O scenarios were as a stand-in for real RDF data.
    const LARGE_S: u32 = 2_000_000;
    const LARGE_P: u32 = 19; // LUBM-scale predicate count
    const LARGE_O: u32 = 2_000_000;
    const LARGE_TRIPLES: usize = 20_000_000;
    for (name, triples) in [
        ("large_scale_pred_skew_only", zipfian_predicates(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.2, 2026)),
        ("large_scale_entity_skew_1.0", zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.0, 1.2, 1.0, 2026)),
        ("large_scale_entity_skew_1.5", zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 1.5, 1.2, 1.5, 2026)),
        ("large_scale_entity_skew_2.0", zipfian_entities(LARGE_S, LARGE_P, LARGE_O, LARGE_TRIPLES, 2.0, 1.2, 2.0, 2026)),
    ] {
        println!("=== {name} ({} triples) ===", triples.len());
        scenarios::bench_large_scale_build_and_size(name, &triples, &mut report);
        scenarios::bench_insertion(name, &triples, &mut report);
    }

    // --- Chain-shaped dataset: path-algorithm (BFS) benchmark.
    {
        let name = "chain";
        let triples = chain(30, 25, 5, 7);
        println!("=== {name} ({} triples) ===", triples.len());
        let (mk2, ik2, omni, radix_static, radix_dynamic) = scenarios::bench_build_and_size(name, &triples, &mut report);
        scenarios::bench_bfs(name, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_bfs_radix(name, &radix_static, "RadixTree(static)", &mut report);
        scenarios::bench_bfs_radix(name, &radix_dynamic, "RadixTree(dynamic)", &mut report);

        let veb = RadixTreeVeb::build_dynamic(&triples);
        scenarios::bench_bfs_variant(name, &veb, "RadixTree(veb)", &mut report);
        let dfs = RadixTreeDfs::build_dynamic(&triples);
        scenarios::bench_bfs_variant(name, &dfs, "RadixTree(dfs)", &mut report);
        let bp = RadixTreeBp::build_dynamic(&triples);
        scenarios::bench_bfs_variant(name, &bp, "RadixTree(bp)", &mut report);
    }

    // --- Star/triangle-shaped dataset: WCOJ multiway-join benchmark.
    {
        let name = "star_triangle";
        let triples = star_triangle(60, 10, 0.8, 3, 6);
        println!("=== {name} ({} triples) ===", triples.len());
        let (mk2, ik2, omni, radix_static, radix_dynamic) = scenarios::bench_build_and_size(name, &triples, &mut report);
        scenarios::bench_join(name, &mk2, &ik2, &omni, &mut report);
        scenarios::bench_join_radix(name, &radix_static, "RadixTree(static)", &mut report);
        scenarios::bench_join_radix(name, &radix_dynamic, "RadixTree(dynamic)", &mut report);

        let veb = RadixTreeVeb::build_dynamic(&triples);
        scenarios::bench_join_variant(name, &veb, "RadixTree(veb)", &mut report);
        let dfs = RadixTreeDfs::build_dynamic(&triples);
        scenarios::bench_join_variant(name, &dfs, "RadixTree(dfs)", &mut report);
        let bp = RadixTreeBp::build_dynamic(&triples);
        scenarios::bench_join_variant(name, &bp, "RadixTree(bp)", &mut report);
    }

    if let Some(rss) = common::metrics::peak_rss_kb() {
        report.push("(overall)", "(process)", "final_peak_rss", format!("{rss} KiB"));
    }

    println!("\n--- Comparison table ---\n");
    report.print_table();

    println!(
        "\nNote: peak RSS is a whole-process, cumulative figure (this harness builds all four \
         structures for every dataset in one process) -- read it as an overall memory ceiling for \
         the run, not a clean per-structure isolate. Per-structure `mem_size` (summed allocated \
         bytes across each structure's own backing arrays) is the precise per-structure comparison."
    );

    if std::env::args().any(|a| a == "--csv") {
        let path = "compare_report.csv";
        std::fs::write(path, report.to_csv()).expect("failed to write CSV report");
        println!("\nWrote {path}");
    }
}
