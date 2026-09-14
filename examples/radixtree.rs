//! Thin standalone demo for the `radix64.rs`-inspired RadixTree prototype:
//! builds a small synthetic RDF dataset, runs point and two-bound/one-free
//! queries, and cross-checks every result against a brute-force
//! [`common::reference::NaiveIndex`] over ~500 randomized probes.

#[path = "common/mod.rs"]
mod common;

use common::metrics::{human_bytes, MemSize};
use common::radixtree::RadixTree;
use common::rdf_gen::{uniform, zipfian_predicates, SplitMix64};
use common::reference::NaiveIndex;

fn main() {
    const NUM_S: u32 = 200;
    const NUM_P: u32 = 20;
    const NUM_O: u32 = 200;
    const NUM_TRIPLES: usize = 3000;

    for (name, triples) in [
        ("uniform", uniform(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 42)),
        ("zipfian", zipfian_predicates(NUM_S, NUM_P, NUM_O, NUM_TRIPLES, 1.2, 42)),
    ] {
        println!("=== dataset: {name} ({} triples) ===", triples.len());
        let naive = NaiveIndex::build(&triples);
        let tree = RadixTree::build(&triples, RadixTree::default_split());
        println!(
            "  RadixTree (static [2,2,2]): height={}, {} node-words, {} logical triples, {} allocated",
            tree.height(),
            tree.node_word_count(),
            naive.len(),
            human_bytes(tree.mem_size_bytes())
        );
        let dyn_tree = RadixTree::build_dynamic(&triples);
        println!(
            "  RadixTree (dynamic split):  height={}, {} node-words, {} logical triples, {} allocated",
            dyn_tree.height(),
            dyn_tree.node_word_count(),
            naive.len(),
            human_bytes(dyn_tree.mem_size_bytes())
        );

        let mut rng = SplitMix64::new(7);
        let mut mismatches = 0;
        for _ in 0..500 {
            let (s, p, o) = if rng.next_f64() < 0.5 {
                triples[rng.next_range(triples.len() as u64) as usize]
            } else {
                (
                    rng.next_range(NUM_S as u64) as u32,
                    rng.next_range(NUM_P as u64) as u32,
                    rng.next_range(NUM_O as u64) as u32,
                )
            };
            if tree.contains(s, p, o) != naive.contains(s, p, o) {
                mismatches += 1;
                eprintln!("  MISMATCH (static) contains({s},{p},{o})");
            }
            if dyn_tree.contains(s, p, o) != naive.contains(s, p, o) {
                mismatches += 1;
                eprintln!("  MISMATCH (dynamic) contains({s},{p},{o})");
            }
        }
        assert_eq!(mismatches, 0, "RadixTree disagreed with NaiveIndex on {mismatches} probes");
        println!("  500 randomized contains() probes (both static and dynamic): all agree with NaiveIndex");

        let (s0, p0, _) = triples[0];
        let row = tree.query_two_bound([0, 1], [s0, p0], 2);
        println!("  query_two_bound(S={s0}, P={p0}, free=O) = {row:?}");
        let unbound = tree.query_one_bound(0, s0, 2);
        println!("  query_one_bound(S={s0}, free=O) -> {} objects", unbound.len());
    }
}
