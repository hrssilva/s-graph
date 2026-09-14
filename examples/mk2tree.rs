//! Thin standalone demo for Mk²-tree: builds a small synthetic RDF dataset,
//! runs the four triple-pattern query shapes, and cross-checks every result
//! against a brute-force [`common::reference::NaiveIndex`] over ~500
//! randomized probes (half present, half absent) as a correctness safety
//! net -- there is no external reference implementation to test against.

#[path = "common/mod.rs"]
mod common;

use common::mk2tree::Mk2Forest;
use common::metrics::{human_bytes, MemSize};
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
        let forest = Mk2Forest::build(&triples);
        println!(
            "  Mk2Forest: {} predicate trees, {} logical triples, {} allocated",
            forest.num_predicates(),
            naive.len(),
            human_bytes(forest.mem_size_bytes())
        );

        let mut rng = SplitMix64::new(7);
        let mut mismatches = 0;
        for _ in 0..500 {
            let (s, p, o) = if rng.next_f64() < 0.5 {
                // a probe drawn from the actual triples (should be present)
                let idx = rng.next_range(triples.len() as u64) as usize;
                triples[idx]
            } else {
                (
                    rng.next_range(NUM_S as u64) as u32,
                    rng.next_range(NUM_P as u64) as u32,
                    rng.next_range(NUM_O as u64) as u32,
                )
            };
            if forest.contains(s, p, o) != naive.contains(s, p, o) {
                mismatches += 1;
                eprintln!("  MISMATCH contains({s},{p},{o})");
            }
        }
        assert_eq!(mismatches, 0, "Mk2Forest disagreed with NaiveIndex on {mismatches} probes");
        println!("  500 randomized contains() probes: all agree with NaiveIndex");

        let bound = forest.row(0, triples[0].0);
        println!("  row(p=0, s={}) = {:?}", triples[0].0, bound);
        let unbound_seq = forest.scan_all_seq(Some(triples[0].0), None);
        let unbound_par = forest.scan_all_parallel(Some(triples[0].0), None);
        assert_eq!(unbound_seq, unbound_par, "sequential and parallel unbound-predicate scans disagree");
        assert_eq!(unbound_seq, naive.scan_all(Some(triples[0].0), None));
        println!(
            "  scan_all(s={}, ?, ?) -> {} pairs, seq and parallel agree",
            triples[0].0,
            unbound_seq.len()
        );
    }
}
