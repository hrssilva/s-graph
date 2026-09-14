//! Thin standalone demo for IK²-tree: builds a small synthetic RDF dataset,
//! runs bound- and unbound-predicate queries, cross-checks every result
//! against a brute-force [`common::reference::NaiveIndex`], and separately
//! cross-checks the eager and lazy unbound-predicate decode paths against
//! each other -- they must agree exactly, since lazy decode is only a
//! different *strategy* for recovering the same predicate ids.

#[path = "common/mod.rs"]
mod common;

use common::ik2tree::Ik2Tree;
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
        let tree = Ik2Tree::build(8, &triples); // k=8: 64-way branching, matching RadixTree's node granularity
        println!(
            "  Ik2Tree: {} predicates, {} logical triples, {} allocated",
            tree.num_predicates(),
            naive.len(),
            human_bytes(tree.mem_size_bytes())
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
            if tree.locate(p, s, o) != naive.contains(s, p, o) {
                mismatches += 1;
                eprintln!("  MISMATCH locate({p},{s},{o})");
            }
        }
        assert_eq!(mismatches, 0, "Ik2Tree disagreed with NaiveIndex on {mismatches} probes");
        println!("  500 randomized locate() probes: all agree with NaiveIndex");

        let (s0, p0, _) = triples[0];
        let row = tree.row_bound(p0, s0);
        println!("  row_bound(p={p0}, s={s0}) = {row:?}");

        let mut eager = tree.row_unbound_eager(s0);
        let mut lazy = tree.row_unbound_lazy(s0);
        eager.sort_unstable();
        lazy.sort_unstable();
        assert_eq!(eager, lazy, "eager and lazy unbound-predicate decode disagree");
        let mut expected: Vec<(u32, u32)> = (0..NUM_P).flat_map(|p| naive.row(p, s0).into_iter().map(move |o| (p, o))).collect();
        expected.sort_unstable();
        assert_eq!(eager, expected, "unbound-predicate row(s={s0}) disagrees with NaiveIndex");
        println!("  row_unbound(s={s0}, ?, ?) -> {} (predicate,object) pairs; eager and lazy agree, match NaiveIndex", eager.len());
    }
}
