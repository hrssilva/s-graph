//! A deliberately naive, `HashSet`-backed triple index used purely as a
//! correctness oracle. None of the compact structures under comparison are
//! checked against an external reference implementation (none is readily
//! available), so every non-trivial query, join, and path algorithm in this
//! comparison is instead cross-checked against this brute-force index over
//! the same synthetic dataset.

use std::collections::HashSet;

pub struct NaiveIndex {
    triples: HashSet<(u32, u32, u32)>,
}

impl NaiveIndex {
    pub fn build(triples: &[(u32, u32, u32)]) -> Self {
        NaiveIndex {
            triples: triples.iter().copied().collect(),
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        self.triples.contains(&(s, p, o))
    }

    /// Sorted, deduplicated objects for a fixed (predicate, subject).
    pub fn row(&self, p: u32, s: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .triples
            .iter()
            .filter(|&&(ts, tp, _)| ts == s && tp == p)
            .map(|&(_, _, to)| to)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Sorted, deduplicated subjects for a fixed (predicate, object).
    pub fn col(&self, p: u32, o: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .triples
            .iter()
            .filter(|&&(_, tp, to)| to == o && tp == p)
            .map(|&(ts, _, _)| ts)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// All (subject, object) pairs across every predicate, optionally
    /// filtered by a bound subject and/or object. Used to check
    /// unbound-predicate queries such as `(s,?,?)`, `(?,?,o)`, `(?,?,?)`.
    pub fn scan_all(&self, s: Option<u32>, o: Option<u32>) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = self
            .triples
            .iter()
            .filter(|&&(ts, _, to)| s.is_none_or(|s| s == ts) && o.is_none_or(|o| o == to))
            .map(|&(ts, _, to)| (ts, to))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Sorted, deduplicated predicates linking a specific (subject, object)
    /// pair -- "how are these two entities related?", the query shape a
    /// genuinely 3D-symmetric structure (Omnitree, RadixTree) can answer
    /// natively by binding S and O and leaving P free, unlike the
    /// k²-tree-family structures (fundamentally 2D over S,O, with predicate
    /// selecting *which* tree/interleaving-context rather than a third
    /// traversal axis).
    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .triples
            .iter()
            .filter(|&&(ts, _, to)| ts == s && to == o)
            .map(|&(_, tp, _)| tp)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Sorted, deduplicated out-neighbors of `node` (as subject), optionally
    /// restricted to one predicate. Used as the BFS reference.
    pub fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .triples
            .iter()
            .filter(|&&(ts, tp, _)| ts == node && predicate.is_none_or(|p| p == tp))
            .map(|&(_, _, to)| to)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn len(&self) -> usize {
        self.triples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}
