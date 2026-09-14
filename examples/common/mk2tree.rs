//! Mk²-tree: one independent [`K2Tree`] per RDF predicate, over the
//! `(subject, object)` adjacency matrix for that predicate.
//!
//! Source: Álvarez-García, Brisaboa, Fernández, Martínez-Prieto,
//! "Compressed k²-triples for full-in-memory RDF engines" (AMCIS 2011);
//! also described as the baseline compared against interleaving in García,
//! Brisaboa, de Bernardo, Navarro, "Interleaved K²-tree: Indexing and
//! Navigating Ternary Relations" (DCC 2014).
//!
//! Bound-predicate queries (`(s,p,o)`, `(s,p,?)`, `(?,p,o)`, `(?,p,?)`)
//! dispatch directly to `trees[p]` in O(1). Unbound-predicate queries
//! (`(s,?,?)`, `(?,?,o)`, `(?,?,?)`) must scan every predicate's tree --
//! this is the structure's known weakness on datasets with many predicates,
//! but the trees are fully independent, so the scan is embarrassingly
//! parallel; both a sequential and a `rayon`-parallel scan are provided so
//! the comparison harness can measure the speedup directly.

use crate::common::k2tree::K2Tree;
use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;
use rayon::prelude::*;

/// `k=8` gives `k^2=64` children per node -- exactly one `u64` word, the
/// same node granularity `RadixTree` uses, so `rank1` reduces to a single
/// hardware `popcnt` over one word here too (each node's 64 existence bits
/// land on a natural word boundary in the shared `BitVector` type). This
/// makes the comparison apples-to-apples rather than comparing a 4-way
/// (`k=2`) tree's rank mechanism against a 64-way one's.
const K: u32 = 8;

pub struct Mk2Forest {
    // Indexed by predicate id; `None` for predicate ids with zero triples
    // (keeps predicate ids as plain indices without needing a HashMap).
    trees: Vec<Option<K2Tree>>,
}

impl Mk2Forest {
    pub fn build(triples: &[Triple]) -> Self {
        let num_predicates = triples.iter().map(|&(_, p, _)| p).max().map(|m| m + 1).unwrap_or(0);
        let mut by_predicate: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_predicates as usize];
        for &(s, p, o) in triples {
            by_predicate[p as usize].push((s, o));
        }
        let trees = by_predicate
            .into_iter()
            .map(|mut pts| if pts.is_empty() { None } else { Some(K2Tree::build(K, &mut pts)) })
            .collect();
        Mk2Forest { trees }
    }

    pub fn num_predicates(&self) -> usize {
        self.trees.len()
    }

    pub fn tree_for(&self, p: u32) -> Option<&K2Tree> {
        self.trees.get(p as usize).and_then(|t| t.as_ref())
    }

    /// Largest matrix side across every predicate's tree, i.e. an upper
    /// bound on any node id that can appear in a query result -- needed by
    /// path algorithms to size a global distance array.
    pub fn node_upper_bound(&self) -> usize {
        self.trees.iter().flatten().map(|t| t.n() as usize).max().unwrap_or(0)
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        self.tree_for(p).is_some_and(|t| t.contains(s, o))
    }

    /// Objects `o` such that `(s,p,o)` is present, ascending. Uses
    /// `K2Tree::row_vec` (plain recursive collection, no `Iterator`/seek
    /// overhead) since this is a one-shot call, not a leapfrog-join stream.
    pub fn row(&self, p: u32, s: u32) -> Vec<u32> {
        self.tree_for(p).map(|t| t.row_vec(s)).unwrap_or_default()
    }

    /// Subjects `s` such that `(s,p,o)` is present, ascending.
    pub fn col(&self, p: u32, o: u32) -> Vec<u32> {
        self.tree_for(p).map(|t| t.col_vec(o)).unwrap_or_default()
    }

    /// All `(s,o)` pairs across every predicate, sequential scan. Used for
    /// unbound-predicate queries such as `(s,?,?)`/`(?,?,o)`/`(?,?,?)`.
    pub fn scan_all_seq(&self, s: Option<u32>, o: Option<u32>) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for tree in self.trees.iter().flatten() {
            collect_pairs(tree, s, o, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same as [`Self::scan_all_seq`] but scans predicate trees in
    /// parallel via `rayon` -- the natural parallel variant for Mk²-tree,
    /// since predicate trees are fully independent.
    pub fn scan_all_parallel(&self, s: Option<u32>, o: Option<u32>) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .trees
            .par_iter()
            .flatten()
            .fold(Vec::new, |mut acc, tree| {
                collect_pairs(tree, s, o, &mut acc);
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Ascending predicates linking a specific (subject, object) pair.
    /// Unlike `RadixTree`/`Omnitree` (genuinely 3D-symmetric structures,
    /// where this is just a two-bound-dims query), Mk2Forest has no
    /// traversal axis for "any predicate" at all -- predicate selects
    /// *which* independent tree to use, so answering this means checking
    /// every predicate's tree individually. O(num_predicates), same
    /// structural cost as the other unbound-predicate queries; sequential
    /// and `rayon`-parallel variants for the same reason those exist.
    pub fn predicates_for_seq(&self, s: u32, o: u32) -> Vec<u32> {
        self.trees
            .iter()
            .enumerate()
            .filter_map(|(p, t)| t.as_ref().filter(|t| t.contains(s, o)).map(|_| p as u32))
            .collect()
    }

    pub fn predicates_for_parallel(&self, s: u32, o: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .trees
            .par_iter()
            .enumerate()
            .filter_map(|(p, t)| t.as_ref().filter(|t| t.contains(s, o)).map(|_| p as u32))
            .collect();
        v.sort_unstable();
        v
    }
}

fn collect_pairs(tree: &K2Tree, s: Option<u32>, o: Option<u32>, out: &mut Vec<(u32, u32)>) {
    match (s, o) {
        (Some(s), Some(o)) => {
            if tree.contains(s, o) {
                out.push((s, o));
            }
        }
        (Some(s), None) => out.extend(tree.row_vec(s).into_iter().map(|o| (s, o))),
        (None, Some(o)) => out.extend(tree.col_vec(o).into_iter().map(|s| (s, o))),
        (None, None) => {
            for s in 0..tree.n() as u32 {
                out.extend(tree.row_vec(s).into_iter().map(|o| (s, o)));
            }
        }
    }
}

impl MemSize for Mk2Forest {
    fn mem_size_bytes(&self) -> usize {
        self.trees.iter().flatten().map(|t| t.mem_size_bytes()).sum()
    }
}

impl Mk2Forest {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        let t: usize = self.trees.iter().flatten().map(|t| t.size_breakdown()[0].1).sum();
        let l: usize = self.trees.iter().flatten().map(|t| t.size_breakdown()[1].1).sum();
        vec![("t (summed over all predicate trees)", t), ("l (summed over all predicate trees)", l)]
    }
}

const MK2_MAGIC: &[u8; 4] = b"MK2A";

impl Mk2Forest {
    /// Serializes this forest to a self-contained byte buffer: a magic
    /// tag, the predicate-slot count, then one presence byte (`0`/`1`)
    /// per slot, each `1` immediately followed by that predicate's
    /// [`K2Tree::write_into`] bytes. See `crate::common::k2_serialize`'s
    /// doc comment for why writes still buffer into a `Vec<u8>` while
    /// reads (below) stream directly.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MK2_MAGIC);
        crate::common::tree_serialize::write_u64(&mut out, self.trees.len() as u64);
        for tree in &self.trees {
            match tree {
                Some(t) => {
                    out.push(1);
                    t.write_into(&mut out);
                }
                None => out.push(0),
            }
        }
        out
    }

    /// Reconstructs a forest from bytes already in memory (written by
    /// [`Self::serialize`]) -- a thin wrapper over
    /// [`Self::deserialize_from_reader`] via [`std::io::Cursor`]. Prefer
    /// [`Self::deserialize_from_path`] when reading a file.
    pub fn deserialize(buf: &[u8]) -> Self {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf))
    }

    /// Reconstructs a forest by reading directly from a file written by
    /// [`Self::serialize`] -- never materializes the whole serialized
    /// content as one in-memory buffer.
    pub fn deserialize_from_path(path: &std::path::Path) -> Self {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path:?}: {e}"));
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file))
    }

    fn deserialize_from_reader<R: std::io::Read>(r: &mut R) -> Self {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).expect("unexpected EOF reading Mk2Forest magic -- truncated or corrupt file");
        assert_eq!(&magic, MK2_MAGIC, "not a serialized Mk2Forest (bad magic)");
        let count = crate::common::tree_serialize::read_u64_from(r) as usize;
        let mut trees = Vec::with_capacity(count);
        let mut flag = [0u8; 1];
        for _ in 0..count {
            r.read_exact(&mut flag).expect("unexpected EOF reading Mk2Forest presence flag");
            trees.push(if flag[0] == 1 { Some(K2Tree::read_from(r)) } else { None });
        }
        Mk2Forest { trees }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::uniform;
    use crate::common::reference::NaiveIndex;

    #[test]
    fn cross_check_against_naive() {
        for seed in 1..5u64 {
            let triples = uniform(12, 4, 12, 200, seed * 101);
            let naive = NaiveIndex::build(&triples);
            let forest = Mk2Forest::build(&triples);

            for s in 0..12u32 {
                for p in 0..4u32 {
                    for o in 0..12u32 {
                        assert_eq!(forest.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o})");
                    }
                    assert_eq!(forest.row(p, s), naive.row(p, s), "row(p={p},s={s})");
                }
                for o in 0..12u32 {
                    for p in 0..4u32 {
                        assert_eq!(forest.col(p, o), naive.col(p, o), "col(p={p},o={o})");
                    }
                    assert_eq!(forest.predicates_for_seq(s, o), naive.predicates_for(s, o), "predicates_for_seq(s={s},o={o})");
                    assert_eq!(forest.predicates_for_parallel(s, o), naive.predicates_for(s, o), "predicates_for_parallel(s={s},o={o})");
                }
            }

            for s in [None, Some(3u32), Some(11)] {
                for o in [None, Some(0u32), Some(7)] {
                    assert_eq!(forest.scan_all_seq(s, o), naive.scan_all(s, o), "scan_all_seq({s:?},{o:?})");
                    assert_eq!(forest.scan_all_parallel(s, o), naive.scan_all(s, o), "scan_all_parallel({s:?},{o:?})");
                }
            }
        }
    }

    #[test]
    fn serialize_round_trip() {
        // Includes a triple domain sparse enough to leave predicate slots
        // with `None` trees (see `Mk2Forest::trees`'s doc comment), so
        // the presence-byte path is exercised too, not just the common
        // case.
        for seed in 1..5u64 {
            let triples = uniform(12, 6, 12, 150, seed * 271);
            let naive = NaiveIndex::build(&triples);
            let forest = Mk2Forest::build(&triples);

            let bytes = forest.serialize();
            let restored = Mk2Forest::deserialize(&bytes);
            assert_eq!(restored.num_predicates(), forest.num_predicates());

            for s in 0..12u32 {
                for p in 0..6u32 {
                    for o in 0..12u32 {
                        assert_eq!(restored.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o}) after round-trip");
                    }
                    assert_eq!(restored.row(p, s), naive.row(p, s), "row(p={p},s={s}) after round-trip");
                }
            }

            // Also round-trip via a real file, exercising
            // `deserialize_from_path`'s streaming read path specifically.
            let path = std::env::temp_dir().join(format!("mk2_round_trip_test_{seed}.bin"));
            std::fs::write(&path, &bytes).unwrap();
            let from_path = Mk2Forest::deserialize_from_path(&path);
            std::fs::remove_file(&path).ok();
            for s in 0..12u32 {
                for p in 0..6u32 {
                    for o in 0..12u32 {
                        assert_eq!(from_path.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o}) after deserialize_from_path");
                    }
                }
            }
        }
    }
}
