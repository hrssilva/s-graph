//! Test-only dataset generation and a brute-force correctness oracle.
//! Not part of the public API -- compiled only under `#[cfg(test)]`.

use crate::Triple;

pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)`.
    pub fn next_range(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Pure uniform random triples over `[0,num_s) x [0,num_p) x [0,num_o)`.
/// May contain duplicates (harmless -- the tree treats the triple set as
/// a set).
pub fn uniform(num_s: u32, num_p: u32, num_o: u32, num_triples: usize, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    (0..num_triples).map(|_| (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32, rng.next_range(num_o as u64) as u32)).collect()
}

fn zipfian_cumulative(n: u32, exponent: f64) -> (Vec<f64>, f64) {
    let mut cumulative = Vec::with_capacity(n as usize);
    let mut total = 0.0f64;
    for rank in 1..=n as u64 {
        total += 1.0 / (rank as f64).powf(exponent);
        cumulative.push(total);
    }
    (cumulative, total)
}

fn sample_zipfian(rng: &mut SplitMix64, cumulative: &[f64], total: f64, n: u32) -> u32 {
    let target = rng.next_f64() * total;
    match cumulative.binary_search_by(|probe| probe.partial_cmp(&target).unwrap()) {
        Ok(i) | Err(i) => i.min(n as usize - 1) as u32,
    }
}

/// Subjects/objects uniform; predicate drawn from a Zipfian distribution
/// (`weight(rank) = 1/rank^exponent`) over `num_p` predicate ids, so a
/// small number of predicates account for most triples -- mirrors real
/// RDF's heavy predicate skew.
pub fn zipfian_predicates(num_s: u32, num_p: u32, num_o: u32, num_triples: usize, exponent: f64, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let (cumulative, total) = zipfian_cumulative(num_p, exponent);
    (0..num_triples).map(|_| (rng.next_range(num_s as u64) as u32, sample_zipfian(&mut rng, &cumulative, total, num_p), rng.next_range(num_o as u64) as u32)).collect()
}

/// A deliberately naive, `HashSet`-backed triple index used purely as a
/// correctness oracle: every non-trivial query in this crate's test
/// suite is cross-checked against this brute-force index over the same
/// synthetic dataset, since no external reference implementation is
/// available.
pub struct NaiveIndex {
    triples: std::collections::HashSet<Triple>,
}

impl NaiveIndex {
    pub fn build(triples: &[Triple]) -> Self {
        NaiveIndex { triples: triples.iter().copied().collect() }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        self.triples.contains(&(s, p, o))
    }

    /// Mirrors `Heap::remove`'s return convention: `true` if it was
    /// present and is now removed, `false` if it was already absent.
    pub fn remove(&mut self, s: u32, p: u32, o: u32) -> bool {
        self.triples.remove(&(s, p, o))
    }

    /// Sorted, deduplicated objects for a fixed (predicate, subject).
    pub fn row(&self, p: u32, s: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self.triples.iter().filter(|&&(ts, tp, _)| ts == s && tp == p).map(|&(_, _, to)| to).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Sorted, deduplicated subjects for a fixed (predicate, object).
    pub fn col(&self, p: u32, o: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self.triples.iter().filter(|&&(_, tp, to)| to == o && tp == p).map(|&(ts, _, _)| ts).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Sorted, deduplicated out-neighbors of `node` (as subject),
    /// optionally restricted to one predicate. Used as the BFS reference.
    pub fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        let mut v: Vec<u32> = self.triples.iter().filter(|&&(ts, tp, _)| ts == node && predicate.is_none_or(|p| p == tp)).map(|&(_, _, to)| to).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}
