//! A rank/select-augmented bitvector, laid out as flat arrays (Structure of
//! Arrays): the raw bits live in one `Vec<u64>`, and a sampled cumulative
//! popcount index (one entry every [`SAMPLE_WORDS`] words) supports
//! near-O(1) `rank1`. `select1` binary-searches that same sample array and
//! then scans at most `SAMPLE_WORDS` words.
//!
//! This is the shared substrate for every tree in this comparison: k²-tree's
//! `T`/`L` topology bitmaps, IK²-tree's per-node variable-width predicate
//! groups, and Omnitree's leaf marker bitmap are all instances of this one
//! type.
//!
//! Convention used everywhere downstream: `rank1(i)` counts the 1-bits in
//! the half-open range `[0, i)` -- i.e. it is *exclusive* at `i`. Getting
//! this backwards by one is the single most common bug in k²-tree-family
//! code, so every navigation routine that uses it says so explicitly.

const SAMPLE_WORDS: usize = 8; // 8 * 64 = 512 bits per sample block

#[derive(Clone, Debug, Default)]
pub struct BitVector {
    words: Vec<u64>,
    len_bits: u64,
    // Cumulative popcount at the start of each sample block, plus one
    // trailing sentinel equal to the total popcount. Empty until
    // `finalize()` is called.
    rank_samples: Vec<u64>,
}

impl BitVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(nbits: usize) -> Self {
        BitVector {
            words: Vec::with_capacity(nbits.div_ceil(64)),
            len_bits: 0,
            rank_samples: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, bit: bool) {
        let word_idx = (self.len_bits / 64) as usize;
        if word_idx == self.words.len() {
            self.words.push(0);
        }
        if bit {
            self.words[word_idx] |= 1u64 << (self.len_bits % 64);
        }
        self.len_bits += 1;
    }

    /// Appends a full 64-bit word in one step -- for callers that already
    /// have a whole node's worth of bits assembled (e.g. `RadixTree`'s
    /// one-word-per-node layout), this replaces 64 individual `push` calls
    /// (each paying a division, a modulo, a growth check, and a bit-or)
    /// with a single `Vec::push`. Requires being called at a word boundary
    /// (`len_bits % 64 == 0`), which every one-word-per-node caller
    /// naturally maintains between nodes.
    #[inline]
    pub fn push_word(&mut self, word: u64) {
        debug_assert_eq!(self.len_bits % 64, 0, "push_word requires a word-aligned position");
        self.words.push(word);
        self.len_bits += 64;
    }

    #[inline]
    pub fn get(&self, i: u64) -> bool {
        debug_assert!(i < self.len_bits, "bit index {i} out of bounds ({})", self.len_bits);
        let word = self.words[(i / 64) as usize];
        (word >> (i % 64)) & 1 == 1
    }

    pub fn len(&self) -> u64 {
        self.len_bits
    }

    /// The raw 64-bit word starting at `bit_offset`, which must be a
    /// multiple of 64 (i.e. word-aligned) -- callers that pack one fixed-
    /// size unit per word (e.g. `RadixTree`'s one-node-per-word layout) can
    /// use this to inspect/mask a whole node in a single load, instead of
    /// bit-by-bit `get` calls.
    pub fn word_at(&self, bit_offset: u64) -> u64 {
        debug_assert_eq!(bit_offset % 64, 0, "word_at requires a word-aligned offset");
        self.words[(bit_offset / 64) as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    /// Reconstructs a `BitVector` from its raw words and bit-length,
    /// finalizing (rebuilding the rank-sample index) immediately -- the
    /// deserialization counterpart to `push`-based construction.
    /// `rank_samples` is never itself stored on disk (see
    /// `crate::common::k2_serialize`'s doc comment): recomputing it here
    /// with one linear pass over `words` is cheap and can't drift out of
    /// sync with a stored copy the way persisting it separately could.
    pub fn from_words(words: Vec<u64>, len_bits: u64) -> Self {
        let mut bv = BitVector { words, len_bits, rank_samples: Vec::new() };
        bv.finalize();
        bv
    }

    /// Builds the rank-sample index. Must be called once after all `push`es
    /// and before any `rank1`/`select1`/`count_ones` call.
    pub fn finalize(&mut self) {
        // `push`-based construction (no upfront `with_capacity`) grows
        // `words` via amortized doubling, which can leave up to ~2x more
        // capacity allocated than the final logical size needs -- and
        // since `MemSize` deliberately reports actual allocated bytes (not
        // logical length), that slack shows up directly in every memory
        // comparison this structure feeds. Trim it now that no more
        // `push`es are coming.
        self.words.shrink_to_fit();
        let n_blocks = self.words.len().div_ceil(SAMPLE_WORDS).max(1);
        self.rank_samples = Vec::with_capacity(n_blocks + 1);
        let mut cum: u64 = 0;
        for block in self.words.chunks(SAMPLE_WORDS) {
            self.rank_samples.push(cum);
            cum += block.iter().map(|w| w.count_ones() as u64).sum::<u64>();
        }
        if self.rank_samples.is_empty() {
            self.rank_samples.push(0);
        }
        self.rank_samples.push(cum); // sentinel = total popcount
    }

    #[inline]
    pub fn count_ones(&self) -> u64 {
        *self.rank_samples.last().unwrap_or(&0)
    }

    /// Number of 1-bits in `[0, i)`. Exclusive at `i`.
    pub fn rank1(&self, i: u64) -> u64 {
        debug_assert!(i <= self.len_bits, "rank index {i} out of bounds ({})", self.len_bits);
        if i == 0 {
            return 0;
        }
        let word_idx = (i / 64) as usize; // word containing bit i-1 (partially, if bit_off > 0)
        let block_idx = word_idx / SAMPLE_WORDS;
        let mut r = self.rank_samples[block_idx];
        let block_start_word = block_idx * SAMPLE_WORDS;
        for w in block_start_word..word_idx {
            r += self.words[w].count_ones() as u64;
        }
        let bit_off = i % 64;
        if bit_off > 0 {
            let mask = (1u64 << bit_off) - 1;
            r += (self.words[word_idx] & mask).count_ones() as u64;
        }
        r
    }

    /// Position of the `k`-th (0-indexed) 1-bit, or `None` if there are
    /// fewer than `k + 1` set bits.
    pub fn select1(&self, k: u64) -> Option<u64> {
        if k >= self.count_ones() {
            return None;
        }
        // Binary search the largest block_idx with rank_samples[block_idx] <= k.
        let mut lo = 0usize;
        let mut hi = self.rank_samples.len() - 1; // rank_samples has n_blocks+1 entries incl. sentinel
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.rank_samples[mid] <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let block_idx = lo;
        let mut remaining = k - self.rank_samples[block_idx];
        let start_word = block_idx * SAMPLE_WORDS;
        let end_word = (start_word + SAMPLE_WORDS).min(self.words.len());
        for w in start_word..end_word {
            let word = self.words[w];
            let pc = word.count_ones() as u64;
            if remaining < pc {
                let mut bits = word;
                for _ in 0..remaining {
                    bits &= bits - 1; // clear lowest set bit
                }
                return Some((w as u64) * 64 + bits.trailing_zeros() as u64);
            }
            remaining -= pc;
        }
        None
    }

    pub fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.words) + crate::common::metrics::vec_bytes(&self.rank_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_rank1(bits: &[bool], i: u64) -> u64 {
        bits[..i as usize].iter().filter(|&&b| b).count() as u64
    }

    fn naive_select1(bits: &[bool], k: u64) -> Option<u64> {
        bits.iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .nth(k as usize)
            .map(|(i, _)| i as u64)
    }

    fn build(bits: &[bool]) -> BitVector {
        let mut bv = BitVector::with_capacity(bits.len());
        for &b in bits {
            bv.push(b);
        }
        bv.finalize();
        bv
    }

    fn check_pattern(bits: Vec<bool>) {
        let bv = build(&bits);
        assert_eq!(bv.len(), bits.len() as u64);
        for i in 0..=bits.len() as u64 {
            assert_eq!(bv.rank1(i), naive_rank1(&bits, i), "rank1({i}) mismatch, len={}", bits.len());
        }
        let ones = bits.iter().filter(|&&b| b).count() as u64;
        assert_eq!(bv.count_ones(), ones);
        for k in 0..ones + 2 {
            assert_eq!(bv.select1(k), naive_select1(&bits, k), "select1({k}) mismatch, len={}", bits.len());
        }
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(bv.get(i as u64), b);
        }
    }

    #[test]
    fn edge_lengths_and_densities() {
        let lengths = [0usize, 1, 63, 64, 65, 511, 512, 513, 1000, 1537];
        for &len in &lengths {
            check_pattern(vec![false; len]); // all-0
            check_pattern(vec![true; len]); // all-1
            check_pattern((0..len).map(|i| i % 2 == 0).collect()); // alternating
            check_pattern((0..len).map(|i| i % 17 == 0).collect()); // sparse
        }
    }

    #[test]
    fn pseudo_random_pattern() {
        let mut x: u64 = 0x243F6A8885A308D3;
        let bits: Vec<bool> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x & 1 == 1
            })
            .collect();
        check_pattern(bits);
    }
}
