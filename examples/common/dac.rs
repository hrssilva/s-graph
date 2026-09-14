//! Directly Addressable Codes (DACs) [Brisaboa, Ladra, Navarro, "DACs:
//! Bringing direct access to variable-length codes", IPM 2013 -- the same
//! research group behind the Mk²/IK²-tree structures this project compares
//! against]: a variable-length integer encoding that supports true random
//! access to the i-th value in O(levels) time, *without* an auxiliary
//! offset array and without scanning from the start.
//!
//! # Why not a plain varint (LEB128) stream
//!
//! A varint stream is exactly as compact per value, but locating element
//! `i` requires knowing the byte length of every element before it --
//! each value's length is data-dependent, only knowable by reading it. An
//! auxiliary offset array would fix that, but costs O(n) space itself,
//! defeating the point of using a variable-width encoding to save memory.
//!
//! # The idea
//!
//! Split every value into fixed-size chunks ([`CHUNK_BITS`] each, least-
//! significant first -- exactly what a varint does), but instead of
//! storing each value's chunks *consecutively*, store them by *level*:
//! `levels[0]` holds every value's first chunk, back to back, in original
//! order -- so `levels[0][i]` is always at index `i`, no lookup needed. A
//! companion bitmap `continues[0]` marks which of those values needed a
//! second chunk. `levels[1]` holds *only* those values' second chunks, in
//! the same relative order -- so a value's position within `levels[1]` is
//! `continues[0].rank1(i)`, one rank operation, not a scan. This repeats
//! per level: `levels[k]`'s existence for a given original index is
//! entirely determined by chasing `continues[0..k]`'s bits and ranks.
//!
//! Reading element `i`: read `levels[0][i]`; if `continues[0].get(i)` is
//! false, that's the whole value. Otherwise `i = continues[0].rank1(i)`,
//! read `levels[1][i]`, check `continues[1].get(i)`, and so on. Cost is
//! O(levels actually present for this value) -- a value that fits in one
//! chunk costs one chunk read and one bit check, nothing more.
//!
//! # Why this beats a fixed-width field
//!
//! A fixed-width `u32` (as [`crate::common::radixtree_dfs`]'s
//! `subtree_word_count` uses) pays 4 bytes per entry regardless of
//! magnitude. Most entries in that specific use (a far node's own subtree
//! word count) are small -- DACs let those cost as little as ~1.1 bytes
//! (one chunk plus one amortized continuation bit) while the rare large
//! ones (subtrees near the root) still get however many chunks they
//! actually need. [`crate::common::radixtree_dfs_dac`] swaps exactly that
//! one field to measure the real memory/navigation-time tradeoff.

use crate::common::bitvector::BitVector;

pub const CHUNK_BITS: u32 = 8;
const CHUNK_MASK: u64 = (1u64 << CHUNK_BITS) - 1;

#[derive(Default)]
pub struct DacArray {
    /// `levels[k][j]` is the k-th (0-indexed) chunk of whichever original
    /// value is the j-th one still "alive" at level k -- see the module
    /// doc comment. `levels[0]` always has one entry per original value;
    /// each later level is a subsequence of the previous one's survivors.
    levels: Vec<Vec<u8>>,
    /// `continues[k]`, index-aligned with `levels[k]`: whether that
    /// value needs a `levels[k+1]` entry. One shorter than `levels` --
    /// the last level's every value is known (by construction: nothing
    /// survived past it) to not continue, so no bitmap is stored for it.
    continues: Vec<BitVector>,
}

impl DacArray {
    pub fn build(values: &[u32]) -> Self {
        let mut levels: Vec<Vec<u8>> = Vec::new();
        let mut continues: Vec<BitVector> = Vec::new();
        let mut active: Vec<u64> = values.iter().map(|&v| v as u64).collect();
        while !active.is_empty() {
            let mut level = Vec::with_capacity(active.len());
            let mut cont = BitVector::with_capacity(active.len());
            let mut next_active = Vec::with_capacity(active.len());
            for &v in &active {
                level.push((v & CHUNK_MASK) as u8);
                let rest = v >> CHUNK_BITS;
                let has_more = rest != 0;
                cont.push(has_more);
                if has_more {
                    next_active.push(rest);
                }
            }
            cont.finalize();
            levels.push(level);
            continues.push(cont);
            active = next_active;
        }
        // The bitmap pushed for the level that turned out to be the last
        // one is entirely `false` by construction (nothing survived past
        // it) and is never consulted by `get` (see the guard there) --
        // drop it rather than storing dead weight.
        continues.pop();
        DacArray { levels, continues }
    }

    pub fn len(&self) -> usize {
        self.levels.first().map_or(0, |l0| l0.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, mut i: usize) -> u32 {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        for level in 0..self.levels.len() {
            value |= (self.levels[level][i] as u64) << shift;
            if level == self.continues.len() || !self.continues[level].get(i as u64) {
                break;
            }
            i = self.continues[level].rank1(i as u64) as usize;
            shift += CHUNK_BITS;
        }
        value as u32
    }

    pub fn mem_size_bytes(&self) -> usize {
        self.levels.iter().map(crate::common::metrics::vec_bytes).sum::<usize>() + self.continues.iter().map(BitVector::mem_size_bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(values: &[u32]) {
        let dac = DacArray::build(values);
        assert_eq!(dac.len(), values.len());
        for (i, &expect) in values.iter().enumerate() {
            assert_eq!(dac.get(i), expect, "get({i}) mismatch, values={values:?}");
        }
    }

    #[test]
    fn empty() {
        check(&[]);
    }

    #[test]
    fn all_fit_one_chunk() {
        check(&[0, 1, 5, 255, 254, 3, 0, 200]);
    }

    #[test]
    fn all_need_full_width() {
        check(&[u32::MAX, u32::MAX - 1, 0xDEADBEEF, 0x12345678]);
    }

    #[test]
    fn mixed_magnitudes() {
        check(&[0, 255, 256, 65535, 65536, 1, u32::MAX, 300, 70000, 16_777_215, 16_777_216]);
    }

    #[test]
    fn single_value_each_chunk_boundary() {
        for &v in &[0u32, 1, 255, 256, 257, 65535, 65536, 65537, 16_777_215, 16_777_216, 16_777_217, u32::MAX] {
            check(&[v]);
        }
    }

    #[test]
    fn pseudo_random_mixed() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let values: Vec<u32> = (0..3000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Skew toward small values (like real subtree word
                // counts) with an occasional large one, rather than
                // uniform over the full u32 range -- exercises the
                // "most values stay at level 0-1" common case alongside
                // the rare deep one.
                match x % 20 {
                    0 => (x >> 3) as u32,
                    1..=2 => ((x >> 3) % 65536) as u32,
                    _ => ((x >> 3) % 256) as u32,
                }
            })
            .collect();
        check(&values);
    }

    #[test]
    fn all_zero() {
        check(&[0; 500]);
    }
}
