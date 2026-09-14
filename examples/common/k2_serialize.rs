//! Shared binary serialization for the k²-tree family
//! (`k2tree.rs`'s [`K2Tree`], reused by `mk2tree.rs`'s `Mk2Forest` and
//! `ik2tree.rs`'s `Ik2Tree`) -- specifically, the one piece every one of
//! them needs: writing/reading a [`BitVector`]'s raw bits.
//!
//! Only `words` and `len_bits` are ever written; `rank_samples` (the
//! sampled cumulative-popcount index `finalize()` builds) is never
//! persisted -- it's rebuilt with one linear pass over `words` on load
//! (see [`BitVector::from_words`]), the same choice
//! `tree_serialize.rs`'s own doc comment explains for
//! `RadixTreeDfsInterleavedVarSkipUnrolled`'s per-level caches: cheap to
//! recompute, and a stored copy could only ever be a second, driftable
//! source of truth for the same data.
//!
//! Like `tree_serialize.rs`, reads are streaming (generic over any
//! `Read`, so `Mk2Forest::deserialize_from_path`/`Ik2Tree::deserialize_from_path`
//! never need the whole serialized file resident as one buffer -- these
//! structures can run into the hundreds of MiB on this project's
//! large-scale scenarios). Writes still go through an in-memory
//! `Vec<u8>` (via the `write_*` free functions this module reuses from
//! `tree_serialize.rs`), matching that module's own convention: the
//! *build* phase's transient peak is already dominated by the raw input
//! dataset and build-time scratch buffers, both far larger than the
//! serialized output, so a streaming write wouldn't move that number.

use crate::common::bitvector::BitVector;
use crate::common::tree_serialize::{read_u64_from, write_u64};

/// Appends `bv`'s raw bits to `out`: `len_bits: u64`, then one `u64` per
/// word (`len_bits.div_ceil(64)` of them).
pub fn write_bitvector(out: &mut Vec<u8>, bv: &BitVector) {
    write_u64(out, bv.len());
    let n_words = bv.len().div_ceil(64);
    for i in 0..n_words {
        write_u64(out, bv.word_at(i * 64));
    }
}

/// Reads a `BitVector` written by [`write_bitvector`], reconstructing
/// `rank_samples` via [`BitVector::from_words`].
pub fn read_bitvector_from<R: std::io::Read>(r: &mut R) -> BitVector {
    let len_bits = read_u64_from(r);
    let n_words = len_bits.div_ceil(64) as usize;
    let mut words = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        words.push(read_u64_from(r));
    }
    BitVector::from_words(words, len_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_various_densities_and_lengths() {
        for &len in &[0u64, 1, 63, 64, 65, 511, 512, 513, 1000, 4096] {
            for density in [0.0, 0.1, 0.5, 0.9, 1.0] {
                let mut bv = BitVector::new();
                let mut rng_state = 0x1234_5678_u64.wrapping_add(len).wrapping_add((density * 1000.0) as u64);
                for _ in 0..len {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let bit = ((rng_state >> 33) as f64 / u32::MAX as f64) < density;
                    bv.push(bit);
                }
                bv.finalize();

                let mut out = Vec::new();
                write_bitvector(&mut out, &bv);
                let restored = read_bitvector_from(&mut std::io::Cursor::new(&out[..]));

                assert_eq!(restored.len(), bv.len(), "len mismatch at len={len} density={density}");
                for i in 0..len {
                    assert_eq!(restored.get(i), bv.get(i), "bit {i} mismatch at len={len} density={density}");
                }
                assert_eq!(restored.count_ones(), bv.count_ones(), "count_ones mismatch at len={len} density={density}");
                for k in 0..restored.count_ones() {
                    assert_eq!(restored.select1(k), bv.select1(k), "select1({k}) mismatch at len={len} density={density}");
                }
            }
        }
    }
}
