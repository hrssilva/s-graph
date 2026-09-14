//! A rank-support variant of [`crate::common::radixtree::RadixTree`]:
//! replaces the baseline's sampled-cumulative-popcount `BitVector` rank1
//! (used by `child_word_start` on every non-leaf descent step) with a
//! two-level checkpoint + delta scheme, to shrink T's rank support while
//! keeping true O(1)-ish random-access lookup.
//!
//! # v1 and v2 dead ends (kept here so the reasoning isn't lost)
//!
//! **v1**: one `u8` per *word* (that word's own popcount, 0..64, always
//! fits). Exactly 1 byte/word -- the *same* as the baseline's `u64` every
//! `SAMPLE_WORDS`(8) words (8 bytes / 8 words = 1 byte/word too). Shrinking
//! block size and entry size by the same factor cancels out; no size win.
//!
//! **v2**: `u8` per 3-word *block* (`3*64=192 <= 255`, the largest block
//! whose popcount still fits a byte) -- a real `1 byte / 3 words ≈ 0.33
//! bytes/word`, but computed via a *cursor* `(block_index,
//! cumulative_rank)` carried across a query's calls, summing (or, on
//! backtracking, subtracting) deltas between the cursor's last position
//! and the next target -- betting that consecutive rank1 targets during a
//! monotonic descent are close together. They aren't: BFS/level-order
//! storage means the gap between level L's position and level L+1's is
//! roughly *the width of level L itself*, and in a wide (64-way), shallow
//! (9-level) tree the deepest levels can hold tens of millions of nodes.
//! Measured: 5.47ms per `contains()` call, ~17,000x slower than the
//! baseline's 324ns -- not a reasonable size/time tradeoff, a design flaw.
//! "Monotonic" only bounds *direction*, not *distance between steps*.
//!
//! # v3 (current): checkpoints bound the scan, independent of query pattern
//!
//! Two levels, closer to what the baseline itself does but denser:
//! - **Coarse**: one `u32` absolute cumulative popcount every
//!   [`CHECKPOINT_WORDS`] (63 = 21 * [`BLOCK_WORDS`], chosen for clean
//!   block alignment rather than exactly 64) words. `u32` comfortably
//!   covers cumulative counts into the billions -- `Triple`'s own S/P/O
//!   fields are `u32`, capping any realistic dataset well under that
//!   anyway, so there's no reason to pay for `u64` here.
//! - **Fine**: one `u8` per [`BLOCK_WORDS`]-word block (from v2), used
//!   *only* to fill the gap from the nearest checkpoint to the target --
//!   at most 21 block-sums plus a final <=3-word scan, a small, *fixed*
//!   bound that never depends on where the previous query happened to
//!   land. No cursor, no state carried between calls -- genuine O(1)-ish
//!   random access, same complexity class as the baseline's own scheme.
//!
//! Size: `t_checkpoints` (`T_words/63 * 4` bytes ≈ 0.063 bytes/word) +
//! `t_block_deltas` (`T_words/3` bytes ≈ 0.333 bytes/word) ≈ 0.40
//! bytes/word total -- still a genuine ~2.5x reduction versus the
//! baseline's 1 byte/word, just short of v2's (broken) 3x, in exchange for
//! actually working.
//!
//! # How this differs from the other three structures
//!
//! - **vs. [`crate::common::radixtree_veb`]**: VEB reorders *where* words
//!   physically live but keeps the baseline's rank1 formula and
//!   `rank_samples` completely unchanged. This variant keeps BFS/level
//!   order untouched but replaces the rank *mechanism* itself.
//! - **vs. [`crate::common::radixtree_dfs`]/[`crate::common::radixtree_bp`]**:
//!   those replace rank-based navigation with skip-size-based navigation
//!   (no rank at all). This keeps rank-based navigation, just backed by a
//!   smaller two-level sampled structure instead of the baseline's single-
//!   level one.

use crate::common::metrics::MemSize;
use crate::common::radixtree::RadixTreeApi;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const WORD_BITS: u64 = K_SQUARE as u64;
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;
/// Words per fine-grained rank-support block: `BLOCK_WORDS * 64` is a
/// block's max possible popcount, and that has to fit a `u8` (0..255), so
/// 3 is the largest value that still fits (`3*64=192 <= 255`;
/// `4*64=256` overflows).
const BLOCK_WORDS: u64 = 3;
/// Words per coarse checkpoint: `21 * BLOCK_WORDS` (63) rather than
/// exactly 64, so every checkpoint boundary is also a block boundary --
/// no partial-block handling needed where the two levels meet.
const CHECKPOINT_WORDS: u64 = 21 * BLOCK_WORDS;
const CHECKPOINT_BLOCKS: u64 = CHECKPOINT_WORDS / BLOCK_WORDS;

impl RadixTreeApi for RadixTreeRankDelta {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeRankDelta::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeRankDelta::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeRankDelta::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeRankDelta::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeRankDelta::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeRankDelta::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeRankDelta::height(self)
    }
}

pub struct RadixTreeRankDelta {
    /// Existence words for every non-leaf-level node, BFS/level order --
    /// same layout as the baseline's `t`, just without an attached rank
    /// structure.
    t_words: Vec<u64>,
    /// One entry per [`CHECKPOINT_WORDS`]-word checkpoint: the absolute
    /// cumulative popcount at that checkpoint's start (`rank1(checkpoint
    /// * CHECKPOINT_WORDS * 64)`). The coarse level of the two-level rank
    /// structure that replaces the baseline's `rank_samples`.
    t_checkpoints: Vec<u32>,
    /// One entry per [`BLOCK_WORDS`]-word block of `t_words`: that block's
    /// own popcount (0..=192, always fits a `u8`). The fine level -- used
    /// only to fill the gap from the nearest checkpoint to a target
    /// position, never summed further back than that.
    t_block_deltas: Vec<u8>,
    /// Existence words for leaf-level nodes. Never rank-queried (only
    /// `contains`/`query_rec` ever call `.get()`/index into it directly),
    /// so it carries no popcount array at all.
    l_words: Vec<u64>,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    dim_order: [usize; NUM_DIMS],
    single_point_exists: bool,
    height: u32,
}

impl RadixTreeRankDelta {
    pub fn default_split() -> [u8; NUM_DIMS] {
        [BASE / NUM_DIMS as u8; NUM_DIMS]
    }

    pub fn build(triples: &[Triple], split: [u8; NUM_DIMS]) -> Self {
        Self::build_with_order(triples, split, [0, 1, 2])
    }

    pub fn build_with_order(triples: &[Triple], split: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS]) -> Self {
        debug_assert!(
            split.iter().map(|&s| s as u32).sum::<u32>() <= BASE as u32,
            "split {split:?} sums to more than BASE ({BASE}) bits -- a single level would then need \
             more than K_SQUARE ({K_SQUARE}) children, overflowing the fixed 64-bit-per-node word \
             layout every other part of this structure assumes"
        );
        let dimension_sizes = dimension_sizes_from(triples);
        let level_dim_bits = static_schedule(total_bits_needed(dimension_sizes), split);
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order)
    }

    pub fn build_dynamic(triples: &[Triple]) -> Self {
        Self::build_dynamic_with_order(triples, [0, 1, 2])
    }

    pub fn build_dynamic_with_order(triples: &[Triple], dim_order: [usize; NUM_DIMS]) -> Self {
        let dimension_sizes = dimension_sizes_from(triples);
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order)
    }

    fn build_from_schedule(triples: &[Triple], dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS]) -> Self {
        if level_dim_bits.is_empty() {
            return RadixTreeRankDelta {
                t_words: Vec::new(),
                t_checkpoints: Vec::new(),
                t_block_deltas: Vec::new(),
                l_words: Vec::new(),
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                dim_order,
                single_point_exists: !triples.is_empty(),
                height: 0,
            };
        }
        let height = level_dim_bits.len() as u32;
        let level_shifts_cache: Vec<[u8; NUM_DIMS]> = level_dim_bits.iter().map(|&db| level_shifts(db, dim_order)).collect();
        let level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]> = {
            let mut remaining = total_bits_needed(dimension_sizes);
            level_dim_bits
                .iter()
                .map(|&db| {
                    for d in 0..NUM_DIMS {
                        remaining[d] -= db[d];
                    }
                    remaining
                })
                .collect()
        };

        let mut points: Vec<(u32, u32, u32)> = triples.to_vec();
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        let mut t_words: Vec<u64> = Vec::new();
        let mut l_words: Vec<u64> = Vec::new();

        struct FrontierEntry {
            start: usize,
            end: usize,
            lo: [u64; NUM_DIMS],
        }
        let mut frontier = vec![FrontierEntry {
            start: 0,
            end: points.len(),
            lo: [0; NUM_DIMS],
        }];

        for level in 0..height as usize {
            let is_last = level == height as usize - 1;
            let dim_bits = level_dim_bits[level];
            let shifts = level_shifts_cache[level];
            let child_cell_bits = level_child_cell_bits_cache[level];
            let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;

            let mut next_frontier = Vec::new();
            for entry in &frontier {
                let qi_of = |pt: &(u32, u32, u32)| -> usize {
                    let coords = [pt.0 as u64, pt.1 as u64, pt.2 as u64];
                    let mut qi = 0usize;
                    for d in 0..NUM_DIMS {
                        let sub = ((coords[d] - entry.lo[d]) >> child_cell_bits[d]) as usize;
                        qi |= sub << shifts[d];
                    }
                    qi
                };

                let slice = &mut points[entry.start..entry.end];
                let mut counts = [0usize; K_SQUARE as usize];
                for pt in slice.iter() {
                    counts[qi_of(pt)] += 1;
                }
                let mut offsets = [0usize; K_SQUARE as usize + 1];
                for i in 0..branching {
                    offsets[i + 1] = offsets[i] + counts[i];
                }

                let buf = &mut scratch[..slice.len()];
                let mut cursor = offsets;
                for &pt in slice.iter() {
                    let qi = qi_of(&pt);
                    buf[cursor[qi]] = pt;
                    cursor[qi] += 1;
                }
                slice.copy_from_slice(buf);

                let mut existence_word = 0u64;
                for qi in 0..branching {
                    let qstart = entry.start + offsets[qi];
                    let qend = entry.start + offsets[qi + 1];
                    if qstart < qend {
                        existence_word |= 1u64 << qi;
                        if !is_last {
                            let child_lo = RadixTreeRankDelta::advance_lo(entry.lo, dim_bits, shifts, qi, child_cell_bits);
                            next_frontier.push(FrontierEntry {
                                start: qstart,
                                end: qend,
                                lo: child_lo,
                            });
                        }
                    }
                }
                if is_last {
                    l_words.push(existence_word);
                } else {
                    t_words.push(existence_word);
                }
            }
            frontier = next_frontier;
        }
        // Plain push-based growth (final node count isn't known up front)
        // leaves real doubling slack, same reasoning as `BitVector::finalize`
        // and the `radixtree_dfs`/`radixtree_bp` builders -- trim it now
        // that construction is done, or `mem_size_bytes` (which reports
        // capacity, not length) reports inflated, non-representative sizes.
        t_words.shrink_to_fit();
        l_words.shrink_to_fit();

        // Both levels computed as single post-passes over the finished
        // `t_words` (not accumulated per-node during the build above) --
        // block/checkpoint boundaries are a pure function of position in
        // `t_words`, not of the tree structure, so there's no benefit to
        // computing them inline versus chunking the finished array.
        let t_block_deltas: Vec<u8> = t_words.chunks(BLOCK_WORDS as usize).map(|chunk| chunk.iter().map(|w| w.count_ones()).sum::<u32>() as u8).collect();
        let t_checkpoints: Vec<u32> = {
            let mut checkpoints = Vec::with_capacity(t_block_deltas.len().div_ceil(CHECKPOINT_BLOCKS as usize).max(1));
            let mut cum = 0u32;
            for chunk in t_block_deltas.chunks(CHECKPOINT_BLOCKS as usize) {
                checkpoints.push(cum);
                cum += chunk.iter().map(|&d| d as u32).sum::<u32>();
            }
            if checkpoints.is_empty() {
                checkpoints.push(0);
            }
            checkpoints
        };

        RadixTreeRankDelta {
            t_words,
            t_checkpoints,
            t_block_deltas,
            l_words,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            dim_order,
            single_point_exists: false,
            height,
        }
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn node_word_count(&self) -> u64 {
        (self.t_words.len() + self.l_words.len()) as u64
    }

    pub fn domain_size(&self, dim: usize) -> u64 {
        self.dimension_sizes[dim]
    }

    #[inline]
    fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        (s as u64) < self.dimension_sizes[0] && (p as u64) < self.dimension_sizes[1] && (o as u64) < self.dimension_sizes[2]
    }

    #[inline]
    fn advance_lo(mut lo: [u64; NUM_DIMS], dim_bits: [u8; NUM_DIMS], shifts: [u8; NUM_DIMS], qi: usize, child_cell_bits: [u8; NUM_DIMS]) -> [u64; NUM_DIMS] {
        for d in 0..NUM_DIMS {
            let mask = (1usize << dim_bits[d]) - 1;
            let sub = (qi >> shifts[d]) & mask;
            lo[d] += (sub as u64) << child_cell_bits[d];
        }
        lo
    }

    /// Exclusive rank1 (count of 1-bits in `[0, bit_pos)`) over `t_words`,
    /// via true random access -- no state carried between calls, so this
    /// costs the same fixed, small amount of work regardless of query
    /// pattern. Three steps: (1) O(1) index into `t_checkpoints` for the
    /// nearest checkpoint at or before `bit_pos`; (2) sum at most
    /// `CHECKPOINT_BLOCKS` (21) `t_block_deltas` entries to reach
    /// `bit_pos`'s own block; (3) popcount at most `BLOCK_WORDS` (3) whole
    /// or partial words of `t_words` within that block. Every step's
    /// bound is fixed by the constants above, never by where a previous
    /// call happened to land.
    #[inline]
    fn rank1_upto_bit(&self, bit_pos: u64) -> u64 {
        let block_bits = BLOCK_WORDS * WORD_BITS;
        let checkpoint_bits = CHECKPOINT_WORDS * WORD_BITS;
        let target_block = bit_pos / block_bits;
        let checkpoint_index = bit_pos / checkpoint_bits;
        let first_block_in_checkpoint = checkpoint_index * CHECKPOINT_BLOCKS;

        let mut rank = self.t_checkpoints[checkpoint_index as usize] as u64;
        for b in first_block_in_checkpoint..target_block {
            rank += self.t_block_deltas[b as usize] as u64;
        }

        let mut remaining_bits = bit_pos - target_block * block_bits;
        let mut word = target_block * BLOCK_WORDS;
        while remaining_bits >= WORD_BITS {
            rank += self.t_words[word as usize].count_ones() as u64;
            remaining_bits -= WORD_BITS;
            word += 1;
        }
        if remaining_bits > 0 {
            let mask = (1u64 << remaining_bits) - 1;
            rank += (self.t_words[word as usize] & mask).count_ones() as u64;
        }
        rank
    }

    #[inline]
    fn child_word_start(&self, bit_pos: u64, next_is_last: bool) -> u64 {
        let virtual_start = self.rank1_upto_bit(bit_pos + 1) * WORD_BITS;
        if next_is_last {
            virtual_start - (self.t_words.len() as u64 * WORD_BITS)
        } else {
            virtual_start
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        if !self.in_bounds(s, p, o) {
            return false;
        }
        if self.height == 0 {
            return self.single_point_exists;
        }
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut word_start = 0u64;
        for level in 0..self.height as usize {
            let is_last = level == self.height as usize - 1;
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let mut qi = 0usize;
            for d in 0..NUM_DIMS {
                let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
                qi |= sub << shifts[d];
            }
            let bit_pos = word_start + qi as u64;
            let exists = if is_last {
                (self.l_words[(word_start / WORD_BITS) as usize] >> qi) & 1 != 0
            } else {
                (self.t_words[(word_start / WORD_BITS) as usize] >> qi) & 1 != 0
            };
            if !exists {
                return false;
            }
            if is_last {
                return true;
            }
            word_start = self.child_word_start(bit_pos, level + 1 == self.height as usize - 1);
            lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 checked above")
    }

    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        self.query_two_bound([0, 2], [s, o], 1)
    }

    pub fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for i in 0..2 {
            if (bound_vals[i] as u64) >= self.dimension_sizes[bound_dims[i]] {
                return out;
            }
        }
        if self.height == 0 {
            if self.single_point_exists {
                out.push(0);
            }
            return out;
        }
        let level_masks = self.precompute_level_masks(bound_dims, bound_vals, |d| d == bound_dims[0] || d == bound_dims[1]);
        self.query_rec(0, 0, [0; NUM_DIMS], &level_masks[..self.height as usize], free_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn precompute_level_masks(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], is_bound: impl Fn(usize) -> bool) -> [u64; MAX_HEIGHT] {
        let mut lo = [0u64; NUM_DIMS];
        let mut masks = [0u64; MAX_HEIGHT];
        debug_assert!(self.height as usize <= MAX_HEIGHT, "tree height {} exceeds MAX_HEIGHT {}", self.height, MAX_HEIGHT);
        for level in 0..self.height as usize {
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let bound_val_for = |d: usize| {
                if !is_bound(d) {
                    None
                } else if d == bound_dims[0] {
                    Some(bound_vals[0])
                } else {
                    Some(bound_vals[1])
                }
            };
            masks[level] = bound_mask(dim_bits, shifts, lo, child_cell_bits, bound_val_for);
            for d in 0..NUM_DIMS {
                if is_bound(d) {
                    let bound_val = bound_val_for(d).unwrap();
                    let sub = (bound_val as u64 - lo[d]) >> child_cell_bits[d];
                    lo[d] += sub << child_cell_bits[d];
                }
            }
        }
        masks
    }

    pub fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        if (bound_val as u64) >= self.dimension_sizes[bound_dim] {
            return out;
        }
        if self.height == 0 {
            if self.single_point_exists {
                out.push(0);
            }
            return out;
        }
        let level_masks = self.precompute_level_masks([bound_dim, bound_dim], [bound_val, bound_val], |d| d == bound_dim);
        self.query_rec(0, 0, [0; NUM_DIMS], &level_masks[..self.height as usize], target_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same shape as the baseline's `query_rec`. Unlike v1/v2, there is no
    /// cursor threaded through the recursion: `child_word_start` (via
    /// `rank1_upto_bit`) is a stateless, self-contained lookup -- checkpoint
    /// plus a bounded scan -- so branching and backtracking need no special
    /// handling at all, exactly like the baseline's own `rank1`.
    #[allow(clippy::too_many_arguments)]
    fn query_rec(&self, level: usize, word_start: u64, lo: [u64; NUM_DIMS], level_masks: &[u64], free_dim: usize, out: &mut Vec<u32>) {
        let is_last = level == self.height as usize - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = level_masks[level];

        let word = if is_last { self.l_words[(word_start / WORD_BITS) as usize] } else { self.t_words[(word_start / WORD_BITS) as usize] };
        let mut relevant = word & mask;
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1; // clear the lowest set bit, advance to the next candidate

            let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            if is_last {
                out.push(child_lo[free_dim] as u32);
                continue;
            }
            let bit_pos = word_start + qi as u64;
            let child_word_start = self.child_word_start(bit_pos, level + 1 == self.height as usize - 1);
            self.query_rec(level + 1, child_word_start, child_lo, level_masks, free_dim, out);
        }
    }
}

fn level_shifts(dim_bits: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS]) -> [u8; NUM_DIMS] {
    let mut shifts = [0u8; NUM_DIMS];
    let mut acc = 0u8;
    for &d in dim_order.iter().rev() {
        shifts[d] = acc;
        acc += dim_bits[d];
    }
    shifts
}

fn bound_mask(dim_bits: [u8; NUM_DIMS], shifts: [u8; NUM_DIMS], lo: [u64; NUM_DIMS], child_cell_bits: [u8; NUM_DIMS], bound_val_for: impl Fn(usize) -> Option<u32>) -> u64 {
    let mut mask: u64 = 1;
    for d in 0..NUM_DIMS {
        match bound_val_for(d) {
            None => {
                let width = 1u32 << dim_bits[d];
                let mut new_mask = 0u64;
                for sub in 0..width {
                    new_mask |= mask << ((sub as u64) << shifts[d]);
                }
                mask = new_mask;
            }
            Some(bound_val) => {
                let sub = (bound_val as u64 - lo[d]) >> child_cell_bits[d];
                mask <<= sub << shifts[d];
            }
        }
    }
    mask
}

fn dimension_sizes_from(triples: &[Triple]) -> [u64; NUM_DIMS] {
    let mut dimension_sizes = [1u64; NUM_DIMS];
    for &(s, p, o) in triples {
        dimension_sizes[0] = dimension_sizes[0].max(s as u64 + 1);
        dimension_sizes[1] = dimension_sizes[1].max(p as u64 + 1);
        dimension_sizes[2] = dimension_sizes[2].max(o as u64 + 1);
    }
    dimension_sizes
}

fn total_bits_needed(dimension_sizes: [u64; NUM_DIMS]) -> [u8; NUM_DIMS] {
    let mut bits = [0u8; NUM_DIMS];
    for d in 0..NUM_DIMS {
        let mut cap = 1u64;
        while cap < dimension_sizes[d] {
            cap *= 2;
            bits[d] += 1;
        }
    }
    bits
}

fn static_schedule(mut remaining: [u8; NUM_DIMS], split: [u8; NUM_DIMS]) -> Vec<[u8; NUM_DIMS]> {
    let mut level_dim_bits = Vec::new();
    while remaining.iter().any(|&r| r > 0) {
        let mut this_level = [0u8; NUM_DIMS];
        for d in 0..NUM_DIMS {
            let take = split[d].min(remaining[d]);
            this_level[d] = take;
            remaining[d] -= take;
        }
        level_dim_bits.push(this_level);
    }
    level_dim_bits
}

fn dynamic_schedule(mut remaining: [u8; NUM_DIMS]) -> Vec<[u8; NUM_DIMS]> {
    let mut level_dim_bits = Vec::new();
    while remaining.iter().any(|&r| r > 0) {
        let mut this_level = [0u8; NUM_DIMS];
        let mut budget = BASE;
        loop {
            let active: Vec<usize> = (0..NUM_DIMS).filter(|&d| remaining[d] > 0).collect();
            if active.is_empty() || budget == 0 {
                break;
            }
            let share = (budget / active.len() as u8).max(1);
            let mut progressed = false;
            for &d in &active {
                if budget == 0 {
                    break;
                }
                let take = share.min(remaining[d]).min(budget);
                if take > 0 {
                    this_level[d] += take;
                    remaining[d] -= take;
                    budget -= take;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        level_dim_bits.push(this_level);
    }
    level_dim_bits
}

impl MemSize for RadixTreeRankDelta {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.t_words)
            + crate::common::metrics::vec_bytes(&self.t_checkpoints)
            + crate::common::metrics::vec_bytes(&self.t_block_deltas)
            + crate::common::metrics::vec_bytes(&self.l_words)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeRankDelta {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("t_words (64-bit-per-node)", crate::common::metrics::vec_bytes(&self.t_words)),
            ("t_checkpoints (4 bytes/63 words, coarse rank support)", crate::common::metrics::vec_bytes(&self.t_checkpoints)),
            ("t_block_deltas (1 byte/3 words, fine rank support)", crate::common::metrics::vec_bytes(&self.t_block_deltas)),
            ("l_words (64-bit-per-node, no rank)", crate::common::metrics::vec_bytes(&self.l_words)),
            ("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)),
            ("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)),
            ("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::{uniform, zipfian_entities, zipfian_predicates};
    use crate::common::reference::NaiveIndex;

    fn cross_check_tree(tree: &RadixTreeRankDelta, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
        for s in 0..num_s {
            for p in 0..num_p {
                for o in 0..num_o {
                    assert_eq!(tree.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o}) {label}");
                }
                let mut got = tree.query_two_bound([0, 1], [s, p], 2);
                let mut expect = naive.row(p, s);
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "query_two_bound(S={s},P={p},free=O) {label}");
            }
        }
        for p in 0..num_p {
            for o in 0..num_o {
                let mut got = tree.query_two_bound([1, 2], [p, o], 0);
                let mut expect = naive.col(p, o);
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "query_two_bound(P={p},O={o},free=S) {label}");
            }
        }
        for s in 0..num_s {
            let mut got = tree.query_one_bound(0, s, 2);
            let mut expect: Vec<u32> = (0..num_p).flat_map(|p| naive.row(p, s)).collect();
            got.sort_unstable();
            got.dedup();
            expect.sort_unstable();
            expect.dedup();
            assert_eq!(got, expect, "query_one_bound(S={s},free=O) {label}");
        }
        for s in 0..num_s {
            for o in 0..num_o {
                assert_eq!(tree.predicates_for(s, o), naive.predicates_for(s, o), "predicates_for(S={s},O={o}) {label}");
            }
        }
    }

    fn cross_check(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, split: [u8; NUM_DIMS]) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeRankDelta::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeRankDelta::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeRankDelta::build(&triples, RadixTreeRankDelta::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeRankDelta::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeRankDelta::build(&triples, RadixTreeRankDelta::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeRankDelta::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeRankDelta::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeRankDelta::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeRankDelta::default_split());
            let triples = zipfian_entities(10, 8, 10, 150, 1.2, 1.3, 1.2, seed * 71);
            cross_check(&triples, 10, 8, 10, RadixTreeRankDelta::default_split());
        }
    }

    #[test]
    fn cross_check_uneven_split_and_larger_domain() {
        for seed in 1..5u64 {
            let triples = uniform(40, 6, 40, 500, seed * 271);
            cross_check(&triples, 40, 6, 40, [3, 1, 2]);
            cross_check(&triples, 40, 6, 40, [1, 1, 1]);
        }
    }

    #[test]
    fn cross_check_dynamic_schedule() {
        for seed in 1..6u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check_dynamic(&triples, 10, 5, 10);
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check_dynamic(&triples, 10, 8, 10);
            let triples = uniform(40, 6, 40, 500, seed * 271);
            cross_check_dynamic(&triples, 40, 6, 40);
        }
    }

    #[test]
    fn cross_check_alternate_dim_orders() {
        for seed in 1..5u64 {
            let triples = uniform(20, 6, 20, 400, seed * 811);
            let naive = NaiveIndex::build(&triples);
            for (order, label) in [([1usize, 0, 2], "PSO"), ([0, 2, 1], "SOP")] {
                let tree = RadixTreeRankDelta::build_with_order(&triples, RadixTreeRankDelta::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeRankDelta::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeRankDelta::build(&triples, RadixTreeRankDelta::default_split());
        let dynamic_tree = RadixTreeRankDelta::build_dynamic(&triples);
        assert!(
            dynamic_tree.node_word_count() < static_tree.node_word_count(),
            "dynamic ({} words) should use fewer node-words than static ({} words)",
            dynamic_tree.node_word_count(),
            static_tree.node_word_count()
        );
    }

    #[test]
    fn duplicate_heavy_dataset() {
        let mut triples = vec![(1u32, 1u32, 1u32); 20];
        triples.push((5, 5, 5));
        cross_check(&triples, 6, 6, 6, RadixTreeRankDelta::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    /// Branching-correctness stress test: a dataset with enough branching
    /// that `query_one_bound`'s free dimension visits many sibling subtrees
    /// at varying depths, exercising `rank1_upto_bit` across a wide spread
    /// of non-adjacent bit positions (not just a simple monotonic walk).
    #[test]
    fn heavy_branching_rank_lookups() {
        let triples = uniform(500, 20, 500, 4000, 999);
        let naive = NaiveIndex::build(&triples);
        let tree = RadixTreeRankDelta::build_dynamic(&triples);
        for s in 0..500u32 {
            let mut got = tree.query_one_bound(0, s, 2);
            let mut expect: Vec<u32> = (0..20u32).flat_map(|p| naive.row(p, s)).collect();
            got.sort_unstable();
            got.dedup();
            expect.sort_unstable();
            expect.dedup();
            assert_eq!(got, expect, "query_one_bound(S={s})");
        }
    }
}
