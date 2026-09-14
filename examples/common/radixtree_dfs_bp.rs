//! A genuine Balanced Parentheses (BP) navigation variant of
//! [`crate::common::radixtree_dfs::RadixTreeDfs`], following Carmona &
//! Manzini, "Depth first representations of k²-trees" (2025)'s BP
//! representation *literally* -- unlike [`crate::common::radixtree_bp`]
//! (whose own doc comment explains why it isn't one).
//!
//! # What "genuine" means here
//!
//! Every node contributes exactly one matched `(`/`)` pair, emitted in DFS
//! preorder: `(`, then recurse into every existing child (each contributing
//! its own matched pair), then `)`. This is the *literal* textbook BP
//! encoding of an ordinal tree. Existence itself is still a flat 64-bit
//! word per node (`words`, identical to [`crate::common::radixtree_dfs`]) --
//! BP only replaces *navigation* (the `subtree_word_count`/`is_far` machinery
//! that file uses), isolating that one variable for comparison.
//!
//! # Why this exists: the literature's answer to "can compaction be free"
//!
//! `radixtree_dfs.rs`'s own doc comment (and the conversation that produced
//! this file) established: LOUDS/BFS-order trees (this project's baseline
//! `t`/`l`) get O(1) child-finding from a single `rank1`, for free, because
//! same-level nodes are stored contiguously. DFS order gives up that
//! property for cache locality, and every known fix is *itself* a
//! rank/min-max-family structure -- there is no way to get O(1) DFS-order
//! subtree-skip with zero auxiliary bookkeeping. The literature's own
//! answer, more refined than storing an explicit `u32` subtree size per
//! node (Enriched DFS, `radixtree_dfs.rs`) or an ad-hoc `rank1` over an
//! "is this node skippable" bitmap (tried and reverted in `radixtree_dfs.rs`'s
//! git history), is: encode the tree as a balanced-parenthesis sequence and
//! support `findclose` via a **range min-max tree** over it (Munro & Raman
//! 2001; Navarro & Sadakane 2014, "Fully-Functional Succinct Trees"). That
//! gets the auxiliary structure down to *o(n)* bits, asymptotically smaller
//! than a `u32`-per-node array.
//!
//! # This implementation: a practical two-level approximation, not the
//! full multi-level min-max tree
//!
//! A textbook range min-max tree is itself a recursive tree of block
//! summaries (giving true O(log n / log log n), with four-Russians tables
//! pushing small cases to O(1)). Implementing that correctly is a
//! significant undertaking on its own. This file implements the
//! **single-level block decomposition** that same literature builds on top
//! of: the `2n`-bit parenthesis sequence is split into fixed 64-bit blocks
//! (one `u64` word each); each block stores its own `min_excess` (the
//! lowest running open-minus-close balance reached inside it, relative to
//! the block's own start) and `total_excess` (net balance change across
//! the block), each fitting an `i8` (a 64-bit block's excess can range at
//! most ±64). `findclose(open_pos)` scans forward: within the block
//! containing `open_pos` bit by bit, then block-by-block using the summary
//! arrays to skip whole blocks that provably can't contain the answer
//! (only scanning bit-by-bit inside whichever block genuinely does). Worst
//! case is O(n / 64) block-skips; in practice, for the local (nearby)
//! `findclose` calls a subtree-skip actually makes, this resolves in very
//! few blocks. A full multi-level tree would bound the worst case far more
//! tightly (`O(log n)` block-skips instead of `O(n / 64)`) at the cost of
//! real extra implementation complexity -- noted here as the honest gap
//! between this and the textbook construction, not implemented.
//!
//! # Why `findclose` alone is enough (the elegant part)
//!
//! Skipping a child's subtree needs two things: how far to jump `word_index`
//! forward in `words`, and where that child's own matched close-paren is
//! (so its *sibling's* open immediately follows). A balanced sequence of
//! length `L` bits covering `N` nodes always has `L == 2*N` (every node
//! contributes exactly one open and one close) -- so once `findclose` gives
//! the close position, the subtree's *node count* (and therefore the
//! `words`-index jump) is simply `(close_pos - open_pos + 1) / 2`, no
//! further storage or lookup needed. `findclose` is the only primitive this
//! variant needs.
//!
//! # How this differs from the other DFS-order variants
//!
//! - **vs. [`crate::common::radixtree_dfs`]**: same DFS storage of `words`,
//!   same O(preceding existing siblings) navigation cost; the only
//!   difference is what pays for the subtree-skip -- an explicit `u32`
//!   subtree-word-count array there, a ~0.31-byte/node BP sequence plus
//!   block summaries here.
//! - **vs. [`crate::common::radixtree_bp`]**: that file stores a sorted
//!   slot list per node (a different *existence*-encoding scheme) plus the
//!   same kind of explicit subtree-node-count array `radixtree_dfs` uses
//!   for navigation. This file keeps the flat 64-bit existence word and
//!   replaces *navigation* with real BP -- a different axis of comparison
//!   entirely.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const WORD_BITS: u64 = K_SQUARE as u64;
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsBp {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsBp::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsBp::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsBp::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsBp::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsBp::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsBp::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsBp::height(self)
    }
}

pub struct RadixTreeDfsBp {
    /// One 64-bit existence word per node, in DFS/preorder visitation
    /// order -- identical in meaning and order to
    /// [`crate::common::radixtree_dfs::RadixTreeDfs`]'s `words`.
    words: Vec<u64>,
    /// The balanced-parenthesis encoding of the tree's topology, packed
    /// one bit per paren (`1` = `(`, `0` = `)`), in DFS preorder: each
    /// node's own `(` immediately precedes its first child's `(` (if any),
    /// and its own `)` immediately follows its last child's `)` (or its
    /// own `(` directly, if it has no children). Length is always
    /// `2 * words.len()` bits.
    bp: Vec<u64>,
    bp_len_bits: u64,
    /// Per 64-bit `bp` block: the lowest running excess (opens minus
    /// closes so far) reached strictly inside the block, relative to the
    /// excess value at the block's own start. Always in `-64..=0`.
    block_min_excess: Vec<i8>,
    /// Per 64-bit `bp` block: the net excess change across the whole
    /// block (`+1` per open, `-1` per close). Always in `-64..=64`.
    block_total_excess: Vec<i8>,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    dim_order: [usize; NUM_DIMS],
    single_point_exists: bool,
    height: u32,
}

impl RadixTreeDfsBp {
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
            return RadixTreeDfsBp {
                words: Vec::new(),
                bp: Vec::new(),
                bp_len_bits: 0,
                block_min_excess: Vec::new(),
                block_total_excess: Vec::new(),
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
        let mut words = Vec::new();
        let mut bp = Vec::new();
        let mut bp_len_bits = 0u64;
        build_bp_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, height, &level_dim_bits, &level_shifts_cache, &level_child_cell_bits_cache, &mut words, &mut bp, &mut bp_len_bits);
        words.shrink_to_fit();
        bp.shrink_to_fit();

        let (block_min_excess, block_total_excess) = compute_block_summaries(&bp, bp_len_bits);

        RadixTreeDfsBp {
            words,
            bp,
            bp_len_bits,
            block_min_excess,
            block_total_excess,
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
        self.words.len() as u64
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

    #[inline]
    fn bp_bit(&self, pos: u64) -> bool {
        (self.bp[(pos / WORD_BITS) as usize] >> (pos % WORD_BITS)) & 1 == 1
    }

    /// Finds the position of the `)` matching the `(` at `open_pos`.
    /// Scans the (possibly partial) block containing `open_pos + 1` bit by
    /// bit, then skips whole later blocks using `block_min_excess` (only
    /// descending into whichever block the summaries prove contains the
    /// answer) -- see the module doc comment for why this isn't a true
    /// O(1) range-min-max tree, just the single-level block scan it's
    /// built on.
    fn findclose(&self, open_pos: u64) -> u64 {
        let mut pos = open_pos + 1;
        let mut excess: i64 = 0;
        let start_block = pos / WORD_BITS;
        let start_block_end = ((start_block + 1) * WORD_BITS).min(self.bp_len_bits);
        while pos < start_block_end {
            excess += if self.bp_bit(pos) { 1 } else { -1 };
            if excess == -1 {
                return pos;
            }
            pos += 1;
        }
        let mut block = (pos / WORD_BITS) as usize;
        loop {
            let candidate = excess + self.block_min_excess[block] as i64;
            if candidate <= -1 {
                let block_start = (block as u64) * WORD_BITS;
                let block_end = (block_start + WORD_BITS).min(self.bp_len_bits);
                let mut p = block_start;
                while p < block_end {
                    excess += if self.bp_bit(p) { 1 } else { -1 };
                    if excess == -1 {
                        return p;
                    }
                    p += 1;
                }
                unreachable!("findclose: target excess not found within the block the summary said contained it");
            }
            excess += self.block_total_excess[block] as i64;
            block += 1;
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
        let mut word_index = 0usize;
        let mut bp_pos = 0u64;
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
            let word = self.words[word_index];
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if is_last {
                return true;
            }
            // Skip past every existing sibling before `qi`: each one's
            // matching close paren (via `findclose`) gives both its span
            // (== its subtree's node count, via the `(close-open+1)/2`
            // identity) and the next sibling's own open position.
            let mut child_word_index = word_index + 1;
            let mut child_bp_pos = bp_pos + 1;
            let mut earlier = word & ((1u64 << qi) - 1);
            while earlier != 0 {
                earlier &= earlier - 1;
                let close_pos = self.findclose(child_bp_pos);
                let span_nodes = ((close_pos - child_bp_pos + 1) / 2) as usize;
                child_word_index += span_nodes;
                child_bp_pos = close_pos + 1;
            }
            word_index = child_word_index;
            bp_pos = child_bp_pos;
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
        self.query_rec(0, 0, [0; NUM_DIMS], 0, &level_masks[..self.height as usize], free_dim, &mut out);
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
        self.query_rec(0, 0, [0; NUM_DIMS], 0, &level_masks[..self.height as usize], target_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same shape as [`crate::common::radixtree_dfs::RadixTreeDfs::query_rec`]:
    /// walks every *existing* child (matched or not) in ascending `qi`
    /// order, since reaching the next sibling requires passing through
    /// every earlier one's `findclose`. Only matched children get recursed
    /// into.
    #[allow(clippy::too_many_arguments)]
    fn query_rec(&self, word_index: usize, bp_pos: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], free_dim: usize, out: &mut Vec<u32>) {
        let is_last = level == self.height as usize - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = level_masks[level];
        let word = self.words[word_index];

        if is_last {
            let mut relevant = word & mask;
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                out.push(child_lo[free_dim] as u32);
            }
            return;
        }

        let mut child_word_index = word_index + 1;
        let mut child_bp_pos = bp_pos + 1;
        let mut existing = word;
        while existing != 0 {
            let qi = existing.trailing_zeros() as usize;
            existing &= existing - 1;
            let close_pos = self.findclose(child_bp_pos);
            if (mask >> qi) & 1 != 0 {
                let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                self.query_rec(child_word_index, child_bp_pos, child_lo, level + 1, level_masks, free_dim, out);
            }
            let span_nodes = ((close_pos - child_bp_pos + 1) / 2) as usize;
            child_word_index += span_nodes;
            child_bp_pos = close_pos + 1;
        }
    }
}

fn push_bp_bit(bp: &mut Vec<u64>, bp_len_bits: &mut u64, bit: bool) {
    let word_idx = (*bp_len_bits / WORD_BITS) as usize;
    if word_idx == bp.len() {
        bp.push(0);
    }
    if bit {
        bp[word_idx] |= 1u64 << (*bp_len_bits % WORD_BITS);
    }
    *bp_len_bits += 1;
}

/// Recursively partitions `points` (in place, via `scratch`) and appends
/// one DFS-preorder-ordered word per visited node to `words`, plus that
/// node's own `(` before recursing into children and `)` right after --
/// the literal balanced-parenthesis encoding, built by the same
/// counting-sort partitioning `radixtree_dfs`'s builder uses.
#[allow(clippy::too_many_arguments)]
fn build_bp_rec(
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: u32,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    words: &mut Vec<u64>,
    bp: &mut Vec<u64>,
    bp_len_bits: &mut u64,
) {
    let is_last = level == height as usize - 1;
    let dim_bits = level_dim_bits[level];
    let shifts = level_shifts_cache[level];
    let child_cell_bits = level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;

    let qi_of = |pt: &(u32, u32, u32)| -> usize {
        let coords = [pt.0 as u64, pt.1 as u64, pt.2 as u64];
        let mut qi = 0usize;
        for d in 0..NUM_DIMS {
            let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
            qi |= sub << shifts[d];
        }
        qi
    };
    let mut counts = [0usize; K_SQUARE as usize];
    for pt in points.iter() {
        counts[qi_of(pt)] += 1;
    }
    let mut offsets = [0usize; K_SQUARE as usize + 1];
    for i in 0..branching {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let buf = &mut scratch[..points.len()];
    let mut cursor = offsets;
    for &pt in points.iter() {
        let qi = qi_of(&pt);
        buf[cursor[qi]] = pt;
        cursor[qi] += 1;
    }
    points.copy_from_slice(buf);

    let mut existence_word = 0u64;
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            existence_word |= 1u64 << qi;
        }
    }
    words.push(existence_word);
    push_bp_bit(bp, bp_len_bits, true); // this node's own '('

    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeDfsBp::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                build_bp_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, words, bp, bp_len_bits);
            }
        }
    }
    push_bp_bit(bp, bp_len_bits, false); // this node's own ')'
}

/// One pass over `bp`, block by block (`WORD_BITS` bits each): for each
/// block, the lowest running excess reached inside it (relative to the
/// block's own start) and its net excess change. The last block may be
/// partial (`bp_len_bits` not a multiple of `WORD_BITS`) -- padding bits
/// beyond `bp_len_bits` are simply not processed, so they can't corrupt
/// either summary.
fn compute_block_summaries(bp: &[u64], bp_len_bits: u64) -> (Vec<i8>, Vec<i8>) {
    let mut block_min_excess = Vec::with_capacity(bp.len());
    let mut block_total_excess = Vec::with_capacity(bp.len());
    for (block_idx, &word) in bp.iter().enumerate() {
        let block_start = (block_idx as u64) * WORD_BITS;
        let bits_in_block = (bp_len_bits - block_start).min(WORD_BITS);
        let mut excess: i32 = 0;
        let mut min_excess: i32 = 0;
        for bit in 0..bits_in_block {
            excess += if (word >> bit) & 1 == 1 { 1 } else { -1 };
            min_excess = min_excess.min(excess);
        }
        block_min_excess.push(min_excess as i8);
        block_total_excess.push(excess as i8);
    }
    (block_min_excess, block_total_excess)
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

impl MemSize for RadixTreeDfsBp {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.words)
            + crate::common::metrics::vec_bytes(&self.bp)
            + crate::common::metrics::vec_bytes(&self.block_min_excess)
            + crate::common::metrics::vec_bytes(&self.block_total_excess)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfsBp {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("words (64-bit-per-node, DFS order)", crate::common::metrics::vec_bytes(&self.words)),
            ("bp (2 bits/node, balanced parens)", crate::common::metrics::vec_bytes(&self.bp)),
            ("block_min_excess (1 byte/64 paren-bits)", crate::common::metrics::vec_bytes(&self.block_min_excess)),
            ("block_total_excess (1 byte/64 paren-bits)", crate::common::metrics::vec_bytes(&self.block_total_excess)),
            ("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)),
            ("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)),
            ("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::{uniform, zipfian_predicates};
    use crate::common::reference::NaiveIndex;

    fn cross_check_tree(tree: &RadixTreeDfsBp, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        let tree = RadixTreeDfsBp::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeDfsBp::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeDfsBp::build(&triples, RadixTreeDfsBp::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeDfsBp::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeDfsBp::build(&triples, RadixTreeDfsBp::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeDfsBp::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsBp::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsBp::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsBp::default_split());
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
                let tree = RadixTreeDfsBp::build_with_order(&triples, RadixTreeDfsBp::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeDfsBp::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeDfsBp::build(&triples, RadixTreeDfsBp::default_split());
        let dynamic_tree = RadixTreeDfsBp::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfsBp::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    /// Directly exercises `findclose` crossing multiple block boundaries:
    /// a wide, moderately deep tree so some subtrees span well over 64
    /// paren-bits, forcing the block-skip path (not just the initial
    /// partial-block scan).
    #[test]
    fn findclose_crosses_block_boundaries() {
        let triples = uniform(500, 20, 500, 6000, 999);
        let naive = NaiveIndex::build(&triples);
        let tree = RadixTreeDfsBp::build_dynamic(&triples);
        assert!(tree.bp_len_bits > 512, "test dataset should produce a BP sequence spanning multiple 64-bit blocks (got {} bits)", tree.bp_len_bits);
        for s in (0..500u32).step_by(7) {
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
