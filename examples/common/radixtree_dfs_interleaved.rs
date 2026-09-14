//! A DFS-order navigation variant of
//! [`crate::common::radixtree_dfs::RadixTreeDfs`] that stores each node's
//! existence word and (when it has one) its subtree-skip value *inline,
//! back to back, in one combined byte stream* -- instead of two separate
//! arrays (`words` and a far-nodes-only `subtree_word_count`) bridged by a
//! `rank1` translation (`radixtree_dfs.rs`'s `is_far`, and this file's own
//! earlier `radixtree_dfs_bp.rs` sibling).
//!
//! # The insight this implements
//!
//! Removing an entry from the middle of an array shifts every later
//! entry's index -- that's true regardless of *why* the entry was removed,
//! and it's what forced `radixtree_dfs.rs`'s `is_far` rank structure (and
//! this project's genuine balanced-parenthesis attempt) to exist at all:
//! both pay a real, measured navigation-time cost (see their own doc
//! comments and git history) to translate "index into the full node
//! sequence" into "index into the entries-only-for-nodes-that-have-one
//! array."
//!
//! But that translation is only needed because the skip value lives in a
//! *separate* array from the word it belongs to. If the skip value is
//! instead written immediately after its own node's word, in the same
//! stream, its address is never "look it up somewhere else" -- it's just
//! "the position I'm already standing on, plus 8 bytes" (a word is always
//! 8 bytes). No index translation of any kind is needed, because there is
//! only ever one position being tracked, not two that need reconciling.
//!
//! # Layout
//!
//! One combined `Vec<u8>`, DFS/preorder, where a node's record is:
//! - **leaf or leaf-parent** (see [`crate::common::radixtree_dfs`]'s doc
//!   comment for why these two need no stored skip at all -- their subtree
//!   size is derivable from `count_ones()` on already-loaded words): 8
//!   bytes, just the existence word.
//! - **far** node (`level + 2 < height`): 12 bytes -- the 8-byte existence
//!   word, immediately followed by a 4-byte `u32` holding the *entire
//!   subtree's total byte span* (itself included, i.e. `self.combined.len()`
//!   this node's record would occupy end-to-end). Reaching the next
//!   sibling is exactly `this_node_pos + that_span` -- no lookup elsewhere,
//!   no rank, one fixed-offset read.
//!
//! This costs the same 8 bytes/leaf-or-leaf-parent-node and 12
//! bytes/far-node as `radixtree_dfs.rs`'s `words` + `subtree_word_count`
//! combination, *without* needing `is_far`'s rank-augmented bitvector at
//! all (that bitvector's entire job was translating between two index
//! spaces this design doesn't have) -- a modest additional memory win, and
//! (the actual point) replacing every far-sibling skip's `rank1` scan with
//! one direct memory read at a statically known offset.
//!
//! # How this differs from the other DFS-order variants
//!
//! - **vs. [`crate::common::radixtree_dfs`]**: identical node-existence
//!   semantics and the identical leaf/leaf-parent/far three-way navigation
//!   split; the only change is how a far node's skip value is addressed --
//!   inline here, via a separate rank-translated array there.
//! - **vs. [`crate::common::radixtree_dfs_bp`]**: that file replaces the
//!   skip value's *meaning* entirely (a balanced-parenthesis `findclose`
//!   instead of a stored subtree span) in pursuit of the same
//!   no-separate-index goal, at real implementation complexity and (as
//!   measured) a real navigation-time cost from its single-level block
//!   scan. This file keeps `radixtree_dfs.rs`'s original stored-span
//!   scheme unchanged and solves the indexing problem by relocating the
//!   value instead of re-deriving it.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;
const WORD_RECORD_BYTES: usize = 8;
const SKIP_RECORD_BYTES: usize = 4;
const FAR_RECORD_BYTES: usize = WORD_RECORD_BYTES + SKIP_RECORD_BYTES;

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleaved {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsInterleaved::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleaved::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleaved::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsInterleaved::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsInterleaved::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsInterleaved::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsInterleaved::height(self)
    }
}

pub struct RadixTreeDfsInterleaved {
    /// One combined byte stream, DFS/preorder: each node contributes an
    /// 8-byte little-endian existence word, and *far* nodes (`level + 2 <
    /// height`) additionally contribute a 4-byte little-endian `u32` right
    /// after it -- see the module doc comment.
    combined: Vec<u8>,
    node_count: u64,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    dim_order: [usize; NUM_DIMS],
    single_point_exists: bool,
    height: u32,
}

#[inline]
fn read_u64(buf: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap())
}

#[inline]
fn read_u32(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

#[inline]
fn push_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Appends a zeroed `u32` placeholder, returning its byte offset so the
/// caller can overwrite it once the real value (the subtree's total byte
/// span) is known, after every child has been built.
#[inline]
fn push_u32_placeholder(buf: &mut Vec<u8>) -> usize {
    let pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    pos
}

#[inline]
fn write_u32_at(buf: &mut [u8], pos: usize, val: u32) {
    buf[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
}

impl RadixTreeDfsInterleaved {
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
            return RadixTreeDfsInterleaved {
                combined: Vec::new(),
                node_count: 0,
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
        let mut combined = Vec::new();
        let mut node_count = 0u64;
        build_interleaved_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, height, &level_dim_bits, &level_shifts_cache, &level_child_cell_bits_cache, &mut combined, &mut node_count);
        combined.shrink_to_fit();

        RadixTreeDfsInterleaved {
            combined,
            node_count,
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
        self.node_count
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

    /// Single-dimension form of [`Self::advance_lo`], for callers that only
    /// ever read `lo[d]` back for one specific `d` (a query's `free_dim`).
    /// `sub` depends only on `qi`/`shifts`/`dim_bits`, never on the
    /// incoming `lo` value -- `advance_lo`'s per-dimension update is a pure
    /// write-only accumulator for any dimension whose result is never read
    /// back, so tracking the other dimensions at all (bound or not) is
    /// dead work, not just "frozen" state worth freezing.
    #[inline]
    fn advance_lo_dim(lo: u64, dim_bits: u8, shifts: u8, qi: usize, child_cell_bits: u8) -> u64 {
        let mask = (1usize << dim_bits) - 1;
        let sub = (qi >> shifts) & mask;
        lo + ((sub as u64) << child_cell_bits)
    }

    /// Whether the node at `level` stores an inline subtree-skip value
    /// after its own word (a "far" node -- see the module doc comment),
    /// hence how many bytes its own record occupies before its first
    /// child (if any) begins.
    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height as usize { FAR_RECORD_BYTES } else { WORD_RECORD_BYTES }
    }

    /// Byte position immediately after one specific child's entire
    /// record -- itself and, transitively, everything in its subtree --
    /// computed without visiting any of it. `child_pos` is that child's
    /// own record's start; `child_level` is its level. One height
    /// cutoff (`child_level + 2 < height`, i.e. "far") decides how:
    /// above it, the child's own stored skip already covers its whole
    /// subtree -- one fixed-offset read. At or under it, a child is
    /// always either a leaf-parent (`child_level == height - 2`: no
    /// stored skip, so derive it -- its span is its own word plus one
    /// fixed 8-byte record per existing leaf child) or a leaf itself
    /// (`child_level == height - 1`: fixed size, no children of its own,
    /// nothing follows it). Far is checked first: it's the dominant case
    /// among skipped siblings (far spans every level from the root down
    /// to two above the leaves, leaf-parent is exactly one level -- see
    /// the module doc comment's memory-cost discussion), so putting it
    /// first lets that common case exit after one comparison instead of
    /// two.
    #[inline]
    fn child_span_end(&self, child_pos: usize, child_level: usize) -> usize {
        if child_level + 2 < self.height as usize {
            let skip = read_u32(&self.combined, child_pos + WORD_RECORD_BYTES) as usize;
            child_pos + skip
        } else if child_level == self.height as usize - 1 {
            child_pos + WORD_RECORD_BYTES
        } else {
            let word = read_u64(&self.combined, child_pos);
            child_pos + WORD_RECORD_BYTES + WORD_RECORD_BYTES * word.count_ones() as usize
        }
    }

    /// Batched form of [`Self::child_span_end`] for callers (namely
    /// [`Self::contains`]) that need to skip a whole run of earlier
    /// existing children at once instead of one at a time. When those
    /// children are themselves leaves, their fixed size means the whole
    /// run's span is one multiply -- no need to visit any of them, not
    /// even to loop -- so that case stays a true O(1) shortcut rather
    /// than degrading to a per-child loop the way the other two cases
    /// unavoidably must (a leaf-parent's or far node's span can only be
    /// known by reading something specific to that one child).
    #[inline]
    fn advance_past_children(&self, child_pos: usize, remaining_bits: u64, child_level: usize) -> usize {
        if child_level == self.height as usize - 1 {
            child_pos + WORD_RECORD_BYTES * remaining_bits.count_ones() as usize
        } else {
            let mut pos = child_pos;
            let mut bits = remaining_bits;
            while bits != 0 {
                bits &= bits - 1;
                pos = self.child_span_end(pos, child_level);
            }
            pos
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
        let mut pos = 0usize;
        for level in 0..self.height as usize {
            let is_last = level == self.height as usize - 1;
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            // `sub` is computed once per dim and reused for both `qi` and
            // the `lo` update below -- `advance_lo` used to re-derive the
            // very same value from `qi` (mask + shift back out), which is
            // strictly redundant when the value it's re-deriving is still
            // sitting right here. `dim_bits` (only ever needed for that
            // re-derivation's mask) drops out entirely along with it.
            let mut qi = 0usize;
            let mut next_lo = lo;
            for d in 0..NUM_DIMS {
                let sub = (coords[d] - lo[d]) >> child_cell_bits[d];
                qi |= (sub as usize) << shifts[d];
                next_lo[d] += sub << child_cell_bits[d];
            }
            let word = read_u64(&self.combined, pos);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if is_last {
                return true;
            }
            let earlier = word & ((1u64 << qi) - 1);
            let first_child_pos = pos + self.own_record_bytes(level);
            pos = self.advance_past_children(first_child_pos, earlier, level + 1);
            lo = next_lo;
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
        let unconstrained_from = self.first_fully_open_level(&level_masks);
        self.query_rec(0, 0, 0, &level_masks[..self.height as usize], free_dim, unconstrained_from, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The first level (if any, else `self.height`) from which
    /// `level_masks` stays fully open for the rest of the descent -- once
    /// every bound dimension's bit budget is exhausted (a fixed point
    /// determined by domain sizes and the schedule, not by the query's
    /// specific values), no level past it ever filters anything, since a
    /// dimension's bit budget only shrinks to `0` and stays there (see
    /// `static_schedule`/`dynamic_schedule`). From this level down, every
    /// existing child matches, so [`Self::query_rec`]'s own masked descent
    /// naturally degenerates into just listing everything -- see its doc
    /// comment.
    fn first_fully_open_level(&self, level_masks: &[u64; MAX_HEIGHT]) -> usize {
        for level in 0..self.height as usize {
            let dim_bits = self.level_dim_bits[level];
            let bit_sum = (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
            let branching: u64 = 1u64 << bit_sum; // number of possible qi slots, 2^bit_sum
            let full_mask = if branching >= 64 { u64::MAX } else { (1u64 << branching) - 1 };
            if level_masks[level] == full_mask {
                return level;
            }
        }
        self.height as usize
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
        let unconstrained_from = self.first_fully_open_level(&level_masks);
        self.query_rec(0, 0, 0, &level_masks[..self.height as usize], target_dim, unconstrained_from, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Iterative, explicit-stack descent -- constrained (mask-checked,
    /// skip-value-reading) while `level < unconstrained_from`, then simply
    /// *listing* every remaining existing value once `level >=
    /// unconstrained_from` (see [`Self::first_fully_open_level`]): from
    /// that level on, every check would trivially pass, so there is
    /// nothing left to filter, only output to produce. Both regimes share
    /// one loop, one stack, and one frame type -- listing isn't a separate
    /// traversal bolted on, just what the same per-node logic degenerates
    /// to once nothing can be excluded any more:
    /// - **descend**: constrained still has to inspect every existing
    ///   child (matched or not) to know where each one ends, using
    ///   [`Self::child_span_end`] to skip past the unmatched ones without
    ///   recursing into them. Listing never needs that -- every existing
    ///   child matches, so the very first one is always right; it just
    ///   goes straight to `first_child_pos`, no span read at all.
    /// - **resume-on-ascend**: a constrained [`ConstrainedFrame`]'s
    ///   `child_pos` is real state -- it may still have unmatched siblings
    ///   left to walk past, via `child_span_end`, before its next match or
    ///   exhaustion. A listing frame needs none of that: DFS lays children
    ///   out consecutively, and since listing never skips any of them,
    ///   `pos` is already sitting exactly where the next sibling begins by
    ///   the time the ascend loop runs it -- `child_pos` goes unused.
    fn query_rec(&self, start_pos: usize, start_lo: u64, start_level: usize, level_masks: &[u64], free_dim: usize, unconstrained_from: usize, out: &mut Vec<u32>) {
        let mut stack: [ConstrainedFrame; MAX_HEIGHT] = [ConstrainedFrame::EMPTY; MAX_HEIGHT];
        let mut stack_len: usize = 0;
        let mut pos = start_pos;
        let mut lo = start_lo;
        let mut level = start_level;

        'outer: loop {
            let mut descended = false;
            let is_last = level == self.height as usize - 1;
            let listing = level >= unconstrained_from;
            let dim_bits = self.level_dim_bits[level][free_dim];
            let shifts = self.level_shifts_cache[level][free_dim];
            let child_cell_bits = self.level_child_cell_bits_cache[level][free_dim];
            let word = read_u64(&self.combined, pos);
            // Every node ever visited here is either the root or an
            // existing, matched child -- and `build_interleaved_rec`
            // only ever records a node for a non-empty point bucket, so
            // its existence word is never all-zero. That makes a
            // childless-non-leaf branch dead code; kept as a debug-only
            // check rather than removed outright, so the invariant stays
            // verified against every cross-check test instead of just
            // assumed.
            debug_assert!(word != 0, "visited node at level {level} has an empty existence word");

            if is_last {
                let mut relevant = if listing { word } else { word & level_masks[level] };
                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                    out.push(child_lo as u32);
                }
                pos += WORD_RECORD_BYTES;
            } else if listing && level + 1 == self.height as usize - 1 {
                // Flat leaf-listing fast path: children are leaves and
                // every one matches, so there's nothing to gain from the
                // general stack machinery below -- walk this node's own
                // bits and each matched leaf child's bits directly. This
                // level is always exactly `height - 2`, which by
                // definition is never a far node (its own record is
                // always the plain 8-byte form), so that's a compile-time
                // constant here, not a call to `own_record_bytes`.
                let leaf_level = level + 1;
                let leaf_dim_bits = self.level_dim_bits[leaf_level][free_dim];
                let leaf_shifts = self.level_shifts_cache[leaf_level][free_dim];
                let leaf_child_cell_bits = self.level_child_cell_bits_cache[leaf_level][free_dim];
                let mut leaf_pos = pos + WORD_RECORD_BYTES;
                let mut own_bits = word;
                while own_bits != 0 {
                    let qi = own_bits.trailing_zeros() as usize;
                    own_bits &= own_bits - 1;
                    let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                    let leaf_word = read_u64(&self.combined, leaf_pos);
                    let mut relevant = leaf_word;
                    while relevant != 0 {
                        let leaf_qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let leaf_lo = Self::advance_lo_dim(child_lo, leaf_dim_bits, leaf_shifts, leaf_qi, leaf_child_cell_bits);
                        out.push(leaf_lo as u32);
                    }
                    leaf_pos += WORD_RECORD_BYTES;
                }
                pos = leaf_pos;
            } else if listing {
                // Every existing child matches -- descend straight into
                // the first one, no span read needed to skip past
                // anything, since there is nothing to skip past. Every
                // level reaching this branch is strictly above the flat
                // path's `height - 2` (that case and `is_last` already
                // claimed the only two non-far levels), so it's always
                // far -- another compile-time constant, not a call to
                // `own_record_bytes`.
                let qi = word.trailing_zeros() as usize;
                let remaining = word & (word - 1);
                let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                debug_assert!(stack_len < MAX_HEIGHT, "query stack overflow: tree height exceeds MAX_HEIGHT");
                stack[stack_len] = ConstrainedFrame { lo, remaining, child_pos: 0 };
                stack_len += 1;
                pos += FAR_RECORD_BYTES;
                lo = child_lo;
                level += 1;
                descended = true;
            } else {
                let mask = level_masks[level];
                let mut child_pos = pos + self.own_record_bytes(level);
                let mut remaining = word;

                while remaining != 0 {
                    let qi = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let matched = (mask >> qi) & 1 != 0;
                    let next_child_pos = self.child_span_end(child_pos, level + 1);

                    if matched {
                        let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                        debug_assert!(stack_len < MAX_HEIGHT, "query stack overflow: tree height exceeds MAX_HEIGHT");
                        stack[stack_len] = ConstrainedFrame { lo, remaining, child_pos: next_child_pos };
                        stack_len += 1;
                        pos = child_pos;
                        lo = child_lo;
                        level += 1;
                        descended = true;
                        break;
                    }
                    child_pos = next_child_pos;
                }

                if !descended {
                    pos = child_pos;
                }
            }

            if descended {
                continue 'outer;
            }

            // Ascend: resume the nearest open ancestor with an unprocessed
            // child left, popping any fully exhausted one and continuing
            // further up. `remaining == 0` (a frame pushed with no more
            // siblings after its one matched child -- the common case for
            // a sparse or near-leaf node) is checked before touching any
            // per-dim level data, so a frame that's about to be popped
            // outright never pays for fetches it won't use.
            loop {
                if stack_len == 0 {
                    return;
                }
                let frame_level = start_level + stack_len - 1;
                let frame = &mut stack[stack_len - 1];
                if frame.remaining == 0 {
                    stack_len -= 1;
                    continue;
                }

                if frame_level >= unconstrained_from {
                    let frame_dim_bits = self.level_dim_bits[frame_level][free_dim];
                    let frame_shifts = self.level_shifts_cache[frame_level][free_dim];
                    let frame_child_cell_bits = self.level_child_cell_bits_cache[frame_level][free_dim];
                    let qi = frame.remaining.trailing_zeros() as usize;
                    frame.remaining &= frame.remaining - 1;
                    lo = Self::advance_lo_dim(frame.lo, frame_dim_bits, frame_shifts, qi, frame_child_cell_bits);
                    level = frame_level + 1;
                    // `pos` already sits exactly where this sibling
                    // begins -- see the doc comment above.
                    continue 'outer;
                }

                let frame_mask = level_masks[frame_level];
                let frame_dim_bits = self.level_dim_bits[frame_level][free_dim];
                let frame_shifts = self.level_shifts_cache[frame_level][free_dim];
                let frame_child_cell_bits = self.level_child_cell_bits_cache[frame_level][free_dim];
                let mut child_pos = frame.child_pos;
                let mut resumed = false;

                while frame.remaining != 0 {
                    let qi = frame.remaining.trailing_zeros() as usize;
                    frame.remaining &= frame.remaining - 1;
                    let matched = (frame_mask >> qi) & 1 != 0;
                    let next_child_pos = self.child_span_end(child_pos, frame_level + 1);

                    if matched {
                        let child_lo = Self::advance_lo_dim(frame.lo, frame_dim_bits, frame_shifts, qi, frame_child_cell_bits);
                        frame.child_pos = next_child_pos;
                        pos = child_pos;
                        lo = child_lo;
                        level = frame_level + 1;
                        resumed = true;
                        break;
                    }
                    child_pos = next_child_pos;
                }

                if resumed {
                    continue 'outer;
                }
                stack_len -= 1;
            }
        }
    }
}

/// One per open ancestor in [`RadixTreeDfsInterleaved::query_rec`]'s
/// merged constrained/listing descent. `child_pos` is real, necessary
/// state for a frame pushed while still constrained (`level <
/// unconstrained_from`): a resumed one may still have several *unmatched*
/// siblings to walk past, via [`RadixTreeDfsInterleaved::child_span_end`],
/// before its next match or exhaustion. A frame pushed while listing
/// (`level >= unconstrained_from`) leaves `child_pos` unused -- every
/// existing child matches there, so resuming needs only whatever position
/// DFS order already propagated into `pos`. `lo` is a single `u64`, not
/// `[u64; NUM_DIMS]`: see
/// [`RadixTreeDfsInterleaved::advance_lo_dim`] for why every dimension
/// except `free_dim` is dead state here, not merely frozen.
#[derive(Clone, Copy)]
struct ConstrainedFrame {
    lo: u64,
    remaining: u64,
    child_pos: usize,
}

impl ConstrainedFrame {
    const EMPTY: ConstrainedFrame = ConstrainedFrame { lo: 0, remaining: 0, child_pos: 0 };
}

/// Recursively partitions `points` (in place, via `scratch`) and appends
/// each visited node's record (8 bytes, or 12 for a far node -- see the
/// module doc comment) directly to `combined`, in DFS preorder. Returns
/// this node's own subtree's total byte span (itself included), which the
/// caller uses to fill in its *own* far-node skip placeholder (if it has
/// one) after every child returns.
#[allow(clippy::too_many_arguments)]
fn build_interleaved_rec(
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: u32,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    combined: &mut Vec<u8>,
    node_count: &mut u64,
) -> u32 {
    let is_last = level == height as usize - 1;
    let is_far_node = level + 2 < height as usize;
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
    push_u64(combined, existence_word);
    *node_count += 1;
    let skip_placeholder_pos = if is_far_node { Some(push_u32_placeholder(combined)) } else { None };

    let mut total_bytes: u32 = WORD_RECORD_BYTES as u32 + if is_far_node { SKIP_RECORD_BYTES as u32 } else { 0 };
    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeDfsInterleaved::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_bytes = build_interleaved_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, combined, node_count);
                total_bytes += child_bytes;
            }
        }
    }
    if let Some(p) = skip_placeholder_pos {
        write_u32_at(combined, p, total_bytes);
    }
    total_bytes
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

impl MemSize for RadixTreeDfsInterleaved {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.combined)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfsInterleaved {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("combined (8B/node + 4B/far-node, inline, DFS order)", crate::common::metrics::vec_bytes(&self.combined)),
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

    fn cross_check_tree(tree: &RadixTreeDfsInterleaved, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        let tree = RadixTreeDfsInterleaved::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeDfsInterleaved::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeDfsInterleaved::build(&triples, RadixTreeDfsInterleaved::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeDfsInterleaved::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeDfsInterleaved::build(&triples, RadixTreeDfsInterleaved::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeDfsInterleaved::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsInterleaved::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsInterleaved::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsInterleaved::default_split());
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
                let tree = RadixTreeDfsInterleaved::build_with_order(&triples, RadixTreeDfsInterleaved::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeDfsInterleaved::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeDfsInterleaved::build(&triples, RadixTreeDfsInterleaved::default_split());
        let dynamic_tree = RadixTreeDfsInterleaved::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfsInterleaved::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    /// Exercises deep, multi-level "far" nesting specifically -- a wide,
    /// deep tree so some far nodes' skip values span other far nodes'
    /// entire 12-byte records nested inside them, not just leaves/leaf-
    /// parents, stressing the inline-skip chain across several levels.
    #[test]
    fn deep_far_nesting_navigates_correctly() {
        let triples = uniform(500, 20, 500, 6000, 999);
        let naive = NaiveIndex::build(&triples);
        let tree = RadixTreeDfsInterleaved::build_dynamic(&triples);
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
