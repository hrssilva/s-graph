//! A topology-*encoding* variant of [`crate::common::radixtree::RadixTree`],
//! inspired by Carmona & Manzini, "Depth first representations of
//! k²-trees" (2025)'s Balanced Parenthesis (BP) representation.
//!
//! # Why not a literal balanced-parenthesis encoding
//!
//! The paper's BP representation encodes a *general* ordered tree: each
//! node contributes exactly one `(` / `)` pair per *actual* child, with no
//! notion of a fixed slot per child (navigation to the i-th child means
//! "the i-th `(` inside this node's span," found via `find_close` jumps).
//! That's a clean fit for a tree where children are addressed purely by
//! sequence. Ours aren't: a child is addressed by a *specific slot*
//! `qi ∈ [0, 64)`, extracted from the query's own coordinate bits, and the
//! existence check ("does slot `qi` exist under this node") needs to
//! answer that *positional* question, not just "does an i-th child exist."
//! A literal parenthesis sequence doesn't carry slot identity at all
//! without extra bookkeeping equivalent to what this variant stores
//! directly, so this implements the *idea* the paper's BP representation
//! is really chasing -- compact encoding when a node's actual branching is
//! far below the maximum -- adapted to our positional addressing:
//!
//! Instead of the baseline's flat 64-bit existence *word* (always 8 bytes,
//! regardless of how many of the 64 possible children actually exist),
//! each node stores a **sorted list of its existing slot indices**
//! (`u8` each) plus an offset/count pair locating that list. For a node
//! with `K` actual children this costs `K + 5` bytes (list + `u32` offset
//! + `u8` count) against the baseline's flat 8 bytes -- a real win while
//! `K` is small, a real loss once `K` gets large (the offset alone costs
//! half of what the baseline's whole word costs). Existence-checking
//! becomes a binary search over the slot list (O(log K) comparisons)
//! instead of one bit-test.
//!
//! Like [`crate::common::radixtree_dfs`], this stores nodes in DFS/preorder
//! order with a `subtree_node_count` skip value per node for navigation
//! (same O(preceding existing siblings) cost, same reasoning for why: no
//! per-node existence word means no rank-based O(1) jump is available
//! either way, so this variant and the DFS one share that exact tradeoff --
//! the two are safe to compare head-to-head on navigation cost, isolating
//! *only* the existence-encoding difference between them).
//!
//! # How this differs from the other three structures
//!
//! - **vs. Mk²-tree/IK²-tree**: up to [`Self::BASE`] (6) bits of combined
//!   branching per step across all three dimensions, same as baseline.
//! - **vs. Omnitree**: schedule fixed in advance from `dimension_sizes`
//!   alone; coordinates stay fully implicit.
//! - **vs. [`crate::common::radixtree_dfs`]**: identical DFS-order storage
//!   and skip-based navigation; the *only* difference is how "which
//!   children exist" is encoded -- a sorted slot list here, a flat 64-bit
//!   word there -- isolating that one variable for comparison.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;

impl crate::common::radixtree::RadixTreeApi for RadixTreeBp {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeBp::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeBp::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeBp::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeBp::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeBp::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeBp::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeBp::height(self)
    }
}

pub struct RadixTreeBp {
    /// Concatenated per-node sorted lists of existing slot indices
    /// (`0..64`), in DFS/preorder node order.
    slots: Vec<u8>,
    /// Per node (DFS order, index-aligned with `slot_count` and
    /// `subtree_node_count`): start offset into `slots`.
    slot_offset: Vec<u32>,
    /// Per node: how many entries this node's own slot list has.
    slot_count: Vec<u8>,
    /// Per node: total node count of this node's own entire subtree
    /// (itself included) -- the skip value that makes navigation possible
    /// without a rank-based structure. `0` (unused) for leaves.
    subtree_node_count: Vec<u32>,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    dim_order: [usize; NUM_DIMS],
    single_point_exists: bool,
    height: u32,
}

impl RadixTreeBp {
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
             more than K_SQUARE ({K_SQUARE}) children, overflowing the fixed 64-way branching \
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
            return RadixTreeBp {
                slots: Vec::new(),
                slot_offset: Vec::new(),
                slot_count: Vec::new(),
                subtree_node_count: Vec::new(),
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
        let mut slots = Vec::new();
        let mut slot_offset = Vec::new();
        let mut slot_count = Vec::new();
        let mut subtree_node_count = Vec::new();
        build_bp_rec(
            &mut points,
            &mut scratch,
            [0; NUM_DIMS],
            0,
            height,
            &level_dim_bits,
            &level_shifts_cache,
            &level_child_cell_bits_cache,
            &mut slots,
            &mut slot_offset,
            &mut slot_count,
            &mut subtree_node_count,
        );
        // Same reasoning as `radixtree_dfs`'s equivalent trim -- recursive
        // push-based growth leaves real doubling slack, trimmed now that
        // construction is done.
        slots.shrink_to_fit();
        slot_offset.shrink_to_fit();
        slot_count.shrink_to_fit();
        subtree_node_count.shrink_to_fit();

        RadixTreeBp {
            slots,
            slot_offset,
            slot_count,
            subtree_node_count,
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
        self.subtree_node_count.len() as u64
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
    fn slots_of(&self, node_index: usize) -> &[u8] {
        let start = self.slot_offset[node_index] as usize;
        let count = self.slot_count[node_index] as usize;
        &self.slots[start..start + count]
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
        let mut node_index = 0usize;
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
            let slots = self.slots_of(node_index);
            let pos = match slots.binary_search(&(qi as u8)) {
                Ok(p) => p,
                Err(_) => return false,
            };
            if is_last {
                return true;
            }
            // Skip past `pos` earlier existing siblings via their stored
            // subtree sizes -- same O(preceding siblings) navigation cost
            // as radixtree_dfs, for the same underlying reason (no
            // rank-based structure to jump directly with).
            let mut child_node_index = node_index + 1;
            for _ in 0..pos {
                child_node_index += self.subtree_node_count[child_node_index] as usize;
            }
            node_index = child_node_index;
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
        self.query_rec(0, [0; NUM_DIMS], 0, &level_masks[..self.height as usize], free_dim, &mut out);
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
        self.query_rec(0, [0; NUM_DIMS], 0, &level_masks[..self.height as usize], target_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The slot list already IS "existing children in ascending qi order"
    /// -- no `trailing_zeros`/mask-and-word trick needed to enumerate
    /// them, just a linear walk of the (already sorted, already only the
    /// existing ones) list. Matched entries get recursed into; every entry
    /// (matched or not) advances the running node-index via its stored
    /// subtree size, exactly like `radixtree_dfs`.
    fn query_rec(&self, node_index: usize, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], free_dim: usize, out: &mut Vec<u32>) {
        let is_last = level == self.height as usize - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = level_masks[level];
        let slots = self.slots_of(node_index);

        if is_last {
            for &qi_u8 in slots {
                let qi = qi_u8 as usize;
                if (mask >> qi) & 1 != 0 {
                    let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                    out.push(child_lo[free_dim] as u32);
                }
            }
            return;
        }

        let mut child_node_index = node_index + 1;
        for &qi_u8 in slots {
            let qi = qi_u8 as usize;
            if (mask >> qi) & 1 != 0 {
                let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                self.query_rec(child_node_index, child_lo, level + 1, level_masks, free_dim, out);
            }
            child_node_index += self.subtree_node_count[child_node_index] as usize;
        }
    }
}

/// Recursive DFS-order builder -- same counting-sort partitioning as
/// `radixtree_dfs`'s, differing only in what gets recorded per node: a
/// sorted slot list (+ offset/count) instead of a flat existence word.
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
    slots: &mut Vec<u8>,
    slot_offset: &mut Vec<u32>,
    slot_count: &mut Vec<u8>,
    subtree_node_count: &mut Vec<u32>,
) -> u32 {
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

    let my_node_index = slot_offset.len();
    let my_slot_start = slots.len() as u32;
    let mut my_count = 0u8;
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            slots.push(qi as u8);
            my_count += 1;
        }
    }
    slot_offset.push(my_slot_start);
    slot_count.push(my_count);
    subtree_node_count.push(0); // placeholder, filled in once every child returns

    let mut total_nodes = 1u32;
    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeBp::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_nodes = build_bp_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, slots, slot_offset, slot_count, subtree_node_count);
                total_nodes += child_nodes;
            }
        }
    }
    subtree_node_count[my_node_index] = total_nodes;
    total_nodes
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

impl MemSize for RadixTreeBp {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.slots)
            + crate::common::metrics::vec_bytes(&self.slot_offset)
            + crate::common::metrics::vec_bytes(&self.slot_count)
            + crate::common::metrics::vec_bytes(&self.subtree_node_count)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeBp {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("slots (1 byte/existing child, DFS order)", crate::common::metrics::vec_bytes(&self.slots)),
            ("slot_offset (O(nodes))", crate::common::metrics::vec_bytes(&self.slot_offset)),
            ("slot_count (O(nodes))", crate::common::metrics::vec_bytes(&self.slot_count)),
            ("subtree_node_count (O(nodes), navigation)", crate::common::metrics::vec_bytes(&self.subtree_node_count)),
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

    fn cross_check_tree(tree: &RadixTreeBp, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        let tree = RadixTreeBp::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeBp::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeBp::build(&triples, RadixTreeBp::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeBp::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeBp::build(&triples, RadixTreeBp::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeBp::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeBp::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeBp::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeBp::default_split());
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
                let tree = RadixTreeBp::build_with_order(&triples, RadixTreeBp::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeBp::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeBp::build(&triples, RadixTreeBp::default_split());
        let dynamic_tree = RadixTreeBp::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeBp::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }
}
