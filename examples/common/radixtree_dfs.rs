//! A physical-layout *and* navigation variant of
//! [`crate::common::radixtree::RadixTree`]: node-words are stored in
//! DFS/preorder visitation order instead of BFS/level order, following
//! Carmona & Manzini, "Depth first representations of k²-trees" (2025).
//!
//! # Why DFS order at all
//!
//! The baseline's BFS/level-order storage means a single query's visited
//! nodes (one per level) can land in far-apart regions of the array once
//! the tree doesn't fit in cache -- exactly the problem the paper names:
//! "nodes which are close in the tree can be stored far apart... for some
//! computation this layout can generate a large number of cache misses."
//! DFS/preorder storage puts a whole subtree contiguously, which is more
//! cache-friendly for anything that visits a node and soon after visits
//! its descendants (which a root-to-leaf query descent does).
//!
//! # The real cost: this is the paper's "Enriched Depth First" (EDF), not
//! "Plain Depth First" (PDF)
//!
//! Plain DFS storage (write each node's word in preorder, nothing else)
//! loses O(1) navigation entirely: finding a specific child means walking
//! past every preceding sibling's *entire subtree content*, since there's
//! no more rank-based random access into a BFS-ordered bitmap. That's
//! catastrophic for point/pattern queries (this project's actual
//! workload), so this variant implements the paper's fix: an extra
//! `subtree_word_count` per *far* node (see below; this node's whole
//! subtree size, itself included, in words), letting navigation *skip* a
//! subtree in O(1) instead of visiting it. Reaching a specific child still
//! costs O(number of earlier existing siblings) -- each one's stored size
//! has to be read and added -- rather than the baseline's O(1) rank1 jump
//! straight to any child. That's the genuine, honest tradeoff this variant
//! measures: better locality, worse per-level navigation, real extra
//! storage (one `u32` per *far* node -- smaller than
//! [`crate::common::radixtree_veb`]'s permutation-table approach, which
//! needs both a permutation *and* a full duplicate copy of every word).
//!
//! # Leaves and leaf-parents never need a stored subtree size
//!
//! Leaf-ness is a *level* property, not a per-node one: only the deepest
//! level has leaves, so a node's *direct children* are either all leaves or
//! all non-leaf together, never mixed, and navigation already knows the
//! current level as it descends -- the same `level + 1 == height - 1` check
//! that detects "the node I'm standing on is a leaf" tells a sibling-skip
//! loop, *before reading anything*, whether the children it's about to walk
//! are leaves. When they are, each one's subtree is itself alone -- word
//! count `1`, a compile-time constant -- so the skip is a plain
//! `count_ones()` over the already-loaded existence word.
//!
//! The same trick extends one level further: a **leaf-parent** (`level + 2
//! == height - 1`, i.e. *its* children are leaves) has a subtree word count
//! of `1 + count_ones(its own existence word)` -- still derivable with zero
//! extra storage, just one extra read of `words` instead of
//! `subtree_word_count`. So `subtree_word_count` holds an entry only for
//! **far** nodes (`level + 2 < height`, i.e. neither a leaf nor a
//! leaf-parent) -- on this project's benchmark datasets, leaves and
//! leaf-parents together are the large majority of all nodes, so this cuts
//! the array to a small fraction of one-entry-per-node.
//!
//! The cost: with leaf/leaf-parent entries removed, a far node's position
//! in `words` (DFS-preorder, all nodes) and its position in the compacted
//! `subtree_word_count` (DFS-preorder, far nodes only) are different
//! indices -- unlike the pure level check above, there's no way to derive
//! one from the other without *some* extra structure, since "word count of
//! a subtree" and "far-node count of a subtree" are different,
//! non-derivable aggregates over the same (data-dependent) shape. `is_far`
//! (a rank-augmented bitvector, one bit per node) bridges the two:
//! `rank1(word_index)` is exactly "how many far nodes precede this one,"
//! i.e. this node's own index into the compacted array -- the same
//! rank1-translates-DFS-index-to-compact-index trick tried earlier for
//! *all* non-leaf nodes (see git history), which cost a real ~22% slowdown
//! because it fired on every non-leaf sibling skip. Restricting it to only
//! the far case -- reached far less often, since leaves and leaf-parents
//! dominate node counts in a wide/shallow tree -- is the bet this version
//! makes instead.
//!
//! # How this differs from the other three structures
//!
//! - **vs. Mk²-tree/IK²-tree (k=2, 2 bits/step across S,O only)**: this
//!   structure spends up to [`Self::BASE`] (6) bits per step, distributed
//!   across all three dimensions via `split`, giving up to 64-way branching
//!   per node instead of 4-way.
//! - **vs. Omnitree**: Omnitree picks which dimensions to split *per node*,
//!   adaptively from the data, which forces it to store an explicit
//!   dimension-select label per node and explicit leaf coordinates. Here
//!   the schedule is fixed *in advance* from `dimension_sizes` alone, so
//!   coordinates stay fully implicit, just like the baseline.
//! - **vs. [`crate::common::radixtree_veb`]**: VEB keeps the baseline's
//!   rank1-based navigation completely unchanged and only reorders where
//!   words physically live (via a permutation table on top of an
//!   unmodified BFS-ordered `t`/`l`, needed because rank1 depends on that
//!   order). This structure instead replaces navigation *and* storage
//!   order together, trading rank1's O(1) jump for O(preceding siblings)
//!   skip-based navigation, at roughly a third of VEB's extra memory.

use crate::common::bitvector::BitVector;
use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const WORD_BITS: u64 = K_SQUARE as u64;
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfs {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfs::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfs::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfs::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfs::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfs::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfs::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfs::height(self)
    }
}

pub struct RadixTreeDfs {
    /// One 64-bit existence word per node, in DFS/preorder visitation
    /// order. Unlike the baseline's separate `t`/`l` (which relied on BFS
    /// order grouping every same-level node together to make a clean
    /// T/L split), DFS order naturally interleaves levels -- a node's word
    /// is immediately followed by its own first child's word, wherever
    /// that child's level is -- so there's one combined array instead.
    words: Vec<u64>,
    /// One entry per *far* node only (`level + 2 < height`, i.e. neither a
    /// leaf nor a leaf-parent -- see the module doc comment): that node's
    /// own subtree word count (itself included), in the same relative
    /// DFS-preorder as `words` restricted to far nodes. `is_far`
    /// translates a `words`-index into the matching index here.
    subtree_word_count: Vec<u32>,
    /// One bit per node, in `words`' own DFS-preorder: `1` if that node is
    /// far (has a `subtree_word_count` entry). `rank1(word_index)` gives
    /// the number of far nodes strictly before `word_index` -- i.e. that
    /// index's own position in the compacted `subtree_word_count`, when
    /// the node at `word_index` is itself far.
    is_far: BitVector,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    dim_order: [usize; NUM_DIMS],
    single_point_exists: bool,
    height: u32,
}

impl RadixTreeDfs {
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
            return RadixTreeDfs {
                words: Vec::new(),
                subtree_word_count: Vec::new(),
                is_far: BitVector::new(),
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
        let mut subtree_word_count = Vec::new();
        let mut is_far = BitVector::new();
        build_dfs_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, height, &level_dim_bits, &level_shifts_cache, &level_child_cell_bits_cache, &mut words, &mut subtree_word_count, &mut is_far);
        // Recursive push-based growth (no way to know the final node count
        // up front) leaves real doubling slack in both arrays -- trim it
        // now that construction is done, same reasoning as
        // `BitVector::finalize`.
        words.shrink_to_fit();
        subtree_word_count.shrink_to_fit();
        is_far.finalize();

        RadixTreeDfs {
            words,
            subtree_word_count,
            is_far,
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

    /// `lo`, advanced into the sub-cell `qi` selects, for every dimension.
    /// Identical to the baseline's -- purely arithmetic, doesn't touch
    /// storage order.
    #[inline]
    fn advance_lo(mut lo: [u64; NUM_DIMS], dim_bits: [u8; NUM_DIMS], shifts: [u8; NUM_DIMS], qi: usize, child_cell_bits: [u8; NUM_DIMS]) -> [u64; NUM_DIMS] {
        for d in 0..NUM_DIMS {
            let mask = (1usize << dim_bits[d]) - 1;
            let sub = (qi >> shifts[d]) & mask;
            lo[d] += (sub as u64) << child_cell_bits[d];
        }
        lo
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
            // Skip past every existing sibling before `qi` (their subtree
            // sizes are stored, so this is O(preceding siblings), not O(1)
            // like the baseline's rank1 jump -- the real navigation cost
            // this layout trades away for locality).
            let mut child_word_index = word_index + 1;
            let earlier = word & ((1u64 << qi) - 1);
            if level + 1 == self.height as usize - 1 {
                // Children are leaves: every one is exactly 1 word, no
                // `subtree_word_count` read needed at all.
                child_word_index += earlier.count_ones() as usize;
            } else if level + 2 == self.height as usize - 1 {
                // Children are leaf-parents: each one's subtree size is
                // `1 + (its own existing-child count)`, derivable straight
                // from its own existence word -- still no
                // `subtree_word_count` read needed.
                let mut earlier = earlier;
                while earlier != 0 {
                    earlier &= earlier - 1;
                    child_word_index += 1 + self.words[child_word_index].count_ones() as usize;
                }
            } else {
                // Children are far nodes: `subtree_word_count` only has
                // entries for far nodes, so `is_far.rank1` translates this
                // `words`-index into its compacted position.
                let mut earlier = earlier;
                while earlier != 0 {
                    earlier &= earlier - 1;
                    let subtree_index = self.is_far.rank1(child_word_index as u64) as usize;
                    child_word_index += self.subtree_word_count[subtree_index] as usize;
                }
            }
            word_index = child_word_index;
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

    /// Per-level masks for a query following one path through the bound
    /// dimensions -- pure arithmetic (`dim_bits`/`shifts`/`child_cell_bits`
    /// and the bound values alone), identical to the baseline's; doesn't
    /// touch storage order or navigation, so it's unmodified here.
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

    /// Unlike the baseline (which only ever visits *matched* candidates
    /// via `word & mask` + `trailing_zeros`), this has to walk every
    /// *existing* child in ascending `qi` order regardless of whether it
    /// matches, since each one's stored subtree size is what advances the
    /// position to the next sibling -- only matched children get
    /// recursed into, but unmatched ones still have to be walked past.
    fn query_rec(&self, word_index: usize, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], free_dim: usize, out: &mut Vec<u32>) {
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

        let children_are_leaves = level + 1 == self.height as usize - 1;
        let children_are_leaf_parents = level + 2 == self.height as usize - 1;
        let mut child_word_index = word_index + 1;
        let mut existing = word;
        while existing != 0 {
            let qi = existing.trailing_zeros() as usize;
            existing &= existing - 1;
            if (mask >> qi) & 1 != 0 {
                let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                self.query_rec(child_word_index, child_lo, level + 1, level_masks, free_dim, out);
            }
            if children_are_leaves {
                // Every leaf is exactly 1 word -- no lookup needed.
                child_word_index += 1;
            } else if children_are_leaf_parents {
                // Each leaf-parent's subtree size is derivable straight
                // from its own existence word -- no `subtree_word_count`
                // read needed.
                child_word_index += 1 + self.words[child_word_index].count_ones() as usize;
            } else {
                let subtree_index = self.is_far.rank1(child_word_index as u64) as usize;
                child_word_index += self.subtree_word_count[subtree_index] as usize;
            }
        }
    }
}

/// Recursively partitions `points` (in place, via `scratch`) and appends
/// one DFS-preorder-ordered word per visited node directly to `words` (with
/// `is_far` kept index-aligned to it, and `subtree_word_count` getting an
/// entry only for far nodes -- see the module doc comment) -- the same
/// counting-sort partitioning the baseline's frontier loop does per level,
/// just invoked recursively (depth-first) instead of iteratively
/// (breadth-first), which is what makes the resulting storage order DFS
/// instead of BFS in the first place. Returns this node's own subtree's
/// total word count (itself included), which the caller uses to fill in
/// `subtree_word_count` for its *own* node (if far) after every child
/// returns.
#[allow(clippy::too_many_arguments)]
fn build_dfs_rec(
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: u32,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    words: &mut Vec<u64>,
    subtree_word_count: &mut Vec<u32>,
    is_far: &mut BitVector,
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
    words.push(existence_word);
    is_far.push(is_far_node);
    // Only far nodes get a `subtree_word_count` slot at all (see the
    // module doc comment) -- `my_subtree_index` is `None` for a leaf or
    // leaf-parent, and the final write below is skipped for it.
    let my_subtree_index = if is_far_node {
        let idx = subtree_word_count.len();
        subtree_word_count.push(0); // placeholder, filled in below once every child's size is known
        Some(idx)
    } else {
        None
    };

    let mut total_words = 1u32;
    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeDfs::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_words = build_dfs_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, words, subtree_word_count, is_far);
                total_words += child_words;
            }
        }
    }
    if let Some(idx) = my_subtree_index {
        subtree_word_count[idx] = total_words;
    }
    total_words
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

impl MemSize for RadixTreeDfs {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.words)
            + crate::common::metrics::vec_bytes(&self.subtree_word_count)
            + self.is_far.mem_size_bytes()
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfs {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("words (64-bit-per-node, DFS order)", crate::common::metrics::vec_bytes(&self.words)),
            ("subtree_word_count (O(far nodes), navigation)", crate::common::metrics::vec_bytes(&self.subtree_word_count)),
            ("is_far (rank bitvector, O(nodes))", self.is_far.mem_size_bytes()),
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

    fn cross_check_tree(tree: &RadixTreeDfs, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        let tree = RadixTreeDfs::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeDfs::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeDfs::build(&triples, RadixTreeDfs::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeDfs::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeDfs::build(&triples, RadixTreeDfs::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeDfs::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfs::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfs::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfs::default_split());
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
                let tree = RadixTreeDfs::build_with_order(&triples, RadixTreeDfs::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeDfs::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeDfs::build(&triples, RadixTreeDfs::default_split());
        let dynamic_tree = RadixTreeDfs::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfs::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }
}
