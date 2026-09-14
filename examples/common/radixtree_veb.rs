//! A physical-layout variant of [`crate::common::radixtree::RadixTree`]:
//! *identical* logical structure, schedule, bit-packing, and rank-based
//! navigation formula -- the only thing that changes is *where in memory*
//! each node's word physically lives.
//!
//! # The problem this targets
//!
//! The baseline structure stores node-words in pure BFS/level order: all of
//! level 0's words, then all of level 1's, etc. Carmona & Manzini,
//! "Depth first representations of k²-trees" (2025), name this exact
//! layout's problem directly: "nodes which are close in the tree can be
//! stored far apart: for some computation this layout can generate a large
//! number of cache misses." A single query's ~5 visited nodes (one per
//! level) can land in five far-apart regions of the array once the tree is
//! large enough not to fit in cache.
//!
//! # Van Emde Boas layout, not DFS order
//!
//! The Carmona & Manzini paper's own fix (DFS/preorder storage) trades
//! locality for navigation: without extra per-node skip-size bookkeeping,
//! finding a specific child requires walking past every preceding sibling
//! subtree's full content (their "Plain Depth First" representation), and
//! their navigation-preserving fix ("Enriched Depth First" / "Balanced
//! Parenthesis") costs real extra bits (BP: roughly *2x* the canonical
//! space) -- see [`crate::common::radixtree_dfs`] and
//! [`crate::common::radixtree_bp`] for those two, implemented as separate
//! variants specifically to isolate that tradeoff.
//!
//! Van Emde Boas layout takes a different approach: recursively split the
//! tree at half its height, lay the top half out first, then each bottom
//! subtree contiguously after it, recursively. This keeps O(1)-style
//! rank-based navigation working *unmodified* -- the recursive splitting is
//! a pure *permutation* of where words live, not a different encoding of
//! which nodes exist -- at the cost of an explicit permutation table
//! (`t_physical_index`/`l_physical_index`, one `u32` per node) to translate
//! between "the Nth existing node in BFS/rank order" (which the schedule
//! and rank formula still address) and "where that node's word physically
//! sits" (VEB order). That's the honest cost here: unlike VEB's classical
//! zero-extra-space presentation for plain binary search trees, our rank1
//! formula's correctness depends on nodes being *counted* in BFS order --
//! separating "counted in" from "physically stored in" order is exactly
//! what the permutation table buys, and it roughly doubles `t`'s memory
//! footprint (the original BFS-ordered `t`, needed for `rank1`, plus a full
//! VEB-reordered copy of the same words for fetching). `l` is never
//! `rank1`-queried (a leaf hit only needs `get`/`word_at`, both of which
//! `l_veb_words` already serves), so its BFS-order copy is discarded right
//! after building the permutation -- unlike `t`, there is no reason to keep
//! two identical-content copies of `l` alive.
//!
//! # How this differs from the other three structures
//!
//! - **vs. Mk²-tree/IK²-tree (k=2, 2 bits/step across S,O only)**: this
//!   structure spends up to [`Self::BASE`] (6) bits per step, distributed
//!   across all three dimensions via `split`, giving up to 64-way branching
//!   per node instead of 4-way -- a shallower tree, and a node's own
//!   existence-bit population count is one hardware `popcount` over a
//!   native word instead of a multi-word sampled-rank lookup.
//! - **vs. Omnitree**: Omnitree picks which dimensions to split *per node*,
//!   adaptively from the data, which forces it to store an explicit
//!   dimension-select label per node and (since the schedule can't be
//!   replayed without seeing the data) explicit leaf coordinates. Here the
//!   schedule is fixed *in advance* from `dimension_sizes` alone -- a
//!   dimension stops contributing bits once its own domain is fully
//!   resolved, deterministically, so a query can recompute it from depth
//!   alone. That means coordinates stay fully implicit (never stored, just
//!   like base k²-tree) while still avoiding Omnitree's "wasted bits on an
//!   exhausted dimension" problem.
//!
//! # Node layout and the child-offset formula
//!
//! Per node, `t`/`l` each reserve exactly [`WORD_BITS`] (64) bits,
//! regardless of that *level*'s actual branching factor (which varies
//! deterministically by depth as dimensions exhaust, but is uniform across
//! all nodes at a given depth) -- unused high bits are always 0 and never
//! contribute to rank, so this is exactly base k²-tree's own
//! `rank1(x+1)*k²` trick with `k²` replaced by a constant 64:
//!
//! `child_word_start = rank1(bit_pos + 1) * 64` (inclusive rank; minus
//! `t.len()` when the child's word lands in `l` instead of `t`, the same
//! virtual-T:L-concatenation correction base k²-tree needs).
//!
//! This works cleanly here specifically *because* the per-node slot size
//! (64 bits) is uniform regardless of depth or data -- unlike IK²-tree,
//! where slot width varies per node based on the active-predicate count and
//! genuinely needs the O(height) region-offset bookkeeping worked out for
//! that structure.

use crate::common::bitvector::BitVector;
use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const WORD_BITS: u64 = K_SQUARE as u64;
const NUM_DIMS: usize = 3; // S, P, O
/// Upper bound on tree height for a stack-allocated per-level scratch array
/// (see [`RadixTreeVeb::precompute_level_masks`]) -- comfortably covers any
/// `u32`-domain dataset (at most ~33 bits per dimension) even under a
/// pathologically thin split like `[1,1,1]` that only consumes 1 bit/level.
const MAX_HEIGHT: usize = 64;

impl crate::common::radixtree::RadixTreeApi for RadixTreeVeb {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeVeb::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeVeb::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeVeb::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeVeb::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeVeb::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeVeb::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeVeb::height(self)
    }
}

pub struct RadixTreeVeb {
    t: BitVector,
    /// Per-level, per-dimension bit counts consumed at that depth (0 once a
    /// dimension is exhausted). Precomputed once from `dimension_sizes` and
    /// `split` -- deterministic, not data-adaptive.
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    /// `level_shifts(level_dim_bits[d], dim_order)`, precomputed once per
    /// level at construction time instead of recomputed on every recursive
    /// query call (it depends only on the level and the fixed schedule).
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    /// Remaining bit-width per dimension *after* this level -- i.e. `log2`
    /// of every node's child cell size at this level, for each dimension.
    /// Every cell size in this structure is a power of two by construction
    /// (`1 << bits`, repeatedly right-shifted), so this exponent is all
    /// that's needed: `(coord - lo) / cell` becomes `(coord - lo) >>
    /// child_cell_bits[d]`, and `sub * cell` becomes `sub <<
    /// child_cell_bits[d]` -- a division/multiplication by a *runtime*
    /// power of two (which the compiler can't turn into a shift on its own,
    /// since it isn't a compile-time constant) replaced by an explicit
    /// shift, with the exponent itself precomputed once here instead of
    /// re-derived from a `cell` value threaded through every call.
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    /// Which logical dimension (0=S,1=P,2=O) is packed into the high/mid/low
    /// bits of each node's word: `dim_order[0]` is packed MSB-first,
    /// `dim_order[2]` LSB-first. Purely an internal bit-layout choice --
    /// `query_two_bound`/`query_one_bound`'s dimension indices always refer
    /// to logical S/P/O regardless of this. Default `[0,1,2]` ("SPO").
    dim_order: [usize; NUM_DIMS],
    /// Degenerate case (every dimension already fully resolved, i.e. all
    /// `dimension_sizes` are 1): no levels at all, just whether the single
    /// possible point `(0,0,0)` was present.
    single_point_exists: bool,
    height: u32,
    /// VEB-reordered copy of every T node's own word (same content as `t`,
    /// just physically permuted) -- `t` itself stays in BFS order
    /// untouched, since `child_word_start`'s `rank1` formula depends on
    /// that order; this is purely for the *fetch* step. `l_veb_words`'s
    /// BFS-order counterpart doesn't exist at all: `l` is never
    /// `rank1`-queried, so there's no reason to keep a second copy of it.
    t_veb_words: Vec<u64>,
    l_veb_words: Vec<u64>,
    /// `t_physical_index[bfs_word_index] = position in t_veb_words` -- the
    /// permutation translating a BFS/rank-order node index into where its
    /// word physically lives. One `u32` per node: the real, explicit
    /// memory cost of separating "counted in" order from "stored in" order
    /// (see the module doc comment).
    t_physical_index: Vec<u32>,
    l_physical_index: Vec<u32>,
}

impl RadixTreeVeb {
    /// `split[d]`: nominal bits dimension `d` contributes per step while
    /// still active (before being capped by its own remaining bits). The
    /// stub's own default is `BASE / N` per dimension; callers can pass a
    /// different split (e.g. weighting predicate to exhaust first).
    pub fn default_split() -> [u8; NUM_DIMS] {
        [BASE / NUM_DIMS as u8; NUM_DIMS]
    }

    /// Static split: every active dimension always contributes exactly
    /// `split[d]` bits per step (capped by its own remaining bits),
    /// regardless of how many other dimensions are still active. Once a
    /// dimension exhausts, its bit budget is simply unused for the rest of
    /// the tree -- e.g. with `split=[2,2,2]`, once the predicate dimension
    /// (few distinct values) exhausts, remaining levels branch only 4-way
    /// (2 bits from S, 2 from O) inside every node's full 64-bit word,
    /// wasting 60 of 64 bits per node at every one of those levels.
    pub fn build(triples: &[Triple], split: [u8; NUM_DIMS]) -> Self {
        Self::build_with_order(triples, split, [0, 1, 2])
    }

    /// Same as [`Self::build`], but `dim_order` chooses which logical
    /// dimension is packed into the high/mid/low bits of each node's word
    /// (e.g. `[1,0,2]` = "PSO", predicate in the high bits instead of the
    /// default `[0,1,2]` = "SPO"). This is purely a bit-layout choice IK²-
    /// tree has no equivalent of: predicate isn't a position in its
    /// traversal order at all there (it's a width overlay on the S,O
    /// skeleton), so there's no "where" for it to be reordered to.
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

    /// Dynamic split: at every level, the full [`BASE`] (6) bit budget is
    /// "water-filled" across whichever dimensions are *still* active,
    /// evenly, redistributing a dimension's unused share (when it's about
    /// to exhaust and needs fewer bits than its even share) to the other
    /// still-active dimensions in the same pass. So once predicate
    /// exhausts, a `[2,2,2]` schedule shifts to `[3,0,3]` for S/O instead
    /// of staying at `[2,0,2]` -- every node keeps using the full 64-bit
    /// word's worth of branching for as long as possible.
    pub fn build_dynamic(triples: &[Triple]) -> Self {
        Self::build_dynamic_with_order(triples, [0, 1, 2])
    }

    /// Same as [`Self::build_dynamic`], but with an explicit dimension
    /// packing order -- see [`Self::build_with_order`].
    pub fn build_dynamic_with_order(triples: &[Triple], dim_order: [usize; NUM_DIMS]) -> Self {
        let dimension_sizes = dimension_sizes_from(triples);
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order)
    }

    fn build_from_schedule(triples: &[Triple], dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS]) -> Self {
        if level_dim_bits.is_empty() {
            return RadixTreeVeb {
                t: BitVector::new(),
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                dim_order,
                single_point_exists: !triples.is_empty(),
                height: 0,
                t_veb_words: Vec::new(),
                l_veb_words: Vec::new(),
                t_physical_index: Vec::new(),
                l_physical_index: Vec::new(),
            };
        }
        let height = level_dim_bits.len() as u32;
        // Precomputed once: depends only on the schedule and dim_order, not
        // on which node/query is being processed, so every recursive query
        // call just indexes this instead of recomputing it from scratch.
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
        // One scratch buffer for the per-level counting-sort passes below,
        // sized once to the largest any frontier entry's slice can ever be
        // (the whole point set) and reused (via a shrinking prefix) for
        // every entry at every level -- avoids a fresh allocation per
        // frontier entry, which without this would be O(total node count)
        // allocations over the whole build.
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        let mut t = BitVector::new();
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

        // `debug_assert!` in `static_schedule`/`dynamic_schedule` call sites
        // already guards the real precondition this depends on: every
        // level's dim_bits must sum to at most BASE (6), since `branching`
        // (and every node's fixed WORD_BITS-wide slot) assumes that.
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
                // `branching` is at most `K_SQUARE` (64), so these are
                // stack arrays, not heap allocations -- see the scratch
                // buffer comment above for why that matters here.
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

                if is_last {
                    // `l` is a plain word-per-node vector (no rank support
                    // needed, see the module doc comment), so there's no
                    // bit-by-bit push/pad dance -- just assemble the whole
                    // existence word and push it once.
                    let mut l_word = 0u64;
                    for qi in 0..branching {
                        let qstart = entry.start + offsets[qi];
                        let qend = entry.start + offsets[qi + 1];
                        if qstart < qend {
                            l_word |= 1u64 << qi;
                        }
                    }
                    l_words.push(l_word);
                } else {
                    for qi in 0..branching {
                        let qstart = entry.start + offsets[qi];
                        let qend = entry.start + offsets[qi + 1];
                        t.push(qstart < qend);
                    }
                    for _ in branching..(WORD_BITS as usize) {
                        t.push(false); // pad the rest of this node's word
                    }
                    for qi in 0..branching {
                        let qstart = entry.start + offsets[qi];
                        let qend = entry.start + offsets[qi + 1];
                        if qstart < qend {
                            let child_lo = RadixTreeVeb::advance_lo(entry.lo, dim_bits, shifts, qi, child_cell_bits);
                            next_frontier.push(FrontierEntry {
                                start: qstart,
                                end: qend,
                                lo: child_lo,
                            });
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        t.finalize();
        l_words.shrink_to_fit();

        let (t_veb_words, t_physical_index, l_veb_words, l_physical_index) = compute_veb_reorder(&t, &l_words, height);

        RadixTreeVeb {
            t,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            dim_order,
            single_point_exists: false,
            height,
            t_veb_words,
            l_veb_words,
            t_physical_index,
            l_physical_index,
        }
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn node_word_count(&self) -> u64 {
        self.t.len() / WORD_BITS + self.l_veb_words.len() as u64
    }

    pub fn domain_size(&self, dim: usize) -> u64 {
        self.dimension_sizes[dim]
    }

    #[inline]
    fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        (s as u64) < self.dimension_sizes[0] && (p as u64) < self.dimension_sizes[1] && (o as u64) < self.dimension_sizes[2]
    }

    /// `lo`, advanced into the sub-cell `qi` (this level's packed child
    /// index) selects, for every dimension. The mask is `0` for a dimension
    /// with `dim_bits[d] == 0` (already fully resolved at an earlier
    /// level), which zeroes `sub` and so leaves `lo[d]` unchanged -- so
    /// unlike a naive per-dimension decode, this doesn't need an `if
    /// dim_bits[d] > 0` guard: the exhausted case already falls out of the
    /// arithmetic. `child_cell_bits[d]` (this level's child cell size, as a
    /// power-of-two exponent) replaces a runtime multiply by a shift --
    /// every cell size in this structure is a power of two by construction.
    #[inline]
    fn advance_lo(mut lo: [u64; NUM_DIMS], dim_bits: [u8; NUM_DIMS], shifts: [u8; NUM_DIMS], qi: usize, child_cell_bits: [u8; NUM_DIMS]) -> [u64; NUM_DIMS] {
        for d in 0..NUM_DIMS {
            let mask = (1usize << dim_bits[d]) - 1;
            let sub = (qi >> shifts[d]) & mask;
            lo[d] += (sub as u64) << child_cell_bits[d];
        }
        lo
    }

    /// Child's own word-slot start, in the virtual `t:l` concatenation --
    /// base k²-tree's `rank1(x+1)*granularity` trick with granularity =
    /// [`WORD_BITS`] instead of `k²`, translated to an `l`-relative offset
    /// when the child lands in the final level.
    #[inline]
    fn child_word_start(&self, bit_pos: u64, next_is_last: bool) -> u64 {
        let virtual_start = self.t.rank1(bit_pos + 1) * WORD_BITS;
        if next_is_last {
            virtual_start - self.t.len()
        } else {
            virtual_start
        }
    }

    /// Fetches a node's own word through the VEB permutation instead of
    /// directly from the BFS-ordered `t`/`l` -- `word_start` is still a
    /// BFS-order bit offset (computed the same way as the baseline, via
    /// `child_word_start`/rank1), only the *physical fetch* is redirected.
    #[inline]
    fn word_at_veb(&self, word_start: u64, is_last: bool) -> u64 {
        let bfs_word_index = (word_start / WORD_BITS) as usize;
        if is_last {
            self.l_veb_words[self.l_physical_index[bfs_word_index] as usize]
        } else {
            self.t_veb_words[self.t_physical_index[bfs_word_index] as usize]
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
            // No `dim_bits[d] > 0` guard needed: for an already-exhausted
            // dimension, `lo[d]` already equals `coords[d]` exactly (its
            // cell has shrunk to size 1), so `sub` is naturally 0 -- and
            // `child_cell_bits[d]` is 0 too, so the shift is a no-op either
            // way.
            let mut qi = 0usize;
            for d in 0..NUM_DIMS {
                let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
                qi |= sub << shifts[d];
            }
            let word = self.word_at_veb(word_start, is_last);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if is_last {
                return true;
            }
            let bit_pos = word_start + qi as u64;
            word_start = self.child_word_start(bit_pos, level + 1 == self.height as usize - 1);
            lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 checked above")
    }

    /// Ascending predicates linking a specific (subject, object) pair --
    /// binds S and O, leaves P free. All three dimensions are genuinely
    /// symmetric in this structure (unlike the k²-tree family, which is
    /// fundamentally 2D over S,O with predicate selecting *which*
    /// tree/interleaving-context), so this is exactly [`Self::query_two_bound`]
    /// with a different pair of dimensions bound -- no special-casing needed.
    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        self.query_two_bound([0, 2], [s, o], 1)
    }

    /// Ascending values of `free_dim` matching both bound dimensions.
    /// `bound_dims`/`bound_vals` name the two fixed dimensions; `free_dim`
    /// is the third. Dimension indices: 0=subject, 1=predicate, 2=object.
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
        // The bound dimensions' local sub-index at a given level is fully
        // determined by the fixed bound values, not by which free-dim branch
        // led here -- so every node visited at a given level shares the same
        // mask. Walk the single (unbranching) bound-dims-only path once up
        // front instead of rebuilding this mask from scratch on every one of
        // the O(height * matches) node visits the recursion below makes.
        let level_masks = self.precompute_level_masks(bound_dims, bound_vals, |d| d == bound_dims[0] || d == bound_dims[1]);
        self.query_rec(0, 0, [0; NUM_DIMS], &level_masks[..self.height as usize], free_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Per-level masks for a query that only ever follows one path through
    /// the bound dimensions (any number of free dimensions may still branch
    /// within a level). `is_bound(d)` reports whether dimension `d` is fixed
    /// (its value looked up positionally in `bound_dims`/`bound_vals`) --
    /// callers with a single bound dimension pass `bound_dims`/`bound_vals`
    /// with only slot 0 populated and `is_bound` checking against it alone.
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
            // Only `is_bound(d)` gates this (not `dim_bits[d] > 0`): for an
            // already-exhausted bound dimension, `lo[d]` already equals
            // `bound_val` exactly, so `sub` is naturally 0 and this is a
            // no-op either way -- but a *free* dimension has no bound_val
            // to read (`bound_val_for` would return `None`), so that gate
            // is load-bearing, not just an optimization.
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

    /// Ascending, deduplicated values of `target_dim` matching `bound_dim`,
    /// with the third dimension left entirely unconstrained -- a
    /// one-bound-dimension scan (unlike [`Self::query_two_bound`]'s two),
    /// needed for predicate-unbound path algorithms.
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

    /// Walks every node whose word (ANDed against `level_masks[level]`) has
    /// at least one surviving candidate bit, pushing `free_dim`'s resolved
    /// value at each leaf hit. Shared by [`Self::query_two_bound`] (one free
    /// dimension, two bound) and [`Self::query_one_bound`] (two free
    /// dimensions, one bound) -- both reduce to exactly this once the
    /// per-level mask already encodes which candidates survive, since
    /// neither caller's recursion body cares how many dimensions the mask
    /// left free, only which one to read the answer back from.
    fn query_rec(&self, level: usize, word_start: u64, lo: [u64; NUM_DIMS], level_masks: &[u64], free_dim: usize, out: &mut Vec<u32>) {
        let is_last = level == self.height as usize - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = level_masks[level];

        let word = self.word_at_veb(word_start, is_last);
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

/// `t`'s own `rank1(bit_pos+1)*WORD_BITS` child-offset formula, as a free
/// function -- needed by [`veb_block`]/[`children_of`] below, which run
/// *before* the final `RadixTreeVeb` struct exists (they only need `t`
/// itself, not any of the query-time convenience fields).
#[inline]
fn child_word_start_of(t: &BitVector, bit_pos: u64, next_is_last: bool) -> u64 {
    let virtual_start = t.rank1(bit_pos + 1) * WORD_BITS;
    if next_is_last {
        virtual_start - t.len()
    } else {
        virtual_start
    }
}

/// Every existing child's own bfs-order word index (i.e. `word_start /
/// WORD_BITS`, in the *original* BFS-ordered `t`/`l`), for the node at
/// `bfs_index` (in `t`, so `level < height-1`) -- read straight off `t`'s
/// own word content via the normal rank-based navigation, no separate
/// adjacency tracking needed.
fn children_of(t: &BitVector, bfs_index: u64, level: usize, height: u32) -> Vec<u64> {
    let word = t.word_at(bfs_index * WORD_BITS);
    let mut out = Vec::new();
    let mut bits = word;
    while bits != 0 {
        let qi = bits.trailing_zeros() as u64;
        bits &= bits - 1;
        let bit_pos = bfs_index * WORD_BITS + qi;
        let child_word_start = child_word_start_of(t, bit_pos, level + 1 == height as usize - 1);
        out.push(child_word_start / WORD_BITS);
    }
    out
}

/// Recursive Van Emde Boas visitation order for the subtree rooted at
/// `root` (bfs-index, at absolute `root_level`), covering exactly
/// `block_height` levels starting there. Writes every node visited
/// straight into `t_visits`/`l_visits` (leaves go to `l_visits` and are
/// never recursed into further); returns the nodes at *this call's own*
/// top-block boundary (its last covered level) so the caller can chase
/// their children into fresh, independently-recursed blocks -- this is
/// exactly "split the tree at half its height, lay the top half out, then
/// each bottom subtree contiguously after it, recursively."
///
/// `root` is visited (pushed into `t_visits`/`l_visits`) *exactly once* --
/// by whichever call ultimately bottoms out on it at `block_height == 1`,
/// not by every recursive level that merely passes it through as the same
/// fixed root of a shrinking top-half call. Boundary entries carry their
/// own level explicitly (rather than relying on the caller to re-derive
/// it), since that's needed to find their children.
fn veb_block(t: &BitVector, height: u32, root: u64, root_level: usize, block_height: usize, t_visits: &mut Vec<u64>, l_visits: &mut Vec<u64>) -> Vec<(u64, usize)> {
    if root_level == height as usize - 1 {
        l_visits.push(root);
        return Vec::new(); // leaf: nothing below it to chase
    }
    if block_height == 1 {
        t_visits.push(root); // visited here, and only here, for this root
        return vec![(root, root_level)]; // boundary node: caller recurses into its children
    }
    let top_h = block_height.div_ceil(2);
    let bottom_h = block_height - top_h;
    let top_boundary = veb_block(t, height, root, root_level, top_h, t_visits, l_visits);
    let mut next_boundary = Vec::new();
    for &(b, b_level) in &top_boundary {
        for child in children_of(t, b, b_level, height) {
            next_boundary.extend(veb_block(t, height, child, b_level + 1, bottom_h, t_visits, l_visits));
        }
    }
    next_boundary
}

/// Builds the VEB-reordered word copies and the BFS-index -> physical-index
/// permutations, from the already-finalized (BFS-ordered) `t` and the
/// plain (no-rank) BFS-ordered `l_words`.
fn compute_veb_reorder(t: &BitVector, l_words: &[u64], height: u32) -> (Vec<u64>, Vec<u32>, Vec<u64>, Vec<u32>) {
    let mut t_visits: Vec<u64> = Vec::new();
    let mut l_visits: Vec<u64> = Vec::new();
    if height > 0 {
        veb_block(t, height, 0, 0, height as usize, &mut t_visits, &mut l_visits);
    }
    let t_node_count = (t.len() / WORD_BITS) as usize;
    let mut t_physical_index = vec![0u32; t_node_count];
    let mut t_veb_words = vec![0u64; t_node_count];
    for (physical_pos, &bfs_idx) in t_visits.iter().enumerate() {
        t_physical_index[bfs_idx as usize] = physical_pos as u32;
        t_veb_words[physical_pos] = t.word_at(bfs_idx * WORD_BITS);
    }
    let l_node_count = l_words.len();
    let mut l_physical_index = vec![0u32; l_node_count];
    let mut l_veb_words = vec![0u64; l_node_count];
    for (physical_pos, &bfs_idx) in l_visits.iter().enumerate() {
        l_physical_index[bfs_idx as usize] = physical_pos as u32;
        l_veb_words[physical_pos] = l_words[bfs_idx as usize];
    }
    (t_veb_words, t_physical_index, l_veb_words, l_physical_index)
}

/// Bit-shift (within a packed multi-dimension index) for each dimension at
/// a level with the given per-dimension bit counts: `dim_order[0]` is
/// packed MSB-first, `dim_order[NUM_DIMS-1]` LSB-first (the default
/// `[0,1,2]` matches the original fixed S,P,O-MSB-first-by-ascending-index
/// convention).
fn level_shifts(dim_bits: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS]) -> [u8; NUM_DIMS] {
    let mut shifts = [0u8; NUM_DIMS];
    let mut acc = 0u8;
    for &d in dim_order.iter().rev() {
        shifts[d] = acc;
        acc += dim_bits[d];
    }
    shifts
}

/// A 64-bit mask, within one node's word, of the `qi` positions consistent
/// with the given per-dimension constraints -- `bound_val_for(d)` returns
/// `Some(value)` for a bound (fixed) dimension, `None` for a free
/// (unconstrained) one. Every node at this level shares the same mask
/// (it depends only on the schedule and the bound values, not on which
/// node's word it's ANDed against), so a query builds it once per level and
/// reuses it: `word & mask`, then iterate only the *set* result bits via
/// `trailing_zeros`/clear-lowest-set-bit -- no candidate-list allocation,
/// no per-candidate `.get()` calls, O(matches) instead of
/// O(candidates-checked).
///
/// No `dim_bits[d] == 0` guard needed: for an exhausted dimension the
/// `None` branch's `width` is `1`, a single no-op iteration re-copying
/// `mask` unchanged, and the `Some` branch's `sub` is naturally `0` (`lo[d]`
/// already equals `bound_val` exactly), so `mask <<= 0` is a no-op too --
/// both branches already reduce to doing nothing on their own.
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

/// Bits needed to fully resolve each dimension's domain (`ceil(log2(size))`).
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

/// At every level, "water-fills" the full [`BASE`] bit budget evenly
/// across whichever dimensions still need bits: a dimension about to
/// exhaust (needing less than its even share) only takes what it needs,
/// and the leftover is redistributed to the other still-active dimensions
/// in the same pass, so the level's total is always exactly `BASE` bits
/// (i.e. full 64-way branching) for as long as any dimension remains
/// active, only shrinking below that on the final level(s) where even an
/// even split across all remaining dimensions can't be fully used.
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

impl MemSize for RadixTreeVeb {
    fn mem_size_bytes(&self) -> usize {
        self.t.mem_size_bytes()
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
            + crate::common::metrics::vec_bytes(&self.t_veb_words)
            + crate::common::metrics::vec_bytes(&self.l_veb_words)
            + crate::common::metrics::vec_bytes(&self.t_physical_index)
            + crate::common::metrics::vec_bytes(&self.l_physical_index)
    }
}

impl RadixTreeVeb {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("t (64-bit-per-node words, BFS order, needed for rank)", self.t.mem_size_bytes()),
            ("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)),
            ("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)),
            ("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)),
            ("t_veb_words (VEB-reordered copy)", crate::common::metrics::vec_bytes(&self.t_veb_words)),
            ("l_veb_words (VEB-reordered copy)", crate::common::metrics::vec_bytes(&self.l_veb_words)),
            ("t_physical_index (permutation, O(nodes))", crate::common::metrics::vec_bytes(&self.t_physical_index)),
            ("l_physical_index (permutation, O(nodes))", crate::common::metrics::vec_bytes(&self.l_physical_index)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::{uniform, zipfian_predicates};
    use crate::common::reference::NaiveIndex;

    fn cross_check_tree(tree: &RadixTreeVeb, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        // The third bound-dimension combination -- S,O bound, P free --
        // exercises the structure's genuine 3D symmetry (unlike row/col,
        // which mirror what a 2D, S-O-only structure can also do).
        for s in 0..num_s {
            for o in 0..num_o {
                assert_eq!(tree.predicates_for(s, o), naive.predicates_for(s, o), "predicates_for(S={s},O={o}) {label}");
            }
        }
    }

    fn cross_check(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, split: [u8; NUM_DIMS]) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeVeb::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeVeb::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeVeb::build(&triples, RadixTreeVeb::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeVeb::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeVeb::build(&triples, RadixTreeVeb::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeVeb::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeVeb::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeVeb::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeVeb::default_split());
        }
    }

    #[test]
    fn cross_check_uneven_split_and_larger_domain() {
        // Predicate exhausts fast (1 bit/step), S/O share the rest --
        // exercises dimensions exhausting at very different depths.
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
        // PSO (predicate packed into the high bits) and SOP (predicate
        // packed into the low bits, after S and O) -- bit-packing order is
        // purely an internal layout choice; query results must be
        // identical to the default SPO order regardless.
        for seed in 1..5u64 {
            let triples = uniform(20, 6, 20, 400, seed * 811);
            let naive = NaiveIndex::build(&triples);
            for (order, label) in [([1usize, 0, 2], "PSO"), ([0, 2, 1], "SOP")] {
                let tree = RadixTreeVeb::build_with_order(&triples, RadixTreeVeb::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeVeb::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        // Predicate has far fewer distinct values than S/O, so it
        // exhausts first: the static [2,2,2] schedule then wastes bits
        // (branching only 4-way inside every 64-bit word) for the rest of
        // the tree, while the dynamic schedule should redistribute
        // predicate's freed bits to S/O and use noticeably fewer words.
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeVeb::build(&triples, RadixTreeVeb::default_split());
        let dynamic_tree = RadixTreeVeb::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeVeb::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }
}
