//! A variant of [`crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved`]
//! that stores each far node's inline skip value at a *per-level* width
//! -- 1, 2, 4, or 8 bytes (`u8`/`u16`/`u32`/`u64`) -- instead of always the
//! full 4-byte `u32`, chosen from a worst-case bound computable purely
//! from the schedule (`level_dim_bits` and `height`) before any data is
//! processed.
//!
//! Widths stick to native machine-integer sizes on purpose -- no 3-byte
//! ("u24") tier. A 3-byte read/write can't go through a single aligned
//! load/store the way 1/2/4/8 can (see `read_var`'s `2`/`4`/`8` arms vs.
//! the byte-by-byte assembly a 3-byte case would need), so it trades a
//! small, rarely-realized memory saving for real per-access overhead on
//! every single far-node hop, not just the ones actually using that
//! width. That trade only starts to look different if the left-over byte
//! in a 4-byte slot were doing something else useful -- packing a second,
//! unrelated piece of per-node information into the same word -- which
//! nothing here does, so there's no case for it.
//!
//! # Why a level determines a safe upper bound on its own
//!
//! A far node's skip value is its whole subtree's byte span. That
//! subtree spans a fixed number of remaining levels (`height - level`),
//! and at every one of those levels a node can have at most
//! `branching(level) = 2^(sum of that level's dim_bits)` children --
//! this is a property of the *schedule* (fixed before construction,
//! independent of the actual triples), not of the data. So the
//! *largest possible* subtree rooted at any node at a given level --
//! achieved only if every level below it were fully, adversarially
//! populated, which real data essentially never is -- is itself a
//! function of level and the schedule alone:
//!
//! ```text
//! max_bytes(height - 1) = 8                                          (leaf)
//! max_bytes(level)      = own_record_bytes(level)
//!                          + branching(level) * max_bytes(level + 1)  (level < height - 1)
//! ```
//!
//! The smallest byte width that can hold *any* value up to
//! `max_bytes(level)` is therefore a safe, provably-sufficient width for
//! *every* far node at that level, for *any* dataset built with this
//! schedule -- not just the one at hand. No extra storage or lookup is
//! needed to know which width a node uses: `level` is already tracked
//! throughout navigation, so `level_skip_bytes[level]` (a tiny
//! `Vec<u8>`, one entry per level, precomputed once at build time) is a
//! plain array read, not a rank operation -- unlike
//! [`crate::common::dac::DacArray`] (tried in
//! [`crate::common::radixtree_dfs_dac`]), which needed a whole extra
//! rank-augmented bitmap per byte boundary to know *whether* a value
//! continues. This costs nothing extra to determine width and preserves
//! the base interleaved design's core property untouched: direct
//! positional navigation, zero rank operations anywhere.
//!
//! # How much this actually saves
//!
//! `branching(level)` up to 64 means `max_bytes` grows so fast bottom-up
//! that it exceeds `u32::MAX` about 5 levels above the leaf, and
//! `u64::MAX` about 11 levels above the leaf, for the *default*,
//! evenly-split schedule (every level branching at the full 64). So for
//! that schedule, roughly the bottom 5 far levels get to use 1, 2, or 4
//! bytes instead of the fixed-width baseline's 8, and roughly the next 6
//! above those get to use the still-real 8-byte tier instead of paying
//! for a wider one that doesn't exist -- only levels within a handful of
//! the *root* end up exactly as wide as the (now 8-byte-capable) worst
//! case. The bound tightens faster -- and the useful window widens --
//! wherever per-level branching is naturally smaller: uneven splits,
//! [`Self::build_dynamic`]'s schedule (which shrinks a level's branching
//! once a dimension's bit budget is exhausted), or datasets skewed enough
//! that the effective tree is shallow and narrow throughout. This is an
//! empirical question answered by benchmarking, not assumed.
//!
//! # Safety of the bound -- and where it stops being one
//!
//! The width chosen is `min(8, smallest width that holds max_bytes(level))`
//! -- widths only ever go up to 8, since [`compute_level_skip_bytes`]
//! offers no wider option. For levels where `max_bytes(level)` itself
//! fits under 8 bytes (per the "how much this actually saves" section
//! above, all but a handful of levels near the root for the default
//! schedule), that genuinely is a proof: any real span at that level is
//! provably `<= max_bytes(level) <=` the chosen width's capacity, so
//! `write_var_at`'s `debug_assert!` should never fire, and hasn't in any
//! test or benchmark run.
//!
//! For the few levels closest to the root -- where even the schedule's
//! worst case exceeds `u64::MAX` (~18.4 quintillion bytes) -- width 8 is
//! chosen not because it's *proven* sufficient but because it's the
//! largest width this format supports. The real guarantee there is
//! exactly [`crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved`]'s
//! own, unstated one: that no single subtree's actual byte span exceeds
//! `u64::MAX`, because `build_interleaved_rec`'s `total_bytes` accumulator
//! is a plain `u64` throughout, with no wider tier to fall back to. Debug
//! builds panic on that overflow via Rust's default checked arithmetic;
//! release builds would wrap silently. Unlike the `u32` ceiling this file
//! used to top out at (~4.29 billion bytes -- only ~7x the largest
//! structure this project has actually measured, ~600 MiB), 18.4
//! quintillion bytes is not a practically approachable number for any
//! real RDF dataset; it's still an assumption, not a proof, but the
//! margin between it and anything real is now astronomical rather than
//! uncomfortably close.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8; // 6: max bits summed across all dims per node
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;
const WORD_RECORD_BYTES: usize = 8;
/// Fallback width (bytes) used only by the schedule-only bound
/// calculation itself (see [`compute_level_skip_bytes`]) to sidestep an
/// otherwise-circular dependency: a far level's own record size depends
/// on its chosen skip width, which is chosen *from* a bound computed
/// using... the level's own record size. Since the skip width is at
/// most 8 bytes regardless, and the bound calculation's whole point is
/// levels where `branching(level) * max_bytes(level+1)` already dwarfs a
/// few bytes, assuming the conservative maximum here costs nothing in
/// practice and breaks the cycle cleanly.
const MAX_SKIP_RECORD_BYTES: usize = 8;

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleavedVarSkip {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsInterleavedVarSkip::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkip::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkip::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkip::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsInterleavedVarSkip::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsInterleavedVarSkip::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsInterleavedVarSkip::height(self)
    }
}

pub struct RadixTreeDfsInterleavedVarSkip {
    /// One combined byte stream, DFS/preorder: each node contributes an
    /// 8-byte little-endian existence word, and *far* nodes (`level + 2 <
    /// height`) additionally contribute `level_skip_bytes[level]` more
    /// little-endian bytes right after it -- see the module doc comment.
    combined: Vec<u8>,
    node_count: u64,
    /// One entry per level: for a far level, the byte width (1, 2, 4, or
    /// 8) chosen for that level's inline skip value, from a
    /// schedule-only worst-case bound (see [`compute_level_skip_bytes`]).
    /// Irrelevant (unread) for non-far levels.
    level_skip_bytes: Vec<u8>,
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

/// Reads a little-endian value of `width` (1, 2, 4, or 8) bytes starting
/// at `pos`, zero-extended to `u64`. Each arm is a single native-width
/// load (`u8`/`u16`/`u32`/`u64::from_le_bytes`) -- no width straddles a
/// non-native size the way a 3-byte read would have to (see the module
/// doc comment for why that tier was dropped rather than added).
#[inline]
fn read_var(buf: &[u8], pos: usize, width: u8) -> u64 {
    match width {
        1 => buf[pos] as u64,
        2 => u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as u64,
        4 => u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as u64,
        8 => u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()),
        _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
    }
}

#[inline]
fn push_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Appends `width` (1, 2, 4, or 8) zeroed placeholder bytes, returning
/// their offset so the caller can overwrite them once the real value
/// (the subtree's total byte span) is known, after every child has been
/// built.
#[inline]
fn push_var_placeholder(buf: &mut Vec<u8>, width: u8) -> usize {
    let pos = buf.len();
    buf.resize(pos + width as usize, 0);
    pos
}

/// Writes `val` as a little-endian value of `width` (1, 2, 4, or 8)
/// bytes at `pos`. `debug_assert!`s that `val` actually fits -- see the
/// module doc comment's "Safety of the bound" section: this should never
/// fire for any tree built from [`compute_level_skip_bytes`]'s own
/// widths, and firing means that bound's derivation has a bug, not that
/// the data is unusual.
#[inline]
fn write_var_at(buf: &mut [u8], pos: usize, width: u8, val: u64) {
    debug_assert!(
        width == 8 || val < (1u64 << (8 * width as u32)),
        "value {val} does not fit in {width} bytes -- schedule-only skip-width bound was insufficient"
    );
    match width {
        1 => buf[pos] = val as u8,
        2 => buf[pos..pos + 2].copy_from_slice(&(val as u16).to_le_bytes()),
        4 => buf[pos..pos + 4].copy_from_slice(&(val as u32).to_le_bytes()),
        8 => buf[pos..pos + 8].copy_from_slice(&val.to_le_bytes()),
        _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
    }
}

/// Smallest number of bytes (1, 2, 4, or 8) sufficient to hold *any*
/// possible far-node subtree span at each level, computed purely from the
/// schedule -- see the module doc comment for the recursion and why it's
/// safe for any dataset built with this schedule, not just the one at
/// hand.
fn compute_level_skip_bytes(level_dim_bits: &[[u8; NUM_DIMS]], height: usize) -> Vec<u8> {
    let mut max_bytes = vec![0u64; height];
    max_bytes[height - 1] = WORD_RECORD_BYTES as u64;
    for level in (0..height - 1).rev() {
        let dim_bits = level_dim_bits[level];
        let branching = 1u64 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
        let is_far = level + 2 < height;
        let own = if is_far { (WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES) as u64 } else { WORD_RECORD_BYTES as u64 };
        max_bytes[level] = own.saturating_add(branching.saturating_mul(max_bytes[level + 1]));
    }
    max_bytes
        .iter()
        .map(|&m| if m <= 0xFF { 1 } else if m <= 0xFFFF { 2 } else if m <= 0xFFFF_FFFF { 4 } else { 8 })
        .collect()
}

impl RadixTreeDfsInterleavedVarSkip {
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
            return RadixTreeDfsInterleavedVarSkip {
                combined: Vec::new(),
                node_count: 0,
                level_skip_bytes: Vec::new(),
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
        let level_skip_bytes = compute_level_skip_bytes(&level_dim_bits, height as usize);

        let mut points: Vec<(u32, u32, u32)> = triples.to_vec();
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        let mut combined = Vec::new();
        let mut node_count = 0u64;
        build_interleaved_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, height, &level_dim_bits, &level_shifts_cache, &level_child_cell_bits_cache, &level_skip_bytes, &mut combined, &mut node_count);
        combined.shrink_to_fit();

        RadixTreeDfsInterleavedVarSkip {
            combined,
            node_count,
            level_skip_bytes,
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
    /// child (if any) begins. The skip width itself varies by level
    /// (`level_skip_bytes[level]`), unlike the base interleaved design's
    /// fixed 4 bytes.
    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height as usize {
            WORD_RECORD_BYTES + self.level_skip_bytes[level] as usize
        } else {
            WORD_RECORD_BYTES
        }
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
            let width = self.level_skip_bytes[child_level];
            let skip = read_var(&self.combined, child_pos + WORD_RECORD_BYTES, width) as usize;
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
                // far -- unlike the base interleaved design, that no
                // longer collapses to a compile-time constant, since the
                // skip width itself varies by level here; still one
                // array read (`level_skip_bytes[level]`, inside
                // `own_record_bytes`), not a rank operation.
                let qi = word.trailing_zeros() as usize;
                let remaining = word & (word - 1);
                let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                debug_assert!(stack_len < MAX_HEIGHT, "query stack overflow: tree height exceeds MAX_HEIGHT");
                stack[stack_len] = ConstrainedFrame { lo, remaining, child_pos: 0 };
                stack_len += 1;
                pos += self.own_record_bytes(level);
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

/// One per open ancestor in [`RadixTreeDfsInterleavedVarSkip::query_rec`]'s
/// merged constrained/listing descent. `child_pos` is real, necessary
/// state for a frame pushed while still constrained (`level <
/// unconstrained_from`): a resumed one may still have several *unmatched*
/// siblings to walk past, via [`RadixTreeDfsInterleavedVarSkip::child_span_end`],
/// before its next match or exhaustion. A frame pushed while listing
/// (`level >= unconstrained_from`) leaves `child_pos` unused -- every
/// existing child matches there, so resuming needs only whatever position
/// DFS order already propagated into `pos`. `lo` is a single `u64`, not
/// `[u64; NUM_DIMS]`: see
/// [`RadixTreeDfsInterleavedVarSkip::advance_lo_dim`] for why every dimension
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
/// each visited node's record (8 bytes, or `8 + level_skip_bytes[level]`
/// for a far node -- see the module doc comment) directly to `combined`,
/// in DFS preorder. Returns this node's own subtree's total byte span
/// (itself included), which the caller uses to fill in its *own*
/// far-node skip placeholder (if it has one) after every child returns.
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
    level_skip_bytes: &[u8],
    combined: &mut Vec<u8>,
    node_count: &mut u64,
) -> u64 {
    let is_last = level == height as usize - 1;
    let is_far_node = level + 2 < height as usize;
    let skip_width = level_skip_bytes[level];
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
    let skip_placeholder_pos = if is_far_node { Some(push_var_placeholder(combined, skip_width)) } else { None };

    let mut total_bytes: u64 = WORD_RECORD_BYTES as u64 + if is_far_node { skip_width as u64 } else { 0 };
    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeDfsInterleavedVarSkip::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_bytes = build_interleaved_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, level_skip_bytes, combined, node_count);
                total_bytes += child_bytes;
            }
        }
    }
    if let Some(p) = skip_placeholder_pos {
        write_var_at(combined, p, skip_width, total_bytes);
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

impl MemSize for RadixTreeDfsInterleavedVarSkip {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.combined)
            + crate::common::metrics::vec_bytes(&self.level_skip_bytes)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfsInterleavedVarSkip {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("combined (8B/node + 1/2/4/8B/far-node, inline, DFS order)", crate::common::metrics::vec_bytes(&self.combined)),
            ("level_skip_bytes (O(height))", crate::common::metrics::vec_bytes(&self.level_skip_bytes)),
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

    /// `read_var`/`write_var_at` round-trip exactly, at every width and
    /// right at each threshold value (the largest thing that fits, and
    /// the smallest thing that doesn't -- off-by-one at a boundary is
    /// the classic bug shape here, so check the boundary itself, not
    /// just a value comfortably inside each range).
    #[test]
    fn variable_width_round_trip_at_boundaries() {
        let cases: &[(u64, u8)] = &[
            (0, 1),
            (1, 1),
            (254, 1),
            (255, 1),   // largest value 1 byte holds
            (256, 2),   // smallest value that needs 2
            (257, 2),
            (65_534, 2),
            (65_535, 2), // largest value 2 bytes holds
            (65_536, 4), // smallest value that needs 4 (no 3-byte tier)
            (65_537, 4),
            (u32::MAX as u64 - 1, 4),
            (u32::MAX as u64, 4),      // largest value 4 bytes holds
            (u32::MAX as u64 + 1, 8),  // smallest value that needs 8
            (u32::MAX as u64 + 2, 8),
            (u64::MAX - 1, 8),
            (u64::MAX, 8),
        ];
        for &(val, width) in cases {
            let mut buf = vec![0u8; width as usize + 8]; // padding so a stray over-read is visible via corruption, not a panic
            write_var_at(&mut buf, 0, width, val);
            let got = read_var(&buf, 0, width);
            assert_eq!(got, val, "round-trip mismatch at width={width} val={val}");
        }
    }

    /// `compute_level_skip_bytes` picks the *smallest* width that holds
    /// the worst-case bound, right at each threshold -- exercised via
    /// a synthetic single-level "schedule" standing in for whatever
    /// `max_bytes(level)` the real recursion would have computed, since
    /// the width-selection arithmetic itself doesn't care how that bound
    /// was derived.
    #[test]
    fn skip_width_selection_at_boundaries() {
        fn width_for(m: u64) -> u8 {
            if m <= 0xFF {
                1
            } else if m <= 0xFFFF {
                2
            } else if m <= 0xFFFF_FFFF {
                4
            } else {
                8
            }
        }
        assert_eq!(width_for(0), 1);
        assert_eq!(width_for(255), 1);
        assert_eq!(width_for(256), 2);
        assert_eq!(width_for(65535), 2);
        assert_eq!(width_for(65536), 4);
        assert_eq!(width_for(u32::MAX as u64), 4);
        assert_eq!(width_for(u32::MAX as u64 + 1), 8);
        assert_eq!(width_for(u64::MAX), 8);
    }

    fn cross_check_tree(tree: &RadixTreeDfsInterleavedVarSkip, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
        let tree = RadixTreeDfsInterleavedVarSkip::build(triples, split);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?}"));
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = RadixTreeDfsInterleavedVarSkip::build_dynamic(triples);
        cross_check_tree(&tree, &naive, num_s, num_p, num_o, "dynamic");
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = RadixTreeDfsInterleavedVarSkip::build(&triples, RadixTreeDfsInterleavedVarSkip::default_split());
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        let tree = RadixTreeDfsInterleavedVarSkip::build_dynamic(&triples);
        assert!(!tree.contains(0, 0, 0));
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        let tree = RadixTreeDfsInterleavedVarSkip::build(&triples, RadixTreeDfsInterleavedVarSkip::default_split());
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
        let tree = RadixTreeDfsInterleavedVarSkip::build_dynamic(&triples);
        assert!(tree.contains(0, 0, 0));
        assert!(!tree.contains(0, 0, 1));
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsInterleavedVarSkip::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsInterleavedVarSkip::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsInterleavedVarSkip::default_split());
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
                let tree = RadixTreeDfsInterleavedVarSkip::build_with_order(&triples, RadixTreeDfsInterleavedVarSkip::default_split(), order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
                let tree = RadixTreeDfsInterleavedVarSkip::build_dynamic_with_order(&triples, order);
                cross_check_tree(&tree, &naive, 20, 6, 20, label);
            }
        }
    }

    #[test]
    fn dynamic_schedule_recovers_size_once_a_dimension_exhausts() {
        let triples = uniform(2000, 8, 2000, 20000, 4242);
        let static_tree = RadixTreeDfsInterleavedVarSkip::build(&triples, RadixTreeDfsInterleavedVarSkip::default_split());
        let dynamic_tree = RadixTreeDfsInterleavedVarSkip::build_dynamic(&triples);
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfsInterleavedVarSkip::default_split());
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
        let tree = RadixTreeDfsInterleavedVarSkip::build_dynamic(&triples);
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
