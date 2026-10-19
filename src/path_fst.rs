//! Experimental prototype, not production code -- see `DESIGN.md`'s
//! "path FST prototype" entry. Builds a minimized (deduplicated)
//! FST-style representation of the same DFS-interleaved, bit-
//! interleaved radix structure `tree` uses -- same per-dimension bit
//! schedule (`total_bits_needed`/`dynamic_schedule`/`compute_qi`/
//! `advance_lo`, reused directly from `tree`, not reimplemented), so a
//! comparison against the production tree is apples-to-apples. Static
//! build only: no `insert`/`remove` yet (deliberately deferred -- see
//! `s-dict`'s own "mutable unrolled-block" history for why that's a
//! separate, harder problem, not a small addition).
//!
//! **Blocked, not a single flat buffer**: records live in a real
//! [`HeapStorage`] ([`BlockStorage`]), the exact same block-chopped
//! backend and `append_record`/`block_capacity` contract the production
//! tree uses -- so any remaining advantage over `Heap` can't be
//! attributed to "no block-indirection hop," only to the dedup itself.
//! Each child reference is a direct [`Pos`] (block, offset), packed into
//! one `u64` (`block << 32 | offset`) -- not a linear byte delta (the
//! first, flat-buffer version of this prototype used a backward delta,
//! which doesn't generalize once "backward" isn't a single scalar
//! direction across block boundaries). This costs 8 bytes per child
//! reference instead of 4, and needs no `BlockIndex`/absolute-position
//! bookkeeping at all: a child's `Pos` is already fully resolved by the
//! time its parent is written (postorder), so there's nothing to
//! compute a delta *from*.
//!
//! **Node identity for dedup is an exact Myhill-Nerode merge**: two
//! subtrees are deduplicated only if their existence word *and* their
//! already-resolved (post-dedup) child positions, in `qi`-ascending
//! order, are identical -- the register key is the literal `(word,
//! Vec<(block, offset)>)` tuple, not a lossy hash (correctness matters
//! for a real structure, unlike `tree::subtree_dedup_report`'s
//! measurement-only 64-bit hash, where a false merge was an acceptable
//! estimate error but would be a real bug here). One register *per tree
//! level*, not a single global one -- nothing in principle stops two
//! different levels' schedules from producing an identical key by
//! coincidence for some future domain/schedule shape, and a state built
//! for level `L+1`'s decode rules must never be reused as if it were
//! level `L'+1`'s; partitioning by level makes that impossible by
//! construction rather than relying on it being empirically unlikely.
//!
//! **Record format**: `[existence_word: 8 bytes][child_pos: u64]*`, one
//! packed `(block, offset)` per real child, `qi`-ascending -- written in
//! one `append_record` call per node (every child's `Pos` is already
//! known, so no placeholder-then-patch is needed the way the production
//! tree's forward-skip spans require). Leaf-level (`level == height -
//! 1`) records are the bare 8-byte word, same as the production tree.

use crate::block_index::BlockIndex;
use crate::storage::{BlockStorage, HeapStorage, Pos};
use crate::tree::{advance_lo, compute_qi, dynamic_schedule, level_shifts, restriction_mask, total_bits_needed, DimRestriction, Triple, K_SQUARE, MAX_HEIGHT, NUM_DIMS};
use std::collections::HashMap;

const WORD_BYTES: usize = 8;
const CHILD_PTR_BYTES: usize = 8;
const DELTA_BYTES: u8 = 4;

fn pack_pos(pos: Pos) -> u64 {
    debug_assert!(pos.block <= u32::MAX as usize && pos.offset <= u32::MAX as usize);
    ((pos.block as u64) << 32) | (pos.offset as u64)
}

fn unpack_pos(packed: u64) -> Pos {
    Pos {
        block: (packed >> 32) as usize,
        offset: (packed & 0xFFFF_FFFF) as usize,
    }
}

/// See the module doc comment. Experimental, read-only (`contains` only
/// for now), not wired into `Heap`/`File`/`Compressed` or any public
/// crate API beyond this type itself.
pub struct PathFst {
    storage: HeapStorage,
    root_pos: Pos,
    height: usize,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    block_capacity: usize,
    /// Records actually written (post-dedup) -- the structure's true
    /// state count.
    node_count: u64,
    /// Total recursive build calls (pre-dedup) -- what the equivalent
    /// plain (non-deduplicated) tree's node count would be.
    visited_count: u64,
}

impl PathFst {
    pub fn build(mut triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        let dim_order = [0usize, 1, 2];
        let total_bits = total_bits_needed(dimension_sizes);
        let level_dim_bits = dynamic_schedule(total_bits);
        let height = level_dim_bits.len();
        if height == 0 {
            return PathFst {
                storage: HeapStorage::empty(),
                root_pos: Pos::START,
                height: 0,
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                block_capacity,
                node_count: 0,
                visited_count: 0,
            };
        }
        let level_shifts_cache: Vec<[u8; NUM_DIMS]> = level_dim_bits.iter().map(|&db| level_shifts(db, dim_order)).collect();
        let level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]> = {
            let mut remaining = total_bits;
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

        triples.sort_unstable();
        triples.dedup();

        let mut storage = HeapStorage::empty();
        let mut register: Vec<HashMap<(u64, Vec<u64>), Pos>> = (0..height).map(|_| HashMap::new()).collect();
        let mut node_count = 0u64;
        let mut visited_count = 0u64;
        let mut scratch = vec![(0u32, 0u32, 0u32); triples.len()];
        let root_pos = build_rec(
            &mut triples,
            &mut scratch,
            [0; NUM_DIMS],
            0,
            height,
            &level_dim_bits,
            &level_shifts_cache,
            &level_child_cell_bits_cache,
            &mut storage,
            block_capacity,
            &mut register,
            &mut node_count,
            &mut visited_count,
        );
        storage.finish_append(block_capacity);

        PathFst {
            storage,
            root_pos,
            height,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            block_capacity,
            node_count,
            visited_count,
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        if self.height == 0 {
            return false;
        }
        if s as u64 >= self.dimension_sizes[0] || p as u64 >= self.dimension_sizes[1] || o as u64 >= self.dimension_sizes[2] {
            return false;
        }
        let coords = [s as u64, p as u64, o as u64];
        let mut pos = self.root_pos;
        let mut lo = [0u64; NUM_DIMS];
        for level in 0..self.height {
            let word = self.storage.read_u64(pos.block, pos.offset);
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let qi = compute_qi(coords, lo, shifts, child_cell_bits);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if level == self.height - 1 {
                return true;
            }
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let child_packed = self.storage.read_u64(pos.block, pos.offset + WORD_BYTES + real_index * CHILD_PTR_BYTES);
            pos = unpack_pos(child_packed);
            lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 loop always returns from inside")
    }

    /// The structure's real, total allocated footprint -- directly
    /// comparable to `Heap::mem_size_bytes`.
    pub fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes()
            + crate::vec_bytes(&self.level_dim_bits)
            + crate::vec_bytes(&self.level_shifts_cache)
            + crate::vec_bytes(&self.level_child_cell_bits_cache)
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn visited_count(&self) -> u64 {
        self.visited_count
    }

    pub fn block_count(&self) -> usize {
        self.storage.block_count()
    }

    pub fn block_capacity(&self) -> usize {
        self.block_capacity
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rec(
    points: &mut [Triple],
    scratch: &mut [Triple],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: usize,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    storage: &mut HeapStorage,
    block_capacity: usize,
    register: &mut [HashMap<(u64, Vec<u64>), Pos>],
    node_count: &mut u64,
    visited_count: &mut u64,
) -> Pos {
    *visited_count += 1;
    let dim_bits = level_dim_bits[level];
    let shifts = level_shifts_cache[level];
    let child_cell_bits = level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
    debug_assert!(branching <= K_SQUARE as usize);

    let qi_of = |pt: &Triple| compute_qi([pt.0 as u64, pt.1 as u64, pt.2 as u64], lo, shifts, child_cell_bits);
    let mut counts = [0usize; K_SQUARE as usize];
    for pt in points.iter() {
        counts[qi_of(pt)] += 1;
    }
    let mut offsets = [0usize; K_SQUARE as usize + 1];
    for i in 0..branching {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let mut cursor = offsets;
    for &pt in points.iter() {
        let qi = qi_of(&pt);
        scratch[cursor[qi]] = pt;
        cursor[qi] += 1;
    }
    points.copy_from_slice(&scratch[..points.len()]);

    let mut existence_word = 0u64;
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            existence_word |= 1u64 << qi;
        }
    }

    if level == height - 1 {
        let key = (existence_word, Vec::new());
        if let Some(&pos) = register[level].get(&key) {
            return pos;
        }
        let pos = storage.append_record(&existence_word.to_le_bytes(), block_capacity);
        register[level].insert(key, pos);
        *node_count += 1;
        return pos;
    }

    let mut child_positions = Vec::with_capacity(existence_word.count_ones() as usize);
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            let (child_points, child_scratch) = (&mut points[offsets[qi]..offsets[qi + 1]], &mut scratch[offsets[qi]..offsets[qi + 1]]);
            let child_pos = build_rec(
                child_points,
                child_scratch,
                child_lo,
                level + 1,
                height,
                level_dim_bits,
                level_shifts_cache,
                level_child_cell_bits_cache,
                storage,
                block_capacity,
                register,
                node_count,
                visited_count,
            );
            child_positions.push(pack_pos(child_pos));
        }
    }

    let key = (existence_word, child_positions.clone());
    if let Some(&pos) = register[level].get(&key) {
        return pos;
    }
    let mut rec = Vec::with_capacity(WORD_BYTES + CHILD_PTR_BYTES * child_positions.len());
    rec.extend_from_slice(&existence_word.to_le_bytes());
    for &packed in &child_positions {
        rec.extend_from_slice(&packed.to_le_bytes());
    }
    let pos = storage.append_record(&rec, block_capacity);
    register[level].insert(key, pos);
    *node_count += 1;
    pos
}

// ==================== delta + BlockIndex variant ====================

/// Second blocked variant, for direct comparison against [`PathFst`]:
/// keeps the *first* (flat-buffer) prototype's compact 4-byte backward
/// delta instead of an 8-byte `Pos`, and uses a real [`BlockIndex`] --
/// the same Fenwick-tree directory `tree::advance`'s own slow path uses
/// -- to turn a delta-resolved absolute byte offset back into a `Pos` on
/// every non-leaf hop. Trades a smaller record (4 bytes/child, same as
/// the unblocked prototype) for one extra `BlockIndex::locate` call
/// (O(log blocks)) per level during `contains` -- exactly the "keep the
/// delta, pay for block navigation with arithmetic instead of a wider
/// pointer" tradeoff this is here to measure, not assume.
///
/// Building needs no `BlockIndex` at all: `HeapStorage::append_record`
/// never pads a block short of its content (a block's length is exactly
/// the sum of the records placed in it), so a running "total bytes
/// appended so far" counter, maintained by the builder itself, already
/// equals the true absolute byte offset -- no Fenwick bookkeeping is
/// needed until the structure is read back. `BlockIndex::build` runs
/// once, after the whole tree is written, in `O(block_count)`.
pub struct PathFstDelta {
    storage: HeapStorage,
    block_index: BlockIndex,
    root_abs_pos: u64,
    height: usize,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    block_capacity: usize,
    node_count: u64,
    visited_count: u64,
}

impl PathFstDelta {
    pub fn build(mut triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        let dim_order = [0usize, 1, 2];
        let total_bits = total_bits_needed(dimension_sizes);
        let level_dim_bits = dynamic_schedule(total_bits);
        let height = level_dim_bits.len();
        if height == 0 {
            return PathFstDelta {
                storage: HeapStorage::empty(),
                block_index: BlockIndex::default(),
                root_abs_pos: 0,
                height: 0,
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                block_capacity,
                node_count: 0,
                visited_count: 0,
            };
        }
        let level_shifts_cache: Vec<[u8; NUM_DIMS]> = level_dim_bits.iter().map(|&db| level_shifts(db, dim_order)).collect();
        let level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]> = {
            let mut remaining = total_bits;
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

        triples.sort_unstable();
        triples.dedup();

        let mut storage = HeapStorage::empty();
        let mut register: Vec<HashMap<(u64, Vec<u64>), u64>> = (0..height).map(|_| HashMap::new()).collect();
        let mut node_count = 0u64;
        let mut visited_count = 0u64;
        let mut total_bytes: u64 = 0;
        let mut scratch = vec![(0u32, 0u32, 0u32); triples.len()];
        let root_abs_pos = build_rec_delta(
            &mut triples,
            &mut scratch,
            [0; NUM_DIMS],
            0,
            height,
            &level_dim_bits,
            &level_shifts_cache,
            &level_child_cell_bits_cache,
            &mut storage,
            block_capacity,
            &mut register,
            &mut node_count,
            &mut visited_count,
            &mut total_bytes,
        );
        storage.finish_append(block_capacity);
        let block_index = BlockIndex::build(&storage);

        PathFstDelta {
            storage,
            block_index,
            root_abs_pos,
            height,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            block_capacity,
            node_count,
            visited_count,
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        if self.height == 0 {
            return false;
        }
        if s as u64 >= self.dimension_sizes[0] || p as u64 >= self.dimension_sizes[1] || o as u64 >= self.dimension_sizes[2] {
            return false;
        }
        let coords = [s as u64, p as u64, o as u64];
        // The root is written *last* under postorder (every other node is
        // its descendant, written first), so it is not at absolute
        // position 0 -- it's wherever the top-level build call actually
        // returned.
        let mut abs_pos: u64 = self.root_abs_pos;
        let mut pos = self.block_index.locate(abs_pos);
        let mut lo = [0u64; NUM_DIMS];
        for level in 0..self.height {
            let word = self.storage.read_u64(pos.block, pos.offset);
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let qi = compute_qi(coords, lo, shifts, child_cell_bits);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if level == self.height - 1 {
                return true;
            }
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(pos.block, pos.offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let target_abs_pos = abs_pos - delta;
            pos = self.block_index.locate(target_abs_pos);
            abs_pos = target_abs_pos;
            lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 loop always returns from inside")
    }

    pub fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes()
            + self.block_index.mem_size_bytes()
            + crate::vec_bytes(&self.level_dim_bits)
            + crate::vec_bytes(&self.level_shifts_cache)
            + crate::vec_bytes(&self.level_child_cell_bits_cache)
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn visited_count(&self) -> u64 {
        self.visited_count
    }

    pub fn block_count(&self) -> usize {
        self.storage.block_count()
    }

    pub fn block_capacity(&self) -> usize {
        self.block_capacity
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rec_delta(
    points: &mut [Triple],
    scratch: &mut [Triple],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: usize,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    storage: &mut HeapStorage,
    block_capacity: usize,
    register: &mut [HashMap<(u64, Vec<u64>), u64>],
    node_count: &mut u64,
    visited_count: &mut u64,
    total_bytes: &mut u64,
) -> u64 {
    *visited_count += 1;
    let dim_bits = level_dim_bits[level];
    let shifts = level_shifts_cache[level];
    let child_cell_bits = level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
    debug_assert!(branching <= K_SQUARE as usize);

    let qi_of = |pt: &Triple| compute_qi([pt.0 as u64, pt.1 as u64, pt.2 as u64], lo, shifts, child_cell_bits);
    let mut counts = [0usize; K_SQUARE as usize];
    for pt in points.iter() {
        counts[qi_of(pt)] += 1;
    }
    let mut offsets = [0usize; K_SQUARE as usize + 1];
    for i in 0..branching {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let mut cursor = offsets;
    for &pt in points.iter() {
        let qi = qi_of(&pt);
        scratch[cursor[qi]] = pt;
        cursor[qi] += 1;
    }
    points.copy_from_slice(&scratch[..points.len()]);

    let mut existence_word = 0u64;
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            existence_word |= 1u64 << qi;
        }
    }

    if level == height - 1 {
        let key = (existence_word, Vec::new());
        if let Some(&abs_addr) = register[level].get(&key) {
            return abs_addr;
        }
        let abs_addr = *total_bytes;
        let rec = existence_word.to_le_bytes();
        storage.append_record(&rec, block_capacity);
        *total_bytes += rec.len() as u64;
        register[level].insert(key, abs_addr);
        *node_count += 1;
        return abs_addr;
    }

    let mut child_addrs = Vec::with_capacity(existence_word.count_ones() as usize);
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            let (child_points, child_scratch) = (&mut points[offsets[qi]..offsets[qi + 1]], &mut scratch[offsets[qi]..offsets[qi + 1]]);
            let child_addr = build_rec_delta(
                child_points,
                child_scratch,
                child_lo,
                level + 1,
                height,
                level_dim_bits,
                level_shifts_cache,
                level_child_cell_bits_cache,
                storage,
                block_capacity,
                register,
                node_count,
                visited_count,
                total_bytes,
            );
            child_addrs.push(child_addr);
        }
    }

    let key = (existence_word, child_addrs.clone());
    if let Some(&abs_addr) = register[level].get(&key) {
        return abs_addr;
    }
    let abs_addr = *total_bytes;
    let mut rec = Vec::with_capacity(WORD_BYTES + DELTA_BYTES as usize * child_addrs.len());
    rec.extend_from_slice(&existence_word.to_le_bytes());
    for &child_addr in &child_addrs {
        let delta = (abs_addr - child_addr) as u32;
        rec.extend_from_slice(&delta.to_le_bytes());
    }
    storage.append_record(&rec, block_capacity);
    *total_bytes += rec.len() as u64;
    register[level].insert(key, abs_addr);
    *node_count += 1;
    abs_addr
}

// ==================== delta + power-of-two block arithmetic variant ====================

/// Third blocked variant -- the user's own follow-up to [`PathFstDelta`]:
/// if `block_capacity` is a power of two, "which block does this
/// absolute position fall in" doesn't need a `BlockIndex` (a Fenwick
/// walk, `O(log blocks)`) at all. Define a *virtual* address as `block *
/// block_capacity + offset` instead of the true packed-byte count
/// [`PathFstDelta`] uses: since `HeapStorage::append_record` never lets
/// a block exceed `block_capacity`, this numbering has no collisions,
/// and inverting it is one shift and one mask -- `block = addr >>
/// capacity_bits`, `offset = addr & capacity_mask` -- genuinely `O(1)`,
/// not just asymptotically cheaper. Same 4-byte delta as [`PathFstDelta`]
/// (the "wasted" tail bytes a block leaves unused when the next record
/// doesn't fit are priced into the delta's numeric value, not into
/// stored bytes -- the `Vec<Vec<u8>>` behind `HeapStorage` is completely
/// unaffected and still holds only real record bytes), but the read path
/// has no Fenwick walk left in it to pay for.
///
/// Building can no longer predict its own resulting address before
/// calling `append_record` (unlike [`PathFstDelta`]'s running byte
/// counter, which works precisely because `HeapStorage` never pads --
/// this scheme's addressing is defined in terms of where
/// `append_record` actually lands, block boundaries included): appends
/// a placeholder record with zeroed deltas, reads back the real `Pos`
/// `append_record` returns, computes this node's own virtual address
/// from it, then backfills every child's delta via `patch_appended` --
/// the identical mechanism `tree::build_rec` already uses for its
/// forward-skip fields, for a different reason (not knowing this node's
/// *own* future address, rather than not knowing a child's future size).
pub struct PathFstDeltaPow2 {
    storage: HeapStorage,
    root_virtual_addr: u64,
    height: usize,
    level_dim_bits: Vec<[u8; NUM_DIMS]>,
    level_shifts_cache: Vec<[u8; NUM_DIMS]>,
    level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]>,
    dimension_sizes: [u64; NUM_DIMS],
    block_capacity: usize,
    capacity_bits: u32,
    capacity_mask: u64,
    node_count: u64,
    visited_count: u64,
}

impl PathFstDeltaPow2 {
    /// `block_capacity` must be a power of two (asserted) -- this
    /// variant exists specifically to exploit that; a non-power-of-two
    /// capacity would need a real division, not a shift/mask, and
    /// should use [`PathFstDelta`] instead.
    pub fn build(mut triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        assert!(block_capacity.is_power_of_two(), "PathFstDeltaPow2 requires a power-of-two block_capacity, got {block_capacity}");
        let capacity_bits = block_capacity.trailing_zeros();
        let capacity_mask = (block_capacity - 1) as u64;

        let dim_order = [0usize, 1, 2];
        let total_bits = total_bits_needed(dimension_sizes);
        let level_dim_bits = dynamic_schedule(total_bits);
        let height = level_dim_bits.len();
        if height == 0 {
            return PathFstDeltaPow2 {
                storage: HeapStorage::empty(),
                root_virtual_addr: 0,
                height: 0,
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                block_capacity,
                capacity_bits,
                capacity_mask,
                node_count: 0,
                visited_count: 0,
            };
        }
        let level_shifts_cache: Vec<[u8; NUM_DIMS]> = level_dim_bits.iter().map(|&db| level_shifts(db, dim_order)).collect();
        let level_child_cell_bits_cache: Vec<[u8; NUM_DIMS]> = {
            let mut remaining = total_bits;
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

        triples.sort_unstable();
        triples.dedup();

        let mut storage = HeapStorage::empty();
        let mut register: Vec<HashMap<(u64, Vec<u64>), u64>> = (0..height).map(|_| HashMap::new()).collect();
        let mut node_count = 0u64;
        let mut visited_count = 0u64;
        let mut scratch = vec![(0u32, 0u32, 0u32); triples.len()];
        let root_virtual_addr = build_rec_pow2(
            &mut triples,
            &mut scratch,
            [0; NUM_DIMS],
            0,
            height,
            &level_dim_bits,
            &level_shifts_cache,
            &level_child_cell_bits_cache,
            &mut storage,
            block_capacity,
            capacity_bits,
            &mut register,
            &mut node_count,
            &mut visited_count,
        );
        storage.finish_append(block_capacity);

        PathFstDeltaPow2 {
            storage,
            root_virtual_addr,
            height,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            block_capacity,
            capacity_bits,
            capacity_mask,
            node_count,
            visited_count,
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        if self.height == 0 {
            return false;
        }
        if s as u64 >= self.dimension_sizes[0] || p as u64 >= self.dimension_sizes[1] || o as u64 >= self.dimension_sizes[2] {
            return false;
        }
        let coords = [s as u64, p as u64, o as u64];
        let mut addr = self.root_virtual_addr;
        let mut lo = [0u64; NUM_DIMS];
        for level in 0..self.height {
            let block = (addr >> self.capacity_bits) as usize;
            let offset = (addr & self.capacity_mask) as usize;
            let word = self.storage.read_u64(block, offset);
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let qi = compute_qi(coords, lo, shifts, child_cell_bits);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if level == self.height - 1 {
                return true;
            }
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            addr -= delta;
            lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 loop always returns from inside")
    }

    pub fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes()
            + crate::vec_bytes(&self.level_dim_bits)
            + crate::vec_bytes(&self.level_shifts_cache)
            + crate::vec_bytes(&self.level_child_cell_bits_cache)
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn visited_count(&self) -> u64 {
        self.visited_count
    }

    pub fn block_count(&self) -> usize {
        self.storage.block_count()
    }

    pub fn block_capacity(&self) -> usize {
        self.block_capacity
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Streams every value of dimension `emit_dim` reachable under
    /// `restrictions` (the "two-Exact"/`[Exact, Exact, Free]` and
    /// "one-Exact"/`[Exact, Free, Free]` BGP shapes named and benchmarked
    /// against `Heap::restricted_descent` elsewhere in this project, now
    /// against the deduplicated DAG), stopping as soon as `emit` returns
    /// `Break` -- no cache, no per-node allocation, same contract as
    /// `Heap::restricted_descent_visit` (not sorted, may repeat).
    ///
    /// Scoped to `Exact`/`Free` restrictions only (`Range`/`Multi` are
    /// rejected the same way `Heap`'s own fast path reserves those for a
    /// slower, per-node engine -- see `tree::exact_bound_dims`'s doc
    /// comment): a `Range`/`Multi` restriction can admit more than one
    /// sub-index per level, so there is no longer a single `lo` to
    /// precompute every later level's mask from, which is exactly what
    /// this method's one-mask-per-level precomputation needs to be
    /// valid.
    ///
    /// # Why one mask per level, not one `restriction_mask` call per node
    ///
    /// The first version of this traversal called [`restriction_mask`]
    /// fresh, from the full `restrictions` array, on every node visited
    /// -- structurally mirroring `Heap::query_rec_restricted`'s general
    /// engine. Measured against `Heap::restricted_descent_visit`, that
    /// lost on `one-Exact` in most domain shapes, including ones where
    /// `Heap`'s own `bulk_list_subtree` fast path never fires at all --
    /// ruling that fast path out as the explanation. The real cause,
    /// found by reading `Heap::restricted_descent_visit` itself
    /// (`src/tree.rs:2310`): `Heap::exact_bound_dims` accepts *zero,
    /// one, or two* `Exact` dimensions, so both `one-Exact` and
    /// `two-Exact` get `Heap`'s `precompute_level_masks`/`query_rec` fast
    /// path -- one `u64` mask computed per tree *level*, before any
    /// recursion starts, so the walk itself never calls anything
    /// mask-computing per node. This method mirrors that trick directly:
    /// [`Self::precompute_level_masks`] walks `lo` forward once,
    /// advancing only the `Exact` dimensions' contribution (the only
    /// ones [`restriction_mask`] ever reads `lo` through -- `Free`'s
    /// branch, `expand_run(mask, 0, width, shifts[d])`, never reads
    /// `lo[d]` at all, the same path-independence argument this file's
    /// `(address, level)`-keyed caching experiments rely on), and
    /// [`Self::visit_rec`] then does only `word & level_masks[level]` --
    /// an array index and a bitwise AND, nothing else, at every node.
    /// Measured fix: closes nearly all of the `one-Exact` gap against
    /// `Heap` (down to 0.53x-1.07x from 1.12x-1.54x across the four
    /// domain shapes this project benchmarks, *beating* `Heap` outright
    /// in two of them) and improves `two-Exact` further on top of an
    /// already-measured win. See `DESIGN.md`'s "BGP-style queries on the
    /// FST" entry and its correction for the full measured table and
    /// the caching experiments (kept separately, still not worth their
    /// own bookkeeping cost) this traversal underlies.
    pub fn restricted_descent_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2::restricted_descent_visit only supports Exact/Free restrictions (dimension {d} got a Range/Multi restriction)"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        self.visit_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, &mut emit)
    }

    /// Mirrors `tree::precompute_level_masks` exactly, including its
    /// fixed-size-array return type (`[u64; MAX_HEIGHT]`, not `Vec<u64>`)
    /// -- a one-query heap allocation here would be pure waste on top of
    /// what's otherwise a handful of levels of cheap arithmetic, and
    /// `MAX_HEIGHT` (64) already bounds every real build (`height` is
    /// `total_bits_needed(..).iter().sum()` divided by the schedule's
    /// per-level minimum, nowhere near 64 for any domain this project
    /// uses). Walks `lo` forward one level at a time, advancing only the
    /// `Exact` dimensions' contribution, calling [`restriction_mask`]
    /// once per level instead of once per node -- see
    /// [`Self::restricted_descent_visit`]'s doc comment for why that's
    /// sound and what it replaced.
    fn precompute_level_masks(&self, restrictions: &[DimRestriction; NUM_DIMS]) -> [u64; MAX_HEIGHT] {
        debug_assert!(self.height <= MAX_HEIGHT);
        let mut lo = [0u64; NUM_DIMS];
        let mut masks = [0u64; MAX_HEIGHT];
        for level in 0..self.height {
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            masks[level] = restriction_mask(dim_bits, shifts, lo, child_cell_bits, restrictions);
            for d in 0..NUM_DIMS {
                if let DimRestriction::Exact(v) = restrictions[d] {
                    let sub = (v as u64 - lo[d]) >> child_cell_bits[d];
                    lo[d] += sub << child_cell_bits[d];
                }
            }
        }
        masks
    }

    fn visit_rec<T>(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        if is_last {
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                emit(child_lo[emit_dim] as u32)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr - delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            self.visit_rec(child_addr, child_lo, level + 1, level_masks, emit_dim, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent_visit`], but sorted,
    /// deduplicated, and collected into a `Vec` -- same contract as
    /// `Heap::restricted_descent`.
    pub fn restricted_descent(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same as [`Self::restricted_descent`], but also reports how many
    /// `(address, level)` pairs were served from the state-reuse cache
    /// vs freshly walked, and takes `use_cache` so the identical
    /// traversal can be measured *without* reuse -- isolating what the
    /// cache itself is worth, not just the restricted-descent shape.
    ///
    /// # Why this is safe to cache by `(address, level)` alone
    ///
    /// `restriction_mask` (reused directly from `tree`, not
    /// reimplemented) folds `lo` into its result only through `Exact`
    /// dimensions (`Free`'s branch, `expand_run(mask, 0, width,
    /// shifts[d])`, never reads `lo[d]` at all) and `advance_lo`'s
    /// per-level increment to any dimension is a pure function of
    /// `(qi, dim_bits, shifts, child_cell_bits)` -- level constants --
    /// never of the incoming `lo` value itself. So as long as every
    /// restriction is `Exact` or `Free` (never `Range`/`Multi`, which
    /// can admit more than one sub-index and so more than one possible
    /// `lo` for the *same* bound dimension at the same level -- the
    /// precondition this whole argument needs), an `Exact` dimension's
    /// `lo` at a given level is a deterministic function of that
    /// level and its bound value alone, not of which physical address
    /// got us there. Two different free-dimension branches that land on
    /// the *same* deduplicated address at the same level are therefore
    /// guaranteed to see an identical mask and recurse into an
    /// identical sub-structure; the only thing that differs between them
    /// is a fixed additive offset to `emit_dim`'s final value. That
    /// means a `(address, level)`'s contribution can be computed once,
    /// *relative* to its own entry `lo[emit_dim]`, cached, and reused by
    /// every other branch that reaches it -- an automaton/NFA-style
    /// "cache this state's accepted outputs" reuse, the FST-native
    /// counterpart to `Heap::query_rec`'s node-by-node descent, not just
    /// a storage-level trick. See `DESIGN.md`'s "BGP-style queries on
    /// the FST" entry for measured cache hit ratios and the resulting
    /// speedup (or lack of one) across domain shapes.
    pub fn restricted_descent_with_stats(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, use_cache: bool) -> (Vec<u32>, u64, u64) {
        for (d, r) in restrictions.iter().enumerate() {
            if let DimRestriction::Exact(v) = *r
                && (v as u64) >= self.dimension_sizes[d]
            {
                return (Vec::new(), 0, 0);
            }
        }
        if self.height == 0 {
            return (Vec::new(), 0, 0);
        }
        let mut cache: HashMap<(u64, usize), Vec<u32>> = HashMap::new();
        let mut cache_hits = 0u64;
        let mut cache_misses = 0u64;
        let mut values = self.collect_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &restrictions, emit_dim, use_cache, &mut cache, &mut cache_hits, &mut cache_misses);
        values.sort_unstable();
        values.dedup();
        (values, cache_hits, cache_misses)
    }

    /// Returns every `emit_dim` value reachable from `addr` (absolute,
    /// unsorted, may contain duplicates across sibling branches -- the
    /// caller sorts/dedups once at the top). When `use_cache`, the
    /// *relative* form (offsets from this call's own `lo[emit_dim]`) is
    /// cached under `(addr, level)` before being shifted back to
    /// absolute values and returned -- see
    /// [`Self::restricted_descent_with_stats`]'s doc comment for why
    /// that's sound.
    #[allow(clippy::too_many_arguments)]
    fn collect_rec(
        &self,
        addr: u64,
        lo: [u64; NUM_DIMS],
        level: usize,
        restrictions: &[DimRestriction; NUM_DIMS],
        emit_dim: usize,
        use_cache: bool,
        cache: &mut HashMap<(u64, usize), Vec<u32>>,
        cache_hits: &mut u64,
        cache_misses: &mut u64,
    ) -> Vec<u32> {
        let base = lo[emit_dim] as u32;
        if use_cache {
            if let Some(rel) = cache.get(&(addr, level)) {
                *cache_hits += 1;
                return rel.iter().map(|&r| base + r).collect();
            }
            *cache_misses += 1;
        }

        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = restriction_mask(dim_bits, shifts, lo, child_cell_bits, restrictions);
        let is_last = level == self.height - 1;

        let mut absolute = Vec::new();
        let mut relative = Vec::new();

        if is_last {
            let mut relevant = word & mask;
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let val = child_lo[emit_dim] as u32;
                relative.push(val - base);
                absolute.push(val);
            }
        } else {
            let mut relevant = word & mask;
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                let child_addr = addr - delta;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_values = self.collect_rec(child_addr, child_lo, level + 1, restrictions, emit_dim, use_cache, cache, cache_hits, cache_misses);
                for v in child_values {
                    relative.push(v - base);
                    absolute.push(v);
                }
            }
        }

        if use_cache {
            cache.insert((addr, level), relative);
        }
        absolute
    }

    /// Follow-up to [`Self::restricted_descent_with_stats`]'s negative
    /// caching result, prompted by reading the `fst` crate's `Automaton`
    /// trait (see `DESIGN.md`'s entry on it): that crate's `can_match`/
    /// `will_always_match` split names a distinction this method
    /// exploits directly, which the earlier caching attempt did not --
    /// caching *every* `(address, level)` visited mixes two very
    /// different kinds of state:
    ///
    /// - while any `Exact` dimension still has bits left to resolve, a
    ///   node's mask is actually restrictive (`word & mask != word` in
    ///   general), and a single query's own descent only ever revisits
    ///   the same `(address, level)` pair by accident (measured hit
    ///   rates: 0.00-9.05%, see the table above) -- not worth caching;
    /// - once every dimension is `Free` or has run out of bits (`tree`'s
    ///   own `first_fully_open_level_restricted` condition, recomputed
    ///   here as [`Self::first_fully_open_level`] since `PathFstDeltaPow2`
    ///   has no `RadixTree` to call it on), the mask is *always* the full
    ///   word -- `will_always_match`, in `fst`'s vocabulary. From that
    ///   point down, "what does this address emit" no longer depends on
    ///   the query at all, only on the DAG's own shared structure -- the
    ///   same quantity `node_count`/`visited_count` already measure a
    ///   15-21% dedup ratio for. Caching is scoped to *only* this region
    ///   ([`Self::bulk_collect_rec`], entered once `level >=
    ///   unconstrained_from`): far fewer distinct `(address, level)`
    ///   pairs than the whole-query caching attempt, and each one is a
    ///   real candidate for the DAG's actual sharing to pay off, not a
    ///   query-path coincidence.
    ///
    /// Everything before `unconstrained_from` still uses a zero-
    /// allocation streaming walk (`visit_rec_fast`'s early-return
    /// branch, predating [`Self::restricted_descent_visit`]'s later
    /// precomputed-mask optimization -- this one still calls
    /// [`restriction_mask`] per node) that already beat the cached/
    /// no-cache `Vec`-merging forms on every shape -- this method is
    /// that traversal with one targeted caching addition, not a
    /// replacement of it.
    pub fn restricted_descent_fastpath(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _ = self.restricted_descent_fastpath_with_stats(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue::<()>(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same shape/contract as [`Self::restricted_descent_visit`], plus
    /// `(cache_hits, cache_misses)` for the bulk fully-open region only
    /// -- see [`Self::restricted_descent_fastpath`]'s doc comment for
    /// why caching is scoped there and not to the whole descent.
    pub fn restricted_descent_fastpath_with_stats<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> (std::ops::ControlFlow<T>, u64, u64) {
        for (d, r) in restrictions.iter().enumerate() {
            if let DimRestriction::Exact(v) = *r
                && (v as u64) >= self.dimension_sizes[d]
            {
                return (std::ops::ControlFlow::Continue(()), 0, 0);
            }
        }
        if self.height == 0 {
            return (std::ops::ControlFlow::Continue(()), 0, 0);
        }
        let unconstrained_from = self.first_fully_open_level(&restrictions);
        let mut cache: HashMap<(u64, usize), Vec<u32>> = HashMap::new();
        let mut hits = 0u64;
        let mut misses = 0u64;
        let flow = self.visit_rec_fast(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &restrictions, emit_dim, unconstrained_from, &mut cache, &mut hits, &mut misses, &mut emit);
        (flow, hits, misses)
    }

    /// `tree::first_fully_open_level_restricted`'s counterpart: the
    /// first level at which every dimension is `Free` or has already
    /// exhausted its bit budget (`level_dim_bits[level][d] == 0`),
    /// computed from the build schedule alone, not any node's actual
    /// content -- same reasoning as `tree`'s version, reimplemented
    /// here rather than called since `PathFstDeltaPow2` has no
    /// `RadixTree` to call it on.
    fn first_fully_open_level(&self, restrictions: &[DimRestriction; NUM_DIMS]) -> usize {
        for level in 0..self.height {
            let dim_bits = self.level_dim_bits[level];
            let all_open = (0..NUM_DIMS).all(|d| matches!(restrictions[d], DimRestriction::Free) || dim_bits[d] == 0);
            if all_open {
                return level;
            }
        }
        self.height
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_rec_fast<T>(
        &self,
        addr: u64,
        lo: [u64; NUM_DIMS],
        level: usize,
        restrictions: &[DimRestriction; NUM_DIMS],
        emit_dim: usize,
        unconstrained_from: usize,
        cache: &mut HashMap<(u64, usize), Vec<u32>>,
        hits: &mut u64,
        misses: &mut u64,
        emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>,
    ) -> std::ops::ControlFlow<T> {
        if level >= unconstrained_from {
            let values = self.bulk_collect_rec(addr, lo, level, emit_dim, cache, hits, misses);
            for v in values {
                emit(v)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mask = restriction_mask(dim_bits, shifts, lo, child_cell_bits, restrictions);
        let is_last = level == self.height - 1;
        let mut relevant = word & mask;

        if is_last {
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                emit(child_lo[emit_dim] as u32)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr - delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            self.visit_rec_fast(child_addr, child_lo, level + 1, restrictions, emit_dim, unconstrained_from, cache, hits, misses, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// The fully-open-region traversal: no `restriction_mask` call at
    /// all (every set bit in `word` is already a match, by
    /// construction once `level >= unconstrained_from`), and every
    /// `(addr, level)` visited is cached -- unlike
    /// [`Self::collect_rec`], which caches unconditionally regardless
    /// of whether the query is still restrictive at that point.
    #[allow(clippy::too_many_arguments)]
    fn bulk_collect_rec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, emit_dim: usize, cache: &mut HashMap<(u64, usize), Vec<u32>>, hits: &mut u64, misses: &mut u64) -> Vec<u32> {
        let base = lo[emit_dim] as u32;
        if let Some(rel) = cache.get(&(addr, level)) {
            *hits += 1;
            return rel.iter().map(|&r| base + r).collect();
        }
        *misses += 1;

        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let is_last = level == self.height - 1;
        let mut remaining = word;
        let mut absolute = Vec::new();
        let mut relative = Vec::new();

        if is_last {
            while remaining != 0 {
                let qi = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let val = child_lo[emit_dim] as u32;
                relative.push(val - base);
                absolute.push(val);
            }
        } else {
            while remaining != 0 {
                let qi = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                let child_addr = addr - delta;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_values = self.bulk_collect_rec(child_addr, child_lo, level + 1, emit_dim, cache, hits, misses);
                for v in child_values {
                    relative.push(v - base);
                    absolute.push(v);
                }
            }
        }

        cache.insert((addr, level), relative);
        absolute
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rec_pow2(
    points: &mut [Triple],
    scratch: &mut [Triple],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: usize,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    storage: &mut HeapStorage,
    block_capacity: usize,
    capacity_bits: u32,
    register: &mut [HashMap<(u64, Vec<u64>), u64>],
    node_count: &mut u64,
    visited_count: &mut u64,
) -> u64 {
    *visited_count += 1;
    let dim_bits = level_dim_bits[level];
    let shifts = level_shifts_cache[level];
    let child_cell_bits = level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
    debug_assert!(branching <= K_SQUARE as usize);

    let qi_of = |pt: &Triple| compute_qi([pt.0 as u64, pt.1 as u64, pt.2 as u64], lo, shifts, child_cell_bits);
    let mut counts = [0usize; K_SQUARE as usize];
    for pt in points.iter() {
        counts[qi_of(pt)] += 1;
    }
    let mut offsets = [0usize; K_SQUARE as usize + 1];
    for i in 0..branching {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let mut cursor = offsets;
    for &pt in points.iter() {
        let qi = qi_of(&pt);
        scratch[cursor[qi]] = pt;
        cursor[qi] += 1;
    }
    points.copy_from_slice(&scratch[..points.len()]);

    let mut existence_word = 0u64;
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            existence_word |= 1u64 << qi;
        }
    }

    let virtual_addr_of = |pos: Pos| -> u64 { (pos.block as u64) << capacity_bits | pos.offset as u64 };

    if level == height - 1 {
        let key = (existence_word, Vec::new());
        if let Some(&addr) = register[level].get(&key) {
            return addr;
        }
        let pos = storage.append_record(&existence_word.to_le_bytes(), block_capacity);
        let addr = virtual_addr_of(pos);
        register[level].insert(key, addr);
        *node_count += 1;
        return addr;
    }

    let mut child_addrs = Vec::with_capacity(existence_word.count_ones() as usize);
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            let (child_points, child_scratch) = (&mut points[offsets[qi]..offsets[qi + 1]], &mut scratch[offsets[qi]..offsets[qi + 1]]);
            let child_addr = build_rec_pow2(
                child_points,
                child_scratch,
                child_lo,
                level + 1,
                height,
                level_dim_bits,
                level_shifts_cache,
                level_child_cell_bits_cache,
                storage,
                block_capacity,
                capacity_bits,
                register,
                node_count,
                visited_count,
            );
            child_addrs.push(child_addr);
        }
    }

    let key = (existence_word, child_addrs.clone());
    if let Some(&addr) = register[level].get(&key) {
        return addr;
    }
    // Placeholder-then-patch: this node's own virtual address isn't known
    // until `append_record` tells us which block it actually landed in.
    let mut rec = vec![0u8; WORD_BYTES + DELTA_BYTES as usize * child_addrs.len()];
    rec[..WORD_BYTES].copy_from_slice(&existence_word.to_le_bytes());
    let pos = storage.append_record(&rec, block_capacity);
    let addr = virtual_addr_of(pos);
    for (i, &child_addr) in child_addrs.iter().enumerate() {
        let delta = (addr - child_addr) as u32;
        storage.patch_appended(pos, WORD_BYTES + i * DELTA_BYTES as usize, DELTA_BYTES, delta as u64);
    }
    register[level].insert(key, addr);
    *node_count += 1;
    addr
}
