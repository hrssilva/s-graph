//! FST layouts and historical comparison implementations.
//! [`PathFstDeltaPow2Bfs`] is the default [`crate::Fst`] storage; the other
//! layouts remain available for measurement. See DESIGN.md for their history. Builds a minimized (deduplicated)
//! FST-style representation of the same DFS-interleaved, bit-
//! interleaved radix structure `tree` uses -- same per-dimension bit
//! schedule (`total_bits_needed`/`dynamic_schedule`/`compute_qi`/
//! `advance_lo`, reused directly from `tree`, not reimplemented), so a
//! comparison against the production tree is apples-to-apples. The promoted
//! packed BFS layout supports all block backends and path-copying updates via
//! `into_mutable`; historical comparison layouts remain static.
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

/// Chunk size for [`PathFstDeltaPow2::restricted_descent_bfs_visit`]'s
/// fixed-size, stack-only frontier buffers -- see that method's doc
/// comment. Large enough to give the CPU plenty of independent
/// outstanding reads to overlap (modern cores track on the order of a
/// few dozen concurrent misses), small enough that the two buffers per
/// recursion frame (`8 + 24` bytes each) stay a trivial fraction of the
/// default stack across `MAX_HEIGHT` levels of recursion.
const BFS_WINDOW: usize = 256;

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
/// delta instead of an 8-byte `Pos`, and uses a real `BlockIndex` --
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
    /// The first version of this traversal called `restriction_mask`
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
    /// `Self::precompute_level_masks` walks `lo` forward once,
    /// advancing only the `Exact` dimensions' contribution (the only
    /// ones `restriction_mask` ever reads `lo` through -- `Free`'s
    /// branch, `expand_run(mask, 0, width, shifts[d])`, never reads
    /// `lo[d]` at all, the same path-independence argument this file's
    /// `(address, level)`-keyed caching experiments rely on), and
    /// `Self::visit_rec` then does only `word & level_masks[level]` --
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
    /// `Exact` dimensions' contribution, calling `restriction_mask`
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

    /// Prototype: same contract and traversal as
    /// [`Self::restricted_descent_visit`], but before recursing into any
    /// of a node's real matching children, first decodes *all* of their
    /// deltas (cheap -- they're bytes immediately after `word`, already
    /// in the same cache line that read pulled in, per `DESIGN.md`'s
    /// "deeper FST-only perf dive" entry) and issues a discarded
    /// `read_u64` at each child's address, before touching any of them
    /// for real -- a software-prefetch *hint* via `_mm_prefetch` isn't
    /// available (this crate is `#![forbid(unsafe_code)]`, and there is
    /// no safe wrapper in `std`), so a real, safe, result-discarded read
    /// stands in for it: it still triggers the hardware's cache fill for
    /// that address, and it carries no correctness risk either way since
    /// a `HeapStorage` read is already infallible and side-effect-free.
    ///
    /// Motivation, from the measured bottleneck: `one-Exact` on the
    /// larger domains runs at IPC ~1.0 with a 35-42% cache-miss rate --
    /// memory-latency-bound, not throughput-bound (see `DESIGN.md`'s
    /// `perf`-rigorous margin entry). Plain `Self::visit_rec` discovers
    /// each child's address and immediately recurses into it, fully
    /// serializing one miss behind the previous one's entire subtree --
    /// zero memory-level parallelism. This method issues up to the
    /// node's own fan-out (bounded, usually small per the node-histogram
    /// data -- ~9 on average at the level this costs the most) many
    /// independent reads back-to-back instead of one at a time, so the
    /// CPU's out-of-order execution can dispatch several of their misses
    /// concurrently instead of queuing them up serially one recursive
    /// call at a time. Scoped locally to one node's own children
    /// (bounded, no growable buffer) rather than a full BFS/frontier
    /// restructuring across the whole level, specifically to test the
    /// idea's value without first paying for a bigger, riskier rewrite
    /// and its own new allocation cost.
    pub fn restricted_descent_prefetch_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("restricted_descent_prefetch_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        self.visit_rec_prefetch(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, &mut emit)
    }

    fn visit_rec_prefetch<T>(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        if is_last {
            let mut relevant = relevant;
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                emit(child_lo[emit_dim] as u32)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        // This crate forbids `unsafe` (`#![forbid(unsafe_code)]`), so
        // there's no `_mm_prefetch` available -- a real, safe
        // `read_u64` whose result is discarded serves the same purpose:
        // it still triggers the hardware's cache fill for that address,
        // and because every one of these reads is independent of the
        // others (no data dependency between siblings), the CPU's own
        // out-of-order execution can already dispatch several of them
        // concurrently once they're issued back-to-back like this,
        // rather than each one only being issued after the previous
        // child's entire recursive subtree has finished. `black_box`
        // stops the optimizer from proving the discarded value is dead
        // and eliding the read entirely.
        let mut prefetch_scan = relevant;
        while prefetch_scan != 0 {
            let qi = prefetch_scan.trailing_zeros() as usize;
            prefetch_scan &= prefetch_scan - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr - delta;
            let child_block = (child_addr >> self.capacity_bits) as usize;
            let child_offset = (child_addr & self.capacity_mask) as usize;
            std::hint::black_box(self.storage.read_u64(child_block, child_offset));
        }

        let mut relevant = relevant;
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr - delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            self.visit_rec_prefetch(child_addr, child_lo, level + 1, level_masks, emit_dim, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Prototype: the BFS/frontier restructuring the sibling-prefetch
    /// attempt's own negative result argued for (see that method's doc
    /// comment and `DESIGN.md`'s matching entry). The problem diagnosed
    /// there wasn't the idea of overlapping memory-level parallelism --
    /// it was that DFS makes the gap between "prefetch a sibling" and
    /// "use it" a whole subtree, long enough that the warmed cache line
    /// is usually evicted again before it's needed. Processing one tree
    /// *level* flat, all at once, instead of recursing depth-first,
    /// makes that gap "the rest of this level's frontier" instead -- a
    /// fundamentally shorter, more survivable distance, at the cost of
    /// giving up the zero-allocation property every other traversal in
    /// this file has: the frontier can be large (tens of thousands of
    /// entries at the level the earlier node-histogram investigation
    /// found costs the most), so it needs a real, growable buffer.
    ///
    /// Issues a discarded `read_u64` (same `#![forbid(unsafe_code)]`
    /// constraint as [`Self::restricted_descent_prefetch_visit`] -- no
    /// `_mm_prefetch` available) for *every* node in the current level's
    /// frontier before reading any of them for real, then does the real
    /// reads and advances to the next level's frontier. Two buffers
    /// (`current`/`next`) swapped each level instead of reallocated.
    pub fn restricted_descent_frontier_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("restricted_descent_frontier_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let mut current: Vec<(u64, [u64; NUM_DIMS])> = vec![(self.root_virtual_addr, [0u64; NUM_DIMS])];
        let mut next: Vec<(u64, [u64; NUM_DIMS])> = Vec::new();

        for level in 0..self.height {
            if current.is_empty() {
                break;
            }
            let is_last = level == self.height - 1;
            let mask = level_masks[level];
            let dim_bits = self.level_dim_bits[level];
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];

            for &(addr, _) in &current {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                std::hint::black_box(self.storage.read_u64(block, offset));
            }

            next.clear();
            for &(addr, lo) in &current {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                let word = self.storage.read_u64(block, offset);
                let mut relevant = word & mask;

                if is_last {
                    while relevant != 0 {
                        let qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                        emit(child_lo[emit_dim] as u32)?;
                    }
                    continue;
                }

                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                    let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                    let child_addr = addr - delta;
                    let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                    next.push((child_addr, child_lo));
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_frontier_visit`] instead.
    pub fn restricted_descent_frontier(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_frontier_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Prototype: the same level-flat prefetch-then-use batching as
    /// [`Self::restricted_descent_frontier_visit`], but without that
    /// method's `Vec<(u64, [u64; NUM_DIMS])>` buffers -- every node held
    /// in flight lives in a fixed-size `[(..); BFS_WINDOW]` array on the
    /// stack, never the heap. A level's frontier can be wider than
    /// `BFS_WINDOW`, so it's processed in sequential chunks of at most
    /// `BFS_WINDOW` nodes each: a chunk is warmed (discarded reads for
    /// every node in it), then read for real, and any children it
    /// produces are buffered into a *second* fixed array that gets
    /// flushed (one recursive call, one level deeper) as soon as it
    /// fills, instead of being allowed to grow. Recursion depth is
    /// bounded by `self.height` (<= `MAX_HEIGHT`), same as the plain
    /// `visit_rec` DFS, so this is no deeper than the existing recursive
    /// traversal already goes -- it just carries a `BFS_WINDOW`-sized
    /// array per frame instead of one node's worth of state.
    ///
    /// Trade-off versus the `Vec`-based frontier: a level wider than
    /// `BFS_WINDOW` loses some prefetch breadth (each chunk only warms
    /// up to `BFS_WINDOW` siblings-across-parents at a time, not the
    /// whole level at once), but needs no heap allocation at all. Kept
    /// as a separate, measured variant -- see `DESIGN.md`.
    pub fn restricted_descent_bfs_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("restricted_descent_bfs_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let root = [(self.root_virtual_addr, [0u64; NUM_DIMS])];
        self.bfs_window_visit(&root, 0, &level_masks, emit_dim, &mut emit)
    }

    /// Helper for [`Self::restricted_descent_bfs_visit`] -- see its doc
    /// comment. `nodes` is a caller-owned slice (a chunk of the parent's
    /// own fixed-size buffer, or the one-element root array), never
    /// something this function allocates.
    fn bfs_window_visit<T>(&self, nodes: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        if nodes.is_empty() {
            return std::ops::ControlFlow::Continue(());
        }
        let is_last = level == self.height - 1;
        let mask = level_masks[level];
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        for chunk in nodes.chunks(BFS_WINDOW) {
            for &(addr, _) in chunk {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                std::hint::black_box(self.storage.read_u64(block, offset));
            }

            let mut next_buf = [(0u64, [0u64; NUM_DIMS]); BFS_WINDOW];
            let mut next_len = 0usize;

            for &(addr, lo) in chunk {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                let word = self.storage.read_u64(block, offset);
                let mut relevant = word & mask;

                if is_last {
                    while relevant != 0 {
                        let qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                        emit(child_lo[emit_dim] as u32)?;
                    }
                    continue;
                }

                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                    let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                    let child_addr = addr - delta;
                    let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);

                    next_buf[next_len] = (child_addr, child_lo);
                    next_len += 1;
                    if next_len == BFS_WINDOW {
                        self.bfs_window_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
                        next_len = 0;
                    }
                }
            }

            if next_len > 0 {
                self.bfs_window_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
            }
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_bfs_visit`] instead.
    pub fn restricted_descent_bfs(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_bfs_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
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

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_prefetch_visit`] instead -- for the
    /// correctness check and timing comparison in
    /// `examples/path_fst_bgp_proto.rs`.
    pub fn restricted_descent_prefetch(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_prefetch_visit(restrictions, emit_dim, |v| {
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
    ///   here as `Self::first_fully_open_level` since `PathFstDeltaPow2`
    ///   has no `RadixTree` to call it on), the mask is *always* the full
    ///   word -- `will_always_match`, in `fst`'s vocabulary. From that
    ///   point down, "what does this address emit" no longer depends on
    ///   the query at all, only on the DAG's own shared structure -- the
    ///   same quantity `node_count`/`visited_count` already measure a
    ///   15-21% dedup ratio for. Caching is scoped to *only* this region
    ///   (`Self::bulk_collect_rec`, entered once `level >=
    ///   unconstrained_from`): far fewer distinct `(address, level)`
    ///   pairs than the whole-query caching attempt, and each one is a
    ///   real candidate for the DAG's actual sharing to pay off, not a
    ///   query-path coincidence.
    ///
    /// Everything before `unconstrained_from` still uses a zero-
    /// allocation streaming walk (`visit_rec_fast`'s early-return
    /// branch, predating [`Self::restricted_descent_visit`]'s later
    /// precomputed-mask optimization -- this one still calls
    /// `restriction_mask` per node) that already beat the cached/
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

// ==================== BFS-ordered layout variant ====================

/// A node discovered during [`PathFstDeltaPow2Bfs::build`]'s bottom-up
/// dedup pass -- `child_indices` refers to positions within
/// `levels_nodes[level + 1]`, not yet real storage addresses (those
/// aren't assigned until the build's second phase).
pub(crate) struct BfsNodeMeta {
    pub(crate) existence_word: u64,
    pub(crate) child_indices: Vec<u32>,
}

/// Pairs with [`PathFstDeltaPow2`]'s own BFS/frontier and fixed-window
/// `restricted_descent_bfs_visit` traversals (see `DESIGN.md`): those
/// win by overlapping or reducing cache-*miss latency* while visiting a
/// whole tree level at once, but they still pay for the *count* of
/// misses a level-flat visit causes, because `PathFstDeltaPow2`'s
/// storage layout is still physically post-order -- a node's address is
/// wherever its *first* discovering parent happened to leave it, with no
/// relationship to the other nodes at its own level.
///
/// This variant changes that: distinct nodes are still deduplicated the
/// same way (two DAG nodes with identical `(existence_word,
/// child_identity)` still share one physical record), but they're
/// *addressed* in BFS/level order -- every distinct level-0 node first,
/// then every distinct level-1 node, and so on -- so that the exact set
/// of nodes a level-flat traversal (`restricted_descent_bfs_visit`,
/// below) touches together is also contiguous in storage, not scattered
/// by whichever parent happened to build each one first.
///
/// Dedup is inherently bottom-up (you can't know two subtrees are
/// identical until both are fully built), but this layout is inherently
/// top-down (level 0 must be *written* before level 1). Building it is
/// therefore three passes instead of `build_rec_pow2`'s one:
/// 1. `collect_rec_bfs` -- the same recursive dedup as
///    `build_rec_pow2`, but instead of writing to storage, it records
///    each distinct node's `(existence_word, child_indices)` into a
///    per-level `Vec`, where `child_indices` are positions within the
///    child level's own `Vec`, not addresses.
/// 2. A level-by-level top-down loop assigns every node's real storage
///    position (`append_record`, with deltas still zeroed -- a node's
///    *own* address is known the moment it's appended, but its
///    children's addresses aren't assigned until the next loop
///    iteration).
/// 3. A second top-down loop backfills every real delta via
///    `patch_appended`, now that every level's addresses are known.
///    Deltas are still unsigned and `DELTA_BYTES` wide, but the sign is
///    the opposite of [`PathFstDeltaPow2`]'s: a child is always at a
///    *higher* address than its parent here (it's written in a later
///    level), so `delta = child_addr - addr`, and every read path adds
///    the delta instead of subtracting it.
///
/// The default immutable graph index, also exported as [`crate::Fst`].
/// Build a set of integer triples, then scan patterns, compose BGPs and paths,
/// or stream adaptive joins with [`crate::Join`]. Snapshots preserve the packed
/// DAG directly through validated, versioned serialization.
///
/// [`Self::scan`] supports exact/wildcard triple bounds, [`Self::select`]
/// supports projected exact/range/set restrictions, and [`Self::bgp`] integrates
/// with binding-aware OPTIONAL, MINUS and existence operators. This type also
/// implements [`crate::join::PatternBackend`] and [`crate::path::AdjacencyBackend`].
///
/// Use `into_mutable` for local insertion/removal while preserving shared
/// suffixes. Explicit traversal variants below remain available for
/// comparison; DESIGN.md records their algorithms and measurements.
/// Backwards-compatible name for the packed heap FST.
pub type PathFstDeltaPow2Bfs = PackedFst<HeapStorage>;

/// Packed suffix-sharing graph, generic over heap, file or compressed blocks.
pub struct PackedFst<S: BlockStorage> {
    storage: S,
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
    level_node_counts: Vec<usize>,
    pub(crate) scan_layout: Vec<crate::fst::ScanLayout>,
    visited_count: u64,
    /// Lazily retained only by diagnostic sweep comparisons. Default queries
    /// and serialization do not allocate this per-node directory.
    level_addrs: std::sync::OnceLock<Vec<Vec<u64>>>,
    /// Update scans need one start cursor per depth, not a per-state directory.
    level_starts: std::sync::OnceLock<Vec<u64>>,
}

impl<S: BlockStorage> PackedFst<S> {
    pub fn build(triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        Self::try_build(triples, dimension_sizes, block_capacity).expect("invalid FST build")
    }

    /// Validated static build. Domains are nonzero sizes, each at most 2^32;
    /// blocks must be powers of two of at least 512 bytes.
    pub fn try_build(mut triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> crate::Result<Self> {
        use crate::{TreeError, OutOfDomain};
        if !dimension_sizes.iter().all(|n| (1..=1u64 << 32).contains(n)) {
            return Err(TreeError::InvalidConfig("FST domains must be in 1..=2^32".into()));
        }
        if !block_capacity.is_power_of_two() || !(512..=u32::MAX as usize).contains(&block_capacity) {
            return Err(TreeError::InvalidConfig("FST block capacity must be a power of two in 512..=2^31".into()));
        }
        if triples.len() > u32::MAX as usize {
            return Err(TreeError::InvalidConfig("FST build exceeds 32-bit node indices".into()));
        }
        for &(s, p, o) in &triples {
            for (dim, value) in [s, p, o].into_iter().enumerate() {
                if value as u64 >= dimension_sizes[dim] {
                    return Err(TreeError::OutOfDomain(OutOfDomain { dim, value, domain_size: dimension_sizes[dim] }));
                }
            }
        }
        let dim_order = [0usize, 1, 2];
        let total_bits = total_bits_needed(dimension_sizes);
        let mut level_dim_bits = dynamic_schedule(total_bits);
        // One zero-bit transition represents (0,0,0), including its absence.
        if level_dim_bits.is_empty() { level_dim_bits.push([0; NUM_DIMS]); }
        let height = level_dim_bits.len();
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

        // Phase 1: bottom-up dedup, no storage writes yet.
        let mut register: Vec<HashMap<(u64, Vec<u32>), u32>> = (0..height).map(|_| HashMap::new()).collect();
        let mut levels_nodes: Vec<Vec<BfsNodeMeta>> = (0..height).map(|_| Vec::new()).collect();
        let mut visited_count = 0u64;
        let mut scratch = vec![(0u32, 0u32, 0u32); triples.len()];
        collect_rec_bfs(
            &mut triples,
            &mut scratch,
            [0; NUM_DIMS],
            0,
            height,
            &level_dim_bits,
            &level_shifts_cache,
            &level_child_cell_bits_cache,
            &mut register,
            &mut levels_nodes,
            &mut visited_count,
        );

        Self::pack_nodes(levels_nodes, dimension_sizes, block_capacity, visited_count)
    }

    pub(crate) fn pack_nodes(levels_nodes: Vec<Vec<BfsNodeMeta>>, dimension_sizes: [u64; 3], block_capacity: usize, visited_count: u64) -> crate::Result<Self> {
        use crate::TreeError;
        let capacity_bits = block_capacity.trailing_zeros();
        let capacity_mask = (block_capacity - 1) as u64;
        let mut level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        if level_dim_bits.is_empty() { level_dim_bits.push([0; 3]); }
        let height = level_dim_bits.len();
        let level_shifts_cache: Vec<_> = level_dim_bits.iter().map(|&bits| level_shifts(bits, [0, 1, 2])).collect();
        let mut remaining = total_bits_needed(dimension_sizes);
        let level_child_cell_bits_cache: Vec<_> = level_dim_bits.iter().map(|bits| {
            for d in 0..3 { remaining[d] -= bits[d]; }
            remaining
        }).collect();
        // Assign addresses without writing placeholder records. Once all child
        // addresses are known, append each complete record once. In particular,
        // compressed storage must not recompress a chunk per delta backfill.
        let mut addr_of: Vec<Vec<u64>> = Vec::with_capacity(height);
        let (mut block, mut offset) = (0u64, 0usize);
        let mut node_count = 0u64;
        for nodes in &levels_nodes {
            let mut addresses = Vec::with_capacity(nodes.len());
            for node in nodes {
                let len = WORD_BYTES + DELTA_BYTES as usize * node.child_indices.len();
                if offset != 0 && offset + len > block_capacity { block += 1; offset = 0; }
                addresses.push((block << capacity_bits) | offset as u64);
                offset += len;
                node_count += 1;
            }
            addr_of.push(addresses);
        }
        let mut storage = S::empty();
        let mut record = [0u8; WORD_BYTES + 64 * DELTA_BYTES as usize];
        for (level, nodes) in levels_nodes.iter().enumerate() {
            for (i, node) in nodes.iter().enumerate() {
                let addr = addr_of[level][i];
                record[..WORD_BYTES].copy_from_slice(&node.existence_word.to_le_bytes());
                for (slot, &child) in node.child_indices.iter().enumerate() {
                    let target = addr_of[level + 1][child as usize];
                    let delta = u32::try_from(target - addr).map_err(|_| TreeError::InvalidConfig("FST child delta exceeds 32 bits; use smaller blocks or partition the graph".into()))?;
                    let start = WORD_BYTES + slot * DELTA_BYTES as usize;
                    record[start..start + 4].copy_from_slice(&delta.to_le_bytes());
                }
                let len = WORD_BYTES + DELTA_BYTES as usize * node.child_indices.len();
                let pos = storage.append_record(&record[..len], block_capacity);
                debug_assert_eq!(((pos.block as u64) << capacity_bits) | pos.offset as u64, addr);
            }
        }
        storage.finish_append(block_capacity);

        let root_virtual_addr = addr_of[0][0];

        let scan_layout = (0..height).map(|l| crate::fst::ScanLayout::new(level_dim_bits[l], level_shifts_cache[l], level_child_cell_bits_cache[l])).collect();
        Ok(Self {
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
            scan_layout,
            level_node_counts: addr_of.iter().map(Vec::len).collect(),
            visited_count,
            level_addrs: std::sync::OnceLock::new(),
            level_starts: std::sync::OnceLock::new(),
        })
    }

    pub(crate) fn read_session(&self) -> impl Drop + '_ { self.storage.begin_read_session() }

    pub(crate) fn release_compaction_block(&mut self, block: usize)
    where S: crate::storage::ReclaimingBlockStorage {
        self.storage.release_block_payload(block);
    }
    pub(crate) fn block_len(&self, block: usize) -> usize { self.storage.block_len(block) }

    pub(crate) fn read_compaction_block(&self, block: usize, out: &mut Vec<u8>)
    where S: crate::storage::ReclaimingBlockStorage {
        self.storage.read_compaction_block(block, out);
    }

    // Trusted construction by the consuming block compactor. It writes only
    // existing canonical live states, in level order, with checked child deltas.
    pub(crate) fn from_compacted_storage(storage: S, dimension_sizes: [u64; 3], block_capacity: usize,
        level_node_counts: Vec<usize>, visited_count: u64) -> Self {
        let mut level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        if level_dim_bits.is_empty() { level_dim_bits.push([0; 3]); }
        let height = level_dim_bits.len();
        let level_shifts_cache: Vec<_> = level_dim_bits.iter().map(|&bits| level_shifts(bits, [0, 1, 2])).collect();
        let mut remaining = total_bits_needed(dimension_sizes);
        let level_child_cell_bits_cache: Vec<_> = level_dim_bits.iter().map(|bits| {
            for d in 0..3 { remaining[d] -= bits[d]; }
            remaining
        }).collect();
        let scan_layout = (0..height).map(|l| crate::fst::ScanLayout::new(level_dim_bits[l], level_shifts_cache[l], level_child_cell_bits_cache[l])).collect();
        Self { storage, root_virtual_addr: 0, height, level_dim_bits, level_shifts_cache,
            level_child_cell_bits_cache, dimension_sizes, block_capacity,
            capacity_bits: block_capacity.trailing_zeros(), capacity_mask: block_capacity as u64 - 1,
            node_count: level_node_counts.iter().sum::<usize>() as u64, level_node_counts,
            scan_layout, visited_count, level_addrs: std::sync::OnceLock::new(), level_starts: std::sync::OnceLock::new() }
    }

    /// Build only O(height) starting cursors, once, without a state directory.
    fn physical_level_starts(&self) -> &[u64] {
        self.level_starts.get_or_init(|| {
            let mut starts = Vec::with_capacity(self.height);
            let (mut block, mut offset) = (0, 0);
            for (level, &count) in self.level_node_counts.iter().enumerate() {
                if block < self.storage.block_count() && offset == self.storage.block_len(block) {
                    block += 1; offset = 0;
                }
                starts.push(((block as u64) << self.capacity_bits) | offset as u64);
                if level + 1 == self.height { break; }
                for _ in 0..count {
                    if offset == self.storage.block_len(block) { block += 1; offset = 0; }
                    let word = self.storage.read_u64(block, offset);
                    offset += WORD_BYTES + DELTA_BYTES as usize * word.count_ones() as usize;
                }
            }
            starts
        })
    }

    /// Scan only the requested depth. The first scan prepares O(height) starts
    /// in a single sequential record pass; later scans jump there directly.
    pub(crate) fn find_physical_state(&self, target_level: usize, mut matches: impl FnMut(u64, u64) -> bool) -> Option<u64> {
        let start = self.physical_level_starts()[target_level];
        let (mut block, mut offset) = ((start >> self.capacity_bits) as usize, (start & self.capacity_mask) as usize);
        for _ in 0..self.level_node_counts[target_level] {
            if offset == self.storage.block_len(block) { block += 1; offset = 0; }
            let cursor = ((block as u64) << self.capacity_bits) | offset as u64;
            let word = self.storage.read_u64(block, offset);
            if matches(cursor, word) { return Some(cursor); }
            offset += WORD_BYTES + if target_level + 1 < self.height {
                DELTA_BYTES as usize * word.count_ones() as usize
            } else { 0 };
        }
        None
    }

    /// Sequential physical records, including states retained only by storage.
    pub(crate) fn for_each_physical_state(&self, mut emit: impl FnMut(usize, u64, u64)) {
        let (mut block, mut offset) = (0, 0);
        for (level, &count) in self.level_node_counts.iter().enumerate() {
            for _ in 0..count {
                if offset == self.storage.block_len(block) { block += 1; offset = 0; }
                let cursor = ((block as u64) << self.capacity_bits) | offset as u64;
                let word = self.storage.read_u64(block, offset);
                emit(level, cursor, word);
                offset += WORD_BYTES + if level + 1 < self.height { DELTA_BYTES as usize * word.count_ones() as usize } else { 0 };
            }
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
            addr += delta;
            lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        }
        unreachable!("height > 0 loop always returns from inside")
    }

    /// Retained heap allocations, including diagnostic directories if requested.
    pub fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes()
            + crate::vec_bytes(&self.scan_layout)
            + crate::vec_bytes(&self.level_node_counts)
            + crate::vec_bytes(&self.level_dim_bits)
            + crate::vec_bytes(&self.level_shifts_cache)
            + crate::vec_bytes(&self.level_child_cell_bits_cache)
            + self.level_starts.get().map_or(0, crate::vec_bytes)
            + self.level_addrs.get().map_or(0, |levels| crate::vec_bytes(levels) + levels.iter().map(crate::vec_bytes).sum::<usize>())
    }

    fn level_addresses(&self) -> &Vec<Vec<u64>> {
        self.level_addrs.get_or_init(|| {
            let mut levels = vec![vec![self.root_virtual_addr]];
            for level in 1..self.height {
                let mut next = Vec::new();
                for &addr in &levels[level - 1] {
                    let word = self.read_word(addr);
                    for i in 0..word.count_ones() as usize { next.push(addr + self.read_delta(addr, i)); }
                }
                next.sort_unstable();
                next.dedup();
                levels.push(next);
            }
            levels
        })
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    /// Distinct physical states per radix level (not logical prefix counts).
    pub fn physical_states_per_level(&self) -> &[usize] { &self.level_node_counts }

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

    /// Crate-internal accessors for `fst_join.rs`'s join compiler/executor
    /// -- not full `LevelAutomaton` walks (see that module's own doc
    /// comment for why), but still native to this struct's own
    /// delta-addressed storage, not `PrefixCursor`'s.
    pub(crate) fn root_addr(&self) -> u64 {
        self.root_virtual_addr
    }

    pub(crate) fn dim_sizes(&self) -> [u64; NUM_DIMS] {
        self.dimension_sizes
    }

    pub(crate) fn level_schedule(&self, level: usize) -> ([u8; NUM_DIMS], [u8; NUM_DIMS], [u8; NUM_DIMS]) {
        (self.level_dim_bits[level], self.level_shifts_cache[level], self.level_child_cell_bits_cache[level])
    }

    /// The full per-level schedule arrays, for callers (like
    /// `automaton_join.rs`) that build a [`crate::level_automaton::
    /// DimRestrictionAutomaton`] directly rather than reading one level
    /// at a time.
    #[allow(clippy::type_complexity)]
    pub(crate) fn full_schedule(&self) -> (&[[u8; NUM_DIMS]], &[[u8; NUM_DIMS]], &[[u8; NUM_DIMS]]) {
        (&self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache)
    }

    pub(crate) fn read_word(&self, addr: u64) -> u64 {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        self.storage.read_u64(block, offset)
    }

    pub(crate) fn read_delta(&self, addr: u64, real_index: usize) -> u64 {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES)
    }

    /// Same contract as [`PathFstDeltaPow2::restricted_descent_visit`] --
    /// plain DFS, one precomputed mask per level -- on this struct's
    /// BFS-ordered storage instead.
    ///
    /// Debugging/comparison only -- prefer
    /// [`Self::restricted_descent_via_composed_automaton`], which takes
    /// the identical fast path for this method's own `Exact`/`Free`-only
    /// case (so there's no speed cost to switching) and also supports
    /// `Range`/`Multi`, which this method panics on. Kept because this
    /// *is* the reference every automaton-driven method on this struct
    /// was validated against.
    pub fn restricted_descent_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        self.visit_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, &mut emit)
    }

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
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            self.visit_rec(child_addr, child_lo, level + 1, level_masks, emit_dim, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent_visit`], sorted,
    /// deduplicated, collected into a `Vec`.
    ///
    /// Debugging/comparison only -- see [`Self::restricted_descent_visit`]'s
    /// own doc comment; prefer [`Self::restricted_descent_via_composed_automaton`].
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

    /// Generic walk half of the `LevelAutomaton` split (see
    /// `level_automaton.rs` and `DESIGN.md`'s "Joins and paths as
    /// automatons" entry) -- structurally identical to `Self::visit_rec`,
    /// just reading `automaton.mask(state, level)` instead of indexing a
    /// flat precomputed array, and carrying `automaton.advance(...)`'s
    /// result forward as `state` instead of nothing. [`Self::
    /// restricted_descent_via_automaton`] below plugs in
    /// `PrecomputedMaskAutomaton` to reproduce today's behavior exactly
    /// through this walk, as the first, lowest-risk check that the split
    /// is sound before anything genuinely state-dependent (a join atom,
    /// a path hop) is built on top of it.
    #[allow(clippy::too_many_arguments)]
    fn visit_automaton_rec<A: crate::level_automaton::LevelAutomaton, T>(
        &self,
        automaton: &A,
        addr: u64,
        lo: [u64; NUM_DIMS],
        level: usize,
        state: A::State,
        emit_dim: usize,
        emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>,
    ) -> std::ops::ControlFlow<T> {
        if !automaton.can_match(&state) {
            return std::ops::ControlFlow::Continue(());
        }
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let mut relevant = word & automaton.mask(&state, level);
        let is_last = level == self.height - 1;

        if is_last {
            while relevant != 0 {
                let qi = relevant.trailing_zeros();
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi as usize, child_cell_bits);
                emit(child_lo[emit_dim] as u32)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        while relevant != 0 {
            let qi = relevant.trailing_zeros();
            relevant &= relevant - 1;
            let next_state = automaton.advance(&state, level, qi);
            if !automaton.can_match(&next_state) {
                continue;
            }
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi as usize, child_cell_bits);
            self.visit_automaton_rec(automaton, child_addr, child_lo, level + 1, next_state, emit_dim, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], walked through the
    /// generic `Self::visit_automaton_rec` with the simplest possible
    /// `LevelAutomaton` (`PrecomputedMaskAutomaton`, wrapping the exact
    /// same `precompute_level_masks` output `restricted_descent_visit`
    /// uses) instead of `visit_rec`'s bespoke recursion -- reproduces
    /// `restricted_descent`'s behavior exactly, validated by
    /// `assert_eq!` against it, as the lowest-risk proof the walk/
    /// automaton split is sound before building anything
    /// state-dependent on top of it.
    ///
    /// Superseded in practice by [`Self::restricted_descent_via_composed_automaton`],
    /// which takes this exact code path internally for `Exact`/`Free`-only
    /// restrictions (so there's no reason to call this one directly
    /// instead) and also handles `Range`/`Multi`. Kept because it's the
    /// validation step this file's own history (`DESIGN.md`) depends on
    /// -- the composed version's fast path was checked against *this*,
    /// not just against `restricted_descent`.
    pub fn restricted_descent_via_automaton(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_via_automaton only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        use crate::level_automaton::LevelAutomaton;
        let level_masks = self.precompute_level_masks(&restrictions);
        let automaton = crate::level_automaton::PrecomputedMaskAutomaton { masks: &level_masks[..self.height] };
        let mut out = Vec::new();
        // Automaton's own `State` happens to be `()` for this particular
        // instance; `clippy::unit_arg` doesn't know that's inherent to a
        // *generic* `start()` call, not a mistake here.
        #[allow(clippy::unit_arg)]
        let _: std::ops::ControlFlow<()> = self.visit_automaton_rec(&automaton, self.root_virtual_addr, [0u64; NUM_DIMS], 0, automaton.start(), emit_dim, &mut |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The preferred general-purpose query entry point on this struct
    /// -- see `DESIGN.md`'s "wired as default" entry. Supports all four
    /// [`DimRestriction`] variants, unlike every other
    /// `PathFstDeltaPow2Bfs` traversal in this file (`restricted_descent`
    /// included), which only ever accepted `Exact`/`Free`.
    ///
    /// `Exact`/`Free`-only restrictions (the common case -- a plain
    /// point or prefix query) take a fast path: one flat
    /// `[u64; MAX_HEIGHT]` mask array, computed once via the same
    /// `precompute_level_masks` mechanism `restricted_descent` itself
    /// uses, zero per-node dispatch. A genuine `Range`/`Multi`
    /// restriction falls back to three independent
    /// `crate::level_automaton::DimRestrictionAutomaton`s (one per
    /// dimension, each computed as if the other two were unconstrained)
    /// composed by intersection (`crate::level_automaton::And`) --
    /// this is the actual proof of "insight 3" from `DESIGN.md`'s
    /// design entry, that mixing restriction kinds can be genuinely
    /// composable rather than a `match` arm per combination.
    ///
    /// The fast path isn't an optimization layered on top of the
    /// composed path -- it's the same lesson `automaton_join.rs` learned
    /// the hard way, applied here before ever recommending this method
    /// as a default: a first version always went through `And`-composed
    /// `DimRestrictionAutomaton`s regardless of restriction kind, and
    /// `perf` on the join side showed that abstraction costs ~1.4-2.1x
    /// over the flat-array path for the `Exact`/`Free`-only case it
    /// never actually needed to pay for. Detecting "no real `Range`/
    /// `Multi` present" and only then reaching for the heavier
    /// composition keeps this method as fast as `restricted_descent`
    /// whenever `restricted_descent` would have worked at all, and
    /// still correct (just slower) when it wouldn't have.
    pub fn restricted_descent_via_composed_automaton(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            if let DimRestriction::Exact(v) = *r
                && v as u64 >= self.dimension_sizes[d]
            {
                return Vec::new();
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let needs_composition = restrictions.iter().any(|r| matches!(r, DimRestriction::Range(..) | DimRestriction::Multi(..)));
        let mut out = Vec::new();
        if needs_composition {
            use crate::level_automaton::{And, DimRestrictionAutomaton, LevelAutomaton};
            let dim0 = DimRestrictionAutomaton::new(0, restrictions[0], &self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache);
            let dim1 = DimRestrictionAutomaton::new(1, restrictions[1], &self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache);
            let dim2 = DimRestrictionAutomaton::new(2, restrictions[2], &self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache);
            let automaton = And(And(dim0, dim1), dim2);
            let state = automaton.start();
            let _: std::ops::ControlFlow<()> = self.visit_automaton_rec(&automaton, self.root_virtual_addr, [0u64; NUM_DIMS], 0, state, emit_dim, &mut |v| {
                out.push(v);
                std::ops::ControlFlow::Continue(())
            });
        } else {
            use crate::level_automaton::{LevelAutomaton, PrecomputedMaskAutomaton};
            let level_masks = self.precompute_level_masks(&restrictions);
            let automaton = PrecomputedMaskAutomaton { masks: &level_masks[..self.height] };
            #[allow(clippy::unit_arg)]
            let _: std::ops::ControlFlow<()> = self.visit_automaton_rec(&automaton, self.root_virtual_addr, [0u64; NUM_DIMS], 0, automaton.start(), emit_dim, &mut |v| {
                out.push(v);
                std::ops::ControlFlow::Continue(())
            });
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same contract and fixed-window batching as
    /// [`PathFstDeltaPow2::restricted_descent_bfs_visit`] -- the
    /// algorithm this whole struct exists to pair with a layout for --
    /// on this struct's BFS-ordered storage instead. Should see a bigger
    /// win here than on `PathFstDeltaPow2`'s post-order storage: every
    /// node a window's discard-then-real-read pass touches is now also
    /// physically contiguous, not merely logically grouped.
    pub fn restricted_descent_bfs_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_bfs_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let root = [(self.root_virtual_addr, [0u64; NUM_DIMS])];
        self.bfs_window_visit(&root, 0, &level_masks, emit_dim, &mut emit)
    }

    fn bfs_window_visit<T>(&self, nodes: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        if nodes.is_empty() {
            return std::ops::ControlFlow::Continue(());
        }
        let is_last = level == self.height - 1;
        let mask = level_masks[level];
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        for chunk in nodes.chunks(BFS_WINDOW) {
            for &(addr, _) in chunk {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                std::hint::black_box(self.storage.read_u64(block, offset));
            }

            let mut next_buf = [(0u64, [0u64; NUM_DIMS]); BFS_WINDOW];
            let mut next_len = 0usize;

            for &(addr, lo) in chunk {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                let word = self.storage.read_u64(block, offset);
                let mut relevant = word & mask;

                if is_last {
                    while relevant != 0 {
                        let qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                        emit(child_lo[emit_dim] as u32)?;
                    }
                    continue;
                }

                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                    let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                    let child_addr = addr + delta;
                    let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);

                    next_buf[next_len] = (child_addr, child_lo);
                    next_len += 1;
                    if next_len == BFS_WINDOW {
                        self.bfs_window_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
                        next_len = 0;
                    }
                }
            }

            if next_len > 0 {
                self.bfs_window_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
            }
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_bfs_visit`] instead.
    pub fn restricted_descent_bfs(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_bfs_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Prototype: same window-batched BFS as [`Self::restricted_descent_bfs_visit`],
    /// but the `word & level_masks[level]` step is pulled out of the
    /// per-node decode loop into its own flat pass over a plain array,
    /// instead of being computed inline, one node at a time, interleaved
    /// with reads/branches/recursive pushes. The reachability question
    /// ("which nodes are even in this window") is unchanged -- it's
    /// still established by the normal delta-walk from the real root,
    /// same as every other traversal in this file; nothing here tries
    /// to mask a whole level's nodes without first knowing they're
    /// actually reachable, which the BFS-ordered layout alone can't
    /// tell you (a node's level doesn't say whether this particular
    /// query's root path reaches it).
    ///
    /// No explicit SIMD intrinsics -- `#![forbid(unsafe_code)]` rules
    /// out `std::arch`'s `_mm256_and_si256` and friends (those need
    /// `unsafe`), and portable `std::simd` is nightly-only
    /// (`#![feature(portable_simd)]`). What this does instead is give
    /// LLVM's auto-vectorizer the cleanest possible shot: a fixed-size,
    /// branch-free, elementwise `masked[i] = words[i] & mask` loop over
    /// plain arrays, with every memory read and every branch moved
    /// *out* of that loop and into separate passes before/after it.
    /// Whether this actually compiles down to a real vector AND
    /// (`pand`/`vpand`) or not is a question for `objdump`, not an
    /// assumption -- see `DESIGN.md` for what was actually found.
    pub fn restricted_descent_bfs_masked_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_bfs_masked_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let root = [(self.root_virtual_addr, [0u64; NUM_DIMS])];
        self.bfs_window_masked_visit(&root, 0, &level_masks, emit_dim, &mut emit)
    }

    fn bfs_window_masked_visit<T>(&self, nodes: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        if nodes.is_empty() {
            return std::ops::ControlFlow::Continue(());
        }
        let is_last = level == self.height - 1;
        let mask = level_masks[level];
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        for chunk in nodes.chunks(BFS_WINDOW) {
            for &(addr, _) in chunk {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                std::hint::black_box(self.storage.read_u64(block, offset));
            }

            // Phase A: every word in the chunk, read into a flat array --
            // no masking, no branching, just the reads.
            let mut words = [0u64; BFS_WINDOW];
            for (i, &(addr, _)) in chunk.iter().enumerate() {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                words[i] = self.storage.read_u64(block, offset);
            }

            // Phase B: the part this method exists to isolate -- one
            // flat, branch-free elementwise AND over plain arrays.
            let mut masked = [0u64; BFS_WINDOW];
            for i in 0..chunk.len() {
                masked[i] = words[i] & mask;
            }

            // Phase C: decode, same logic as `bfs_window_visit`, just
            // reading `words`/`masked` instead of recomputing them.
            let mut next_buf = [(0u64, [0u64; NUM_DIMS]); BFS_WINDOW];
            let mut next_len = 0usize;

            for (i, &(addr, lo)) in chunk.iter().enumerate() {
                let mut relevant = masked[i];
                if relevant == 0 {
                    continue;
                }
                let word = words[i];

                if is_last {
                    while relevant != 0 {
                        let qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                        emit(child_lo[emit_dim] as u32)?;
                    }
                    continue;
                }

                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                    let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                    let child_addr = addr + delta;
                    let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);

                    next_buf[next_len] = (child_addr, child_lo);
                    next_len += 1;
                    if next_len == BFS_WINDOW {
                        self.bfs_window_masked_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
                        next_len = 0;
                    }
                }
            }

            if next_len > 0 {
                self.bfs_window_masked_visit(&next_buf[..next_len], level + 1, level_masks, emit_dim, emit)?;
            }
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_bfs_masked_visit`] instead.
    pub fn restricted_descent_bfs_masked(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_bfs_masked_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Prototype: the "mask tree + sweep" idea -- for every tree level,
    /// read *every distinct node's* word (not just the ones this
    /// query's root actually reaches) and AND it with that level's one
    /// precomputed mask, in a single flat pass over the level's address
    /// list, before any real traversal starts. Then walk the real
    /// parent-to-child links exactly like [`Self::restricted_descent_visit`],
    /// but reading the precomputed `(word, word & mask)` pair instead of
    /// recomputing the AND inline. Modeled directly on Hyperscan/Teddy's
    /// and simdjson's "cheap bulk filter pass over the whole input, then
    /// a slower confirm pass on survivors" shape -- see
    /// `FST_LITERATURE_REVIEW.md`.
    ///
    /// Finding each accepted node during the walk needs a position
    /// within its level's `Vec`, which is found by binary search over
    /// `level_addrs[level]` (sound because addresses are assigned in
    /// strictly increasing order during `build`'s Phase 2, so each
    /// level's address list is already sorted).
    ///
    /// See `DESIGN.md` for why this loses badly in this project's
    /// domains specifically: the DAG's dedup means a level's *distinct*
    /// node count can be orders of magnitude larger than the number of
    /// nodes any single query actually reaches (measured directly, not
    /// assumed), because the same shared node is stored once but
    /// reachable from many different root bindings. Teddy/simdjson sweep
    /// every byte of their input because every byte genuinely needs
    /// classifying -- there is no equivalent "most of this is already
    /// known-unreachable before you've even read it" concept in their
    /// domain the way there is here.
    pub fn restricted_descent_sweep_visit<T>(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, mut emit: impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return std::ops::ControlFlow::Continue(()),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_sweep_visit only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return std::ops::ControlFlow::Continue(());
        }
        let level_masks = self.precompute_level_masks(&restrictions);

        let filtered: Vec<Vec<(u64, u64)>> = (0..self.height)
            .map(|level| {
                let mask = level_masks[level];
                self.level_addresses()[level]
                    .iter()
                    .map(|&addr| {
                        let block = (addr >> self.capacity_bits) as usize;
                        let offset = (addr & self.capacity_mask) as usize;
                        let word = self.storage.read_u64(block, offset);
                        (word, word & mask)
                    })
                    .collect()
            })
            .collect();

        self.sweep_walk_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &filtered, emit_dim, &mut emit)
    }

    fn sweep_walk_rec<T>(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, filtered: &[Vec<(u64, u64)>], emit_dim: usize, emit: &mut impl FnMut(u32) -> std::ops::ControlFlow<T>) -> std::ops::ControlFlow<T> {
        let idx = self
            .level_addresses()[level]
            .binary_search(&addr)
            .expect("every real child address must exist in its own level's address list");
        let (word, mut relevant) = filtered[level][idx];
        let is_last = level == self.height - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                emit(child_lo[emit_dim] as u32)?;
            }
            return std::ops::ControlFlow::Continue(());
        }

        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            self.sweep_walk_rec(child_addr, child_lo, level + 1, filtered, emit_dim, emit)?;
        }
        std::ops::ControlFlow::Continue(())
    }

    /// Same contract as [`Self::restricted_descent`], built on
    /// [`Self::restricted_descent_sweep_visit`] instead.
    pub fn restricted_descent_sweep(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.restricted_descent_sweep_visit(restrictions, emit_dim, |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Prototype only -- see `DESIGN.md`. Inverts the usual shape of
    /// every other traversal in this file: instead of descending from
    /// the root and pruning as it goes, this sweeps *only* the leaf
    /// level's existence words in one branch-free elementwise AND pass
    /// (same "give LLVM the cleanest shot at auto-vectorizing, verify
    /// with `objdump`, don't assume" convention as
    /// [`Self::restricted_descent_bfs_masked_visit`] -- `#![forbid(unsafe_code)]`
    /// rules out real intrinsics), then, for every leaf with at least
    /// one surviving bit, walks newly-built parent back-pointers *up*
    /// toward the root to find out which triple(s) it actually belongs
    /// to, instead of finding it by descending down to it.
    ///
    /// Why this needs back-pointers at all, and why that's a real cost:
    /// this structure is a DAG, not a plain trie -- [`Self::build`]
    /// dedups identical subtrees, so one leaf (or any node) can be the
    /// real child of more than one parent. A forward delta only ever
    /// points down, so recovering "what paths lead here" needs an
    /// explicit reverse index built freshly by [`Self::build_parent_index`]
    /// for every call to this method -- real, uncached work this
    /// prototype does *not* try to make free, unlike `level_addrs`
    /// (precomputed once, in [`Self::build`] itself, and reused by
    /// every sweep-shaped method already in this file).
    ///
    /// The DAG sharing cuts both ways here: a leaf with multiple
    /// parents can have multiple root-to-leaf paths, and *not every one
    /// of them need satisfy this query's `Exact` restrictions* -- the
    /// leaf-level mask only ever checked the free dimensions' bits, not
    /// the pinned ones, so every reconstructed path gets checked against
    /// the full restriction set before being accepted, exactly like
    /// re-deriving the full triple and checking it from scratch. And
    /// because a shared ancestor can be reached by exponentially many
    /// paths in the worst case, `Self::paths_to_root` memoizes per
    /// address for the duration of one call -- without that, this is
    /// not just slow, it can fail to terminate in practical time on a
    /// real domain.
    pub fn restricted_descent_leaf_sweep(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        let parent_index = self.build_parent_index();
        self.restricted_descent_leaf_sweep_with_index(restrictions, emit_dim, &parent_index)
    }

    /// Same as [`Self::restricted_descent_leaf_sweep`], but takes an
    /// already-built [`Self::build_parent_index`] instead of rebuilding
    /// one -- lets a caller amortize that cost across many queries
    /// (what a real implementation would do) to measure the leaf-sweep
    /// idea's own cost in isolation from the index-rebuild cost this
    /// prototype doesn't otherwise try to hide.
    pub fn restricted_descent_leaf_sweep_with_index(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, parent_index: &HashMap<u64, Vec<(u64, u8)>>) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_leaf_sweep only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let leaf_level = self.height - 1;
        let leaf_mask = level_masks[leaf_level];

        // Phase A: every leaf word, read into a flat array -- no
        // masking, no branching yet.
        let leaf_addrs = &self.level_addresses()[leaf_level];
        let mut words = vec![0u64; leaf_addrs.len()];
        for (i, &addr) in leaf_addrs.iter().enumerate() {
            let block = (addr >> self.capacity_bits) as usize;
            let offset = (addr & self.capacity_mask) as usize;
            words[i] = self.storage.read_u64(block, offset);
        }

        // Phase B: one flat, branch-free elementwise AND over plain
        // arrays -- the part this method exists to isolate.
        let mut masked = vec![0u64; leaf_addrs.len()];
        for i in 0..leaf_addrs.len() {
            masked[i] = words[i] & leaf_mask;
        }

        let dim_bits_by_level = &self.level_dim_bits;
        let shifts_by_level = &self.level_shifts_cache;
        let child_cell_bits_by_level = &self.level_child_cell_bits_cache;

        let mut path_cache: HashMap<u64, Vec<Vec<u8>>> = HashMap::new();
        let mut out = Vec::new();
        for (i, &addr) in leaf_addrs.iter().enumerate() {
            let mut relevant = masked[i];
            if relevant == 0 {
                continue;
            }
            let chains = self.paths_to_root(addr, parent_index, &mut path_cache);
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                for chain in &chains {
                    let mut lo = [0u64; NUM_DIMS];
                    for (level, &step_qi) in chain.iter().enumerate() {
                        lo = advance_lo(lo, dim_bits_by_level[level], shifts_by_level[level], step_qi as usize, child_cell_bits_by_level[level]);
                    }
                    let final_lo = advance_lo(lo, dim_bits_by_level[leaf_level], shifts_by_level[leaf_level], qi, child_cell_bits_by_level[leaf_level]);
                    let matches_restrictions = restrictions.iter().enumerate().all(|(d, r)| match *r {
                        DimRestriction::Exact(v) => final_lo[d] == v as u64,
                        DimRestriction::Free => true,
                        DimRestriction::Range(..) | DimRestriction::Multi(..) => unreachable!("validated above"),
                    });
                    if matches_restrictions {
                        out.push(final_lo[emit_dim] as u32);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every real forward edge in the whole structure, inverted: for
    /// every node below the root, the list of `(parent_addr, qi)` pairs
    /// describing every real parent that points to it and which of the
    /// parent's bits was used to do so. Unlike every forward structure
    /// in this file (`level_addrs` included), this is *not* built in
    /// [`Self::build`] and cached as part of the struct -- a caller gets
    /// one by calling this directly, once, and may reuse it across many
    /// [`Self::restricted_descent_leaf_sweep_with_index`] calls to
    /// amortize the cost; [`Self::restricted_descent_leaf_sweep`]
    /// itself rebuilds one fresh on every call, which is real,
    /// unamortized work this prototype does not try to hide.
    ///
    /// Walks every non-leaf level's *unrestricted* existence word (every
    /// real child, not filtered by any query's mask) and decodes each
    /// one's address via the same delta read every other traversal in
    /// this file uses, so this is one full, unfiltered structural pass
    /// over every node that isn't a leaf.
    /// Diagnostic only, for measuring [`Self::restricted_descent_leaf_sweep`]
    /// honestly: the number of distinct leaf-level nodes `build_parent_index`
    /// and the leaf sweep itself have to touch, regardless of what any
    /// one query actually needs -- the same "distinct nodes stored once,
    /// reachable from many different root bindings" count that made the
    /// whole-level sweep lose badly (see `DESIGN.md`).
    pub fn leaf_level_node_count(&self) -> usize {
        if self.height == 0 {
            0
        } else {
            self.level_node_counts[self.height - 1]
        }
    }

    pub fn build_parent_index(&self) -> HashMap<u64, Vec<(u64, u8)>> {
        let mut parents: HashMap<u64, Vec<(u64, u8)>> = HashMap::new();
        if self.height == 0 {
            return parents;
        }
        for level in 0..self.height - 1 {
            for &addr in &self.level_addresses()[level] {
                let block = (addr >> self.capacity_bits) as usize;
                let offset = (addr & self.capacity_mask) as usize;
                let word = self.storage.read_u64(block, offset);
                let mut remaining = word;
                while remaining != 0 {
                    let qi = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                    let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                    let child_addr = addr + delta;
                    parents.entry(child_addr).or_default().push((addr, qi as u8));
                }
            }
        }
        parents
    }

    /// Every distinct sequence of transitions (`qi` per level, root
    /// first) that reaches `addr` from the real root, found by walking
    /// [`Self::build_parent_index`]'s reverse edges upward. The root
    /// itself has exactly one path: the empty sequence. Memoized in
    /// `cache` for the lifetime of one [`Self::restricted_descent_leaf_sweep`]
    /// call -- a shared ancestor reached from many different leaves (or
    /// many different paths to the *same* leaf) would otherwise redo
    /// the same upward walk repeatedly, and in a sufficiently shared DAG
    /// the raw path *count* itself, not just the per-node work, can
    /// still grow large even with memoization -- a real scaling
    /// question for this prototype, not assumed away.
    fn paths_to_root(&self, addr: u64, parent_index: &HashMap<u64, Vec<(u64, u8)>>, cache: &mut HashMap<u64, Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
        if addr == self.root_virtual_addr {
            return vec![Vec::new()];
        }
        if let Some(cached) = cache.get(&addr) {
            return cached.clone();
        }
        let mut result = Vec::new();
        if let Some(parents) = parent_index.get(&addr) {
            for &(parent_addr, qi) in parents {
                for parent_chain in self.paths_to_root(parent_addr, parent_index, cache) {
                    let mut chain = parent_chain;
                    chain.push(qi);
                    result.push(chain);
                }
            }
        }
        cache.insert(addr, result.clone());
        result
    }
}

/// Prototype: a maximally naive rayon parallelization of the plain DFS
/// `visit_rec` pseudocode (see the conversation/`DESIGN.md`) -- every
/// recursive call is a candidate for running on a different worker
/// thread, with no depth or fanout cutoff at all. This project already
/// has a measured cautionary tale about exactly this shape of naivety,
/// on a different traversal (`Heap`'s query path, `DESIGN.md`'s "Rayon
/// fan-out depth sweep" entry): rayon's per-task dispatch overhead is
/// roughly fixed, so fanning out at every level, all the way down to
/// near-leaf nodes whose own work is a handful of instructions, is
/// expected to collapse once dispatch cost dominates the real work
/// being split. Built and measured anyway, deliberately uncapped, to
/// see exactly where that happens on *this* traversal rather than
/// assume the old number transfers -- see `DESIGN.md`.
///
/// Returns a `Vec<u32>` directly instead of streaming through a
/// `ControlFlow`-based `emit` callback like every other traversal in
/// this file: early-exit (`ControlFlow::Break`) doesn't have an obvious
/// meaning once multiple branches are running concurrently on different
/// threads, so this prototype sidesteps that question entirely rather
/// than guess at an answer.
#[cfg(feature = "parallel")]
impl<S: BlockStorage + Sync> PackedFst<S> {
    pub fn restricted_descent_parallel(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let mut out = self.visit_parallel_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn visit_parallel_rec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize) -> Vec<u32> {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut out = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                out.push(child_lo[emit_dim] as u32);
            }
            return out;
        }

        // Decoding each accepted transition's child address is cheap
        // (one popcount, one small `read_var`) -- stays sequential. Only
        // the recursive descent into each resulting child is handed to
        // rayon, in `parallel_collect`.
        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        self.parallel_collect(&children, level + 1, level_masks, emit_dim)
    }

    /// Unconditional binary fan-out -- every split becomes a
    /// `rayon::join` call, all the way down, no size/depth cutoff (the
    /// "naive" part). `rayon::join` is still just a *candidate* for
    /// running on another worker thread (it falls back to sequential
    /// execution under its own work-stealing logic when the pool is
    /// busy), so this can't explode into one OS thread per node the way
    /// `std::thread::spawn` per call would -- but the *coordination*
    /// overhead (two closures, a join point) is paid at every split
    /// regardless of whether a second thread ever actually picks it up.
    fn parallel_collect(&self, children: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize) -> Vec<u32> {
        match children.len() {
            0 => Vec::new(),
            1 => {
                let (addr, lo) = children[0];
                self.visit_parallel_rec(addr, lo, level, level_masks, emit_dim)
            }
            n => {
                let mid = n / 2;
                let (left, right) = children.split_at(mid);
                let (mut l, r) = rayon::join(
                    || self.parallel_collect(left, level, level_masks, emit_dim),
                    || self.parallel_collect(right, level, level_masks, emit_dim),
                );
                l.extend(r);
                l
            }
        }
    }

    /// Same idea as [`Self::restricted_descent_parallel`], but with the
    /// depth cutoff that entry's own doc comment flagged as the obvious
    /// next step: fan out with `rayon::join` for only the first
    /// `parallel_depth` tree levels below the root, then fall back to
    /// the plain sequential `visit_rec` (via
    /// `Self::visit_sequential_to_vec`) for the rest of each subtree.
    /// `parallel_depth` counts tree levels, not bisection splits --
    /// bisecting one level's sibling list in half doesn't itself consume
    /// budget, only actually descending to the next level does -- so
    /// this is directly comparable to `Heap`'s own `parallel_depth`
    /// parameter from the "Rayon fan-out depth sweep" entry in
    /// `DESIGN.md`, not a different kind of knob with the same name.
    pub fn restricted_descent_parallel_cutoff(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, parallel_depth: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_cutoff only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let mut out = self.visit_parallel_cutoff_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, parallel_depth);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same contract as `Self::visit_rec`, just collecting into an
    /// owned `Vec` instead of streaming through a `ControlFlow` `emit`
    /// callback -- reuses the already-correct sequential traversal
    /// rather than re-deriving it, for the part of a subtree
    /// [`Self::restricted_descent_parallel_cutoff`] decides isn't worth
    /// parallelizing any further.
    fn visit_sequential_to_vec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let _: std::ops::ControlFlow<()> = self.visit_rec(addr, lo, level, level_masks, emit_dim, &mut |v| {
            out.push(v);
            std::ops::ControlFlow::Continue(())
        });
        out
    }

    fn visit_parallel_cutoff_rec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, remaining_depth: usize) -> Vec<u32> {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut out = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                out.push(child_lo[emit_dim] as u32);
            }
            return out;
        }

        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        self.parallel_collect_cutoff(&children, level + 1, level_masks, emit_dim, remaining_depth)
    }

    /// Same unconditional-binary-fan-out shape as [`Self::parallel_collect`],
    /// except once `remaining_depth` reaches zero the rest of every
    /// child's subtree finishes through `Self::visit_sequential_to_vec`
    /// instead of recursing into more `rayon::join` calls. `remaining_depth`
    /// is only decremented when a child actually descends to the next
    /// tree level (the `children.len() == 1` arm) -- bisecting a level's
    /// sibling list in half, the `n => ..` arm, doesn't consume budget on
    /// its own.
    fn parallel_collect_cutoff(&self, children: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize, remaining_depth: usize) -> Vec<u32> {
        if remaining_depth == 0 {
            return children.iter().flat_map(|&(addr, lo)| self.visit_sequential_to_vec(addr, lo, level, level_masks, emit_dim)).collect();
        }
        match children.len() {
            0 => Vec::new(),
            1 => {
                let (addr, lo) = children[0];
                self.visit_parallel_cutoff_rec(addr, lo, level, level_masks, emit_dim, remaining_depth - 1)
            }
            n => {
                let mid = n / 2;
                let (left, right) = children.split_at(mid);
                let (mut l, r) = rayon::join(
                    || self.parallel_collect_cutoff(left, level, level_masks, emit_dim, remaining_depth),
                    || self.parallel_collect_cutoff(right, level, level_masks, emit_dim, remaining_depth),
                );
                l.extend(r);
                l
            }
        }
    }

    /// A different cutoff signal than [`Self::restricted_descent_parallel_cutoff`]'s
    /// depth: instead of a fixed, global, once-spent budget, gate every
    /// fan-out decision on a purely local property -- how many
    /// explorable subtrees (real children) are available to split
    /// *right here* -- and re-evaluate it fresh at every node, all the
    /// way down. `rayon::join` is only used when a node has *more than*
    /// `min_parallel_size` real children; at or below that, every one of
    /// those children is processed by a plain sequential loop (not one
    /// fork per child the way the depth-cutoff version's single-child
    /// case works) -- but each of *those* children's own fan-out is
    /// still re-checked against `min_parallel_size` when its own turn
    /// comes, so a node deep in an otherwise-sequential region with an
    /// unusually wide fan-out of its own still gets to parallelize.
    /// Depth-cutoff can't do that: once its budget hits zero, everything
    /// below is sequential regardless of how wide any single node's own
    /// fan-out turns out to be.
    pub fn restricted_descent_parallel_size_cutoff(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, min_parallel_size: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_size_cutoff only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let mut out = self.visit_parallel_size_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, min_parallel_size);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn visit_parallel_size_rec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, min_parallel_size: usize) -> Vec<u32> {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut out = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                out.push(child_lo[emit_dim] as u32);
            }
            return out;
        }

        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        self.parallel_collect_by_size(&children, level + 1, level_masks, emit_dim, min_parallel_size)
    }

    fn parallel_collect_by_size(&self, children: &[(u64, [u64; NUM_DIMS])], level: usize, level_masks: &[u64], emit_dim: usize, min_parallel_size: usize) -> Vec<u32> {
        if children.len() <= min_parallel_size {
            return children.iter().flat_map(|&(addr, lo)| self.visit_parallel_size_rec(addr, lo, level, level_masks, emit_dim, min_parallel_size)).collect();
        }
        let mid = children.len() / 2;
        let (left, right) = children.split_at(mid);
        let (mut l, r) = rayon::join(
            || self.parallel_collect_by_size(left, level, level_masks, emit_dim, min_parallel_size),
            || self.parallel_collect_by_size(right, level, level_masks, emit_dim, min_parallel_size),
        );
        l.extend(r);
        l
    }

    /// Answers a question raised about every `parallel*` variant above:
    /// what happens to a branch that hits a dead end? Today, nothing
    /// expensive -- but it still pays a real, avoidable cost. Every
    /// `rayon::join` call is a value-returning rendezvous: both sides
    /// must finish and hand back a `Vec<u32>` before the caller can
    /// `extend` them together, *even when one side is a dead end that
    /// never had anything to contribute*. That's a `Vec::new()`
    /// allocation plus an empty `extend` call, paid on every single
    /// dead-end branch, purely because `join`'s shape requires a result
    /// to merge regardless of whether there's anything in it.
    ///
    /// A thread genuinely can't be "dropped" without being waited on at
    /// all -- the top-level call still needs every branch's contribution
    /// (empty or not) to have actually finished before it can return a
    /// complete answer, and there's no way to know a branch is a dead
    /// end without running it. But the *merge* doesn't have to go
    /// through a return value. `rayon::scope` + `Scope::spawn` replaces
    /// `join`'s "both sides return a `Vec`, caller merges them" shape
    /// with "every branch writes *directly* into one shared sink, if it
    /// has anything to write at all, and the only synchronization point
    /// is the single outer `scope` call waiting for every spawned task
    /// (however deeply nested) to finish." A dead end -- `relevant == 0`
    /// at any level, including before any children are even decoded --
    /// now costs exactly one function call that returns immediately:
    /// no allocation, no lock, no merge, nothing to join back. Only a
    /// branch that actually reaches accepted leaves ever touches the
    /// shared `Mutex`, and only once (results are gathered into a local
    /// `Vec` first, then pushed in one lock acquisition), not once per
    /// emitted value.
    pub fn restricted_descent_parallel_scoped(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, parallel_depth: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_scoped only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let sink: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        rayon::scope(|scope| {
            self.visit_scoped_rec(scope, &sink, self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, parallel_depth);
        });
        let mut out = sink.into_inner().expect("sink mutex never panics while locked");
        out.sort_unstable();
        out.dedup();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_scoped_rec<'a>(&'a self, scope: &rayon::Scope<'a>, sink: &'a std::sync::Mutex<Vec<u32>>, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &'a [u64], emit_dim: usize, remaining_depth: usize) {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        // Dead end: nothing matched at this node at all. No allocation,
        // no lock, no spawn -- just return. This is the case the
        // question was actually about.
        if relevant == 0 {
            return;
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut local = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                local.push(child_lo[emit_dim] as u32);
            }
            sink.lock().expect("sink mutex never panics while locked").extend(local);
            return;
        }

        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        if remaining_depth == 0 {
            // Still worth gathering sequentially before touching the
            // lock at all: a sequential dead end anywhere in this
            // subtree costs nothing extra, same as the parallel case.
            let mut local = Vec::new();
            for &(a, l) in &children {
                local.extend(self.visit_sequential_to_vec(a, l, level + 1, level_masks, emit_dim));
            }
            if !local.is_empty() {
                sink.lock().expect("sink mutex never panics while locked").extend(local);
            }
            return;
        }

        for (child_addr, child_lo) in children {
            scope.spawn(move |s| {
                self.visit_scoped_rec(s, sink, child_addr, child_lo, level + 1, level_masks, emit_dim, remaining_depth - 1);
            });
        }
    }

    /// Same mechanism as [`Self::restricted_descent_parallel_scoped`],
    /// plus one more lever raised directly: `rayon::scope`'s own docs
    /// say every `Scope::spawn` call heap-allocates its task, unlike
    /// `rayon::join`'s stack-only two-way split -- so spawning one task
    /// *per child*, unconditionally, pays that allocation once per
    /// accepted transition even when a node's fan-out is wide. This
    /// variant chunks a node's children into at most `max_branch` groups
    /// (not groups of size `max_branch` -- the cap is on how many tasks
    /// get spawned *from one node*, matching "branch up to `x` ways,"
    /// not on how many children end up in each one) and spawns one task
    /// per chunk; each task then runs its own chunk's children one
    /// after another, recursing normally (still subject to the same
    /// `remaining_depth` budget for any *further* fan-out below it).
    pub fn restricted_descent_parallel_scoped_chunked(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, parallel_depth: usize, max_branch: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_scoped_chunked only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let max_branch = max_branch.max(1);
        let level_masks = self.precompute_level_masks(&restrictions);
        let sink: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        rayon::scope(|scope| {
            self.visit_scoped_chunked_rec(scope, &sink, self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, parallel_depth, max_branch);
        });
        let mut out = sink.into_inner().expect("sink mutex never panics while locked");
        out.sort_unstable();
        out.dedup();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_scoped_chunked_rec<'a>(&'a self, scope: &rayon::Scope<'a>, sink: &'a std::sync::Mutex<Vec<u32>>, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &'a [u64], emit_dim: usize, remaining_depth: usize, max_branch: usize) {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        if relevant == 0 {
            return;
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut local = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                local.push(child_lo[emit_dim] as u32);
            }
            sink.lock().expect("sink mutex never panics while locked").extend(local);
            return;
        }

        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        if remaining_depth == 0 {
            let mut local = Vec::new();
            for &(a, l) in &children {
                local.extend(self.visit_sequential_to_vec(a, l, level + 1, level_masks, emit_dim));
            }
            if !local.is_empty() {
                sink.lock().expect("sink mutex never panics while locked").extend(local);
            }
            return;
        }

        let num_chunks = max_branch.min(children.len());
        let chunk_size = children.len().div_ceil(num_chunks);
        for chunk in children.chunks(chunk_size) {
            let chunk = chunk.to_vec();
            scope.spawn(move |s| {
                for (child_addr, child_lo) in chunk {
                    self.visit_scoped_chunked_rec(s, sink, child_addr, child_lo, level + 1, level_masks, emit_dim, remaining_depth - 1, max_branch);
                }
            });
        }
    }

    /// Lazy binary splitting (Tzannes, Caragea, Barua & Vishkin, 2010),
    /// adapted to this traversal via `rayon::join_context` instead of
    /// hand-tuned the way every other `parallel*` variant above is.
    /// Every other variant decides *in advance*, from a fixed constant,
    /// whether a given node is worth splitting -- the LBS paper's named
    /// weakness of eager splitting: a threshold "over-fits... to a
    /// particular dataset, platform, and calling context." LBS's idea:
    /// don't decide up front -- split in response to *observed* demand,
    /// via `rayon::join_context`'s `FnContext::migrated()` (whether a
    /// closure actually got picked up by a different worker thread),
    /// the same signal rayon's own parallel-iterator bridge uses
    /// internally to decide whether to keep splitting a range.
    ///
    /// Two real bugs found and fixed while measuring this, not just
    /// tuning it (see `DESIGN.md` for the numbers both times):
    ///
    /// 1. A node with exactly one live child isn't evidence *against*
    ///    splitting -- `one-Exact`-shaped queries spend their first
    ///    several levels on a single live path before fan-out actually
    ///    appears. The first version treated "nothing to split here" the
    ///    same as "not migrated," permanently abandoning any chance of
    ///    splitting the instant it passed through that corridor -- often
    ///    at level 1, well before the real fan-out region. Fixed by
    ///    recursing straight through a single-child node, carrying
    ///    `migrated` forward unchanged instead of consuming it.
    /// 2. Migration *alone*, with no floor, never converges: measured
    ///    directly (temporary counters, since removed), this traversal
    ///    sees `rayon::join_context` succeed at migrating roughly 75-80%
    ///    of attempts on this machine's 12 cores -- there's simply
    ///    enough idle capacity around that the adaptive signal keeps
    ///    saying "yes, split more" almost everywhere, reproducing the
    ///    exact `CountLatch`-dominated collapse the fixed-depth cutoff
    ///    hit at `parallel_depth=4`, just reached adaptively instead of
    ///    by a bad constant. This is why rayon's own internal splitter
    ///    (`plumbing::bridge_producer_consumer`) combines migration
    ///    status *with* a length-based floor, not migration alone --
    ///    `max_depth_from_root` plays that role here: splitting still
    ///    requires both migration *and* being within the first
    ///    `max_depth_from_root` tree levels.
    ///
    /// The root call is seeded as already migrated -- otherwise nothing
    /// would ever get offered to steal in the first place.
    pub fn restricted_descent_parallel_lazy(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, max_depth_from_root: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_lazy only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let sink: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        // Seeded as `migrated = true`: the one unconditional split
        // attempt that gets the whole thing started (see doc comment).
        self.visit_lazy_rec(&sink, self.root_virtual_addr, [0u64; NUM_DIMS], 0, &level_masks, emit_dim, true, max_depth_from_root);
        let mut out = sink.into_inner().expect("sink mutex never panics while locked");
        out.sort_unstable();
        out.dedup();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_lazy_rec(&self, sink: &std::sync::Mutex<Vec<u32>>, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, migrated: bool, max_depth_from_root: usize) {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        if relevant == 0 {
            return;
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut local = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                local.push(child_lo[emit_dim] as u32);
            }
            sink.lock().expect("sink mutex never panics while locked").extend(local);
            return;
        }

        let mut children: Vec<(u64, [u64; NUM_DIMS])> = Vec::with_capacity(relevant.count_ones() as usize);
        while relevant != 0 {
            let qi = relevant.trailing_zeros() as usize;
            relevant &= relevant - 1;
            let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
            let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
            let child_addr = addr + delta;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            children.push((child_addr, child_lo));
        }

        // Exactly one live child: there's nothing to split *here*, but
        // that's not evidence against splitting -- `one-Exact`-shaped
        // queries spend their first several levels on a single live
        // path (the pinned dimension's bits haven't exhausted yet)
        // before fan-out actually appears. Recurse straight through,
        // carrying `migrated` forward unchanged, so the real
        // split-or-not decision gets asked again once there's an
        // actual choice to make, instead of being permanently decided
        // by a corridor that had no choice at all.
        if children.len() == 1 {
            let (addr, lo) = children[0];
            self.visit_lazy_rec(sink, addr, lo, level + 1, level_masks, emit_dim, migrated, max_depth_from_root);
            return;
        }

        // Migration alone turned out not to be enough (measured, not
        // assumed -- see `DESIGN.md`): with 12 idle cores and a workload
        // this size, migration succeeds ~79% of the time regardless of
        // tree depth, so a pure migration gate never actually converges
        // to a small number of splits -- it just keeps splitting almost
        // everywhere, landing in the exact same `CountLatch`-dominated
        // collapse as an unbounded fixed-depth cutoff, just reached
        // adaptively instead of by a bad constant. This is exactly why
        // rayon's own internal splitter (`plumbing::bridge_producer_consumer`,
        // used by `par_iter`) combines migration status *with* a
        // length-based floor, not migration alone -- so this does too:
        // require that we're still within the first `max_depth_from_root`
        // tree levels *and* migration, not either one on its own.
        if !migrated || level >= max_depth_from_root {
            let mut local = Vec::new();
            for &(a, l) in &children {
                local.extend(self.visit_sequential_to_vec(a, l, level + 1, level_masks, emit_dim));
            }
            if !local.is_empty() {
                sink.lock().expect("sink mutex never panics while locked").extend(local);
            }
            return;
        }

        // This call *was* migrated -- an idle thread actually reached
        // out for this exact piece of work a moment ago, real evidence
        // there's spare capacity nearby right now. Worth offering more.
        //
        // `join_context`, unlike `Scope::spawn`, is synchronous -- it
        // blocks until both closures finish, so `children` is
        // guaranteed to outlive the call. No need to own/clone `left`
        // and `right` into fresh `Vec`s the way `Scope::spawn` requires;
        // borrowing the slices directly (same pattern every `join`-based
        // variant above already uses) avoids two allocations and two
        // copies per split for free.
        let mid = children.len() / 2;
        let (left, right) = children.split_at(mid);
        rayon::join_context(
            |ctx| {
                let migrated = ctx.migrated();
                for &(a, l) in left {
                    self.visit_lazy_rec(sink, a, l, level + 1, level_masks, emit_dim, migrated, max_depth_from_root);
                }
            },
            |ctx| {
                let migrated = ctx.migrated();
                for &(a, l) in right {
                    self.visit_lazy_rec(sink, a, l, level + 1, level_masks, emit_dim, migrated, max_depth_from_root);
                }
            },
        );
    }

    /// Corrected after a first attempt misunderstood the mechanism being
    /// proposed (see `DESIGN.md`): not one shared, growing pool of work
    /// items, but exactly `N` fixed slots, one per worker thread, each
    /// holding the `(node, open_mask)` pair described directly -- which
    /// node this thread is currently at, and a bitmask of that node's
    /// accepted transitions nobody has claimed yet. A thread that runs
    /// out of work at its own slot looks at every *other* slot for one
    /// with open bits, claims one, and continues DFS from there --
    /// exactly the worked example given: thread 0 finds thread 1 at node
    /// 57 with children 245 and 401 open, claims 245 (clearing that bit
    /// in slot 1), and starts exploring 245 by writing it, with its own
    /// first child already claimed, into slot 0.
    ///
    /// Claiming a bit is lock-free (a `compare_exchange` loop on an
    /// `AtomicU64` mask, the same "clear the lowest set bit" trick used
    /// throughout this file) -- no `Mutex` on the hot path at all. The
    /// `Mutex` the proposal asked for exists, but only for the *rare*
    /// event of a slot changing which node it refers to (`Slot::position`),
    /// not for every claim: a slot's mask is only ever set non-zero
    /// *after* its `position` is already correct, and a slot is only
    /// ever republished (by its own owning thread, never a thief) once
    /// its mask has already hit zero -- so no thief can ever observe a
    /// non-zero mask paired with a stale position. This also means no
    /// per-node allocation at all: unlike every other `parallel*`
    /// variant, a node's children are never collected into a `Vec` --
    /// the raw accepted-transitions bitmask *is* the published state,
    /// and a specific child's address is decoded only once actually
    /// claimed, by whoever claims it.
    pub fn restricted_descent_parallel_queue(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_queue only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let sink: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let num_workers = rayon::current_num_threads().max(1);
        let slots: Vec<Slot> = (0..num_workers)
            .map(|_| Slot {
                data: std::sync::Mutex::new(SlotData {
                    position: (0u64, [0u64; NUM_DIMS], 0usize),
                    open_mask: 0,
                }),
            })
            .collect();
        // Same global completion counter as the first attempt -- an
        // empty slot everywhere isn't proof nothing remains (another
        // thread could be mid-transition), so termination still needs
        // a count of "discovered but not yet dispatched" work, not just
        // "every slot I can currently see is empty." Seeded to 1 for
        // the root before any worker starts.
        let pending = std::sync::atomic::AtomicUsize::new(1);
        rayon::scope(|s| {
            for worker_id in 0..num_workers {
                let slots = &slots;
                let level_masks = &level_masks;
                let sink = &sink;
                let pending = &pending;
                s.spawn(move |_| {
                    if worker_id == 0 {
                        self.dfs_slot_rec(self.root_virtual_addr, [0u64; NUM_DIMS], 0, level_masks, emit_dim, sink, slots, 0, pending);
                        pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    self.slot_worker_loop(worker_id, level_masks, emit_dim, sink, slots, pending);
                });
            }
        });
        let mut out = sink.into_inner().expect("sink mutex never panics while locked");
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Claim attempt against one slot (which may be the caller's own,
    /// or another thread's -- the slot doesn't know or care which).
    /// Returns the claimed node's identity plus which transition (`qi`)
    /// was claimed, or `None` if the slot has nothing open right now.
    ///
    /// A first version split this into a lock-free CAS on the mask
    /// followed by a *separate* locked read of the position, reasoning
    /// that position only changes after the mask reaches zero so the
    /// two could never disagree. That reasoning had a real hole: mask
    /// reaching zero only means *no more bits are claimable*, not that
    /// every thread that already claimed an earlier bit has finished
    /// reading the position that went with it. A thief could win the
    /// CAS, then stall (for any reason -- scheduling, cache miss) before
    /// reading position; in that window the owner could legitimately
    /// see the mask at zero, conclude the slot is free, and republish
    /// it for a completely different node -- leaving the stalled thief
    /// to read the *new* position paired with a `qi` that was only ever
    /// valid for the *old* one. Measured directly, not hypothetically:
    /// this produced real, reproducible out-of-bounds panics once
    /// multiple real threads were involved (see `DESIGN.md`). Fixed by
    /// merging the mask and the position into one `Mutex<SlotData>` and
    /// doing the whole "check mask, pick a bit, clear it, read position"
    /// sequence as a single locked operation -- trading the lock-free
    /// claim path away in exchange for there being no gap left for a
    /// stale read to land in at all.
    fn slot_try_claim(slot: &Slot) -> Option<(u64, [u64; NUM_DIMS], usize, usize)> {
        let mut data = slot.data.lock().expect("slot mutex never panics while locked");
        if data.open_mask == 0 {
            return None;
        }
        let qi = data.open_mask.trailing_zeros() as usize;
        data.open_mask &= data.open_mask - 1;
        let (addr, lo, level) = data.position;
        Some((addr, lo, level, qi))
    }

    /// Decodes the one transition (`qi`) just claimed from a node at
    /// `(addr, lo, level)` -- the delta read every other variant in this
    /// file does once a child is already known to be wanted -- then
    /// recurses into it as *this* thread's own current work, publishing
    /// to `slots[my_slot]`, never the slot the claim came from.
    #[allow(clippy::too_many_arguments)]
    fn decode_claimed_and_recurse(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, qi: usize, level_masks: &[u64], emit_dim: usize, sink: &std::sync::Mutex<Vec<u32>>, slots: &[Slot], my_slot: usize, pending: &std::sync::atomic::AtomicUsize) {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
        let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
        let child_addr = addr + delta;
        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        self.dfs_slot_rec(child_addr, child_lo, level + 1, level_masks, emit_dim, sink, slots, my_slot, pending);
        pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Visits one node under `slots[my_slot]`'s exclusive ownership --
    /// only the thread whose own slot this is ever *writes* to it (reads
    /// via [`Self::slot_try_claim`] are shared with every other thread).
    /// Publishes `position` before `open_mask` (never the reverse): a
    /// thief can only ever observe a non-zero mask once the position it
    /// describes is already correct, because nothing sets the mask
    /// non-zero until the position write has already completed, and
    /// nothing re-publishes a new position until the previous mask has
    /// already reached zero (no thief can be mid-claim against an
    /// identity that's about to change, because "about to change" only
    /// happens once there's nothing left to claim).
    ///
    /// Crucial subtlety a first version got wrong (see `DESIGN.md`): a
    /// thread recursing into a child it just claimed must check whether
    /// its *own* slot is still occupied by an ancestor's still-open
    /// siblings before touching it again. Overwriting an occupied slot
    /// -- which plain recursion would do unconditionally, since every
    /// level of the tree wants to publish its own fan-out -- silently
    /// strands those siblings: nothing points at them anymore, nobody
    /// (not even this thread) can ever claim them, and the global
    /// `pending` count they were added under can never be decremented,
    /// hanging the whole query forever. The fix: only publish/share a
    /// node's fan-out when `slots[my_slot]` is currently empty;
    /// otherwise finish that node's entire subtree by plain recursion
    /// instead (still correct, just unshared at that level) -- and
    /// since this same check runs again at every level below, sharing
    /// resumes the instant the slot actually frees up, however deep
    /// that turns out to be, rather than never resuming at all.
    #[allow(clippy::too_many_arguments)]
    fn dfs_slot_rec(&self, addr: u64, lo: [u64; NUM_DIMS], level: usize, level_masks: &[u64], emit_dim: usize, sink: &std::sync::Mutex<Vec<u32>>, slots: &[Slot], my_slot: usize, pending: &std::sync::atomic::AtomicUsize) {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let word = self.storage.read_u64(block, offset);
        let mut relevant = word & level_masks[level];
        let is_last = level == self.height - 1;

        if relevant == 0 {
            return;
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];

        if is_last {
            let mut local = Vec::new();
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                local.push(child_lo[emit_dim] as u32);
            }
            sink.lock().expect("sink mutex never panics while locked").extend(local);
            return;
        }

        // Check whether my own slot is free *and*, if so, publish this
        // node into it, as one locked operation -- nothing else can
        // change a slot while its mask reads zero (only this node's own
        // owning thread ever sets a new, non-zero mask), so there's no
        // gap here for a thief to land in, but keeping the check and
        // the publish under the same lock acquisition makes that true
        // by construction instead of by argument.
        let published = {
            let mut data = slots[my_slot].data.lock().expect("slot mutex never panics while locked");
            if data.open_mask != 0 {
                false
            } else {
                // `pending` goes up *before* releasing the lock that
                // makes the new mask visible to anyone -- by the time a
                // thief can see a non-zero mask here, these units are
                // already accounted for.
                pending.fetch_add(relevant.count_ones() as usize, std::sync::atomic::Ordering::SeqCst);
                data.position = (addr, lo, level);
                data.open_mask = relevant;
                true
            }
        };
        if !published {
            // My own slot is still occupied by an ancestor's open
            // siblings -- don't touch it. Finish this node's whole
            // subtree by plain recursion; each child re-runs this same
            // check, so sharing resumes the moment the slot frees up.
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
                let child_addr = addr + delta;
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                self.dfs_slot_rec(child_addr, child_lo, level + 1, level_masks, emit_dim, sink, slots, my_slot, pending);
            }
            return;
        }

        // Race for a share of what was just published, exactly like any
        // thief would through the same `slot_try_claim` entry point --
        // no special-cased "this is mine first" fast path.
        while let Some((a2, l2, lvl2, qi)) = Self::slot_try_claim(&slots[my_slot]) {
            self.decode_claimed_and_recurse(a2, l2, lvl2, qi, level_masks, emit_dim, sink, slots, my_slot, pending);
        }
    }

    /// A worker with nothing left at its own slot: scan every other
    /// slot once for a claimable bit. `pending == 0` is the only
    /// global stop signal; finding nothing claimable anywhere on one
    /// pass just means yielding and trying again, since real work may
    /// still be arriving from a thread that's mid-recursion.
    fn slot_worker_loop(&self, my_slot: usize, level_masks: &[u64], emit_dim: usize, sink: &std::sync::Mutex<Vec<u32>>, slots: &[Slot], pending: &std::sync::atomic::AtomicUsize) {
        loop {
            if pending.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                return;
            }
            if let Some((a2, l2, lvl2, qi)) = Self::slot_try_claim(&slots[my_slot]) {
                self.decode_claimed_and_recurse(a2, l2, lvl2, qi, level_masks, emit_dim, sink, slots, my_slot, pending);
                continue;
            }
            let mut stole = false;
            for (other, slot) in slots.iter().enumerate() {
                if other == my_slot {
                    continue;
                }
                if let Some((a2, l2, lvl2, qi)) = Self::slot_try_claim(slot) {
                    self.decode_claimed_and_recurse(a2, l2, lvl2, qi, level_masks, emit_dim, sink, slots, my_slot, pending);
                    stole = true;
                    break;
                }
            }
            if !stole {
                std::thread::yield_now();
            }
        }
    }

    /// Same per-thread-slot idea as [`Self::restricted_descent_parallel_queue`],
    /// but with the recursion flattened out: instead of one cell per
    /// thread, this is `N` threads by `height` columns, one cell per
    /// *tree depth* a thread is currently holding along its own path --
    /// so every ancestor level of a thread's current descent is its own
    /// independently stealable cell, not just its single deepest one.
    /// This removes the single-slot design's "my slot is occupied by an
    /// ancestor, fall back to plain recursion" escape hatch entirely:
    /// there is no longer a reason for it to exist, since an ancestor
    /// and its descendant never compete for the same cell.
    ///
    /// A column's identity is *not* the real tree level -- after a
    /// thread steals a fresh starting point, its own column 0 can be
    /// any real level, which is exactly why `MatrixCell::real_level`
    /// is stored explicitly rather than inferred from the column index.
    ///
    /// `prefer_shallow` selects between two steal-order heuristics
    /// compared directly rather than guessed at: `true` always steals
    /// from the shallowest open column found across every other
    /// thread's row (more remaining subtree, in expectation, since
    /// higher levels of this tree tend to still have more than one
    /// live child), `false` always steals from the deepest. Both are
    /// plain linear scans with no per-row "anything open" summary bit
    /// -- deliberately: that is a real optimization opportunity if the
    /// scan itself turns out to matter, but it is not built until a
    /// measurement says it is worth building, not decided from
    /// reasoning alone.
    pub fn restricted_descent_parallel_matrix(&self, restrictions: [DimRestriction<'_>; NUM_DIMS], emit_dim: usize, prefer_shallow: bool) -> Vec<u32> {
        for (d, r) in restrictions.iter().enumerate() {
            match *r {
                DimRestriction::Exact(v) if (v as u64) >= self.dimension_sizes[d] => return Vec::new(),
                DimRestriction::Range(..) | DimRestriction::Multi(..) => panic!("PathFstDeltaPow2Bfs::restricted_descent_parallel_matrix only supports Exact/Free restrictions (dimension {d})"),
                _ => {}
            }
        }
        if self.height == 0 {
            return Vec::new();
        }
        let level_masks = self.precompute_level_masks(&restrictions);
        let sink: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let num_workers = rayon::current_num_threads().max(1);
        let matrix: Vec<Vec<std::sync::Mutex<MatrixCell>>> =
            (0..num_workers).map(|_| (0..self.height).map(|_| std::sync::Mutex::new(MatrixCell::empty())).collect()).collect();
        let pending = std::sync::atomic::AtomicUsize::new(0);

        // Initialization lives entirely outside the per-thread loop, as
        // its own one-time bootstrap step: the root is a single node,
        // there is nothing to parallelize about visiting it, so it is
        // handled directly here rather than by special-casing thread 0
        // inside its own loop body. One bit is claimed from it
        // immediately so that no thread's *own* active position is ever
        // real tree level 0 -- level 0 only ever exists, from this point
        // on, as an ordinary stealable backlog cell at `matrix[0][0]`,
        // found through the exact same scan every other steal goes
        // through, whether root's fan-out is narrower or wider than
        // `num_workers`.
        let thread0_descended = match self.matrix_initialize(&level_masks, emit_dim, &sink, &matrix, &pending) {
            None => return Vec::new(),
            Some(descended) => descended,
        };

        rayon::scope(|s| {
            for worker_id in 0..num_workers {
                let matrix = &matrix;
                let level_masks = &level_masks;
                let sink = &sink;
                let pending = &pending;
                s.spawn(move |_| {
                    let start_depth = if worker_id == 0 && thread0_descended { Some(1) } else { None };
                    self.matrix_worker_loop(worker_id, start_depth, level_masks, emit_dim, sink, matrix, pending, prefer_shallow);
                });
            }
        });
        let mut out = sink.into_inner().expect("sink mutex never panics while locked");
        out.sort_unstable();
        out.dedup();
        out
    }

    /// One-time bootstrap for [`Self::restricted_descent_parallel_matrix`]:
    /// publishes the root into `matrix[0][0]`, then immediately claims
    /// and dispatches exactly one of its transitions so `matrix[0][1]`
    /// (the real starting point for thread 0's own loop) is already
    /// filled before any thread starts running. Returns `None` only when
    /// the root itself has nothing accepted (an empty result for the
    /// whole query); otherwise `Some(did_descend)`, where `did_descend`
    /// is `false` only in the degenerate `height == 1` case (the root's
    /// own level is already the last one, so the one claimed transition
    /// was emitted directly instead of producing a child cell).
    fn matrix_initialize(
        &self,
        level_masks: &[u64],
        emit_dim: usize,
        sink: &std::sync::Mutex<Vec<u32>>,
        matrix: &[Vec<std::sync::Mutex<MatrixCell>>],
        pending: &std::sync::atomic::AtomicUsize,
    ) -> Option<bool> {
        let block = (self.root_virtual_addr >> self.capacity_bits) as usize;
        let offset = (self.root_virtual_addr & self.capacity_mask) as usize;
        let root_word = self.storage.read_u64(block, offset);
        let root_mask = root_word & level_masks[0];
        if root_mask == 0 {
            return None;
        }
        // One pending unit per bit in the published mask, exactly like
        // every other publish site (`dispatch_claimed_bit`'s own
        // `child_mask` case) -- not one unit for "the root" as a whole,
        // which would undercount whenever root has more than one
        // accepted child and let `pending` hit zero while real siblings
        // are still sitting unclaimed in `matrix[0][0]`.
        pending.fetch_add(root_mask.count_ones() as usize, std::sync::atomic::Ordering::SeqCst);
        let published = Self::matrix_publish(&matrix[0][0], self.root_virtual_addr, [0u64; NUM_DIMS], 0, root_word, root_mask);
        debug_assert!(published, "matrix_initialize: root cell cannot already be occupied, nothing has run yet");
        let (addr, lo, real_level, word, qi) =
            Self::matrix_try_claim(&matrix[0][0]).expect("just published a non-zero mask with no other thread running yet");
        let descended = self.dispatch_claimed_bit(addr, lo, real_level, word, qi, level_masks, emit_dim, sink, pending, matrix, 0, 1);
        Some(descended)
    }

    /// A worker's steady-state loop: work its own column, deepest-first,
    /// backing up a column at a time when the current one is exhausted
    /// -- mirroring [`Self::restricted_descent_parallel_queue`]'s "while
    /// let claim, else return to the caller" recursion, just flattened
    /// into an explicit loop over an explicit stack instead of the call
    /// stack. Once backed up past column 0 with nothing left (`depth`
    /// becomes `None`), falls back to stealing a brand new starting
    /// point via [`Self::matrix_scan_and_steal`], always landing at
    /// column 0 of its *own* row regardless of which column it was
    /// stolen from.
    #[allow(clippy::too_many_arguments)]
    fn matrix_worker_loop(
        &self,
        my_id: usize,
        mut depth: Option<usize>,
        level_masks: &[u64],
        emit_dim: usize,
        sink: &std::sync::Mutex<Vec<u32>>,
        matrix: &[Vec<std::sync::Mutex<MatrixCell>>],
        pending: &std::sync::atomic::AtomicUsize,
        prefer_shallow: bool,
    ) {
        loop {
            if pending.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                return;
            }
            let d = match depth {
                None => match Self::matrix_scan_and_steal(matrix, my_id, prefer_shallow) {
                    None => {
                        std::thread::yield_now();
                        continue;
                    }
                    Some((addr, lo, real_level, word, qi)) => {
                        let descended = self.dispatch_claimed_bit(addr, lo, real_level, word, qi, level_masks, emit_dim, sink, pending, matrix, my_id, 0);
                        depth = if descended { Some(0) } else { None };
                        continue;
                    }
                },
                Some(d) => d,
            };
            match Self::matrix_try_claim(&matrix[my_id][d]) {
                None => {
                    depth = if d == 0 { None } else { Some(d - 1) };
                }
                Some((addr, lo, real_level, word, qi)) => {
                    let descended = self.dispatch_claimed_bit(addr, lo, real_level, word, qi, level_masks, emit_dim, sink, pending, matrix, my_id, d + 1);
                    depth = Some(if descended { d + 1 } else { d });
                }
            }
        }
    }

    /// Claim one transition from a cell, regardless of whether it is the
    /// caller's own current column or a steal target -- the cell itself
    /// carries everything needed to decode it (`word`, cached once at
    /// publish time, never re-read from storage after), so a claim never
    /// needs `level_masks` at all; that array is only ever touched in
    /// [`Self::dispatch_claimed_bit`], exactly once, at the moment a
    /// *new* child cell is created.
    fn matrix_try_claim(cell: &std::sync::Mutex<MatrixCell>) -> Option<(u64, [u64; NUM_DIMS], usize, u64, usize)> {
        let mut data = cell.lock().expect("matrix cell mutex never panics while locked");
        if data.open_mask == 0 {
            return None;
        }
        let qi = data.open_mask.trailing_zeros() as usize;
        data.open_mask &= data.open_mask - 1;
        Some((data.addr, data.lo, data.real_level, data.word, qi))
    }

    /// Read-only peek at a cell's real level, without claiming anything
    /// -- used only by the steal scan to compare candidates by depth
    /// before committing to one. Still requires the lock (there is no
    /// lock-free summary field here; see this method's own doc on
    /// [`Self::restricted_descent_parallel_matrix`] for why that is
    /// deliberately deferred).
    fn matrix_peek_level(cell: &std::sync::Mutex<MatrixCell>) -> Option<usize> {
        let data = cell.lock().expect("matrix cell mutex never panics while locked");
        if data.open_mask == 0 {
            None
        } else {
            Some(data.real_level)
        }
    }

    /// Publish a freshly discovered node into a cell. Every call site
    /// only ever targets a cell it has just confirmed (by construction,
    /// not by re-checking here) is not currently occupied: a column
    /// backed away from because it was exhausted (`open_mask == 0`) can
    /// only become occupied again by this same thread, and a steal
    /// target is always this thread's *own* column 0, never written by
    /// anyone else. Returns whether the publish actually happened, but
    /// every caller treats a `false` here as a real bug (`debug_assert!`),
    /// not a condition to handle -- if it is ever hit, the surrounding
    /// invariant has broken somewhere and needs to be understood, not
    /// papered over.
    fn matrix_publish(cell: &std::sync::Mutex<MatrixCell>, addr: u64, lo: [u64; NUM_DIMS], real_level: usize, word: u64, open_mask: u64) -> bool {
        let mut data = cell.lock().expect("matrix cell mutex never panics while locked");
        if data.open_mask != 0 {
            return false;
        }
        data.addr = addr;
        data.lo = lo;
        data.real_level = real_level;
        data.word = word;
        data.open_mask = open_mask;
        true
    }

    /// Scans every *other* thread's row for an open cell, preferring
    /// the shallowest or the deepest real level found depending on
    /// `prefer_shallow`, then makes exactly one claim attempt against
    /// the best candidate seen. The candidate can lose the race between
    /// being peeked and being claimed (someone else takes its last bit
    /// first) -- handled by simply returning `None` for this poll
    /// iteration, same as finding nothing at all; the caller's loop
    /// just tries again, since real work may still be arriving from
    /// threads still mid-descent.
    fn matrix_scan_and_steal(matrix: &[Vec<std::sync::Mutex<MatrixCell>>], my_id: usize, prefer_shallow: bool) -> Option<(u64, [u64; NUM_DIMS], usize, u64, usize)> {
        let mut best: Option<(usize, usize, usize)> = None;
        for (t, row) in matrix.iter().enumerate() {
            if t == my_id {
                continue;
            }
            for (c, cell) in row.iter().enumerate() {
                if let Some(lvl) = Self::matrix_peek_level(cell) {
                    let better = match best {
                        None => true,
                        Some((_, _, best_lvl)) => {
                            if prefer_shallow {
                                lvl < best_lvl
                            } else {
                                lvl > best_lvl
                            }
                        }
                    };
                    if better {
                        best = Some((t, c, lvl));
                    }
                }
            }
        }
        let (t, c, _) = best?;
        Self::matrix_try_claim(&matrix[t][c])
    }

    /// Decodes and dispatches one claimed transition `qi` from a node
    /// already fully described by `(addr, lo, real_level, word)` --
    /// `word` came from the cell itself, never re-read from storage,
    /// since it cannot have changed since the cell was published (this
    /// structure is immutable once built).
    ///
    /// Three outcomes, matching [`Self::dfs_slot_rec`]'s leaf/dead-end/
    /// descend split exactly, just returning which one happened instead
    /// of recursing:
    /// - leaf level: emits directly, returns `false` (nothing to
    ///   publish, caller stays at its current column and tries claiming
    ///   more from the same cell);
    /// - dead end (the child exists but has nothing accepted under it):
    ///   returns `false` for the same reason;
    /// - real descent: publishes the child into `matrix[target_thread][target_column]`
    ///   and returns `true`, so the caller knows to move its own active
    ///   column to `target_column`.
    ///
    /// `pending`'s increment for the child's own fan-out happens
    /// *before* its decrement for this now-resolved claim, so the count
    /// can never be observed transiently at zero while real work still
    /// exists -- the same ordering [`Self::dfs_slot_rec`] uses.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_claimed_bit(
        &self,
        addr: u64,
        lo: [u64; NUM_DIMS],
        real_level: usize,
        word: u64,
        qi: usize,
        level_masks: &[u64],
        emit_dim: usize,
        sink: &std::sync::Mutex<Vec<u32>>,
        pending: &std::sync::atomic::AtomicUsize,
        matrix: &[Vec<std::sync::Mutex<MatrixCell>>],
        target_thread: usize,
        target_column: usize,
    ) -> bool {
        let dim_bits = self.level_dim_bits[real_level];
        let shifts = self.level_shifts_cache[real_level];
        let child_cell_bits = self.level_child_cell_bits_cache[real_level];
        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);

        if real_level == self.height - 1 {
            sink.lock().expect("sink mutex never panics while locked").push(child_lo[emit_dim] as u32);
            pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            return false;
        }

        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
        let delta = self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES);
        let child_addr = addr + delta;
        let child_block = (child_addr >> self.capacity_bits) as usize;
        let child_offset = (child_addr & self.capacity_mask) as usize;
        let child_word = self.storage.read_u64(child_block, child_offset);
        // The one and only place `level_masks` is touched for this
        // child: `open_mask` collapses the level's structural
        // constraint and this specific node's own existence bits into
        // one value, permanently, right here -- nothing downstream
        // (`matrix_try_claim`, `matrix_peek_level`, the worker loop)
        // ever indexes `level_masks` again for this cell.
        let child_mask = child_word & level_masks[real_level + 1];

        if child_mask == 0 {
            pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            return false;
        }
        pending.fetch_add(child_mask.count_ones() as usize, std::sync::atomic::Ordering::SeqCst);
        let published = Self::matrix_publish(&matrix[target_thread][target_column], child_addr, child_lo, real_level + 1, child_word, child_mask);
        debug_assert!(published, "dispatch_claimed_bit: target cell was unexpectedly occupied");
        pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        true
    }
}

/// One worker thread's current position, published for any other idle
/// thread to steal from -- the `(node_id, open_mask)` pair proposed
/// directly. `position` (which node) and `open_mask` (which of that
/// node's transitions remain unclaimed) are guarded by the *same* lock
/// deliberately, not two independent ones -- see
/// [`PathFstDeltaPow2Bfs::slot_try_claim`]'s doc comment for the race
/// that motivated merging them.
#[cfg(feature = "parallel")]
struct SlotData {
    position: (u64, [u64; NUM_DIMS], usize),
    open_mask: u64,
}

#[cfg(feature = "parallel")]
struct Slot {
    data: std::sync::Mutex<SlotData>,
}

/// One cell of [`PathFstDeltaPow2Bfs::restricted_descent_parallel_matrix`]'s
/// `threads x height` matrix -- a single tree-depth's worth of a single
/// thread's current descent path. `real_level` is stored explicitly
/// rather than inferred from the column index, because after a steal a
/// thread's own column 0 can hold a node from *any* real tree level, not
/// necessarily 0. `word` is cached here too, read from storage exactly
/// once at publish time: since this structure is immutable once built,
/// nothing can ever invalidate it, so every claim against this cell
/// afterward (by its owner or by a thief) decodes straight from the
/// cell's own fields with no further storage access for the word itself.
#[cfg(feature = "parallel")]
struct MatrixCell {
    addr: u64,
    lo: [u64; NUM_DIMS],
    real_level: usize,
    word: u64,
    open_mask: u64,
}

#[cfg(feature = "parallel")]
impl MatrixCell {
    fn empty() -> Self {
        MatrixCell { addr: 0, lo: [0u64; NUM_DIMS], real_level: 0, word: 0, open_mask: 0 }
    }
}

/// Phase 1 of [`PathFstDeltaPow2Bfs::build`] -- bottom-up dedup, same
/// partitioning/dedup logic as `build_rec_pow2`, but recording each
/// distinct node's `(existence_word, child_indices)` into `levels_nodes`
/// instead of writing storage. Returns this node's index within
/// `levels_nodes[level]`.
#[allow(clippy::too_many_arguments)]
fn collect_rec_bfs(
    points: &mut [Triple],
    scratch: &mut [Triple],
    lo: [u64; NUM_DIMS],
    level: usize,
    height: usize,
    level_dim_bits: &[[u8; NUM_DIMS]],
    level_shifts_cache: &[[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &[[u8; NUM_DIMS]],
    register: &mut [HashMap<(u64, Vec<u32>), u32>],
    levels_nodes: &mut [Vec<BfsNodeMeta>],
    visited_count: &mut u64,
) -> u32 {
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
        if let Some(&idx) = register[level].get(&key) {
            return idx;
        }
        let idx = levels_nodes[level].len() as u32;
        levels_nodes[level].push(BfsNodeMeta { existence_word, child_indices: Vec::new() });
        register[level].insert(key, idx);
        return idx;
    }

    let mut child_indices = Vec::with_capacity(existence_word.count_ones() as usize);
    for qi in 0..branching {
        if offsets[qi] < offsets[qi + 1] {
            let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
            let (child_points, child_scratch) = (&mut points[offsets[qi]..offsets[qi + 1]], &mut scratch[offsets[qi]..offsets[qi + 1]]);
            let child_idx = collect_rec_bfs(
                child_points,
                child_scratch,
                child_lo,
                level + 1,
                height,
                level_dim_bits,
                level_shifts_cache,
                level_child_cell_bits_cache,
                register,
                levels_nodes,
                visited_count,
            );
            child_indices.push(child_idx);
        }
    }

    let key = (existence_word, child_indices.clone());
    if let Some(&idx) = register[level].get(&key) {
        return idx;
    }
    let idx = levels_nodes[level].len() as u32;
    levels_nodes[level].push(BfsNodeMeta { existence_word, child_indices });
    register[level].insert(key, idx);
    idx
}

#[path = "fst_snapshot.rs"]
mod snapshot;

#[cfg(feature = "file")]
impl PackedFst<crate::storage_file::FileStorage> {
    /// Space occupied by the live temporary backing file (not a snapshot).
    pub fn disk_bytes(&self) -> u64 { self.storage.file_len }
}
