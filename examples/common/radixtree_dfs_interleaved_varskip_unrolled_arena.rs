//! A third storage variant alongside
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled`]
//! (heap-backed blocks, one `Vec<u8>` allocation each) and
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap`]
//! (`mmap`-backed blocks, one OS mapping each, `unsafe`). This one is a
//! *hand-rolled slab allocator*, staying in 100% safe Rust, built to
//! answer a specific question the mmap comparison raised: mmap's real
//! win turned out to be memory, not speed -- it eliminated almost all of
//! the heap variant's ~19% overhead by allocating each block at exactly
//! its target size once, with none of `Vec::extend_from_slice`'s
//! amortized-doubling overshoot. That's a property of *how the
//! allocation is sized*, not of `mmap` specifically -- so can safe Rust
//! get the same property without OS-level mapping at all?
//!
//! # The design: one arena, fixed-size slots, no per-block allocation
//!
//! Every block's bytes live in *one* shared `arena: Vec<u8>`, divided
//! into fixed-`block_capacity`-sized slots addressed by plain integer
//! index (`slot * block_capacity` is where it starts) -- there is no
//! separate allocation per block at all, heap or otherwise. The arena is
//! still sized exactly, not amortized: the initial bulk build does a
//! dry-run pass to count exactly how many slots it will need (see
//! [`plan_slot_lengths`]) before allocating `vec![0u8; num_slots *
//! block_capacity]` once, so -- like mmap, unlike the heap variant's
//! per-block `Vec::new()` grown incrementally -- there is no
//! reallocation, and hence no overshoot, anywhere in the bulk-build
//! path. Only [`Self::allocate_slot`] (reached during
//! [`Self::insert`]-triggered new-block creation, where the eventual
//! total truly isn't known in advance) falls back to the arena's own
//! ordinary `Vec::resize`, which *can* over-allocate via normal
//! amortized growth -- the same honest, unavoidable tradeoff every
//! growable safe-Rust collection makes when it can't see the future.
//!
//! Shifting bytes within a slot (for an in-place insert) or moving a
//! split's tail into a freshly allocated slot both go through
//! [`Vec::copy_within`] -- a safe, `std`-provided memmove that correctly
//! handles overlapping ranges, doing exactly what `MmapBlock::insert_at`
//! needed raw `ptr::copy` for, with no `unsafe` anywhere in this file at
//! all (confirmed via `grep -rn "unsafe"` on this file specifically,
//! same standard this whole project otherwise holds to). Logical
//! block-order (DFS order, what `Pos.block` indexes into) is kept
//! entirely separate from *physical* slot placement in the arena, via
//! `blocks: Vec<usize>` mapping logical position to slot index -- a
//! split or new-block insertion is still just `Vec::insert` on that
//! (small, block-count-sized, not node-count-sized) indirection vector,
//! exactly as it was for `Vec<MmapBlock>` and `Vec<Vec<u8>>` before it.
//!
//! # What this doesn't attempt
//!
//! A vacated slot (e.g. the original half of a block that split entirely
//! into its tail) is never reclaimed or reused -- there is no free-list.
//! Over many inserts this would eventually waste arena space on
//! abandoned slots; this prototype accepts that rather than building a
//! genuine allocator with reuse, matching the same "isolate the one
//! question being asked" scoping every variant in this exploration has
//! used.
//!
//! # The problem this solves
//!
//! Every structure in this comparison is built once, from a full triple
//! slice, and is otherwise read-only -- there is no `insert` anywhere in
//! `examples/common` except the one this file adds. The reason: varskip's
//! `combined: Vec<u8>` is one contiguous allocation, addressed by plain
//! byte offsets, with every "next sibling" reachable via `pos + skip`.
//! Inserting a new node's record *anywhere but the very end* means
//! shifting every byte physically after it -- an O(N) memmove over the
//! *entire* structure, however small the actual change.
//!
//! # The fix: block-relative addressing
//!
//! Split the same byte stream into many independently-allocated `Block`s
//! (each up to `block_capacity` bytes, holding as many whole node
//! records as fit -- no record is ever split across a block boundary),
//! linked purely by their order in `blocks: Vec<Block>`. A position is
//! now `Pos { block, offset }` instead of a bare `usize`. Inserting a new
//! node's record then costs:
//! - O(block_capacity) to shift bytes *within* one block (if it has
//!   room), or
//! - O(block_capacity) to split one full block into two (if it doesn't),
//!
//! plus O(height) to increment every far ancestor's stored skip value by
//! the inserted span (see [`Self::insert`]) -- never a pass over the
//! whole structure. The trade: advancing a position by a *large* skip
//! (`pos + skip` in the flat design) now potentially means walking
//! across several blocks (see [`Self::advance`]) instead of one
//! constant-time addition -- a genuine, honest cost this file accepts in
//! exchange for insertion no longer being all-or-nothing "rebuild the
//! whole tree," which is what every other structure in this project
//! still requires (see `examples/compare/scenarios.rs`'s
//! `bench_insertion`, which measures exactly that full-rebuild cost as
//! the *only* option for every other structure).
//!
//! # What did *not* change
//!
//! The bit-packing, per-level schedule, leaf/leaf-parent/far
//! classification, and per-level variable skip-width scheme are all
//! identical to
//! [`crate::common::radixtree_dfs_interleaved_varskip::RadixTreeDfsInterleavedVarSkip`]
//! -- copied here unmodified where the logic doesn't touch storage
//! addressing. In particular, [`compute_level_skip_bytes`]'s
//! schedule-only worst-case bound is exactly what makes insertion safe
//! *without* ever needing to widen an existing skip value's storage: it
//! was already sized for the worst case any dataset with this schedule
//! could produce, growing the dataset by one point included.
//!
//! # Scope
//!
//! [`Self::insert`] only supports points within the domain established
//! at build time (`debug_assert!`ed) -- growing the domain would change
//! `height` and the whole per-level schedule, a different and much
//! larger problem than the storage-addressing one this file targets.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8;
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;
const WORD_RECORD_BYTES: usize = 8;
const MAX_SKIP_RECORD_BYTES: usize = 8;
/// Default block capacity (bytes) -- deliberately a build/runtime
/// parameter, not a const, so the insert-cost/read-cost tradeoff can be
/// swept without recompiling, the same way
/// [`crate::common::radixtree_dfs_interleaved_varskip`]'s sibling
/// experiment swept `unroll`.
pub const DEFAULT_BLOCK_CAPACITY: usize = 4096;

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleavedVarSkipUnrolledArena {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsInterleavedVarSkipUnrolledArena::height(self)
    }
}

/// A position in the block-list: which block, and the byte offset within
/// it. Never lands mid-record -- every record (word, plus its skip value
/// if far) is fully contained in one block, so ordinary field reads
/// never need to consult [`RadixTreeDfsInterleavedVarSkipUnrolledArena::advance`]
/// at all; only *jumping to a different record* does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Pos {
    block: usize,
    offset: usize,
}

impl Pos {
    const START: Pos = Pos { block: 0, offset: 0 };
}

pub struct RadixTreeDfsInterleavedVarSkipUnrolledArena {
    /// One shared backing store for every block's bytes -- see the
    /// module doc comment. Divided into fixed-`block_capacity`-sized
    /// slots; slot `s` occupies `arena[s*block_capacity .. s*block_capacity
    /// + slot_used_len[s]]` (the rest of that slot's span, if any, is
    /// unused padding, present only because the arena's own length is
    /// always kept a multiple of `block_capacity`).
    arena: Vec<u8>,
    /// `slot_start[s]`: byte offset in `arena` where slot `s` begins.
    /// Almost always `s * block_capacity` in spirit, but tracked
    /// explicitly (rather than computed) because a handful of slots are
    /// deliberately larger than `block_capacity` -- see `slot_capacity`.
    slot_start: Vec<usize>,
    /// `slot_capacity[s]`: how many bytes are reserved for slot `s`.
    /// Normally `block_capacity`, except for a slot created specifically
    /// to hold one oversized record/chain that wouldn't fit in a normal
    /// slot (see [`Self::insert_bytes`] and `chop_into_blocks`) -- the
    /// arena analogue of `MmapBlock::new(block.capacity.max(new_bytes.len()))`
    /// in the sibling mmap variant, which also gives individual blocks
    /// their own capacity rather than assuming a single global one.
    slot_capacity: Vec<usize>,
    /// `slot_used_len[s]`: how many of slot `s`'s reserved bytes
    /// currently hold real content. Indexed by *slot* (physical arena
    /// position), not by logical DFS-order block index -- see `blocks`.
    slot_used_len: Vec<usize>,
    /// `blocks[i]` is the slot index holding the `i`-th block in
    /// DFS/logical order. This indirection is what lets a split insert
    /// a new logical block "between" two others (`Vec::insert` on this
    /// small, block-count-sized vector) without needing the *physical*
    /// arena layout to match logical order at all -- new slots are
    /// always appended to the end of the arena (see
    /// [`Self::allocate_slot`]), wherever they end up in `blocks`.
    blocks: Vec<usize>,
    block_capacity: usize,
    node_count: u64,
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

#[inline]
fn push_var(buf: &mut Vec<u8>, width: u8, val: u64) {
    debug_assert!(width == 8 || val < (1u64 << (8 * width as u32)), "value {val} does not fit in {width} bytes");
    match width {
        1 => buf.push(val as u8),
        2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&val.to_le_bytes()),
        _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
    }
}

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

/// Identical to varskip's own bound (see its module doc comment) --
/// schedule-only, safe for any dataset with this schedule including one
/// grown one point at a time via [`RadixTreeDfsInterleavedVarSkipUnrolledArena::insert`].
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
    max_bytes.iter().map(|&m| width_for(m)).collect()
}

impl RadixTreeDfsInterleavedVarSkipUnrolledArena {
    pub fn default_split() -> [u8; NUM_DIMS] {
        [BASE / NUM_DIMS as u8; NUM_DIMS]
    }

    pub fn build(triples: &[Triple], split: [u8; NUM_DIMS], block_capacity: usize) -> Self {
        Self::build_with_order(triples, split, [0, 1, 2], block_capacity)
    }

    pub fn build_with_order(triples: &[Triple], split: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        debug_assert!(split.iter().map(|&s| s as u32).sum::<u32>() <= BASE as u32, "split {split:?} sums to more than BASE ({BASE}) bits");
        let dimension_sizes = dimension_sizes_from(triples);
        let level_dim_bits = static_schedule(total_bits_needed(dimension_sizes), split);
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    pub fn build_dynamic(triples: &[Triple], block_capacity: usize) -> Self {
        Self::build_dynamic_with_order(triples, [0, 1, 2], block_capacity)
    }

    pub fn build_dynamic_with_order(triples: &[Triple], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        let dimension_sizes = dimension_sizes_from(triples);
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    fn build_from_schedule(triples: &[Triple], dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        debug_assert!(block_capacity >= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES, "block_capacity must fit at least one far-node record");
        if level_dim_bits.is_empty() {
            return RadixTreeDfsInterleavedVarSkipUnrolledArena {
                arena: Vec::new(),
                slot_start: Vec::new(),
                slot_capacity: Vec::new(),
                slot_used_len: Vec::new(),
                blocks: Vec::new(),
                block_capacity,
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

        // Bulk-build into one flat buffer first, exactly like varskip's
        // own recursive builder (unmodified) -- far simpler than making
        // the recursive counting-sort build itself block-aware, and
        // batch construction has no insertion-cost requirement to begin
        // with. `record_starts` collects each node's offset in DFS order
        // as a side effect, giving exact record boundaries to chop on
        // afterward without re-deriving them from level/width info.
        let mut points: Vec<(u32, u32, u32)> = triples.to_vec();
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        let mut flat = Vec::new();
        let mut node_count = 0u64;
        let mut record_starts = Vec::new();
        build_interleaved_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, height, &level_dim_bits, &level_shifts_cache, &level_child_cell_bits_cache, &level_skip_bytes, &mut flat, &mut node_count, &mut record_starts);

        let chopped = chop_into_blocks(&flat, &record_starts, block_capacity);

        RadixTreeDfsInterleavedVarSkipUnrolledArena {
            arena: chopped.arena,
            slot_start: chopped.slot_start,
            slot_capacity: chopped.slot_capacity,
            slot_used_len: chopped.slot_used_len,
            blocks: chopped.blocks,
            block_capacity,
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

    #[inline]
    fn advance_lo_dim(lo: u64, dim_bits: u8, shifts: u8, qi: usize, child_cell_bits: u8) -> u64 {
        let mask = (1usize << dim_bits) - 1;
        let sub = (qi >> shifts) & mask;
        lo + ((sub as u64) << child_cell_bits)
    }

    /// Appends a brand-new slot of `capacity` bytes to the end of the
    /// arena and returns its slot index. The only place arena growth can
    /// overshoot (via `Vec::resize`'s ordinary amortized doubling) --
    /// deliberately: reached only from insert-triggered new-block
    /// creation, where (unlike the one-time bulk build) the eventual
    /// total size genuinely isn't known in advance, and reserving
    /// exactly `capacity` more bytes on *this* one shared `Vec` every
    /// single insert would risk a full-arena copy each time instead of
    /// mmap's/heap's independent per-block allocations.
    fn allocate_slot(&mut self, capacity: usize) -> usize {
        let slot = self.slot_capacity.len();
        let start = self.arena.len();
        self.arena.resize(start + capacity, 0);
        self.slot_start.push(start);
        self.slot_capacity.push(capacity);
        self.slot_used_len.push(0);
        slot
    }

    #[inline]
    fn slot_slice(&self, slot: usize) -> &[u8] {
        let start = self.slot_start[slot];
        &self.arena[start..start + self.slot_used_len[slot]]
    }

    /// In-place insertion at `offset` within `slot`: shifts the existing
    /// `[offset, used)` tail right by `bytes.len()` via [`Vec::copy_within`]
    /// -- a safe, `std`-provided memmove that handles the overlapping
    /// ranges this shift produces -- then writes `bytes` into the gap.
    /// Panics (never reallocates a slot in place) if it wouldn't fit;
    /// callers check capacity first, same discipline `MmapBlock::insert_at`
    /// required of its own callers.
    fn slot_insert_at(&mut self, slot: usize, offset: usize, bytes: &[u8]) {
        let start = self.slot_start[slot];
        let used = self.slot_used_len[slot];
        debug_assert!(offset <= used, "slot_insert_at offset {offset} beyond slot's used length {used}");
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        self.arena.copy_within(start + offset..start + used, start + offset + bytes.len());
        self.arena[start + offset..start + offset + bytes.len()].copy_from_slice(bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }

    /// Appends `bytes` at the end of `slot`'s used region. Panics rather
    /// than reallocating if it wouldn't fit.
    fn slot_extend_from_slice(&mut self, slot: usize, bytes: &[u8]) {
        let start = self.slot_start[slot];
        let used = self.slot_used_len[slot];
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        self.arena[start + used..start + used + bytes.len()].copy_from_slice(bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }

    #[inline]
    fn block_slice(&self, logical: usize) -> &[u8] {
        self.slot_slice(self.blocks[logical])
    }

    #[inline]
    fn block_len(&self, logical: usize) -> usize {
        self.slot_used_len[self.blocks[logical]]
    }

    fn block_write_u64_at(&mut self, logical: usize, offset: usize, val: u64) {
        let slot = self.blocks[logical];
        let start = self.slot_start[slot] + offset;
        self.arena[start..start + 8].copy_from_slice(&val.to_le_bytes());
    }

    fn block_write_var_at(&mut self, logical: usize, offset: usize, width: u8, val: u64) {
        let slot = self.blocks[logical];
        let start = self.slot_start[slot];
        let capacity = self.slot_capacity[slot];
        write_var_at(&mut self.arena[start..start + capacity], offset, width, val);
    }

    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height as usize {
            WORD_RECORD_BYTES + self.level_skip_bytes[level] as usize
        } else {
            WORD_RECORD_BYTES
        }
    }

    /// Advances `pos` by `n` bytes, walking across block boundaries as
    /// needed. O(1) when `pos` and the target land in the same block
    /// (the overwhelmingly common case for nearby siblings); O(blocks
    /// spanned) for a skip large enough to cross several -- the honest
    /// cost this file's design accepts (see the module doc comment) in
    /// exchange for O(block_capacity) insertion instead of O(structure
    /// size).
    #[inline]
    fn advance(&self, mut pos: Pos, mut n: usize) -> Pos {
        loop {
            let avail = self.block_len(pos.block) - pos.offset;
            if n < avail {
                pos.offset += n;
                return pos;
            }
            n -= avail;
            if pos.block + 1 >= self.blocks.len() {
                debug_assert_eq!(n, 0, "advance() ran past the end of all blocks");
                pos.offset += avail; // off-the-end sentinel; never dereferenced
                return pos;
            }
            pos.block += 1;
            pos.offset = 0;
        }
    }

    /// Byte position immediately after one specific child's entire
    /// record. Identical rule set to varskip's own (see its doc comment)
    /// -- the only change is that reading a child's own word/skip is a
    /// same-block read (records never straddle a block boundary), while
    /// the final "jump to where this subtree ends" goes through
    /// [`Self::advance`], which may cross blocks.
    #[inline]
    fn child_span_end(&self, child_pos: Pos, child_level: usize) -> Pos {
        if child_level + 2 < self.height as usize {
            let width = self.level_skip_bytes[child_level];
            let skip = read_var(self.block_slice(child_pos.block), child_pos.offset + WORD_RECORD_BYTES, width) as usize;
            self.advance(child_pos, skip)
        } else if child_level == self.height as usize - 1 {
            self.advance(child_pos, WORD_RECORD_BYTES)
        } else {
            let word = read_u64(self.block_slice(child_pos.block), child_pos.offset);
            self.advance(child_pos, WORD_RECORD_BYTES + WORD_RECORD_BYTES * word.count_ones() as usize)
        }
    }

    #[inline]
    fn advance_past_children(&self, child_pos: Pos, remaining_bits: u64, child_level: usize) -> Pos {
        if child_level == self.height as usize - 1 {
            self.advance(child_pos, WORD_RECORD_BYTES * remaining_bits.count_ones() as usize)
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
        let mut pos = Pos::START;
        for level in 0..self.height as usize {
            let is_last = level == self.height as usize - 1;
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let mut qi = 0usize;
            let mut next_lo = lo;
            for d in 0..NUM_DIMS {
                let sub = (coords[d] - lo[d]) >> child_cell_bits[d];
                qi |= (sub as usize) << shifts[d];
                next_lo[d] += sub << child_cell_bits[d];
            }
            let word = read_u64(self.block_slice(pos.block), pos.offset);
            if (word >> qi) & 1 == 0 {
                return false;
            }
            if is_last {
                return true;
            }
            let earlier = word & ((1u64 << qi) - 1);
            let first_child_pos = self.advance(pos, self.own_record_bytes(level));
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
        self.query_rec(Pos::START, 0, 0, &level_masks[..self.height as usize], free_dim, unconstrained_from, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn first_fully_open_level(&self, level_masks: &[u64; MAX_HEIGHT]) -> usize {
        for level in 0..self.height as usize {
            let dim_bits = self.level_dim_bits[level];
            let bit_sum = (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
            let branching: u64 = 1u64 << bit_sum;
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
        debug_assert!(self.height as usize <= MAX_HEIGHT);
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
        self.query_rec(Pos::START, 0, 0, &level_masks[..self.height as usize], target_dim, unconstrained_from, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Same merged constrained/listing descent as varskip's `query_rec`
    /// (see its doc comment) -- only the position type changed, from a
    /// bare `usize` to [`Pos`].
    fn query_rec(&self, start_pos: Pos, start_lo: u64, start_level: usize, level_masks: &[u64], free_dim: usize, unconstrained_from: usize, out: &mut Vec<u32>) {
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
            let word = read_u64(self.block_slice(pos.block), pos.offset);
            debug_assert!(word != 0, "visited node at level {level} has an empty existence word");

            if is_last {
                let mut relevant = if listing { word } else { word & level_masks[level] };
                while relevant != 0 {
                    let qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                    out.push(child_lo as u32);
                }
                pos = self.advance(pos, WORD_RECORD_BYTES);
            } else if listing && level + 1 == self.height as usize - 1 {
                let leaf_level = level + 1;
                let leaf_dim_bits = self.level_dim_bits[leaf_level][free_dim];
                let leaf_shifts = self.level_shifts_cache[leaf_level][free_dim];
                let leaf_child_cell_bits = self.level_child_cell_bits_cache[leaf_level][free_dim];
                let mut leaf_pos = self.advance(pos, WORD_RECORD_BYTES);
                let mut own_bits = word;
                while own_bits != 0 {
                    let qi = own_bits.trailing_zeros() as usize;
                    own_bits &= own_bits - 1;
                    let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                    let leaf_word = read_u64(self.block_slice(leaf_pos.block), leaf_pos.offset);
                    let mut relevant = leaf_word;
                    while relevant != 0 {
                        let leaf_qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let leaf_lo = Self::advance_lo_dim(child_lo, leaf_dim_bits, leaf_shifts, leaf_qi, leaf_child_cell_bits);
                        out.push(leaf_lo as u32);
                    }
                    leaf_pos = self.advance(leaf_pos, WORD_RECORD_BYTES);
                }
                pos = leaf_pos;
            } else if listing {
                let qi = word.trailing_zeros() as usize;
                let remaining = word & (word - 1);
                let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                debug_assert!(stack_len < MAX_HEIGHT);
                stack[stack_len] = ConstrainedFrame { lo, remaining, child_pos: Pos::START };
                stack_len += 1;
                pos = self.advance(pos, self.own_record_bytes(level));
                lo = child_lo;
                level += 1;
                descended = true;
            } else {
                let mask = level_masks[level];
                let mut child_pos = self.advance(pos, self.own_record_bytes(level));
                let mut remaining = word;

                while remaining != 0 {
                    let qi = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let matched = (mask >> qi) & 1 != 0;
                    let next_child_pos = self.child_span_end(child_pos, level + 1);

                    if matched {
                        let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                        debug_assert!(stack_len < MAX_HEIGHT);
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

    /// Inserts one point, mutating the structure in place. Returns
    /// `false` (no change) if it already existed. Only supports points
    /// within the domain established at build time -- see the module
    /// doc comment's "Scope" section.
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> bool {
        debug_assert!(self.height > 0, "insert() on an empty/degenerate tree is out of scope for this prototype");
        debug_assert!(self.in_bounds(s, p, o), "insert({s},{p},{o}) is outside the domain established at build time");
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut pos = Pos::START;
        // One entry per visited ancestor: its own record's position, its
        // level, and the word value *before* any mutation this insert
        // makes (needed to recompute `earlier`/skip-propagation
        // correctly even after this function has already written new
        // bytes earlier in the same call).
        let mut trail: [(Pos, usize, u64); MAX_HEIGHT] = [(Pos::START, 0, 0); MAX_HEIGHT];
        let mut trail_len = 0usize;

        for level in 0..self.height as usize {
            let is_last = level == self.height as usize - 1;
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let mut qi = 0usize;
            let mut next_lo = lo;
            for d in 0..NUM_DIMS {
                let sub = (coords[d] - lo[d]) >> child_cell_bits[d];
                qi |= (sub as usize) << shifts[d];
                next_lo[d] += sub << child_cell_bits[d];
            }
            let word = read_u64(self.block_slice(pos.block), pos.offset);
            trail[trail_len] = (pos, level, word);
            trail_len += 1;

            if (word >> qi) & 1 == 0 {
                self.insert_new_branch(pos, level, qi, word, coords, lo, &trail[..trail_len]);
                return true;
            }
            if is_last {
                return false; // already present
            }
            let earlier = word & ((1u64 << qi) - 1);
            let first_child_pos = self.advance(pos, self.own_record_bytes(level));
            pos = self.advance_past_children(first_child_pos, earlier, level + 1);
            lo = next_lo;
        }
        unreachable!("height > 0 checked above")
    }

    /// Handles the divergence case found by [`Self::insert`]: `qi`'s bit
    /// isn't set in the word at `node_pos` (level `level`). Sets it, and
    /// -- unless `level` is the leaf level, where that bit flip alone is
    /// the entire change -- builds and splices in a brand-new straight
    /// chain of records for levels `level + 1..height` (there is only
    /// one point below this divergence, so no branching, no counting
    /// sort needed), then walks `trail` to increment every far
    /// ancestor's stored skip by the newly-inserted byte count.
    fn insert_new_branch(&mut self, node_pos: Pos, level: usize, qi: usize, word: u64, coords: [u64; NUM_DIMS], lo: [u64; NUM_DIMS], trail: &[(Pos, usize, u64)]) {
        self.set_bit_in_word(node_pos, qi);
        let is_last = level == self.height as usize - 1;
        if is_last {
            return; // no new bytes: existing leaf word already covers this value
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let child_lo = Self::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        let chain = build_single_point_chain(coords, level + 1, self.height, &self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache, &self.level_skip_bytes, child_lo);

        let earlier = word & ((1u64 << qi) - 1);
        let first_child_pos = self.advance(node_pos, self.own_record_bytes(level));
        let insert_pos = self.advance_past_children(first_child_pos, earlier, level + 1);

        self.insert_bytes(insert_pos, &chain);
        self.node_count += (self.height as usize - (level + 1)) as u64;

        for &(anc_pos, anc_level, _) in trail {
            if anc_level + 2 < self.height as usize {
                let width = self.level_skip_bytes[anc_level];
                let skip_offset = anc_pos.offset + WORD_RECORD_BYTES;
                let old = read_var(self.block_slice(anc_pos.block), skip_offset, width);
                self.block_write_var_at(anc_pos.block, skip_offset, width, old + chain.len() as u64);
            }
        }
    }

    #[inline]
    fn set_bit_in_word(&mut self, pos: Pos, qi: usize) {
        let old = read_u64(self.block_slice(pos.block), pos.offset);
        let new = old | (1u64 << qi);
        self.block_write_u64_at(pos.block, pos.offset, new);
    }

    /// Inserts `new_bytes` into the block stream at `pos`. A slot has a
    /// *fixed* capacity for its whole lifetime (see the module doc
    /// comment) -- it can never grow past `slot_capacity[slot]` in
    /// place, not even for a tail append. So the only two cases are: it
    /// fits in place (`slot_insert_at`, an in-slot shift via
    /// `Vec::copy_within`), or it doesn't, in which case `new_bytes`
    /// always gets its own fresh slot rather than trying to force it
    /// into an existing one -- splitting the slot at `pos.offset` first
    /// (an empty, effectively-abandoned tail slot for the common
    /// tail-append case, since `is_tail` skips allocating one at all; a
    /// real second half otherwise), then inserting the new slot between
    /// the two halves via `Vec::insert` on the small `blocks` indirection
    /// vector. This trades a small amount of block-count growth (a lone
    /// inserted chain gets its own slot rather than being packed
    /// alongside a neighbor) for never having to reason about "does it
    /// still fit after the split" -- always correct regardless of how
    /// full either half ends up. Every position recorded *before* `pos`
    /// in the same [`Self::insert`] call stays valid: a split only ever
    /// moves bytes at-or-after `pos.offset` into new slots referenced
    /// after `pos.block` in `blocks`, and every earlier position is (by
    /// construction, since it's an ancestor of whatever is being
    /// inserted) strictly before `pos` within that same block or in an
    /// earlier block altogether.
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8]) {
        let slot = self.blocks[pos.block];
        let used = self.slot_used_len[slot];
        let capacity = self.slot_capacity[slot];
        if used + new_bytes.len() <= capacity {
            self.slot_insert_at(slot, pos.offset, new_bytes);
            return;
        }
        let is_tail = pos.offset == used;
        let tail_slot = if is_tail {
            None
        } else {
            let tail_len = used - pos.offset;
            let new_slot = self.allocate_slot(capacity);
            let src_start = self.slot_start[slot] + pos.offset;
            let dst_start = self.slot_start[new_slot];
            self.arena.copy_within(src_start..src_start + tail_len, dst_start);
            self.slot_used_len[new_slot] = tail_len;
            self.slot_used_len[slot] = pos.offset;
            Some(new_slot)
        };
        let new_block_capacity = self.block_capacity.max(new_bytes.len());
        let new_block_slot = self.allocate_slot(new_block_capacity);
        self.slot_extend_from_slice(new_block_slot, new_bytes);
        self.blocks.insert(pos.block + 1, new_block_slot);
        if let Some(tail_slot) = tail_slot {
            self.blocks.insert(pos.block + 2, tail_slot);
        }
    }
}

/// Builds the byte chain for a brand-new straight-line path (levels
/// `start_level..height`) for exactly one point -- no branching, so
/// unlike [`build_interleaved_rec`] this needs no counting sort: each
/// level's existence word has exactly one bit set, and (since there's
/// only one child at every level) each level's own span is `own_bytes +
/// child's span`, computable bottom-up in one pass before writing
/// anything, so skip values can be pushed directly instead of reserved
/// as placeholders and backfilled.
#[allow(clippy::too_many_arguments)]
fn build_single_point_chain(coords: [u64; NUM_DIMS], start_level: usize, height: u32, level_dim_bits: &[[u8; NUM_DIMS]], level_shifts_cache: &[[u8; NUM_DIMS]], level_child_cell_bits_cache: &[[u8; NUM_DIMS]], level_skip_bytes: &[u8], mut lo: [u64; NUM_DIMS]) -> Vec<u8> {
    let height = height as usize;
    let mut sizes = vec![0u64; height];
    sizes[height - 1] = WORD_RECORD_BYTES as u64;
    for lvl in (start_level..height - 1).rev() {
        let own = if lvl + 2 < height { WORD_RECORD_BYTES as u64 + level_skip_bytes[lvl] as u64 } else { WORD_RECORD_BYTES as u64 };
        sizes[lvl] = own + sizes[lvl + 1];
    }

    let mut chain = Vec::with_capacity(sizes[start_level] as usize);
    for lvl in start_level..height {
        let dim_bits = level_dim_bits[lvl];
        let shifts = level_shifts_cache[lvl];
        let child_cell_bits = level_child_cell_bits_cache[lvl];
        let mut qi = 0usize;
        for d in 0..NUM_DIMS {
            let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
            qi |= sub << shifts[d];
        }
        push_u64(&mut chain, 1u64 << qi);
        if lvl + 2 < height {
            // `sizes[lvl]` is exactly this level's own header plus every
            // descendant's span -- the same quantity `total_bytes`
            // accumulates in `build_interleaved_rec`, just computed
            // bottom-up in advance here instead of via post-order return
            // values, since there's no branching to sum over.
            push_var(&mut chain, level_skip_bytes[lvl], sizes[lvl]);
        }
        lo = RadixTreeDfsInterleavedVarSkipUnrolledArena::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
    }
    debug_assert_eq!(chain.len() as u64, sizes[start_level], "chain length must match the precomputed size it reports to its own parent");
    chain
}

/// The output of [`chop_into_blocks`]: an arena already carved into
/// slots and filled with every node record, ready to become a
/// [`RadixTreeDfsInterleavedVarSkipUnrolledArena`]'s fields directly.
struct ChoppedArena {
    arena: Vec<u8>,
    slot_start: Vec<usize>,
    slot_capacity: Vec<usize>,
    slot_used_len: Vec<usize>,
    blocks: Vec<usize>,
}

/// Dry-run of `chop_into_blocks`'s own greedy-packing decision (see
/// there) -- same walk over `record_starts`, but only counting each
/// slot's eventual capacity, in the order slots will be created, instead
/// of writing anything. Lets the caller size the arena exactly once
/// (`vec![0u8; total]`) up front rather than growing it incrementally
/// while copying record bytes in, which is what would reintroduce the
/// amortized-doubling overshoot this whole variant exists to avoid (see
/// the module doc comment).
fn plan_slot_lengths(record_starts: &[usize], flat_len: usize, capacity: usize) -> Vec<usize> {
    let mut slot_capacities = Vec::new();
    let mut current_len = 0usize;
    for (i, &start) in record_starts.iter().enumerate() {
        let end = record_starts.get(i + 1).copied().unwrap_or(flat_len);
        let record_len = end - start;
        if current_len != 0 && current_len + record_len > capacity {
            slot_capacities.push(capacity);
            current_len = 0;
        }
        // A record wider than `capacity` (never happens at this
        // project's node-record sizes vs. any realistic capacity, but
        // defensively handled anyway) gets a slot sized just for it,
        // rather than a `debug_assert!` failure deep in a copy later.
        if record_len > capacity {
            slot_capacities.push(record_len);
        } else {
            current_len += record_len;
        }
    }
    if current_len != 0 {
        slot_capacities.push(capacity);
    }
    slot_capacities
}

/// Chops `flat` into arena slots of at most `capacity` bytes, never
/// splitting a record (using `record_starts`, DFS-order offsets
/// collected by [`build_interleaved_rec`]). Two passes: [`plan_slot_lengths`]
/// first decides exactly how many slots of what sizes are needed (so the
/// arena is allocated exactly once, at its final size), then this
/// function re-walks the same records -- making the identical
/// packing decision, now against real slot boundaries -- and copies each
/// one directly into place.
fn chop_into_blocks(flat: &[u8], record_starts: &[usize], capacity: usize) -> ChoppedArena {
    let slot_capacities = plan_slot_lengths(record_starts, flat.len(), capacity);
    let mut slot_start = Vec::with_capacity(slot_capacities.len());
    let mut total = 0usize;
    for &cap in &slot_capacities {
        slot_start.push(total);
        total += cap;
    }
    let mut arena = vec![0u8; total];
    let mut slot_used_len = vec![0usize; slot_capacities.len()];

    let mut slot_idx = 0usize;
    let mut current_len = 0usize;
    for (i, &start) in record_starts.iter().enumerate() {
        let end = record_starts.get(i + 1).copied().unwrap_or(flat.len());
        let record = &flat[start..end];
        if current_len != 0 && current_len + record.len() > capacity {
            slot_idx += 1;
            current_len = 0;
        }
        let dest = slot_start[slot_idx] + current_len;
        arena[dest..dest + record.len()].copy_from_slice(record);
        current_len += record.len();
        slot_used_len[slot_idx] = current_len;
        if record.len() > capacity {
            slot_idx += 1;
            current_len = 0;
        }
    }

    let blocks: Vec<usize> = (0..slot_capacities.len()).collect();
    ChoppedArena { arena, slot_start, slot_capacity: slot_capacities, slot_used_len, blocks }
}

#[derive(Clone, Copy)]
struct ConstrainedFrame {
    lo: u64,
    remaining: u64,
    child_pos: Pos,
}

impl ConstrainedFrame {
    const EMPTY: ConstrainedFrame = ConstrainedFrame { lo: 0, remaining: 0, child_pos: Pos::START };
}

/// Identical to varskip's own builder, plus `record_starts`: pushes
/// `combined.len()` right before every node's word, in DFS order, so the
/// caller can chop the finished flat buffer into blocks without
/// re-deriving record boundaries from level/width info afterward.
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
    record_starts: &mut Vec<usize>,
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
    record_starts.push(combined.len());
    push_u64(combined, existence_word);
    *node_count += 1;
    let skip_placeholder_pos = if is_far_node {
        let pos = combined.len();
        combined.resize(pos + skip_width as usize, 0);
        Some(pos)
    } else {
        None
    };

    let mut total_bytes: u64 = WORD_RECORD_BYTES as u64 + if is_far_node { skip_width as u64 } else { 0 };
    if !is_last {
        for qi in 0..branching {
            let qstart = offsets[qi];
            let qend = offsets[qi + 1];
            if qstart < qend {
                let child_lo = RadixTreeDfsInterleavedVarSkipUnrolledArena::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_bytes = build_interleaved_rec(&mut points[qstart..qend], scratch, child_lo, level + 1, height, level_dim_bits, level_shifts_cache, level_child_cell_bits_cache, level_skip_bytes, combined, node_count, record_starts);
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

impl MemSize for RadixTreeDfsInterleavedVarSkipUnrolledArena {
    fn mem_size_bytes(&self) -> usize {
        // Unlike the mmap variant (whose blocks' real footprint is each
        // mapping's fixed `capacity`, not its used length), the arena is
        // one ordinary `Vec<u8>` -- `vec_bytes`'s usual "actual allocated
        // bytes" convention already covers every slot, used or not,
        // since slot capacity is exactly what the arena's own length
        // was grown by (see `allocate_slot`/`chop_into_blocks`).
        crate::common::metrics::vec_bytes(&self.arena)
            + crate::common::metrics::vec_bytes(&self.slot_start)
            + crate::common::metrics::vec_bytes(&self.slot_capacity)
            + crate::common::metrics::vec_bytes(&self.slot_used_len)
            + crate::common::metrics::vec_bytes(&self.blocks)
            + crate::common::metrics::vec_bytes(&self.level_skip_bytes)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfsInterleavedVarSkipUnrolledArena {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("arena: Vec<u8> backing store", crate::common::metrics::vec_bytes(&self.arena)),
            ("slot_start (O(slots))", crate::common::metrics::vec_bytes(&self.slot_start)),
            ("slot_capacity (O(slots))", crate::common::metrics::vec_bytes(&self.slot_capacity)),
            ("slot_used_len (O(slots))", crate::common::metrics::vec_bytes(&self.slot_used_len)),
            ("blocks: logical-to-slot indirection (O(blocks))", crate::common::metrics::vec_bytes(&self.blocks)),
            ("level_skip_bytes (O(height))", crate::common::metrics::vec_bytes(&self.level_skip_bytes)),
            ("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)),
            ("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)),
            ("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)),
        ]
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::{uniform, zipfian_predicates, SplitMix64};
    use crate::common::reference::NaiveIndex;

    /// Small capacities deliberately force block splits on almost every
    /// insert -- the whole point of these tests is to exercise
    /// `insert_bytes`'s split path, not just its happy-path in-place
    /// growth.
    const TEST_CAPACITIES: [usize; 4] = [32, 64, 256, DEFAULT_BLOCK_CAPACITY];

    fn cross_check_tree(tree: &RadixTreeDfsInterleavedVarSkipUnrolledArena, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
    }

    fn cross_check(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, split: [u8; NUM_DIMS]) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build(triples, split, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?} cap={cap}"));
        }
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(triples, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("dynamic cap={cap}"));
        }
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split(), cap);
            assert!(!tree.contains(0, 0, 0));
            assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        }
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split(), cap);
            assert!(tree.contains(0, 0, 0));
            assert!(!tree.contains(0, 0, 1));
        }
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split());
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
        }
    }

    #[test]
    fn duplicate_heavy_dataset() {
        let mut triples = vec![(1u32, 1u32, 1u32); 20];
        triples.push((5, 5, 5));
        cross_check(&triples, 6, 6, 6, RadixTreeDfsInterleavedVarSkipUnrolledArena::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    #[test]
    fn deep_far_nesting_navigates_correctly() {
        let triples = uniform(500, 20, 500, 6000, 999);
        let naive = NaiveIndex::build(&triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(&triples, cap);
            for s in (0..500u32).step_by(7) {
                let mut got = tree.query_one_bound(0, s, 2);
                let mut expect: Vec<u32> = (0..20u32).flat_map(|p| naive.row(p, s)).collect();
                got.sort_unstable();
                got.dedup();
                expect.sort_unstable();
                expect.dedup();
                assert_eq!(got, expect, "query_one_bound(S={s}) cap={cap}");
            }
        }
    }

    /// The core new capability: build from *part* of a dataset, `insert`
    /// the rest one point at a time, and verify against a `NaiveIndex`
    /// built from the *full* set after every single insert -- not just
    /// at the end, so a bug that only shows up transiently (e.g. a
    /// dangling `Pos` after a block split) can't hide behind a
    /// coincidentally-correct final state.
    #[test]
    fn insert_one_at_a_time_matches_naive_after_every_insert() {
        for cap in TEST_CAPACITIES {
            let all = uniform(30, 6, 30, 400, 12345 + cap as u64);
            let split_at = all.len() / 3;
            let (initial, rest) = all.split_at(split_at);

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(initial, cap);
            let mut seen: Vec<Triple> = initial.to_vec();
            for &(s, p, o) in rest {
                let was_new = !seen.contains(&(s, p, o));
                let inserted = tree.insert(s, p, o);
                assert_eq!(inserted, was_new, "insert({s},{p},{o}) return value mismatch, cap={cap}");
                if was_new {
                    seen.push((s, p, o));
                }
                let naive = NaiveIndex::build(&seen);
                cross_check_tree(&tree, &naive, 30, 6, 30, &format!("after inserting ({s},{p},{o}) cap={cap}"));
            }
        }
    }

    /// Same idea, but inserting in a *randomized* order (not the
    /// dataset's own generation order) and skipping the expensive
    /// after-every-insert full cross-check in favor of one at the end --
    /// covers many more distinct insertion positions (first/middle/last
    /// child at various levels) than the sequential test above, at a
    /// fraction of the runtime.
    #[test]
    fn insert_in_random_order_matches_naive_at_the_end() {
        for cap in TEST_CAPACITIES {
            let all = uniform(50, 8, 50, 1500, 999_000 + cap as u64);
            let split_at = all.len() / 2;
            let (initial, rest) = all.split_at(split_at);
            let mut rest_shuffled = rest.to_vec();
            let mut rng = SplitMix64::new(0xBADC0FFEE ^ cap as u64);
            for i in (1..rest_shuffled.len()).rev() {
                let j = rng.next_range(i as u64 + 1) as usize;
                rest_shuffled.swap(i, j);
            }

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(initial, cap);
            for &(s, p, o) in &rest_shuffled {
                tree.insert(s, p, o);
            }
            let naive = NaiveIndex::build(&all);
            cross_check_tree(&tree, &naive, 50, 8, 50, &format!("random-order insert cap={cap}"));
        }
    }

    /// Explicitly targets duplicate insertion: re-inserting an existing
    /// point must be a no-op (`false` return, no structural change), not
    /// merely "the final query results happen to still be right."
    #[test]
    fn reinserting_existing_point_is_a_no_op() {
        let triples = uniform(20, 4, 20, 100, 55);
        let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledArena::build_dynamic(&triples, 64);
        let before = tree.node_word_count();
        for &(s, p, o) in &triples {
            assert!(!tree.insert(s, p, o), "re-inserting an existing point should return false");
        }
        assert_eq!(tree.node_word_count(), before, "re-inserting existing points must not change node count");
    }
}
