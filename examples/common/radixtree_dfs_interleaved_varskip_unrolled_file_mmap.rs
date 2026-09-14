//! A fifth storage variant, sitting between
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap`]
//! (anonymous `mmap`-backed blocks -- fast, but every touched page stays
//! resident in RAM forever, since `MADV_DONTNEED` on an anonymous mapping
//! *discards* its content) and
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile`]
//! (real file, hand-rolled `pread`/`pwrite`, near-zero RAM but a syscall
//! on every field access). This one asks: what if we get `mmap`'s
//! plain-memory-speed access for whatever's currently being traversed,
//! *and* bound resident memory by actively releasing pages once the
//! traversal has moved past them -- using `madvise(MADV_DONTNEED)` the
//! way it's actually meant to be used, over a *file-backed* mapping where
//! eviction is safe?
//!
//! # Why `MAP_SHARED` over a real file, not `MAP_ANONYMOUS`
//!
//! `MADV_DONTNEED`'s semantics differ sharply by mapping kind (see `man
//! 2 madvise`): on an anonymous mapping, or on a `MAP_PRIVATE` mapping
//! that has ever been written to (copy-on-write), the advised pages are
//! discarded and the *next* touch either zero-fills (anonymous) or
//! re-reads the mapping's *original, pre-write* file content
//! (`MAP_PRIVATE`) -- either way, any bytes this structure wrote after
//! the mapping was created would be silently lost. That rules out both
//! the sibling mmap variant's `MAP_ANONYMOUS` scheme and an ordinary
//! `MAP_PRIVATE` file mapping for a structure this file *mutates* (via
//! [`Self::insert`]). `MAP_SHARED` over a real, open file descriptor is
//! the one mode where this is safe: writes to the mapped memory go
//! straight into the page cache for that file (the same page cache a
//! `read()`/`write()` on the file would see), so `MADV_DONTNEED` there
//! just evicts the resident pages from *this process's* page tables --
//! the up-to-date content stays in the page cache (or gets written back
//! to disk if it's been evicted from there too), and the next touch
//! transparently re-faults it in, correctly, no data lost. See `man 2
//! mmap`'s `MAP_SHARED` section and `madvise`'s own note that
//! `MADV_DONTNEED` "will result in ... repopulating the memory contents
//! from the up-to-date contents of the underlying mapped file (shared
//! file mappings ...)" -- this is the exact guarantee the design leans
//! on.
//!
//! # Why this needs `unsafe` -- and why it's confined to this file (and
//! its sibling mmap variant)
//!
//! `mmap`/`mremap`/`munmap`/`madvise` are OS syscalls; there is no
//! safe-Rust equivalent. This project otherwise holds to zero `unsafe`
//! (confirmed via `grep -rn "unsafe" examples/`); the anonymous-mmap
//! variant was already an explicit, user-approved, scoped exception "for
//! this file" -- this file extends that same precedent to the one other
//! file in this project that structurally cannot exist without raw
//! `mmap`/`madvise` calls, for the same reason (there's no way to test
//! what file-backed `mmap` + `madvise` actually buys without it). Every
//! `unsafe` block below carries its own safety-invariant comment.
//!
//! # What changed from the anonymous-mmap variant
//!
//! Instead of one independent `mmap` per block (as the anonymous variant
//! and this file's own first draft both did), every block is a *view*
//! into **one single, shared `MAP_SHARED` mapping** of one backing file
//! (created via `open_backing_file`, unlinked immediately after creation
//! -- same pattern the pure file-backed variant uses, so its space is
//! reclaimed once every reference to it is gone). Addressing is
//! otherwise identical to
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_arena::RadixTreeDfsInterleavedVarSkipUnrolledArena`]'s
//! own scheme -- `slot_start`/`slot_capacity`/`slot_used_len` plus a
//! `blocks: Vec<usize>` logical-to-slot indirection -- with `base_ptr`
//! pointer arithmetic standing in for the arena's `Vec<u8>` indexing.
//! Growing the mapping (build time or [`Self::insert`]-triggered new
//! slot) uses [`std::fs::File::set_len`] to extend the file, then
//! `mremap(MREMAP_MAYMOVE)` to extend the mapping -- no page-alignment
//! constraint on individual slots this time (only the *file's* offset 0
//! needs alignment, trivially true), so slots pack exactly as tightly as
//! the arena variant's do, with none of that design's first draft's
//! per-block page-rounding waste.
//!
//! # Why a *single* mapping, not one per block: two measured problems
//! with the first draft
//!
//! An earlier version of this file gave each block its own independent
//! `mmap`, and called `madvise(MADV_DONTNEED)` once per evicted block.
//! Benchmarked against the other four variants across this project's
//! four large-scale scenarios, it was *never* faster than plain
//! anonymous mmap and was, in the worst case, ~700x slower on `contains`
//! -- two distinct, compounding problems, both fixed here:
//!
//! 1. **One `madvise` call per block is too many.** With
//!    `block_capacity = 65536` and millions of node records, a single
//!    row query can touch hundreds of blocks; one `madvise` syscall each
//!    dominated the cost of the (otherwise near-free) memory reads it
//!    was supposed to protect. [`Self::advise_behind`] now groups a
//!    contiguous run of logical blocks whose *physical* slots also
//!    happen to be byte-contiguous in the shared mapping (always true
//!    before any [`Self::insert`] has split anything, since slots are
//!    then allocated in the same order as `blocks`) into **one**
//!    `madvise` call spanning the whole run, instead of one call per
//!    block.
//! 2. **Evicting the root thrashes the single hottest block in the
//!    structure.** Every [`Self::contains`]/[`Self::query_rec`]/
//!    [`Self::insert`] call starts at `Pos::START` -- block 0 is read by
//!    *every single call*, regardless of which point it's about. The
//!    first draft's "evict everything when finishing" policy dutifully
//!    evicted block 0 at the end of every call, so the very next call
//!    immediately paid a page fault to bring the busiest block in the
//!    whole structure straight back -- constant thrashing on exactly the
//!    data that should never have left RAM. [`Self::hot_prefix_blocks`]
//!    now excludes a small fixed prefix of low block indices from
//!    eviction entirely (see its own doc comment for why one block is
//!    enough: only the *root's own record*, not a deep universally-hot
//!    path, is actually guaranteed touched by every query).
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
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Default for [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::hot_prefix_blocks`]
/// -- see that field's own doc comment.
const HOT_PREFIX_BLOCKS: usize = 1;

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Creates and opens a fresh backing file for one tree's node records,
/// then immediately unlinks it -- same pattern as
/// [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile`]'s
/// own `open_backing_file`: an already-open file descriptor (and, here,
/// the one shared mapping of it) keeps working after its directory entry
/// is removed, so the file's storage is freed once the last reference
/// (the `File` and the mapping) goes away -- the tree's own `Drop` impl
/// still needs to `munmap` that mapping explicitly, since unmapping
/// isn't automatic the way closing a `File` is, but nothing needs to
/// unlink or otherwise clean up the file itself.
fn open_backing_file() -> File {
    let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("s_graph_filemmap_{}_{id}.bin", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap_or_else(|e| panic!("failed to create backing file {path:?}: {e}"));
    let _ = std::fs::remove_file(&path);
    file
}

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::height(self)
    }
}

/// A position in the block-list: which block, and the byte offset within
/// it. Never lands mid-record -- every record (word, plus its skip value
/// if far) is fully contained in one block, so ordinary field reads
/// never need to consult [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::advance`]
/// at all; only *jumping to a different record* does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Pos {
    block: usize,
    offset: usize,
}

impl Pos {
    const START: Pos = Pos { block: 0, offset: 0 };
}

pub struct RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
    /// The one shared backing file the mapping is a window into -- see
    /// the module doc comment. Kept open for the tree's whole lifetime
    /// since inserts may need to grow it.
    file: File,
    /// Base address of the current `MAP_SHARED` mapping of `file`.
    /// Never cached across a call that might grow it (every access goes
    /// through `self.base_ptr` freshly) since [`Self::ensure_mapping_capacity`]
    /// may relocate it via `mremap(MREMAP_MAYMOVE)`.
    base_ptr: *mut u8,
    /// Current size of the mapping at `base_ptr` -- always `>= file_len`,
    /// grown via `mremap` in [`Self::ensure_mapping_capacity`].
    mapped_capacity: u64,
    /// Bytes of `file`/the mapping actually allocated to slots so far --
    /// always `<= mapped_capacity`. Unlike the first draft of this file,
    /// slots are packed exactly as tightly as the arena variant's
    /// (no per-slot page-rounding): only the mapping's own growth needs
    /// page-granularity reasoning, not individual slot placement.
    file_len: u64,
    /// `slot_start[s]`: byte offset (into the mapping) where slot `s`
    /// begins. Same role as
    /// [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_arena::RadixTreeDfsInterleavedVarSkipUnrolledArena`]'s
    /// own field of the same name.
    slot_start: Vec<u64>,
    /// `slot_capacity[s]`: how many bytes are reserved for slot `s`.
    /// Normally `block_capacity`, except for a slot created specifically
    /// to hold one oversized record/chain.
    slot_capacity: Vec<usize>,
    /// `slot_used_len[s]`: how many of slot `s`'s reserved bytes
    /// currently hold real content.
    slot_used_len: Vec<usize>,
    /// `blocks[i]` is the slot index holding the `i`-th block in
    /// DFS/logical order -- the same logical-to-physical indirection the
    /// arena variant uses, letting a split insert a new logical block
    /// via `Vec::insert` on this small vector without needing physical
    /// slot placement to match logical order.
    blocks: Vec<usize>,
    /// How far [`Self::advise_behind`] has already evicted within the
    /// *current* top-level call -- see that method's doc comment.
    advised_up_to_block: Cell<usize>,
    /// Number of low logical block indices [`Self::advise_behind`] never
    /// evicts, regardless of how far the traversal has progressed --
    /// see the module doc comment's "two measured problems" section.
    /// One is enough: only the root's own record (always in block 0) is
    /// actually guaranteed touched by *every* query: everything past the
    /// first divergence point depends on which child a specific query's
    /// `s`/`p`/`o` selects, so there is no deeper universally-hot path to
    /// protect, and protecting more than block 0 would just pin
    /// arbitrary extra data in RAM for no measured benefit.
    hot_prefix_blocks: usize,
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

// SAFETY: this struct owns its mapping exclusively -- nothing else ever
// holds a pointer into it -- so moving one between threads is exactly as
// sound as moving a `Vec<u8>` (which also wraps a raw allocation).
unsafe impl Send for RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {}

impl Drop for RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
    fn drop(&mut self) {
        // SAFETY: `base_ptr`/`mapped_capacity` are exactly the address
        // and size of the mapping most recently established by `mmap`
        // (build time) or `mremap` (growth), never mutated outside those
        // two call sites, and `drop` runs at most once (the standard
        // `Drop` guarantee) -- so this never double-unmaps or unmaps
        // memory this struct doesn't own. `mapped_capacity == 0` only
        // for a struct built via the degenerate/empty-tree path, which
        // never calls `mmap` at all -- guarded explicitly since
        // `munmap(ptr, 0)` is not a meaningful call.
        if self.mapped_capacity > 0 {
            unsafe {
                libc::munmap(self.base_ptr as *mut libc::c_void, self.mapped_capacity as usize);
            }
        }
    }
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
/// grown one point at a time via [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::insert`].
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

impl RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
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
            return RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
                file: open_backing_file(),
                base_ptr: std::ptr::null_mut(),
                mapped_capacity: 0,
                file_len: 0,
                slot_start: Vec::new(),
                slot_capacity: Vec::new(),
                slot_used_len: Vec::new(),
                blocks: Vec::new(),
                advised_up_to_block: Cell::new(0),
                hot_prefix_blocks: HOT_PREFIX_BLOCKS,
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

        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
            file: chopped.file,
            base_ptr: chopped.base_ptr,
            mapped_capacity: chopped.mapped_capacity,
            file_len: chopped.file_len,
            slot_start: chopped.slot_start,
            slot_capacity: chopped.slot_capacity,
            slot_used_len: chopped.slot_used_len,
            blocks: chopped.blocks,
            advised_up_to_block: Cell::new(0),
            hot_prefix_blocks: HOT_PREFIX_BLOCKS,
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

    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height as usize {
            WORD_RECORD_BYTES + self.level_skip_bytes[level] as usize
        } else {
            WORD_RECORD_BYTES
        }
    }

    /// Grows the mapping to at least `needed` bytes if it isn't already
    /// that large: extends the file first (`File::set_len`, metadata
    /// only), then `mremap(MREMAP_MAYMOVE)`s the mapping to match. Exact
    /// growth (not amortized like `Vec::resize`'s doubling): for a
    /// `MAP_SHARED` file-backed mapping, growing in place (or relocating
    /// to a fresh address if the kernel can't extend it in place) never
    /// needs to *copy* any byte's content -- the pages are already
    /// backed by the file, `mremap` just needs new page-table entries --
    /// so there's no reallocation-copy cost here to amortize away the
    /// way there is for the arena variant's `Vec<u8>`.
    fn ensure_mapping_capacity(&mut self, needed: u64) {
        if needed <= self.mapped_capacity {
            return;
        }
        self.file.set_len(needed).expect("failed to extend backing file");
        let new_ptr = if self.mapped_capacity == 0 {
            // SAFETY: `self.file` is a valid, open file descriptor now
            // covering at least `[0, needed)` (just extended above);
            // `MAP_SHARED` + a real file is required for
            // `madvise(MADV_DONTNEED)` eviction to be safe (see the
            // module doc comment); the returned pointer is checked
            // immediately below before being trusted for anything.
            unsafe { libc::mmap(std::ptr::null_mut(), needed as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, self.file.as_raw_fd(), 0) }
        } else {
            // SAFETY: `self.base_ptr` is the address of the mapping most
            // recently established by `mmap`/`mremap` (invariant this
            // method alone maintains), `self.mapped_capacity` is exactly
            // that mapping's current size, and `self.file` now covers at
            // least `[0, needed)` -- `mremap`'s contract is satisfied.
            // `MREMAP_MAYMOVE` permits the kernel to relocate the
            // mapping if it can't grow in place, which is why `base_ptr`
            // is always re-read from `self` rather than cached across a
            // call that might trigger this.
            unsafe { libc::mremap(self.base_ptr as *mut libc::c_void, self.mapped_capacity as usize, needed as usize, libc::MREMAP_MAYMOVE) }
        };
        assert_ne!(new_ptr, libc::MAP_FAILED, "{}({needed} bytes) failed: {}", if self.mapped_capacity == 0 { "mmap" } else { "mremap" }, std::io::Error::last_os_error());
        self.base_ptr = new_ptr as *mut u8;
        self.mapped_capacity = needed;
    }

    /// Appends a fresh slot of `capacity` bytes at the end of the used
    /// region, growing the mapping first if needed -- the file-mmap
    /// analogue of the arena variant's own `allocate_slot`.
    fn allocate_slot(&mut self, capacity: usize) -> usize {
        let slot = self.slot_capacity.len();
        let start = self.file_len;
        let needed = start + capacity as u64;
        self.ensure_mapping_capacity(needed);
        self.file_len = needed;
        self.slot_start.push(start);
        self.slot_capacity.push(capacity);
        self.slot_used_len.push(0);
        slot
    }

    #[inline]
    fn slot_slice(&self, slot: usize) -> &[u8] {
        let start = self.slot_start[slot] as usize;
        let len = self.slot_used_len[slot];
        // SAFETY: `base_ptr` maps at least `mapped_capacity >= file_len`
        // bytes for the mapping currently in effect, and
        // `start + len <= file_len` is an invariant every mutating
        // method maintains -- so `[base_ptr+start, base_ptr+start+len)`
        // is always a valid, initialized range. A prior
        // `madvise(MADV_DONTNEED)` (see `advise_behind`) doesn't change
        // this: the mapping stays valid, and a touch here transparently
        // re-faults the (correct, file-backed) content in.
        unsafe { std::slice::from_raw_parts(self.base_ptr.add(start), len) }
    }

    /// In-place insertion at `offset` within `slot`: shifts the existing
    /// `[offset, used)` tail right by `bytes.len()`, then writes `bytes`
    /// into the gap. Panics (never reallocates a slot in place) if it
    /// wouldn't fit.
    fn slot_insert_at(&mut self, slot: usize, offset: usize, bytes: &[u8]) {
        let start = self.slot_start[slot] as usize;
        let used = self.slot_used_len[slot];
        debug_assert!(offset <= used, "slot_insert_at offset {offset} beyond slot's used length {used}");
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        // SAFETY: both the source range `[start+offset, start+used)` and
        // destination range `[start+offset+bytes.len(), start+used+bytes.len())`
        // are within the mapped region (just asserted); `copy` (not
        // `copy_nonoverlapping`) since those ranges can genuinely
        // overlap when `bytes.len() < used - offset`. The second copy
        // writes `bytes` into the now-vacated gap, which cannot overlap
        // `bytes` itself (a distinct allocation).
        unsafe {
            std::ptr::copy(self.base_ptr.add(start + offset), self.base_ptr.add(start + offset + bytes.len()), used - offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base_ptr.add(start + offset), bytes.len());
        }
        self.slot_used_len[slot] = used + bytes.len();
    }

    /// Appends `bytes` at the end of `slot`'s used region. Panics rather
    /// than reallocating if it wouldn't fit.
    fn slot_extend_from_slice(&mut self, slot: usize, bytes: &[u8]) {
        let start = self.slot_start[slot] as usize;
        let used = self.slot_used_len[slot];
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        // SAFETY: destination range `[start+used, start+used+bytes.len())`
        // is within the mapped region (just asserted), doesn't overlap
        // `bytes` (a distinct allocation).
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base_ptr.add(start + used), bytes.len());
        }
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
        let start = self.slot_start[slot] as usize + offset;
        // SAFETY: `[start, start+8)` is within the slot's reserved
        // capacity (every caller writes only within a record it already
        // validated the size of).
        unsafe {
            std::ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), self.base_ptr.add(start), 8);
        }
    }

    fn block_write_var_at(&mut self, logical: usize, offset: usize, width: u8, val: u64) {
        let slot = self.blocks[logical];
        let start = self.slot_start[slot] as usize;
        let capacity = self.slot_capacity[slot];
        // SAFETY: `[start, start+capacity)` is this slot's whole
        // reserved (mapped) region; `write_var_at` only ever writes
        // within `[offset, offset+width)`, itself within that region by
        // the same per-level-skip-width bound every sibling variant
        // relies on.
        let buf = unsafe { std::slice::from_raw_parts_mut(self.base_ptr.add(start), capacity) };
        write_var_at(buf, offset, width, val);
    }

    /// Advises away (`madvise(MADV_DONTNEED)`) every block in
    /// `[max(advised_up_to_block, hot_prefix_blocks), upto)`, then
    /// records `upto` as the new high-water mark -- see the module doc
    /// comment's "two measured problems" section for why both the
    /// `hot_prefix_blocks` floor and the batching below exist; without
    /// them this was measured to be up to ~700x *slower* than no
    /// eviction at all, not a win.
    ///
    /// Called two ways: with the *current* position's block at the top
    /// of [`Self::contains`]/[`Self::query_rec`]/[`Self::insert`]'s main
    /// descent loop (evicting everything strictly behind the position
    /// we're about to read -- covers both forward descent and
    /// backtrack-resume, since both reach this same call site), and with
    /// `self.blocks.len()` right before each of those methods' terminal
    /// return points (flushing whatever is left -- "when finishing").
    /// [`Self::insert`]'s ancestor-skip-update pass is the one
    /// intentional exception: it revisits `trail` positions *after* the
    /// main descent has already advised them away, so those specific
    /// reads/writes pay one extra re-fault each -- accepted rather than
    /// special-cased, since inserts are rare next to the read-heavy
    /// workloads this eviction scheme targets.
    fn advise_behind(&self, upto: usize) {
        let already = self.advised_up_to_block.get();
        let from = already.max(self.hot_prefix_blocks);
        let capped = upto.min(self.blocks.len());
        if capped <= from {
            return;
        }
        // Group [from, capped) into maximal runs of physically-contiguous
        // slots -- always the *entire* range in one run before any
        // insert has split a block (slots are allocated in the same
        // order as `blocks` then), since [`Self::allocate_slot`] always
        // appends -- and issue one `madvise` per run instead of one per
        // block, which is what makes calling this every outer-loop
        // iteration (not just at coarse checkpoints) actually cheap.
        let mut run_start = self.slot_start[self.blocks[from]];
        let mut run_end = run_start + self.slot_capacity[self.blocks[from]] as u64;
        for i in (from + 1)..capped {
            let slot = self.blocks[i];
            let byte_start = self.slot_start[slot];
            if byte_start == run_end {
                run_end += self.slot_capacity[slot] as u64;
            } else {
                self.advise_range(run_start, run_end - run_start);
                run_start = byte_start;
                run_end = byte_start + self.slot_capacity[slot] as u64;
            }
        }
        self.advise_range(run_start, run_end - run_start);
        self.advised_up_to_block.set(capped);
    }

    /// Advises the kernel that `[byte_offset, byte_offset + len)` of the
    /// mapping is no longer needed in this process's resident set right
    /// now -- see the module doc comment for why this is safe (no data
    /// loss) specifically because this mapping is `MAP_SHARED` over a
    /// real file. Advisory only: unlike every other syscall in this
    /// file, a failure here costs nothing but a missed optimization (the
    /// pages simply stay resident), so it does not panic on error.
    fn advise_range(&self, byte_offset: u64, len: u64) {
        // SAFETY: `[base_ptr+byte_offset, base_ptr+byte_offset+len)` is
        // within the current mapping (every caller derives the range
        // from `slot_start`/`slot_capacity`, which are always within
        // `[0, file_len) <= mapped_capacity`), still alive (this method
        // never runs after `Drop`), and `madvise(MADV_DONTNEED)` never
        // invalidates the mapping itself (only its resident pages) -- so
        // this struct's own invariants hold whether the call succeeds or
        // not; the return value is deliberately ignored for exactly that
        // reason.
        unsafe {
            libc::madvise(self.base_ptr.add(byte_offset as usize) as *mut libc::c_void, len as usize, libc::MADV_DONTNEED);
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
        self.advised_up_to_block.set(0);
        for level in 0..self.height as usize {
            self.advise_behind(pos.block);
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
                self.advise_behind(self.blocks.len());
                return false;
            }
            if is_last {
                self.advise_behind(self.blocks.len());
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
        self.advised_up_to_block.set(0);

        'outer: loop {
            self.advise_behind(pos.block);
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
                    self.advise_behind(self.blocks.len());
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
        self.advised_up_to_block.set(0);

        for level in 0..self.height as usize {
            self.advise_behind(pos.block);
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
                self.advise_behind(self.blocks.len());
                return true;
            }
            if is_last {
                self.advise_behind(self.blocks.len());
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
    /// *fixed* capacity for its whole lifetime -- it can never grow past
    /// `slot_capacity[slot]` in place, not even for a tail append. So
    /// the only two cases are: it fits in place (`slot_insert_at`, an
    /// in-slot shift), or it doesn't, in which case `new_bytes` always
    /// gets its own fresh slot rather than trying to force it into an
    /// existing one -- splitting the slot at `pos.offset` first (an
    /// empty, effectively-abandoned tail slot for the common tail-append
    /// case, since `is_tail` skips allocating one at all; a real second
    /// half otherwise), then inserting the new slot between the two
    /// halves via `Vec::insert` on the small `blocks` indirection
    /// vector -- exactly the arena variant's own `insert_bytes` scheme.
    /// Every position recorded *before* `pos` in the same
    /// [`Self::insert`] call stays valid: a split only ever moves bytes
    /// at-or-after `pos.offset` into new slots referenced after
    /// `pos.block` in `blocks`, and every earlier position is (by
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
            let tail_start = self.slot_start[slot] as usize + pos.offset;
            let mut tail_buf = vec![0u8; tail_len];
            // SAFETY: `[tail_start, tail_start+tail_len)` is within the
            // mapped region (part of `slot`'s already-used bytes).
            unsafe {
                std::ptr::copy_nonoverlapping(self.base_ptr.add(tail_start), tail_buf.as_mut_ptr(), tail_len);
            }
            let new_slot = self.allocate_slot(capacity);
            self.slot_extend_from_slice(new_slot, &tail_buf);
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
        lo = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
    }
    debug_assert_eq!(chain.len() as u64, sizes[start_level], "chain length must match the precomputed size it reports to its own parent");
    chain
}

/// The output of [`chop_into_blocks`]: a fresh backing file, already
/// mapped and filled with every node record packed into slots, ready to
/// become a [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap`]'s fields
/// directly.
struct ChoppedBlocks {
    file: File,
    base_ptr: *mut u8,
    mapped_capacity: u64,
    file_len: u64,
    slot_start: Vec<u64>,
    slot_capacity: Vec<usize>,
    slot_used_len: Vec<usize>,
    blocks: Vec<usize>,
}

/// Dry-run of `chop_into_blocks`'s own greedy-packing decision (see
/// there) -- identical in spirit to
/// [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_arena`]'s
/// own `plan_slot_lengths`, duplicated here rather than shared since
/// every variant in this exploration stays self-contained.
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

/// Chops `flat` into slots of at most `capacity` bytes, never splitting a
/// record (using `record_starts`, DFS-order offsets collected by
/// [`build_interleaved_rec`]) -- a single record larger than `capacity`
/// still gets its own (oversized) slot rather than being split, since
/// splitting mid-record would corrupt it. Two passes, exactly like the
/// arena variant's own `chop_into_blocks`: [`plan_slot_lengths`] first
/// decides exactly how many slots of what sizes are needed, then this
/// function re-walks the same records into one in-RAM scratch buffer,
/// and only then creates the backing file (sized exactly once) and its
/// single `MAP_SHARED` mapping, copying the whole scratch buffer into it
/// in one pass.
fn chop_into_blocks(flat: &[u8], record_starts: &[usize], capacity: usize) -> ChoppedBlocks {
    let slot_capacities = plan_slot_lengths(record_starts, flat.len(), capacity);
    let mut slot_start = Vec::with_capacity(slot_capacities.len());
    let mut total: u64 = 0;
    for &cap in &slot_capacities {
        slot_start.push(total);
        total += cap as u64;
    }
    let mut scratch = vec![0u8; total as usize];
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
        let dest = slot_start[slot_idx] as usize + current_len;
        scratch[dest..dest + record.len()].copy_from_slice(record);
        current_len += record.len();
        slot_used_len[slot_idx] = current_len;
        if record.len() > capacity {
            slot_idx += 1;
            current_len = 0;
        }
    }

    let file = open_backing_file();
    let (base_ptr, mapped_capacity) = if total > 0 {
        file.set_len(total).expect("failed to size backing file");
        // Populate the file via a plain `pwrite` *before* mapping it --
        // this is the "write-all fix": writing through a `write()`-family
        // syscall lands in the kernel's page cache, which is not charged
        // to this process's RSS, whereas the mmap-then-memcpy this
        // function used to do touches every page through this process's
        // own mapping, charging the *entire* structure to RSS the moment
        // it's built (measured to make this variant's peak RSS the
        // highest of the four in this comparison, ~2.79 GiB vs. ~2.24
        // GiB for the plain file/anon-mmap/heap variants on the largest
        // scenario). `mmap`, called only afterward, does not itself fault
        // in any pages -- reads/writes through it will lazily pull pages
        // from the (now correctly populated) page cache on first touch,
        // exactly like the plain file-backed variant's own build path.
        file.write_all_at(&scratch, 0).expect("failed to bulk-write backing file");
        // SAFETY: `file` is a valid, open file descriptor now covering
        // `[0, total)` (just written above); `MAP_SHARED` is required for
        // `madvise(MADV_DONTNEED)` eviction to be safe (see the module
        // doc comment); the returned pointer is checked immediately
        // below.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), total as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap({total} bytes) failed: {}", std::io::Error::last_os_error());
        (ptr as *mut u8, total)
    } else {
        (std::ptr::null_mut(), 0)
    };

    let blocks: Vec<usize> = (0..slot_capacities.len()).collect();
    ChoppedBlocks { file, base_ptr, mapped_capacity, file_len: total, slot_start, slot_capacity: slot_capacities, slot_used_len, blocks }
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
                let child_lo = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
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

impl MemSize for RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
    /// **Not a faithful resident-memory number for this particular
    /// variant** -- unlike the sibling anonymous-mmap variant (where
    /// summing each block's mapped `capacity` is a reasonable proxy for
    /// what's actually resident, since nothing there is ever evicted),
    /// this variant *actively* releases pages via `madvise(MADV_DONTNEED)`
    /// (see [`Self::advise_behind`]), so a block's `capacity` measures
    /// how much *virtual address space* it reserves, not how much
    /// physical RAM it holds at any given moment -- that number
    /// legitimately varies call to call. Kept here anyway, for the same
    /// "allocated capacity" convention every other variant reports, but
    /// the number this variant's whole design is meant to be judged on
    /// is peak RSS (`VmHWM`, via [`crate::common::metrics::peak_rss_kb`]),
    /// measured in its own isolated process -- see `examples/rss_file_mmap.rs`.
    fn mem_size_bytes(&self) -> usize {
        self.mapped_capacity as usize
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

impl RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("mapped_capacity (virtual, not necessarily resident)", self.mapped_capacity as usize),
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

    /// Serializes this structure to a self-contained byte buffer -- see
    /// `crate::common::tree_serialize`'s own doc comment for the format
    /// and why it exists (letting `examples/phase_file_mmap.rs` separate
    /// build/query/insert into isolated processes, each with its own
    /// honest peak-RSS reading). Reads each logical block via
    /// [`Self::block_slice`] -- plain memory access through the mapping,
    /// no syscall per block the way the pure file-backed variant's own
    /// `serialize` needs.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        crate::common::tree_serialize::write_header(
            &mut out,
            &crate::common::tree_serialize::Header {
                height: self.height,
                node_count: self.node_count,
                block_capacity: self.block_capacity,
                dimension_sizes: self.dimension_sizes,
                dim_order: self.dim_order,
                single_point_exists: self.single_point_exists,
                level_dim_bits: self.level_dim_bits.clone(),
                level_skip_bytes: self.level_skip_bytes.clone(),
                level_shifts_cache: self.level_shifts_cache.clone(),
                level_child_cell_bits_cache: self.level_child_cell_bits_cache.clone(),
            },
        );
        let n = self.blocks.len();
        let capacities: Vec<usize> = (0..n).map(|i| self.slot_capacity[self.blocks[i]]).collect();
        let used_lens: Vec<usize> = (0..n).map(|i| self.slot_used_len[self.blocks[i]]).collect();
        crate::common::tree_serialize::write_block_table(&mut out, &capacities, &used_lens);
        for i in 0..n {
            out.extend_from_slice(self.block_slice(i));
        }
        out
    }

    /// Reconstructs a structure from bytes already in memory (written by
    /// [`Self::serialize`]) -- a thin wrapper over
    /// [`Self::deserialize_from_reader`] via [`std::io::Cursor`]. Prefer
    /// [`Self::deserialize_from_path`] when reading a file.
    pub fn deserialize(buf: &[u8]) -> Self {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf))
    }

    /// Reconstructs a structure by reading directly from a file written
    /// by [`Self::serialize`]: streams each block straight from the
    /// source file into a fresh backing file (one reused, block-sized
    /// `Vec<u8>`, never the whole content at once -- see
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize_from_path`]'s
    /// doc comment for the general shape), and only then `mmap`s that
    /// file -- the same write-before-map ordering [`chop_into_blocks`]
    /// uses (see the module doc comment's "two measured problems"
    /// section) so this doesn't reintroduce the build-time RSS spike
    /// that ordering was specifically changed to avoid.
    pub fn deserialize_from_path(path: &std::path::Path) -> Self {
        let file_in = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path:?}: {e}"));
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file_in))
    }

    fn deserialize_from_reader<R: std::io::Read>(r: &mut R) -> Self {
        let h = crate::common::tree_serialize::read_header_from(r);
        let table = crate::common::tree_serialize::read_block_table_from(r);
        let mut slot_start = Vec::with_capacity(table.len());
        let mut total: u64 = 0;
        for &(cap, _) in &table {
            slot_start.push(total);
            total += cap as u64;
        }

        let file = open_backing_file();
        let mut slot_used_len = vec![0usize; table.len()];
        if total > 0 {
            file.set_len(total).expect("failed to size backing file");
        }
        let mut xfer = Vec::new();
        for (i, &(_, len)) in table.iter().enumerate() {
            xfer.resize(len, 0);
            r.read_exact(&mut xfer).expect("unexpected EOF while reading serialized tree content");
            if len > 0 {
                file.write_all_at(&xfer, slot_start[i]).expect("failed to write block content");
            }
            slot_used_len[i] = len;
        }
        let (base_ptr, mapped_capacity) = if total > 0 {
            // SAFETY: `file` is a valid, open file descriptor now
            // covering `[0, total)` (just written above); `MAP_SHARED`
            // is required for `madvise(MADV_DONTNEED)` eviction to be
            // safe (see the module doc comment); the returned pointer is
            // checked immediately below.
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), total as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0) };
            assert_ne!(ptr, libc::MAP_FAILED, "mmap({total} bytes) failed: {}", std::io::Error::last_os_error());
            (ptr as *mut u8, total)
        } else {
            (std::ptr::null_mut(), 0)
        };

        let slot_capacity: Vec<usize> = table.iter().map(|&(cap, _)| cap).collect();
        let blocks: Vec<usize> = (0..table.len()).collect();

        RadixTreeDfsInterleavedVarSkipUnrolledFileMmap {
            file,
            base_ptr,
            mapped_capacity,
            file_len: total,
            slot_start,
            slot_capacity,
            slot_used_len,
            blocks,
            advised_up_to_block: Cell::new(0),
            hot_prefix_blocks: HOT_PREFIX_BLOCKS,
            block_capacity: h.block_capacity,
            node_count: h.node_count,
            level_skip_bytes: h.level_skip_bytes,
            level_dim_bits: h.level_dim_bits,
            level_shifts_cache: h.level_shifts_cache,
            level_child_cell_bits_cache: h.level_child_cell_bits_cache,
            dimension_sizes: h.dimension_sizes,
            dim_order: h.dim_order,
            single_point_exists: h.single_point_exists,
            height: h.height,
        }
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

    fn cross_check_tree(tree: &RadixTreeDfsInterleavedVarSkipUnrolledFileMmap, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build(triples, split, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?} cap={cap}"));
        }
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(triples, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("dynamic cap={cap}"));
        }
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split(), cap);
            assert!(!tree.contains(0, 0, 0));
            assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        }
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split(), cap);
            assert!(tree.contains(0, 0, 0));
            assert!(!tree.contains(0, 0, 1));
        }
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split());
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    #[test]
    fn deep_far_nesting_navigates_correctly() {
        let triples = uniform(500, 20, 500, 6000, 999);
        let naive = NaiveIndex::build(&triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(&triples, cap);
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

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(initial, cap);
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

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(initial, cap);
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
        let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(&triples, 64);
        let before = tree.node_word_count();
        for &(s, p, o) in &triples {
            assert!(!tree.insert(s, p, o), "re-inserting an existing point should return false");
        }
        assert_eq!(tree.node_word_count(), before, "re-inserting existing points must not change node count");
    }

    /// Round-trips through [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::serialize`]/
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::deserialize`],
    /// both right after build and again after a batch of inserts (so
    /// reconstructed block capacity, not just content, is exercised).
    #[test]
    fn serialize_round_trip() {
        for cap in TEST_CAPACITIES {
            let all = uniform(30, 6, 30, 400, 8080 + cap as u64);
            let split_at = all.len() / 2;
            let (initial, rest) = all.split_at(split_at);
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::build_dynamic(initial, cap);

            let bytes = tree.serialize();
            let mut restored = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::deserialize(&bytes);
            let naive = NaiveIndex::build(initial);
            cross_check_tree(&restored, &naive, 30, 6, 30, &format!("round-trip right after build, cap={cap}"));

            let mut seen: Vec<Triple> = initial.to_vec();
            for &(s, p, o) in rest {
                if restored.insert(s, p, o) {
                    seen.push((s, p, o));
                }
            }
            let naive_after_insert = NaiveIndex::build(&seen);
            cross_check_tree(&restored, &naive_after_insert, 30, 6, 30, &format!("round-trip after post-deserialize inserts, cap={cap}"));

            let bytes2 = restored.serialize();
            let restored2 = RadixTreeDfsInterleavedVarSkipUnrolledFileMmap::deserialize(&bytes2);
            cross_check_tree(&restored2, &naive_after_insert, 30, 6, 30, &format!("second round-trip, cap={cap}"));
        }
    }
}
