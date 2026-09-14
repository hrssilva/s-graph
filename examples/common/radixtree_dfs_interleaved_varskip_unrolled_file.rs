//! A fourth storage variant alongside
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled`]
//! (heap-backed blocks, one `Vec<u8>` allocation each),
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_mmap::RadixTreeDfsInterleavedVarSkipUnrolledMmap`]
//! (`mmap`-backed blocks, one OS mapping each, `unsafe`), and
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_arena::RadixTreeDfsInterleavedVarSkipUnrolledArena`]
//! (one shared in-RAM `Vec<u8>` arena, safe Rust). This one asks a
//! different question: what if the node records live on *disk*, not in
//! RAM at all, read and written with plain `File` I/O -- no `mmap`, no
//! in-process byte buffer, every touch a real syscall?
//!
//! # The design: one backing file, fixed-size slots, hand-rolled I/O
//!
//! The layout is otherwise identical to the arena variant's: one shared
//! backing store divided into fixed-`block_capacity`-sized slots,
//! addressed by `slot_start[slot]` (almost always `slot * block_capacity`,
//! except for the rare oversized slot -- see `slot_capacity`), with
//! `blocks: Vec<usize>` mapping logical (DFS-order) block index to
//! physical slot index exactly as before. The only real difference is
//! *where that backing store lives*: instead of `arena: Vec<u8>`, this
//! file has `file: std::fs::File`, and every read or write of a node's
//! bytes goes through [`std::os::unix::fs::FileExt::read_exact_at`] /
//! `write_all_at` -- explicit `pread`/`pwrite` calls at a byte offset,
//! safe (they take the offset as a plain argument, no seek-then-read
//! race), but genuinely hitting the kernel's file I/O path, unlike
//! `mmap`'s page-fault-once-then-plain-memory-access model. This is the
//! "by hand" half of the exercise.
//!
//! # Fetching in page-sized chunks, not one node at a time
//!
//! The first version of this file issued one `pread` per field read --
//! 8 bytes at a time for every node word, a handful for every skip
//! value -- even though a single `block_capacity`-sized slot (65536
//! bytes by default) typically holds thousands of tiny node records. The
//! obvious fix -- a single-entry cache holding one *whole slot* at a
//! time, refetched only when a read targets a different slot -- was
//! tried first and *measured*, not assumed: across every benchmarked
//! scenario it made row queries (which scan many siblings of one parent
//! consecutively via repeated [`Self::child_span_end`] calls, genuinely
//! reusing the same slot many times in a row) 9-17x faster, but made
//! `contains` and `insert` 2-6.5x *slower*, because those touch exactly
//! one 8-byte word per tree level with little same-slot reuse across
//! levels, and paying for a full up-to-64KB `pread` when 8 bytes were
//! needed was pure waste.
//!
//! [`Self::block_stack`] instead holds a [`READ_CHUNK_BYTES`]-sized
//! (4096, one typical OS page), chunk-aligned slice of one slot,
//! refilled with a single `pread` only when a read falls outside the
//! currently cached chunk (see [`Self::ensure_stack_entry`]), and kept
//! in sync with any write to a cached slot (see [`Self::slot_write_all`],
//! which patches the cached bytes in place rather than merely
//! invalidating them) so a stale cached copy is never served. A
//! branching node's children comfortably fit within a few KB, so this
//! keeps most of the whole-slot cache's sibling-scan win while bounding
//! how much a low-locality single-field read (the common case for
//! `contains`'s per-level descent) can overpay -- see
//! [`READ_CHUNK_BYTES`]'s own doc comment for the exact numbers this
//! tradeoff was tuned against. [`Self::slot_insert_at`]'s tail-move
//! (already reading/writing a multi-byte contiguous span in one call,
//! not one field at a time) bypasses the cache entirely via
//! [`Self::slot_read_exact`]/[`Self::slot_write_all`] directly -- a
//! cache buys nothing there, since that read is already as batched as
//! it can be.
//!
//! # A stack of cached blocks, not one sliding slot
//!
//! A single sliding cache (one slot, replaced wholesale on every miss)
//! has two distinct costs. First: every top-level call
//! ([`Self::contains`]/[`Self::query_rec`]/[`Self::insert`]) starts at
//! `Pos::START`, reading the root's record first, but ends wherever its
//! own traversal last touched -- almost never back at the root's own
//! chunk -- so a single cache slot would force a fresh `pread` for the
//! root at the *start* of literally every call, the same "evict the one
//! thing every call needs" pattern this project already found and fixed
//! once, in the file-mmap variant's `madvise` eviction policy (see that
//! file's module doc comment's "two measured problems" section). Second,
//! more general: *any* backtrack during a query or insert -- returning
//! to an ancestor already read earlier in the same call, then descending
//! into a different child -- re-evicts and re-fetches that ancestor's
//! block under a one-slot cache, even though its bytes were resident
//! moments ago. [`Self::insert_new_branch`]'s own ancestor-skip-update
//! pass is the sharpest example: it deliberately revisits every position
//! on the path from the root down to the insertion point a second time,
//! to bump each ancestor's stored skip value.
//!
//! [`Self::block_stack`] fixes both at once by mirroring the actual
//! shape of a DFS descent instead of caching a single "current" slot. It
//! is a stack of [`StackCacheEntry`], one per block currently *open* on
//! the path from the root to wherever the traversal is right now.
//! [`Self::ensure_stack_entry`] -- the single place every read or write
//! goes through -- reacts to whichever slot a call names next: if it's
//! already the top entry, nothing changes; if it's a backtrack (the slot
//! is already on the stack, just not at the top -- true by construction,
//! since a block only ever leaves the stack once the traversal has fully
//! left its subtree), every entry above it is popped, exposing the
//! already-resident bytes with no `pread` at all; otherwise it's
//! genuinely new territory and gets pushed fresh. No traversal function
//! calls push/pop explicitly -- the policy falls entirely out of which
//! slot each read/write already names, so none of `contains`/
//! `query_rec`/`insert`'s own control flow had to change to get this.
//!
//! Entry 0 is always the root's block, and it is the one entry
//! [`StackCacheGuard`] -- an RAII guard constructed at the top of every
//! top-level call, popping the stack back down to just that entry when
//! the call returns (on every exit path, including early returns) --
//! never removes, giving the root the same permanent residency a
//! separate, dedicated cache would, without needing one: this is sound
//! specifically because `blocks[0]`'s slot never changes for the life of
//! the tree -- [`Self::insert_bytes`]'s splits only ever insert new
//! slots *after* logical position 0, never replace it. Everything above
//! entry 0 is scratch for the *current* call only, popped off at its end
//! so nothing about one call's descent path lingers to confuse the next.
//!
//! "Popped" here means the [`BlockStack`] sense, not `Vec::truncate`:
//! the entries above the new top stay physically allocated rather than
//! being dropped, so a call that pushes back to the same depth a later
//! call also reaches doesn't pay for a fresh heap allocation -- see
//! [`BlockStack`]'s own doc comment.
//!
//! [`Self::insert_new_branch`]'s ancestor-skip-update loop walks `trail`
//! in *reverse* (leaf-most ancestor first, root last) specifically to
//! keep this a genuine stack discipline -- monotonically popping down
//! from wherever the insert's forward descent left the stack, instead of
//! walking root-to-leaf, which would pop everything down to the root on
//! its first step and then have to re-push every deeper ancestor right
//! back, defeating the cache entirely. Each ancestor's skip-value update
//! is independent of every other's, so the reversed order changes
//! nothing about correctness.
//!
//! # A depth that depends on who's asking
//!
//! An unbounded stack turned out to be the right shape for
//! [`Self::insert`] but not for reads. Measured separately:
//! [`Self::contains`] never backtracks at all (a single top-down walk),
//! so its only win is the permanent root entry -- nothing a deeper stack
//! adds. [`Self::query_rec`]'s own `ConstrainedFrame` backtrack resumes
//! through a `child_pos` already computed (via skip pointers) *before*
//! descending into the sibling being backtracked from -- it never
//! re-reads the ancestor's own bytes on resume, so there is no
//! repeated-access pattern for a deeper stack to serve there either;
//! measured row-query latency was flat (within noise) with the
//! unbounded stack, and a touch worse than the original one-slot cache,
//! from the extra `Vec` search [`Self::ensure_stack_entry`] does on every
//! push. Only [`Self::insert_new_branch`]'s ancestor-skip-update pass
//! has a genuine, guaranteed repeated-block-access pattern (see above).
//!
//! So [`Self::stack_cap`] makes the depth a per-call choice instead of a
//! fixed one: `1` for [`Self::contains`]/[`Self::query_two_bound`]/
//! [`Self::query_one_bound`], reproducing the original single
//! [`READ_CHUNK_BYTES`] sliding window (plus the permanent root) exactly
//! -- [`Self::ensure_stack_entry`] evicts back down to just the root
//! before any push once that one non-root slot is already occupied, so
//! at most two entries ever coexist -- and `MAX_HEIGHT` (effectively
//! unbounded) for [`Self::insert`], keeping the full stack described
//! above. Same code path, same struct, no duplicated cache logic:
//! [`Self::stack_cap`] is the only thing that differs between the two
//! modes.
//!
//! Because [`BlockStack`] never actually frees a popped entry (see
//! above), `stack_cap` `1` isn't just behaviorally identical to the
//! original one-slot cache -- after the very first top-level call has
//! run once, it's allocation-identical too: `top` only ever oscillates
//! between `1` and `2`, both physical slots already exist, and every
//! later call's push is an in-place overwrite plus (at most) one
//! `Vec::resize` that reuses existing capacity. Read-only calls pay
//! nothing extra for having a `Vec` of entries instead of two plain
//! fields.
//!
//! The **bulk build** is still done as one efficient pass, not one
//! syscall per record: [`plan_slot_lengths`] dry-runs the same
//! greedy-packing decision `chop_into_blocks` uses to size everything
//! exactly, the whole slotted layout is assembled in one scratch
//! `Vec<u8>` in RAM, and only then is it handed to the backing file in a
//! *single* `write_all_at` call. Only the per-block bookkeeping
//! (`slot_start`/`slot_capacity`/`slot_used_len`/`blocks`, plus the same
//! O(height) `level_*` caches every variant carries) stays resident in
//! RAM after that -- see [`Self::mem_size_bytes`]'s doc comment for what
//! that means for comparing this variant's "memory" number to the
//! others'.
//!
//! [`Self::allocate_slot`] (insert-triggered new-slot creation) grows the
//! file with [`std::fs::File::set_len`] instead of `Vec::resize`. This is
//! the one structural advantage a real file has over the arena's shared
//! `Vec<u8>`: extending a file's length is metadata-only -- the
//! filesystem does not need to relocate and copy every byte already
//! written the way `Vec::resize` can when its backing allocation must
//! grow, so this variant should not reproduce the arena's occasional
//! large-reallocation insert-latency spikes (see that file's own
//! benchmark results). What it pays instead is the constant-factor cost
//! of a real syscall on every single field access, everywhere, not just
//! on growth.
//!
//! Shifting bytes within a slot (for an in-place insert) means reading
//! the shifted-over tail into a small buffer, then writing it back at
//! its new offset -- the file-I/O analogue of what [`Vec::copy_within`]
//! does for the arena variant and what `MmapBlock::insert_at` did with
//! raw `ptr::copy` for the mmap variant. No `unsafe` anywhere in this
//! file (confirmed via `grep -rn "unsafe"` on this file specifically,
//! same standard this whole project otherwise holds to).
//!
//! # What this doesn't attempt
//!
//! A vacated slot (e.g. the original half of a block that split entirely
//! into its tail) is never reclaimed or reused -- there is no free-list.
//! Over many inserts this would eventually waste file space on abandoned
//! slots; this prototype accepts that rather than building a genuine
//! allocator with reuse, matching the same "isolate the one question
//! being asked" scoping every variant in this exploration has used. The
//! backing file itself is a `std::env::temp_dir()` file, unlinked
//! immediately after creation (its file descriptor keeps working after
//! unlink on Linux, and the space is reclaimed automatically once the
//! `File` closes, including on panic -- no `Drop` impl needed for
//! cleanup).
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
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Creates and opens a fresh backing file for one tree's node records,
/// then immediately unlinks it. On Linux (and POSIX generally) an
/// already-open file descriptor keeps working after its directory entry
/// is removed -- the file's storage is freed automatically when the last
/// open handle closes, whether that's an ordinary `Drop` or unwinding
/// past one after a panic. This sidesteps needing a manual `Drop` impl
/// (and the "did every code path actually clean up" risk that comes with
/// one) entirely.
fn open_backing_file() -> File {
    let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("s_graph_filebacked_{}_{id}.bin", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap_or_else(|e| panic!("failed to create backing file {path:?}: {e}"));
    let _ = std::fs::remove_file(&path);
    file
}

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

/// Size of one [`RadixTreeDfsInterleavedVarSkipUnrolledFile::block_stack`]
/// entry's fetch, in bytes -- deliberately much
/// smaller than a typical `block_capacity` (65536 in this project's own
/// benchmarks). A whole-slot cache was tried first and measured to be a
/// net loss on `contains`/`insert` (2-6.5x *slower* than no caching at
/// all, across every benchmarked scenario) because those touch one
/// 8-byte word per level with little same-slot reuse, so paying for a
/// full up-to-64KB `pread` when 8 bytes were needed was pure waste; it
/// was, however, a large win for row queries (9-17x faster) which really
/// do scan many siblings consecutively within one slot. 4096 (one
/// typical OS page) keeps most of the sibling-scan win -- a branching
/// node's children easily fit in a few KB -- while bounding how much a
/// low-locality single-field read can overpay. See the module doc
/// comment's "Fetching in page-sized chunks" section.
const READ_CHUNK_BYTES: usize = 4096;

// TEMPORARY diagnostic counter (not part of the structure itself, no
// struct field changes needed): counts real `pread` calls
// (`ensure_stack_entry`'s `read_exact_at`) since the counting thread
// started, to let a caller isolate how many actual reads one query
// costs. `pub(crate)` and thread-local so it's zero-cost for every
// caller not using it.
thread_local! {
    pub(crate) static DIAG_PREAD_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
pub fn diag_pread_count() -> u64 {
    DIAG_PREAD_COUNT.with(|c| c.get())
}

#[cfg(test)]
thread_local! {
    static DIAG_BULK_LIST_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// One cached, page-aligned chunk of a single slot's bytes -- one entry
/// of [`RadixTreeDfsInterleavedVarSkipUnrolledFile::block_stack`]. See
/// that field's doc comment and the module doc comment's "A stack of
/// cached blocks" section for how entries come and go.
struct StackCacheEntry {
    slot: usize,
    /// Byte offset within the slot where `buf` begins (a multiple of
    /// [`READ_CHUNK_BYTES`], except when widened past that to cover a
    /// read that straddled the aligned boundary -- see
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledFile::ensure_stack_entry`]).
    chunk_start: usize,
    buf: Vec<u8>,
}

/// Backing storage for [`RadixTreeDfsInterleavedVarSkipUnrolledFile::block_stack`]:
/// a physical array of [`StackCacheEntry`] that only ever *grows*, plus a
/// logical `top` marking how many entries are currently live
/// (`entries[..top]`). "Popping" is just decrementing `top` -- the
/// entries above it stay physically allocated, inert, ready for
/// [`RadixTreeDfsInterleavedVarSkipUnrolledFile::ensure_stack_entry`] to
/// silently reuse (via `Vec::clear` + `Vec::resize`, which keeps the
/// existing heap allocation as long as it's already big enough, instead
/// of dropping and reallocating) the next time a push needs that depth
/// again. For [`RadixTreeDfsInterleavedVarSkipUnrolledFile::stack_cap`]
/// `1` (every read-only top-level call), `top` settles into oscillating
/// between `1` and `2` for the rest of the tree's lifetime once the
/// first call has run once -- so from then on, no read-only call's push
/// or pop here touches the allocator at all; only a chunk actually
/// needing to grow past what's already reserved does.
struct BlockStack {
    entries: Vec<StackCacheEntry>,
    top: usize,
}

impl BlockStack {
    fn new() -> Self {
        BlockStack { entries: Vec::with_capacity(MAX_HEIGHT), top: 0 }
    }
}

/// RAII guard, constructed at the top of every top-level read-only call
/// ([`RadixTreeDfsInterleavedVarSkipUnrolledFile::contains`]/
/// `query_two_bound`/`query_one_bound`), that pops
/// [`RadixTreeDfsInterleavedVarSkipUnrolledFile::block_stack`] back down
/// to just its permanently-resident root entry when that call returns --
/// on *every* exit path, including early returns, since this runs on
/// drop rather than needing to be threaded through each `return`. Pops
/// rather than truncates the underlying `Vec` (see [`BlockStack`]), so
/// this never touches the allocator either. See the module doc comment's
/// "A stack of cached blocks" section. [`Self::insert`] doesn't use this
/// -- see its own doc comment for why -- and pops the same way manually.
struct StackCacheGuard<'a>(&'a RadixTreeDfsInterleavedVarSkipUnrolledFile);

impl Drop for StackCacheGuard<'_> {
    fn drop(&mut self) {
        let mut stack = self.0.block_stack.borrow_mut();
        stack.top = stack.top.min(1);
    }
}

impl crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleavedVarSkipUnrolledFile {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        RadixTreeDfsInterleavedVarSkipUnrolledFile::height(self)
    }
}

/// A position in the block-list: which block, and the byte offset within
/// it. Never lands mid-record -- every record (word, plus its skip value
/// if far) is fully contained in one block, so ordinary field reads
/// never need to consult [`RadixTreeDfsInterleavedVarSkipUnrolledFile::advance`]
/// at all; only *jumping to a different record* does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Pos {
    block: usize,
    offset: usize,
}

impl Pos {
    const START: Pos = Pos { block: 0, offset: 0 };
}

pub struct RadixTreeDfsInterleavedVarSkipUnrolledFile {
    /// The one shared backing store for every block's bytes -- see the
    /// module doc comment. Every node record lives here, on disk, never
    /// cached across calls; `file_len` tracks its current length so
    /// [`Self::allocate_slot`] knows where to append the next slot
    /// without an extra `metadata()` syscall.
    file: File,
    file_len: u64,
    /// `slot_start[s]`: byte offset in `file` where slot `s` begins.
    /// Almost always `s * block_capacity` in spirit, but tracked
    /// explicitly (rather than computed) because a handful of slots are
    /// deliberately larger than `block_capacity` -- see `slot_capacity`.
    slot_start: Vec<u64>,
    /// `slot_capacity[s]`: how many bytes are reserved for slot `s`.
    /// Normally `block_capacity`, except for a slot created specifically
    /// to hold one oversized record/chain that wouldn't fit in a normal
    /// slot (see [`Self::insert_bytes`] and `chop_into_blocks`) -- the
    /// same per-block-capacity idea `MmapBlock::new(block.capacity.max(new_bytes.len()))`
    /// uses in the sibling mmap variant, and the arena variant's own
    /// `slot_capacity` field, which also give individual blocks their
    /// own capacity rather than assuming a single global one.
    slot_capacity: Vec<usize>,
    /// `slot_used_len[s]`: how many of slot `s`'s reserved bytes
    /// currently hold real content. Indexed by *slot* (physical file
    /// position), not by logical DFS-order block index -- see `blocks`.
    slot_used_len: Vec<usize>,
    /// `blocks[i]` is the slot index holding the `i`-th block in
    /// DFS/logical order. This indirection is what lets a split insert
    /// a new logical block "between" two others (`Vec::insert` on this
    /// small, block-count-sized vector) without needing the *physical*
    /// on-disk layout to match logical order at all -- new slots are
    /// always appended to the end of the file (see
    /// [`Self::allocate_slot`]), wherever they end up in `blocks`.
    blocks: Vec<usize>,
    /// A stack of cached, page-chunked slot windows, one entry per block
    /// currently *open* on the path from the root to wherever the
    /// current top-level call's traversal is right now -- entry 0 is
    /// always `blocks[0]`'s slot and is never popped (see
    /// [`StackCacheGuard`]), so the root stays permanently resident the
    /// way a separate dedicated cache used to provide explicitly; every
    /// entry above it is pushed on first touching a new block, or popped
    /// back to (exposing already-resident bytes, no `pread`) on a
    /// backtrack to an ancestor still open on the stack -- up to
    /// [`Self::stack_cap`] entries above the root; see that field's doc
    /// comment for why the effective depth differs between reads and
    /// writes. See [`Self::ensure_stack_entry`] -- the single place every
    /// read or write goes through -- and the module doc comment's "A
    /// stack of cached blocks, not one sliding slot" section for the
    /// full rationale. `RefCell` because the read path
    /// (`contains`/`query_rec`/descent) is `&self` -- this is ordinary
    /// safe-Rust interior mutability for a cache, not a soundness escape
    /// hatch.
    block_stack: RefCell<BlockStack>,
    /// How many *non-root* entries [`Self::block_stack`] is allowed to
    /// hold before [`Self::ensure_stack_entry`] evicts down to just the
    /// root on the next genuinely-new push -- set to `1` at the start of
    /// every read-only top-level call ([`Self::contains`]/
    /// [`Self::query_two_bound`]/[`Self::query_one_bound`]) and to
    /// `MAX_HEIGHT` at the start of [`Self::insert`]. This is what makes
    /// [`Self::block_stack`] behave as a single [`READ_CHUNK_BYTES`]
    /// sliding window (plus the permanently-pinned root) during
    /// reads -- measured to help point lookups only marginally (via the
    /// root pin alone) and range queries not at all (their own
    /// `ConstrainedFrame` backtrack resumes via a precomputed skip
    /// position, never by re-reading an already-cached ancestor, so
    /// neither query shape has a genuine repeated-block-access pattern
    /// for a deeper stack to exploit) -- while letting it grow to the full
    /// descent depth during [`Self::insert`], whose ancestor-skip-update
    /// pass (see [`Self::insert_new_branch`]) *does* deliberately
    /// re-read every position on the path a second time. `Cell` (not
    /// `RefCell`): a plain `Copy` value set once per top-level call, not
    /// a cache needing interior mutation mid-traversal.
    stack_cap: Cell<usize>,
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
/// grown one point at a time via [`RadixTreeDfsInterleavedVarSkipUnrolledFile::insert`].
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

impl RadixTreeDfsInterleavedVarSkipUnrolledFile {
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
            return RadixTreeDfsInterleavedVarSkipUnrolledFile {
                file: open_backing_file(),
                file_len: 0,
                slot_start: Vec::new(),
                slot_capacity: Vec::new(),
                slot_used_len: Vec::new(),
                blocks: Vec::new(),
                block_stack: RefCell::new(BlockStack::new()),
                stack_cap: Cell::new(1),
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

        RadixTreeDfsInterleavedVarSkipUnrolledFile {
            file: chopped.file,
            file_len: chopped.file_len,
            slot_start: chopped.slot_start,
            slot_capacity: chopped.slot_capacity,
            slot_used_len: chopped.slot_used_len,
            blocks: chopped.blocks,
            block_stack: RefCell::new(BlockStack::new()),
            stack_cap: Cell::new(1),
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
    /// backing file and returns its slot index, extending the file with
    /// [`File::set_len`] -- metadata-only on a real filesystem, unlike
    /// `Vec::resize`'s potential full-buffer copy in the sibling arena
    /// variant (see the module doc comment).
    fn allocate_slot(&mut self, capacity: usize) -> usize {
        let slot = self.slot_capacity.len();
        let start = self.file_len;
        let new_len = start + capacity as u64;
        self.file.set_len(new_len).expect("failed to extend backing file");
        self.file_len = new_len;
        self.slot_start.push(start);
        self.slot_capacity.push(capacity);
        self.slot_used_len.push(0);
        slot
    }

    /// Reads exactly `buf.len()` bytes starting at `offset` within `slot`
    /// -- one `pread` (via `read_exact_at`), unconditionally, bypassing
    /// [`Self::block_stack`]. Used for reads that are already
    /// multi-byte/one-shot by nature (the tail-move in
    /// [`Self::slot_insert_at`]) where a whole-block cache buys nothing;
    /// [`Self::block_read_u64`]/[`Self::block_read_var`] are the
    /// cache-aware entry points for per-node field reads.
    fn slot_read_exact(&self, slot: usize, offset: usize, buf: &mut [u8]) {
        self.file.read_exact_at(buf, self.slot_start[slot] + offset as u64).expect("file-backed block read failed");
    }

    /// Writes `bytes` at `offset` within `slot` -- one `pwrite` (via
    /// `write_all_at`) -- then, if `slot` has a [`Self::block_stack`]
    /// entry, patches the written range directly into that entry's
    /// cached buffer (rather than merely invalidating it), so a
    /// subsequent read of the same slot -- extremely common right after
    /// a write, e.g. [`Self::insert_new_branch`]'s read-then-write
    /// ancestor-skip update -- is served from cache instead of paying a
    /// redundant `pread` for bytes this call just wrote itself. Only
    /// falls back to dropping the cached buffer (forcing a re-fetch on
    /// next read) when the written range doesn't fully fall within what
    /// happens to be cached, which never loses correctness, only a bit
    /// of the cache's usefulness in that rare case.
    fn slot_write_all(&self, slot: usize, offset: usize, bytes: &[u8]) {
        self.file.write_all_at(bytes, self.slot_start[slot] + offset as u64).expect("file-backed block write failed");
        let mut stack = self.block_stack.borrow_mut();
        let top = stack.top;
        if let Some(entry) = stack.entries[..top].iter_mut().find(|e| e.slot == slot) {
            if offset >= entry.chunk_start && offset + bytes.len() <= entry.chunk_start + entry.buf.len() {
                let local = offset - entry.chunk_start;
                entry.buf[local..local + bytes.len()].copy_from_slice(bytes);
            } else {
                entry.buf.clear();
            }
        }
    }

    /// Ensures [`Self::block_stack`] has `slot` at its top, with its
    /// cached chunk covering `[offset, offset + len)` -- fetching
    /// (`pread`) whatever's missing. This is the single place that
    /// implements the whole push-on-descend/pop-on-backtrack policy (see
    /// the module doc comment's "A stack of cached blocks" section): if
    /// `slot` is already the top entry, only the chunk-coverage check
    /// below runs; if `slot` is present deeper in the stack (a
    /// backtrack), every entry above it is popped, exposing it as the
    /// new top with no `pread` needed for the pop itself; otherwise
    /// (`slot` seen nowhere on the stack -- genuinely new territory) a
    /// fresh entry is pushed -- unless the stack already holds
    /// [`Self::stack_cap`] non-root entries, in which case it's first
    /// popped down to just the root, reproducing a single sliding window
    /// when `stack_cap` is `1` (see that field's doc comment). A "push"
    /// never allocates a new [`StackCacheEntry`] if one is already
    /// physically there from a deeper call that has since popped past it
    /// (see [`BlockStack`]) -- it's overwritten in place instead, and its
    /// `buf` is only `clear`ed (dropping its length, keeping its heap
    /// capacity) rather than replaced, so the widen step below reuses
    /// that capacity via `Vec::resize` whenever it's already big enough.
    /// Either way, the resulting top entry's cached chunk is then widened
    /// with one `pread` if it doesn't already cover `[offset, offset +
    /// len)` -- see [`READ_CHUNK_BYTES`]'s doc comment for why the fetch
    /// is bounded rather than covering the whole slot, and widens past
    /// one chunk only in the rare case a read straddles the aligned
    /// boundary.
    fn ensure_stack_entry(&self, slot: usize, offset: usize, len: usize) {
        let mut stack = self.block_stack.borrow_mut();
        let top_matches = stack.top > 0 && stack.entries[stack.top - 1].slot == slot;
        if !top_matches {
            if let Some(idx) = stack.entries[..stack.top].iter().position(|e| e.slot == slot) {
                stack.top = idx + 1;
            } else {
                if stack.top.saturating_sub(1) >= self.stack_cap.get() {
                    stack.top = 1;
                }
                if stack.top < stack.entries.len() {
                    let top = stack.top;
                    let entry = &mut stack.entries[top];
                    entry.slot = slot;
                    entry.chunk_start = 0;
                    entry.buf.clear();
                } else {
                    stack.entries.push(StackCacheEntry { slot, chunk_start: 0, buf: Vec::new() });
                }
                stack.top += 1;
            }
        }
        let top = stack.top;
        let entry = &mut stack.entries[top - 1];
        if offset >= entry.chunk_start && offset + len <= entry.chunk_start + entry.buf.len() {
            return;
        }
        let used = self.slot_used_len[slot];
        let chunk_start = (offset / READ_CHUNK_BYTES) * READ_CHUNK_BYTES;
        let chunk_end = (chunk_start + READ_CHUNK_BYTES).max(offset + len).min(used);
        entry.buf.resize(chunk_end.saturating_sub(chunk_start), 0);
        if !entry.buf.is_empty() {
            self.file.read_exact_at(&mut entry.buf, self.slot_start[slot] + chunk_start as u64).expect("file-backed block read failed");
            DIAG_PREAD_COUNT.with(|c| c.set(c.get() + 1));
        }
        entry.chunk_start = chunk_start;
    }

    /// In-place insertion at `offset` within `slot`: reads the existing
    /// `[offset, used)` tail into a scratch buffer, writes it back
    /// shifted right by `bytes.len()`, then writes `bytes` into the
    /// vacated gap -- the file-I/O analogue of [`Vec::copy_within`] (the
    /// arena variant's approach) or `ptr::copy` (the mmap variant's).
    /// Panics (never reallocates a slot in place) if it wouldn't fit;
    /// callers check capacity first.
    fn slot_insert_at(&mut self, slot: usize, offset: usize, bytes: &[u8]) {
        let used = self.slot_used_len[slot];
        debug_assert!(offset <= used, "slot_insert_at offset {offset} beyond slot's used length {used}");
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        let tail_len = used - offset;
        if tail_len > 0 {
            let mut tail_buf = vec![0u8; tail_len];
            self.slot_read_exact(slot, offset, &mut tail_buf);
            self.slot_write_all(slot, offset + bytes.len(), &tail_buf);
        }
        self.slot_write_all(slot, offset, bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }

    /// Appends `bytes` at the end of `slot`'s used region. Panics rather
    /// than reallocating if it wouldn't fit.
    fn slot_extend_from_slice(&mut self, slot: usize, bytes: &[u8]) {
        let used = self.slot_used_len[slot];
        debug_assert!(used + bytes.len() <= self.slot_capacity[slot], "slot capacity exceeded");
        self.slot_write_all(slot, used, bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }

    #[inline]
    fn block_len(&self, logical: usize) -> usize {
        self.slot_used_len[self.blocks[logical]]
    }

    /// Reads an 8-byte word at `offset` within logical block `logical`.
    /// Served straight from [`Self::block_stack`] -- a real `pread` only
    /// when this exact slot/offset isn't already cached somewhere on the
    /// stack (see [`Self::ensure_stack_entry`]); for block 0 (the root)
    /// that's true from the second top-level call onward, since entry 0
    /// is never popped.
    fn block_read_u64(&self, logical: usize, offset: usize) -> u64 {
        let slot = self.blocks[logical];
        self.ensure_stack_entry(slot, offset, 8);
        let stack = self.block_stack.borrow();
        let c = &stack.entries[stack.top - 1];
        debug_assert_eq!(c.slot, slot);
        let local = offset - c.chunk_start;
        u64::from_le_bytes(c.buf[local..local + 8].try_into().unwrap())
    }

    fn block_read_var(&self, logical: usize, offset: usize, width: u8) -> u64 {
        let slot = self.blocks[logical];
        self.ensure_stack_entry(slot, offset, width as usize);
        let stack = self.block_stack.borrow();
        let c = &stack.entries[stack.top - 1];
        debug_assert_eq!(c.slot, slot);
        let local = offset - c.chunk_start;
        match width {
            1 => c.buf[local] as u64,
            2 => u16::from_le_bytes(c.buf[local..local + 2].try_into().unwrap()) as u64,
            4 => u32::from_le_bytes(c.buf[local..local + 4].try_into().unwrap()) as u64,
            8 => u64::from_le_bytes(c.buf[local..local + 8].try_into().unwrap()),
            _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
        }
    }

    fn block_write_u64_at(&mut self, logical: usize, offset: usize, val: u64) {
        let slot = self.blocks[logical];
        self.slot_write_all(slot, offset, &val.to_le_bytes());
    }

    fn block_write_var_at(&mut self, logical: usize, offset: usize, width: u8, val: u64) {
        let slot = self.blocks[logical];
        let bytes = val.to_le_bytes();
        self.slot_write_all(slot, offset, &bytes[..width as usize]);
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
            let skip = self.block_read_var(child_pos.block, child_pos.offset + WORD_RECORD_BYTES, width) as usize;
            self.advance(child_pos, skip)
        } else if child_level == self.height as usize - 1 {
            self.advance(child_pos, WORD_RECORD_BYTES)
        } else {
            let word = self.block_read_u64(child_pos.block, child_pos.offset);
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
        self.stack_cap.set(1);
        let _stack_guard = StackCacheGuard(self);
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
            let word = self.block_read_u64(pos.block, pos.offset);
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
        self.stack_cap.set(1);
        let _stack_guard = StackCacheGuard(self);
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
        self.stack_cap.set(1);
        let _stack_guard = StackCacheGuard(self);
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

            if level >= unconstrained_from {
                // Every bound dimension has exhausted its bits by this
                // level -- the rest of this subtree is a pure listing, no
                // node in it can fail the (now-vacuous) mask check. See
                // `Self::bulk_list_subtree`'s doc comment for why this
                // reads the whole subtree in one shot instead of walking
                // it node by node, the way the rest of this function
                // still does for the genuinely-constrained levels above
                // `unconstrained_from`. No stack frame is pushed here (no
                // per-child backtracking needed -- `bulk_list_subtree`
                // itself is fully recursive over the in-memory buffer),
                // so `descended` stays `false` and control falls straight
                // through to the backtrack-resume loop below, exactly as
                // if a constrained subtree had been fully exhausted.
                self.bulk_list_subtree(pos, lo, level, free_dim, out);
            } else {
                let is_last = level == self.height as usize - 1;
                let dim_bits = self.level_dim_bits[level][free_dim];
                let shifts = self.level_shifts_cache[level][free_dim];
                let child_cell_bits = self.level_child_cell_bits_cache[level][free_dim];
                let word = self.block_read_u64(pos.block, pos.offset);
                debug_assert!(word != 0, "visited node at level {level} has an empty existence word");

                if is_last {
                    let mut relevant = word & level_masks[level];
                    while relevant != 0 {
                        let qi = relevant.trailing_zeros() as usize;
                        relevant &= relevant - 1;
                        let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                        out.push(child_lo as u32);
                    }
                    pos = self.advance(pos, WORD_RECORD_BYTES);
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

                // No `frame_level >= unconstrained_from` fast path here
                // (unlike the pre-bulk-read version of this function):
                // every frame on this stack was pushed by the constrained
                // branch above, which only runs while `level <
                // unconstrained_from` -- so `frame_level <
                // unconstrained_from` always holds, and a frame at or
                // past that boundary can no longer occur.
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

    /// Handles one entire "listing" subtree -- everything at or below
    /// `level` once every bound dimension has exhausted its bits, so
    /// every node in it is a guaranteed match -- in as few `pread`s as
    /// possible instead of one per individual node record. The old
    /// per-node listing walk (still visible in git history) paid one
    /// `Self::block_read_u64` -- one potential cache miss, one potential
    /// `pread` -- for *every visited node*, even though none of them
    /// could ever fail to match: measured on a query returning ~20
    /// results over a 9-level subtree, that was ~218 separate reads, the
    /// dominant cost by far (contrast the genuinely-constrained levels
    /// above `unconstrained_from`, which still walk node by node since
    /// there a wrong branch really can be pruned without reading it).
    /// [`Self::child_span_end`] already computes this whole subtree's
    /// exact byte span (via the node's own stored skip value, for a
    /// "far" node -- the same skip value the constrained path already
    /// reads to prune whole branches) without needing to touch anything
    /// beyond `pos` itself, so that span can be read in one bulk
    /// operation (one `pread` per distinct slot spanned, via
    /// [`Self::bulk_read`], not one per node) and then decoded entirely
    /// from memory via [`Self::decode_listing`], paying zero further I/O
    /// no matter how many nodes the subtree turns out to contain.
    fn bulk_list_subtree(&self, pos: Pos, lo: u64, level: usize, free_dim: usize, out: &mut Vec<u32>) {
        #[cfg(test)]
        DIAG_BULK_LIST_CALLS.with(|c| c.set(c.get() + 1));
        let span_end = self.child_span_end(pos, level);
        let byte_len = self.byte_distance(pos, span_end);
        let buf = self.bulk_read(pos, byte_len);
        self.decode_listing(&buf, 0, lo, level, free_dim, out);
    }

    /// Total bytes from `start` (inclusive) to `end` (exclusive), which
    /// may span multiple blocks -- the same block-walking
    /// [`Self::advance`] does, just measuring distance instead of
    /// producing a new [`Pos`].
    fn byte_distance(&self, start: Pos, end: Pos) -> usize {
        if start.block == end.block {
            return end.offset - start.offset;
        }
        let mut total = self.block_len(start.block) - start.offset;
        for b in (start.block + 1)..end.block {
            total += self.block_len(b);
        }
        total += end.offset;
        total
    }

    /// Reads `byte_len` bytes starting at `start` into one flat buffer:
    /// one [`Self::slot_read_exact`] call per distinct slot the range
    /// touches (a slot's content is always physically contiguous, so
    /// this is already as batched as it can be without changing how
    /// slots themselves are laid out), rather than one read per
    /// individual node record inside that range.
    fn bulk_read(&self, start: Pos, byte_len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; byte_len];
        let mut written = 0usize;
        let mut block = start.block;
        let mut offset = start.offset;
        while written < byte_len {
            let avail = self.block_len(block) - offset;
            let take = avail.min(byte_len - written);
            self.slot_read_exact(self.blocks[block], offset, &mut buf[written..written + take]);
            written += take;
            block += 1;
            offset = 0;
        }
        buf
    }

    /// Decodes one subtree from a buffer captured by
    /// [`Self::bulk_list_subtree`], appending every matching free-dim
    /// value to `out`. Returns the buffer offset immediately after this
    /// node's entire subtree, so a recursive call for the next sibling
    /// can chain directly off it without re-deriving byte lengths.
    /// Mirrors the three cases the old per-node listing walk in
    /// [`Self::query_rec`] handled -- `is_last`, near-leaf double
    /// nesting, and the general interior case (recursing per set bit,
    /// exactly like [`build_interleaved_rec`]'s own recursive descent
    /// mirrors this format's encoding) -- but walks `buf` directly via a
    /// byte offset instead of [`Pos`]/[`Self::block_read_u64`], so it
    /// costs zero I/O regardless of how many nodes the subtree contains.
    fn decode_listing(&self, buf: &[u8], off: usize, lo: u64, level: usize, free_dim: usize, out: &mut Vec<u32>) -> usize {
        let word = u64::from_le_bytes(buf[off..off + WORD_RECORD_BYTES].try_into().unwrap());
        let is_last = level == self.height as usize - 1;
        let dim_bits = self.level_dim_bits[level][free_dim];
        let shifts = self.level_shifts_cache[level][free_dim];
        let child_cell_bits = self.level_child_cell_bits_cache[level][free_dim];

        if is_last {
            let mut relevant = word;
            while relevant != 0 {
                let qi = relevant.trailing_zeros() as usize;
                relevant &= relevant - 1;
                out.push(Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits) as u32);
            }
            off + WORD_RECORD_BYTES
        } else if level + 1 == self.height as usize - 1 {
            let leaf_level = level + 1;
            let leaf_dim_bits = self.level_dim_bits[leaf_level][free_dim];
            let leaf_shifts = self.level_shifts_cache[leaf_level][free_dim];
            let leaf_child_cell_bits = self.level_child_cell_bits_cache[leaf_level][free_dim];
            let mut cursor = off + WORD_RECORD_BYTES;
            let mut own_bits = word;
            while own_bits != 0 {
                let qi = own_bits.trailing_zeros() as usize;
                own_bits &= own_bits - 1;
                let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                let leaf_word = u64::from_le_bytes(buf[cursor..cursor + WORD_RECORD_BYTES].try_into().unwrap());
                let mut relevant = leaf_word;
                while relevant != 0 {
                    let leaf_qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    out.push(Self::advance_lo_dim(child_lo, leaf_dim_bits, leaf_shifts, leaf_qi, leaf_child_cell_bits) as u32);
                }
                cursor += WORD_RECORD_BYTES;
            }
            cursor
        } else {
            let mut cursor = off + self.own_record_bytes(level);
            let mut remaining = word;
            while remaining != 0 {
                let qi = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let child_lo = Self::advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                cursor = self.decode_listing(buf, cursor, child_lo, level + 1, free_dim, out);
            }
            cursor
        }
    }

    /// Inserts one point, mutating the structure in place. Returns
    /// `false` (no change) if it already existed. Only supports points
    /// within the domain established at build time -- see the module
    /// doc comment's "Scope" section.
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> bool {
        // No `StackCacheGuard` here (unlike `contains`/`query_two_bound`/
        // `query_one_bound`): this method needs `&mut self` throughout
        // (e.g. `self.insert_new_branch(...)` below), which a guard
        // holding `&self` alive until its `Drop` would conflict with.
        // `self.block_stack` is a `RefCell`, so truncating it at each
        // return point below needs only a momentary `&self` each time,
        // not a borrow held across the whole call.
        //
        // `stack_cap` is set to `MAX_HEIGHT` (effectively unbounded, see
        // that field's doc comment) for the whole call, since it's
        // `insert_new_branch`'s ancestor-skip-update pass -- not this
        // method's own forward descent -- that needs the deeper stack;
        // there's no cheaper point to set it from without threading a
        // flag through every call in between.
        self.stack_cap.set(MAX_HEIGHT);
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
            let word = self.block_read_u64(pos.block, pos.offset);
            trail[trail_len] = (pos, level, word);
            trail_len += 1;

            if (word >> qi) & 1 == 0 {
                self.insert_new_branch(pos, level, qi, word, coords, lo, &trail[..trail_len]);
                self.block_stack.borrow_mut().top = 1;
                return true;
            }
            if is_last {
                self.block_stack.borrow_mut().top = 1;
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
    /// sort needed), then walks `trail` -- in *reverse*, leaf-most
    /// ancestor first -- to increment every far ancestor's stored skip
    /// by the newly-inserted byte count. The reverse order matters for
    /// [`Self::block_stack`]: it keeps the walk a monotonic pop back
    /// down the stack from wherever the forward descent left it, so
    /// every ancestor is served from cache; walking root-to-leaf instead
    /// would pop the whole stack down to the root on the first
    /// iteration and then have to re-push every deeper ancestor right
    /// back, paying a `pread` for each one all over again. See the
    /// module doc comment's "A stack of cached blocks" section.
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

        for &(anc_pos, anc_level, _) in trail.iter().rev() {
            if anc_level + 2 < self.height as usize {
                let width = self.level_skip_bytes[anc_level];
                let skip_offset = anc_pos.offset + WORD_RECORD_BYTES;
                let old = self.block_read_var(anc_pos.block, skip_offset, width);
                self.block_write_var_at(anc_pos.block, skip_offset, width, old + chain.len() as u64);
            }
        }
    }

    #[inline]
    fn set_bit_in_word(&mut self, pos: Pos, qi: usize) {
        let old = self.block_read_u64(pos.block, pos.offset);
        let new = old | (1u64 << qi);
        self.block_write_u64_at(pos.block, pos.offset, new);
    }

    /// Inserts `new_bytes` into the block stream at `pos`. A slot has a
    /// *fixed* capacity for its whole lifetime (see the module doc
    /// comment) -- it can never grow past `slot_capacity[slot]` in
    /// place, not even for a tail append. So the only two cases are: it
    /// fits in place (`slot_insert_at`, an in-slot shift via
    /// read-then-write), or it doesn't, in which case `new_bytes`
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
            let mut tail_buf = vec![0u8; tail_len];
            self.slot_read_exact(slot, pos.offset, &mut tail_buf);
            let new_slot = self.allocate_slot(capacity);
            self.slot_write_all(new_slot, 0, &tail_buf);
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
        lo = RadixTreeDfsInterleavedVarSkipUnrolledFile::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
    }
    debug_assert_eq!(chain.len() as u64, sizes[start_level], "chain length must match the precomputed size it reports to its own parent");
    chain
}

/// The output of [`chop_into_blocks`]: a backing file already carved
/// into slots and filled with every node record, ready to become a
/// [`RadixTreeDfsInterleavedVarSkipUnrolledFile`]'s fields directly.
struct ChoppedFile {
    file: File,
    file_len: u64,
    slot_start: Vec<u64>,
    slot_capacity: Vec<usize>,
    slot_used_len: Vec<usize>,
    blocks: Vec<usize>,
}

/// Dry-run of `chop_into_blocks`'s own greedy-packing decision (see
/// there) -- same walk over `record_starts`, but only counting each
/// slot's eventual capacity, in the order slots will be created, instead
/// of writing anything. Lets the caller size the whole slotted layout
/// exactly once (one scratch `Vec<u8>`, one `write_all_at` to the file)
/// up front rather than growing the file incrementally while copying
/// record bytes in one `pwrite` at a time.
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

/// Chops `flat` into file-backed slots of at most `capacity` bytes,
/// never splitting a record (using `record_starts`, DFS-order offsets
/// collected by [`build_interleaved_rec`]). Two passes: [`plan_slot_lengths`]
/// first decides exactly how many slots of what sizes are needed, then
/// this function re-walks the same records -- making the identical
/// packing decision, now against real slot boundaries -- assembling the
/// whole slotted layout in one in-RAM scratch buffer, and finally hands
/// that to the backing file in a *single* `write_all_at` call -- the
/// bulk build stays one efficient pass, in contrast to the per-op
/// `pread`/`pwrite`-per-touch cost every other method in this file pays
/// (see the module doc comment).
fn chop_into_blocks(flat: &[u8], record_starts: &[usize], capacity: usize) -> ChoppedFile {
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
    file.set_len(total).expect("failed to size backing file");
    if total > 0 {
        file.write_all_at(&scratch, 0).expect("failed to bulk-write backing file");
    }

    let blocks: Vec<usize> = (0..slot_capacities.len()).collect();
    ChoppedFile { file, file_len: total, slot_start, slot_capacity: slot_capacities, slot_used_len, blocks }
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
                let child_lo = RadixTreeDfsInterleavedVarSkipUnrolledFile::advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
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

impl MemSize for RadixTreeDfsInterleavedVarSkipUnrolledFile {
    /// **Not apples-to-apples with the other three variants' numbers.**
    /// Every node record lives on disk (`self.file`, see the module doc
    /// comment), not in RAM -- this only totals the RAM-resident
    /// bookkeeping (slot table + per-level caches), which is why this
    /// variant's whole reason for existing is that this number should be
    /// dramatically *smaller* than heap/mmap/arena's, at the cost of
    /// real per-access I/O latency those don't pay and real disk space
    /// this alone uses (see [`Self::disk_bytes`]). Report both numbers
    /// together, never this one alone, when comparing against the other
    /// variants.
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.slot_start)
            + crate::common::metrics::vec_bytes(&self.slot_capacity)
            + crate::common::metrics::vec_bytes(&self.slot_used_len)
            + crate::common::metrics::vec_bytes(&self.blocks)
            + crate::common::metrics::vec_bytes(&self.level_skip_bytes)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
            + self.stack_cache_bytes()
    }
}

impl RadixTreeDfsInterleavedVarSkipUnrolledFile {
    /// Total bytes currently held across every *physical*
    /// [`Self::block_stack`] entry -- not just the currently-live ones
    /// (`entries[..top]`). [`BlockStack`] deliberately never frees an
    /// entry's buffer once allocated, so a tree that has ever run one
    /// [`Self::insert`] (`stack_cap` = `MAX_HEIGHT` there) keeps up to
    /// `height` entries' worth of capacity reserved for the rest of its
    /// life, even after every later call is read-only (`stack_cap` = `1`,
    /// `top` back down to at most `2`) -- an honest, deliberate
    /// space-for-reuse tradeoff (see [`BlockStack`]'s doc comment), not
    /// a leak, so it's counted in full here rather than only for the
    /// live prefix.
    fn stack_cache_bytes(&self) -> usize {
        self.block_stack.borrow().entries.iter().map(|e| e.buf.capacity()).sum()
    }

    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("slot_start (O(slots))", crate::common::metrics::vec_bytes(&self.slot_start)),
            ("slot_capacity (O(slots))", crate::common::metrics::vec_bytes(&self.slot_capacity)),
            ("slot_used_len (O(slots))", crate::common::metrics::vec_bytes(&self.slot_used_len)),
            ("blocks: logical-to-slot indirection (O(blocks))", crate::common::metrics::vec_bytes(&self.blocks)),
            ("level_skip_bytes (O(height))", crate::common::metrics::vec_bytes(&self.level_skip_bytes)),
            ("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)),
            ("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)),
            ("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)),
            ("block_stack (physical high-water mark, <= height * READ_CHUNK_BYTES; never freed once grown)", self.stack_cache_bytes()),
        ]
    }

    /// Bytes actually occupied in the backing file -- the disk-space
    /// counterpart to [`Self::mem_size_bytes`]'s RAM-only number. See
    /// that method's doc comment for why the two need to be read
    /// together, not interchangeably.
    pub fn disk_bytes(&self) -> u64 {
        self.file_len
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Serializes this structure to a self-contained byte buffer -- see
    /// `crate::common::tree_serialize`'s own doc comment for the format
    /// and why it exists (letting `examples/phase_file.rs` separate
    /// build/query/insert into isolated processes, each with its own
    /// honest peak-RSS reading). Reads each logical block's used content
    /// with one `pread` (via [`Self::slot_read_exact`]) -- the same
    /// per-slot granularity the read-cache is tuned around, not the
    /// per-field granularity the whole point of this variant is to
    /// avoid on the hot path.
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
            let slot = self.blocks[i];
            let len = self.slot_used_len[slot];
            let mut tmp = vec![0u8; len];
            self.slot_read_exact(slot, 0, &mut tmp);
            out.extend_from_slice(&tmp);
        }
        out
    }

    /// Reconstructs a structure from bytes already in memory (written by
    /// [`Self::serialize`]) -- a thin wrapper over
    /// [`Self::deserialize_from_reader`] via [`std::io::Cursor`]. Prefer
    /// [`Self::deserialize_from_path`] when reading a file: this method
    /// still requires the whole serialized buffer resident before it
    /// starts, which [`Self::deserialize_from_path`] avoids -- exactly
    /// the buffering [`Self::deserialize_from_reader`]'s own doc comment
    /// describes this variant's `deserialize` as having once needlessly
    /// doubled internally, before it was rewritten to write directly
    /// from `buf`'s slices; reading the *whole file* into a `buf` first
    /// (what every `examples/phase_*.rs` binary did before
    /// `deserialize_from_path` existed) was the same problem one layer
    /// up.
    pub fn deserialize(buf: &[u8]) -> Self {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf))
    }

    /// Reconstructs a structure by reading directly from a file written
    /// by [`Self::serialize`] -- never materializes the whole serialized
    /// content as one in-memory buffer; only one block's worth (a
    /// reused `Vec<u8>`, no bigger than the largest block, typically
    /// `block_capacity`) is ever transient at a time, copied straight
    /// from the source file into the target file with one `read_exact` +
    /// `write_all_at` pair per block. See
    /// `crate::common::tree_serialize`'s own doc comment for why this
    /// exists and what it replaced.
    pub fn deserialize_from_path(path: &std::path::Path) -> Self {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path:?}: {e}"));
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file))
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
        if total > 0 {
            file.set_len(total).expect("failed to size backing file");
        }
        let mut slot_used_len = vec![0usize; table.len()];
        let mut xfer = Vec::new();
        for (i, &(_, len)) in table.iter().enumerate() {
            xfer.resize(len, 0);
            r.read_exact(&mut xfer).expect("unexpected EOF while reading serialized tree content");
            if len > 0 {
                file.write_all_at(&xfer, slot_start[i]).expect("failed to write block content");
            }
            slot_used_len[i] = len;
        }

        let slot_capacity: Vec<usize> = table.iter().map(|&(cap, _)| cap).collect();
        let blocks: Vec<usize> = (0..table.len()).collect();

        RadixTreeDfsInterleavedVarSkipUnrolledFile {
            file,
            file_len: total,
            slot_start,
            slot_capacity,
            slot_used_len,
            blocks,
            block_stack: RefCell::new(BlockStack::new()),
            stack_cap: Cell::new(1),
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

    fn cross_check_tree(tree: &RadixTreeDfsInterleavedVarSkipUnrolledFile, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build(triples, split, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("split={split:?} cap={cap}"));
        }
    }

    fn cross_check_dynamic(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(triples, cap);
            cross_check_tree(&tree, &naive, num_s, num_p, num_o, &format!("dynamic cap={cap}"));
        }
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split(), cap);
            assert!(!tree.contains(0, 0, 0));
            assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
        }
    }

    #[test]
    fn single_point() {
        let triples: Vec<Triple> = vec![(0, 0, 0)];
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build(&triples, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split(), cap);
            assert!(tree.contains(0, 0, 0));
            assert!(!tree.contains(0, 0, 1));
        }
    }

    #[test]
    fn tiny_hand_traceable_case() {
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split());
        cross_check_dynamic(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_default_split() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split());
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split());
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
        cross_check(&triples, 6, 6, 6, RadixTreeDfsInterleavedVarSkipUnrolledFile::default_split());
        cross_check_dynamic(&triples, 6, 6, 6);
    }

    #[test]
    fn deep_far_nesting_navigates_correctly() {
        let triples = uniform(500, 20, 500, 6000, 999);
        let naive = NaiveIndex::build(&triples);
        for cap in TEST_CAPACITIES {
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, cap);
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

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(initial, cap);
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

            let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(initial, cap);
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
        let mut tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, 64);
        let before = tree.node_word_count();
        for &(s, p, o) in &triples {
            assert!(!tree.insert(s, p, o), "re-inserting an existing point should return false");
        }
        assert_eq!(tree.node_word_count(), before, "re-inserting existing points must not change node count");
    }

    /// Round-trips through [`RadixTreeDfsInterleavedVarSkipUnrolledFile::serialize`]/
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize`], both
    /// right after build and again after a batch of inserts (so
    /// reconstructed block capacity, not just content, is exercised).
    #[test]
    fn serialize_round_trip() {
        for cap in TEST_CAPACITIES {
            let all = uniform(30, 6, 30, 400, 8080 + cap as u64);
            let split_at = all.len() / 2;
            let (initial, rest) = all.split_at(split_at);
            let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(initial, cap);

            let bytes = tree.serialize();
            let mut restored = RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize(&bytes);
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
            let restored2 = RadixTreeDfsInterleavedVarSkipUnrolledFile::deserialize(&bytes2);
            cross_check_tree(&restored2, &naive_after_insert, 30, 6, 30, &format!("second round-trip, cap={cap}"));
        }
    }

    /// A tiny S domain paired with a much larger O domain, so a
    /// `[0,1]`-bound (S,P) query's bound bits exhaust several levels
    /// before the leaf, leaving a genuine listing tail -- unlike every
    /// other test/benchmark dataset in this project, none of which
    /// happen to shape a domain this way (see the doc comment on
    /// `Self::bulk_list_subtree`'s call site in `Self::query_rec`).
    /// Confirms both that `Self::bulk_list_subtree` is real, exercised
    /// code (not dead code that merely compiles) via
    /// `DIAG_BULK_LIST_CALLS`, and that its results are correct.
    #[test]
    fn bulk_listing_tail_is_exercised_and_correct() {
        DIAG_BULK_LIST_CALLS.with(|c| c.set(0));
        let mut rng = SplitMix64::new(0xB0111);
        let mut triples = Vec::new();
        for _ in 0..2000 {
            let s = rng.next_range(6) as u32;
            let p = rng.next_range(4) as u32;
            let o = rng.next_range(50_000) as u32;
            triples.push((s, p, o));
        }
        let naive = NaiveIndex::build(&triples);
        let tree = RadixTreeDfsInterleavedVarSkipUnrolledFile::build_dynamic(&triples, DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&tree, &naive, 6, 4, 50_000, "bulk listing tail (S=6, P=4, O=50000)");
        assert!(
            DIAG_BULK_LIST_CALLS.with(|c| c.get()) > 0,
            "bulk_list_subtree was never called -- this domain shape no longer produces a listing tail, so this test isn't exercising the code it's meant to"
        );
    }
}
