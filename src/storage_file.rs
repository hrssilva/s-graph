//! `FileStorage`: a real backing file, a slot table, and a page-chunked,
//! write-through stack cache -- near-zero steady-state RAM at the cost of
//! `pread`/`pwrite` latency. Requires the `file` feature.
//!
//! # I/O error policy
//!
//! Reads/writes against the backing file during ordinary `contains`/
//! `insert`/etc. traversal are treated as unrecoverable: a mid-traversal
//! disk failure (`ENOSPC`, `EIO`, a revoked permission) leaves no sensible
//! partial state to hand back to the caller, so these panic with a clear
//! message rather than threading `Result` through the entire read/query
//! API -- the same call the `examples/` prototype this was ported from
//! made, kept deliberately rather than inherited by oversight (see the
//! crate-level docs' "Error handling" section). The one boundary that
//! *does* return `Result` is reconstructing storage from serialized bytes
//! ([`BlockStorage::read_blocks_from`]): that's genuinely untrusted
//! external input (a truncated/corrupt file), not an internal invariant,
//! and it's cleanly separable from any hot traversal loop.

use crate::error::{Result, TreeError};
use crate::storage::{BlockStorage, Pos};
use crate::tree::{read_var_at, write_var_at, MAX_HEIGHT, WORD_RECORD_BYTES};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::fs::{File as StdFile, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Opens a fresh, uniquely-named temp file and immediately unlinks its
/// path -- the classic Linux idiom for a scratch file that needs zero
/// explicit cleanup: the open `File` handle keeps the underlying inode
/// (and its disk space) alive for as long as it's held, even across a
/// panic, and the space is reclaimed automatically the moment every
/// handle to it closes, with no path ever left behind to leak.
fn open_and_unlink_temp_file(prefix: &str) -> std::io::Result<StdFile> {
    let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("{prefix}_{}_{id}.bin", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)?;
    let _ = std::fs::remove_file(&path);
    Ok(file)
}

fn open_backing_file() -> StdFile {
    open_and_unlink_temp_file("s_graph_filebacked").unwrap_or_else(|e| panic!("failed to create backing scratch file: {e}"))
}

/// One cached, page-aligned chunk of a single slot's bytes -- one entry
/// of `FileStorage::block_stack`.
///
/// # History: two more complicated designs were tried here and reverted
///
/// This backend went through two other cache designs before landing back
/// here, both real correctness bugs, both root-caused the hard way --
/// worth recording so a future redesign doesn't re-walk the same path:
///
/// 1. **Fixed-capacity, always-whole-slot, write-back entries** (`buf`
///    always exactly `slot_capacity[slot]` bytes, a `dirty` flag, flushed
///    lazily on eviction). Reintroduced a *worse* version of the same bug
///    class this windowed design fixes (a stale, unflushed entry
///    surviving eviction and later getting silently resurrected) and
///    cost real, measured performance: resolving a whole up-to-64KB slot
///    to touch 8 bytes measured at ~8x a small windowed read.
/// 2. **A second, write-capable windowed cache layered on top of (1)**,
///    to claw back the lost performance without reverting (1) itself.
///    This introduced a *second* correctness bug (two windows for the
///    same slot could overlap, since window size wasn't fixed), plus
///    evidence of at least one more that wasn't fully root-caused, before
///    reverting both back to this file's original shape.
///
/// The common failure mode across both: any design where a write can be
/// *deferred* (a dirty flag, a "flush later" promise) has to prove "at
/// most one authoritative copy of a slot's content exists at a time" as
/// an actively-maintained invariant across every path that creates or
/// evicts an entry -- and every attempt here found a path that didn't.
///
/// # Why this design doesn't have that problem
///
/// Writes here ([`FileStorage::slot_write_all`]) go straight to disk with
/// one `pwrite`, *immediately*, every time -- never deferred, no `dirty`
/// flag, no flush-on-eviction. `buf`/`chunk_start` are then just a
/// read-side accelerator: patched in place if the write falls inside
/// what's cached, or the entry is dropped (`buf.clear()`) if it doesn't.
/// Disk is *always* the single authoritative copy, so there is no "which
/// copy is current" question for any invalidation logic to get wrong.
///
/// This does re-accept the original small-window tradeoff measured
/// against a whole-slot cache: `contains`/`insert` (little same-slot
/// reuse across tree levels) ~9-17x faster than a whole-slot cache, but
/// row queries (many siblings of one parent, genuinely reusing the same
/// slot) up to ~2-6.5x slower than one would be. That tradeoff -- not a
/// bug -- is why [`FileStorage::stack_cap`] exists: a *stack* of these
/// entries (mirroring the actual shape of a DFS descent) instead of one
/// sliding window recovers most of the sibling-scan locality a single
/// whole-slot cache would give, without paying its point-read cost.
struct StackCacheEntry {
    slot: usize,
    /// Byte offset within the slot where `buf` begins (a multiple of
    /// [`READ_CHUNK_BYTES`], except when widened past that to cover a
    /// read/write that straddled the aligned boundary). Meaningless
    /// whenever `buf` is empty (a freshly pushed or invalidated entry).
    chunk_start: usize,
    buf: Vec<u8>,
}

struct BlockStack {
    entries: Vec<StackCacheEntry>,
    top: usize,
}

const READ_CHUNK_BYTES: usize = 4096;

impl BlockStack {
    fn new() -> Self {
        BlockStack { entries: Vec::with_capacity(MAX_HEIGHT), top: 0 }
    }
}

/// RAII guard, constructed at the top of every top-level read-only call
/// (`contains`/`restricted_descent`), that pops
/// `FileStorage::block_stack` back down to just its root entry when
/// that call returns -- on every exit path, since this runs on drop.
/// `insert`/`insert_batch` don't use this and pop the same way manually
/// via [`FileStorage::end_insert`]. Sound down to `top.min(1)` (not `0`)
/// specifically because every top-level call -- including `insert`,
/// which reads the root's own word first thing every time -- always
/// touches at least one entry before this ever runs, so `entries` is
/// never empty at this point.
struct StackCacheGuard<'a>(&'a FileStorage);

impl Drop for StackCacheGuard<'_> {
    fn drop(&mut self) {
        let mut stack = self.0.block_stack.borrow_mut();
        stack.top = stack.top.min(1);
    }
}

/// A real backing `File`, a slot table (`slot_start`/`slot_capacity`/
/// `slot_used_len`) with `blocks[i]` mapping logical block `i` to a
/// physical slot, and the stack of small windowed `StackCacheEntry`
/// reads. Writes (`Self::slot_write_all`) are write-through -- straight
/// to disk every time, the cache just kept in sync opportunistically --
/// see `StackCacheEntry`'s own doc comment for why that's a deliberate,
/// hard-won choice, not an oversight.
pub struct FileStorage {
    file: StdFile,
    pub(crate) file_len: u64,
    slot_start: Vec<u64>,
    slot_capacity: Vec<usize>,
    slot_used_len: Vec<usize>,
    blocks: Vec<usize>,
    block_stack: RefCell<BlockStack>,
    stack_cap: Cell<usize>,
    /// The streaming bulk-build path's currently-open block, held in RAM
    /// and not yet given a real file slot -- `Some` only between the
    /// first `append_record` of a build and the matching `finish_append`.
    /// Buffering one block's worth of records in RAM before issuing a
    /// single batched `pwrite` avoids a `pwrite` syscall per node record.
    build_pending: Option<Vec<u8>>,
    /// Reused across [`Self::slot_insert_at`]/[`Self::slot_remove_at`]
    /// calls -- every `insert`/`remove` on this backend used to allocate
    /// a fresh tail-shifting scratch buffer, every time.
    scratch: Vec<u8>,
}

impl FileStorage {
    fn allocate_slot(&mut self, capacity: usize) -> usize {
        let slot = self.slot_capacity.len();
        let start = self.file_len;
        let new_len = start + capacity as u64;
        self.file.set_len(new_len).unwrap_or_else(|e| panic!("failed to extend backing file: {e}"));
        self.file_len = new_len;
        self.slot_start.push(start);
        self.slot_capacity.push(capacity);
        self.slot_used_len.push(0);
        slot
    }

    /// Moves [`FileStorage::build_pending`]'s buffered block to a real
    /// file slot with one batched `pwrite`, then registers it in
    /// [`FileStorage::blocks`].
    fn flush_pending_block(&mut self, capacity: usize) {
        let buf = self.build_pending.take().expect("flush_pending_block called with no pending block");
        let slot = self.allocate_slot(capacity);
        if !buf.is_empty() {
            self.file.write_all_at(&buf, self.slot_start[slot]).unwrap_or_else(|e| panic!("failed to flush pending block to backing file: {e}"));
        }
        self.slot_used_len[slot] = buf.len();
        self.blocks.push(slot);
    }

    /// One `pread`, unconditionally, bypassing [`Self::block_stack`]
    /// entirely. Correct on its own -- no cache check needed -- only
    /// because `Self::slot_write_all` is write-through: disk is always
    /// the current, authoritative copy of every slot.
    fn slot_read_exact(&self, slot: usize, offset: usize, buf: &mut [u8]) {
        self.file.read_exact_at(buf, self.slot_start[slot] + offset as u64).unwrap_or_else(|e| panic!("file-backed block read failed: {e}"));
    }

    /// Writes `bytes` at `offset` within `slot` -- one `pwrite`,
    /// immediately, every time. Then, if `slot` has an *active*
    /// [`Self::block_stack`] entry, patches the written range directly
    /// into that entry's cached buffer rather than merely dropping it.
    fn slot_write_all(&self, slot: usize, offset: usize, bytes: &[u8]) {
        self.file.write_all_at(bytes, self.slot_start[slot] + offset as u64).unwrap_or_else(|e| panic!("file-backed block write failed: {e}"));
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
    /// (`pread`) whatever's missing. See `StackCacheEntry`'s doc
    /// comment for the push-on-descend/pop-on-backtrack policy this
    /// implements.
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
            self.file.read_exact_at(&mut entry.buf, self.slot_start[slot] + chunk_start as u64).unwrap_or_else(|e| panic!("file-backed block read failed: {e}"));
        }
        entry.chunk_start = chunk_start;
    }

    /// In-place insertion at `offset` within `slot`: reads the existing
    /// `[offset, used)` tail into a scratch buffer, writes it back
    /// shifted right by `bytes.len()`, then writes `bytes` into the
    /// vacated gap.
    fn slot_insert_at(&mut self, slot: usize, offset: usize, bytes: &[u8]) {
        let used = self.slot_used_len[slot];
        assert!(offset <= used, "slot_insert_at: offset {offset} > used {used} for slot {slot}");
        assert!(
            used + bytes.len() <= self.slot_capacity[slot],
            "slot_insert_at: used {used} + new {} > capacity {} for slot {slot} (offset {offset})",
            bytes.len(),
            self.slot_capacity[slot]
        );
        let tail_len = used - offset;
        if tail_len > 0 {
            self.scratch.clear();
            self.scratch.resize(tail_len, 0);
            self.file.read_exact_at(&mut self.scratch, self.slot_start[slot] + offset as u64).unwrap_or_else(|e| panic!("file-backed block read failed: {e}"));
            self.slot_write_all(slot, offset + bytes.len(), &self.scratch);
        }
        self.slot_write_all(slot, offset, bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }

    /// In-place removal of `len` bytes at `offset` within `slot` -- the
    /// mirror of [`Self::slot_insert_at`]: reads the tail past the
    /// removed range into a scratch buffer, writes it back shifted left
    /// by `len` (via [`Self::slot_write_all`], which already patches or
    /// invalidates the stack cache correctly for a write that over- or
    /// under-shoots whatever's cached), then shrinks `slot_used_len`.
    fn slot_remove_at(&mut self, slot: usize, offset: usize, len: usize) {
        let used = self.slot_used_len[slot];
        assert!(offset + len <= used, "slot_remove_at: range [{offset}, {}) exceeds used {used} for slot {slot}", offset + len);
        let tail_len = used - offset - len;
        if tail_len > 0 {
            self.scratch.clear();
            self.scratch.resize(tail_len, 0);
            self.file.read_exact_at(&mut self.scratch, self.slot_start[slot] + (offset + len) as u64).unwrap_or_else(|e| panic!("file-backed block read failed: {e}"));
            self.slot_write_all(slot, offset, &self.scratch);
        }
        self.slot_used_len[slot] = used - len;
    }

    fn slot_extend_from_slice(&mut self, slot: usize, bytes: &[u8]) {
        let used = self.slot_used_len[slot];
        assert!(
            used + bytes.len() <= self.slot_capacity[slot],
            "slot_extend_from_slice: used {used} + new {} > capacity {} for slot {slot}",
            bytes.len(),
            self.slot_capacity[slot]
        );
        self.slot_write_all(slot, used, bytes);
        self.slot_used_len[slot] = used + bytes.len();
    }
}

impl BlockStorage for FileStorage {
    fn debug_slot(&self, block: usize) -> usize {
        self.blocks[block]
    }
    fn read_u64(&self, block: usize, offset: usize) -> u64 {
        let slot = self.blocks[block];
        self.ensure_stack_entry(slot, offset, 8);
        let stack = self.block_stack.borrow();
        let c = &stack.entries[stack.top - 1];
        let local = offset - c.chunk_start;
        u64::from_le_bytes(c.buf[local..local + 8].try_into().unwrap())
    }
    fn read_var(&self, block: usize, offset: usize, width: u8) -> u64 {
        let slot = self.blocks[block];
        self.ensure_stack_entry(slot, offset, width as usize);
        let stack = self.block_stack.borrow();
        let c = &stack.entries[stack.top - 1];
        let local = offset - c.chunk_start;
        match width {
            1 => c.buf[local] as u64,
            2 => u16::from_le_bytes(c.buf[local..local + 2].try_into().unwrap()) as u64,
            4 => u32::from_le_bytes(c.buf[local..local + 4].try_into().unwrap()) as u64,
            8 => u64::from_le_bytes(c.buf[local..local + 8].try_into().unwrap()),
            _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
        }
    }
    fn read_word_and_skip(&self, block: usize, offset: usize, skip_width: u8) -> (u64, u64) {
        let slot = self.blocks[block];
        let total = WORD_RECORD_BYTES + skip_width as usize;
        self.ensure_stack_entry(slot, offset, total);
        let stack = self.block_stack.borrow();
        let c = &stack.entries[stack.top - 1];
        let local = offset - c.chunk_start;
        let word = u64::from_le_bytes(c.buf[local..local + 8].try_into().unwrap());
        let skip = if skip_width > 0 { read_var_at(&c.buf, local + 8, skip_width) } else { 0 };
        (word, skip)
    }
    fn write_u64_at(&mut self, block: usize, offset: usize, val: u64) {
        let slot = self.blocks[block];
        self.slot_write_all(slot, offset, &val.to_le_bytes());
    }
    fn write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64) {
        let slot = self.blocks[block];
        assert!(width == 8 || val < (1u64 << (8 * width as u32)), "write_var_at: value {val} does not fit in {width} bytes (block={block} offset={offset})");
        let bytes = val.to_le_bytes();
        self.slot_write_all(slot, offset, &bytes[..width as usize]);
    }
    #[inline]
    fn block_len(&self, block: usize) -> usize {
        self.slot_used_len[self.blocks[block]]
    }
    fn block_count(&self) -> usize {
        self.blocks.len()
    }
    fn bulk_read(&self, block: usize, offset: usize, len: usize) -> Cow<'_, [u8]> {
        let mut buf = vec![0u8; len];
        let mut written = 0usize;
        let mut b = block;
        let mut off = offset;
        while written < len {
            let avail = self.block_len(b) - off;
            let take = avail.min(len - written);
            self.slot_read_exact(self.blocks[b], off, &mut buf[written..written + take]);
            written += take;
            b += 1;
            off = 0;
        }
        Cow::Owned(buf)
    }
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8], block_capacity: usize) {
        let slot = self.blocks[pos.block];
        let used = self.slot_used_len[slot];
        let capacity = self.slot_capacity[slot];
        if used + new_bytes.len() <= capacity {
            self.slot_insert_at(slot, pos.offset, new_bytes);
            return;
        }
        if pos.offset == 0 {
            // Prepend a block without copying the old block into a new tail
            // slot or leaving an empty logical block behind.
            let new_slot = self.allocate_slot(block_capacity.max(new_bytes.len()));
            self.slot_extend_from_slice(new_slot, new_bytes);
            self.blocks.insert(pos.block, new_slot);
            return;
        }
        let is_tail = pos.offset == used;
        let tail_slot = if is_tail {
            None
        } else {
            let tail_len = used - pos.offset;
            self.scratch.clear();
            self.scratch.resize(tail_len, 0);
            self.file.read_exact_at(&mut self.scratch, self.slot_start[slot] + pos.offset as u64).unwrap_or_else(|e| panic!("file-backed block read failed: {e}"));
            let new_slot = self.allocate_slot(capacity);
            self.slot_write_all(new_slot, 0, &self.scratch);
            self.slot_used_len[new_slot] = tail_len;
            self.slot_used_len[slot] = pos.offset;
            Some(new_slot)
        };
        let new_block_capacity = block_capacity.max(new_bytes.len());
        let new_block_slot = self.allocate_slot(new_block_capacity);
        self.slot_extend_from_slice(new_block_slot, new_bytes);
        self.blocks.insert(pos.block + 1, new_block_slot);
        if let Some(tail_slot) = tail_slot {
            self.blocks.insert(pos.block + 2, tail_slot);
        }
    }
    fn remove_bytes(&mut self, pos: Pos, len: usize) {
        let slot = self.blocks[pos.block];
        self.slot_remove_at(slot, pos.offset, len);
    }
    fn begin_top_level_call(&self) -> impl Drop + '_ {
        self.stack_cap.set(1);
        StackCacheGuard(self)
    }
    fn set_insert_mode(&self, active: bool) {
        self.stack_cap.set(if active { MAX_HEIGHT } else { 1 });
    }
    /// Manually pops `FileStorage::block_stack` back down to at most
    /// just the root entry, the same place `StackCacheGuard` leaves it
    /// for read-only calls (`.min(1)`, matching that guard's own Drop
    /// exactly, not an unconditional `= 1` -- nothing here currently
    /// drives `top` below `1` the way `CompressedStorage::invalidate_block`
    /// can, but there is no reason for this to be the one place in
    /// either backend that would silently resurrect a deactivated entry
    /// if that ever changed) -- `insert`/`insert_batch` can't hold that
    /// guard (its `&self` borrow would conflict with the `&mut self`
    /// they need throughout), so they call this explicitly at every
    /// return point.
    fn end_insert(&self) {
        let mut stack = self.block_stack.borrow_mut();
        stack.top = stack.top.min(1);
    }

    fn append_record(&mut self, bytes: &[u8], capacity: usize) -> Pos {
        debug_assert!(bytes.len() <= capacity, "record of {} bytes exceeds block_capacity {capacity}", bytes.len());
        let need_new_block = match &self.build_pending {
            None => true,
            Some(buf) => !buf.is_empty() && buf.len() + bytes.len() > capacity,
        };
        if need_new_block {
            if self.build_pending.is_some() {
                self.flush_pending_block(capacity);
            }
            self.build_pending = Some(Vec::with_capacity(capacity));
        }
        let block = self.blocks.len();
        let pending = self.build_pending.as_mut().expect("build_pending just ensured Some");
        let offset = pending.len();
        pending.extend_from_slice(bytes);
        Pos { block, offset }
    }
    fn patch_appended(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64) {
        if pos.block == self.blocks.len() {
            let pending = self.build_pending.as_mut().expect("patch_appended: pending build block missing");
            write_var_at(pending, pos.offset + rel_offset, width, val);
        } else {
            let slot = self.blocks[pos.block];
            let bytes = val.to_le_bytes();
            self.slot_write_all(slot, pos.offset + rel_offset, &bytes[..width as usize]);
        }
    }
    fn finish_append(&mut self, capacity: usize) {
        if self.build_pending.is_some() {
            self.flush_pending_block(capacity);
        }
        self.shrink_to_fit();
    }
    fn empty() -> Self {
        FileStorage {
            file: open_backing_file(),
            file_len: 0,
            slot_start: Vec::new(),
            slot_capacity: Vec::new(),
            slot_used_len: Vec::new(),
            blocks: Vec::new(),
            block_stack: RefCell::new(BlockStack::new()),
            stack_cap: Cell::new(1),
            build_pending: None,
            scratch: Vec::new(),
        }
    }
    fn shrink_to_fit(&mut self) {
        self.slot_start.shrink_to_fit();
        self.slot_capacity.shrink_to_fit();
        self.slot_used_len.shrink_to_fit();
        self.blocks.shrink_to_fit();
        self.block_stack.get_mut().entries.shrink_to_fit();
        self.scratch.shrink_to_fit();
    }
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize) {
        if records.is_empty() {
            return;
        }
        let mut groups: Vec<&[(Pos, usize)]> = Vec::new();
        let mut group_start = 0usize;
        let mut current_len = 0usize;
        for (i, &(_, len)) in records.iter().enumerate() {
            if i > group_start && current_len + len > block_capacity {
                groups.push(&records[group_start..i]);
                group_start = i;
                current_len = 0;
            }
            current_len += len;
        }
        groups.push(&records[group_start..]);

        let file = open_backing_file();
        let mut slot_start = Vec::with_capacity(groups.len());
        let mut slot_capacity = Vec::with_capacity(groups.len());
        let mut slot_used_len = Vec::with_capacity(groups.len());
        let mut cursor = 0u64;
        let mut buf = Vec::new();
        for group in &groups {
            let total_len: usize = group.iter().map(|&(_, len)| len).sum();
            buf.clear();
            buf.reserve(total_len);
            for &(pos, len) in *group {
                let old_slot = self.blocks[pos.block];
                let start = buf.len();
                buf.resize(start + len, 0);
                self.slot_read_exact(old_slot, pos.offset, &mut buf[start..start + len]);
            }
            let cap = total_len.max(block_capacity);
            if cap > 0 {
                file.write_all_at(&buf, cursor).unwrap_or_else(|e| panic!("failed to write compacted block content: {e}"));
            }
            slot_start.push(cursor);
            slot_capacity.push(cap);
            slot_used_len.push(total_len);
            cursor += cap as u64;
        }
        file.set_len(cursor).unwrap_or_else(|e| panic!("failed to size compacted backing file: {e}"));

        self.file = file;
        self.file_len = cursor;
        self.slot_start = slot_start;
        self.slot_capacity = slot_capacity;
        self.slot_used_len = slot_used_len;
        self.blocks = (0..groups.len()).collect();
        *self.block_stack.get_mut() = BlockStack::new();
    }

    fn block_table(&self) -> Vec<(usize, usize)> {
        (0..self.blocks.len()).map(|i| (self.slot_capacity[self.blocks[i]], self.slot_used_len[self.blocks[i]])).collect()
    }
    fn write_blocks_into(&self, out: &mut Vec<u8>) {
        for i in 0..self.blocks.len() {
            let slot = self.blocks[i];
            let len = self.slot_used_len[slot];
            let mut tmp = vec![0u8; len];
            self.slot_read_exact(slot, 0, &mut tmp);
            out.extend_from_slice(&tmp);
        }
    }
    /// Reconstructs from serialized bytes. Unlike every other method on
    /// this impl, this genuinely is an untrusted-input boundary (a
    /// truncated or corrupted serialized stream), so it returns `Result`
    /// instead of panicking.
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> Result<Self> {
        let mut slot_start = Vec::with_capacity(table.len());
        let mut total: u64 = 0;
        for &(cap, _) in table {
            slot_start.push(total);
            total += cap as u64;
        }
        let file = open_and_unlink_temp_file("s_graph_filebacked")?;
        if total > 0 {
            file.set_len(total)?;
        }
        let mut slot_used_len = vec![0usize; table.len()];
        let mut xfer = Vec::new();
        for (i, &(_, len)) in table.iter().enumerate() {
            xfer.resize(len, 0);
            r.read_exact(&mut xfer).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    TreeError::CorruptData("truncated block content".to_string())
                } else {
                    TreeError::Io(e)
                }
            })?;
            if len > 0 {
                file.write_all_at(&xfer, slot_start[i])?;
            }
            slot_used_len[i] = len;
        }
        let slot_capacity: Vec<usize> = table.iter().map(|&(cap, _)| cap).collect();
        let blocks: Vec<usize> = (0..table.len()).collect();
        Ok(FileStorage {
            file,
            file_len: total,
            slot_start,
            slot_capacity,
            slot_used_len,
            blocks,
            block_stack: RefCell::new(BlockStack::new()),
            stack_cap: Cell::new(1),
            build_pending: None,
            scratch: Vec::new(),
        })
    }

    fn mem_size_bytes(&self) -> usize {
        crate::vec_bytes(&self.slot_start)
            + crate::vec_bytes(&self.slot_capacity)
            + crate::vec_bytes(&self.slot_used_len)
            + crate::vec_bytes(&self.blocks)
            + self.stack_cache_bytes()
            + crate::vec_bytes(&self.scratch)
    }
    fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("slot_start (O(slots))", crate::vec_bytes(&self.slot_start)),
            ("slot_capacity (O(slots))", crate::vec_bytes(&self.slot_capacity)),
            ("slot_used_len (O(slots))", crate::vec_bytes(&self.slot_used_len)),
            ("blocks: logical-to-slot indirection (O(blocks))", crate::vec_bytes(&self.blocks)),
            ("block_stack (windowed read cache)", self.stack_cache_bytes()),
            ("scratch (insert/remove tail-shift buffer)", crate::vec_bytes(&self.scratch)),
        ]
    }
}

impl FileStorage {
    /// Bounded by O(height) entries at up to [`READ_CHUNK_BYTES`] each
    /// (not O(dataset size)) -- a small, stack-shaped read cache, never
    /// holding anything not already durably on disk.
    fn stack_cache_bytes(&self) -> usize {
        self.block_stack.borrow().entries.iter().map(|e| e.buf.capacity()).sum()
    }
}
