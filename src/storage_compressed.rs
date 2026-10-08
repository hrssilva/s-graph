//! `CompressedStorage`: LZ4-compressed, in-RAM blocks. For deployments
//! with real RAM pressure but no disk available at all (the premise
//! `FileStorage` can't help with). Requires the `compressed` feature.
//!
//! Structurally this is [`crate::storage_file::FileStorage`]'s own
//! write-through, push-on-descend/pop-on-backtrack stack-cache discipline
//! (see that module's `StackCacheEntry` doc comment for the reverted
//! write-back designs and the bug class each one hit -- that history is
//! exactly why this backend does *not* attempt a dirty/deferred-
//! recompress cache) with the resolve cost moved from a `pread`/`pwrite`
//! syscall to a decompress/recompress call, and the payoff moved from
//! "near-zero steady-state RAM" to "compressed-instead-of-full-size
//! steady-state RAM".
//!
//! One structural difference from `FileStorage` worth calling out:
//! `CompressedStorage` has **no slot indirection** (`blocks:
//! Vec<CompressedBlock>`, a block's index *is* its address, matching
//! `HeapStorage`) rather than `FileStorage`'s logical-block-to-physical-
//! slot table. A real file's newly allocated space is physically
//! append-only, which is *why* `FileStorage` needs that indirection; RAM
//! has no such constraint, so `Vec::insert` at the right logical position
//! works directly. The one place this costs something `FileStorage`
//! doesn't pay: a `Vec::insert` shifts every later logical block index by
//! one, which would silently desynchronize any *cached* entry keyed by
//! the old index -- see `CompressedStorage::invalidate_shifted` for why
//! this is sound (root-to-here cache entries are always non-decreasing in
//! block index, so popping the cache down to below the shift point is
//! sufficient).
//!
//! # Chunked compression, not whole-block
//!
//! **Measured, not assumed -- both the original problem and this fix.**
//! An earlier version of this backend compressed each block as *one*
//! monolithic LZ4 blob, up to the full `block_capacity` (e.g. 64 KiB). At
//! 20M triples, that measured a real ~2.1x compression ratio but
//! `contains()` latency ~300-600x slower than `Heap`. Root-caused
//! directly: a single point query triggered ~12-90 real decompress calls,
//! because `advance_past_children`'s existing, backend-agnostic skip-
//! ahead loop reads every earlier active sibling's own skip pointer to
//! jump past its subtree -- an O(active earlier siblings) cost every
//! backend already pays on every level, which `Heap`/`File` don't feel
//! (a cache-cold memory read / page-cache-warm `pread`) but a whole-block
//! decompress felt acutely.
//!
//! The fix, below: compress each block as many small independent
//! `COMPRESS_CHUNK_BYTES`-sized windows instead of one blob (mirroring
//! `FileStorage`'s own `READ_CHUNK_BYTES` `pread` windowing), so a touch
//! decompresses one small chunk, not the whole block. Measured after:
//! `contains()` 9.4x faster, row queries 4.8x faster, inserts 6.7x
//! faster, at a small cost (`mem_size_bytes` +8%, build time +9%).
//! `Compressed` remains genuinely slower than `Heap`/`File` for point/row
//! queries -- that's an inherent property of "decompress on every
//! touch," not something chunking erases.
//!
//! # `unsafe` policy
//!
//! `lz4_flex`'s default `safe-encode`/`safe-decode` build (zero unsafe
//! code) is used deliberately, not its faster unsafe-pointer-path
//! configuration. One measurement on a single run showed that
//! configuration ~18-25% faster across `contains`/row/insert -- but this
//! crate's policy is no `unsafe` in `src/` at all, and that number wasn't
//! re-validated across multiple runs before this decision. Revisit behind
//! a separate opt-in feature if a real deployment profiles LZ4 decode as
//! its bottleneck.

use crate::error::{Result, TreeError};
use crate::storage::{BlockStorage, Pos};
use crate::tree::{read_var_at, write_var_at, MAX_HEIGHT};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};

/// Independently-compressed window size within one block. Matches
/// `storage_file::READ_CHUNK_BYTES` deliberately (same role, same proven
/// size, just for a compress boundary instead of a `pread` boundary), so
/// a single-field touch only ever needs to decompress one (rarely two,
/// only when straddling a boundary) small chunk instead of a whole
/// up-to-`block_capacity` block. Kept as its own constant (not shared
/// with `storage_file`) since the `file` and `compressed` features are
/// independent -- either can be compiled without the other.
const COMPRESS_CHUNK_BYTES: usize = 4096;

struct CompressedBlock {
    /// LZ4-compressed independently per `COMPRESS_CHUNK_BYTES`-aligned
    /// window of this block's uncompressed content: `chunks[i]` covers
    /// uncompressed bytes `[i*COMPRESS_CHUNK_BYTES,
    /// min((i+1)*COMPRESS_CHUNK_BYTES, uncompressed_len))`.
    chunks: Vec<Vec<u8>>,
    /// The block's real, current logical (uncompressed) length --
    /// decompression needs this up front (LZ4's block format doesn't
    /// self-describe its output size).
    uncompressed_len: usize,
    /// The logical/uncompressed capacity budget this block was allocated
    /// under -- identical meaning to `HeapStorage`/`FileStorage`'s own
    /// per-block capacity bookkeeping.
    capacity: usize,
}

/// Splits `content` into `COMPRESS_CHUNK_BYTES`-aligned windows and
/// compresses each independently -- the one place chunking policy lives.
fn chunk_and_compress(content: &[u8]) -> Vec<Vec<u8>> {
    content.chunks(COMPRESS_CHUNK_BYTES).map(lz4_flex::compress).collect()
}

/// Decompresses one block's *entire* content, concatenating every chunk
/// -- bypasses the windowed cache entirely, for call sites that already
/// need the whole block regardless of the windowing optimization.
fn decompress_whole(cb: &CompressedBlock) -> Vec<u8> {
    let mut out = Vec::with_capacity(cb.uncompressed_len);
    for (i, chunk) in cb.chunks.iter().enumerate() {
        let start = i * COMPRESS_CHUNK_BYTES;
        let this_len = (cb.uncompressed_len - start).min(COMPRESS_CHUNK_BYTES);
        let old_len = out.len();
        out.resize(old_len + this_len, 0);
        if this_len > 0 {
            // Decompressing a chunk this backend compressed itself
            // moments earlier (or reconstructed from just-read plaintext
            // in `read_blocks_from`) -- an internal invariant, not an
            // untrusted-input boundary, so this stays a panic.
            lz4_flex::decompress_into(chunk, &mut out[old_len..]).expect("corrupt compressed block");
        }
    }
    out
}

/// One cached, chunk-aligned *window* of a single block's decompressed
/// content -- the `CompressedStorage` counterpart of `StackCacheEntry`.
struct CompressedCacheEntry {
    block: usize,
    chunk_start: usize,
    buf: Vec<u8>,
}

struct CompressedBlockStack {
    entries: Vec<CompressedCacheEntry>,
    top: usize,
}

impl CompressedBlockStack {
    fn new() -> Self {
        CompressedBlockStack { entries: Vec::with_capacity(MAX_HEIGHT), top: 0 }
    }
}

/// RAII guard, the `CompressedStorage` counterpart of `FileStorage`'s
/// `StackCacheGuard`: pops [`CompressedStorage::block_stack`] back to
/// just its root entry when a top-level read-only call returns.
struct CompressedStackCacheGuard<'a>(&'a CompressedStorage);

impl Drop for CompressedStackCacheGuard<'_> {
    fn drop(&mut self) {
        let mut stack = self.0.block_stack.borrow_mut();
        stack.top = stack.top.min(1);
    }
}

pub struct CompressedStorage {
    blocks: Vec<CompressedBlock>,
    block_stack: RefCell<CompressedBlockStack>,
    stack_cap: Cell<usize>,
    /// The streaming bulk-build path's currently-open block, held
    /// **uncompressed** and not yet committed.
    build_pending: Option<Vec<u8>>,
}

impl CompressedStorage {
    /// Ensures `Self::block_stack` has `block`'s decompressed content
    /// covering `[offset, offset + len)` at its top -- the same
    /// push-on-descend/pop-on-backtrack policy as `FileStorage`'s own
    /// cache, just decompressing chunk(s) instead of `pread`ing.
    fn ensure_stack_entry(&self, block: usize, offset: usize, len: usize) {
        let mut stack = self.block_stack.borrow_mut();
        let top_matches = stack.top > 0 && stack.entries[stack.top - 1].block == block;
        if !top_matches {
            if let Some(idx) = stack.entries[..stack.top].iter().position(|e| e.block == block) {
                stack.top = idx + 1;
            } else {
                if stack.top.saturating_sub(1) >= self.stack_cap.get() {
                    stack.top = 1;
                }
                if stack.top < stack.entries.len() {
                    let top = stack.top;
                    let entry = &mut stack.entries[top];
                    entry.block = block;
                    entry.chunk_start = 0;
                    entry.buf.clear();
                } else {
                    stack.entries.push(CompressedCacheEntry { block, chunk_start: 0, buf: Vec::new() });
                }
                stack.top += 1;
            }
        }
        let top = stack.top;
        let entry = &mut stack.entries[top - 1];
        if offset >= entry.chunk_start && offset + len <= entry.chunk_start + entry.buf.len() {
            return;
        }
        let cb = &self.blocks[block];
        let chunk_start = (offset / COMPRESS_CHUNK_BYTES) * COMPRESS_CHUNK_BYTES;
        let want_end = (offset + len).max(chunk_start + COMPRESS_CHUNK_BYTES).min(cb.uncompressed_len);
        entry.buf.clear();
        let mut pos = chunk_start;
        while pos < want_end {
            let chunk_idx = pos / COMPRESS_CHUNK_BYTES;
            let this_len = (cb.uncompressed_len - pos).min(COMPRESS_CHUNK_BYTES);
            let start = entry.buf.len();
            entry.buf.resize(start + this_len, 0);
            if this_len > 0 {
                lz4_flex::decompress_into(&cb.chunks[chunk_idx], &mut entry.buf[start..]).expect("corrupt compressed block");
            }
            pos += this_len;
        }
        entry.chunk_start = chunk_start;
    }

    /// Shared body of `write_u64_at`/`write_var_at`/`patch_appended`'s
    /// already-committed branch: decompress-through-cache (just the
    /// spanned chunk(s), not the whole block), mutate, recompress exactly
    /// those chunk(s) and store back immediately -- write-through, exactly
    /// `FileStorage::slot_write_all`'s discipline, now at chunk
    /// granularity instead of whole-block.
    fn write_block_at(&mut self, block: usize, offset: usize, bytes: &[u8]) {
        self.ensure_stack_entry(block, offset, bytes.len());
        let mut stack = self.block_stack.borrow_mut();
        let top = stack.top;
        let entry = &mut stack.entries[top - 1];
        let local = offset - entry.chunk_start;
        entry.buf[local..local + bytes.len()].copy_from_slice(bytes);
        let chunk_start = entry.chunk_start;
        let new_chunks = chunk_and_compress(&entry.buf);
        drop(stack);
        let first_chunk_idx = chunk_start / COMPRESS_CHUNK_BYTES;
        for (i, c) in new_chunks.into_iter().enumerate() {
            self.blocks[block].chunks[first_chunk_idx + i] = c;
        }
    }

    /// Moves [`Self::build_pending`]'s buffered block to a real,
    /// compressed, committed block.
    fn flush_pending_block(&mut self, capacity: usize) {
        let buf = self.build_pending.take().expect("flush_pending_block called with no pending block");
        let chunks = chunk_and_compress(&buf);
        self.blocks.push(CompressedBlock { chunks, uncompressed_len: buf.len(), capacity });
    }

    /// Pops any cached entries whose `block` index was shifted by a
    /// `Vec::insert` at `from_block` (unlike `FileStorage`, which has no
    /// such shift to begin with).
    fn invalidate_shifted(&mut self, from_block: usize) {
        let stack = self.block_stack.get_mut();
        while stack.top > 0 && stack.entries[stack.top - 1].block >= from_block {
            stack.top -= 1;
        }
    }

    /// Pops the cache stack down to (and removing) any entry for `block`
    /// -- used after a *whole-block* content rewrite (`insert_bytes`)
    /// replaces that block's chunks wholesale.
    fn invalidate_block(&mut self, block: usize) {
        let stack = self.block_stack.get_mut();
        if let Some(idx) = stack.entries[..stack.top].iter().position(|e| e.block == block) {
            stack.top = idx;
        }
    }

    fn cache_bytes(&self) -> usize {
        self.block_stack.borrow().entries.iter().map(|e| e.buf.capacity()).sum()
    }
}

impl BlockStorage for CompressedStorage {
    fn read_u64(&self, block: usize, offset: usize) -> u64 {
        self.ensure_stack_entry(block, offset, 8);
        let stack = self.block_stack.borrow();
        let entry = &stack.entries[stack.top - 1];
        let local = offset - entry.chunk_start;
        u64::from_le_bytes(entry.buf[local..local + 8].try_into().unwrap())
    }
    fn read_var(&self, block: usize, offset: usize, width: u8) -> u64 {
        self.ensure_stack_entry(block, offset, width as usize);
        let stack = self.block_stack.borrow();
        let entry = &stack.entries[stack.top - 1];
        let local = offset - entry.chunk_start;
        read_var_at(&entry.buf, local, width)
    }
    fn write_u64_at(&mut self, block: usize, offset: usize, val: u64) {
        self.write_block_at(block, offset, &val.to_le_bytes());
    }
    fn write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64) {
        let bytes = val.to_le_bytes();
        self.write_block_at(block, offset, &bytes[..width as usize]);
    }
    #[inline]
    fn block_len(&self, block: usize) -> usize {
        self.blocks[block].uncompressed_len
    }
    fn block_count(&self) -> usize {
        self.blocks.len()
    }
    /// Bypasses `Self::block_stack` entirely, like
    /// `FileStorage::slot_read_exact` does for its own bulk reads: a
    /// listing query is one-shot by nature and typically wants most/all
    /// of a spanned block's content anyway.
    fn bulk_read(&self, block: usize, offset: usize, len: usize) -> Cow<'_, [u8]> {
        let mut buf = Vec::with_capacity(len);
        let mut b = block;
        let mut off = offset;
        while buf.len() < len {
            let decompressed = decompress_whole(&self.blocks[b]);
            let avail = decompressed.len() - off;
            let take = avail.min(len - buf.len());
            buf.extend_from_slice(&decompressed[off..off + take]);
            b += 1;
            off = 0;
        }
        Cow::Owned(buf)
    }
    /// Mirrors `HeapStorage::insert_bytes`'s exact splice-or-split shape.
    /// Unlike the simple field-write path, this changes a block's
    /// *length*, so it works against the whole decompressed block and
    /// re-chunks it from scratch rather than patching one window.
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8], block_capacity: usize) {
        let cb_len = self.blocks[pos.block].uncompressed_len;
        if cb_len + new_bytes.len() <= block_capacity {
            let mut content = decompress_whole(&self.blocks[pos.block]);
            content.splice(pos.offset..pos.offset, new_bytes.iter().copied());
            let new_len = content.len();
            self.blocks[pos.block].chunks = chunk_and_compress(&content);
            self.blocks[pos.block].uncompressed_len = new_len;
            self.invalidate_block(pos.block);
            return;
        }
        if pos.offset == cb_len {
            let cap = block_capacity.max(new_bytes.len());
            let chunks = chunk_and_compress(new_bytes);
            self.blocks.insert(pos.block + 1, CompressedBlock { chunks, uncompressed_len: new_bytes.len(), capacity: cap });
            self.invalidate_shifted(pos.block + 1);
            return;
        }
        let mut content = decompress_whole(&self.blocks[pos.block]);
        let tail: Vec<u8> = content.split_off(pos.offset);
        content.extend_from_slice(new_bytes);
        let new_len = content.len();
        self.blocks[pos.block].chunks = chunk_and_compress(&content);
        self.blocks[pos.block].uncompressed_len = new_len;
        self.invalidate_block(pos.block);

        let tail_cap = block_capacity.max(tail.len());
        let tail_chunks = chunk_and_compress(&tail);
        self.blocks.insert(pos.block + 1, CompressedBlock { chunks: tail_chunks, uncompressed_len: tail.len(), capacity: tail_cap });
        self.invalidate_shifted(pos.block + 1);
    }
    /// Mirrors `insert_bytes`'s in-place branch only -- removal never
    /// needs the overflow/split branches, since it only ever shrinks a
    /// block.
    fn remove_bytes(&mut self, pos: Pos, len: usize) {
        let mut content = decompress_whole(&self.blocks[pos.block]);
        content.splice(pos.offset..pos.offset + len, std::iter::empty());
        let new_len = content.len();
        self.blocks[pos.block].chunks = chunk_and_compress(&content);
        self.blocks[pos.block].uncompressed_len = new_len;
        self.invalidate_block(pos.block);
    }
    fn begin_top_level_call(&self) -> impl Drop + '_ {
        self.stack_cap.set(1);
        CompressedStackCacheGuard(self)
    }
    fn set_insert_mode(&self, active: bool) {
        self.stack_cap.set(if active { MAX_HEIGHT } else { 1 });
    }
    fn end_insert(&self) {
        self.block_stack.borrow_mut().top = 1;
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
            let bytes = val.to_le_bytes();
            self.write_block_at(pos.block, pos.offset + rel_offset, &bytes[..width as usize]);
        }
    }
    fn finish_append(&mut self, capacity: usize) {
        if self.build_pending.is_some() {
            self.flush_pending_block(capacity);
        }
        self.shrink_to_fit();
        // A bulk build's own skip-pointer backfilling can populate this
        // cache as a side effect of patching an already-committed block
        // (`write_block_at` always decompresses a window first, unlike
        // `FileStorage`'s write-through `pwrite`, which never needs to
        // read anything to write) -- `shrink_to_fit` above only trims
        // each entry's capacity down to its content length, it doesn't
        // empty them. Left as-is, that read cache is dead weight for a
        // tree nobody has queried yet, and at small scale it can dwarf
        // the structure's own content (measured directly: an 8 KiB
        // cache next to ~4 KiB of actual compressed content, on a
        // 758-triple tree). Reset it to genuinely empty here; the next
        // real read populates it fresh, same as a freshly deserialized
        // tree would.
        *self.block_stack.get_mut() = CompressedBlockStack::new();
    }
    fn empty() -> Self {
        CompressedStorage { blocks: Vec::new(), block_stack: RefCell::new(CompressedBlockStack::new()), stack_cap: Cell::new(1), build_pending: None }
    }
    fn shrink_to_fit(&mut self) {
        for b in &mut self.blocks {
            for c in &mut b.chunks {
                c.shrink_to_fit();
            }
            b.chunks.shrink_to_fit();
        }
        self.blocks.shrink_to_fit();
        let stack = self.block_stack.get_mut();
        for e in &mut stack.entries {
            e.buf.shrink_to_fit();
        }
        stack.entries.shrink_to_fit();
    }
    /// Same greedy record-granular repacking as `HeapStorage`'s own
    /// `compact_from_records`, plus a one-entry "last decompressed
    /// source block" cache local to this call.
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize) {
        let mut new_blocks: Vec<CompressedBlock> = Vec::new();
        let mut current: Vec<u8> = Vec::with_capacity(block_capacity);
        let mut last_source_block: Option<usize> = None;
        let mut last_source_decompressed: Vec<u8> = Vec::new();
        for &(pos, len) in records {
            if last_source_block != Some(pos.block) {
                last_source_decompressed = decompress_whole(&self.blocks[pos.block]);
                last_source_block = Some(pos.block);
            }
            if !current.is_empty() && current.len() + len > block_capacity {
                let chunks = chunk_and_compress(&current);
                new_blocks.push(CompressedBlock { chunks, uncompressed_len: current.len(), capacity: block_capacity });
                current = Vec::with_capacity(block_capacity);
            }
            current.extend_from_slice(&last_source_decompressed[pos.offset..pos.offset + len]);
        }
        if !current.is_empty() {
            let chunks = chunk_and_compress(&current);
            new_blocks.push(CompressedBlock { chunks, uncompressed_len: current.len(), capacity: block_capacity });
        }
        self.blocks = new_blocks;
        *self.block_stack.get_mut() = CompressedBlockStack::new();
    }

    fn block_table(&self) -> Vec<(usize, usize)> {
        self.blocks.iter().map(|b| (b.capacity, b.uncompressed_len)).collect()
    }
    /// Decompresses every block and writes its plain, uncompressed bytes
    /// -- matching `HeapStorage`/`FileStorage`'s existing behavior
    /// exactly, so the serialized wire format stays identical and
    /// interchangeable across all three backends.
    fn write_blocks_into(&self, out: &mut Vec<u8>) {
        for cb in &self.blocks {
            out.extend_from_slice(&decompress_whole(cb));
        }
    }
    /// Reconstructs from serialized bytes -- the one genuinely
    /// untrusted-input boundary in this backend, so it returns `Result`.
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> Result<Self> {
        let mut blocks = Vec::with_capacity(table.len());
        for &(capacity, len) in table {
            let mut v = vec![0u8; len];
            r.read_exact(&mut v).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    TreeError::CorruptData("truncated block content".to_string())
                } else {
                    TreeError::Io(e)
                }
            })?;
            let chunks = chunk_and_compress(&v);
            blocks.push(CompressedBlock { chunks, uncompressed_len: len, capacity });
        }
        Ok(CompressedStorage { blocks, block_stack: RefCell::new(CompressedBlockStack::new()), stack_cap: Cell::new(1), build_pending: None })
    }

    fn mem_size_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<CompressedBlock>()
            + self.blocks.iter().map(|b| b.chunks.capacity() * std::mem::size_of::<Vec<u8>>() + b.chunks.iter().map(|c| c.capacity()).sum::<usize>()).sum::<usize>()
            + self.cache_bytes()
    }
    fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        let chunk_vec_overhead: usize = self.blocks.iter().map(|b| b.chunks.capacity() * std::mem::size_of::<Vec<u8>>()).sum();
        let chunk_content: usize = self.blocks.iter().map(|b| b.chunks.iter().map(|c| c.capacity()).sum::<usize>()).sum();
        vec![
            ("blocks: outer Vec<CompressedBlock> bookkeeping", self.blocks.capacity() * std::mem::size_of::<CompressedBlock>()),
            ("blocks: per-block Vec<Vec<u8>> chunk-pointer overhead", chunk_vec_overhead),
            ("blocks: compressed chunk content (sum)", chunk_content),
            ("block_stack (decompressed window cache)", self.cache_bytes()),
        ]
    }
}
