//! The storage abstraction: [`BlockStorage`], the trait every backend
//! implements, plus [`Pos`] (a logical block/offset address) and
//! `HeapStorage`, the `heap` feature’s zero-dependency, full-RAM-
//! resident backend. `file`/`compressed`-feature backends live in
//! `crate::storage_file`/`crate::storage_compressed`.

#[cfg(feature = "heap")]
use crate::tree::{read_var_at, write_var_at};
use std::borrow::Cow;

/// A position in the block-list: which (logical) block, and the byte
/// offset within it. Backend-agnostic -- `S::read_u64`/etc. resolve
/// `block` to wherever that backend actually keeps it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pos {
    pub(crate) block: usize,
    pub(crate) offset: usize,
}

impl Pos {
    pub(crate) const START: Pos = Pos { block: 0, offset: 0 };
}

// ==================== the storage abstraction ====================

/// Everything the shared tree algorithm needs from wherever node bytes
/// actually live, addressed by *logical* (DFS-order) block index -- each
/// implementation owns its own physical-block/slot bookkeeping entirely
/// privately (no shared `slot_start`-style fields live on the outer
/// generic struct). Reads take `&self` (both impls' caching, where they
/// have any, is interior-mutability-based); writes take `&mut self`
/// (`HeapStorage` genuinely needs exclusive access to mutate its `Vec`s;
/// `FileStorage`'s writes are always called from a `&mut self` context
/// anyway, so this costs it nothing).
pub trait BlockStorage: Sized {
    /// Scope a graph automaton traversal. The default has no runtime state;
    /// paged backends can retain a bounded path of chunks until it returns.
    fn begin_read_session(&self) -> impl Drop + '_ { NoopReadSession }
    fn read_u64(&self, block: usize, offset: usize) -> u64;
    fn read_var(&self, block: usize, offset: usize, width: u8) -> u64;

    fn write_u64_at(&mut self, block: usize, offset: usize, val: u64);
    fn write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64);
    fn block_len(&self, block: usize) -> usize;
    fn block_count(&self) -> usize;

    /// Maps a logical block index to whatever stable physical identity
    /// the backend has (a no-op for `HeapStorage`, where logical index
    /// *is* physical identity; `FileStorage` overrides this to return the
    /// slot number, which stays stable across the `self.blocks`
    /// reordering/insertion that churns logical indices). Used only for
    /// diagnostics -- never on a hot path.
    fn debug_slot(&self, block: usize) -> usize {
        block
    }

    /// Whole-subtree bulk read for the tree's own bulk-listing
    /// optimization: zero-copy borrow when the whole range lives in one
    /// physical block (`HeapStorage`'s common case), an owned copy
    /// otherwise (`FileStorage`, always; `HeapStorage` only when a range
    /// spans more than one block, which -- since a block never splits a
    /// record -- only happens for a subtree wider than one block).
    fn bulk_read(&self, block: usize, offset: usize, len: usize) -> Cow<'_, [u8]>;

    /// Splices `new_bytes` in at `pos`, growing/splitting storage as
    /// needed -- see each impl for its own mechanism; genuinely
    /// different between backends, so this stays a whole trait method
    /// rather than being decomposed further.
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8], block_capacity: usize);

    /// Removes `len` bytes at `pos` -- the mirror of [`Self::insert_bytes`].
    /// Always shrinks the owning block in place; unlike `insert_bytes`,
    /// this never needs a split-equivalent (removal can't overflow a
    /// destination). Some backends drop a block that becomes entirely
    /// empty as a result (see `HeapStorage`'s own impl for why), so
    /// [`Self::block_count`] may or may not change -- callers that care
    /// must check before/after, exactly as `Tree::remove_cascade` does.
    fn remove_bytes(&mut self, pos: Pos, len: usize);

    /// RAII guard, constructed at the top of every top-level call
    /// (`contains`/`restricted_descent`), dropped at its
    /// end. No-op for `HeapStorage`; pops `FileStorage`'s stack cache
    /// back to just the root.
    fn begin_top_level_call(&self) -> impl Drop + '_;
    /// No-op for `HeapStorage`; toggles `FileStorage`'s cache depth
    /// (`1` for reads, unbounded for insert's ancestor-revisit pass).
    fn set_insert_mode(&self, active: bool);
    /// Called at the end of `insert` (both return paths) since `insert`
    /// can't hold a `begin_top_level_call` guard across its own
    /// `&mut self` calls.
    fn end_insert(&self);

    /// Appends one atomic node record to whichever block the streaming
    /// bulk-build path is currently filling, opening a fresh block first
    /// if `bytes` wouldn't fit in what's left of the current one.
    /// `bytes.len()` must never exceed `block_capacity` (a node record's
    /// worst case -- its existence word plus one hoisted span entry per
    /// real child -- is `crate::tree::MAX_RECORD_BYTES`; every caller of
    /// bulk build already asserts `block_capacity` is at least that).
    fn append_record(&mut self, bytes: &[u8], block_capacity: usize) -> Pos;

    /// Finish the current append block so the next record starts a new block.
    /// Returns false if this backend does not support explicit boundaries.
    /// Unlike `finish_append`, this need not trim all prior block metadata.
    fn start_new_append_block(&mut self, _block_capacity: usize) -> bool { false }

    /// Overwrites `width` bytes at `pos.offset + rel_offset`, where `pos`
    /// was returned by an earlier, not-yet-[`Self::finish_append`]ed call
    /// to [`Self::append_record`] on the same builder -- how a far node's
    /// skip placeholder gets backfilled once its subtree's total byte
    /// length is known, without requiring that node's own block to
    /// already be finalized. Unlike [`BlockStorage::write_var_at`] (which
    /// requires `block` to already be a committed, addressable block),
    /// this must also work when `pos` still refers to storage's own
    /// currently-open, not-yet-finalized block.
    fn patch_appended(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64);

    /// Finalizes whatever block [`Self::append_record`] left open (if
    /// any) and reclaims any leftover reserved-but-unused capacity --
    /// called exactly once, after a streaming bulk build's last
    /// `append_record` call.
    fn finish_append(&mut self, block_capacity: usize);

    fn empty() -> Self;

    /// (capacity, used_len) per logical block, in DFS order.
    fn block_table(&self) -> Vec<(usize, usize)>;
    fn write_blocks_into(&self, out: &mut Vec<u8>);
    /// Reconstructs storage from a serialized block table + content
    /// stream. Fallible: the bytes may come from an untrusted source
    /// (corrupt file, truncated transfer), so a short read is a real
    /// [`crate::error::TreeError::Io`]/[`crate::error::TreeError::CorruptData`],
    /// never a panic.
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> crate::error::Result<Self>;

    fn mem_size_bytes(&self) -> usize;
    fn size_breakdown(&self) -> Vec<(&'static str, usize)>;

    /// Releases any reserved-but-unused capacity, without changing a
    /// single byte of content. Meant to be called by a caller who knows
    /// they're done with a burst of insertion activity (a whole
    /// `insert_batch` call, or "no more inserts for now") -- calling it
    /// after *every* insert or every batch would be counterproductive
    /// (see `HeapStorage::insert_bytes`'s own doc comment: every block
    /// a caller is still actively growing is deliberately pre-reserved to
    /// `block_capacity` precisely so repeated nearby inserts don't each
    /// pay for a reallocation -- shrinking it back down immediately would
    /// just force the next insert to regrow it again).
    fn shrink_to_fit(&mut self);

    /// Rebuilds the entire block layout from scratch, packing `records`
    /// -- every node's own `(position, byte length)` in DFS-preorder --
    /// greedily into `block_capacity`-sized blocks, exactly the way the
    /// streaming bulk-build path packs node records as they're produced,
    /// except sourcing each record's bytes from the *existing* storage
    /// instead of a live build.
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize);
}

#[doc(hidden)]
pub struct NoopReadSession;
impl Drop for NoopReadSession { fn drop(&mut self) {} }

/// Storage used by consuming compaction. Retiring a block frees RAM-resident
/// payload while preserving every logical block number. The block must never
/// be read again; only a consumed graph may use this interface.
pub trait ReclaimingBlockStorage: BlockStorage {
    fn release_block_payload(&mut self, block: usize);

    /// Read a committed block into reusable compaction scratch. Specialized
    /// backends avoid allocating a fresh owned `bulk_read` result per block.
    fn read_compaction_block(&self, block: usize, out: &mut Vec<u8>) {
        let bytes = self.bulk_read(block, 0, self.block_len(block));
        out.clear();
        out.extend_from_slice(&bytes);
    }

    /// Replace a committed block with equally sized content. The default is
    /// correct for custom backends; paged/compressed backends batch the write.
    fn rewrite_compaction_block(&mut self, block: usize, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.block_len(block));
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for (i, chunk) in chunks.iter().enumerate() {
            self.write_u64_at(block, i * 8, u64::from_le_bytes(*chunk));
        }
        let start = bytes.len() / 8 * 8;
        for (i, &byte) in remainder.iter().enumerate() {
            self.write_var_at(block, start + i, 1, byte as u64);
        }
    }
}

/// Builder support for child-first compaction into a root-first logical layout.
/// Reserved blocks have no readable payload until installed. Each slot must be
/// installed exactly once, before exposing the resulting graph to readers.
pub trait ReservedBlockStorage: ReclaimingBlockStorage {
    /// Reserve metadata only; return the first new logical block number.
    fn reserve_compaction_blocks(&mut self, count: usize, capacity: usize) -> usize;
    /// Install a complete block, consuming or clearing `bytes` for reuse.
    fn install_compaction_block(&mut self, block: usize, bytes: &mut Vec<u8>, capacity: usize);
}

// -------------------- HeapStorage --------------------

/// Direct port of the original heap variant's storage: one `Vec<u8>` per
/// logical block, no slot indirection at all (a block's index in
/// `blocks` *is* its address). Growing/splitting a block is plain
/// `Vec::splice`/`split_off`/`insert` on these.
#[cfg(feature = "heap")]
pub struct HeapStorage {
    blocks: Vec<Vec<u8>>,
}

#[cfg(feature = "heap")]
impl ReclaimingBlockStorage for HeapStorage {
    fn release_block_payload(&mut self, block: usize) {
        self.blocks[block] = Vec::new();
    }
    fn rewrite_compaction_block(&mut self, block: usize, bytes: &[u8]) {
        self.blocks[block].copy_from_slice(bytes);
    }
}

#[cfg(feature = "heap")]
impl ReservedBlockStorage for HeapStorage {
    fn reserve_compaction_blocks(&mut self, count: usize, _capacity: usize) -> usize {
        let first = self.blocks.len();
        self.blocks.resize_with(first + count, Vec::new);
        first
    }
    fn install_compaction_block(&mut self, block: usize, bytes: &mut Vec<u8>, _capacity: usize) {
        debug_assert!(self.blocks[block].is_empty());
        let mut payload = std::mem::take(bytes);
        payload.shrink_to_fit();
        self.blocks[block] = payload;
    }
}

#[cfg(feature = "heap")]
struct NoopGuard;
#[cfg(feature = "heap")]
impl Drop for NoopGuard {
    fn drop(&mut self) {}
}

#[cfg(feature = "heap")]
impl BlockStorage for HeapStorage {
    #[inline]
    fn read_u64(&self, block: usize, offset: usize) -> u64 {
        u64::from_le_bytes(self.blocks[block][offset..offset + 8].try_into().unwrap())
    }
    #[inline]
    fn read_var(&self, block: usize, offset: usize, width: u8) -> u64 {
        read_var_at(&self.blocks[block], offset, width)
    }
    #[inline]
    fn write_u64_at(&mut self, block: usize, offset: usize, val: u64) {
        self.blocks[block][offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    }
    #[inline]
    fn write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64) {
        write_var_at(&mut self.blocks[block], offset, width, val);
    }
    #[inline]
    fn block_len(&self, block: usize) -> usize {
        self.blocks[block].len()
    }
    fn block_count(&self) -> usize {
        self.blocks.len()
    }
    fn bulk_read(&self, block: usize, offset: usize, len: usize) -> Cow<'_, [u8]> {
        let avail = self.blocks[block].len() - offset;
        if len <= avail {
            return Cow::Borrowed(&self.blocks[block][offset..offset + len]);
        }
        let mut buf = Vec::with_capacity(len);
        buf.extend_from_slice(&self.blocks[block][offset..]);
        let mut b = block + 1;
        while buf.len() < len {
            let take = (len - buf.len()).min(self.blocks[b].len());
            buf.extend_from_slice(&self.blocks[b][..take]);
            b += 1;
        }
        Cow::Owned(buf)
    }
    /// A tail-position insert (`pos.offset == block.len()`) that doesn't
    /// fit within `block_capacity` gets a fresh block of its own instead
    /// of being let through onto the existing block regardless of size.
    /// An earlier, unconditional tail bypass meant a block at a
    /// repeatedly-hit tail position could grow arbitrarily far past
    /// `block_capacity` via plain `Vec` amortized doubling, never
    /// splitting -- measured to balloon a build's `mem_size_bytes` by
    /// ~9x (53.55 MiB -> 476.21 MiB at 2,000,000 triples) once
    /// Morton-sorted batch insertion made hitting the same block's tail
    /// repeatedly, back to back, the common case instead of a rare one.
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8], block_capacity: usize) {
        let block_len = self.blocks[pos.block].len();
        if block_len + new_bytes.len() <= block_capacity {
            let block = &mut self.blocks[pos.block];
            block.splice(pos.offset..pos.offset, new_bytes.iter().copied());
            return;
        }
        if pos.offset == block_len {
            // Doesn't fit, and there's no existing tail content to carry
            // into a split -- give new_bytes a block of its own, right
            // after this one, instead of widening this block past its
            // intended capacity.
            let mut new_block = Vec::with_capacity(block_capacity.max(new_bytes.len()));
            new_block.extend_from_slice(new_bytes);
            self.blocks.insert(pos.block + 1, new_block);
            return;
        }
        let block = &mut self.blocks[pos.block];
        let mut tail = block.split_off(pos.offset);
        // Pre-reserve the split-off tail up to the full block budget too
        // -- it's a real, ongoing block from here on (more inserts can
        // land in it later), not a one-off.
        tail.reserve_exact(block_capacity.saturating_sub(tail.len()));
        block.extend_from_slice(new_bytes);
        self.blocks.insert(pos.block + 1, tail);
    }
    /// Unlike `File`/`CompressedStorage` (which track a block's capacity
    /// separately from its used length, so a zero-content block still
    /// serializes with a valid nonzero capacity), `HeapStorage::block_table`
    /// reports `(b.len(), b.len())` -- capacity equals length by
    /// construction here, so a block emptied down to zero bytes would
    /// serialize as `cap == 0`, which deserialization correctly rejects
    /// as corrupt. Drop the block entirely instead of ever leaving that
    /// state behind; `Self::block_count` changing is exactly the case
    /// `Tree::remove_cascade` already checks for before deciding whether
    /// a point update or a full `BlockIndex` rebuild is needed.
    fn remove_bytes(&mut self, pos: Pos, len: usize) {
        self.blocks[pos.block].splice(pos.offset..pos.offset + len, std::iter::empty());
        if self.blocks[pos.block].is_empty() {
            self.blocks.remove(pos.block);
        }
    }
    fn begin_top_level_call(&self) -> impl Drop + '_ {
        NoopGuard
    }
    fn set_insert_mode(&self, _active: bool) {}
    fn end_insert(&self) {}

    /// Every block is allocated at its full `capacity` budget up front
    /// (`Vec::with_capacity`) the moment it's opened, so accumulating
    /// records into it never triggers `Vec`'s own amortized-doubling
    /// reallocation-and-copy along the way. The trade: a block that ends
    /// up smaller than `capacity` carries reserved capacity it never
    /// used -- [`Self::shrink_to_fit`], called from
    /// [`Self::finish_append`] once the whole streaming build is done,
    /// reclaims exactly that.
    fn append_record(&mut self, bytes: &[u8], capacity: usize) -> Pos {
        debug_assert!(bytes.len() <= capacity, "record of {} bytes exceeds block_capacity {capacity}", bytes.len());
        if self.blocks.is_empty() {
            self.blocks.push(Vec::with_capacity(capacity));
        } else {
            let last = self.blocks.len() - 1;
            if !self.blocks[last].is_empty() && self.blocks[last].len() + bytes.len() > capacity {
                self.blocks.push(Vec::with_capacity(capacity));
            }
        }
        let last = self.blocks.len() - 1;
        let offset = self.blocks[last].len();
        self.blocks[last].extend_from_slice(bytes);
        Pos { block: last, offset }
    }
    fn start_new_append_block(&mut self, capacity: usize) -> bool {
        if let Some(last) = self.blocks.last_mut() {
            if last.is_empty() { return true; }
            last.shrink_to_fit();
        }
        self.blocks.push(Vec::with_capacity(capacity));
        true
    }
    fn patch_appended(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64) {
        write_var_at(&mut self.blocks[pos.block], pos.offset + rel_offset, width, val);
    }
    fn finish_append(&mut self, _capacity: usize) {
        self.shrink_to_fit();
    }
    fn empty() -> Self {
        HeapStorage { blocks: Vec::new() }
    }
    fn shrink_to_fit(&mut self) {
        for block in &mut self.blocks {
            block.shrink_to_fit();
        }
        self.blocks.shrink_to_fit();
    }
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize) {
        let mut new_blocks: Vec<Vec<u8>> = Vec::new();
        let mut current: Vec<u8> = Vec::with_capacity(block_capacity);
        for &(pos, len) in records {
            if !current.is_empty() && current.len() + len > block_capacity {
                new_blocks.push(std::mem::replace(&mut current, Vec::with_capacity(block_capacity)));
            }
            current.extend_from_slice(&self.blocks[pos.block][pos.offset..pos.offset + len]);
        }
        if !current.is_empty() {
            new_blocks.push(current);
        }
        self.blocks = new_blocks;
    }

    fn block_table(&self) -> Vec<(usize, usize)> {
        self.blocks.iter().map(|b| (b.len(), b.len())).collect()
    }
    fn write_blocks_into(&self, out: &mut Vec<u8>) {
        for b in &self.blocks {
            out.extend_from_slice(b);
        }
    }
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> crate::error::Result<Self> {
        let mut blocks = Vec::with_capacity(table.len());
        for &(_, len) in table {
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            blocks.push(v);
        }
        Ok(HeapStorage { blocks })
    }

    fn mem_size_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Vec<u8>>() + self.blocks.iter().map(crate::vec_bytes).sum::<usize>()
    }
    fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("blocks: outer Vec<Vec<u8>> pointers", self.blocks.capacity() * std::mem::size_of::<Vec<u8>>()),
            ("blocks: node records (sum of inner Vecs)", self.blocks.iter().map(crate::vec_bytes).sum()),
        ]
    }
}
