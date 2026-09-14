//! Generalized version of the `radixtree_dfs_interleaved_varskip_unrolled*`
//! family: one generic struct,
//! [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>`], monomorphized
//! over a [`BlockStorage`] backend instead of being copy-pasted per
//! backend the way
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled::RadixTreeDfsInterleavedVarSkipUnrolled`]
//! (heap) and
//! [`crate::common::radixtree_dfs_interleaved_varskip_unrolled_file::RadixTreeDfsInterleavedVarSkipUnrolledFile`]
//! (file) are today.
//!
//! # What's shared vs. what's per-backend
//!
//! Everything about the *tree* -- the per-level bit schedule, the
//! existence-word/skip-pointer record format, `contains`/`query_rec`'s
//! descent logic, `insert`'s trail walk, the bulk-listing optimization --
//! is written exactly once here, generic over `S: BlockStorage`. Only
//! "given a logical block index, get/put bytes, grow/split the block
//! sequence" is backend-specific, behind the [`BlockStorage`] trait --
//! [`HeapStorage`] (a plain `Vec<Vec<u8>>`, no slot indirection, matching
//! the original heap variant exactly) and [`FileStorage`] (the original
//! file variant's slot table + `pread`/`pwrite` + stack cache, moved here
//! unmodified in spirit).
//!
//! # Why this is a separate file, not a replacement
//!
//! To measure whether the trait indirection itself costs anything
//! relative to the original, hand-specialized `heap.rs`/`file.rs` --
//! [`Heap`]/[`File`] here are meant to be benchmarked side by side with
//! the originals on identical workloads, not to replace them yet.
//!
//! # One deliberate algorithmic difference from the *original* heap variant
//!
//! `query_rec`'s "listing" branch uses the bulk-read-and-decode-in-memory
//! optimization (see [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::bulk_list_subtree`])
//! for *both* backends here, since it's now shared, generic code -- the
//! original file variant already has this; the original heap variant
//! still walks a listing subtree node by node (it never needed the I/O
//! reduction, so nobody had gone back to also give it the simpler bulk
//! decode path). Functionally identical output either way, but this
//! means a `Heap` vs. original-heap latency comparison isn't *purely*
//! isolating abstraction overhead the way the `File` vs. original-file
//! comparison is -- worth keeping in mind when reading benchmark deltas.

use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cell::Cell;
use std::cell::RefCell;
use std::fs::{File as StdFile, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

const K_SQUARE: u8 = 64;
const BASE: u8 = K_SQUARE.trailing_zeros() as u8;
const NUM_DIMS: usize = 3; // S, P, O
const MAX_HEIGHT: usize = 64;
const WORD_RECORD_BYTES: usize = 8;
const MAX_SKIP_RECORD_BYTES: usize = 8;
pub const DEFAULT_BLOCK_CAPACITY: usize = 4096;

// ==================== shared, backend-agnostic helpers ====================
// Identical to both `radixtree_dfs_interleaved_varskip_unrolled.rs` and
// `radixtree_dfs_interleaved_varskip_unrolled_file.rs` -- none of this
// touches storage addressing.

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
    debug_assert!(width == 8 || val < (1u64 << (8 * width as u32)), "value {val} does not fit in {width} bytes");
    match width {
        1 => buf[pos] = val as u8,
        2 => buf[pos..pos + 2].copy_from_slice(&(val as u16).to_le_bytes()),
        4 => buf[pos..pos + 4].copy_from_slice(&(val as u32).to_le_bytes()),
        8 => buf[pos..pos + 8].copy_from_slice(&val.to_le_bytes()),
        _ => unreachable!("skip width must be 1, 2, 4, or 8, got {width}"),
    }
}

#[inline]
fn read_var_at(buf: &[u8], pos: usize, width: u8) -> u64 {
    match width {
        1 => buf[pos] as u64,
        2 => u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as u64,
        4 => u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as u64,
        8 => u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()),
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

/// Bundles the per-level bit schedule so the streaming/parallel bulk-build
/// path (see [`build_rec`] and its own doc comment) doesn't have to thread
/// five separate slice parameters through every recursive call.
struct BuildSchedule<'a> {
    height: usize,
    level_dim_bits: &'a [[u8; NUM_DIMS]],
    level_shifts_cache: &'a [[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &'a [[u8; NUM_DIMS]],
    level_skip_bytes: &'a [u8],
}

impl BuildSchedule<'_> {
    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height {
            WORD_RECORD_BYTES + self.level_skip_bytes[level] as usize
        } else {
            WORD_RECORD_BYTES
        }
    }
}

/// Where a bulk build's node records go as they're produced, in strict
/// DFS-preorder. [`build_rec`] is generic over this instead of writing
/// into one flat `Vec<u8>` (the pre-streaming design), so the exact same
/// recursive tree-building logic can target either real, block-chopped
/// [`BlockStorage`] (streaming records straight into their final home,
/// [`StorageSink`]) or a small in-memory scratch buffer for one subtree at
/// a time ([`FlatBuf`], used by [`build_rec`]'s own parallel branches
/// before their finished bytes get stitched into the real storage -- see
/// [`build_rec`]'s doc comment for why that split exists).
trait RecordSink {
    /// Appends one atomic node record (existence word, plus its skip
    /// placeholder if the node is a far node) and returns its position.
    fn append(&mut self, bytes: &[u8]) -> Pos;
    /// Overwrites `width` bytes at `pos.offset + rel_offset` -- used to
    /// backfill a far node's skip placeholder once its subtree's total
    /// byte length is known, after every descendant has already been
    /// appended.
    fn patch(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64);
    /// Appends an entire *already-encoded* subtree's bytes (the finished
    /// output of a nested [`build_rec`] call, most often one of its
    /// parallel branches) as a unit. Every skip/span value inside `bytes`
    /// is a relative byte distance, never an absolute position (see the
    /// module's own doc comment on `pos + span` navigation), so the bytes
    /// are fully relocatable: nothing inside them needs patching just
    /// because they're being copied somewhere else. [`FlatBuf`] can
    /// therefore just concatenate; [`StorageSink`] has to walk `bytes`
    /// record-by-record (`sched`, `child_level`) to preserve the
    /// "a block never splits a record" invariant real storage depends on.
    fn append_subtree(&mut self, bytes: &[u8], child_level: usize, sched: &BuildSchedule);
}

/// One subtree's worth of bulk-build output, fully in memory -- the unit
/// [`build_rec`]'s parallel branches build into (bounded by that
/// subtree's own size, never the whole dataset) before being stitched
/// into the real [`StorageSink`] by the caller that spawned them.
struct FlatBuf {
    bytes: Vec<u8>,
}

impl FlatBuf {
    fn new() -> Self {
        FlatBuf { bytes: Vec::new() }
    }
}

impl RecordSink for FlatBuf {
    #[inline]
    fn append(&mut self, bytes: &[u8]) -> Pos {
        let offset = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        Pos { block: 0, offset }
    }
    #[inline]
    fn patch(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64) {
        write_var_at(&mut self.bytes, pos.offset + rel_offset, width, val);
    }
    #[inline]
    fn append_subtree(&mut self, bytes: &[u8], _child_level: usize, _sched: &BuildSchedule) {
        self.bytes.extend_from_slice(bytes);
    }
}

/// The real, final destination: a [`RecordSink`] adapter over `S`'s own
/// [`BlockStorage::append_record`]/[`BlockStorage::patch_appended`], so
/// [`build_rec`]'s single-threaded/no-fan-out path writes node records
/// straight into their final block-chopped home as they're produced,
/// instead of ever materializing the whole encoded tree in RAM first (the
/// pre-streaming design's `flat: Vec<u8>` + `record_starts: Vec<usize>`).
struct StorageSink<'a, S: BlockStorage> {
    storage: &'a mut S,
    block_capacity: usize,
}

impl<S: BlockStorage> RecordSink for StorageSink<'_, S> {
    #[inline]
    fn append(&mut self, bytes: &[u8]) -> Pos {
        self.storage.append_record(bytes, self.block_capacity)
    }
    #[inline]
    fn patch(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64) {
        self.storage.patch_appended(pos, rel_offset, width, val);
    }
    fn append_subtree(&mut self, bytes: &[u8], child_level: usize, sched: &BuildSchedule) {
        replay_records(bytes, 0, child_level, sched, &mut |record| {
            self.storage.append_record(record, self.block_capacity);
        });
    }
}

/// Walks an already-encoded subtree buffer (as produced by a nested
/// [`build_rec`] call into a [`FlatBuf`]) in the same DFS-preorder its own
/// build produced it in, calling `emit` once per node record -- the
/// record-granular counterpart of [`FlatBuf::append_subtree`]'s plain
/// concatenation, needed only when the destination is real, block-chopped
/// storage that must never let a block boundary split a record. Pure
/// byte-format parsing (word + `count_ones()` fan-out + schedule-derived
/// record length), the same shape as
/// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::dfs_records_rec`]'s
/// live-tree walk, just over a buffer instead of storage. Returns the
/// cursor position just past the subtree it walked.
fn replay_records(buf: &[u8], off: usize, level: usize, sched: &BuildSchedule, emit: &mut impl FnMut(&[u8])) -> usize {
    let word = u64::from_le_bytes(buf[off..off + WORD_RECORD_BYTES].try_into().unwrap());
    let is_last = level == sched.height - 1;
    let rec_len = sched.own_record_bytes(level);
    emit(&buf[off..off + rec_len]);
    let mut cursor = off + rec_len;
    if !is_last {
        for _ in 0..word.count_ones() {
            cursor = replay_records(buf, cursor, level + 1, sched, emit);
        }
    }
    cursor
}

/// Minimum point count a subtree needs before [`build_rec`] considers
/// splitting its children across `rayon` tasks worth the overhead of a
/// task spawn plus a small heap-allocated [`FlatBuf`] per branch. Not
/// tuned to any particular dataset shape -- it only gates *whether* to
/// fan out, never *how* (that's [`build_rec`]'s own per-node branching,
/// unconditionally schedule-driven either way), so a value in the tens of
/// thousands is conservative in both directions: small enough that any
/// dataset with real parallel work available finds it, large enough that
/// the many small subtrees near tree leaves never pay task overhead for
/// no benefit.
const BUILD_PARALLEL_MIN_POINTS: usize = 100_000;

/// The bulk-build tree-construction algorithm, in exactly one place,
/// generic over where node records end up ([`RecordSink`]). This single
/// function is both this project's streaming-build fix and its
/// parallel-build addition, at once, because both are really the same
/// question -- "who writes this subtree's bytes, and where do they end
/// up" -- answered differently depending on subtree size:
///
/// - **Small subtree, or no more parallel headroom worth spending**:
///   partition `points` by this node's radix digit exactly as the
///   original single-threaded design did, append this node's own record
///   to `sink`, then recurse into each active child *with the same
///   `sink`* -- so when `sink` is a [`StorageSink`] (always true at the
///   top-level call, and true all the way down for any dataset too small
///   to ever trigger fan-out), every node's record streams straight into
///   its final block-chopped home the moment it's produced. Peak RAM
///   for the whole bulk-build byte stream is then whatever's `open` in
///   [`FileStorage`]'s pending block (one `block_capacity`, not the whole
///   tree) or already committed to [`HeapStorage`]'s own blocks (which a
///   caller who chose `Heap` has already accepted paying for) -- never a
///   second, separate full copy of the encoded tree the way the
///   flatten-then-chop design used to need.
/// - **Subtree large enough to be worth it** (`points.len() >=
///   [`BUILD_PARALLEL_MIN_POINTS`]` and at least two of its children are
///   non-empty): this node's own record still streams to `sink`
///   immediately (so an ancestor chain stays fully correct regardless of
///   what happens below), but each active child subtree is handed to its
///   own `rayon` task, building into a fresh, private [`FlatBuf`] --
///   never sharing `sink`, since `S: BlockStorage` isn't safely
///   accessible from multiple threads at once and doesn't need to be.
///   Each task recurses into this same function and may itself fan out
///   further if its own subtree still clears the size bar -- the
///   textbook fork-join-with-size-cutoff shape (Karras 2012's parallel
///   BVH/octree/k-d-tree construction is the closest published analogue:
///   sort once, then treat sibling-subtree construction as embarrassingly
///   parallel work, no top-down pointer chasing needed since each
///   subtree's output is self-contained). Once every branch's `FlatBuf`
///   is done, the *spawning* call stitches them into `sink`, one at a
///   time, in original child order (`sink.append_subtree`) -- trivial
///   concatenation if `sink` is itself a `FlatBuf` (a nested fan-out
///   stitching into its own parent's buffer), or a record-boundary-aware
///   replay if `sink` is the real [`StorageSink`].
///
///   **Honest limit, not glossed over**: because `.collect()` below
///   waits for *every* sibling branch before any of them gets stitched,
///   the peak extra RAM one fan-out episode can cost is the *whole* of
///   that subtree's encoded bytes held an additional time (split across
///   as many `FlatBuf`s as there are active children, but all resident
///   at once) -- not merely "whichever few branches happen to be
///   in-flight at once." For a fan-out that lands near the root of a
///   very large tree, that's effectively the whole dataset's encoded
///   size, on top of the counting sort's own input/scratch buffers (see
///   [`Self::build`]'s own doc comment) -- measured directly: a 20M-triple
///   bulk build's peak RSS is dominated by exactly these transients, not
///   by the final structure (which can be two orders of magnitude
///   smaller on `File`). Turning this into a true bounded-memory pipeline
///   -- stitching each branch into `sink` as soon as *it* finishes,
///   without waiting on its still-running siblings -- needs an ordered,
///   incremental producer/consumer handoff rather than `rayon`'s
///   synchronous `map`+`collect`, which is real additional machinery
///   deliberately left for a follow-up rather than rushed into this
///   pass. What *is* fixed here: each `FlatBuf` is `shrink_to_fit()`'d
///   the moment it's done (below) so at least none of them individually
///   overshoots its own final size via `Vec`'s amortized doubling while
///   they're all resident together.
///
/// Every skip/span value this writes is a relative byte distance (never
/// an absolute position), so a subtree's bytes are correct regardless of
/// *where* they end up being copied to -- the property `append_subtree`
/// leans on to avoid ever needing to patch a value twice.
///
/// A single free function, not a closure re-derived at each call site:
/// this is the one and only place a triple's child-slot digit at a given
/// level gets computed, shared verbatim by [`build_rec`]'s own partition
/// step and by the external bucket-spilling path's
/// [`distribute_to_buckets`] -- so "the bounded-build path produces the
/// exact same tree as the in-RAM path" is a structural fact (the two
/// paths run the identical digit computation), not something that has to
/// be separately proven true of two hand-written copies of the same math.
#[inline]
fn compute_qi(coords: [u64; NUM_DIMS], lo: [u64; NUM_DIMS], shifts: [u8; NUM_DIMS], child_cell_bits: [u8; NUM_DIMS]) -> usize {
    let mut qi = 0usize;
    for d in 0..NUM_DIMS {
        let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
        qi |= sub << shifts[d];
    }
    qi
}

#[allow(clippy::too_many_arguments)]
fn build_rec<Sink: RecordSink>(points: &mut [(u32, u32, u32)], scratch: &mut [(u32, u32, u32)], lo: [u64; NUM_DIMS], level: usize, sched: &BuildSchedule, sink: &mut Sink, node_count: &mut u64) -> u64 {
    let is_last = level == sched.height - 1;
    let is_far_node = level + 2 < sched.height;
    let skip_width = sched.level_skip_bytes[level];
    let dim_bits = sched.level_dim_bits[level];
    let shifts = sched.level_shifts_cache[level];
    let child_cell_bits = sched.level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;

    let qi_of = |pt: &(u32, u32, u32)| -> usize { compute_qi([pt.0 as u64, pt.1 as u64, pt.2 as u64], lo, shifts, child_cell_bits) };
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
    let record_len = WORD_RECORD_BYTES + if is_far_node { skip_width as usize } else { 0 };
    debug_assert!(record_len <= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES);
    let mut record_buf = [0u8; WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES];
    record_buf[..WORD_RECORD_BYTES].copy_from_slice(&existence_word.to_le_bytes());
    let own_pos = sink.append(&record_buf[..record_len]);
    *node_count += 1;

    let mut total_bytes: u64 = record_len as u64;
    if !is_last {
        let active: Vec<(usize, usize, usize)> = (0..branching).filter(|&qi| offsets[qi] < offsets[qi + 1]).map(|qi| (qi, offsets[qi], offsets[qi + 1])).collect();
        let fan_out = active.len() >= 2 && points.len() >= BUILD_PARALLEL_MIN_POINTS && rayon::current_num_threads() > 1;
        if fan_out {
            let mut rest_points: &mut [(u32, u32, u32)] = points;
            let mut rest_scratch: &mut [(u32, u32, u32)] = scratch;
            let mut prev = 0usize;
            let mut branches: Vec<([u64; NUM_DIMS], &mut [(u32, u32, u32)], &mut [(u32, u32, u32)])> = Vec::with_capacity(active.len());
            for &(qi, _, qend) in &active {
                let (child_points, remainder_points) = rest_points.split_at_mut(qend - prev);
                let (child_scratch, remainder_scratch) = rest_scratch.split_at_mut(qend - prev);
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                branches.push((child_lo, child_points, child_scratch));
                rest_points = remainder_points;
                rest_scratch = remainder_scratch;
                prev = qend;
            }
            let results: Vec<(Vec<u8>, u64)> = branches
                .into_par_iter()
                .map(|(child_lo, child_points, child_scratch)| {
                    let mut local_sink = FlatBuf::new();
                    let mut local_count = 0u64;
                    build_rec(child_points, child_scratch, child_lo, level + 1, sched, &mut local_sink, &mut local_count);
                    // `local_sink.bytes` grew via plain `extend_from_slice`
                    // calls with no capacity reservation, so -- exactly
                    // the `Vec` amortized-doubling overshoot this
                    // project's own `HeapStorage`/`FileStorage` block
                    // allocation deliberately avoids (see
                    // `HeapStorage::append_record`'s doc comment) -- its
                    // *capacity* can run up to ~2x its final *length* by
                    // the time this subtree is done. All these buffers
                    // are momentarily resident together (one per branch,
                    // until the caller's stitch loop below frees them one
                    // at a time), so this reclaims that slack before it
                    // can compound across branches into a real peak-RSS
                    // cost of its own, on top of `results` collecting
                    // every branch's full subtree at once in the first
                    // place (see this function's own doc comment).
                    local_sink.bytes.shrink_to_fit();
                    (local_sink.bytes, local_count)
                })
                .collect();
            for (bytes, count) in results {
                total_bytes += bytes.len() as u64;
                *node_count += count;
                sink.append_subtree(&bytes, level + 1, sched);
            }
        } else {
            for &(qi, qstart, qend) in &active {
                let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
                let child_bytes = build_rec(&mut points[qstart..qend], &mut scratch[qstart..qend], child_lo, level + 1, sched, sink, node_count);
                total_bytes += child_bytes;
            }
        }
    }
    if is_far_node {
        sink.patch(own_pos, WORD_RECORD_BYTES, skip_width, total_bytes);
    }
    total_bytes
}

// ==================== bounded-memory (external-bucketing) bulk build ====================
//
// Everything above this point holds `points`/`scratch` -- O(N) each -- in
// RAM for the whole build, plus (via `build_rec`'s own parallel fan-out)
// a transient extra O(N) while sibling `FlatBuf`s are collected before
// stitching. Measured directly this session: ~56 bytes of peak RAM per
// input triple, independent of backend -- which, projected to a billion
// triples, is tens of gigabytes just to *build* the tree, regardless of
// whether the finished structure ends up on `Heap` or `File`. That
// defeats the entire premise of choosing `File` for RAM control at the
// scale this project actually targets.
//
// What follows generalizes `build_rec`'s own "small enough: build
// directly; big enough: fan out in parallel" decision with a third
// option, gated by a caller-supplied `BulkBuildBudget` rather than a
// fixed constant: a subtree whose point count exceeds the budget is
// never materialized as a `Vec<Triple>` at all. Instead it's spilled,
// with one streaming pass, into up to `branching` (<=64) on-disk bucket
// files -- direct digit-addressed distribution, not a comparison sort,
// so no second pass is needed to find bucket boundaries the way a
// merge-based external sort would -- and each bucket is recursed into
// the same way. The moment a bucket's own count fits the budget, it's
// read back and handed to the ordinary, completely unmodified `build_rec`
// above. This is the standard external-memory distribution-sort pattern
// (Vitter's I/O model; the same "build small, budget-bounded pieces"
// idea HDTCat uses to bulk-load RDF at scale) applied as a natural
// extension of the recursion already here, not a parallel rewrite of it.

/// Caller-supplied bound on how much bulk-build state is ever allowed to
/// be resident in RAM at once, gating [`build_external`]'s spill-to-disk
/// decision the same way [`BUILD_PARALLEL_MIN_POINTS`] gates `build_rec`'s
/// in-RAM fan-out: a plain point-count comparison, never anything
/// dataset-shape-aware.
///
/// # What this does and does not bound
///
/// Bounded to roughly `O(max_resident_points)` per subtree actually held
/// in RAM, plus another `O(max_resident_points)` while a batch of
/// same-level, budget-fitting siblings is fanned out in parallel (see
/// [`flush_fit_batch`], which proactively flushes once a *batch's own
/// aggregate* size would exceed the budget -- not just when a single
/// sibling does -- specifically so this stays `O(budget)` per flush
/// regardless of how many siblings fit into it or how many threads are
/// available; measured directly during this feature's own development
/// that omitting that cap lets peak RSS track N almost exactly instead of
/// plateauing, by batching *every* fitting sibling into one `.collect()`
/// regardless of their combined size), plus a small
/// `O(branching x io_buffer_bytes)` per currently-spilling node for open
/// bucket files (<=64 x `io_buffer_bytes`, ~16 MiB by default). None of
/// this scales with total input size N.
///
/// **`HeapStorage` caveat, stated plainly**: this bounds *build-time
/// transients* only. A `Heap`-backed tree's own steady-state size is
/// still `O(final tree size)` by definition (`HeapStorage` holds the
/// entire structure resident, always) -- this budget cannot and does not
/// change that floor. The N-independent *peak* win this type exists for
/// is a `FileStorage` story specifically; `Heap` only benefits from the
/// smaller build-time transient on top of a steady-state cost it always
/// had anyway.
#[derive(Clone, Copy, Debug)]
pub struct BulkBuildBudget {
    /// A subtree with more than this many points gets spilled to disk
    /// instead of being read into a `Vec<Triple>`.
    pub max_resident_points: usize,
    /// RAM staging buffer per currently-open bucket file, flushed to disk
    /// via one batched `pwrite` once full (see [`SpillBucket::push`]) --
    /// never one syscall per triple.
    pub io_buffer_bytes: usize,
}

impl BulkBuildBudget {
    pub const DEFAULT_IO_BUFFER_BYTES: usize = 256 * 1024;

    /// Degenerates the whole external-bucketing path to a no-op: no
    /// subtree can ever exceed `usize::MAX` points, so
    /// `build_bounded_from_schedule`'s root-level collect never overflows
    /// and it takes the exact same code path `build_from_schedule` always
    /// has. Lets a caller (or a test) use the bounded-build API and
    /// provably get today's unbounded behavior back, rather than having
    /// to keep two separate build call sites around.
    pub const UNBOUNDED: Self = Self { max_resident_points: usize::MAX, io_buffer_bytes: Self::DEFAULT_IO_BUFFER_BYTES };

    pub fn new(max_resident_points: usize) -> Self {
        Self { max_resident_points, io_buffer_bytes: Self::DEFAULT_IO_BUFFER_BYTES }
    }

    /// ~56 bytes of peak RAM per resident point is this session's own
    /// directly measured points+scratch+parallel-fan-out-slack breakdown
    /// (20M-triple scenario, see this file's own history) -- a documented,
    /// revisitable estimate for callers who'd rather reason in "GB of RAM
    /// I'm willing to spend" than in raw point counts, not a guarantee.
    /// [`Self::new`] is always available to bypass the estimate entirely.
    const BYTES_PER_RESIDENT_POINT: usize = 56;

    pub fn from_ram_budget_bytes(bytes: usize) -> Self {
        Self::new((bytes / Self::BYTES_PER_RESIDENT_POINT).max(1))
    }
}

/// One (subject, predicate, object) triple's on-disk row format while
/// spilled to a [`SpillBucket`]: three little-endian `u32`s, back to back.
const SPILL_RECORD_BYTES: usize = 12;

/// One bucket's on-disk staging area during [`distribute_to_buckets`]:
/// raw 12-byte triple rows, appended via a RAM-buffered, batched `pwrite`
/// (mirroring `FileStorage::flush_pending_block`'s style rather than
/// `std::io::BufWriter`, to stay inside this file's own pread/pwrite
/// idiom instead of introducing a second I/O abstraction) -- never one
/// syscall per triple. `count` is a free byproduct of routing triples
/// here, so nothing above ever needs a separate counting pass.
struct SpillBucket {
    file: StdFile,
    buf: Vec<u8>,
    file_len: u64,
    count: u64,
}

impl SpillBucket {
    fn open(io_buffer_bytes: usize) -> Self {
        SpillBucket { file: open_and_unlink_temp_file("s_graph_bulk_spill_bucket"), buf: Vec::with_capacity(io_buffer_bytes), file_len: 0, count: 0 }
    }

    fn push(&mut self, t: Triple, io_buffer_bytes: usize) {
        self.buf.extend_from_slice(&t.0.to_le_bytes());
        self.buf.extend_from_slice(&t.1.to_le_bytes());
        self.buf.extend_from_slice(&t.2.to_le_bytes());
        self.count += 1;
        if self.buf.len() + SPILL_RECORD_BYTES > io_buffer_bytes {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            self.file.write_all_at(&self.buf, self.file_len).expect("failed to flush bulk-build spill bucket to disk");
            self.file_len += self.buf.len() as u64;
            self.buf.clear();
        }
    }

    /// Reads this bucket's entire content back as an owned `Vec<Triple>`,
    /// for handoff to the ordinary in-RAM [`build_rec`] once its `count`
    /// has been confirmed to fit a [`BulkBuildBudget`].
    fn into_vec(mut self) -> Vec<Triple> {
        self.flush();
        let mut raw = vec![0u8; self.file_len as usize];
        if !raw.is_empty() {
            self.file.read_exact_at(&mut raw, 0).expect("failed to read back bulk-build spill bucket");
        }
        raw.chunks_exact(SPILL_RECORD_BYTES).map(decode_spill_record).collect()
    }

    /// Streams this bucket's content back in `io_buffer_bytes`-sized
    /// chunks -- used when a bucket is still over budget and needs to be
    /// spilled again, one level deeper, without ever materializing the
    /// whole thing in RAM at once.
    fn into_stream(mut self, io_buffer_bytes: usize) -> SpillBucketReader {
        self.flush();
        let chunk_bytes = (io_buffer_bytes / SPILL_RECORD_BYTES).max(1) * SPILL_RECORD_BYTES;
        SpillBucketReader { file: self.file, file_len: self.file_len, pos: 0, buf: Vec::new(), buf_pos: 0, chunk_bytes }
    }
}

#[inline]
fn decode_spill_record(c: &[u8]) -> Triple {
    (u32::from_le_bytes(c[0..4].try_into().unwrap()), u32::from_le_bytes(c[4..8].try_into().unwrap()), u32::from_le_bytes(c[8..12].try_into().unwrap()))
}

/// Chunked reader for [`SpillBucket::into_stream`] -- an `Iterator` so it
/// can feed straight back into [`build_external`]/[`distribute_to_buckets`]
/// without ever holding more than one `chunk_bytes`-sized buffer.
struct SpillBucketReader {
    file: StdFile,
    file_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_pos: usize,
    chunk_bytes: usize,
}

impl Iterator for SpillBucketReader {
    type Item = Triple;
    fn next(&mut self) -> Option<Triple> {
        if self.buf_pos >= self.buf.len() {
            if self.pos >= self.file_len {
                return None;
            }
            let take = self.chunk_bytes.min((self.file_len - self.pos) as usize);
            self.buf.resize(take, 0);
            self.file.read_exact_at(&mut self.buf, self.pos).expect("failed to read bulk-build spill bucket chunk");
            self.pos += take as u64;
            self.buf_pos = 0;
        }
        let t = decode_spill_record(&self.buf[self.buf_pos..self.buf_pos + SPILL_RECORD_BYTES]);
        self.buf_pos += SPILL_RECORD_BYTES;
        Some(t)
    }
}

/// One streaming pass over `source`, routing each triple to
/// `buckets[compute_qi(..)]` (a [`SpillBucket`] opened lazily, the first
/// time anything lands in it) -- the external-storage analogue of
/// `build_rec`'s in-RAM counting-sort partition, except direct-addressing
/// into up to `branching` separate files needs no count-then-scatter
/// shuffle the way in-place array partitioning does: routing a triple
/// *is* placing it in its final bucket. Returns the existence word (one
/// bit per non-empty bucket -- identical meaning to `build_rec`'s own)
/// alongside the buckets themselves.
fn distribute_to_buckets<I: Iterator<Item = Triple>>(source: I, lo: [u64; NUM_DIMS], level: usize, sched: &BuildSchedule, io_buffer_bytes: usize) -> (u64, Vec<Option<SpillBucket>>) {
    let dim_bits = sched.level_dim_bits[level];
    let shifts = sched.level_shifts_cache[level];
    let child_cell_bits = sched.level_child_cell_bits_cache[level];
    let branching: usize = 1 << (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;

    let mut buckets: Vec<Option<SpillBucket>> = (0..branching).map(|_| None).collect();
    let mut existence_word = 0u64;
    for t in source {
        let qi = compute_qi([t.0 as u64, t.1 as u64, t.2 as u64], lo, shifts, child_cell_bits);
        buckets[qi].get_or_insert_with(|| SpillBucket::open(io_buffer_bytes)).push(t, io_buffer_bytes);
        existence_word |= 1u64 << qi;
    }
    (existence_word, buckets)
}

/// The external-bucketing counterpart of `build_rec`: same per-node
/// contract (append this node's own record to `sink`, recurse into
/// active children in ascending-qi/DFS order, patch the far-node skip
/// pointer, return total bytes written) but sourcing points from a
/// stream instead of an in-RAM slice, for a subtree whose point count is
/// too large to safely hold as a `Vec<Triple>` under `budget`.
///
/// Active children are walked in qi order, batching consecutive
/// budget-fitting siblings into `pending_batch` for [`flush_fit_batch`]
/// (parallel-eligible, `build_rec`-style); the moment an over-budget
/// sibling is hit, any pending batch is flushed first (to preserve DFS
/// order), then that sibling recurses straight back into *this same
/// `sink`* -- never through a `FlatBuf` -- since an over-budget subtree's
/// encoded bytes are themselves too large to buffer without reopening the
/// exact O(N) cost this whole path exists to avoid.
fn build_external<Sink: RecordSink, I: Iterator<Item = Triple>>(source: I, lo: [u64; NUM_DIMS], level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    let is_last = level == sched.height - 1;
    let dim_bits = sched.level_dim_bits[level];
    let shifts = sched.level_shifts_cache[level];
    let child_cell_bits = sched.level_child_cell_bits_cache[level];

    if is_last {
        // No recursion follows a last-level node -- it needs only
        // *presence*, not routed content, so this skips bucket-file I/O
        // entirely and just folds the existence word, exactly as cheap
        // as `build_rec`'s own `is_last` branch.
        let mut existence_word = 0u64;
        for t in source {
            existence_word |= 1u64 << compute_qi([t.0 as u64, t.1 as u64, t.2 as u64], lo, shifts, child_cell_bits);
        }
        sink.append(&existence_word.to_le_bytes());
        *node_count += 1;
        return WORD_RECORD_BYTES as u64;
    }

    let is_far_node = level + 2 < sched.height;
    let skip_width = sched.level_skip_bytes[level];
    let (existence_word, mut buckets) = distribute_to_buckets(source, lo, level, sched, budget.io_buffer_bytes);

    let record_len = WORD_RECORD_BYTES + if is_far_node { skip_width as usize } else { 0 };
    let mut record_buf = [0u8; WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES];
    record_buf[..WORD_RECORD_BYTES].copy_from_slice(&existence_word.to_le_bytes());
    let own_pos = sink.append(&record_buf[..record_len]);
    *node_count += 1;
    let mut total_bytes = record_len as u64;

    let mut pending_batch: Vec<(usize, [u64; NUM_DIMS], SpillBucket)> = Vec::new();
    let mut pending_points: u64 = 0;
    for (qi, slot) in buckets.iter_mut().enumerate() {
        let Some(bucket) = slot.take() else { continue };
        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        if bucket.count as usize <= budget.max_resident_points {
            pending_points += bucket.count;
            pending_batch.push((qi, child_lo, bucket));
            // Flush proactively once the *batch itself* would exceed the
            // budget, not only when a single sibling does. Without this,
            // a node with many individually-small-enough children (the
            // common case for wide/uniform data) would batch *all* of
            // them into one `flush_fit_batch` call regardless of their
            // combined size -- reintroducing, for the fitting part of the
            // tree, exactly the "whole subtree's bytes resident at once
            // via collected FlatBufs" cost this feature exists to remove
            // (measured directly: without this cap, peak RSS tracked N
            // almost exactly rather than plateauing at the budget).
            // Capping each flush's own aggregate size is what makes
            // `flush_fit_batch`'s own `O(budget)`-per-flush accounting
            // (see its doc comment) actually true, not just asserted.
            if pending_points as usize >= budget.max_resident_points {
                total_bytes += flush_fit_batch(&mut pending_batch, level, sched, budget, sink, node_count);
                pending_points = 0;
            }
        } else {
            total_bytes += flush_fit_batch(&mut pending_batch, level, sched, budget, sink, node_count);
            pending_points = 0;
            total_bytes += build_external(bucket.into_stream(budget.io_buffer_bytes), child_lo, level + 1, sched, budget, sink, node_count);
        }
    }
    total_bytes += flush_fit_batch(&mut pending_batch, level, sched, budget, sink, node_count);

    if is_far_node {
        sink.patch(own_pos, WORD_RECORD_BYTES, skip_width, total_bytes);
    }
    total_bytes
}

/// Stitches every entry of `batch` (each already confirmed to fit
/// `budget`) into `sink`, in original (ascending-qi/DFS) order --
/// `build_rec`'s own parallel-fan-out shape, reapplied one layer up:
/// parallelize via a private `FlatBuf` per branch when there's enough of
/// them and enough combined work to be worth it (same
/// [`BUILD_PARALLEL_MIN_POINTS`] threshold, no separate constant),
/// otherwise handle them sequentially with no extra buffering at all.
/// Empties `batch` either way; returns the total bytes appended.
fn flush_fit_batch<Sink: RecordSink>(batch: &mut Vec<(usize, [u64; NUM_DIMS], SpillBucket)>, level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    if batch.is_empty() {
        return 0;
    }
    let taken = std::mem::take(batch);
    let total_points: u64 = taken.iter().map(|(_, _, b)| b.count).sum();
    let fan_out = taken.len() >= 2 && total_points as usize >= BUILD_PARALLEL_MIN_POINTS && rayon::current_num_threads() > 1;
    let mut total_bytes = 0u64;
    if fan_out {
        // Vec's IntoParallelIterator is indexed, so `.collect()` below
        // preserves `taken`'s own order (ascending qi, by construction) --
        // no re-sort needed, matching `build_rec`'s own fan-out, which
        // doesn't re-sort its results either.
        let results: Vec<(Vec<u8>, u64)> = taken
            .into_par_iter()
            .map(|(_, child_lo, bucket)| {
                let mut local_sink = FlatBuf::new();
                let mut local_count = 0u64;
                build_subtree_bounded(bucket, child_lo, level + 1, sched, budget, &mut local_sink, &mut local_count);
                local_sink.bytes.shrink_to_fit();
                (local_sink.bytes, local_count)
            })
            .collect();
        for (bytes, count) in results {
            total_bytes += bytes.len() as u64;
            *node_count += count;
            sink.append_subtree(&bytes, level + 1, sched);
        }
    } else {
        for (_, child_lo, bucket) in taken {
            total_bytes += build_subtree_bounded(bucket, child_lo, level + 1, sched, budget, sink, node_count);
        }
    }
    total_bytes
}

/// The one dispatch point between "small enough for the ordinary in-RAM
/// path" and "still too big, keep spilling": reads `bucket` back if its
/// own count already fits `budget`, handing off to the existing,
/// completely unmodified [`build_rec`] (a bucket that fits can never need
/// to spill internally, since every descendant's point count is at most
/// its own) -- or recurses into [`build_external`] one level deeper
/// otherwise.
fn build_subtree_bounded<Sink: RecordSink>(bucket: SpillBucket, lo: [u64; NUM_DIMS], level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    if bucket.count as usize <= budget.max_resident_points {
        let mut points = bucket.into_vec();
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        build_rec(&mut points, &mut scratch, lo, level, sched, sink, node_count)
    } else {
        build_external(bucket.into_stream(budget.io_buffer_bytes), lo, level, sched, budget, sink, node_count)
    }
}

// ==================== end bounded-memory bulk build ====================

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
            push_var(&mut chain, level_skip_bytes[lvl], sizes[lvl]);
        }
        lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
    }
    debug_assert_eq!(chain.len() as u64, sizes[start_level], "chain length must match the precomputed size it reports to its own parent");
    chain
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

/// A position in the block-list: which (logical) block, and the byte
/// offset within it. Backend-agnostic -- `S::read_u64`/etc. resolve
/// `block` to wherever that backend actually keeps it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pos {
    block: usize,
    offset: usize,
}

impl Pos {
    const START: Pos = Pos { block: 0, offset: 0 };
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
    fn read_u64(&self, block: usize, offset: usize) -> u64;
    fn read_var(&self, block: usize, offset: usize, width: u8) -> u64;

    /// Reads a node's existence word and, if `skip_width > 0`, its
    /// immediately-following skip field, together. Default impl just
    /// calls [`Self::read_u64`]/[`Self::read_var`] separately -- correct
    /// for any backend, and all `HeapStorage` needs, since there's no
    /// I/O to amortize there. `FileStorage` overrides this to fetch both
    /// in one operation: `insert_no_reset`'s descent reads every level's
    /// word anyway, and (see [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert_new_branch`]'s
    /// ancestor loop) almost always ends up needing that level's skip
    /// value too, so fetching it opportunistically here means that loop
    /// never has to read it again -- only write the updated value.
    /// `skip_width == 0` means this level has no skip field (the last
    /// two levels); the second return value is then meaningless.
    fn read_word_and_skip(&self, block: usize, offset: usize, skip_width: u8) -> (u64, u64) {
        let word = self.read_u64(block, offset);
        let skip = if skip_width > 0 { self.read_var(block, offset + WORD_RECORD_BYTES, skip_width) } else { 0 };
        (word, skip)
    }

    fn write_u64_at(&mut self, block: usize, offset: usize, val: u64);
    fn write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64);
    fn block_len(&self, block: usize) -> usize;
    fn block_count(&self) -> usize;

    /// Maps a logical block index to whatever stable physical identity
    /// the backend has (a no-op for `HeapStorage`, where logical index
    /// *is* physical identity; `FileStorage` overrides this to return the
    /// slot number, which stays stable across the `self.blocks`
    /// reordering/insertion that churns logical indices). Used only for
    /// diagnostics -- e.g. [`advance`](RadixTreeDfsInterleavedVarSkipUnrolledGeneric::advance)'s
    /// panic message -- never on a hot path.
    fn debug_slot(&self, block: usize) -> usize {
        block
    }

    /// Whole-subtree bulk read for [`bulk_list_subtree`](RadixTreeDfsInterleavedVarSkipUnrolledGeneric::bulk_list_subtree):
    /// zero-copy borrow when the whole range lives in one physical block
    /// (`HeapStorage`'s common case), an owned copy otherwise
    /// (`FileStorage`, always; `HeapStorage` only when a range spans
    /// more than one block, which -- since a block never splits a
    /// record -- only happens for a subtree wider than one block).
    fn bulk_read(&self, block: usize, offset: usize, len: usize) -> Cow<'_, [u8]>;

    /// Splices `new_bytes` in at `pos`, growing/splitting storage as
    /// needed -- see each impl for its own mechanism; genuinely
    /// different between backends (see the module doc comment), so this
    /// stays a whole trait method rather than being decomposed further.
    fn insert_bytes(&mut self, pos: Pos, new_bytes: &[u8], block_capacity: usize);

    /// RAII guard, constructed at the top of every top-level call
    /// (`contains`/`query_two_bound`/`query_one_bound`), dropped at its
    /// end. No-op for `HeapStorage`; pops `FileStorage`'s stack cache
    /// back to just the root.
    fn begin_top_level_call(&self) -> impl Drop + '_;
    /// No-op for `HeapStorage`; toggles `FileStorage`'s cache depth
    /// (`1` for reads, unbounded for insert's ancestor-revisit pass).
    fn set_insert_mode(&self, active: bool);
    /// Called at the end of [`insert`](RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert)
    /// (both return paths) since `insert` can't hold a `begin_top_level_call`
    /// guard across its own `&mut self` calls -- see that method's own
    /// comment.
    fn end_insert(&self);

    /// Appends one atomic node record to whichever block the streaming
    /// bulk-build path ([`build_rec`] via [`StorageSink`]) is currently
    /// filling, opening a fresh block first if `bytes` wouldn't fit in
    /// what's left of the current one -- the incremental counterpart of
    /// the old two-pass "flatten the whole tree, then chop it into
    /// blocks" design (a `from_chopped` method that lived here before
    /// this streaming rewrite, since removed): records are
    /// chopped into blocks as they're produced, so this never needs a
    /// second, separate full copy of the encoded tree in RAM (nor a
    /// `record_starts`-style index of it) the way that design did.
    /// `bytes.len()` must never exceed `block_capacity` (guaranteed by
    /// [`build_rec`]: a node record is at most
    /// `WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES`, and every caller of
    /// bulk build already asserts `block_capacity` is at least that).
    fn append_record(&mut self, bytes: &[u8], block_capacity: usize) -> Pos;

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
    /// `append_record` call, leaving storage in the same shape the old
    /// two-pass `from_chopped` design used to produce for the same
    /// sequence of records.
    fn finish_append(&mut self, block_capacity: usize);

    fn empty() -> Self;

    /// (capacity, used_len) per logical block, in DFS order.
    fn block_table(&self) -> Vec<(usize, usize)>;
    fn write_blocks_into(&self, out: &mut Vec<u8>);
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> Self;

    fn mem_size_bytes(&self) -> usize;
    fn size_breakdown(&self) -> Vec<(&'static str, usize)>;

    /// Releases any reserved-but-unused capacity, without changing a
    /// single byte of content. Meant to be called by a caller who knows
    /// they're done with a burst of insertion activity (a whole
    /// `insert_batch` call, or "no more inserts for now") -- calling it
    /// after *every* insert or every batch would be counterproductive
    /// (see [`HeapStorage::insert_bytes`]'s own doc comment: every block
    /// a caller is still actively growing is deliberately pre-reserved to
    /// `block_capacity` precisely so repeated nearby inserts don't each
    /// pay for a reallocation -- shrinking it back down immediately would
    /// just force the next insert to regrow it again). `HeapStorage`
    /// shrinks every block's `Vec<u8>`; `FileStorage` has no equivalent
    /// capacity-vs-content slack to reclaim (its RAM cost is the small
    /// `slot_*`/`blocks` bookkeeping vectors and the stack cache, not
    /// per-block content, and those are shrunk too, though they're not
    /// where any real savings live).
    fn shrink_to_fit(&mut self);

    /// Rebuilds the entire block layout from scratch, packing `records`
    /// -- every node's own `(position, byte length)` in DFS-preorder,
    /// from [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::dfs_records`]
    /// -- greedily into `block_capacity`-sized blocks, exactly the way
    /// the streaming bulk-build path ([`BlockStorage::append_record`],
    /// via [`build_rec`]/[`StorageSink`]) packs node records as they're
    /// produced, except sourcing each record's bytes from the *existing*
    /// storage instead of a live build. Streaming construction
    /// leaves blocks split wherever a given insert happened to first
    /// overflow one -- an artifact of insertion order, not of the
    /// content -- so grouping by pre-existing block boundaries (this
    /// method's earlier form) can't do better than whatever fragmentation
    /// streaming already committed to. Grouping at record granularity
    /// instead closes that gap down to (very close to) what a bulk build
    /// of the identical content would have produced. Called by
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::shrink_to_fit`]
    /// before [`Self::shrink_to_fit`] itself runs -- which then only has
    /// leftover reserved-but-unused capacity to trim, no fragmentation.
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize);
}

// -------------------- HeapStorage --------------------

/// Direct port of the original heap variant's storage: one `Vec<u8>` per
/// logical block, no slot indirection at all (a block's index in
/// `blocks` *is* its address). Growing/splitting a block is plain
/// `Vec::splice`/`split_off`/`insert` on these.
pub struct HeapStorage {
    blocks: Vec<Vec<u8>>,
}

struct NoopGuard;
impl Drop for NoopGuard {
    fn drop(&mut self) {}
}

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
    /// Unlike the *original* heap variant's own `insert_bytes` (which this
    /// started as a faithful port of), a tail-position insert
    /// (`pos.offset == block.len()`) that doesn't fit within
    /// `block_capacity` gets a fresh block of its own instead of being
    /// let through onto the existing block regardless of size. The
    /// original's unconditional tail bypass (`if tail_append || ...`)
    /// meant a block at a repeatedly-hit tail position could grow
    /// arbitrarily far past `block_capacity` via plain `Vec` amortized
    /// doubling, never splitting -- harmless under the random,
    /// scattered single-triple insertion every benchmark exercised
    /// before `insert_batch` existed, but measured to balloon a build's
    /// `mem_size_bytes` by ~9x (53.55 MiB -> 476.21 MiB at 2,000,000
    /// triples) once `insert_batch`'s Morton-sorted insertion order made
    /// hitting the very same block's tail repeatedly, back to back, the
    /// *common* case instead of a rare one -- see
    /// `bench_streaming_insert_build.rs`'s own doc comment for the full
    /// measured story, including the re-validation this fix went
    /// through. `FileStorage::insert_bytes`'s `is_tail` branch already
    /// takes exactly this approach (a fresh slot, never widening the
    /// original past its capacity); this brings `HeapStorage` in line
    /// with it instead of leaving them inconsistent.
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
        // land in it later), not a one-off, so it gets the same
        // "allocate the full block once, fill it, don't reallocate on
        // every subsequent insert into it" treatment as a brand-new
        // block. `split_off` itself only allocates enough for the
        // content it's carrying over, not the full budget.
        tail.reserve_exact(block_capacity.saturating_sub(tail.len()));
        block.extend_from_slice(new_bytes);
        self.blocks.insert(pos.block + 1, tail);
    }
    fn begin_top_level_call(&self) -> impl Drop + '_ {
        NoopGuard
    }
    fn set_insert_mode(&self, _active: bool) {}
    fn end_insert(&self) {}

    /// Every block is allocated at its full `capacity` budget up front
    /// (`Vec::with_capacity`) the moment it's opened, so accumulating
    /// records into it never triggers `Vec`'s own amortized-doubling
    /// reallocation-and-copy along the way (a plain `Vec::new()` +
    /// repeated `extend_from_slice` would -- the same class of overshoot
    /// `insert_bytes`'s own doc comment describes for the insert path).
    /// The trade: a block that ends up smaller than `capacity` (every
    /// block that isn't immediately followed by another one squeezed in
    /// right after it, plus the very last block) carries reserved
    /// capacity it never used -- [`Self::shrink_to_fit`], called from
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
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> Self {
        let mut blocks = Vec::with_capacity(table.len());
        for &(_, len) in table {
            let mut v = vec![0u8; len];
            r.read_exact(&mut v).expect("unexpected EOF while reading serialized tree content");
            blocks.push(v);
        }
        HeapStorage { blocks }
    }

    fn mem_size_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Vec<u8>>() + self.blocks.iter().map(crate::common::metrics::vec_bytes).sum::<usize>()
    }
    fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("blocks: outer Vec<Vec<u8>> pointers", self.blocks.capacity() * std::mem::size_of::<Vec<u8>>()),
            ("blocks: node records (sum of inner Vecs)", self.blocks.iter().map(crate::common::metrics::vec_bytes).sum()),
        ]
    }
}

// -------------------- FileStorage --------------------

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Opens a fresh, uniquely-named temp file and immediately unlinks its
/// path -- the classic Linux idiom for a scratch file that needs zero
/// explicit cleanup: the open `File` handle keeps the underlying inode
/// (and its disk space) alive for as long as it's held, even across a
/// panic, and the space is reclaimed automatically the moment every
/// handle to it closes, with no path ever left behind to leak. Shared by
/// [`open_backing_file`] (a tree's real backing store) and
/// [`SpillBucket::open`] (one bulk-build bucket's scratch storage) so
/// both get this guarantee from one place instead of two copies of the
/// same unlink-after-open dance.
fn open_and_unlink_temp_file(prefix: &str) -> StdFile {
    let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("{prefix}_{}_{id}.bin", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap_or_else(|e| panic!("failed to create scratch file {path:?}: {e}"));
    let _ = std::fs::remove_file(&path);
    file
}

fn open_backing_file() -> StdFile {
    open_and_unlink_temp_file("s_graph_filebacked_generic")
}

/// One cached, page-aligned chunk of a single slot's bytes -- one entry
/// of [`FileStorage::block_stack`].
///
/// # History: two more complicated designs were tried here and reverted
///
/// This module went through two other cache designs before landing back
/// here, both real correctness bugs, both root-caused the hard way (see
/// the project history behind this file for the full account) -- worth
/// recording so the next redesign doesn't re-walk the same path:
///
/// 1. **Fixed-capacity, always-whole-slot, write-back entries**
///    (`buf` always exactly `slot_capacity[slot]` bytes, a `dirty` flag,
///    flushed lazily on eviction). Motivated by an even earlier bug in
///    *this* windowed design (see point 3 below) -- but going all the way
///    to "cache the whole slot" both reintroduced a *worse* version of
///    the same class of bug (a stale, unflushed entry surviving eviction
///    and later getting silently resurrected -- fixed once by neutralizing
///    every orphaned entry per slot on a miss, not just a dirty one) and
///    cost real, measured performance: resolving a whole up-to-64KB slot
///    to touch 8 bytes measured at ~8x a small windowed read.
/// 2. **A second, write-capable windowed cache layered on top of (1)**
///    specifically for the ancestor skip-pointer pattern, to claw back
///    the lost performance without reverting (1) itself. This introduced
///    a *second* correctness bug (two windows for the same slot could
///    overlap, since window size wasn't fixed) that was found and fixed,
///    and evidence of at least one more that wasn't fully root-caused
///    before reverting both (1) and (2) back to this file's original
///    shape.
///
/// The common failure mode across both: any design where a write can be
/// *deferred* (a dirty flag, a "flush later" promise) has to prove "at
/// most one authoritative copy of a slot's content exists at a time" as
/// an actively-maintained invariant across every path that creates or
/// evicts an entry -- and every attempt here found a path that didn't.
///
/// # Why this design doesn't have that problem
///
/// Writes here (`FileStorage::slot_write_all`) go straight to disk with
/// one `pwrite`, *immediately*, every time -- never deferred, no `dirty`
/// flag, no flush-on-eviction. `buf`/`chunk_start` are then just a
/// read-side accelerator: patched in place if the write falls inside
/// what's cached (`Self::slot_write_all`), or the entry is dropped
/// (`buf.clear()`, `chunk_start` becomes meaningless as a bonus, no
/// separate invalidation flag needed) if it doesn't. Disk is *always*
/// the single authoritative copy, so there is no "which copy is current"
/// question for any invalidation logic to get wrong: a cache entry is
/// either an accurate mirror of some byte range of disk, or empty.
/// Nothing here can ever be silently resurrected as authoritative,
/// because nothing here is ever authoritative in the first place.
///
/// This does re-accept the original small-window tradeoff measured
/// against a whole-slot cache: `contains`/`insert` (little same-slot
/// reuse across tree levels) ~9-17x faster than a whole-slot cache, but
/// row queries (many siblings of one parent, genuinely reusing the same
/// slot) up to ~2-6.5x slower than one would be. That tradeoff -- not a
/// bug -- is why [`FileStorage::stack_cap`] exists: a *stack* of these
/// entries (mirroring the actual shape of a DFS descent, see
/// [`FileStorage::ensure_stack_entry`]) instead of one sliding window
/// recovers most of the sibling-scan locality a single whole-slot cache
/// would give, without paying its point-read cost.
struct StackCacheEntry {
    slot: usize,
    /// Byte offset within the slot where `buf` begins (a multiple of
    /// [`READ_CHUNK_BYTES`], except when widened past that to cover a
    /// read/write that straddled the aligned boundary -- see
    /// [`FileStorage::ensure_stack_entry`]). Meaningless whenever `buf`
    /// is empty (a freshly pushed or invalidated entry).
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
/// (`contains`/`query_two_bound`/`query_one_bound`), that pops
/// [`FileStorage::block_stack`] back down to just its root entry when
/// that call returns -- on every exit path, since this runs on drop.
/// `insert`/`insert_batch` don't use this (see
/// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert_no_reset`]'s
/// own doc comment) and pop the same way manually via
/// [`FileStorage::end_insert`]. Sound down to `top.min(1)` (not `0`)
/// specifically because every top-level call -- including `insert`,
/// which reads the root's own word first thing every time -- always
/// touches at least one entry before this ever runs, so `entries` is
/// never empty at this point; see `Self::end_insert`'s own comment.
struct StackCacheGuard<'a>(&'a FileStorage);

impl Drop for StackCacheGuard<'_> {
    fn drop(&mut self) {
        let mut stack = self.0.block_stack.borrow_mut();
        stack.top = stack.top.min(1);
    }
}

/// Direct port of the original file variant's storage (see that file's
/// own module doc comment for the full page-chunked-stack-cache
/// rationale this one shares byte-for-byte): a real backing `File`, a
/// slot table (`slot_start`/`slot_capacity`/`slot_used_len`) with
/// `blocks[i]` mapping logical block `i` to a physical slot, and the
/// stack of small windowed [`StackCacheEntry`] reads. Writes
/// ([`Self::slot_write_all`]) are write-through -- straight to disk
/// every time, the cache just kept in sync opportunistically -- see
/// [`StackCacheEntry`]'s own doc comment for why that's a deliberate,
/// hard-won choice, not an oversight.
pub struct FileStorage {
    file: StdFile,
    file_len: u64,
    slot_start: Vec<u64>,
    slot_capacity: Vec<usize>,
    slot_used_len: Vec<usize>,
    blocks: Vec<usize>,
    block_stack: RefCell<BlockStack>,
    stack_cap: Cell<usize>,
    /// The streaming bulk-build path's ([`build_rec`] via [`StorageSink`])
    /// currently-open block, held in RAM and not yet given a real file
    /// slot -- `Some` only between the first `append_record` of a build
    /// and the matching `finish_append`; `None` at every other time
    /// (including the whole steady-state lifetime of a tree, and during
    /// `insert`/`insert_batch`, which go through [`Self::insert_bytes`]
    /// instead and never touch this field). Buffering one block's worth
    /// of records in RAM before issuing a single batched `pwrite` (see
    /// [`Self::flush_pending_block`]) avoids a `pwrite` syscall per node
    /// record, which -- at up to `WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES`
    /// bytes per record -- would otherwise dominate build time at any
    /// real scale.
    build_pending: Option<Vec<u8>>,
}

impl FileStorage {
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

    /// Moves [`FileStorage::build_pending`]'s buffered block to a real
    /// file slot with one batched `pwrite` (`write_all_at`), then
    /// registers it in [`FileStorage::blocks`] -- from this point on it's
    /// an ordinary, already-committed block like any other, addressable
    /// via the normal `write_var_at`/`read_u64`/etc. paths. Every logical
    /// block index [`BlockStorage::append_record`] hands out via a `Pos`
    /// while this block was still pending (`pos.block == self.blocks.len()`
    /// at the time) becomes valid the instant this call returns, since
    /// it's exactly `self.blocks.len()` (pre-push) that this new slot
    /// gets pushed to.
    fn flush_pending_block(&mut self, capacity: usize) {
        let buf = self.build_pending.take().expect("flush_pending_block called with no pending block");
        let slot = self.allocate_slot(capacity);
        if !buf.is_empty() {
            self.file.write_all_at(&buf, self.slot_start[slot]).expect("failed to flush pending block to backing file");
        }
        self.slot_used_len[slot] = buf.len();
        self.blocks.push(slot);
    }

    /// One `pread`, unconditionally, bypassing [`Self::block_stack`]
    /// entirely. Correct on its own -- no cache check needed -- only
    /// because [`Self::slot_write_all`] is write-through: disk is always
    /// the current, authoritative copy of every slot, so there is no
    /// "pending write" state a raw read could ever miss. Used for reads
    /// that are already multi-byte/one-shot by nature where a small
    /// window buys nothing: [`Self::slot_insert_at`]'s tail-move,
    /// `write_blocks_into` (serialization), `bulk_read` (listing
    /// queries), and [`FileStorage::compact_from_records`]'s bulk copy.
    /// [`BlockStorage::read_u64`]/[`BlockStorage::read_var`] are the
    /// cache-aware entry points for per-node field reads.
    fn slot_read_exact(&self, slot: usize, offset: usize, buf: &mut [u8]) {
        self.file.read_exact_at(buf, self.slot_start[slot] + offset as u64).expect("file-backed block read failed");
    }

    /// Writes `bytes` at `offset` within `slot` -- one `pwrite`,
    /// immediately, every time. Then, if `slot` has an *active*
    /// [`Self::block_stack`] entry (`stack.entries[..top]`; anything
    /// beyond `top` is inert and never read again, see
    /// [`Self::ensure_stack_entry`]), patches the written range directly
    /// into that entry's cached buffer -- rather than merely dropping
    /// it -- so a read of the same slot immediately after a write (e.g.
    /// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert_new_branch`]'s
    /// read-then-write ancestor-skip update) is served from cache instead
    /// of paying a redundant `pread`. Falls back to dropping the cached
    /// buffer (forcing a re-fetch on the next read) only when the written
    /// range doesn't fully fall within what's currently cached -- never
    /// a correctness issue, just a bit less cache reuse in that case.
    ///
    /// Write-through, not write-back, deliberately: see
    /// [`StackCacheEntry`]'s own doc comment for the two more
    /// complicated (and, each in its own way, buggy) designs this
    /// replaced. With every write immediately durable, there is exactly
    /// one authoritative copy of any slot's content -- disk -- at every
    /// instant, so no invalidation logic anywhere in this file has to
    /// reason about "is this cached copy still current."
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
    /// (`pread`) whatever's missing.
    ///
    /// This is the single place implementing the whole
    /// push-on-descend/pop-on-backtrack policy: if `slot` is already the
    /// top entry, only the chunk-coverage check below runs; if `slot` is
    /// present deeper in the stack (a backtrack -- true by construction,
    /// since a block only ever leaves the stack once the traversal has
    /// fully left its subtree), every entry above it is popped, exposing
    /// the already-resident bytes with no `pread` needed for the pop
    /// itself; otherwise it's genuinely new territory and gets pushed
    /// fresh -- unless the stack already holds [`Self::stack_cap`]
    /// non-root entries, in which case it's first popped down to just
    /// the root, reproducing a single sliding window when `stack_cap` is
    /// `1` (read-only calls; see that field's own doc comment) or the
    /// full unbounded stack when it's `MAX_HEIGHT` (`insert`/
    /// `insert_batch`, which need every ancestor on the path to the
    /// insertion point to stay resident for the skip-pointer-update pass
    /// that revisits them all a second time). A "push" never allocates a
    /// new [`StackCacheEntry`] if one is already physically there from a
    /// deeper call that has since popped past it -- overwritten in
    /// place, `buf` only `clear`ed (dropping length, keeping heap
    /// capacity) rather than replaced.
    ///
    /// Either way, the resulting top entry's cached chunk is then
    /// widened with one `pread` if it doesn't already cover `[offset,
    /// offset + len)` -- bounded to [`READ_CHUNK_BYTES`] rather than the
    /// whole slot (see [`StackCacheEntry`]'s own doc comment for why),
    /// widening past one chunk only in the rare case a read straddles
    /// the aligned boundary.
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
        }
        entry.chunk_start = chunk_start;
    }

    /// In-place insertion at `offset` within `slot`: reads the existing
    /// `[offset, used)` tail into a scratch buffer, writes it back
    /// shifted right by `bytes.len()`, then writes `bytes` into the
    /// vacated gap. Bypasses [`Self::block_stack`] entirely (via
    /// [`Self::slot_read_exact`]/[`Self::slot_write_all`] directly) --
    /// this is already one multi-byte read and two multi-byte writes, a
    /// small window buys nothing over the same one/two `pread`/`pwrite`
    /// calls a cache lookup would also need.
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
            let mut tail_buf = vec![0u8; tail_len];
            self.slot_read_exact(slot, offset, &mut tail_buf);
            self.slot_write_all(slot, offset + bytes.len(), &tail_buf);
        }
        self.slot_write_all(slot, offset, bytes);
        self.slot_used_len[slot] = used + bytes.len();
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
            // No need to touch `slot`'s own cached entry here, if it has
            // one: nothing legitimate ever reads at-or-past the new
            // (shrunk) `used` again (that content now lives in
            // `new_slot`), so any stale tail bytes still sitting in a
            // cached window past that point are simply never read.
            // If `slot` is later extended again, `Self::slot_write_all`
            // writes the real disk content through first and only then
            // patches whatever's cached -- so a cache entry is always
            // refreshed with exactly what was actually written, never
            // trusted to already be right.
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
    fn begin_top_level_call(&self) -> impl Drop + '_ {
        self.stack_cap.set(1);
        StackCacheGuard(self)
    }
    fn set_insert_mode(&self, active: bool) {
        self.stack_cap.set(if active { MAX_HEIGHT } else { 1 });
    }
    /// Manually pops [`FileStorage::block_stack`] back down to just the
    /// root entry, the same place [`StackCacheGuard`] leaves it for
    /// read-only calls -- `insert`/`insert_batch` can't hold that guard
    /// (its `&self` borrow would conflict with the `&mut self` they need
    /// throughout), so they call this explicitly at every return point
    /// instead. Safe to set unconditionally, no `entries.len()` check
    /// needed: every `insert` call reads the root's own word first thing
    /// (`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::insert_no_reset`'s
    /// descent), which always resolves `entries[0]` before this could
    /// ever run.
    fn end_insert(&self) {
        self.block_stack.borrow_mut().top = 1;
    }

    /// Appends `bytes` to [`FileStorage::build_pending`] (opening it, or
    /// flushing-then-reopening it via [`Self::flush_pending_block`], as
    /// needed), then reports the record's position as belonging to
    /// logical block `self.blocks.len()` -- the index the pending block
    /// will actually get the moment it's flushed, so a `Pos` handed out
    /// here stays valid (readable via [`Self::patch_appended`], and later
    /// via the normal `write_var_at`/`read_u64` paths once the whole
    /// build finishes) whether or not this particular block has been
    /// flushed by the time something looks it up.
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
    /// Patches a position returned by a still-in-progress
    /// [`Self::append_record`] sequence. `pos.block == self.blocks.len()`
    /// means it's still sitting in [`FileStorage::build_pending`] (patch
    /// the RAM buffer directly); any other value means it was already
    /// flushed to a real slot by a later `append_record` call, so this
    /// goes through the same write-through path `write_var_at` uses.
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
        }
    }
    /// Only trims leftover bookkeeping now -- [`Self::compact_from_records`]
    /// (called first by [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::shrink_to_fit`],
    /// the only caller of either) already rebuilt the whole block layout
    /// from scratch and reset [`Self::block_stack`] to empty in the
    /// process, so there's nothing left here to compact -- just the small
    /// `slot_*`/`blocks` vectors, now much shorter than they were before
    /// compaction (nothing to *flush* either way: `Self::slot_write_all`
    /// is write-through, so there's never anything dirty sitting in the
    /// cache to begin with).
    fn shrink_to_fit(&mut self) {
        self.slot_start.shrink_to_fit();
        self.slot_capacity.shrink_to_fit();
        self.slot_used_len.shrink_to_fit();
        self.blocks.shrink_to_fit();
        self.block_stack.get_mut().entries.shrink_to_fit();
    }
    fn compact_from_records(&mut self, records: &[(Pos, usize)], block_capacity: usize) {
        if records.is_empty() {
            return;
        }
        // Greedily group consecutive records (in DFS-preorder, exactly
        // as passed in) whose combined length still fits one
        // `block_capacity` budget -- record-granular, unlike grouping
        // whole pre-existing blocks, so this can pack right up to (and
        // only stop at) an actual record boundary, the same freedom a
        // bulk build has.
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
                file.write_all_at(&buf, cursor).expect("failed to write compacted block content");
            }
            slot_start.push(cursor);
            slot_capacity.push(cap);
            slot_used_len.push(total_len);
            cursor += cap as u64;
        }
        file.set_len(cursor).expect("failed to size compacted backing file");

        self.file = file;
        self.file_len = cursor;
        self.slot_start = slot_start;
        self.slot_capacity = slot_capacity;
        self.slot_used_len = slot_used_len;
        self.blocks = (0..groups.len()).collect();
        // Every cached entry refers to a slot number from the layout
        // that just got replaced -- none of them mean anything anymore.
        // Nothing to flush first: `Self::slot_write_all` is write-through,
        // so the old file (read from above via `Self::slot_read_exact`)
        // was already the complete, current picture.
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
    fn read_blocks_from<R: std::io::Read>(r: &mut R, table: &[(usize, usize)]) -> Self {
        let mut slot_start = Vec::with_capacity(table.len());
        let mut total: u64 = 0;
        for &(cap, _) in table {
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
        FileStorage {
            file,
            file_len: total,
            slot_start,
            slot_capacity,
            slot_used_len,
            blocks,
            block_stack: RefCell::new(BlockStack::new()),
            stack_cap: Cell::new(1),
            build_pending: None,
        }
    }

    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.slot_start) + crate::common::metrics::vec_bytes(&self.slot_capacity) + crate::common::metrics::vec_bytes(&self.slot_used_len) + crate::common::metrics::vec_bytes(&self.blocks) + self.stack_cache_bytes()
    }
    fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("slot_start (O(slots))", crate::common::metrics::vec_bytes(&self.slot_start)),
            ("slot_capacity (O(slots))", crate::common::metrics::vec_bytes(&self.slot_capacity)),
            ("slot_used_len (O(slots))", crate::common::metrics::vec_bytes(&self.slot_used_len)),
            ("blocks: logical-to-slot indirection (O(blocks))", crate::common::metrics::vec_bytes(&self.blocks)),
            ("block_stack (windowed read cache)", self.stack_cache_bytes()),
        ]
    }
}

impl FileStorage {
    /// Bounded by O(height) entries at up to [`READ_CHUNK_BYTES`] each
    /// (not O(dataset size)) -- a small, stack-shaped read cache, never
    /// holding anything not already durably on disk (`Self::slot_write_all`
    /// is write-through), so this is purely a RAM-vs-syscall bookkeeping
    /// number, not a "how much unflushed data exists" one.
    fn stack_cache_bytes(&self) -> usize {
        self.block_stack.borrow().entries.iter().map(|e| e.buf.capacity()).sum()
    }
}

// ==================== the generic tree ====================

pub struct RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S: BlockStorage> {
    storage: S,
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

/// Not colliding with the originals' own `Heap`/`File` names -- use
/// `radixtree_dfs_interleaved_varskip_unrolled_generic::Heap`/`::File`.
pub type Heap = RadixTreeDfsInterleavedVarSkipUnrolledGeneric<HeapStorage>;
pub type File = RadixTreeDfsInterleavedVarSkipUnrolledGeneric<FileStorage>;

impl<S: BlockStorage> crate::common::radixtree::RadixTreeApi for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        Self::contains(self, s, p, o)
    }
    fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        Self::query_two_bound(self, bound_dims, bound_vals, free_dim)
    }
    fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        Self::query_one_bound(self, bound_dim, bound_val, target_dim)
    }
    fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        Self::predicates_for(self, s, o)
    }
    fn node_word_count(&self) -> u64 {
        Self::node_word_count(self)
    }
    fn domain_size(&self, dim: usize) -> u64 {
        Self::domain_size(self, dim)
    }
    fn height(&self) -> u32 {
        Self::height(self)
    }
}

impl<S: BlockStorage> MemSize for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes()
            + crate::common::metrics::vec_bytes(&self.level_skip_bytes)
            + crate::common::metrics::vec_bytes(&self.level_dim_bits)
            + crate::common::metrics::vec_bytes(&self.level_shifts_cache)
            + crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)
    }
}

impl RadixTreeDfsInterleavedVarSkipUnrolledGeneric<FileStorage> {
    /// Bytes actually occupied in the backing file -- disk-space
    /// counterpart to [`MemSize::mem_size_bytes`]'s RAM-only number. Only
    /// meaningful for the `File` backend, so this lives outside the
    /// generic `impl<S: BlockStorage>` block rather than in the shared
    /// trait.
    pub fn disk_bytes(&self) -> u64 {
        self.storage.file_len
    }
}

impl<S: BlockStorage> RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    pub fn default_split() -> [u8; NUM_DIMS] {
        [BASE / NUM_DIMS as u8; NUM_DIMS]
    }

    /// Takes `triples` by value, not `&[Triple]`: [`build_from_schedule`]
    /// needs its own owned, mutable, sortable copy of the data for the
    /// in-place counting sort regardless (it can't mutate a caller's
    /// borrowed slice), so a by-reference signature would force an extra
    /// full clone inside every single build -- measured as a real,
    /// avoidable ~229 MiB (at 20M triples) contributor to bulk build's
    /// peak RSS, on top of the counting sort's own same-sized `scratch`
    /// flip buffer. Taking ownership means `triples` itself becomes
    /// `build_from_schedule`'s `points` array via a plain move, zero
    /// copies. A caller who still needs the original data after building
    /// (as this project's own benchmarks do, for post-build probes) just
    /// clones before calling -- exactly one clone either way, but now it's
    /// the caller's explicit choice rather than an unconditional cost
    /// hidden inside every build.
    pub fn build(triples: Vec<Triple>, split: [u8; NUM_DIMS], block_capacity: usize) -> Self {
        Self::build_with_order(triples, split, [0, 1, 2], block_capacity)
    }

    pub fn build_with_order(triples: Vec<Triple>, split: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        debug_assert!(split.iter().map(|&s| s as u32).sum::<u32>() <= BASE as u32);
        let dimension_sizes = dimension_sizes_from(&triples);
        let level_dim_bits = static_schedule(total_bits_needed(dimension_sizes), split);
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    pub fn build_dynamic(triples: Vec<Triple>, block_capacity: usize) -> Self {
        Self::build_dynamic_with_order(triples, [0, 1, 2], block_capacity)
    }

    pub fn build_dynamic_with_order(triples: Vec<Triple>, dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        let dimension_sizes = dimension_sizes_from(&triples);
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    /// An empty tree (zero triples) whose per-level bit schedule and
    /// height are fully determined by `dimension_sizes` alone -- the
    /// starting point for a *streaming* build: generate triples one (or a
    /// few) at a time and [`Self::insert`] each as it's produced, instead
    /// of materializing a full `Vec<Triple>` (and the O(N) scratch buffer
    /// [`build_from_schedule`]'s bulk path needs for its counting sort)
    /// before building anything at all. See
    /// `examples/bench_streaming_insert_build.rs`.
    ///
    /// Unlike [`Self::build_dynamic`], `dimension_sizes` has to be given
    /// explicitly rather than inferred from data (there's no data yet) --
    /// callers need to already know their domain's upper bound, exactly
    /// as any real streaming-ingestion pipeline already must (a
    /// dictionary-encoded RDF store knows its id space's size before it
    /// knows every id that will ever appear in it).
    ///
    /// This works out to calling the ordinary bulk build path with an
    /// empty triple slice: `build_interleaved_rec` over zero points still
    /// emits exactly one node (the root, existence word `0`, no
    /// children) -- the correctly-formatted, single-record tree
    /// `insert()` expects to already find at `Pos::START`.
    pub fn build_dynamic_empty(dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        Self::build_dynamic_empty_with_order(dimension_sizes, [0, 1, 2], block_capacity)
    }

    pub fn build_dynamic_empty_with_order(dimension_sizes: [u64; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(Vec::new(), dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    /// The bounded-memory sibling of [`Self::build_dynamic`], for
    /// datasets too large to materialize as a single `Vec<Triple>` at
    /// all (see [`BulkBuildBudget`]'s own doc comment for exactly what
    /// "bounded" does and doesn't mean here). Takes `triples` as a plain
    /// `Iterator` rather than a `Vec` -- a real billion-triple caller
    /// can't hand over a many-gigabyte `Vec` any more than this method
    /// can afford to build one internally -- and `dimension_sizes`
    /// explicitly, for the same reason [`Self::build_dynamic_empty`]
    /// already requires it: a single-pass streaming source can't be
    /// scanned twice to first learn the domain and then build.
    ///
    /// `BulkBuildBudget::UNBOUNDED` degenerates this to exactly
    /// [`Self::build_dynamic`]'s own behavior (see that constant's own
    /// doc comment) -- useful for a caller who wants one call site
    /// regardless of dataset size, or for tests proving the two paths
    /// agree.
    pub fn build_dynamic_bounded<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        Self::build_dynamic_bounded_with_order(triples, dimension_sizes, [0, 1, 2], block_capacity, budget)
    }

    pub fn build_dynamic_bounded_with_order<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_bounded_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity, budget)
    }

    fn build_bounded_from_schedule<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        debug_assert!(block_capacity >= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES);
        let mut triples = triples;
        if level_dim_bits.is_empty() {
            return RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
                storage: S::empty(),
                block_capacity,
                node_count: 0,
                level_skip_bytes: Vec::new(),
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                dim_order,
                single_point_exists: triples.next().is_some(),
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

        // Collect at most `max_resident_points + 1` triples up front --
        // one pass, no double materialization either way: if that's
        // everything (`prefix.len() <= max_resident_points`), `prefix`
        // simply *is* what `points` would have been for the ordinary
        // fast path below, at zero extra cost over `build_from_schedule`.
        // Only when there's strictly more left (the `+1` overflowed) does
        // this fall through to external bucketing, chaining the already-
        // collected prefix back in front of whatever's left in `triples`
        // so no triple is skipped or duplicated.
        let mut prefix: Vec<Triple> = Vec::new();
        for t in triples.by_ref() {
            prefix.push(t);
            if prefix.len() > budget.max_resident_points {
                break;
            }
        }

        let mut node_count = 0u64;
        let mut storage = S::empty();
        {
            let sched = BuildSchedule { height: height as usize, level_dim_bits: &level_dim_bits, level_shifts_cache: &level_shifts_cache, level_child_cell_bits_cache: &level_child_cell_bits_cache, level_skip_bytes: &level_skip_bytes };
            let mut sink = StorageSink { storage: &mut storage, block_capacity };
            if prefix.len() <= budget.max_resident_points {
                // FITS: the exact same fast path `build_from_schedule`
                // takes, no external-bucketing machinery ever touched.
                let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); prefix.len()];
                build_rec(&mut prefix, &mut scratch, [0; NUM_DIMS], 0, &sched, &mut sink, &mut node_count);
            } else {
                // SPILL: stream straight into real storage from the root
                // -- no root-level `FlatBuf` needed, `build_external` is
                // generic over `Sink` the same way `build_rec` is.
                build_external(prefix.into_iter().chain(triples), [0; NUM_DIMS], 0, &sched, &budget, &mut sink, &mut node_count);
            }
        }
        storage.finish_append(block_capacity);

        RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
            storage,
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

    fn build_from_schedule(triples: Vec<Triple>, dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        debug_assert!(block_capacity >= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES);
        if level_dim_bits.is_empty() {
            return RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
                storage: S::empty(),
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

        // A plain move, not a clone: `Triple` already *is* `(u32, u32,
        // u32)`, and `triples` is owned here (see this fn's caller-side
        // doc comment on `Self::build`), so the counting sort gets to
        // reuse the caller's own allocation as its mutable working array
        // instead of paying for a second, identical one.
        let mut points: Vec<(u32, u32, u32)> = triples;
        let mut scratch: Vec<(u32, u32, u32)> = vec![(0, 0, 0); points.len()];
        let mut node_count = 0u64;
        let mut storage = S::empty();
        {
            let sched = BuildSchedule { height: height as usize, level_dim_bits: &level_dim_bits, level_shifts_cache: &level_shifts_cache, level_child_cell_bits_cache: &level_child_cell_bits_cache, level_skip_bytes: &level_skip_bytes };
            let mut sink = StorageSink { storage: &mut storage, block_capacity };
            build_rec(&mut points, &mut scratch, [0; NUM_DIMS], 0, &sched, &mut sink, &mut node_count);
        }
        storage.finish_append(block_capacity);

        RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
            storage,
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
    pub fn block_count(&self) -> usize {
        self.storage.block_count()
    }

    #[inline]
    fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        (s as u64) < self.dimension_sizes[0] && (p as u64) < self.dimension_sizes[1] && (o as u64) < self.dimension_sizes[2]
    }

    #[inline]
    fn block_len(&self, logical: usize) -> usize {
        self.storage.block_len(logical)
    }
    #[inline]
    fn block_read_u64(&self, block: usize, offset: usize) -> u64 {
        self.storage.read_u64(block, offset)
    }
    #[inline]
    fn block_read_var(&self, block: usize, offset: usize, width: u8) -> u64 {
        self.storage.read_var(block, offset, width)
    }
    #[inline]
    fn block_read_word_and_skip(&self, block: usize, offset: usize, skip_width: u8) -> (u64, u64) {
        self.storage.read_word_and_skip(block, offset, skip_width)
    }
    #[inline]
    fn block_write_u64_at(&mut self, block: usize, offset: usize, val: u64) {
        self.storage.write_u64_at(block, offset, val);
    }
    #[inline]
    fn block_write_var_at(&mut self, block: usize, offset: usize, width: u8, val: u64) {
        self.storage.write_var_at(block, offset, width, val);
    }

    #[inline]
    fn own_record_bytes(&self, level: usize) -> usize {
        if level + 2 < self.height as usize {
            WORD_RECORD_BYTES + self.level_skip_bytes[level] as usize
        } else {
            WORD_RECORD_BYTES
        }
    }

    #[inline]
    fn advance(&self, mut pos: Pos, mut n: usize) -> Pos {
        loop {
            let bl = self.block_len(pos.block);
            assert!(pos.offset <= bl, "advance: pos.offset {} > block_len {bl} for block {} (n={n})", pos.offset, pos.block);
            let avail = bl - pos.offset;
            if n < avail {
                pos.offset += n;
                return pos;
            }
            n -= avail;
            if pos.block + 1 >= self.storage.block_count() {
                assert_eq!(n, 0, "advance() ran past the end of all blocks (block={}, slot={}, avail={avail})", pos.block, self.storage.debug_slot(pos.block));
                pos.offset += avail;
                return pos;
            }
            pos.block += 1;
            pos.offset = 0;
        }
    }

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
        let _guard = self.storage.begin_top_level_call();
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
        let _guard = self.storage.begin_top_level_call();
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

    pub fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        let _guard = self.storage.begin_top_level_call();
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

    fn query_rec(&self, start_pos: Pos, start_lo: u64, start_level: usize, level_masks: &[u64], free_dim: usize, unconstrained_from: usize, out: &mut Vec<u32>) {
        let mut stack: [ConstrainedFrame; MAX_HEIGHT] = [ConstrainedFrame::EMPTY; MAX_HEIGHT];
        let mut stack_len: usize = 0;
        let mut pos = start_pos;
        let mut lo = start_lo;
        let mut level = start_level;

        'outer: loop {
            let mut descended = false;

            if level >= unconstrained_from {
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
                        let child_lo = advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
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
                            let child_lo = advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
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
                        let child_lo = advance_lo_dim(frame.lo, frame_dim_bits, frame_shifts, qi, frame_child_cell_bits);
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

    /// Bulk-reads one entire "listing" subtree (see
    /// `radixtree_dfs_interleaved_varskip_unrolled_file.rs`'s own module
    /// doc comment for the full rationale) in one shot and decodes it
    /// entirely from memory -- shared, generic code, so both backends
    /// get it here (see this module's own doc comment for the one place
    /// that makes `Heap`'s behavior differ from the *original* heap
    /// variant, which doesn't have this).
    fn bulk_list_subtree(&self, pos: Pos, lo: u64, level: usize, free_dim: usize, out: &mut Vec<u32>) {
        let span_end = self.child_span_end(pos, level);
        let byte_len = self.byte_distance(pos, span_end);
        let buf = self.storage.bulk_read(pos.block, pos.offset, byte_len);
        self.decode_listing(&buf, 0, lo, level, free_dim, out);
    }

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
                out.push(advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits) as u32);
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
                let child_lo = advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                let leaf_word = u64::from_le_bytes(buf[cursor..cursor + WORD_RECORD_BYTES].try_into().unwrap());
                let mut relevant = leaf_word;
                while relevant != 0 {
                    let leaf_qi = relevant.trailing_zeros() as usize;
                    relevant &= relevant - 1;
                    out.push(advance_lo_dim(child_lo, leaf_dim_bits, leaf_shifts, leaf_qi, leaf_child_cell_bits) as u32);
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
                let child_lo = advance_lo_dim(lo, dim_bits, shifts, qi, child_cell_bits);
                cursor = self.decode_listing(buf, cursor, child_lo, level + 1, free_dim, out);
            }
            cursor
        }
    }

    /// Inserts one point. No `begin_top_level_call` guard here (unlike
    /// the read methods): this needs `&mut self` throughout (e.g.
    /// `self.insert_new_branch(...)` below), which a guard holding a
    /// live borrow across the call would conflict with -- `set_insert_mode`/
    /// `end_insert` are momentary `&self` calls instead, matching what
    /// the original file variant's own `insert` already does for the
    /// same reason.
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> bool {
        self.storage.set_insert_mode(true);
        let result = self.insert_no_reset(s, p, o);
        self.storage.end_insert();
        result
    }

    /// Sorts `triples` into this tree's own DFS-preorder (Morton/
    /// interleaved-coordinate) order -- see [`Self::morton_key`] -- then
    /// inserts them one at a time via [`Self::insert_no_reset`], resetting
    /// the file backend's block-stack cache only once, after the whole
    /// batch, instead of after every single triple.
    ///
    /// This changes *only* physical insertion order and how often the
    /// cache is reset -- not which bits end up set anywhere. The
    /// resulting tree is byte-for-byte equivalent (same `node_word_count`,
    /// same query results) to inserting the same points one at a time in
    /// any other order, or to a bulk `build_dynamic` over the same set:
    /// [`Self::insert_new_branch`] only ever reads/writes the one root-to-
    /// insertion-point path a point's own coordinates determine, which
    /// doesn't depend on what was inserted before it or in what order.
    /// See `streaming_batch_insert_matches_per_triple_insert_and_bulk_build`
    /// in this module's own test suite.
    ///
    /// Two effects, both purely about *how* the insertion work is spent,
    /// not *what* the result is:
    /// - **File backend**: consecutive triples in sorted order tend to
    ///   share the longest possible common ancestor prefix with each
    ///   other, so [`Self::ensure_stack_entry`]'s cache -- which the plain
    ///   per-triple [`Self::insert`] deliberately truncates back to just
    ///   the root after every single call (see its own doc comment) --
    ///   stays warm across many consecutive inserts instead of missing on
    ///   every ancestor above the root every single time, cutting real
    ///   `pread`/`pwrite` syscall counts.
    /// - **Both backends, block packing**: [`Self::insert_bytes`] grows
    ///   whichever block the insertion point currently falls in. Sorted,
    ///   spatially-adjacent triples are far more likely to fall into the
    ///   *same* freshly-grown block as their immediate predecessor than
    ///   randomly-ordered ones are, so a batch produces fewer, better-
    ///   filled blocks than the same triples inserted in arbitrary order
    ///   would -- see `bench_streaming_insert_build.rs`'s own measured
    ///   numbers for the size of this effect.
    ///
    /// Still O(batch memory), not O(whole dataset): callers choose their
    /// own batch size (a few thousand triples is enough to see most of
    /// the benefit -- see the bench binary), never needing to hold the
    /// full triple set at once the way `build_dynamic` does.
    pub fn insert_batch(&mut self, triples: &mut [Triple]) -> u64 {
        if triples.is_empty() {
            return 0;
        }
        triples.sort_unstable_by_key(|&(s, p, o)| self.morton_key(s, p, o));
        self.storage.set_insert_mode(true);
        let mut inserted = 0u64;
        for &(s, p, o) in triples.iter() {
            if self.insert_no_reset(s, p, o) {
                inserted += 1;
            }
        }
        self.storage.end_insert();
        inserted
    }

    /// Reclaims reserved-but-unused block capacity accumulated by
    /// [`Self::insert`]/[`Self::insert_batch`] -- call this once, when
    /// the caller knows they're done with a burst of insertion activity
    /// (e.g. after a whole streamed dataset has been ingested), not
    /// after every individual call. See [`BlockStorage::shrink_to_fit`]'s
    /// own doc comment for why "after every insert" would be
    /// counterproductive rather than merely unnecessary. A bulk
    /// [`Self::build`]/[`Self::build_dynamic`] already ends up shrunk
    /// automatically -- this is only needed after the streaming/insert
    /// path.
    pub fn shrink_to_fit(&mut self) {
        if self.height > 0 {
            // `dfs_records`' walk is read-only and, in file order, purely
            // sequential -- exactly the shape `BlockStack`'s own stack
            // discipline already handles well (see its doc comment): under
            // this guard's `stack_cap == 1`, the walk settles into the
            // same small sliding window every other read-only call uses,
            // no extra reset needed (the last `insert`/`insert_batch`
            // already left `top` at `1` -- see `FileStorage::end_insert`).
            let records = {
                let _guard = self.storage.begin_top_level_call();
                self.dfs_records()
            };
            self.storage.compact_from_records(&records, self.block_capacity);
        }
        self.storage.shrink_to_fit();
    }

    /// Every node's own `(position, byte length)`, walked in DFS-preorder
    /// -- derived on the fly from the live tree (just
    /// `own_record_bytes`/`block_read_u64`/`advance`, all backend-agnostic
    /// and already used elsewhere for navigation), never stored anywhere.
    /// Feeds [`BlockStorage::compact_from_records`]: packing at this
    /// granularity, instead of whatever boundaries pre-existing physical
    /// blocks happen to have, is what lets `shrink_to_fit` close (most
    /// of) the gap to what a bulk build of the same content would have
    /// produced. Memory is `O(node_count)` pairs of small integers (16
    /// bytes each), not the record bytes themselves, and the walk's own
    /// recursion is bounded by `height` (<= `MAX_HEIGHT`) -- neither
    /// materializes the tree's actual content.
    fn dfs_records(&self) -> Vec<(Pos, usize)> {
        let mut out = Vec::with_capacity(self.node_count as usize + 1);
        self.dfs_records_rec(Pos::START, 0, &mut out);
        out
    }

    fn dfs_records_rec(&self, pos: Pos, level: usize, out: &mut Vec<(Pos, usize)>) -> Pos {
        let word = self.block_read_u64(pos.block, pos.offset);
        let len = self.own_record_bytes(level);
        out.push((pos, len));
        let mut child_pos = self.advance(pos, len);
        if level == self.height as usize - 1 {
            return child_pos;
        }
        for _ in 0..word.count_ones() {
            child_pos = self.dfs_records_rec(child_pos, level + 1, out);
        }
        child_pos
    }

    /// This point's `qi` (child slot index) at every level, in descent
    /// order -- exactly the sequence [`build_interleaved_rec`]'s own
    /// per-level counting sort partitions by, so sorting a batch of
    /// points lexicographically by this key reproduces the identical
    /// relative order a bulk build over the same points would visit them
    /// in (i.e. this tree's own physical DFS-preorder / "Morton code" for
    /// the current schedule). Pure geometry -- reads only the schedule
    /// (`level_shifts_cache`/`level_child_cell_bits_cache`), never the
    /// tree's actual stored content, so it's identical whether or not any
    /// of these points already exist in the tree.
    fn morton_key(&self, s: u32, p: u32, o: u32) -> [u8; MAX_HEIGHT] {
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut key = [0u8; MAX_HEIGHT];
        for level in 0..self.height as usize {
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let mut qi = 0usize;
            let mut next_lo = lo;
            for d in 0..NUM_DIMS {
                let sub = (coords[d] - lo[d]) >> child_cell_bits[d];
                qi |= (sub as usize) << shifts[d];
                next_lo[d] += sub << child_cell_bits[d];
            }
            key[level] = qi as u8;
            lo = next_lo;
        }
        key
    }

    /// The shared body of [`Self::insert`] and [`Self::insert_batch`] --
    /// everything except setting/resetting the file backend's insert-mode
    /// cache depth, which the two callers deliberately do at different
    /// granularities (every call vs. once per batch). See their own doc
    /// comments.
    fn insert_no_reset(&mut self, s: u32, p: u32, o: u32) -> bool {
        debug_assert!(self.height > 0, "insert() on an empty/degenerate tree is out of scope for this prototype");
        debug_assert!(self.in_bounds(s, p, o), "insert({s},{p},{o}) is outside the domain established at build time");
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut pos = Pos::START;
        // Fourth element: this level's *own* skip-field value, read
        // alongside its word in one combined fetch (see
        // `block_read_word_and_skip`) whenever this level has one
        // (`level + 2 < height`; 0 otherwise, meaningless). Almost every
        // trail entry ends up needing this in `insert_new_branch`'s
        // ancestor loop anyway -- fetching it now, piggybacked on the
        // word read this loop was already doing, means that loop never
        // needs to re-read it at all, just write the updated value.
        let mut trail: [(Pos, usize, u64, u64); MAX_HEIGHT] = [(Pos::START, 0, 0, 0); MAX_HEIGHT];
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
            let skip_width = if level + 2 < self.height as usize { self.level_skip_bytes[level] } else { 0 };
            let (word, skip_old) = self.block_read_word_and_skip(pos.block, pos.offset, skip_width);
            trail[trail_len] = (pos, level, word, skip_old);
            trail_len += 1;

            if (word >> qi) & 1 == 0 {
                self.insert_new_branch(pos, level, qi, word, coords, lo, &trail[..trail_len]);
                return true;
            }
            if is_last {
                return false;
            }
            let earlier = word & ((1u64 << qi) - 1);
            let first_child_pos = self.advance(pos, self.own_record_bytes(level));
            pos = self.advance_past_children(first_child_pos, earlier, level + 1);
            lo = next_lo;
        }
        unreachable!("height > 0 checked above")
    }

    fn insert_new_branch(&mut self, node_pos: Pos, level: usize, qi: usize, word: u64, coords: [u64; NUM_DIMS], lo: [u64; NUM_DIMS], trail: &[(Pos, usize, u64, u64)]) {
        self.set_bit_in_word(node_pos, qi, word);
        let is_last = level == self.height as usize - 1;
        if is_last {
            return;
        }

        let dim_bits = self.level_dim_bits[level];
        let shifts = self.level_shifts_cache[level];
        let child_cell_bits = self.level_child_cell_bits_cache[level];
        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        let chain = build_single_point_chain(coords, level + 1, self.height, &self.level_dim_bits, &self.level_shifts_cache, &self.level_child_cell_bits_cache, &self.level_skip_bytes, child_lo);

        let earlier = word & ((1u64 << qi) - 1);
        let first_child_pos = self.advance(node_pos, self.own_record_bytes(level));
        let insert_pos = self.advance_past_children(first_child_pos, earlier, level + 1);

        self.storage.insert_bytes(insert_pos, &chain, self.block_capacity);
        self.node_count += (self.height as usize - (level + 1)) as u64;

        for &(anc_pos, anc_level, _, skip_old) in trail.iter().rev() {
            if anc_level + 2 < self.height as usize {
                let width = self.level_skip_bytes[anc_level];
                let skip_offset = anc_pos.offset + WORD_RECORD_BYTES;
                self.block_write_var_at(anc_pos.block, skip_offset, width, skip_old + chain.len() as u64);
            }
        }
    }

    #[inline]
    fn set_bit_in_word(&mut self, pos: Pos, qi: usize, old: u64) {
        let new = old | (1u64 << qi);
        self.block_write_u64_at(pos.block, pos.offset, new);
    }

    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.storage.size_breakdown();
        v.push(("level_skip_bytes (O(height))", crate::common::metrics::vec_bytes(&self.level_skip_bytes)));
        v.push(("level_dim_bits (O(height))", crate::common::metrics::vec_bytes(&self.level_dim_bits)));
        v.push(("level_shifts_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_shifts_cache)));
        v.push(("level_child_cell_bits_cache (O(height))", crate::common::metrics::vec_bytes(&self.level_child_cell_bits_cache)));
        v
    }

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
        let table = self.storage.block_table();
        let capacities: Vec<usize> = table.iter().map(|&(c, _)| c).collect();
        let used_lens: Vec<usize> = table.iter().map(|&(_, l)| l).collect();
        crate::common::tree_serialize::write_block_table(&mut out, &capacities, &used_lens);
        self.storage.write_blocks_into(&mut out);
        out
    }

    pub fn deserialize(buf: &[u8]) -> Self {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf))
    }

    pub fn deserialize_from_path(path: &std::path::Path) -> Self {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path:?}: {e}"));
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file))
    }

    fn deserialize_from_reader<R: std::io::Read>(r: &mut R) -> Self {
        let h = crate::common::tree_serialize::read_header_from(r);
        let table = crate::common::tree_serialize::read_block_table_from(r);
        let storage = S::read_blocks_from(r, &table);
        RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
            storage,
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

    const TEST_CAPACITIES: [usize; 4] = [32, 64, 256, DEFAULT_BLOCK_CAPACITY];

    fn cross_check_tree<S: BlockStorage>(tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
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

    fn cross_check_both_backends(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, label: &str) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let heap = Heap::build_dynamic(triples.to_vec(), cap);
            cross_check_tree(&heap, &naive, num_s, num_p, num_o, &format!("heap cap={cap} {label}"));
            let file = File::build_dynamic(triples.to_vec(), cap);
            cross_check_tree(&file, &naive, num_s, num_p, num_o, &format!("file cap={cap} {label}"));
        }
    }

    #[test]
    fn cross_check_uniform_and_zipfian() {
        for seed in 1..5u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            cross_check_both_backends(&triples, 30, 6, 30, "uniform");
            let triples = zipfian_predicates(40, 10, 40, 600, 1.2, seed * 2027);
            cross_check_both_backends(&triples, 40, 10, 40, "zipfian");
        }
    }

    #[test]
    fn insert_one_at_a_time_matches_naive_after_every_insert() {
        let mut rng = SplitMix64::new(0xC0FFEE);
        let all = uniform(20, 4, 20, 300, 555);
        let split_at = all.len() / 2;
        let (initial, rest) = all.split_at(split_at);

        let mut heap = Heap::build_dynamic(initial.to_vec(), 64);
        let mut file = File::build_dynamic(initial.to_vec(), 64);
        let mut seen: Vec<Triple> = initial.to_vec();
        for &(s, p, o) in rest {
            let _ = &mut rng;
            let heap_new = heap.insert(s, p, o);
            let file_new = file.insert(s, p, o);
            assert_eq!(heap_new, file_new, "heap/file insert({s},{p},{o}) agreement");
            if heap_new {
                seen.push((s, p, o));
            }
            let naive = NaiveIndex::build(&seen);
            cross_check_tree(&heap, &naive, 20, 4, 20, "heap after insert");
            cross_check_tree(&file, &naive, 20, 4, 20, "file after insert");
        }
    }

    #[test]
    fn streaming_insert_build_matches_bulk_build_and_naive() {
        // Builds the *same* dataset two ways -- today's bulk path
        // (materialize every triple, then `build_dynamic`) and the
        // streaming path this test exists to validate
        // (`build_dynamic_empty` + one `insert()` per triple, as
        // `examples/bench_streaming_insert_build.rs` does at scale) --
        // and checks both land on a tree indistinguishable from
        // `NaiveIndex`, for both backends.
        for seed in 1..4u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [30u64, 6u64, 30u64];

            let mut heap_stream = Heap::build_dynamic_empty(dimension_sizes, 64);
            let mut file_stream = File::build_dynamic_empty(dimension_sizes, 64);
            for &(s, p, o) in &triples {
                heap_stream.insert(s, p, o);
                file_stream.insert(s, p, o);
            }
            cross_check_tree(&heap_stream, &naive, 30, 6, 30, &format!("heap streaming-insert build seed={seed}"));
            cross_check_tree(&file_stream, &naive, 30, 6, 30, &format!("file streaming-insert build seed={seed}"));

            let heap_bulk = Heap::build_dynamic(triples.to_vec(), 64);
            assert_eq!(heap_stream.mem_size_bytes() > 0, heap_bulk.mem_size_bytes() > 0);
            assert_eq!(heap_stream.node_word_count(), heap_bulk.node_word_count(), "streaming vs. bulk build should produce the same node count, seed={seed}");
        }
    }

    #[test]
    fn streaming_batch_insert_matches_per_triple_insert_and_bulk_build() {
        // insert_batch sorts into a different physical insertion order
        // than one-triple-at-a-time and only resets the file backend's
        // cache once per batch instead of once per triple -- neither
        // should change a single bit of the resulting tree's content.
        for seed in 1..4u64 {
            let mut triples = uniform(40, 6, 40, 600, seed * 9001);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [40u64, 6u64, 40u64];

            let mut heap_batch = Heap::build_dynamic_empty(dimension_sizes, 64);
            let batch_inserted = heap_batch.insert_batch(&mut triples);
            cross_check_tree(&heap_batch, &naive, 40, 6, 40, &format!("heap batch-insert build seed={seed}"));

            let mut file_batch = File::build_dynamic_empty(dimension_sizes, 64);
            let file_batch_inserted = file_batch.insert_batch(&mut triples);
            cross_check_tree(&file_batch, &naive, 40, 6, 40, &format!("file batch-insert build seed={seed}"));
            assert_eq!(batch_inserted, file_batch_inserted, "heap/file insert_batch new-triple count agreement, seed={seed}");

            let heap_bulk = Heap::build_dynamic(triples.to_vec(), 64);
            assert_eq!(heap_batch.node_word_count(), heap_bulk.node_word_count(), "batch-insert vs. bulk build should produce the same node count, seed={seed}");

            // Sorting a batch that's already sorted (idempotent re-insert
            // of the same points, now in a different starting order) must
            // report zero *new* insertions and leave the tree unchanged.
            let mut same_again = triples.clone();
            let reinserted = heap_batch.insert_batch(&mut same_again);
            assert_eq!(reinserted, 0, "re-inserting the same batch must find nothing new, seed={seed}");
            cross_check_tree(&heap_batch, &naive, 40, 6, 40, &format!("heap batch-insert idempotency seed={seed}"));
        }
    }

    #[test]
    fn file_streaming_insert_stress_small_capacity() {
        // Small block_capacity forces splits (and stack-cache wraps) far
        // more often per triple than production capacities would --
        // originally added to cheaply reproduce a real corruption bug in
        // an earlier, now-reverted write-back cache design (a stale
        // entry could survive eviction and later get silently
        // resurrected as authoritative). `FileStorage`'s current
        // write-through cache (see `StackCacheEntry`'s own doc comment)
        // is structurally immune to that whole bug class -- there is no
        // deferred/dirty state to go stale -- but this remains a cheap,
        // useful general stress test for splits/wraps interleaved with
        // an immediate `contains()` self-check, so it stays.
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        let mut file = File::build_dynamic_empty(dimension_sizes, 64);
        let mut rng = SplitMix64::new(2026);
        let mut seen: Vec<Triple> = Vec::new();
        for i in 0..3_000u32 {
            let s = rng.next_range(200_000) as u32;
            let p = rng.next_range(19) as u32;
            let o = rng.next_range(200_000) as u32;
            let is_new = file.insert(s, p, o);
            if is_new {
                seen.push((s, p, o));
            }
            if !file.contains(s, p, o) {
                panic!("self-check failed at triple #{i}: ({s},{p},{o}) not found immediately after insert");
            }
        }
        let naive = NaiveIndex::build(&seen);
        cross_check_tree(&file, &naive, 200, 19, 200, "file small-capacity stress (partial domain slice)");
    }

    #[test]
    fn shrink_to_fit_changes_capacity_not_content() {
        for seed in 1..4u64 {
            let mut triples = uniform(50, 8, 50, 800, seed * 12345);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [50u64, 8u64, 50u64];

            let mut heap = Heap::build_dynamic_empty(dimension_sizes, 64);
            heap.insert_batch(&mut triples);
            let mem_before = heap.mem_size_bytes();
            heap.shrink_to_fit();
            let mem_after = heap.mem_size_bytes();
            assert!(mem_after <= mem_before, "shrink_to_fit should never increase mem_size_bytes, seed={seed} ({mem_after} > {mem_before})");
            cross_check_tree(&heap, &naive, 50, 8, 50, &format!("heap after shrink_to_fit seed={seed}"));

            let mut file = File::build_dynamic_empty(dimension_sizes, 64);
            file.insert_batch(&mut triples);
            file.shrink_to_fit();
            cross_check_tree(&file, &naive, 50, 8, 50, &format!("file after shrink_to_fit seed={seed}"));

            // shrink_to_fit must be safe to call and then keep inserting
            // into -- it shouldn't leave the structure unable to grow
            // further.
            let extra = uniform(50, 8, 50, 50, seed * 999);
            for &(s, p, o) in &extra {
                heap.insert(s, p, o);
                file.insert(s, p, o);
            }
            let mut all: Vec<Triple> = triples.clone();
            all.extend_from_slice(&extra);
            let naive_after = NaiveIndex::build(&all);
            cross_check_tree(&heap, &naive_after, 50, 8, 50, &format!("heap after shrink_to_fit then more inserts seed={seed}"));
            cross_check_tree(&file, &naive_after, 50, 8, 50, &format!("file after shrink_to_fit then more inserts seed={seed}"));
        }
    }

    #[test]
    fn serialize_round_trip() {
        for cap in TEST_CAPACITIES {
            let all = uniform(30, 6, 30, 400, 8080 + cap as u64);
            let split_at = all.len() / 2;
            let (initial, rest) = all.split_at(split_at);
            let naive = NaiveIndex::build(initial);

            let heap = Heap::build_dynamic(initial.to_vec(), cap);
            let heap_bytes = heap.serialize();
            let mut heap_restored = Heap::deserialize(&heap_bytes);
            cross_check_tree(&heap_restored, &naive, 30, 6, 30, &format!("heap round-trip cap={cap}"));

            let file = File::build_dynamic(initial.to_vec(), cap);
            let file_bytes = file.serialize();
            let mut file_restored = File::deserialize(&file_bytes);
            cross_check_tree(&file_restored, &naive, 30, 6, 30, &format!("file round-trip cap={cap}"));

            let mut seen: Vec<Triple> = initial.to_vec();
            for &(s, p, o) in rest {
                if heap_restored.insert(s, p, o) {
                    seen.push((s, p, o));
                }
                file_restored.insert(s, p, o);
            }
            let naive_after = NaiveIndex::build(&seen);
            cross_check_tree(&heap_restored, &naive_after, 30, 6, 30, &format!("heap round-trip after insert cap={cap}"));
            cross_check_tree(&file_restored, &naive_after, 30, 6, 30, &format!("file round-trip after insert cap={cap}"));
        }
    }

    /// Every other test in this module builds well under
    /// [`BUILD_PARALLEL_MIN_POINTS`] triples, so none of them ever
    /// exercise [`build_rec`]'s parallel fan-out branch -- only its
    /// always-correct-regardless-of-scale sequential/streaming path. This
    /// test's whole purpose is to actually cross the threshold (a uniform
    /// dataset wide enough in S that the root node alone has many active
    /// buckets, guaranteeing fan-out triggers at the very first level)
    /// and verify the result is still bit-for-bit consistent with
    /// [`NaiveIndex`] for both backends -- i.e. that stitching parallel
    /// branches' [`FlatBuf`]s back into real storage (record-boundary-
    /// aware for `File`, since [`RecordSink::append_subtree`] has to walk
    /// them; trivial concatenation for the nested case) reproduces
    /// exactly the same tree a fully sequential build would.
    #[test]
    fn bulk_build_parallel_path_matches_naive_uniform() {
        let triples = uniform(100, 6, 100, BUILD_PARALLEL_MIN_POINTS + 10_000, 0xFACE);
        let naive = NaiveIndex::build(&triples);
        let heap = Heap::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&heap, &naive, 100, 6, 100, "heap parallel-path uniform");
        let file = File::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&file, &naive, 100, 6, 100, "file parallel-path uniform");
        assert_eq!(heap.node_word_count(), file.node_word_count(), "heap/file parallel-path node count agreement");
    }

    /// Same idea as [`bulk_build_parallel_path_matches_naive_uniform`],
    /// but with a dataset concentrated enough (a tiny S domain) that the
    /// root -- and likely several levels below it -- may have only one
    /// active bucket, so [`build_rec`] can't fan out there at all. This
    /// exercises the "keep descending sequentially, re-checking fan-out
    /// at every level" behavior: fan-out should still trigger *somewhere*
    /// once a level with >= 2 active buckets is reached (here, the O
    /// dimension), and the result must still match [`NaiveIndex`] exactly
    /// -- concentrated/skewed data must never be treated as a special
    /// case by this structure (see this project's own "always measure,
    /// never assume" standard re: the k²-tree family's traversal
    /// pathology on skewed data, which this design avoids structurally).
    #[test]
    fn bulk_build_parallel_path_matches_naive_skewed() {
        let triples = zipfian_predicates(3, 4, 4000, BUILD_PARALLEL_MIN_POINTS + 10_000, 1.5, 0xBEEF);
        let naive = NaiveIndex::build(&triples);
        let heap = Heap::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&heap, &naive, 3, 4, 4000, "heap parallel-path skewed");
        let file = File::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&file, &naive, 3, 4, 4000, "file parallel-path skewed");
        assert_eq!(heap.node_word_count(), file.node_word_count(), "heap/file parallel-path (skewed) node count agreement");
    }

    // ==================== bounded-memory (external-bucketing) build ====================

    /// `BulkBuildBudget::UNBOUNDED` must be a strict no-op relative to
    /// `build_dynamic`: no subtree can ever exceed `usize::MAX` points,
    /// so `build_bounded_from_schedule`'s root-level collect always takes
    /// the exact same fast path `build_from_schedule` does, never
    /// touching `build_external`/`SpillBucket` at all.
    #[test]
    fn bounded_build_unbounded_matches_dynamic_build() {
        for seed in 1..4u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            let naive = NaiveIndex::build(&triples);
            let dims = [30u64, 6u64, 30u64];

            let heap_dynamic = Heap::build_dynamic(triples.clone(), 64);
            let heap_bounded = Heap::build_dynamic_bounded(triples.clone().into_iter(), dims, 64, BulkBuildBudget::UNBOUNDED);
            assert_eq!(heap_dynamic.node_word_count(), heap_bounded.node_word_count(), "unbounded budget must match build_dynamic exactly, seed={seed}");
            cross_check_tree(&heap_bounded, &naive, 30, 6, 30, &format!("heap unbounded-budget seed={seed}"));

            let file_dynamic = File::build_dynamic(triples.clone(), 64);
            let file_bounded = File::build_dynamic_bounded(triples.into_iter(), dims, 64, BulkBuildBudget::UNBOUNDED);
            assert_eq!(file_dynamic.node_word_count(), file_bounded.node_word_count(), "unbounded budget must match build_dynamic exactly (file), seed={seed}");
            cross_check_tree(&file_bounded, &naive, 30, 6, 30, &format!("file unbounded-budget seed={seed}"));
        }
    }

    /// A deliberately tiny budget (well under the dataset size) forces
    /// `build_external` to recurse through several levels of on-disk
    /// bucketing even for a small dataset -- and the result must be
    /// byte-for-byte identical (not just query-equivalent) to what
    /// `build_dynamic`'s ordinary in-RAM path produces for the same
    /// triples, proven via `serialize()` rather than only cross-checking
    /// query behavior.
    #[test]
    fn bounded_build_byte_identical_with_tiny_budget() {
        for seed in 1..4u64 {
            let triples = uniform(40, 6, 40, 600, seed * 9001);
            let dims = [40u64, 6u64, 40u64];
            let tiny_budget = BulkBuildBudget::new(25);

            let heap_dynamic = Heap::build_dynamic(triples.clone(), 64);
            let heap_bounded = Heap::build_dynamic_bounded(triples.clone().into_iter(), dims, 64, tiny_budget);
            assert_eq!(heap_dynamic.serialize(), heap_bounded.serialize(), "bounded build with a tiny budget must serialize identically to build_dynamic, seed={seed}");

            let file_dynamic = File::build_dynamic(triples.clone(), 64);
            let file_bounded = File::build_dynamic_bounded(triples.into_iter(), dims, 64, tiny_budget);
            assert_eq!(file_dynamic.serialize(), file_bounded.serialize(), "bounded build with a tiny budget must serialize identically to build_dynamic (file), seed={seed}");
        }
    }

    /// The bounded-build analogue of `bulk_build_parallel_path_matches_naive_uniform`:
    /// a budget far smaller than the dataset forces `build_external` to
    /// spill and recurse, with plenty of same-level siblings fitting the
    /// budget at once (exercising `flush_fit_batch`'s own parallel-fan-out
    /// branch too, since `BUILD_PARALLEL_MIN_POINTS` is still the gate).
    #[test]
    fn bulk_build_bounded_path_matches_naive_uniform() {
        let triples = uniform(100, 6, 100, BUILD_PARALLEL_MIN_POINTS + 10_000, 0xFACE);
        let naive = NaiveIndex::build(&triples);
        let dims = [100u64, 6u64, 100u64];
        let budget = BulkBuildBudget::new(5_000);

        let heap = Heap::build_dynamic_bounded(triples.clone().into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&heap, &naive, 100, 6, 100, "heap bounded-path uniform");
        let file = File::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&file, &naive, 100, 6, 100, "file bounded-path uniform");
        assert_eq!(heap.node_word_count(), file.node_word_count(), "heap/file bounded-path node count agreement");
    }

    /// The bounded-build analogue of `bulk_build_parallel_path_matches_naive_skewed`:
    /// a tiny S domain means the root (and likely several levels below
    /// it) has only one active bucket, forcing `build_external`'s
    /// sequential "keep spilling straight into `sink`" branch to run for
    /// many consecutive levels before any `flush_fit_batch` ever fires --
    /// concentrated/skewed data must not be a special case here either.
    #[test]
    fn bulk_build_bounded_path_matches_naive_skewed() {
        let triples = zipfian_predicates(3, 4, 4000, BUILD_PARALLEL_MIN_POINTS + 10_000, 1.5, 0xBEEF);
        let naive = NaiveIndex::build(&triples);
        let dims = [3u64, 4u64, 4000u64];
        let budget = BulkBuildBudget::new(5_000);

        let heap = Heap::build_dynamic_bounded(triples.clone().into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&heap, &naive, 3, 4, 4000, "heap bounded-path skewed");
        let file = File::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&file, &naive, 3, 4, 4000, "file bounded-path skewed");
        assert_eq!(heap.node_word_count(), file.node_word_count(), "heap/file bounded-path (skewed) node count agreement");
    }

    /// One node with a mix of active children: some fit the budget
    /// (batched, possibly parallel, via `flush_fit_batch`), one doesn't
    /// (forced sequential, straight into `sink`). Constructed via a
    /// skewed subject domain (one dominant S value drowning out a long
    /// tail of distinct ones) so a single top-level bucket is far larger
    /// than its siblings. This is the one control-flow shape
    /// (batch/spill interleaving within a single node) with no analogue
    /// in the parallel-fan-out tests -- what's being checked is that
    /// interleaving a flush with a sequential spill still produces exact
    /// DFS/qi order, not just "some tree that happens to match query-wise".
    #[test]
    fn bulk_build_bounded_mixed_fits_and_spills_siblings() {
        let mut rng = SplitMix64::new(0xC0DE_1234);
        let mut triples: Vec<Triple> = Vec::new();
        // One hot subject, heavily duplicated across predicates/objects --
        // guaranteed to overflow a small budget on its own.
        for _ in 0..4_000u32 {
            triples.push((0, rng.next_range(6) as u32, rng.next_range(50) as u32));
        }
        // A long tail of distinct, low-volume subjects -- each comfortably
        // fits any reasonable budget on its own.
        for s in 1..60u32 {
            for _ in 0..20u32 {
                triples.push((s, rng.next_range(6) as u32, rng.next_range(50) as u32));
            }
        }
        let naive = NaiveIndex::build(&triples);
        let dims = [60u64, 6u64, 50u64];
        let budget = BulkBuildBudget::new(500);

        let heap = Heap::build_dynamic_bounded(triples.clone().into_iter(), dims, 64, budget);
        cross_check_tree(&heap, &naive, 60, 6, 50, "heap bounded-path mixed fits/spills");
        let file = File::build_dynamic_bounded(triples.clone().into_iter(), dims, 64, budget);
        cross_check_tree(&file, &naive, 60, 6, 50, "file bounded-path mixed fits/spills");

        let heap_dynamic = Heap::build_dynamic(triples, 64);
        assert_eq!(heap.node_word_count(), heap_dynamic.node_word_count(), "bounded build with mixed fits/spills siblings must match the ordinary in-RAM build's node count");
    }

    /// A small budget together with wide (64-way) branching at every
    /// level forces many `SpillBucket`s open concurrently, including
    /// across `flush_fit_batch`'s own parallel fan-out -- a soft guard
    /// against real file-descriptor exhaustion (see this feature's own
    /// module doc comment for the honest, non-zero risk at very high
    /// thread counts) rather than a hard limit test.
    #[test]
    fn bulk_build_bounded_wide_branching_many_open_buckets() {
        let triples = uniform(4_000, 60, 4_000, 200_000, 0x0FD_5EED);
        let naive_probe_s: u32 = 17;
        let naive = NaiveIndex::build(&triples);
        let dims = [4_000u64, 60u64, 4_000u64];
        let budget = BulkBuildBudget::new(2_000);

        let file = File::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        // A full exhaustive cross-check here would be O(4000*60*4000)
        // contains calls -- far more than this test needs to prove the
        // point -- so this spot-checks one row instead of the full grid.
        let mut got = file.query_two_bound([0, 1], [naive_probe_s, 3], 2);
        let mut expect = naive.row(3, naive_probe_s);
        got.sort_unstable();
        expect.sort_unstable();
        assert_eq!(got, expect, "wide-branching bounded build query_two_bound(S={naive_probe_s},P=3,free=O)");
    }
}
