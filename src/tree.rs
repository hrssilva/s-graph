//! The core generic radix trie: the per-level bit schedule, the bulk
//! build (in-RAM, parallel-fan-out, and RAM-bounded/external-bucketing
//! variants), and [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric`]
//! itself -- `contains`/`query_two_bound`/`query_one_bound`'s descent,
//! `insert`/`insert_batch`'s ancestor trail, and serialization.
//! Generic over [`crate::storage::BlockStorage`]; see [`crate::Heap`],
//! [`crate::File`], [`crate::Compressed`] for the concrete backends.

use crate::error::{OutOfDomain, Result, TreeError};
use crate::storage::{BlockStorage, Pos};
#[cfg(feature = "heap")]
use crate::storage::HeapStorage;
#[cfg(feature = "compressed")]
use crate::storage_compressed::CompressedStorage;
#[cfg(feature = "file")]
use crate::storage_file::FileStorage;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::fs::{File as StdFile, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const K_SQUARE: u8 = 64;
pub(crate) const BASE: u8 = K_SQUARE.trailing_zeros() as u8;
pub(crate) const NUM_DIMS: usize = 3; // S, P, O
pub(crate) const MAX_HEIGHT: usize = 64;
pub(crate) const WORD_RECORD_BYTES: usize = 8;
/// One level's entry in `Self::remove`/`Self::remove_batch`'s root-to-
/// leaf trail: `(position, level, this level's own existence word, its
/// skip field if it has one, the `qi` used to descend past it)`.
type RemoveTrailEntry = (Pos, usize, u64, u64, usize);
type RemoveTrail = [RemoveTrailEntry; MAX_HEIGHT];
const MAX_SKIP_RECORD_BYTES: usize = 8;
pub const DEFAULT_BLOCK_CAPACITY: usize = 4096;

/// One `(subject, predicate, object)` triple.
pub type Triple = (u32, u32, u32);

// ==================== shared, backend-agnostic helpers ====================

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
#[cfg(any(feature = "heap", feature = "file", feature = "compressed", feature = "parallel"))]
pub(crate) fn write_var_at(buf: &mut [u8], pos: usize, width: u8, val: u64) {
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
#[cfg(any(feature = "heap", feature = "file", feature = "compressed"))]
pub(crate) fn read_var_at(buf: &[u8], pos: usize, width: u8) -> u64 {
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
    assert!(dimension_sizes.iter().all(|&n| (1..=1u64 << 32).contains(&n)), "domains must fit u32 coordinates and be nonzero");
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

fn validate_config(domains: [u64; 3], mut order: [usize; 3], capacity: usize) -> Result<()> {
    order.sort_unstable();
    if order != [0, 1, 2] || capacity < 16 || capacity > u32::MAX as usize || domains.iter().any(|&n| n == 0 || n > 1u64 << 32) {
        return Err(TreeError::InvalidConfig("expected a dimension permutation, domains in 1..=2^32, and block capacity in 16..=u32::MAX".into()));
    }
    Ok(())
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
/// path doesn't have to thread five separate slice parameters through
/// every recursive call.
struct BuildSchedule<'a> {
    height: usize,
    level_dim_bits: &'a [[u8; NUM_DIMS]],
    level_shifts_cache: &'a [[u8; NUM_DIMS]],
    level_child_cell_bits_cache: &'a [[u8; NUM_DIMS]],
    level_skip_bytes: &'a [u8],
}

#[cfg(feature = "parallel")]
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
/// into one flat `Vec<u8>`, so the exact same recursive tree-building
/// logic can target either real, block-chopped [`BlockStorage`]
/// (streaming records straight into their final home, [`StorageSink`])
/// or a small in-memory scratch buffer for one subtree at a time
/// ([`FlatBuf`], used by [`build_rec`]'s own parallel branches before
/// their finished bytes get stitched into the real storage).
trait RecordSink {
    /// Appends one atomic node record (existence word, plus its skip
    /// placeholder if the node is a far node) and returns its position.
    fn append(&mut self, bytes: &[u8]) -> Pos;
    /// Overwrites `width` bytes at `pos.offset + rel_offset` -- used to
    /// backfill a far node's skip placeholder once its subtree's total
    /// byte length is known.
    fn patch(&mut self, pos: Pos, rel_offset: usize, width: u8, val: u64);
    /// Appends an entire *already-encoded* subtree's bytes as a unit.
    /// Every skip/span value inside `bytes` is a relative byte distance,
    /// never an absolute position, so the bytes are fully relocatable.
    /// [`FlatBuf`] can therefore just concatenate; [`StorageSink`] has to
    /// walk `bytes` record-by-record to preserve the "a block never
    /// splits a record" invariant real storage depends on.
    #[cfg(feature = "parallel")]
    fn append_subtree(&mut self, bytes: &[u8], child_level: usize, sched: &BuildSchedule);
}

/// One subtree's worth of bulk-build output, fully in memory -- the unit
/// [`build_rec`]'s parallel branches build into before being stitched
/// into the real [`StorageSink`] by the caller that spawned them.
#[cfg(feature = "parallel")]
struct FlatBuf {
    bytes: Vec<u8>,
}

#[cfg(feature = "parallel")]
impl FlatBuf {
    fn new() -> Self {
        FlatBuf { bytes: Vec::new() }
    }
}

#[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
    fn append_subtree(&mut self, bytes: &[u8], _child_level: usize, _sched: &BuildSchedule) {
        self.bytes.extend_from_slice(bytes);
    }
}

/// The real, final destination: a [`RecordSink`] adapter over `S`'s own
/// [`BlockStorage::append_record`]/[`BlockStorage::patch_appended`], so
/// [`build_rec`]'s single-threaded/no-fan-out path writes node records
/// straight into their final block-chopped home as they're produced.
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
    #[cfg(feature = "parallel")]
    fn append_subtree(&mut self, bytes: &[u8], child_level: usize, sched: &BuildSchedule) {
        replay_records(bytes, 0, child_level, sched, &mut |record| {
            self.storage.append_record(record, self.block_capacity);
        });
    }
}

/// Walks an already-encoded subtree buffer (as produced by a nested
/// [`build_rec`] call into a [`FlatBuf`]) in the same DFS-preorder its
/// own build produced it in, calling `emit` once per node record.
/// Returns the cursor position just past the subtree it walked.
#[cfg(feature = "parallel")]
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

/// Minimum point count a subtree needs before the bulk build considers
/// splitting its children across `rayon` tasks worth the overhead of a
/// task spawn plus a small heap-allocated [`FlatBuf`] per branch.
#[cfg(any(feature = "parallel", all(test, feature = "heap")))]
const BUILD_PARALLEL_MIN_POINTS: usize = 100_000;

/// A single free function, not a closure re-derived at each call site:
/// the one and only place a triple's child-slot digit at a given level
/// gets computed, shared verbatim by [`build_rec`]'s own partition step
/// and by the external bucket-spilling path's [`distribute_to_buckets`].
#[inline]
fn compute_qi(coords: [u64; NUM_DIMS], lo: [u64; NUM_DIMS], shifts: [u8; NUM_DIMS], child_cell_bits: [u8; NUM_DIMS]) -> usize {
    let mut qi = 0usize;
    for d in 0..NUM_DIMS {
        let sub = ((coords[d] - lo[d]) >> child_cell_bits[d]) as usize;
        qi |= sub << shifts[d];
    }
    qi
}

/// The bulk-build tree-construction algorithm, in exactly one place,
/// generic over where node records end up ([`RecordSink`]). Partitions
/// `points` by this node's radix digit via counting sort, appends this
/// node's own record to `sink`, then recurses into each active child.
/// Every skip/span value this writes is a relative byte distance (never
/// an absolute position), so a subtree's bytes are correct regardless of
/// *where* they end up being copied to.
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
        total_bytes += build_children(&active, points, scratch, lo, level, dim_bits, shifts, child_cell_bits, sched, sink, node_count);
    }
    if is_far_node {
        sink.patch(own_pos, WORD_RECORD_BYTES, skip_width, total_bytes);
    }
    total_bytes
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn build_children<Sink: RecordSink>(
    active: &[(usize, usize, usize)],
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    dim_bits: [u8; NUM_DIMS],
    shifts: [u8; NUM_DIMS],
    child_cell_bits: [u8; NUM_DIMS],
    sched: &BuildSchedule,
    sink: &mut Sink,
    node_count: &mut u64,
) -> u64 {
    let fan_out = active.len() >= 2 && points.len() >= BUILD_PARALLEL_MIN_POINTS && rayon::current_num_threads() > 1;
    if !fan_out {
        return build_children_sequential(active, points, scratch, lo, level, dim_bits, shifts, child_cell_bits, sched, sink, node_count);
    }
    // See this function's module-level context: sibling subtrees are
    // fully disjoint slices of `points`/`scratch` (Karras 2012's parallel
    // BVH/octree/k-d-tree construction is the closest published analogue
    // -- sort once, then treat sibling-subtree construction as
    // embarrassingly parallel work), never shared mutable state.
    let mut rest_points: &mut [(u32, u32, u32)] = points;
    let mut rest_scratch: &mut [(u32, u32, u32)] = scratch;
    let mut prev = 0usize;
    type BuildBranch<'a> = ([u64; NUM_DIMS], &'a mut [Triple], &'a mut [Triple]);
    let mut branches: Vec<BuildBranch<'_>> = Vec::with_capacity(active.len());
    for &(qi, _, qend) in active {
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
            // Reclaim `Vec` amortized-doubling slack before it can
            // compound across branches (all resident together until the
            // stitch loop below frees them one at a time) into a real
            // peak-RSS cost of its own.
            local_sink.bytes.shrink_to_fit();
            (local_sink.bytes, local_count)
        })
        .collect();
    let mut total_bytes = 0u64;
    for (bytes, count) in results {
        total_bytes += bytes.len() as u64;
        *node_count += count;
        sink.append_subtree(&bytes, level + 1, sched);
    }
    total_bytes
}

#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
fn build_children<Sink: RecordSink>(
    active: &[(usize, usize, usize)],
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    dim_bits: [u8; NUM_DIMS],
    shifts: [u8; NUM_DIMS],
    child_cell_bits: [u8; NUM_DIMS],
    sched: &BuildSchedule,
    sink: &mut Sink,
    node_count: &mut u64,
) -> u64 {
    build_children_sequential(active, points, scratch, lo, level, dim_bits, shifts, child_cell_bits, sched, sink, node_count)
}

#[allow(clippy::too_many_arguments)]
fn build_children_sequential<Sink: RecordSink>(
    active: &[(usize, usize, usize)],
    points: &mut [(u32, u32, u32)],
    scratch: &mut [(u32, u32, u32)],
    lo: [u64; NUM_DIMS],
    level: usize,
    dim_bits: [u8; NUM_DIMS],
    shifts: [u8; NUM_DIMS],
    child_cell_bits: [u8; NUM_DIMS],
    sched: &BuildSchedule,
    sink: &mut Sink,
    node_count: &mut u64,
) -> u64 {
    let mut total_bytes = 0u64;
    for &(qi, qstart, qend) in active {
        let child_lo = advance_lo(lo, dim_bits, shifts, qi, child_cell_bits);
        total_bytes += build_rec(&mut points[qstart..qend], &mut scratch[qstart..qend], child_lo, level + 1, sched, sink, node_count);
    }
    total_bytes
}

// ==================== bounded-memory (external-bucketing) bulk build ====================
//
// Everything above holds `points`/`scratch` -- O(N) each -- in RAM for
// the whole build, plus (via `build_children`'s own parallel fan-out) a
// transient extra O(N) while sibling `FlatBuf`s are collected before
// stitching -- tens of gigabytes just to *build* the tree at billion-
// triple scale, regardless of which backend the finished structure ends
// up on. What follows generalizes that "small enough: build directly;
// big enough: fan out" decision with a third option, gated by a
// caller-supplied `BulkBuildBudget`: a subtree whose point count exceeds
// the budget is never materialized as a `Vec<Triple>` at all. Instead
// it's spilled, with one streaming pass, into up to `branching` (<=64)
// on-disk bucket files -- direct digit-addressed distribution, not a
// comparison sort -- and each bucket is recursed into the same way. The
// standard external-memory distribution-sort pattern (Vitter's I/O
// model; the same "build small, budget-bounded pieces" idea HDTCat uses
// to bulk-load RDF at scale).
//
// This machinery always uses real temp files for its spill buckets,
// regardless of which `BlockStorage` backend the caller ultimately
// chose -- a `Heap`-backed tree's bulk build can still be RAM-bounded
// this way, even though the finished tree itself is fully resident.

static NEXT_SPILL_FILE_ID: AtomicU64 = AtomicU64::new(0);

/// Opens a fresh, uniquely-named temp file and immediately unlinks its
/// path, for [`SpillBucket`]'s own scratch storage.
fn open_spill_file() -> StdFile {
    let id = NEXT_SPILL_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("s_graph_bulk_spill_{}_{id}.bin", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap_or_else(|e| panic!("failed to create bulk-build spill scratch file {path:?}: {e}"));
    let _ = std::fs::remove_file(&path);
    file
}

/// Caller-supplied bound on how much bulk-build state is ever allowed to
/// be resident in RAM at once, gating `build_external`'s spill-to-disk
/// decision the same way `BUILD_PARALLEL_MIN_POINTS` gates the in-RAM
/// path's fan-out: a plain point-count comparison, never anything
/// dataset-shape-aware.
///
/// # What this does and does not bound
///
/// Bounded to roughly `O(max_resident_points)` per subtree actually held
/// in RAM, plus another `O(max_resident_points)` while a batch of
/// same-level, budget-fitting siblings is fanned out in parallel, plus a
/// small `O(branching x io_buffer_bytes)` per currently-spilling node for
/// open bucket files (<=64 x `io_buffer_bytes`, ~16 MiB by default). None
/// of this scales with total input size N.
///
/// **`Heap` caveat, stated plainly**: this bounds *build-time transients*
/// only. A `Heap`-backed tree's own steady-state size is still `O(final
/// tree size)` by definition -- this budget cannot and does not change
/// that floor. The N-independent *peak* win this type exists for is a
/// `File` story specifically; `Heap` only benefits from the smaller
/// build-time transient on top of a steady-state cost it always had.
#[derive(Clone, Copy, Debug)]
pub struct BulkBuildBudget {
    /// A subtree with more than this many points gets spilled to disk
    /// instead of being read into a `Vec<Triple>`.
    pub max_resident_points: usize,
    /// RAM staging buffer per currently-open bucket file, flushed to disk
    /// via one batched `pwrite` once full -- never one syscall per
    /// triple.
    pub io_buffer_bytes: usize,
}

impl BulkBuildBudget {
    pub const DEFAULT_IO_BUFFER_BYTES: usize = 256 * 1024;

    /// Degenerates the whole external-bucketing path to a no-op: no
    /// subtree can ever exceed `usize::MAX` points, so this provably
    /// reproduces today's unbounded bulk-build behavior exactly.
    pub const UNBOUNDED: Self = Self { max_resident_points: usize::MAX, io_buffer_bytes: Self::DEFAULT_IO_BUFFER_BYTES };

    pub fn new(max_resident_points: usize) -> Self {
        Self { max_resident_points, io_buffer_bytes: Self::DEFAULT_IO_BUFFER_BYTES }
    }

    /// ~56 bytes of peak RAM per resident point is a directly measured
    /// points+scratch+parallel-fan-out-slack estimate at 20M-triple
    /// scale -- a documented, revisitable estimate for callers who'd
    /// rather reason in "GB of RAM I'm willing to spend" than in raw
    /// point counts, not a guarantee. [`Self::new`] bypasses the
    /// estimate entirely.
    const BYTES_PER_RESIDENT_POINT: usize = 56;

    pub fn from_ram_budget_bytes(bytes: usize) -> Self {
        Self::new((bytes / Self::BYTES_PER_RESIDENT_POINT).max(1))
    }
}

/// One (subject, predicate, object) triple's on-disk row format while
/// spilled to a [`SpillBucket`]: three little-endian `u32`s, back to
/// back.
const SPILL_RECORD_BYTES: usize = 12;

/// One bucket's on-disk staging area during [`distribute_to_buckets`]:
/// raw 12-byte triple rows, appended via a RAM-buffered, batched
/// `pwrite` -- never one syscall per triple. `count` is a free byproduct
/// of routing triples here, so nothing above ever needs a separate
/// counting pass.
struct SpillBucket {
    file: StdFile,
    buf: Vec<u8>,
    file_len: u64,
    count: u64,
}

impl SpillBucket {
    fn open(io_buffer_bytes: usize) -> Self {
        SpillBucket { file: open_spill_file(), buf: Vec::with_capacity(io_buffer_bytes), file_len: 0, count: 0 }
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
            self.file.write_all_at(&self.buf, self.file_len).unwrap_or_else(|e| panic!("failed to flush bulk-build spill bucket to disk: {e}"));
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
            self.file.read_exact_at(&mut raw, 0).unwrap_or_else(|e| panic!("failed to read back bulk-build spill bucket: {e}"));
        }
        raw.as_chunks::<SPILL_RECORD_BYTES>().0.iter().map(|row| decode_spill_record(row)).collect()
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
/// can feed straight back into `build_external`/[`distribute_to_buckets`]
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
            self.file.read_exact_at(&mut self.buf, self.pos).unwrap_or_else(|e| panic!("failed to read bulk-build spill bucket chunk: {e}"));
            self.pos += take as u64;
            self.buf_pos = 0;
        }
        let t = decode_spill_record(&self.buf[self.buf_pos..self.buf_pos + SPILL_RECORD_BYTES]);
        self.buf_pos += SPILL_RECORD_BYTES;
        Some(t)
    }
}

/// One streaming pass over `source`, routing each triple to
/// `buckets[compute_qi(..)]` -- the external-storage analogue of
/// `build_rec`'s in-RAM counting-sort partition. Returns the existence
/// word (one bit per non-empty bucket) alongside the buckets themselves.
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
/// contract but sourcing points from a stream instead of an in-RAM
/// slice, for a subtree whose point count is too large to safely hold as
/// a `Vec<Triple>` under `budget`.
fn build_external<Sink: RecordSink, I: Iterator<Item = Triple>>(source: I, lo: [u64; NUM_DIMS], level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    let is_last = level == sched.height - 1;
    let dim_bits = sched.level_dim_bits[level];
    let shifts = sched.level_shifts_cache[level];
    let child_cell_bits = sched.level_child_cell_bits_cache[level];

    if is_last {
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
            // budget, not only when a single sibling does -- otherwise a
            // node with many individually-small-enough children would
            // batch all of them into one flush regardless of combined
            // size, reintroducing the "whole subtree resident at once"
            // cost this feature exists to remove.
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
/// `budget`) into `sink`, in original (ascending-qi/DFS) order. Empties
/// `batch` either way; returns the total bytes appended.
fn flush_fit_batch<Sink: RecordSink>(batch: &mut Vec<(usize, [u64; NUM_DIMS], SpillBucket)>, level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    if batch.is_empty() {
        return 0;
    }
    let taken = std::mem::take(batch);
    flush_taken_batch(taken, level, sched, budget, sink, node_count)
}

#[cfg(feature = "parallel")]
fn flush_taken_batch<Sink: RecordSink>(taken: Vec<(usize, [u64; NUM_DIMS], SpillBucket)>, level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    let total_points: u64 = taken.iter().map(|(_, _, b)| b.count).sum();
    let fan_out = taken.len() >= 2 && total_points as usize >= BUILD_PARALLEL_MIN_POINTS && rayon::current_num_threads() > 1;
    if !fan_out {
        return flush_taken_batch_sequential(taken, level, sched, budget, sink, node_count);
    }
    // Vec's IntoParallelIterator is indexed, so `.collect()` below
    // preserves `taken`'s own order (ascending qi, by construction) --
    // no re-sort needed.
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
    let mut total_bytes = 0u64;
    for (bytes, count) in results {
        total_bytes += bytes.len() as u64;
        *node_count += count;
        sink.append_subtree(&bytes, level + 1, sched);
    }
    total_bytes
}

#[cfg(not(feature = "parallel"))]
fn flush_taken_batch<Sink: RecordSink>(taken: Vec<(usize, [u64; NUM_DIMS], SpillBucket)>, level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    flush_taken_batch_sequential(taken, level, sched, budget, sink, node_count)
}

fn flush_taken_batch_sequential<Sink: RecordSink>(taken: Vec<(usize, [u64; NUM_DIMS], SpillBucket)>, level: usize, sched: &BuildSchedule, budget: &BulkBuildBudget, sink: &mut Sink, node_count: &mut u64) -> u64 {
    let mut total_bytes = 0u64;
    for (_, child_lo, bucket) in taken {
        total_bytes += build_subtree_bounded(bucket, child_lo, level + 1, sched, budget, sink, node_count);
    }
    total_bytes
}

/// The one dispatch point between "small enough for the ordinary in-RAM
/// path" and "still too big, keep spilling": reads `bucket` back if its
/// own count already fits `budget`, handing off to the existing,
/// completely unmodified [`build_rec`] -- or recurses into
/// `build_external` one level deeper otherwise.
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

#[derive(Clone, Copy)]
struct ConstrainedFrame {
    lo: u64,
    remaining: u64,
    child_pos: Pos,
}

impl ConstrainedFrame {
    const EMPTY: ConstrainedFrame = ConstrainedFrame { lo: 0, remaining: 0, child_pos: Pos::START };
}

// ==================== the generic tree ====================

pub struct RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S: BlockStorage> {
    storage: S,
    block_index: crate::block_index::BlockIndex,
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

/// The in-RAM backend: fastest, full RAM footprint, no slot indirection.
/// Requires the `heap` feature (on by default).
#[cfg(feature = "heap")]
pub type Heap = RadixTreeDfsInterleavedVarSkipUnrolledGeneric<HeapStorage>;
/// The real-file backend: near-zero steady-state RAM, `pread`/`pwrite`
/// latency. Requires the `file` feature.
#[cfg(feature = "file")]
pub type File = RadixTreeDfsInterleavedVarSkipUnrolledGeneric<FileStorage>;
/// The LZ4-compressed in-RAM backend: for RAM-constrained deployments
/// with no disk available at all. Requires the `compressed` feature.
#[cfg(feature = "compressed")]
pub type Compressed = RadixTreeDfsInterleavedVarSkipUnrolledGeneric<CompressedStorage>;

#[derive(Clone, Copy)]
struct ScanFrame {
    lo: [u64; 3],
    remaining: u64,
    child: Pos,
}

/// Monotone child navigation shared by native prefix joins. No borrowed cache
/// buffers: positions remain valid across reads on File/Compressed storage.
#[derive(Clone, Copy)]
pub(crate) struct PrefixCursor {
    pub(crate) unskipped: u64,
    child: Pos,
}

impl PrefixCursor {
    pub(crate) const EMPTY: Self = Self { unskipped: 0, child: Pos::START };

    pub(crate) fn position(&self) -> Pos { self.child }
}

impl ScanFrame {
    const EMPTY: Self = Self { lo: [0; 3], remaining: 0, child: Pos::START };
}

/// Allocation-free, filtered DFS traversal. Order is Morton order, not
/// lexicographic order. Borrows the tree, preventing mutation while scanning.
pub struct Scan<'a, S: BlockStorage> {
    tree: &'a RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>,
    frames: [ScanFrame; MAX_HEIGHT],
    masks: [u64; MAX_HEIGHT],
    depth: usize,
    single: bool,
}

impl<S: BlockStorage> Iterator for Scan<'_, S> {
    type Item = Triple;

    fn next(&mut self) -> Option<Triple> {
        if std::mem::take(&mut self.single) {
            return Some((0, 0, 0));
        }
        while self.depth != 0 {
            let level = self.depth - 1;
            let frame = &mut self.frames[level];
            if frame.remaining == 0 {
                self.depth -= 1;
                continue;
            }
            let qi = frame.remaining.trailing_zeros() as usize;
            frame.remaining &= frame.remaining - 1;
            let last = self.depth == self.tree.height as usize;
            let child = frame.child;
            if !last {
                frame.child = self.tree.child_span_end(child, level + 1);
            }
            if self.masks[level] & (1u64 << qi) == 0 {
                continue;
            }
            let lo = advance_lo(frame.lo, self.tree.level_dim_bits[level], self.tree.level_shifts_cache[level], qi, self.tree.level_child_cell_bits_cache[level]);
            if last {
                return Some((lo[0] as u32, lo[1] as u32, lo[2] as u32));
            }
            self.frames[self.depth] = ScanFrame {
                lo,
                remaining: self.tree.block_read_u64(child.block, child.offset),
                child: self.tree.advance(child, self.tree.own_record_bytes(level + 1)),
            };
            self.depth += 1;
        }
        None
    }
}

impl<S: BlockStorage> std::iter::FusedIterator for Scan<'_, S> {}

impl<S: BlockStorage> Drop for Scan<'_, S> {
    fn drop(&mut self) {
        let _guard = self.tree.storage.begin_top_level_call();
    }
}

#[cfg(feature = "file")]
impl RadixTreeDfsInterleavedVarSkipUnrolledGeneric<FileStorage> {
    /// Bytes actually occupied in the backing file -- disk-space
    /// counterpart to [`Self::mem_size_bytes`]'s RAM-only number.
    pub fn disk_bytes(&self) -> u64 {
        self.storage.file_len
    }
}

impl<S: BlockStorage> RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    pub(crate) fn prefix_domains(&self) -> [u64; 3] { self.dimension_sizes }

    pub(crate) fn prefix_schedule(&self) -> impl ExactSizeIterator<Item = ([u8; 3], [u8; 3], [u8; 3])> + '_ {
        self.level_dim_bits.iter().zip(&self.level_shifts_cache).zip(&self.level_child_cell_bits_cache)
            .map(|((&bits, &shifts), &remaining)| (bits, shifts, remaining))
    }

    pub(crate) fn prefix_guard(&self) -> impl Drop + '_ { self.storage.begin_top_level_call() }

    pub(crate) fn prefix_single(&self) -> bool { self.single_point_exists }

    #[inline]
    pub(crate) fn prefix_cursor(&self, pos: Pos, level: usize) -> PrefixCursor {
        let occupied = self.block_read_u64(pos.block, pos.offset);
        PrefixCursor { unskipped: occupied, child: self.advance(pos, self.own_record_bytes(level)) }
    }

    /// Select occupied slots in increasing order. Advance only when another
    /// child is requested, avoiding a subtree-span read on early cancellation.
    #[inline]
    pub(crate) fn prefix_child(&self, cursor: &mut PrefixCursor, level: usize, slot: u32) {
        if level + 1 == self.height as usize { return; }
        let preceding = cursor.unskipped & ((1u64 << slot) - 1);
        cursor.child = self.advance_past_children(cursor.child, preceding, level + 1);
        cursor.unskipped &= !preceding;
    }

    /// Streams matching triples without allocating an adjacency list or sorting.
    /// `None` is a wildcard in any coordinate; every stored triple occurs once.
    pub fn scan(&self, bounds: [Option<u32>; 3]) -> Scan<'_, S> {
        let mut scan = Scan { tree: self, frames: [ScanFrame::EMPTY; MAX_HEIGHT], masks: [0; MAX_HEIGHT], depth: 0, single: false };
        if bounds.iter().enumerate().any(|(d, v)| v.is_some_and(|v| v as u64 >= self.dimension_sizes[d])) {
            return scan;
        }
        if self.height == 0 {
            scan.single = self.single_point_exists;
            return scan;
        }
        let mut lo = [0; 3];
        for (level, mask) in scan.masks[..self.height as usize].iter_mut().enumerate() {
            let bits = self.level_child_cell_bits_cache[level];
            *mask = bound_mask(self.level_dim_bits[level], self.level_shifts_cache[level], lo, bits, |d| bounds[d]);
            for d in 0..3 {
                if let Some(v) = bounds[d] {
                    lo[d] += ((v as u64 - lo[d]) >> bits[d]) << bits[d];
                }
            }
        }
        scan.frames[0] = ScanFrame { lo: [0; 3], remaining: self.block_read_u64(0, 0), child: self.advance(Pos::START, self.own_record_bytes(0)) };
        scan.depth = 1;
        scan
    }

    /// Counts triples matching `bounds` -- same convention as
    /// [`Self::scan`]: `None` is a wildcard in any coordinate -- without
    /// decoding, allocating or even computing the actual coordinates of
    /// any matching triple, the way [`Self::scan`]/[`Self::query_two_bound`]
    /// must to list them.
    ///
    /// At a last-level (leaf) node, this is `u64::count_ones` on the
    /// masked word: O(1), not one iteration per matching bit the way
    /// listing pays to materialize and decode each one. Internal nodes
    /// still have to be visited one at a time to reach each child's byte
    /// position -- there is no cached per-node "matches below this
    /// point" count to skip that with, so this is a constant-factor win
    /// concentrated at the leaf level (the majority of nodes in a deep
    /// trie), not an asymptotic one. See `DESIGN.md` for measurements.
    pub fn count_matching(&self, bounds: [Option<u32>; 3]) -> u64 {
        let _guard = self.storage.begin_top_level_call();
        if bounds.iter().enumerate().any(|(d, v)| v.is_some_and(|v| v as u64 >= self.dimension_sizes[d])) {
            return 0;
        }
        if self.height == 0 {
            // Any `Some(v)` bound dim already passed the check above,
            // so it's necessarily 0 -- the domain's only valid value --
            // whether the tree's one possible point matches or not
            // therefore depends only on whether it exists at all.
            return self.single_point_exists as u64;
        }
        let mut lo = [0u64; NUM_DIMS];
        let mut masks = [0u64; MAX_HEIGHT];
        for (level, mask) in masks[..self.height as usize].iter_mut().enumerate() {
            let bits = self.level_child_cell_bits_cache[level];
            *mask = bound_mask(self.level_dim_bits[level], self.level_shifts_cache[level], lo, bits, |d| bounds[d]);
            for d in 0..NUM_DIMS {
                if let Some(v) = bounds[d] {
                    lo[d] += ((v as u64 - lo[d]) >> bits[d]) << bits[d];
                }
            }
        }
        self.count_rec(Pos::START, 0, &masks[..self.height as usize])
    }

    fn count_rec(&self, pos: Pos, level: usize, masks: &[u64]) -> u64 {
        let word = self.block_read_u64(pos.block, pos.offset);
        let mask = masks[level];
        if level == self.height as usize - 1 {
            return (word & mask).count_ones() as u64;
        }
        let mut remaining = word;
        let mut child_pos = self.advance(pos, self.own_record_bytes(level));
        let mut total = 0u64;
        while remaining != 0 {
            let qi = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let next_child_pos = self.child_span_end(child_pos, level + 1);
            if (mask >> qi) & 1 != 0 {
                total += self.count_rec(child_pos, level + 1, masks);
            }
            child_pos = next_child_pos;
        }
        total
    }

    pub fn default_split() -> [u8; NUM_DIMS] {
        [BASE / NUM_DIMS as u8; NUM_DIMS]
    }

    /// Takes `triples` by value, not `&[Triple]`: the counting sort needs
    /// its own owned, mutable, sortable copy of the data regardless, so a
    /// by-reference signature would force an extra full clone inside
    /// every single build. A caller who still needs the original data
    /// after building just clones before calling.
    ///
    /// `split` must sum to at most `BASE` (6) bits per level; a caller-
    /// supplied value that doesn't is rejected with
    /// [`TreeError::InvalidConfig`] rather than silently producing a
    /// schedule that overflows a node's 64-way branching budget. Use
    /// [`Self::build_dynamic`] to have the schedule computed
    /// automatically instead of supplying one.
    pub fn build(triples: Vec<Triple>, split: [u8; NUM_DIMS], block_capacity: usize) -> Result<Self> {
        Self::build_with_order(triples, split, [0, 1, 2], block_capacity)
    }

    pub fn build_with_order(triples: Vec<Triple>, split: [u8; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Result<Self> {
        validate_config(dimension_sizes_from(&triples), dim_order, block_capacity)?;
        let split_sum: u32 = split.iter().map(|&s| s as u32).sum();
        if split_sum > BASE as u32 {
            return Err(TreeError::InvalidConfig(format!("split {split:?} sums to {split_sum} bits, exceeding the {BASE}-bit-per-level branching budget")));
        }
        let dimension_sizes = dimension_sizes_from(&triples);
        if total_bits_needed(dimension_sizes).iter().zip(split).any(|(&bits, step)| bits != 0 && step == 0) {
            return Err(TreeError::InvalidConfig("split must advance every nontrivial dimension".into()));
        }
        let level_dim_bits = static_schedule(total_bits_needed(dimension_sizes), split);
        Ok(Self::build_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity))
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
    /// of materializing a full `Vec<Triple>` before building anything.
    ///
    /// Unlike [`Self::build_dynamic`], `dimension_sizes` has to be given
    /// explicitly rather than inferred from data (there's no data yet) --
    /// callers need to already know their domain's upper bound, exactly
    /// as any real streaming-ingestion pipeline already must (a
    /// dictionary-encoded RDF store knows its id space's size before it
    /// knows every id that will ever appear in it).
    pub fn build_dynamic_empty(dimension_sizes: [u64; NUM_DIMS], block_capacity: usize) -> Self {
        Self::build_dynamic_empty_with_order(dimension_sizes, [0, 1, 2], block_capacity)
    }

    pub fn build_dynamic_empty_with_order(dimension_sizes: [u64; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize) -> Self {
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_from_schedule(Vec::new(), dimension_sizes, level_dim_bits, dim_order, block_capacity)
    }

    /// The bounded-memory sibling of [`Self::build_dynamic`], for
    /// datasets too large to materialize as a single `Vec<Triple>` at all
    /// (see [`BulkBuildBudget`]'s own doc comment). Takes `triples` as a
    /// plain `Iterator` rather than a `Vec`, and `dimension_sizes`
    /// explicitly, for the same reason [`Self::build_dynamic_empty`]
    /// already requires it: a single-pass streaming source can't be
    /// scanned twice to first learn the domain and then build.
    ///
    /// `BulkBuildBudget::UNBOUNDED` degenerates this to exactly
    /// [`Self::build_dynamic`]'s own behavior.
    pub fn build_dynamic_bounded<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        Self::build_dynamic_bounded_with_order(triples, dimension_sizes, [0, 1, 2], block_capacity, budget)
    }

    pub fn build_dynamic_bounded_with_order<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], dim_order: [usize; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        let level_dim_bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        Self::build_bounded_from_schedule(triples, dimension_sizes, level_dim_bits, dim_order, block_capacity, budget)
    }

    fn build_bounded_from_schedule<I: Iterator<Item = Triple>>(triples: I, dimension_sizes: [u64; NUM_DIMS], level_dim_bits: Vec<[u8; NUM_DIMS]>, dim_order: [usize; NUM_DIMS], block_capacity: usize, budget: BulkBuildBudget) -> Self {
        validate_config(dimension_sizes, dim_order, block_capacity).expect("invalid tree configuration");
        debug_assert!(block_capacity >= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES);
        let mut triples = triples.inspect(|&(s, p, o)| {
            assert!([s, p, o].iter().zip(dimension_sizes).all(|(&v, size)| (v as u64) < size), "bounded-build coordinate exceeds domain");
        });
        if level_dim_bits.is_empty() {
            return RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
                storage: S::empty(),
                block_index: crate::block_index::BlockIndex::default(),
                block_capacity,
                node_count: 0,
                level_skip_bytes: Vec::new(),
                level_dim_bits,
                level_shifts_cache: Vec::new(),
                level_child_cell_bits_cache: Vec::new(),
                dimension_sizes,
                dim_order,
                single_point_exists: triples.fold(false, |_, _| true),
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
        // everything, `prefix` simply *is* what `points` would have been
        // for the ordinary fast path below, at zero extra cost. Only
        // when there's strictly more left does this fall through to
        // external bucketing, chaining the already-collected prefix back
        // in front of whatever's left in `triples` so no triple is
        // skipped or duplicated.
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
            block_index: crate::block_index::BlockIndex::build(&storage),
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
        validate_config(dimension_sizes, dim_order, block_capacity).expect("invalid tree configuration");
        debug_assert!(block_capacity >= WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES);
        if level_dim_bits.is_empty() {
            return RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
                storage: S::empty(),
                block_index: crate::block_index::BlockIndex::default(),
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
        // u32)`, and `triples` is owned here, so the counting sort gets
        // to reuse the caller's own allocation as its mutable working
        // array instead of paying for a second, identical one.
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
            block_index: crate::block_index::BlockIndex::build(&storage),
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

    /// Whether `(s, p, o)` falls within the domain established at build
    /// time. Exposed so a caller inserting a large batch can pre-filter
    /// or pre-validate without paying for constructing an error value
    /// per candidate -- [`Self::insert`]/[`Self::insert_batch`] already
    /// call this internally and return
    /// [`TreeError::OutOfDomain`]
    /// on a `false` result, so this is a convenience for the caller, not
    /// a substitute for it being checked.
    #[inline]
    pub fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        (s as u64) < self.dimension_sizes[0] && (p as u64) < self.dimension_sizes[1] && (o as u64) < self.dimension_sizes[2]
    }

    /// Builds the specific [`TreeError::OutOfDomain`] for a coordinate
    /// already known (via [`Self::in_bounds`]) to be out of range,
    /// identifying which of the three dimensions actually violated its
    /// bound.
    fn out_of_domain_error(&self, s: u32, p: u32, o: u32) -> TreeError {
        let coords = [s, p, o];
        for (d, &coordinate) in coords.iter().enumerate() {
            if (coordinate as u64) >= self.dimension_sizes[d] {
                return TreeError::OutOfDomain(OutOfDomain { dim: d, value: coords[d], domain_size: self.dimension_sizes[d] });
            }
        }
        unreachable!("out_of_domain_error called on a coordinate that was actually in bounds")
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
    fn advance(&self, pos: Pos, n: usize) -> Pos {
        let len = self.block_len(pos.block);
        assert!(pos.offset <= len, "invalid block offset");
        let available = len - pos.offset;
        if n < available || (n == available && pos.block + 1 == self.storage.block_count()) {
            return Pos { block: pos.block, offset: pos.offset + n };
        }
        if n == available && self.block_len(pos.block + 1) != 0 { return Pos { block: pos.block + 1, offset: 0 }; }
        self.block_index.locate(self.block_index.prefix(pos.block) + pos.offset as u64 + n as u64)
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

    /// Like calling [`Self::contains`] once per triple in `triples`, in
    /// `triples`' own order, but shares descent work across queries that
    /// land on the same physical trie node. Internally sorts a *copy* of
    /// the queries into this tree's own DFS-preorder (see
    /// [`Self::morton_key`] -- the same order [`Self::insert_batch`]/
    /// [`Self::remove_batch`] already sort into), then walks once,
    /// keeping the previous query's root-to-leaf path cached: a query
    /// whose key shares its first `k` digits with the previous query's
    /// key reuses that query's already-read node at every one of those
    /// `k` levels -- same word, same physical byte position, since the
    /// trie's shape is fully determined by the digits queried so far --
    /// and only re-reads from the trie starting at the first level where
    /// the two keys actually diverge.
    ///
    /// When a batch has no shared prefixes at all (fully scattered
    /// queries against a wide, shallow tree) this degenerates to exactly
    /// `triples.len()` independent [`Self::contains`] descents: the only
    /// added cost per query is an O(min(height, shared)) byte comparison
    /// against the previous key, so this is never worse than the naive
    /// per-triple loop. When queries cluster -- many triples sharing a
    /// subject, or checking membership of one graph's triples against
    /// another, mostly-overlapping one -- the shared levels, *including*
    /// their [`Self::advance_past_children`] sibling-hopping (the real
    /// dominant cost on a high-fanout, high-occupancy node), are paid
    /// exactly once for the whole cluster rather than once per query.
    ///
    /// Returns one `bool` per input triple, in `triples`' own order, not
    /// sorted order -- unlike `insert_batch`/`remove_batch`, this never
    /// reorders the caller's slice, since there's no sort order worth
    /// exposing from a read-only query. An out-of-domain triple reports
    /// `false`, the same rule `contains` already documents.
    pub fn contains_batch(&self, triples: &[Triple]) -> Vec<bool> {
        let mut out = Vec::new();
        self.contains_batch_into(triples, &mut out);
        out
    }

    /// Same as [`Self::contains_batch`], writing into (and resizing)
    /// `out` instead of allocating a fresh `Vec`, for repeated calls
    /// that want to reuse the output buffer's capacity.
    pub fn contains_batch_into(&self, triples: &[Triple], out: &mut Vec<bool>) {
        let _guard = self.storage.begin_top_level_call();
        out.clear();
        out.resize(triples.len(), false);
        if triples.is_empty() {
            return;
        }
        if self.height == 0 {
            for (i, &(s, p, o)) in triples.iter().enumerate() {
                out[i] = self.in_bounds(s, p, o) && self.single_point_exists;
            }
            return;
        }
        let height = self.height as usize;
        // Out-of-domain triples are left as `false` in `out` and simply
        // never enter the sorted descent batch -- same rule `contains`
        // documents, applied up front instead of per-level.
        let mut order: Vec<(u32, [u8; MAX_HEIGHT])> = Vec::with_capacity(triples.len());
        for (i, &(s, p, o)) in triples.iter().enumerate() {
            if self.in_bounds(s, p, o) {
                order.push((i as u32, self.morton_key(s, p, o)));
            }
        }
        order.sort_unstable_by(|a, b| a.1[..height].cmp(&b.1[..height]));

        let mut stack_pos = [Pos::START; MAX_HEIGHT];
        let mut prev_key = [0u8; MAX_HEIGHT];
        let mut prev_depth = 0usize;

        for &(orig_idx, key) in order.iter() {
            let shared = key[..height]
                .iter()
                .zip(prev_key[..height].iter())
                .take_while(|(a, b)| a == b)
                .count()
                .min(prev_depth);
            let mut pos = stack_pos[shared];
            let mut found = true;
            let mut level = shared;
            while level < height {
                let word = self.block_read_u64(pos.block, pos.offset);
                let qi = key[level] as u64;
                if (word >> qi) & 1 == 0 {
                    found = false;
                    break;
                }
                if level == height - 1 {
                    break;
                }
                let earlier = word & ((1u64 << qi) - 1);
                let first_child_pos = self.advance(pos, self.own_record_bytes(level));
                pos = self.advance_past_children(first_child_pos, earlier, level + 1);
                stack_pos[level + 1] = pos;
                level += 1;
            }
            out[orig_idx as usize] = found;
            prev_key = key;
            // Not `level + 1`: `level` is the deepest level this query
            // actually tested a bit at (whether it failed there or
            // succeeded at the last level), so it's the deepest level a
            // following query may resume *at* -- re-testing that bit
            // fresh -- without ever resuming *past* a level nobody has
            // tested the matching digit for yet.
            prev_depth = level;
        }
    }

    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        self.query_two_bound([0, 2], [s, o], 1)
    }

    pub fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        self.query_two_bound_into(bound_dims, bound_vals, free_dim, &mut out);
        out
    }

    /// Replaces `out`, retaining its allocation for repeated adjacency queries.
    pub fn query_two_bound_into(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize, out: &mut Vec<u32>) {
        let _guard = self.storage.begin_top_level_call();
        out.clear();
        for i in 0..2 {
            if (bound_vals[i] as u64) >= self.dimension_sizes[bound_dims[i]] {
                return;
            }
        }
        if self.height == 0 {
            if self.single_point_exists {
                out.push(0);
            }
            return;
        }
        let level_masks = self.precompute_level_masks(bound_dims, bound_vals, |d| d == bound_dims[0] || d == bound_dims[1]);
        let unconstrained_from = self.first_fully_open_level(&level_masks);
        self.query_rec(Pos::START, 0, 0, &level_masks[..self.height as usize], free_dim, unconstrained_from, out);
        out.sort_unstable();
        out.dedup();
    }

    pub fn query_one_bound(&self, bound_dim: usize, bound_val: u32, target_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        self.query_one_bound_into(bound_dim, bound_val, target_dim, &mut out);
        out
    }

    /// Replaces `out`, retaining its allocation for repeated adjacency queries.
    pub fn query_one_bound_into(&self, bound_dim: usize, bound_val: u32, target_dim: usize, out: &mut Vec<u32>) {
        let _guard = self.storage.begin_top_level_call();
        out.clear();
        if (bound_val as u64) >= self.dimension_sizes[bound_dim] {
            return;
        }
        if self.height == 0 {
            if self.single_point_exists {
                out.push(0);
            }
            return;
        }
        let level_masks = self.precompute_level_masks([bound_dim, bound_dim], [bound_val, bound_val], |d| d == bound_dim);
        let unconstrained_from = self.first_fully_open_level(&level_masks);
        self.query_rec(Pos::START, 0, 0, &level_masks[..self.height as usize], target_dim, unconstrained_from, out);
        out.sort_unstable();
        out.dedup();
    }

    fn first_fully_open_level(&self, level_masks: &[u64; MAX_HEIGHT]) -> usize {
        for (level, &mask) in level_masks.iter().enumerate().take(self.height as usize) {
            let dim_bits = self.level_dim_bits[level];
            let bit_sum = (dim_bits[0] + dim_bits[1] + dim_bits[2]) as u32;
            let branching: u64 = 1u64 << bit_sum;
            let full_mask = if branching >= 64 { u64::MAX } else { (1u64 << branching) - 1 };
            if mask == full_mask {
                return level;
            }
        }
        self.height as usize
    }

    fn precompute_level_masks(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], is_bound: impl Fn(usize) -> bool) -> [u64; MAX_HEIGHT] {
        let mut lo = [0u64; NUM_DIMS];
        let mut masks = [0u64; MAX_HEIGHT];
        debug_assert!(self.height as usize <= MAX_HEIGHT);
        for (level, mask) in masks.iter_mut().enumerate().take(self.height as usize) {
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
            *mask = bound_mask(dim_bits, shifts, lo, child_cell_bits, bound_val_for);
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

    #[allow(clippy::too_many_arguments)]
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
                // A visited node's word can legitimately be all zero here:
                // `Self::remove`'s cascade never physically removes the
                // root's own record even when clearing its last bit
                // leaves the whole tree empty (there's nothing above the
                // root to clear a pointing bit in). Every branch below
                // already handles `word == 0` correctly on its own
                // (`relevant`/`remaining` both become `0`, so no results
                // are pushed and the right amount of storage is skipped).

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

    /// Bulk-reads one entire "listing" subtree in one shot and decodes it
    /// entirely from memory, rather than walking it node by node.
    fn bulk_list_subtree(&self, pos: Pos, lo: u64, level: usize, free_dim: usize, out: &mut Vec<u32>) {
        let span_end = self.child_span_end(pos, level);
        let byte_len = self.byte_distance(pos, span_end);
        let buf = self.storage.bulk_read(pos.block, pos.offset, byte_len);
        self.decode_listing(&buf, 0, lo, level, free_dim, out);
    }

    fn byte_distance(&self, start: Pos, end: Pos) -> usize {
        if start.block == end.block { return end.offset - start.offset; }
        (self.block_index.prefix(end.block) + end.offset as u64 - self.block_index.prefix(start.block) - start.offset as u64) as usize
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

    /// Inserts one point. Returns `Ok(true)` if it was newly inserted,
    /// `Ok(false)` if it already existed, or
    /// [`TreeError::OutOfDomain`]
    /// if the coordinate falls outside the domain established at build
    /// time. Domain checks precede traversal and mutation. Navigation costs
    /// O(height log B) in the worst case; block splits also cost O(B), where B
    /// is the logical block count.
    ///
    /// No `begin_top_level_call` guard here (unlike the read methods):
    /// this needs `&mut self` throughout, which a guard holding a live
    /// borrow across the call would conflict with -- `set_insert_mode`/
    /// `end_insert` are momentary `&self` calls instead.
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> Result<bool> {
        if !self.in_bounds(s, p, o) {
            return Err(self.out_of_domain_error(s, p, o));
        }
        self.storage.set_insert_mode(true);
        let result = self.insert_no_reset(s, p, o);
        self.storage.end_insert();
        Ok(result)
    }

    /// Sorts `triples` into this tree's own DFS-preorder (Morton/
    /// interleaved-coordinate) order -- see `Self::morton_key` -- then
    /// inserts them one at a time via `Self::insert_no_reset`, resetting
    /// the file backend's block-stack cache only once, after the whole
    /// batch, instead of after every single triple.
    ///
    /// Every triple is validated against the domain up front, before any
    /// mutation happens: if any one of them is out of bounds, this
    /// returns `Err` and the tree is left completely unchanged (not
    /// partially inserted up to the bad triple) -- both because sorting
    /// itself would otherwise run `Self::morton_key` on an out-of-domain
    /// coordinate, and because a batch call should not have a partial-
    /// failure state a caller has to reason about.
    ///
    /// This changes *only* physical insertion order and how often the
    /// cache is reset -- not which bits end up set anywhere. The
    /// resulting tree is byte-for-byte equivalent to inserting the same
    /// points one at a time in any other order, or to a bulk
    /// `build_dynamic` over the same set.
    pub fn insert_batch(&mut self, triples: &mut [Triple]) -> Result<u64> {
        if triples.is_empty() {
            return Ok(0);
        }
        for &(s, p, o) in triples.iter() {
            if !self.in_bounds(s, p, o) {
                return Err(self.out_of_domain_error(s, p, o));
            }
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
        Ok(inserted)
    }

    /// Removes `(s, p, o)` if present. Returns `Ok(true)` if it was
    /// removed, `Ok(false)` if it was absent or out of domain -- an
    /// out-of-domain coordinate can't match anything, the same rule
    /// `contains`/`query_two_bound` already document, so this never
    /// errors on one (unlike `insert`: accepting an invalid coordinate
    /// here wouldn't misrepresent the domain going forward either way,
    /// since `remove` never changes it).
    ///
    /// O(height) plus the physical cost of shifting whatever content
    /// follows the removed record within its own block -- full
    /// cascading reclamation: a leaf (or any ancestor) left entirely
    /// empty by the removal has its own record physically removed too,
    /// all the way up to (but never including) the root, with every
    /// surviving ancestor's own cached skip field decremented to match.
    /// Does not merge or drop now-underfull blocks; that block-to-block
    /// rebalancing is separate, not-yet-implemented future work.
    pub fn remove(&mut self, s: u32, p: u32, o: u32) -> Result<bool> {
        if !self.in_bounds(s, p, o) {
            return Ok(false);
        }
        if self.height == 0 {
            return Ok(std::mem::take(&mut self.single_point_exists));
        }
        self.storage.set_insert_mode(true);
        let found = if let Some(trail) = self.descend_for_remove(s, p, o) {
            self.remove_cascade(&trail[..self.height as usize]);
            true
        } else {
            false
        };
        self.storage.end_insert();
        Ok(found)
    }

    /// Descends from the root to `(s, p, o)`'s slot, recording the full
    /// root-to-leaf trail -- shared by `Self::remove` (which cascades
    /// immediately) and `Self::remove_batch` (which defers cascading so
    /// it can combine it across the whole sorted batch). `None` if the
    /// point isn't present. Caller must already know `self.height > 0`
    /// and that `(s, p, o)` is in bounds.
    fn descend_for_remove(&self, s: u32, p: u32, o: u32) -> Option<RemoveTrail> {
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut pos = Pos::START;
        // Fifth element (vs. insert's own trail): this level's own `qi`
        // -- the cascade needs to know which bit to clear at every
        // level, not just the deepest one.
        let mut trail: RemoveTrail = [(Pos::START, 0, 0, 0, 0); MAX_HEIGHT];

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
            trail[level] = (pos, level, word, skip_old, qi);

            if (word >> qi) & 1 == 0 {
                return None;
            }
            if is_last {
                return Some(trail);
            }
            let earlier = word & ((1u64 << qi) - 1);
            let first_child_pos = self.advance(pos, self.own_record_bytes(level));
            pos = self.advance_past_children(first_child_pos, earlier, level + 1);
            lo = next_lo;
        }
        unreachable!("height > 0 checked by caller")
    }

    /// Clears the target bit at `trail`'s deepest (leaf) level, then
    /// walks the rest of `trail` in reverse (root-ward), mirroring
    /// `Self::insert_new_branch`'s own ancestor-patch loop but
    /// subtracting: while every node visited so far was removed in its
    /// entirety (`still_cascading`), clear this level's own bit for the
    /// now-dead child below it too, and if this node's word is now also
    /// zero, remove its own record as well -- except the root (level
    /// `0`), which is never physically removed even if its word goes to
    /// zero, since there is nothing above it to clear a pointing bit in.
    /// The first surviving ancestor stops the bit-clearing; every level
    /// from there to the root still gets its own cached skip field
    /// decremented by the total bytes removed below it, exactly as every
    /// trail entry in `insert_new_branch`'s loop gets patched regardless
    /// of level.
    fn remove_cascade(&mut self, trail: &[RemoveTrailEntry]) {
        let mut removed_bytes: u64 = 0;
        let mut still_cascading = true;
        for &(pos, level, word, skip_old, qi) in trail.iter().rev() {
            let new_word = if still_cascading { word & !(1u64 << qi) } else { word };
            if new_word != word {
                self.block_write_u64_at(pos.block, pos.offset, new_word);
            }
            if still_cascading && new_word == 0 && level != 0 {
                let own_len = self.own_record_bytes(level);
                let old_blocks = self.storage.block_count();
                self.storage.remove_bytes(pos, own_len);
                if self.storage.block_count() == old_blocks {
                    self.block_index.shrink(pos.block, own_len);
                } else {
                    // A backend that drops a block it just emptied (see
                    // `HeapStorage::remove_bytes`) shifts every later
                    // logical block index down by one -- cheaper to
                    // rebuild in O(B) than to track that shift through
                    // every remaining trail entry.
                    self.block_index.rebuild(&self.storage);
                }
                self.node_count -= 1;
                removed_bytes += own_len as u64;
            } else {
                still_cascading = false;
                if removed_bytes > 0 && level + 2 < self.height as usize {
                    let width = self.level_skip_bytes[level];
                    let skip_offset = pos.offset + WORD_RECORD_BYTES;
                    self.block_write_var_at(pos.block, skip_offset, width, skip_old - removed_bytes);
                }
            }
        }
    }

    /// Sorts `triples` into this tree's own DFS-preorder (Morton) order
    /// -- see `Self::morton_key` -- then removes them one at a time via
    /// `Self::remove`, mirroring `Self::insert_batch`'s own signature
    /// and Morton-locality rationale: nearby triples share cache-warm
    /// descent paths.
    ///
    /// A true per-level batched cascade (group every node touched by
    /// the whole sorted batch, apply once, and compact every block
    /// holding several now-dead records in one combined pass instead
    /// of one shift per record) was built and measured here, not just
    /// theorized: it's a real 2-3x win when `block_capacity` is small
    /// enough that many blocks each hold several dead records (so the
    /// sequential loop below pays for re-shifting the same block's
    /// tail repeatedly) -- but at this crate's actual
    /// [`DEFAULT_BLOCK_CAPACITY`], and for any batch without that kind
    /// of physical clustering, the batched version's own bookkeeping
    /// (one descent trail per triple, grouping, a block-indexed map)
    /// cost more than it recovered, making it *slower* than this plain
    /// loop in the common case. Reverted in favor of this simpler,
    /// measured-faster version rather than keeping a second code path
    /// whose benefit doesn't show up at the capacities this crate
    /// actually defaults to or measures against in `s-bench`. Unlike
    /// `insert_batch`, there's no up-front domain validation here --
    /// `remove` never errors on an out-of-domain coordinate, it simply
    /// can't match anything, so an out-of-domain triple is silently not
    /// found.
    pub fn remove_batch(&mut self, triples: &mut [Triple]) -> Result<u64> {
        if triples.is_empty() {
            return Ok(0);
        }
        triples.sort_unstable_by_key(|&(s, p, o)| self.morton_key(s, p, o));
        let mut removed = 0u64;
        for &(s, p, o) in triples.iter() {
            if self.remove(s, p, o)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Reclaims reserved-but-unused block capacity accumulated by
    /// [`Self::insert`]/[`Self::insert_batch`] -- call this once, when
    /// the caller knows they're done with a burst of insertion activity,
    /// not after every individual call.
    pub fn shrink_to_fit(&mut self) {
        if self.height > 0 {
            let records = {
                let _guard = self.storage.begin_top_level_call();
                self.dfs_records()
            };
            self.storage.compact_from_records(&records, self.block_capacity);
            self.block_index.rebuild(&self.storage);
        }
        self.storage.shrink_to_fit();
        self.block_index.shrink_to_fit();
    }

    /// Every node's own `(position, byte length)`, walked in DFS-preorder
    /// -- derived on the fly from the live tree, never stored anywhere.
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
    /// order -- exactly the sequence the bulk build's own per-level
    /// counting sort partitions by, so sorting a batch of points
    /// lexicographically by this key reproduces the identical relative
    /// order a bulk build over the same points would visit them in.
    fn morton_key(&self, s: u32, p: u32, o: u32) -> [u8; MAX_HEIGHT] {
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut key = [0u8; MAX_HEIGHT];
        for (level, digit) in key.iter_mut().enumerate().take(self.height as usize) {
            let shifts = self.level_shifts_cache[level];
            let child_cell_bits = self.level_child_cell_bits_cache[level];
            let mut qi = 0usize;
            let mut next_lo = lo;
            for d in 0..NUM_DIMS {
                let sub = (coords[d] - lo[d]) >> child_cell_bits[d];
                qi |= (sub as usize) << shifts[d];
                next_lo[d] += sub << child_cell_bits[d];
            }
            *digit = qi as u8;
            lo = next_lo;
        }
        key
    }

    /// The shared body of [`Self::insert`] and [`Self::insert_batch`] --
    /// both callers have already validated `(s, p, o)` is in bounds, so
    /// this only asserts it (a cheap, redundant internal sanity check,
    /// not a substitute for the real check both callers already made).
    fn insert_no_reset(&mut self, s: u32, p: u32, o: u32) -> bool {
        if self.height == 0 {
            return !std::mem::replace(&mut self.single_point_exists, true);
        }
        debug_assert!(self.height > 0, "insert() on an empty/degenerate tree is out of scope for this prototype");
        debug_assert!(self.in_bounds(s, p, o), "insert_no_reset({s},{p},{o}) called without an out-of-bounds check by its caller -- internal bug, not a caller-facing error");
        let coords = [s as u64, p as u64, o as u64];
        let mut lo = [0u64; NUM_DIMS];
        let mut pos = Pos::START;
        // Fourth element: this level's *own* skip-field value, read
        // alongside its word in one combined fetch. Almost every trail
        // entry ends up needing this in `insert_new_branch`'s ancestor
        // loop anyway.
        let mut trail: [(Pos, usize, u64, u64); MAX_HEIGHT] = [(Pos::START, 0, 0, 0); MAX_HEIGHT];

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
            trail[level] = (pos, level, word, skip_old);

            if (word >> qi) & 1 == 0 {
                self.insert_new_branch(pos, level, qi, word, coords, lo, &trail[..=level]);
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

    #[allow(clippy::too_many_arguments)]
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

        let old_blocks = self.storage.block_count();
        self.storage.insert_bytes(insert_pos, &chain, self.block_capacity);
        if self.storage.block_count() == old_blocks {
            self.block_index.grow(insert_pos.block, chain.len());
        } else {
            // Splits already shift the backend's block vector. Rebuild in O(B);
            // the common no-split insertion updates only O(log B) counters.
            self.block_index.rebuild(&self.storage);
        }
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

    /// Actual allocated-capacity bytes this tree occupies -- see
    /// `crate::vec_bytes`'s own doc comment for what that means in
    /// practice.
    pub fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes() + self.block_index.mem_size_bytes() + crate::vec_bytes(&self.level_skip_bytes) + crate::vec_bytes(&self.level_dim_bits) + crate::vec_bytes(&self.level_shifts_cache) + crate::vec_bytes(&self.level_child_cell_bits_cache)
    }

    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.storage.size_breakdown();
        v.push(("block navigation index", self.block_index.mem_size_bytes()));
        v.push(("level_skip_bytes (O(height))", crate::vec_bytes(&self.level_skip_bytes)));
        v.push(("level_dim_bits (O(height))", crate::vec_bytes(&self.level_dim_bits)));
        v.push(("level_shifts_cache (O(height))", crate::vec_bytes(&self.level_shifts_cache)));
        v.push(("level_child_cell_bits_cache (O(height))", crate::vec_bytes(&self.level_child_cell_bits_cache)));
        v
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        crate::serialize::write_header(
            &mut out,
            &crate::serialize::Header {
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
        crate::serialize::write_block_table(&mut out, &capacities, &used_lens);
        self.storage.write_blocks_into(&mut out);
        out
    }

    /// Reconstructs a tree from bytes produced by [`Self::serialize`] --
    /// on *any* backend, not just the one that produced them (see
    /// [`crate::serialize`]'s own module docs). Returns
    /// [`TreeError::CorruptData`]/
    /// [`TreeError::UnsupportedVersion`]
    /// rather than panicking: `buf` is untrusted external input by
    /// construction (a file on disk, bytes off the wire), not a value
    /// this process produced itself.
    pub fn deserialize(buf: &[u8]) -> Result<Self> {
        Self::deserialize_with_limits(buf, crate::serialize::DeserializeLimits::default())
    }

    pub fn deserialize_with_limits(buf: &[u8], limits: crate::serialize::DeserializeLimits) -> Result<Self> {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf), buf.len() as u64, limits)
    }

    pub fn deserialize_from_path(path: &std::path::Path) -> Result<Self> {
        Self::deserialize_from_path_with_limits(path, crate::serialize::DeserializeLimits::default())
    }

    pub fn deserialize_from_path_with_limits(path: &std::path::Path, limits: crate::serialize::DeserializeLimits) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file), len, limits)
    }

    fn deserialize_from_reader<R: std::io::Read>(r: &mut R, input_len: u64, limits: crate::serialize::DeserializeLimits) -> Result<Self> {
        let corrupt = |s: &str| TreeError::CorruptData(s.into());
        let h = crate::serialize::read_header_from(r)?;
        validate_config(h.dimension_sizes, h.dim_order, h.block_capacity).map_err(|_| corrupt("invalid tree configuration"))?;
        if h.block_capacity > limits.max_block_bytes { return Err(corrupt("block capacity exceeds limit")); }
        let mut remaining = total_bits_needed(h.dimension_sizes);
        for (i, &bits) in h.level_dim_bits.iter().enumerate() {
            let sum: u32 = bits.iter().map(|&b| b as u32).sum();
            if sum == 0 || sum > BASE as u32 || (0..3).any(|d| bits[d] > remaining[d] || (bits[d] == 0 && remaining[d] != 0)) {
                return Err(corrupt("invalid bit schedule"));
            }
            for d in 0..3 { remaining[d] -= bits[d]; }
            if h.level_shifts_cache[i] != level_shifts(bits, h.dim_order) || h.level_child_cell_bits_cache[i] != remaining {
                return Err(corrupt("inconsistent cached schedule"));
            }
        }
        if remaining != [0; 3] || (h.height != 0 && h.single_point_exists) {
            return Err(corrupt("incomplete schedule or invalid singleton flag"));
        }
        if h.height != 0 && h.level_skip_bytes != compute_level_skip_bytes(&h.level_dim_bits, h.height as usize) {
            return Err(corrupt("invalid skip widths"));
        }
        let mut table = crate::serialize::read_block_table_limited(r, limits)?;
        let used: u64 = table.iter().map(|&(_, n)| n as u64).sum();
        // Exact version-1 framing, checked before any payload allocation.
        let expected = 61u64 + 10 * h.height as u64 + 8 * table.len() as u64 + used;
        if expected != input_len { return Err(corrupt("snapshot length does not match its block table")); }
        // Earlier file-backend splits could leave empty logical blocks. They
        // contain no records and can be removed without changing byte spans.
        table.retain(|&(_, len)| len != 0);
        if (h.height == 0 && (!table.is_empty() || h.node_count != 0)) || (h.height != 0 && table.is_empty()) {
            return Err(corrupt("missing root or invalid singleton storage"));
        }
        let storage = S::read_blocks_from(r, &table)?;
        let tree = RadixTreeDfsInterleavedVarSkipUnrolledGeneric {
            block_index: crate::block_index::BlockIndex::build(&storage),
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
        };
        if tree.height != 0 {
            let mut pos = Pos::START;
            let mut nodes = 0;
            let bytes = tree.validate_node(&mut pos, 0, [0; 3], &mut nodes)?;
            if bytes != used || nodes != tree.node_count { return Err(corrupt("node count or topology mismatch")); }
        }
        Ok(tree)
    }

    // Validate physical record boundaries and topology without following any
    // untrusted skip pointer. Recursion is bounded by the checked height.
    fn validate_node(&self, pos: &mut Pos, level: usize, lo: [u64; 3], nodes: &mut u64) -> Result<u64> {
        let corrupt = |s: &str| TreeError::CorruptData(s.into());
        if pos.block >= self.storage.block_count() { return Err(corrupt("truncated node topology")); }
        let len = self.own_record_bytes(level);
        if pos.offset > self.block_len(pos.block) || len > self.block_len(pos.block) - pos.offset {
            return Err(corrupt("node record crosses a block boundary"));
        }
        let word = self.block_read_u64(pos.block, pos.offset);
        let bits = self.level_dim_bits[level];
        let branching = 1u32 << bits.iter().map(|&b| b as u32).sum::<u32>();
        if (word == 0 && level != 0) || (branching < 64 && word >> branching != 0) {
            return Err(corrupt("invalid existence bitmap"));
        }
        let skip = if level + 2 < self.height as usize {
            Some(self.block_read_var(pos.block, pos.offset + 8, self.level_skip_bytes[level]))
        } else { None };
        *nodes += 1;
        pos.offset += len;
        if pos.offset == self.block_len(pos.block) { pos.block += 1; pos.offset = 0; }
        let mut bytes = len as u64;
        let mut children = word;
        while children != 0 {
            let qi = children.trailing_zeros() as usize;
            children &= children - 1;
            let next_lo = advance_lo(lo, bits, self.level_shifts_cache[level], qi, self.level_child_cell_bits_cache[level]);
            if next_lo.iter().zip(self.dimension_sizes).any(|(&v, size)| v >= size) { return Err(corrupt("stored coordinate exceeds domain")); }
            if level + 1 < self.height as usize { bytes += self.validate_node(pos, level + 1, next_lo, nodes)?; }
        }
        if skip.is_some_and(|s| s != bytes) { return Err(corrupt("subtree skip does not match topology")); }
        Ok(bytes)
    }

}

#[cfg(all(test, feature = "heap"))]
mod tests {
    use super::*;
    use crate::test_support::{uniform, zipfian_predicates, NaiveIndex, SplitMix64};

    const TEST_CAPACITIES: [usize; 4] = [32, 64, 256, DEFAULT_BLOCK_CAPACITY];

    /// `Result::unwrap_err` requires the `Ok` type to implement `Debug`,
    /// which the tree struct itself deliberately doesn't (no test in
    /// this suite needs it) -- this sidesteps that bound.
    fn expect_err<T>(r: Result<T>) -> TreeError {
        match r {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        }
    }

    fn cross_check_tree<S: BlockStorage>(tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>, naive: &NaiveIndex, num_s: u32, num_p: u32, num_o: u32, label: &str) {
        let mut all_triples: Vec<Triple> = Vec::new();
        for s in 0..num_s {
            for p in 0..num_p {
                for o in 0..num_o {
                    assert_eq!(tree.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o}) {label}");
                    all_triples.push((s, p, o));
                }
                let mut got = tree.query_two_bound([0, 1], [s, p], 2);
                let mut expect = naive.row(p, s);
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "query_two_bound(S={s},P={p},free=O) {label}");
                assert_eq!(tree.count_matching([Some(s), Some(p), None]), expect.len() as u64, "count_matching(S={s},P={p},O=?) {label}");
            }
        }
        let expect_batch: Vec<bool> = all_triples.iter().map(|&(s, p, o)| naive.contains(s, p, o)).collect();
        assert_eq!(tree.contains_batch(&all_triples), expect_batch, "contains_batch over the full domain {label}");
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

    /// Cross-checks `Heap` unconditionally; `File`/`Compressed` too when
    /// their features are enabled -- `cargo test --all-features` for full
    /// coverage, `cargo test` (default: `heap` only) still exercises the
    /// shared algorithm against the default backend.
    fn cross_check_all_backends(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, label: &str) {
        let naive = NaiveIndex::build(triples);
        for cap in TEST_CAPACITIES {
            let heap = Heap::build_dynamic(triples.to_vec(), cap);
            cross_check_tree(&heap, &naive, num_s, num_p, num_o, &format!("heap cap={cap} {label}"));
            #[cfg(feature = "file")]
            {
                let file = File::build_dynamic(triples.to_vec(), cap);
                cross_check_tree(&file, &naive, num_s, num_p, num_o, &format!("file cap={cap} {label}"));
            }
            #[cfg(feature = "compressed")]
            {
                let compressed = Compressed::build_dynamic(triples.to_vec(), cap);
                cross_check_tree(&compressed, &naive, num_s, num_p, num_o, &format!("compressed cap={cap} {label}"));
            }
        }
    }

    #[test]
    fn cross_check_uniform_and_zipfian() {
        for seed in 1..5u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            cross_check_all_backends(&triples, 30, 6, 30, "uniform");
            let triples = zipfian_predicates(40, 10, 40, 600, 1.2, seed * 2027);
            cross_check_all_backends(&triples, 40, 10, 40, "zipfian");
        }
    }

    #[test]
    fn insert_rejects_out_of_domain_coordinate() {
        // The concrete reproduction this crate's own production hardening
        // was built around: an out-of-domain `insert()` must be rejected
        // with `Err`, not silently accepted (and, worse, not silently
        // corrupt an unrelated in-domain coordinate -- see the sibling
        // assertion below).
        let triples = vec![(0u32, 0u32, 0u32), (1, 0, 1), (2, 1, 2), (3, 1, 3)];
        let mut tree = Heap::build_dynamic(triples.clone(), 64);
        let dom_s = tree.domain_size(0) as u32;
        assert!(!tree.in_bounds(dom_s, 0, 0));

        let before: Vec<bool> = (0..dom_s).flat_map(|s| (0..2u32).flat_map(move |p| (0..4u32).map(move |o| (s, p, o)))).map(|(s, p, o)| tree.contains(s, p, o)).collect();

        let err = tree.insert(dom_s, 0, 0).unwrap_err();
        match err {
            TreeError::OutOfDomain(e) => assert_eq!(e.dim, 0),
            other => panic!("expected OutOfDomain, got {other:?}"),
        }

        let after: Vec<bool> = (0..dom_s).flat_map(|s| (0..2u32).flat_map(move |p| (0..4u32).map(move |o| (s, p, o)))).map(|(s, p, o)| tree.contains(s, p, o)).collect();
        assert_eq!(before, after, "a rejected out-of-domain insert must not change any existing query result");
    }

    #[test]
    fn insert_batch_rejects_out_of_domain_atomically() {
        let mut tree = Heap::build_dynamic(vec![(0u32, 0u32, 0u32)], 64);
        let dom_s = tree.domain_size(0) as u32;
        let mut batch = vec![(1u32, 0u32, 0u32), (dom_s, 0, 0)];
        let err = tree.insert_batch(&mut batch).unwrap_err();
        assert!(matches!(err, TreeError::OutOfDomain(_)));
        assert!(!tree.contains(1, 0, 0), "insert_batch must not partially apply when any triple is out of domain");
    }

    #[test]
    fn build_with_order_rejects_oversized_split() {
        let triples = vec![(0u32, 0u32, 0u32)];
        let bad_split = [BASE, BASE, BASE]; // sums to 3*BASE, far over budget
        let err = expect_err(Heap::build(triples, bad_split, 64));
        assert!(matches!(err, TreeError::InvalidConfig(_)));
    }

    #[test]
    fn insert_one_at_a_time_matches_naive_after_every_insert() {
        let all = uniform(20, 4, 20, 300, 555);
        let split_at = all.len() / 2;
        let (initial, rest) = all.split_at(split_at);

        let mut heap = Heap::build_dynamic(initial.to_vec(), 64);
        let mut seen: Vec<Triple> = initial.to_vec();
        for &(s, p, o) in rest {
            let heap_new = heap.insert(s, p, o).unwrap();
            if heap_new {
                seen.push((s, p, o));
            }
            let naive = NaiveIndex::build(&seen);
            cross_check_tree(&heap, &naive, 20, 4, 20, "heap after insert");
        }
    }

    // ==================== deletion ====================

    #[test]
    fn remove_absent_triple_is_noop() {
        let triples = uniform(20, 4, 20, 100, 11);
        let present: std::collections::HashSet<Triple> = triples.iter().copied().collect();
        let absent = (0..20u32)
            .flat_map(|s| (0..4u32).flat_map(move |p| (0..20u32).map(move |o| (s, p, o))))
            .find(|t| !present.contains(t))
            .expect("a dense 100-triple sample over a 1600-point domain leaves some point absent");

        let mut heap = Heap::build_dynamic(triples.clone(), 64);
        let mem_before = heap.mem_size_bytes();
        let words_before = heap.node_word_count();
        assert!(!heap.remove(absent.0, absent.1, absent.2).unwrap());
        assert_eq!(heap.mem_size_bytes(), mem_before);
        assert_eq!(heap.node_word_count(), words_before);
        let naive = NaiveIndex::build(&triples);
        cross_check_tree(&heap, &naive, 20, 4, 20, "heap after no-op remove");
    }

    #[test]
    fn remove_out_of_domain_returns_false_not_err() {
        let triples = vec![(0u32, 0u32, 0u32), (1, 0, 1), (2, 1, 2), (3, 1, 3)];
        let mut tree = Heap::build_dynamic(triples.clone(), 64);
        let dom_s = tree.domain_size(0) as u32;
        assert!(!tree.in_bounds(dom_s, 0, 0));
        assert!(!tree.remove(dom_s, 0, 0).unwrap());
        let naive = NaiveIndex::build(&triples);
        cross_check_tree(&tree, &naive, dom_s, 2, 4, "heap after out-of-domain remove attempt");
    }

    #[test]
    fn remove_then_contains_false_matches_naive() {
        let all = uniform(20, 4, 20, 300, 9009);
        let mut naive = NaiveIndex::build(&all);
        let mut heap = Heap::build_dynamic(all, 64);
        let distinct: Vec<Triple> = (0..20u32).flat_map(|s| (0..4u32).flat_map(move |p| (0..20u32).map(move |o| (s, p, o)))).filter(|&(s, p, o)| naive.contains(s, p, o)).collect();
        for (s, p, o) in distinct {
            assert!(heap.remove(s, p, o).unwrap());
            assert!(!heap.contains(s, p, o));
            assert!(naive.remove(s, p, o), "naive and tree must agree on what's present before removing it");
            cross_check_tree(&heap, &naive, 20, 4, 20, "heap after remove");
        }
    }

    #[test]
    fn remove_one_of_two_triples_in_the_same_node_preserves_the_other() {
        // A domain small enough (3 total coordinate bits) to fit entirely
        // in one node, guaranteeing both triples live in the exact same
        // existence word regardless of schedule details.
        let dimension_sizes = [2u64, 2u64, 2u64];
        let a: Triple = (0, 0, 0);
        let b: Triple = (1, 1, 1);

        for (first, second) in [(a, b), (b, a)] {
            let mut heap = Heap::build_dynamic_empty(dimension_sizes, 64);
            heap.insert(a.0, a.1, a.2).unwrap();
            heap.insert(b.0, b.1, b.2).unwrap();
            assert_eq!(heap.height(), 1, "test assumes the whole domain fits in one node");

            assert!(heap.remove(first.0, first.1, first.2).unwrap());
            assert!(!heap.contains(first.0, first.1, first.2));
            assert!(heap.contains(second.0, second.1, second.2), "removing one sibling must not disturb the other");

            let naive = NaiveIndex::build(&[second]);
            cross_check_tree(&heap, &naive, 2, 2, 2, "heap after removing one of two co-resident triples");
        }
    }

    #[test]
    fn remove_last_triple_in_a_subtree_cascades_up_and_shrinks_storage() {
        let dimension_sizes = [64u64, 4u64, 64u64];
        let mut heap = Heap::build_dynamic_empty(dimension_sizes, 64);
        heap.insert(0, 0, 0).unwrap();
        heap.insert(63, 3, 63).unwrap();
        assert!(heap.height() >= 2, "test assumes a multi-level tree");

        let words_before = heap.node_word_count();
        let mem_before = heap.mem_size_bytes();

        assert!(heap.remove(0, 0, 0).unwrap());
        assert!(!heap.contains(0, 0, 0));
        assert!(heap.contains(63, 3, 63));
        assert!(heap.node_word_count() < words_before, "cascading remove must physically shrink node_word_count, not just flip a bit");
        assert!(heap.mem_size_bytes() <= mem_before);

        // The root must have survived in a reusable state: further
        // inserts and removes still work correctly afterward.
        assert!(heap.insert(10, 1, 10).unwrap());
        assert!(heap.contains(10, 1, 10));
        assert!(heap.contains(63, 3, 63));
        assert!(!heap.contains(0, 0, 0));
        assert!(heap.remove(63, 3, 63).unwrap());
        assert!(heap.contains(10, 1, 10));
        assert!(!heap.contains(63, 3, 63));
    }

    #[test]
    fn remove_down_to_a_fully_empty_tree_leaves_root_intact() {
        let triples = uniform(10, 2, 10, 30, 42);
        let mut seen: std::collections::HashSet<Triple> = triples.iter().copied().collect();
        let mut heap = Heap::build_dynamic(seen.iter().copied().collect::<Vec<_>>(), 64);
        for &t in seen.clone().iter() {
            assert!(heap.remove(t.0, t.1, t.2).unwrap());
            seen.remove(&t);
        }
        for s in 0..10u32 {
            for p in 0..2u32 {
                for o in 0..10u32 {
                    assert!(!heap.contains(s, p, o), "every triple must be gone after removing them all");
                }
            }
        }
        // The root survived in a sane, reusable state.
        assert!(heap.insert(1, 1, 1).unwrap());
        assert!(heap.contains(1, 1, 1));
        let naive = NaiveIndex::build(&[(1u32, 1u32, 1u32)]);
        cross_check_tree(&heap, &naive, 10, 2, 10, "heap after remove-everything-then-insert");
    }

    #[test]
    fn remove_matches_naive_across_all_backends() {
        for cap in TEST_CAPACITIES {
            let triples = zipfian_predicates(40, 10, 40, 600, 1.2, 0xBADA55 + cap as u64);
            let mut to_remove = triples.clone();
            to_remove.truncate(triples.len() / 2);
            let mut remaining: std::collections::HashSet<Triple> = triples.iter().copied().collect();
            for t in &to_remove {
                remaining.remove(t);
            }
            let remaining_vec: Vec<Triple> = remaining.into_iter().collect();
            let naive = NaiveIndex::build(&remaining_vec);

            let mut heap = Heap::build_dynamic(triples.clone(), cap);
            heap.remove_batch(&mut to_remove.clone()).unwrap();
            cross_check_tree(&heap, &naive, 40, 10, 40, &format!("heap cap={cap} after remove_batch"));

            #[cfg(feature = "file")]
            {
                let mut file = File::build_dynamic(triples.clone(), cap);
                file.remove_batch(&mut to_remove.clone()).unwrap();
                cross_check_tree(&file, &naive, 40, 10, 40, &format!("file cap={cap} after remove_batch"));
            }
            #[cfg(feature = "compressed")]
            {
                let mut compressed = Compressed::build_dynamic(triples.clone(), cap);
                compressed.remove_batch(&mut to_remove.clone()).unwrap();
                cross_check_tree(&compressed, &naive, 40, 10, 40, &format!("compressed cap={cap} after remove_batch"));
            }
        }
    }

    /// Regression test for a real bug found while measuring a (since
    /// reverted -- see `Self::remove_batch`'s own doc comment) batched
    /// bulk-removal design: `CompressedStorage::end_insert` popped its
    /// cache stack back to an unconditional `top = 1` rather than
    /// `top.min(1)`. A root-level branch that needs a brand-new child
    /// chain (`height` small enough -- here exactly 2 -- that the root
    /// itself has no skip field) calls `insert_bytes`, which grows the
    /// block and calls `invalidate_block`, deactivating that block's
    /// cache entry (`top` down to 0) -- but with no skip field to
    /// patch afterward, nothing in the rest of that one `insert()` call
    /// touches storage again to naturally refresh it. The old
    /// unconditional `= 1` then resurrected that now-stale, pre-growth
    /// cached window; a later top-level call (here, a second `insert`,
    /// then `contains`) reusing it without refetching would read the
    /// wrong, truncated content -- confirmed by reverting the fix
    /// locally and watching this exact test fail. `FileStorage` gets
    /// the same defensive `.min(1)` for consistency, though its
    /// write-through-to-disk model was never actually exposed to this
    /// specific hazard.
    #[test]
    fn insert_root_level_branch_in_a_height_two_tree_does_not_resurrect_a_stale_cache() {
        let dims = [16u64, 16u64, 16u64]; // 4+4+4 = 12 bits -> height == 2
        for cap in TEST_CAPACITIES {
            macro_rules! check_backend {
                ($backend:ty) => {{
                    let mut tree = <$backend>::build_dynamic_empty(dims, cap);
                    assert_eq!(tree.height(), 2, "test assumes height == 2");
                    assert!(tree.insert(0, 0, 0).unwrap());
                    assert!(tree.insert(15, 15, 15).unwrap()); // a second, distinct root-level branch
                    assert!(tree.contains(0, 0, 0), "{} cap={cap}: stale cache after root-level branch inserts", stringify!($backend));
                    assert!(tree.contains(15, 15, 15), "{} cap={cap}: stale cache after root-level branch inserts", stringify!($backend));
                    assert!(!tree.contains(1, 1, 1));
                }};
            }
            check_backend!(Heap);
            #[cfg(feature = "file")]
            check_backend!(File);
            #[cfg(feature = "compressed")]
            check_backend!(Compressed);
        }
    }

    #[test]
    fn streaming_insert_build_matches_bulk_build_and_naive() {
        for seed in 1..4u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [30u64, 6u64, 30u64];

            let mut heap_stream = Heap::build_dynamic_empty(dimension_sizes, 64);
            for &(s, p, o) in &triples {
                heap_stream.insert(s, p, o).unwrap();
            }
            cross_check_tree(&heap_stream, &naive, 30, 6, 30, &format!("heap streaming-insert build seed={seed}"));

            let heap_bulk = Heap::build_dynamic(triples.to_vec(), 64);
            assert_eq!(heap_stream.node_word_count(), heap_bulk.node_word_count(), "streaming vs. bulk build should produce the same node count, seed={seed}");
        }
    }

    #[test]
    fn streaming_batch_insert_matches_per_triple_insert_and_bulk_build() {
        // insert_batch sorts into a different physical insertion order
        // than one-triple-at-a-time -- neither should change a single bit
        // of the resulting tree's content.
        for seed in 1..4u64 {
            let mut triples = uniform(40, 6, 40, 600, seed * 9001);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [40u64, 6u64, 40u64];

            let mut heap_batch = Heap::build_dynamic_empty(dimension_sizes, 64);
            let batch_inserted = heap_batch.insert_batch(&mut triples).unwrap();
            cross_check_tree(&heap_batch, &naive, 40, 6, 40, &format!("heap batch-insert build seed={seed}"));

            let heap_bulk = Heap::build_dynamic(triples.clone(), 64);
            assert_eq!(heap_batch.node_word_count(), heap_bulk.node_word_count(), "batch-insert vs. bulk build should produce the same node count, seed={seed}");
            assert!(batch_inserted > 0);

            // Sorting a batch that's already sorted (idempotent re-insert
            // of the same points, now in a different starting order) must
            // report zero *new* insertions and leave the tree unchanged.
            let mut same_again = triples.clone();
            let reinserted = heap_batch.insert_batch(&mut same_again).unwrap();
            assert_eq!(reinserted, 0, "re-inserting the same batch must find nothing new, seed={seed}");
            cross_check_tree(&heap_batch, &naive, 40, 6, 40, &format!("heap batch-insert idempotency seed={seed}"));
        }
    }

    #[cfg(feature = "file")]
    #[test]
    fn file_streaming_insert_stress_small_capacity() {
        // Small block_capacity forces splits (and stack-cache wraps) far
        // more often per triple than production capacities would.
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        let mut file = File::build_dynamic_empty(dimension_sizes, 64);
        let mut rng = SplitMix64::new(2026);
        let mut seen: Vec<Triple> = Vec::new();
        for i in 0..3_000u32 {
            let s = rng.next_range(200_000) as u32;
            let p = rng.next_range(19) as u32;
            let o = rng.next_range(200_000) as u32;
            let is_new = file.insert(s, p, o).unwrap();
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

    #[cfg(feature = "compressed")]
    #[test]
    fn compressed_streaming_insert_stress_small_capacity() {
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        let mut compressed = Compressed::build_dynamic_empty(dimension_sizes, 64);
        let mut rng = SplitMix64::new(2026);
        let mut seen: Vec<Triple> = Vec::new();
        for i in 0..3_000u32 {
            let s = rng.next_range(200_000) as u32;
            let p = rng.next_range(19) as u32;
            let o = rng.next_range(200_000) as u32;
            let is_new = compressed.insert(s, p, o).unwrap();
            if is_new {
                seen.push((s, p, o));
            }
            if !compressed.contains(s, p, o) {
                panic!("self-check failed at triple #{i}: ({s},{p},{o}) not found immediately after insert");
            }
        }
        let naive = NaiveIndex::build(&seen);
        cross_check_tree(&compressed, &naive, 200, 19, 200, "compressed small-capacity stress (partial domain slice)");
    }

    /// Randomly interleaves inserts and removes against a small-capacity,
    /// multi-level tree (the same wide domain as the streaming-insert
    /// stress tests above, forcing a height well past 2 so skip fields
    /// actually exist and get exercised by the cascading removes'
    /// ancestor decrements, not just leaf-level bit flips), self-checking
    /// after every op and fully cross-checking periodically.
    fn interleaved_insert_remove_stress<S: BlockStorage>(mut tree: RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>, label: &str) {
        let mut rng = SplitMix64::new(4242);
        let mut seen: std::collections::HashSet<Triple> = std::collections::HashSet::new();
        for i in 0..3_000u32 {
            let s = rng.next_range(200_000) as u32;
            let p = rng.next_range(19) as u32;
            let o = rng.next_range(200_000) as u32;
            if seen.is_empty() || rng.next_f64() < 0.5 {
                let is_new = tree.insert(s, p, o).unwrap();
                if is_new {
                    seen.insert((s, p, o));
                }
                if !tree.contains(s, p, o) {
                    panic!("{label}: self-check failed at op #{i}: insert({s},{p},{o}) not found immediately after");
                }
            } else {
                let idx = rng.next_range(seen.len() as u64) as usize;
                let target = *seen.iter().nth(idx).unwrap();
                seen.remove(&target);
                if !tree.remove(target.0, target.1, target.2).unwrap() {
                    panic!("{label}: self-check failed at op #{i}: remove({target:?}) reported absent but was tracked as live");
                }
                if tree.contains(target.0, target.1, target.2) {
                    panic!("{label}: self-check failed at op #{i}: remove({target:?}) left it still contains()==true");
                }
            }
            if i % 200 == 199 {
                let naive = NaiveIndex::build(&seen.iter().copied().collect::<Vec<_>>());
                cross_check_tree(&tree, &naive, 200, 19, 200, &format!("{label} interleaved stress op #{i} (partial domain slice)"));
            }
        }
    }

    #[test]
    fn interleaved_insert_remove_stress_small_capacity() {
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        interleaved_insert_remove_stress(Heap::build_dynamic_empty(dimension_sizes, 64), "heap");
    }

    #[cfg(feature = "file")]
    #[test]
    fn file_interleaved_insert_remove_stress_small_capacity() {
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        interleaved_insert_remove_stress(File::build_dynamic_empty(dimension_sizes, 64), "file");
    }

    #[cfg(feature = "compressed")]
    #[test]
    fn compressed_interleaved_insert_remove_stress_small_capacity() {
        let dimension_sizes = [200_000u64, 19u64, 200_000u64];
        interleaved_insert_remove_stress(Compressed::build_dynamic_empty(dimension_sizes, 64), "compressed");
    }

    #[test]
    fn shrink_to_fit_changes_capacity_not_content() {
        for seed in 1..4u64 {
            let mut triples = uniform(50, 8, 50, 800, seed * 12345);
            let naive = NaiveIndex::build(&triples);
            let dimension_sizes = [50u64, 8u64, 50u64];

            let mut heap = Heap::build_dynamic_empty(dimension_sizes, 64);
            heap.insert_batch(&mut triples).unwrap();
            let mem_before = heap.mem_size_bytes();
            heap.shrink_to_fit();
            let mem_after = heap.mem_size_bytes();
            assert!(mem_after <= mem_before, "shrink_to_fit should never increase mem_size_bytes, seed={seed} ({mem_after} > {mem_before})");
            cross_check_tree(&heap, &naive, 50, 8, 50, &format!("heap after shrink_to_fit seed={seed}"));

            // shrink_to_fit must be safe to call and then keep inserting
            // into -- it shouldn't leave the structure unable to grow
            // further.
            let extra = uniform(50, 8, 50, 50, seed * 999);
            for &(s, p, o) in &extra {
                heap.insert(s, p, o).unwrap();
            }
            let mut all: Vec<Triple> = triples.clone();
            all.extend_from_slice(&extra);
            let naive_after = NaiveIndex::build(&all);
            cross_check_tree(&heap, &naive_after, 50, 8, 50, &format!("heap after shrink_to_fit then more inserts seed={seed}"));
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
            let mut heap_restored = Heap::deserialize(&heap_bytes).unwrap();
            cross_check_tree(&heap_restored, &naive, 30, 6, 30, &format!("heap round-trip cap={cap}"));

            let mut seen: Vec<Triple> = initial.to_vec();
            for &(s, p, o) in rest {
                if heap_restored.insert(s, p, o).unwrap() {
                    seen.push((s, p, o));
                }
            }
            let naive_after = NaiveIndex::build(&seen);
            cross_check_tree(&heap_restored, &naive_after, 30, 6, 30, &format!("heap round-trip after insert cap={cap}"));
        }
    }

    #[test]
    fn remove_then_serialize_roundtrip() {
        for cap in TEST_CAPACITIES {
            let all = uniform(30, 6, 30, 400, 7070 + cap as u64);
            let mut seen: std::collections::HashSet<Triple> = all.iter().copied().collect();
            let mut heap = Heap::build_dynamic(seen.iter().copied().collect::<Vec<_>>(), cap);

            let to_remove: Vec<Triple> = seen.iter().take(seen.len() / 2).copied().collect();
            for &(s, p, o) in &to_remove {
                assert!(heap.remove(s, p, o).unwrap());
                seen.remove(&(s, p, o));
            }

            let naive = NaiveIndex::build(&seen.iter().copied().collect::<Vec<_>>());
            cross_check_tree(&heap, &naive, 30, 6, 30, &format!("heap after remove, pre-serialize cap={cap}"));

            let bytes = heap.serialize();
            let restored = Heap::deserialize(&bytes).unwrap();
            cross_check_tree(&restored, &naive, 30, 6, 30, &format!("heap after remove, round-tripped cap={cap}"));
        }
    }

    #[cfg(all(feature = "file", feature = "compressed"))]
    #[test]
    fn serialize_cross_backend_round_trip() {
        // Validates the "serialize/deserialize round-trips through
        // uncompressed bytes" design decision directly: the format is
        // meant to be opaque to which backend produced or consumes it.
        let triples = zipfian_predicates(40, 10, 40, 600, 1.2, 0xC5A55);
        let naive = NaiveIndex::build(&triples);
        let cap = 128;

        let heap_bytes = Heap::build_dynamic(triples.clone(), cap).serialize();
        let file_bytes = File::build_dynamic(triples.clone(), cap).serialize();
        let compressed_bytes = Compressed::build_dynamic(triples.clone(), cap).serialize();

        cross_check_tree(&Heap::deserialize(&file_bytes).unwrap(), &naive, 40, 10, 40, "heap<-file");
        cross_check_tree(&Heap::deserialize(&compressed_bytes).unwrap(), &naive, 40, 10, 40, "heap<-compressed");
        cross_check_tree(&File::deserialize(&heap_bytes).unwrap(), &naive, 40, 10, 40, "file<-heap");
        cross_check_tree(&File::deserialize(&compressed_bytes).unwrap(), &naive, 40, 10, 40, "file<-compressed");
        cross_check_tree(&Compressed::deserialize(&heap_bytes).unwrap(), &naive, 40, 10, 40, "compressed<-heap");
        cross_check_tree(&Compressed::deserialize(&file_bytes).unwrap(), &naive, 40, 10, 40, "compressed<-file");
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bad = vec![0u8; 32];
        let err = expect_err(Heap::deserialize(&bad));
        assert!(matches!(err, TreeError::CorruptData(_)), "expected CorruptData, got {err:?}");
    }

    #[test]
    fn deserialize_rejects_truncated_input() {
        let heap = Heap::build_dynamic(uniform(10, 2, 10, 50, 1), 64);
        let bytes = heap.serialize();
        let truncated = &bytes[..bytes.len() / 2];
        let err = expect_err(Heap::deserialize(truncated));
        assert!(matches!(err, TreeError::CorruptData(_) | TreeError::Io(_)), "expected a data error, got {err:?}");
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        let heap = Heap::build_dynamic(uniform(10, 2, 10, 50, 1), 64);
        let mut bytes = heap.serialize();
        bytes[4] = 200; // version byte, right after the 4-byte magic
        let err = expect_err(Heap::deserialize(&bytes));
        assert!(matches!(err, TreeError::UnsupportedVersion(200)), "expected UnsupportedVersion(200), got {err:?}");
    }

    #[test]
    fn bulk_build_parallel_path_matches_naive_uniform() {
        // Wide enough in S that the root node alone has many active
        // buckets: on a `parallel`-feature build this actually crosses
        // BUILD_PARALLEL_MIN_POINTS and exercises the fan-out path; on a
        // non-parallel build it's still a perfectly good large-uniform
        // correctness check via the sequential path.
        let triples = uniform(100, 6, 100, BUILD_PARALLEL_MIN_POINTS + 10_000, 0xFACE);
        let naive = NaiveIndex::build(&triples);
        let heap = Heap::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&heap, &naive, 100, 6, 100, "heap parallel-path uniform");
    }

    #[test]
    fn bulk_build_parallel_path_matches_naive_skewed() {
        let triples = zipfian_predicates(3, 4, 4000, BUILD_PARALLEL_MIN_POINTS + 10_000, 1.5, 0xBEEF);
        let naive = NaiveIndex::build(&triples);
        let heap = Heap::build_dynamic(triples.to_vec(), DEFAULT_BLOCK_CAPACITY);
        cross_check_tree(&heap, &naive, 3, 4, 4000, "heap parallel-path skewed");
    }

    // ==================== bounded-memory (external-bucketing) build ====================

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
        }
    }

    #[test]
    fn bounded_build_byte_identical_with_tiny_budget() {
        for seed in 1..4u64 {
            let triples = uniform(40, 6, 40, 600, seed * 9001);
            let dims = [40u64, 6u64, 40u64];
            let tiny_budget = BulkBuildBudget::new(25);

            let heap_dynamic = Heap::build_dynamic(triples.clone(), 64);
            let heap_bounded = Heap::build_dynamic_bounded(triples.into_iter(), dims, 64, tiny_budget);
            assert_eq!(heap_dynamic.serialize(), heap_bounded.serialize(), "bounded build with a tiny budget must serialize identically to build_dynamic, seed={seed}");
        }
    }

    #[test]
    fn bulk_build_bounded_path_matches_naive_uniform() {
        let triples = uniform(100, 6, 100, BUILD_PARALLEL_MIN_POINTS + 10_000, 0xFACE);
        let naive = NaiveIndex::build(&triples);
        let dims = [100u64, 6u64, 100u64];
        let budget = BulkBuildBudget::new(5_000);

        let heap = Heap::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&heap, &naive, 100, 6, 100, "heap bounded-path uniform");
    }

    #[test]
    fn bulk_build_bounded_path_matches_naive_skewed() {
        let triples = zipfian_predicates(3, 4, 4000, BUILD_PARALLEL_MIN_POINTS + 10_000, 1.5, 0xBEEF);
        let naive = NaiveIndex::build(&triples);
        let dims = [3u64, 4u64, 4000u64];
        let budget = BulkBuildBudget::new(5_000);

        let heap = Heap::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        cross_check_tree(&heap, &naive, 3, 4, 4000, "heap bounded-path skewed");
    }

    #[test]
    fn bulk_build_bounded_mixed_fits_and_spills_siblings() {
        let mut rng = SplitMix64::new(0xC0DE_1234);
        let mut triples: Vec<Triple> = Vec::new();
        for _ in 0..4_000u32 {
            triples.push((0, rng.next_range(6) as u32, rng.next_range(50) as u32));
        }
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

        let heap_dynamic = Heap::build_dynamic(triples, 64);
        assert_eq!(heap.node_word_count(), heap_dynamic.node_word_count(), "bounded build with mixed fits/spills siblings must match the ordinary in-RAM build's node count");
    }

    #[test]
    fn bulk_build_bounded_wide_branching_many_open_buckets() {
        let triples = uniform(4_000, 60, 4_000, 200_000, 0x0FD_5EED);
        let naive_probe_s: u32 = 17;
        let naive = NaiveIndex::build(&triples);
        let dims = [4_000u64, 60u64, 4_000u64];
        let budget = BulkBuildBudget::new(2_000);

        let heap = Heap::build_dynamic_bounded(triples.into_iter(), dims, DEFAULT_BLOCK_CAPACITY, budget);
        let mut got = heap.query_two_bound([0, 1], [naive_probe_s, 3], 2);
        let mut expect = naive.row(3, naive_probe_s);
        got.sort_unstable();
        expect.sort_unstable();
        assert_eq!(got, expect, "wide-branching bounded build query_two_bound(S={naive_probe_s},P=3,free=O)");
    }

    #[test]
    fn contains_batch_empty_input_is_empty_output() {
        let tree = Heap::build_dynamic(vec![(0, 0, 0)], 64);
        assert_eq!(tree.contains_batch(&[]), Vec::<bool>::new());
    }

    #[test]
    fn contains_batch_matches_contains_scattered_and_clustered() {
        let triples = uniform(500, 20, 500, 20_000, 0x5CA7);
        let naive = NaiveIndex::build(&triples);
        let mut rng = SplitMix64::new(0xBA7C4);

        for cap in TEST_CAPACITIES {
            let tree = Heap::build_dynamic(triples.clone(), cap);

            // Scattered: independent random coordinates, almost no shared
            // Morton-key prefixes expected.
            let scattered: Vec<Triple> = (0..2_000).map(|_| (rng.next_range(500) as u32, rng.next_range(20) as u32, rng.next_range(500) as u32)).collect();
            let expect: Vec<bool> = scattered.iter().map(|&(s, p, o)| naive.contains(s, p, o)).collect();
            assert_eq!(tree.contains_batch(&scattered), expect, "scattered batch cap={cap}");

            // Clustered: many queries sharing the same subject (and often
            // predicate too), the case this method is specifically for.
            let mut clustered: Vec<Triple> = Vec::new();
            for _ in 0..40 {
                let s = rng.next_range(500) as u32;
                for _ in 0..50 {
                    clustered.push((s, rng.next_range(20) as u32, rng.next_range(500) as u32));
                }
            }
            let expect: Vec<bool> = clustered.iter().map(|&(s, p, o)| naive.contains(s, p, o)).collect();
            assert_eq!(tree.contains_batch(&clustered), expect, "clustered batch cap={cap}");
        }
    }

    #[test]
    fn contains_batch_preserves_input_order_handles_duplicates_and_out_of_domain() {
        let triples = vec![(0, 0, 1), (1, 1, 2), (2, 2, 3)];
        let tree = Heap::build_dynamic(triples, 64);
        // Deliberately unsorted, with a repeated triple and an
        // out-of-domain one interleaved -- output must align 1:1 with
        // this exact input order, not any internally-sorted order.
        let queries = [
            (2, 2, 3),       // present
            (99, 0, 0),      // out of domain -> false, not a panic
            (0, 0, 1),       // present
            (2, 2, 3),       // duplicate of index 0, present
            (1, 1, 1),       // absent (in domain)
            (0, 0, 1),       // duplicate of index 2, present
        ];
        let expect = [true, false, true, true, false, true];
        assert_eq!(tree.contains_batch(&queries), expect);
    }

    #[test]
    fn contains_batch_matches_contains_on_degenerate_height_zero_tree() {
        let tree = Heap::build_dynamic(vec![(0, 0, 0)], 64);
        assert_eq!(tree.height(), 0, "test assumes a [1,1,1]-style degenerate domain");
        let queries = [(0, 0, 0), (0, 0, 0)];
        assert_eq!(tree.contains_batch(&queries), [true, true]);
    }

    #[test]
    fn contains_batch_into_reuses_buffer_capacity_across_calls() {
        let tree = Heap::build_dynamic(uniform(50, 5, 50, 2_000, 0x0B0F), 64);
        let mut out = Vec::with_capacity(1_000);
        let cap_before = out.capacity();
        let queries: Vec<Triple> = (0..1_000u32).map(|i| (i % 50, i % 5, i % 50)).collect();
        tree.contains_batch_into(&queries, &mut out);
        assert_eq!(out.len(), queries.len());
        assert_eq!(out.capacity(), cap_before, "contains_batch_into should not need to reallocate when the buffer already has enough capacity");
        let expect: Vec<bool> = queries.iter().map(|&(s, p, o)| tree.contains(s, p, o)).collect();
        assert_eq!(out, expect);
    }

    fn all_bound_patterns(s: u32, p: u32, o: u32) -> [[Option<u32>; 3]; 8] {
        [[None, None, None], [Some(s), None, None], [None, Some(p), None], [None, None, Some(o)], [Some(s), Some(p), None], [Some(s), None, Some(o)], [None, Some(p), Some(o)], [Some(s), Some(p), Some(o)]]
    }

    #[test]
    fn count_matching_matches_naive_and_scan_across_all_bound_patterns() {
        let triples = uniform(30, 5, 30, 3_000, 0xC0117);
        let naive = NaiveIndex::build(&triples);
        let probe_s = triples[0].0;
        let probe_p = triples[0].1;
        let probe_o = triples[0].2;

        for cap in TEST_CAPACITIES {
            let heap = Heap::build_dynamic(triples.clone(), cap);
            for bounds in all_bound_patterns(probe_s, probe_p, probe_o) {
                let got = heap.count_matching(bounds);
                let expect_naive = naive.count_matching(bounds);
                let expect_scan = heap.scan(bounds).count() as u64;
                assert_eq!(got, expect_naive, "count_matching({bounds:?}) vs naive, heap cap={cap}");
                assert_eq!(got, expect_scan, "count_matching({bounds:?}) vs scan, heap cap={cap}");
            }
            #[cfg(feature = "file")]
            {
                let file = File::build_dynamic(triples.clone(), cap);
                for bounds in all_bound_patterns(probe_s, probe_p, probe_o) {
                    assert_eq!(file.count_matching(bounds), naive.count_matching(bounds), "count_matching({bounds:?}) vs naive, file cap={cap}");
                }
            }
            #[cfg(feature = "compressed")]
            {
                let compressed = Compressed::build_dynamic(triples.clone(), cap);
                for bounds in all_bound_patterns(probe_s, probe_p, probe_o) {
                    assert_eq!(compressed.count_matching(bounds), naive.count_matching(bounds), "count_matching({bounds:?}) vs naive, compressed cap={cap}");
                }
            }
        }
    }

    #[test]
    fn count_matching_out_of_domain_bound_is_zero_not_panic() {
        let tree = Heap::build_dynamic(vec![(0, 0, 0), (1, 1, 1)], 64);
        assert_eq!(tree.count_matching([Some(99), None, None]), 0);
        assert_eq!(tree.count_matching([None, Some(99), None]), 0);
        assert_eq!(tree.count_matching([None, None, Some(99)]), 0);
    }

    #[test]
    fn count_matching_all_wildcards_equals_total_distinct_triple_count() {
        let triples = uniform(40, 6, 40, 5_000, 0xBADD1E);
        let naive = NaiveIndex::build(&triples);
        for cap in TEST_CAPACITIES {
            let heap = Heap::build_dynamic(triples.clone(), cap);
            assert_eq!(heap.count_matching([None, None, None]), naive.len(), "cap={cap}");
        }
    }

    #[test]
    fn count_matching_matches_contains_on_degenerate_height_zero_tree() {
        let tree = Heap::build_dynamic(vec![(0, 0, 0)], 64);
        assert_eq!(tree.height(), 0, "test assumes a [1,1,1]-style degenerate domain");
        assert_eq!(tree.count_matching([None, None, None]), 1);
        assert_eq!(tree.count_matching([Some(0), Some(0), Some(0)]), 1);
    }

    #[test]
    fn count_matching_after_remove_matches_naive() {
        let triples = uniform(25, 4, 25, 1_500, 0x5EED5);
        let mut heap = Heap::build_dynamic(triples.clone(), 64);
        let mut naive = NaiveIndex::build(&triples);
        let mut live: Vec<Triple> = triples.clone();
        live.sort_unstable();
        live.dedup();
        let mut rng = SplitMix64::new(0x9001);
        for _ in 0..300.min(live.len()) {
            let idx = rng.next_range(live.len() as u64) as usize;
            let (s, p, o) = live.swap_remove(idx);
            assert!(heap.remove(s, p, o).unwrap());
            assert!(naive.remove(s, p, o));
        }
        for bounds in all_bound_patterns(triples[0].0, triples[0].1, triples[0].2) {
            assert_eq!(heap.count_matching(bounds), naive.count_matching(bounds), "count_matching({bounds:?}) after removals");
        }
    }
}
