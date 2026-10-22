//! Rank-free root-first BFS compaction. Existing live edit slots provide depth
//! chains and forwarding addresses; stored records use block displacement runs.
use super::{
    BlockCompactionStats, BlockStorage, EDIT, MutableFst, PackedFst, Result, SEALED, TreeError,
};
use crate::automata_core::BitmapSource;
use crate::storage::ReclaimingBlockStorage;
use std::ops::Range;
use std::time::Instant;

// Most old identities fit a 32-bit temporary output reference after removing
// the cursor tags and base alignment. Larger identities use a per-depth escape
// stream in output-edge order, preserving arbitrary supported source addresses.
pub(super) struct Codec {
    pub(super) base_slots: u64,
    pub(super) sealed_slots: u64,
}
impl Codec {
    pub(super) fn key(&self, cursor: u64) -> u64 {
        if cursor & EDIT != 0 {
            self.base_slots + self.sealed_slots + (cursor & !EDIT)
        } else if cursor & SEALED != 0 {
            self.base_slots + (cursor & !SEALED)
        } else {
            cursor / 4
        }
    }
    pub(super) fn cursor(&self, key: u64) -> u64 {
        if key < self.base_slots {
            key * 4
        } else if key < self.base_slots + self.sealed_slots {
            SEALED | (key - self.base_slots)
        } else {
            EDIT | (key - self.base_slots - self.sealed_slots)
        }
    }
    pub(super) fn encode(&self, cursor: u64, escapes: &mut Vec<u64>) -> u32 {
        let key = self.key(cursor);
        if key < u32::MAX as u64 {
            key as u32
        } else {
            escapes.push(cursor);
            u32::MAX
        }
    }
}

pub(super) fn word(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

// No rank prefixes. These are only reached-record bitmaps and source-block
// enumeration. Edit-state liveness already exists in the reference-counted arena.
struct Pages {
    directory: Vec<usize>,
    blocks: Vec<usize>,
    words: Vec<u64>,
    words_per_page: usize,
    growths: usize,
    count: usize,
}
impl Pages {
    fn new(blocks: usize, slots_per_block: usize) -> Self {
        Self {
            directory: vec![0; blocks],
            blocks: Vec::new(),
            words: Vec::new(),
            words_per_page: slots_per_block.div_ceil(64),
            growths: 0,
            count: 0,
        }
    }
    fn mark(&mut self, block: usize, slot: usize) {
        if self.directory[block] == 0 {
            let cap = self.blocks.capacity();
            self.blocks.push(block);
            self.growths += usize::from(cap != self.blocks.capacity());
            self.directory[block] = self.blocks.len();
            let cap = self.words.capacity();
            self.words.resize(self.words.len() + self.words_per_page, 0);
            self.growths += usize::from(cap != self.words.capacity());
        }
        let page = self.directory[block] - 1;
        self.words[page * self.words_per_page + slot / 64] |= 1 << (slot % 64);
    }
    fn finish(&mut self) {
        self.blocks.sort_unstable();
        self.count = self.words.iter().map(|w| w.count_ones() as usize).sum();
    }
    fn clear(&mut self) {
        for &block in &self.blocks {
            self.directory[block] = 0;
        }
        self.blocks.clear();
        self.words.clear();
        self.count = 0;
    }
    fn page(&self, block: usize) -> usize {
        debug_assert_ne!(self.directory[block], 0);
        self.directory[block] - 1
    }
    fn slots(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        let start = self.page(block) * self.words_per_page;
        self.words[start..start + self.words_per_page]
            .iter()
            .enumerate()
            .flat_map(|(i, &w)| {
                let mut w = w;
                std::iter::from_fn(move || {
                    if w == 0 {
                        return None;
                    }
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    Some(i * 64 + bit)
                })
            })
    }
    fn bytes(&self) -> usize {
        crate::vec_bytes(&self.directory)
            + crate::vec_bytes(&self.blocks)
            + crate::vec_bytes(&self.words)
    }
}

struct Frontier {
    base: Pages,
    sealed: Pages,
    base_bits: u32,
    sealed_bits: u32,
}
impl Frontier {
    fn new<S: BlockStorage>(g: &MutableFst<S>) -> Self {
        let sealed_capacity = g.sealed.as_ref().map_or(1024, |s| s.capacity);
        Self {
            base: Pages::new(g.base.block_count(), g.block_capacity() / 4),
            sealed: Pages::new(
                g.sealed.as_ref().map_or(0, |s| s.storage.block_count()),
                sealed_capacity,
            ),
            base_bits: g.block_capacity().trailing_zeros(),
            sealed_bits: sealed_capacity.trailing_zeros(),
        }
    }
    fn mark(&mut self, cursor: u64) {
        if cursor & EDIT != 0 {
            return;
        } // live edits are already depth-linked
        if cursor & SEALED != 0 {
            let address = cursor & !SEALED;
            self.sealed.mark(
                (address >> self.sealed_bits) as usize,
                (address & ((1 << self.sealed_bits) - 1)) as usize,
            );
        } else {
            self.base.mark(
                (cursor >> self.base_bits) as usize,
                (cursor & ((1 << self.base_bits) - 1)) as usize / 4,
            );
        }
    }
    fn finish(&mut self) {
        self.base.finish();
        self.sealed.finish();
    }
    fn clear(&mut self) {
        self.base.clear();
        self.sealed.clear();
    }
    fn total(&self) -> usize {
        self.base.count + self.sealed.count
    }
    fn bytes(&self) -> usize {
        self.base.bytes() + self.sealed.bytes()
    }
    fn growths(&self) -> usize {
        self.base.growths + self.sealed.growths
    }
}

#[derive(Clone, Copy)]
struct Run {
    start: u32,
    // Modular displacement also handles a target address below the old offset.
    // Both sides use the same arithmetic; the final edge delta is checked.
    displacement: u64,
}
struct Runs {
    ranges: Vec<Range<usize>>,
    entries: Vec<Run>,
    growths: usize,
}
impl Runs {
    fn new() -> Self {
        Self {
            ranges: Vec::new(),
            entries: Vec::new(),
            growths: 0,
        }
    }
    fn reset(&mut self, pages: usize) {
        self.entries.clear();
        let cap = self.ranges.capacity();
        if pages > cap {
            self.ranges.reserve_exact(pages - self.ranges.len());
        }
        self.ranges.resize(pages, 0..0);
        self.growths += usize::from(cap != self.ranges.capacity());
    }
    fn push(&mut self, start: usize, offset: usize, address: u64) {
        let displacement = address.wrapping_sub(offset as u64);
        if self.entries.len() == start || self.entries.last().unwrap().displacement != displacement
        {
            let cap = self.entries.capacity();
            self.entries.push(Run {
                start: offset as u32,
                displacement,
            });
            self.growths += usize::from(cap != self.entries.capacity());
        }
    }
    fn get(&self, page: usize, offset: usize) -> u64 {
        let range = &self.ranges[page];
        let entries = &self.entries[range.clone()];
        let run = if entries.len() == 1 {
            entries[0]
        } else {
            entries[entries.partition_point(|r| r.start as usize <= offset) - 1]
        };
        (offset as u64).wrapping_add(run.displacement)
    }
    fn bytes(&self) -> usize {
        crate::vec_bytes(&self.ranges) + crate::vec_bytes(&self.entries)
    }
}

struct Writer<S: ReclaimingBlockStorage> {
    storage: S,
    capacity: usize,
    codec: Codec,
    base_runs: Runs,
    sealed_runs: Runs,
    pending_escapes: Vec<u64>,
    stats: BlockCompactionStats,
    source_bytes: usize,
    ordinal: usize,
}
impl<S: ReclaimingBlockStorage> Writer<S> {
    fn emit(&mut self, word: u64, children: impl Iterator<Item = u64>, next: &mut Frontier) -> u64 {
        let mut record = [0u8; 8 + 64 * 4];
        record[..8].copy_from_slice(&word.to_le_bytes());
        let mut length = 8;
        for child in children {
            next.mark(child);
            let token = self.codec.encode(child, &mut self.pending_escapes);
            record[length..length + 4].copy_from_slice(&token.to_le_bytes());
            length += 4;
            self.stats.reachable_transitions += 1;
        }
        let pos = self.storage.append_record(&record[..length], self.capacity);
        let address = ((pos.block as u64) << self.capacity.trailing_zeros()) | pos.offset as u64;
        self.ordinal += 1; // the traversal itself supplies the ordinal
        self.stats.output_record_bytes += length;
        self.stats.peak_uncompressed_block_bytes = self
            .stats
            .peak_uncompressed_block_bytes
            .max(self.source_bytes + self.stats.output_record_bytes);
        address
    }
    fn fixup(
        &mut self,
        blocks: Range<usize>,
        escapes: &[u64],
        current: &Frontier,
        graph: &MutableFst<S>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        let mut escaped = 0;
        for block in blocks {
            self.storage.read_compaction_block(block, scratch);
            let mut offset = 0;
            while offset < scratch.len() {
                let count = word(scratch, offset).count_ones() as usize;
                let parent = ((block as u64) << self.capacity.trailing_zeros()) | offset as u64;
                offset += 8;
                for _ in 0..count {
                    let token = u32::from_le_bytes(scratch[offset..offset + 4].try_into().unwrap());
                    let child = if token == u32::MAX {
                        let c = escapes[escaped];
                        escaped += 1;
                        c
                    } else {
                        self.codec.cursor(token as u64)
                    };
                    let target = if child & EDIT != 0 {
                        // Its occupancy word became its forwarding address
                        // after emission. No new edit-address array is needed.
                        graph.nodes[(child & !EDIT) as usize]
                            .as_ref()
                            .unwrap()
                            .signature
                            .word
                    } else if child & SEALED != 0 {
                        let address = child & !SEALED;
                        let block = (address >> current.sealed_bits) as usize;
                        let offset = (address & ((1 << current.sealed_bits) - 1)) as usize;
                        self.sealed_runs.get(current.sealed.page(block), offset)
                    } else {
                        let block = (child >> current.base_bits) as usize;
                        let offset = (child & ((1 << current.base_bits) - 1)) as usize;
                        self.base_runs.get(current.base.page(block), offset)
                    };
                    let delta = target
                        .checked_sub(parent)
                        .and_then(|v| u32::try_from(v).ok())
                        .filter(|&v| v != 0)
                        .ok_or_else(|| {
                            TreeError::InvalidConfig("FST child delta exceeds 32 bits".into())
                        })?;
                    scratch[offset..offset + 4].copy_from_slice(&delta.to_le_bytes());
                    offset += 4;
                }
            }
            self.storage.rewrite_compaction_block(block, scratch);
            self.stats.output_fixup_blocks += 1;
        }
        debug_assert_eq!(escaped, escapes.len());
        Ok(())
    }
}

pub(in crate::mutable_fst) fn run<S: ReclaimingBlockStorage>(
    mut graph: MutableFst<S>,
) -> Result<(PackedFst<S>, BlockCompactionStats)> {
    let capacity = graph.block_capacity();
    let domains = graph.dimension_sizes();
    let height = graph.height();
    let visits = u64::try_from(graph.logical_visits)
        .map_err(|_| TreeError::InvalidConfig("FST logical visit count exceeds u64".into()))?;
    let stats = BlockCompactionStats {
        input_states: graph.base.node_count() as usize
            + graph.live_nodes
            + graph
                .sealed
                .as_ref()
                .map_or(0, |s| s.per_level.iter().sum()),
        source_blocks: graph.base.block_count()
            + graph.sealed.as_ref().map_or(0, |s| s.storage.block_count()),
        ..Default::default()
    };
    graph.register = Vec::new();
    graph.memo = Vec::new();
    graph.base_register = None;
    graph.sampled_register = None;
    graph.free = Vec::new();
    if graph.source().word(graph.root) == 0 {
        drop(graph);
        return Ok((
            PackedFst::try_build(Vec::new(), domains, capacity)?,
            BlockCompactionStats {
                reachable_states: 1,
                released_blocks: stats.source_blocks,
                ..stats
            },
        ));
    }
    // All occupied arena slots are live by reference-count maintenance.
    // Reuse the consumed nodes' reference counters as next-slot links.
    let mut edit_heads = vec![usize::MAX; height];
    let mut edit_counts = vec![0usize; height];
    for (index, node) in graph.nodes.iter_mut().enumerate().rev() {
        if let Some(node) = node {
            debug_assert_ne!(node.references, 0);
            let depth = node.level;
            node.references = edit_heads[depth];
            edit_heads[depth] = index;
            edit_counts[depth] += 1;
        }
    }
    debug_assert_eq!(edit_counts.iter().sum::<usize>(), graph.live_nodes);
    let mut current = Frontier::new(&graph);
    let mut next = Frontier::new(&graph);
    current.mark(graph.root);
    current.finish();
    let sealed_blocks = graph.sealed.as_ref().map_or(0, |s| s.storage.block_count());
    let mut sealed_max = vec![u8::MAX; sealed_blocks];
    let mut retire: Vec<Vec<usize>> = (0..height).map(|_| Vec::new()).collect();
    let codec = Codec {
        base_slots: graph.base.block_count() as u64 * (capacity / 4) as u64,
        sealed_slots: graph
            .sealed
            .as_ref()
            .map_or(0, |s| s.storage.block_count() as u64 * s.capacity as u64),
    };
    let source_bytes = (0..graph.base.block_count())
        .map(|b| graph.base.block_len(b))
        .sum::<usize>()
        + graph.sealed.as_ref().map_or(0, |s| {
            (0..s.storage.block_count())
                .map(|b| s.storage.block_len(b))
                .sum()
        });
    let mut writer = Writer {
        storage: S::empty(),
        capacity,
        codec,
        base_runs: Runs::new(),
        sealed_runs: Runs::new(),
        pending_escapes: Vec::new(),
        source_bytes,
        ordinal: 0,
        stats: BlockCompactionStats {
            input_uncompressed_block_bytes: source_bytes,
            peak_uncompressed_block_bytes: source_bytes,
            ..stats
        },
    };
    let mut scratch =
        Vec::with_capacity(capacity.max(graph.sealed.as_ref().map_or(0, |s| s.capacity)));
    let mut previous = 0..0;
    let mut previous_escapes = Vec::new();
    let mut base_retired = 0;
    let mut counts = Vec::with_capacity(height);
    for depth in 0..height {
        let start = Instant::now();
        let nodes = current.total() + edit_counts[depth];
        debug_assert_ne!(nodes, 0);
        counts.push(nodes);
        writer.stats.reachable_states += nodes;
        let leaf = depth + 1 == height;
        writer.ordinal = 0;
        writer.stats.peak_relocation_states =
            writer.stats.peak_relocation_states.max(current.total());
        writer.base_runs.reset(current.base.blocks.len());
        writer.sealed_runs.reset(current.sealed.blocks.len());
        if !writer.storage.start_new_append_block(capacity) {
            return Err(TreeError::InvalidConfig(
                "depth compaction requires explicit append-block boundaries".into(),
            ));
        }
        let mut first = None;
        let mut last = 0;
        for &block in &current.base.blocks {
            let run_start = writer.base_runs.entries.len();
            graph.base.read_compaction_block(block, &mut scratch);
            writer.stats.source_block_reads += 1;
            for slot in current.base.slots(block) {
                let offset = slot * 4;
                let cursor = ((block as u64) << current.base_bits) | offset as u64;
                let w = word(&scratch, offset);
                let n = if leaf { 0 } else { w.count_ones() as usize };
                let children = (0..n).map(|i| {
                    cursor
                        + u32::from_le_bytes(
                            scratch[offset + 8 + i * 4..offset + 12 + i * 4]
                                .try_into()
                                .unwrap(),
                        ) as u64
                });
                let address = writer.emit(w, children, &mut next);
                writer.base_runs.push(run_start, offset, address);
                first.get_or_insert((address >> current.base_bits) as usize);
                last = (address >> current.base_bits) as usize + 1;
            }
            writer.base_runs.ranges[current.base.page(block)] =
                run_start..writer.base_runs.entries.len();
            // Original records are BFS ordered. A later block with live
            // records at this depth proves all earlier blocks are finished.
            while base_retired < block {
                writer.source_bytes -= graph.base.block_len(base_retired);
                graph.base.release_compaction_block(base_retired);
                writer.stats.released_blocks += 1;
                base_retired += 1;
            }
        }
        if let Some(sealed) = &mut graph.sealed {
            for &block in &current.sealed.blocks {
                let run_start = writer.sealed_runs.entries.len();
                sealed.storage.read_compaction_block(block, &mut scratch);
                writer.stats.source_block_reads += 1;
                if sealed_max[block] == u8::MAX {
                    let mut offset = 0;
                    let mut max = 0;
                    while offset < scratch.len() {
                        let level = scratch[offset] as usize;
                        max = max.max(level);
                        let count = if level + 1 == height {
                            0
                        } else {
                            word(&scratch, offset + 1).count_ones() as usize
                        };
                        offset += 9 + 8 * count;
                    }
                    sealed_max[block] = max as u8;
                    retire[max].push(block);
                }
                for offset in current.sealed.slots(block) {
                    debug_assert_eq!(scratch[offset] as usize, depth);
                    let w = word(&scratch, offset + 1);
                    let n = if leaf { 0 } else { w.count_ones() as usize };
                    let children = (0..n).map(|i| word(&scratch, offset + 9 + i * 8));
                    let address = writer.emit(w, children, &mut next);
                    writer.sealed_runs.push(run_start, offset, address);
                    first.get_or_insert((address >> current.base_bits) as usize);
                    last = (address >> current.base_bits) as usize + 1;
                }
                writer.sealed_runs.ranges[current.sealed.page(block)] =
                    run_start..writer.sealed_runs.entries.len();
            }
            for &block in &retire[depth] {
                writer.source_bytes -= sealed.storage.block_len(block);
                sealed.storage.release_block_payload(block);
                writer.stats.released_blocks += 1;
            }
        }
        let mut edit = edit_heads[depth];
        while edit != usize::MAX {
            let node = graph.nodes[edit].as_ref().expect("live depth-linked edit");
            debug_assert_eq!(node.level, depth);
            let next_edit = node.references;
            let address = writer.emit(
                node.signature.word,
                node.signature.children.iter().copied(),
                &mut next,
            );
            let signature =
                std::sync::Arc::get_mut(&mut graph.nodes[edit].as_mut().unwrap().signature)
                    .expect("consuming compaction released signature registers");
            signature.word = address; // forward in its existing arena slot
            // The child payload will never be read again; release it while
            // retaining only the existing word as a forwarding address.
            signature.children = Vec::new();
            writer.stats.forwarded_edit_states += 1;
            first.get_or_insert((address >> current.base_bits) as usize);
            last = (address >> current.base_bits) as usize + 1;
            edit = next_edit;
        }
        debug_assert_eq!(writer.ordinal, nodes);
        let fixup = Instant::now();
        writer.fixup(previous, &previous_escapes, &current, &graph, &mut scratch)?;
        // Every incoming edge to this depth has now been resolved. Forwarding
        // slots and their remaining allocation can be released immediately.
        let mut edit = edit_heads[depth];
        while edit != usize::MAX {
            let node = graph.nodes[edit].take().unwrap();
            edit = node.references;
        }
        writer.stats.fixup_ns += fixup.elapsed().as_nanos() as u64;
        writer.stats.copy_ns += start.elapsed().as_nanos() as u64;
        let plan = Instant::now();
        next.finish();
        writer.stats.planning_ns += plan.elapsed().as_nanos() as u64;
        writer.stats.frontier_bytes = writer
            .stats
            .frontier_bytes
            .max(current.bytes() + next.bytes());
        writer.stats.peak_frontier_states = writer
            .stats
            .peak_frontier_states
            .max(nodes + next.total() + edit_counts.get(depth + 1).copied().unwrap_or(0));
        let relocation_bytes = writer.base_runs.bytes() + writer.sealed_runs.bytes();
        writer.stats.relocation_bytes = writer.stats.relocation_bytes.max(relocation_bytes);
        writer.stats.relocation_runs = writer
            .stats
            .relocation_runs
            .max(writer.base_runs.entries.len() + writer.sealed_runs.entries.len());
        writer.stats.scratch_bytes = writer.stats.scratch_bytes.max(crate::vec_bytes(&scratch));
        let auxiliary = crate::vec_bytes(&edit_heads)
            + crate::vec_bytes(&edit_counts)
            + crate::vec_bytes(&sealed_max)
            + crate::vec_bytes(&retire)
            + retire.iter().map(crate::vec_bytes).sum::<usize>()
            + crate::vec_bytes(&counts)
            + crate::vec_bytes(&previous_escapes)
            + crate::vec_bytes(&writer.pending_escapes);
        writer.stats.workspace_bytes = writer
            .stats
            .workspace_bytes
            .max(current.bytes() + next.bytes() + relocation_bytes + auxiliary);
        previous = first.expect("nonempty live depth")..last;
        previous_escapes.clear();
        std::mem::swap(&mut previous_escapes, &mut writer.pending_escapes);
        std::mem::swap(&mut current, &mut next);
        next.clear();
    }
    debug_assert_eq!(current.total(), 0);
    debug_assert!(previous_escapes.is_empty()); // leaf depth has no references
    writer.stats.frontier_buffer_growths = current.growths() + next.growths();
    writer.stats.relocation_buffer_growths = writer.base_runs.growths + writer.sealed_runs.growths;
    // Unreached source blocks never need a liveness/header scan. Retire their
    // payload now; earlier monotone/max-depth retirement handles reached ones.
    for block in base_retired..graph.base.block_count() {
        graph.base.release_compaction_block(block);
        writer.stats.released_blocks += 1;
    }
    if let Some(sealed) = &mut graph.sealed {
        for (block, &max) in sealed_max.iter().enumerate() {
            if max == u8::MAX {
                sealed.storage.release_block_payload(block);
                writer.stats.released_blocks += 1;
            }
        }
    }
    debug_assert_eq!(writer.stats.released_blocks, writer.stats.source_blocks);
    writer.storage.finish_append(capacity);
    drop(graph);
    Ok((
        PackedFst::from_compacted_storage(writer.storage, domains, capacity, counts, visits),
        writer.stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutable_fst::UpdateLookup;

    #[test]
    fn displacement_runs_handle_holes_backward_lookups_and_wide_addresses() {
        let mut runs = Runs::new();
        runs.reset(1);
        let start = runs.entries.len();
        for (old, new) in [(240, 0), (248, 8), (260, 40), (276, 56), (500, 280)] {
            runs.push(start, old, new);
        }
        runs.ranges[0] = start..runs.entries.len();
        assert_eq!(runs.entries.len(), 2);
        for (old, new) in [(500, 280), (240, 0), (276, 56), (248, 8), (260, 40)] {
            assert_eq!(runs.get(0, old), new);
        }
        let bytes = runs.bytes();
        let growths = runs.growths;
        runs.reset(1);
        runs.push(0, 12, (1 << 40) + 12);
        runs.push(0, 24, (1 << 40) + 24);
        runs.ranges[0] = 0..1;
        assert_eq!(runs.get(0, 24), (1 << 40) + 24);
        assert_eq!(runs.bytes(), bytes);
        assert_eq!(runs.growths, growths);
    }

    fn edits<S: ReclaimingBlockStorage>(lookup: UpdateLookup) -> MutableFst<S> {
        let initial: Vec<_> = (0..512u32)
            .flat_map(|s| (0..2).map(move |p| (s, p, (s * 197 + p * 73) % 512)))
            .collect();
        let mut g =
            PackedFst::<S>::build(initial, [512, 8, 512], 512).into_mutable_with_lookup(lookup);
        for s in 0..300u32 {
            if s % 3 != 0 {
                assert!(g.remove(s, 0, s * 197 % 512).unwrap());
            }
            assert!(g.insert(s, 3, (s * 37 + 11) % 512).unwrap());
        }
        assert!(g.insert(511, 7, 511).unwrap());
        assert!(g.remove(511, 7, 511).unwrap());
        assert!(!g.free.is_empty());
        g
    }

    fn exercise<S: ReclaimingBlockStorage>() {
        for lookup in [
            UpdateLookup::Indexed,
            UpdateLookup::Sampled(8),
            UpdateLookup::Memoized,
            UpdateLookup::ScanBase,
            UpdateLookup::ScanAll,
        ] {
            let old = edits::<S>(lookup).compact_depths().unwrap().0;
            let graph = edits::<S>(lookup);
            let edit_states = graph.live_nodes;
            let (packed, stats) = graph.compact_streaming().unwrap();
            assert_eq!(stats.forwarded_edit_states, edit_states);
            assert_eq!(stats.rank_lookups, 0);
            assert_eq!(stats.rank_bytes, 0);
            assert_eq!(stats.released_blocks, stats.source_blocks);
            assert_eq!(packed.serialize(), old.serialize());
            // The public default must use precisely this layout/algorithm,
            // rather than the older borrowed rebuild used by to_fst().
            assert_eq!(
                edits::<S>(lookup).freeze().unwrap().serialize(),
                packed.serialize()
            );
            assert_eq!(
                edits::<S>(lookup)
                    .freeze_in_place()
                    .unwrap()
                    .compact()
                    .unwrap()
                    .serialize(),
                edits::<S>(lookup)
                    .freeze_in_place()
                    .unwrap()
                    .compact_streaming()
                    .unwrap()
                    .0
                    .serialize()
            );
            let oracle = crate::Fst::deserialize(&packed.serialize()).unwrap();
            assert_eq!(
                packed.scan([None; 3]).collect::<Vec<_>>(),
                oracle.scan([None; 3]).collect::<Vec<_>>()
            );
        }
        for domains in [[1, 1, 1], [1 << 32, 1, 1 << 32]] {
            let graph = PackedFst::<S>::build(Vec::new(), domains, 512).into_mutable();
            let (packed, stats) = graph.compact_streaming().unwrap();
            assert_eq!(packed.node_count(), 1);
            assert_eq!(stats.released_blocks, stats.source_blocks);
            let t = ((domains[0] - 1) as u32, 0, (domains[2] - 1) as u32);
            let mut graph = packed.into_mutable();
            graph.insert(t.0, t.1, t.2).unwrap();
            let (packed, stats) = graph.compact_streaming().unwrap();
            assert_eq!(stats.rank_lookups, 0);
            assert_eq!(stats.released_blocks, stats.source_blocks);
            assert_eq!(packed.scan([None; 3]).collect::<Vec<_>>(), vec![t]);
        }
    }

    #[test]
    fn arena_metadata_and_output_match_rank_based_compaction() {
        exercise::<crate::storage::HeapStorage>();
        #[cfg(feature = "file")]
        exercise::<crate::storage_file::FileStorage>();
        #[cfg(feature = "compressed")]
        exercise::<crate::storage_compressed::CompressedStorage>();
    }

    #[test]
    fn escaped_edit_identity_resolves_through_existing_arena_slot() {
        let mut graph = crate::Fst::build(Vec::new(), [16, 4, 16], 512).into_mutable();
        graph.insert(1, 0, 2).unwrap();
        graph.register = Vec::new();
        let child = graph.root;
        assert_ne!(child & EDIT, 0);
        std::sync::Arc::get_mut(
            &mut graph.nodes[(child & !EDIT) as usize]
                .as_mut()
                .unwrap()
                .signature,
        )
        .unwrap()
        .word = 512;
        let current = Frontier::new(&graph);
        let mut next = Frontier::new(&graph);
        let mut writer = Writer {
            storage: crate::storage::HeapStorage::empty(),
            capacity: 512,
            codec: Codec {
                base_slots: 1 << 40,
                sealed_slots: 0,
            },
            base_runs: Runs::new(),
            sealed_runs: Runs::new(),
            pending_escapes: Vec::new(),
            stats: BlockCompactionStats::default(),
            source_bytes: 0,
            ordinal: 0,
        };
        writer.emit(1, std::iter::once(child), &mut next);
        let escapes = std::mem::take(&mut writer.pending_escapes);
        assert_eq!(escapes, [child]);
        writer
            .fixup(0..1, &escapes, &current, &graph, &mut Vec::new())
            .unwrap();
        assert_eq!(writer.storage.read_var(0, 8, 4), 512);
    }
}
