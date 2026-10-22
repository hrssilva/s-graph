//! One source-edge traversal, two reusable live frontiers, one relocation map.
//! Pending output references are fixed up one committed block at a time.
use super::*;
use std::time::Instant;

use super::super::streaming::{Codec, word};

// Directories are per source block, not per source byte/state. Bitmap pages
// exist only for blocks reached at this depth. Page slots retain their indices
// when `blocks` is sorted for source-local enumeration.
struct Pages {
    directory: Vec<usize>,
    blocks: Vec<usize>,
    words: Vec<u64>,
    prefix: Vec<usize>,
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
            prefix: Vec::new(),
            words_per_page: slots_per_block.div_ceil(64),
            growths: 0,
            count: 0,
        }
    }
    fn mark(&mut self, block: usize, slot: usize) {
        if self.directory[block] == 0 {
            let capacity = self.blocks.capacity();
            self.blocks.push(block);
            self.growths += usize::from(capacity != self.blocks.capacity());
            self.directory[block] = self.blocks.len();
            let capacity = self.words.capacity();
            self.words.resize(self.words.len() + self.words_per_page, 0);
            self.growths += usize::from(capacity != self.words.capacity());
        }
        let page = self.directory[block] - 1;
        self.words[page * self.words_per_page + slot / 64] |= 1 << (slot % 64);
    }
    fn finish(&mut self) {
        self.blocks.sort_unstable();
        self.prefix.clear();
        let capacity = self.prefix.capacity();
        // This complete depth's required length is already known. Amortized
        // doubling here can retain almost twice the largest rank directory.
        self.prefix.reserve_exact(self.words.len() + 1);
        self.growths += usize::from(capacity != self.prefix.capacity());
        self.prefix.resize(self.words.len() + 1, 0);
        let mut count = 0;
        // Rank follows physical enumeration, even though pages were allocated
        // in discovery order. Fixed-width leaf addresses then need no table.
        for &block in &self.blocks {
            let start = (self.directory[block] - 1) * self.words_per_page;
            for i in start..start + self.words_per_page {
                self.prefix[i] = count;
                count += self.words[i].count_ones() as usize;
            }
        }
        self.prefix[self.words.len()] = count;
        self.count = count;
    }
    fn clear(&mut self) {
        for &block in &self.blocks {
            self.directory[block] = 0;
        }
        self.blocks.clear();
        self.words.clear();
        self.prefix.clear();
        self.count = 0;
    }
    fn total(&self) -> usize {
        self.count
    }
    fn index(&self, block: usize, slot: usize) -> usize {
        debug_assert_ne!(self.directory[block], 0);
        let word = (self.directory[block] - 1) * self.words_per_page + slot / 64;
        debug_assert_ne!(self.words[word] & (1 << (slot % 64)), 0);
        self.prefix[word] + (self.words[word] & ((1u64 << (slot % 64)) - 1)).count_ones() as usize
    }
    fn slots(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        let start = (self.directory[block] - 1) * self.words_per_page;
        self.words[start..start + self.words_per_page]
            .iter()
            .enumerate()
            .flat_map(|(i, &word)| {
                let mut word = word;
                std::iter::from_fn(move || {
                    if word == 0 {
                        None
                    } else {
                        let slot = word.trailing_zeros() as usize;
                        word &= word - 1;
                        Some(i * 64 + slot)
                    }
                })
            })
    }
    fn rank_bytes(&self) -> usize {
        crate::vec_bytes(&self.words) + crate::vec_bytes(&self.prefix)
    }
    fn bytes(&self) -> usize {
        self.rank_bytes() + crate::vec_bytes(&self.directory) + crate::vec_bytes(&self.blocks)
    }
}

struct Frontier {
    base: Pages,
    sealed: Pages,
    edits: Pages,
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
            edits: Pages::new(g.nodes.len().div_ceil(1024), 1024),
            base_bits: g.block_capacity().trailing_zeros(),
            sealed_bits: sealed_capacity.trailing_zeros(),
        }
    }
    fn mark(&mut self, cursor: u64) {
        if cursor & EDIT != 0 {
            let index = (cursor & !EDIT) as usize;
            self.edits.mark(index / 1024, index % 1024);
        } else if cursor & SEALED != 0 {
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
    fn index(&self, cursor: u64) -> usize {
        if cursor & EDIT != 0 {
            let index = (cursor & !EDIT) as usize;
            self.base.total() + self.sealed.total() + self.edits.index(index / 1024, index % 1024)
        } else if cursor & SEALED != 0 {
            let address = cursor & !SEALED;
            self.base.total()
                + self.sealed.index(
                    (address >> self.sealed_bits) as usize,
                    (address & ((1 << self.sealed_bits) - 1)) as usize,
                )
        } else {
            self.base.index(
                (cursor >> self.base_bits) as usize,
                (cursor & ((1 << self.base_bits) - 1)) as usize / 4,
            )
        }
    }
    fn finish(&mut self) {
        self.base.finish();
        self.sealed.finish();
        self.edits.finish();
    }
    fn clear(&mut self) {
        self.base.clear();
        self.sealed.clear();
        self.edits.clear();
    }
    fn total(&self) -> usize {
        self.base.total() + self.sealed.total() + self.edits.total()
    }
    fn bytes(&self) -> usize {
        self.base.bytes() + self.sealed.bytes() + self.edits.bytes()
    }
    fn rank_bytes(&self) -> usize {
        self.base.rank_bytes() + self.sealed.rank_bytes() + self.edits.rank_bytes()
    }
    fn growths(&self) -> usize {
        self.base.growths + self.sealed.growths + self.edits.growths
    }
    fn recycle_prefixes_into(&mut self, next: &mut Self) {
        // The preceding output depth has been fixed up. Only the newly
        // discovered frontier needs rank now; move capacity, never copy it.
        for (old, new) in [
            (&mut self.base, &mut next.base),
            (&mut self.sealed, &mut next.sealed),
            (&mut self.edits, &mut next.edits),
        ] {
            debug_assert_eq!(new.prefix.capacity(), 0);
            std::mem::swap(&mut old.prefix, &mut new.prefix);
        }
    }
}

struct Locations {
    map: Relocations,
    growths: usize,
    peak_bytes: usize,
}
impl Locations {
    fn new() -> Self {
        Self {
            map: Relocations::Small(Vec::new()),
            growths: 0,
            peak_bytes: 0,
        }
    }
    fn resize(&mut self, n: usize) {
        let before = self.map.bytes();
        match &mut self.map {
            Relocations::Small(v) => {
                if n > v.capacity() {
                    v.reserve_exact(n - v.len());
                }
                v.resize(n, 0);
            }
            Relocations::Large(v) => {
                if n > v.capacity() {
                    v.reserve_exact(n - v.len());
                }
                v.resize(n, 0);
            }
        }
        self.growths += usize::from(before != self.map.bytes());
        self.peak_bytes = self.peak_bytes.max(self.map.bytes());
    }
    fn set(&mut self, index: usize, address: u64) {
        if address > u32::MAX as u64
            && let Relocations::Small(v) = &self.map
        {
            let wide: Vec<u64> = v.iter().map(|&n| n as u64).collect();
            self.peak_bytes = self
                .peak_bytes
                .max(self.map.bytes() + crate::vec_bytes(&wide));
            self.map = Relocations::Large(wide);
            self.growths += 1;
        }
        self.map.set(index, address);
    }
}

struct Writer<S: ReclaimingBlockStorage> {
    storage: S,
    capacity: usize,
    codec: Codec,
    locations: Locations,
    pending_escapes: Vec<u64>,
    stats: BlockCompactionStats,
    source_bytes: usize,
    fixed_leaf: bool,
}
impl<S: ReclaimingBlockStorage> Writer<S> {
    fn emit(
        &mut self,
        cursor: u64,
        word: u64,
        children: impl Iterator<Item = u64>,
        current: &Frontier,
        next: &mut Frontier,
    ) -> u64 {
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
        if !self.fixed_leaf {
            self.locations.set(current.index(cursor), address);
        }
        self.stats.output_record_bytes += length;
        self.stats.peak_uncompressed_block_bytes = self
            .stats
            .peak_uncompressed_block_bytes
            .max(self.source_bytes + self.stats.output_record_bytes);
        address
    }
    fn fixup(
        &mut self,
        blocks: std::ops::Range<usize>,
        escapes: &[u64],
        current: &Frontier,
        scratch: &mut Vec<u8>,
        leaf_start: Option<u64>,
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
                        let cursor = escapes[escaped];
                        escaped += 1;
                        cursor
                    } else {
                        self.codec.cursor(token as u64)
                    };
                    let index = current.index(child);
                    let target = if let Some(start) = leaf_start {
                        start + index as u64 * 8
                    } else {
                        self.locations.map.get(index)
                    };
                    let delta = target
                        .checked_sub(parent)
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|&n| n != 0)
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
        locations: Locations::new(),
        pending_escapes: Vec::new(),
        source_bytes,
        fixed_leaf: false,
        stats: BlockCompactionStats {
            input_uncompressed_block_bytes: source_bytes,
            peak_uncompressed_block_bytes: source_bytes,
            ..stats
        },
    };
    let mut scratch = Vec::new();
    let mut previous = 0..0;
    let mut previous_escapes = Vec::new();
    let mut base_retired = 0;
    let mut counts = Vec::with_capacity(height);
    for depth in 0..height {
        let start = Instant::now();
        let nodes = current.total();
        debug_assert_ne!(nodes, 0);
        counts.push(nodes);
        writer.stats.reachable_states += nodes;
        let leaf = depth + 1 == height;
        writer.fixed_leaf = leaf;
        if !leaf {
            writer.stats.peak_relocation_states = writer.stats.peak_relocation_states.max(nodes);
            writer.locations.resize(nodes);
        }
        if !writer.storage.start_new_append_block(capacity) {
            return Err(TreeError::InvalidConfig(
                "depth compaction requires explicit append-block boundaries".into(),
            ));
        }
        let mut first = None;
        let mut last = 0;
        for &block in &current.base.blocks {
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
                let address = writer.emit(cursor, w, children, &current, &mut next);
                first.get_or_insert((address >> current.base_bits) as usize);
                last = (address >> current.base_bits) as usize + 1;
            }
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
                    let cursor = SEALED | ((block as u64) << current.sealed_bits) | offset as u64;
                    let w = word(&scratch, offset + 1);
                    let n = if leaf { 0 } else { w.count_ones() as usize };
                    let children = (0..n).map(|i| word(&scratch, offset + 9 + i * 8));
                    let address = writer.emit(cursor, w, children, &current, &mut next);
                    first.get_or_insert((address >> current.base_bits) as usize);
                    last = (address >> current.base_bits) as usize + 1;
                }
            }
            for &block in &retire[depth] {
                writer.source_bytes -= sealed.storage.block_len(block);
                sealed.storage.release_block_payload(block);
                writer.stats.released_blocks += 1;
            }
        }
        for &block in &current.edits.blocks {
            for slot in current.edits.slots(block) {
                let index = block * 1024 + slot;
                let node = graph.nodes[index].take().expect("live edit frontier");
                debug_assert_eq!(node.level, depth);
                let address = writer.emit(
                    EDIT | index as u64,
                    node.signature.word,
                    node.signature.children.iter().copied(),
                    &current,
                    &mut next,
                );
                first.get_or_insert((address >> current.base_bits) as usize);
                last = (address >> current.base_bits) as usize + 1;
            }
        }
        let fixup = Instant::now();
        let leaf_start = if leaf {
            Some((first.unwrap() as u64) << current.base_bits)
        } else {
            None
        };
        writer.fixup(
            previous,
            &previous_escapes,
            &current,
            &mut scratch,
            leaf_start,
        )?;
        writer.stats.fixup_ns += fixup.elapsed().as_nanos() as u64;
        writer.stats.copy_ns += start.elapsed().as_nanos() as u64;
        let plan = Instant::now();
        current.recycle_prefixes_into(&mut next);
        next.finish();
        writer.stats.planning_ns += plan.elapsed().as_nanos() as u64;
        writer.stats.rank_bytes = writer
            .stats
            .rank_bytes
            .max(current.rank_bytes() + next.rank_bytes());
        writer.stats.frontier_bytes = writer
            .stats
            .frontier_bytes
            .max(current.bytes() + next.bytes());
        writer.stats.peak_frontier_states =
            writer.stats.peak_frontier_states.max(nodes + next.total());
        writer.stats.relocation_bytes = writer.locations.peak_bytes;
        writer.stats.scratch_bytes = writer.stats.scratch_bytes.max(crate::vec_bytes(&scratch));
        let auxiliary = crate::vec_bytes(&sealed_max)
            + crate::vec_bytes(&retire)
            + retire.iter().map(crate::vec_bytes).sum::<usize>()
            + crate::vec_bytes(&counts)
            + crate::vec_bytes(&previous_escapes)
            + crate::vec_bytes(&writer.pending_escapes);
        writer.stats.workspace_bytes = writer
            .stats
            .workspace_bytes
            .max(current.bytes() + next.bytes() + writer.locations.peak_bytes + auxiliary);
        previous = first.expect("nonempty live depth")..last;
        previous_escapes.clear();
        std::mem::swap(&mut previous_escapes, &mut writer.pending_escapes);
        std::mem::swap(&mut current, &mut next);
        next.clear();
    }
    debug_assert_eq!(current.total(), 0);
    debug_assert!(previous_escapes.is_empty()); // leaf depth has no references
    writer.stats.frontier_buffer_growths = current.growths() + next.growths();
    writer.stats.relocation_buffer_growths = writer.locations.growths;
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
    // The fixed call sites perform one rank per internal state and per edge.
    // Derive the diagnostic outside hot loops to avoid biasing comparisons.
    writer.stats.rank_lookups = writer.stats.reachable_transitions + writer.stats.reachable_states
        - counts.last().copied().unwrap_or(0);
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
    #[test]
    fn empty_full_width_and_full_fanout_records() {
        for domains in [[1, 1, 1], [1 << 32, 1, 1 << 32]] {
            let graph = crate::Fst::build(Vec::new(), domains, 512).into_mutable();
            let (packed, stats) = graph.compact_depths().unwrap();
            assert_eq!(stats.released_blocks, stats.source_blocks);
            assert_eq!(packed.node_count(), 1);
            let triple = ((domains[0] - 1) as u32, 0, (domains[2] - 1) as u32);
            let mut graph = packed.into_mutable();
            graph.insert(triple.0, triple.1, triple.2).unwrap();
            let (packed, stats) = graph.compact_depths().unwrap();
            assert_eq!(stats.released_blocks, stats.source_blocks);
            assert_eq!(packed.scan([None; 3]).collect::<Vec<_>>(), vec![triple]);
            let packed = crate::Fst::deserialize(&packed.serialize()).unwrap();
            let mut graph = packed.into_mutable();
            graph.remove(triple.0, triple.1, triple.2).unwrap();
            assert_eq!(graph.compact_depths().unwrap().0.node_count(), 1);
        }
        let triples: Vec<_> = (0..4)
            .flat_map(|s| (0..4).flat_map(move |p| (0..4).map(move |o| (s * 4, p * 4, o * 4))))
            .collect();
        let mut graph = crate::Fst::build(triples, [16; 3], 512).into_mutable();
        graph.insert(1, 1, 1).unwrap();
        let frozen = graph.freeze_in_place().unwrap();
        let expected = frozen.scan([None; 3]).collect::<Vec<_>>();
        let (packed, stats) = frozen.compact_depths().unwrap();
        assert_eq!(stats.released_blocks, stats.source_blocks);
        assert_eq!(packed.scan([None; 3]).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn escape_references_are_fixed_up_in_edge_order() {
        let mut current = Frontier {
            base: Pages::new(1, 128),
            sealed: Pages::new(1, 1024),
            edits: Pages::new(1, 1024),
            base_bits: 9,
            sealed_bits: 10,
        };
        current.mark(0);
        current.finish();
        let mut next = Frontier {
            base: Pages::new(1, 128),
            sealed: Pages::new(1, 1024),
            edits: Pages::new(1, 1024),
            base_bits: 9,
            sealed_bits: 10,
        };
        let children = [SEALED | 12, EDIT | 3, SEALED | 12];
        let mut writer = Writer {
            storage: crate::storage::HeapStorage::empty(),
            capacity: 512,
            codec: Codec {
                base_slots: 1 << 40,
                sealed_slots: 1024,
            },
            locations: Locations::new(),
            pending_escapes: Vec::new(),
            stats: BlockCompactionStats::default(),
            source_bytes: 0,
            fixed_leaf: false,
        };
        writer.locations.resize(1);
        writer.emit(0, 7, children.into_iter(), &current, &mut next);
        next.finish();
        writer.locations.resize(next.total());
        writer.locations.set(next.index(children[0]), 512);
        writer.locations.set(next.index(children[1]), 520);
        let escapes = std::mem::take(&mut writer.pending_escapes);
        writer
            .fixup(0..1, &escapes, &next, &mut Vec::new(), None)
            .unwrap();
        assert_eq!(writer.storage.read_var(0, 8, 4), 512);
        assert_eq!(writer.storage.read_var(0, 12, 4), 520);
        assert_eq!(writer.storage.read_var(0, 16, 4), 512);
    }

    #[test]
    fn reordered_pages_reuse_storage_and_wide_tokens() {
        let mut pages = Pages::new(20, 1024);
        for (block, slot) in [(19, 63), (2, 64), (19, 64), (0, 1023), (2, 64)] {
            pages.mark(block, slot);
        }
        pages.finish();
        assert_eq!(pages.blocks, [0, 2, 19]);
        assert_eq!(pages.total(), 4);
        assert_eq!(pages.index(0, 1023), 0);
        assert_eq!(pages.index(2, 64), 1);
        assert_eq!(pages.index(19, 63), 2);
        assert_eq!(pages.index(19, 64), 3);
        let indices: std::collections::BTreeSet<_> = [(19, 63), (2, 64), (19, 64), (0, 1023)]
            .into_iter()
            .map(|(b, s)| pages.index(b, s))
            .collect();
        assert_eq!(indices, (0..4).collect());
        let bytes = pages.bytes();
        let growths = pages.growths;
        pages.clear();
        for block in [0, 2, 19] {
            pages.mark(block, 64);
        }
        pages.finish();
        assert_eq!(pages.bytes(), bytes);
        assert_eq!(pages.growths, growths);
        assert_eq!(pages.total(), 3);
        let codec = Codec {
            base_slots: u32::MAX as u64,
            sealed_slots: 1 << 40,
        };
        let mut escapes = Vec::new();
        for cursor in [0, 256, SEALED | 1234, EDIT | 37] {
            let token = codec.encode(cursor, &mut escapes);
            if token != u32::MAX {
                assert_eq!(codec.cursor(token as u64), cursor);
            } else {
                assert_eq!(*escapes.last().unwrap(), cursor);
            }
        }
        let mut locations = Locations::new();
        locations.resize(3);
        locations.set(2, 19);
        locations.set(1, 1 << 40);
        assert_eq!(locations.map.get(2), 19);
        assert_eq!(locations.map.get(1), 1 << 40);
    }
}
