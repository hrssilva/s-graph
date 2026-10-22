//! Consuming, in-memory block compaction. No signatures or graph copies are
//! materialized: mark live record starts, plan relocations, copy and retire.
use super::*;
use crate::storage::ReclaimingBlockStorage;

#[path = "fst_recursive_compact.rs"]
pub(super) mod recursive;

#[path = "fst_depth_compact.rs"]
pub(super) mod depths;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_rank_boundaries_and_wide_relocations() {
        let positions = [0, 63, 64, 1023, 1024, 1039, 2048, 4095, 8192];
        let mut bits = RankBits::new(8193);
        for position in positions {
            assert!(bits.mark(position));
            assert!(!bits.mark(position));
        }
        bits.finish();
        for slot in 0..=8193 {
            assert_eq!(bits.rank(slot), positions.partition_point(|&v| v < slot));
        }
        // Empty bitmaps and exact word ends use the terminal prefix without
        // reading a word past the allocation.
        for slots in [0, 1, 63, 64, 65, 1023, 1024, 1025] {
            let mut bits = RankBits::new(slots);
            for slot in 0..slots {
                bits.mark(slot);
            }
            bits.finish();
            assert_eq!(bits.total(), slots);
            for slot in 0..=slots {
                assert_eq!(bits.rank(slot), slot);
            }
        }
        assert_eq!(
            bits.ones(60, 1030).collect::<Vec<_>>(),
            vec![63, 64, 1023, 1024]
        );
        assert_eq!(bits.ones(69, 69).count(), 0);
        let mut wide = Relocations::new(2, false);
        wide.set(1, 1 << 40);
        assert_eq!(wide.get(1), 1 << 40);
    }
    #[test]
    fn block_release_across_large_sealed_segments() {
        let mut peaks = Vec::new();
        for method in [0, 1, 2, 3] {
            for order in [FreezeOrder::Dfs, FreezeOrder::Levels] {
                let mut graph = crate::Fst::build(Vec::new(), [4096, 4, 4096], 512).into_mutable();
                for s in 0u32..4096 {
                    for p in 0..2 {
                        let mut x = s ^ (p * 0x1234);
                        x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
                        x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
                        x ^= x >> 16;
                        graph.insert(s, p, x % 4096).unwrap();
                    }
                }
                let frozen = graph.freeze_in_place_with_order(order).unwrap();
                if order == FreezeOrder::Levels {
                    let sealed = frozen.graph.sealed.as_ref().unwrap();
                    let mut previous_level = 0;
                    for block in 0..sealed.storage.block_count() {
                        let level = sealed.storage.read_var(block, 0, 1);
                        assert!(level >= previous_level);
                        previous_level = level;
                        let mut offset = 0;
                        while offset < sealed.storage.block_len(block) {
                            assert_eq!(sealed.storage.read_var(block, offset, 1), level);
                            let word = sealed.storage.read_u64(block, offset + 1);
                            offset += 9 + if level as usize + 1 == frozen.height() {
                                0
                            } else {
                                8 * word.count_ones() as usize
                            };
                        }
                    }
                }
                let expected = frozen.to_fst::<crate::storage::HeapStorage>().unwrap();
                let (packed, stats) = match method {
                    0 => frozen.compact_blocks().unwrap(),
                    1 => frozen.compact_recursive().unwrap(),
                    2 => frozen.compact_depths().unwrap(),
                    _ => frozen.compact_streaming().unwrap(),
                };
                assert_eq!(
                    packed.scan([None; 3]).collect::<Vec<_>>(),
                    expected.scan([None; 3]).collect::<Vec<_>>()
                );
                assert_eq!(packed.visited_count(), expected.visited_count());
                assert_eq!(stats.released_blocks, stats.source_blocks);
                peaks.push(stats.peak_uncompressed_block_bytes);
                eprintln!(
                    "method={method} {order:?} sealed segment block-swap accounting: {stats:?}"
                );
            }
        }
        assert!(peaks[1] < peaks[0], "BFS should retire blocks sooner");
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlockCompactionStats {
    pub input_states: usize,
    pub reachable_states: usize,
    pub reachable_transitions: usize,
    pub source_blocks: usize,
    pub released_blocks: usize,
    pub relocation_bytes: usize,
    pub rank_bytes: usize,
    /// Explicit temporary heap allocations, excluding output block buffers,
    /// backend caches,
    /// allocator bookkeeping, recursion stack and output block descriptors.
    pub workspace_bytes: usize,
    pub input_uncompressed_block_bytes: usize,
    /// Used source-block bytes plus written output bytes; excludes arena,
    /// capacities and compressed-vs-uncompressed differences. Not process RSS.
    pub peak_uncompressed_block_bytes: usize,
    pub output_record_bytes: usize,
    /// Phase timings for local comparisons; excludes final graph construction.
    pub mark_ns: u64,
    pub planning_ns: u64,
    pub copy_ns: u64,
    /// Depth compactor diagnostics (zero for older algorithms).
    pub frontier_bytes: usize,
    pub scratch_bytes: usize,
    pub peak_frontier_states: usize,
    pub peak_relocation_states: usize,
    pub source_block_reads: usize,
    pub output_fixup_blocks: usize,
    pub frontier_buffer_growths: usize,
    pub relocation_buffer_growths: usize,
    pub fixup_ns: u64,
    /// Rank lookup counts for the depth/streaming comparison.
    pub rank_lookups: usize,
    /// Maximum retained block relocation runs (streaming compactor).
    pub relocation_runs: usize,
    /// Live edit states forwarded using the existing arena (streaming).
    pub forwarded_edit_states: usize,
}

struct RankBits {
    slots: usize,
    words: Vec<u64>,
    prefix: Vec<usize>, // one prefix per 64-bit word, plus the total
}
impl RankBits {
    fn new(slots: usize) -> Self {
        Self {
            slots,
            words: vec![0; slots.div_ceil(64)],
            prefix: Vec::new(),
        }
    }
    fn mark(&mut self, slot: usize) -> bool {
        let word = &mut self.words[slot / 64];
        let mask = 1 << (slot % 64);
        let fresh = *word & mask == 0;
        *word |= mask;
        fresh
    }
    fn finish(&mut self) {
        self.prefix = Vec::with_capacity(self.words.len() + 1);
        let mut count = 0;
        for word in &self.words {
            self.prefix.push(count);
            count += word.count_ones() as usize;
        }
        self.prefix.push(count);
    }
    fn total(&self) -> usize {
        *self.prefix.last().unwrap()
    }
    fn rank(&self, slot: usize) -> usize {
        let word = slot / 64;
        let mut n = self.prefix[word];
        if !slot.is_multiple_of(64) {
            n += (self.words[word] & ((1 << (slot % 64)) - 1)).count_ones() as usize;
        }
        n
    }
    fn ones(&self, start: usize, end: usize) -> impl Iterator<Item = usize> + '_ {
        self.words[start / 64..end.div_ceil(64)]
            .iter()
            .enumerate()
            .flat_map(move |(i, &w)| {
                let index = start / 64 + i;
                let mut word = w;
                if index == start / 64 {
                    word &= u64::MAX << (start % 64);
                }
                if index == end / 64 && !end.is_multiple_of(64) {
                    word &= (1 << (end % 64)) - 1;
                }
                std::iter::from_fn(move || {
                    if word == 0 {
                        None
                    } else {
                        let bit = word.trailing_zeros() as usize;
                        word &= word - 1;
                        Some(index * 64 + bit)
                    }
                })
            })
    }
    fn bytes(&self) -> usize {
        crate::vec_bytes(&self.words) + crate::vec_bytes(&self.prefix)
    }
}
struct Marks {
    base: RankBits,
    sealed: RankBits,
    edits: RankBits,
}
impl Marks {
    fn mark(&mut self, cursor: u64) -> bool {
        if cursor & EDIT != 0 {
            self.edits.mark((cursor & !EDIT) as usize)
        } else if cursor & SEALED != 0 {
            self.sealed.mark((cursor & !SEALED) as usize)
        } else {
            self.base.mark(cursor as usize / 4)
        }
    }
    fn index(&self, cursor: u64) -> usize {
        if cursor & EDIT != 0 {
            self.base.total() + self.sealed.total() + self.edits.rank((cursor & !EDIT) as usize)
        } else if cursor & SEALED != 0 {
            self.base.total() + self.sealed.rank((cursor & !SEALED) as usize)
        } else {
            self.base.rank(cursor as usize / 4)
        }
    }
    fn total(&self) -> usize {
        self.base.total() + self.sealed.total() + self.edits.total()
    }
    fn bytes(&self) -> usize {
        self.base.bytes() + self.sealed.bytes() + self.edits.bytes()
    }
}
enum Relocations {
    Small(Vec<u32>),
    Large(Vec<u64>),
}
impl Relocations {
    fn new(n: usize, small: bool) -> Self {
        if small {
            Self::Small(vec![0; n])
        } else {
            Self::Large(vec![0; n])
        }
    }
    fn get(&self, index: usize) -> u64 {
        match self {
            Self::Small(v) => v[index] as u64,
            Self::Large(v) => v[index],
        }
    }
    fn set(&mut self, index: usize, address: u64) {
        match self {
            Self::Small(v) => v[index] = u32::try_from(address).unwrap(),
            Self::Large(v) => v[index] = address,
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Self::Small(v) => crate::vec_bytes(v),
            Self::Large(v) => crate::vec_bytes(v),
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Level {
    start: u64,
    end: u64,
    nodes: usize,
}

fn mark<S: BlockStorage>(
    graph: &MutableFst<S>,
    cursor: u64,
    level: usize,
    marks: &mut Marks,
    ranges: &mut [Option<(usize, usize)>],
    edges: &mut usize,
) {
    if !marks.mark(cursor) {
        return;
    }
    if cursor & (EDIT | SEALED) == 0 {
        let slot = cursor as usize / 4;
        ranges[level] = Some(match ranges[level] {
            Some((lo, hi)) => (lo.min(slot), hi.max(slot + 1)),
            None => (slot, slot + 1),
        });
    }
    if level + 1 == graph.height() {
        return;
    }
    let source = graph.source();
    let word = source.word(cursor);
    let mut todo = word;
    while todo != 0 {
        let slot = todo.trailing_zeros();
        todo &= todo - 1;
        *edges += 1;
        mark(
            graph,
            source.follow(cursor, word, slot),
            level + 1,
            marks,
            ranges,
            edges,
        );
    }
}

// Original records are already level ordered. Sealed records can be DFS
// ordered: before copying, use their level byte; while copying use the planned
// destination interval so an already retired block is never inspected again.
fn level<S: BlockStorage>(
    graph: &mut MutableFst<S>,
    marks: &Marks,
    ranges: &[Option<(usize, usize)>],
    n: usize,
    planned: Option<(&Relocations, &[Level])>,
    mut emit: impl FnMut(&mut MutableFst<S>, u64) -> Result<()>,
) -> Result<()> {
    if let Some((lo, hi)) = ranges[n] {
        for slot in marks.base.ones(lo, hi) {
            emit(graph, (slot * 4) as u64)?;
        }
    }
    for slot in marks.sealed.ones(0, marks.sealed.slots) {
        let cursor = SEALED | slot as u64;
        let matches = if let Some((map, levels)) = planned {
            let address = map.get(marks.index(cursor));
            address >= levels[n].start && address < levels[n].end
        } else {
            let sealed = graph.sealed.as_ref().unwrap();
            let (block, offset) = sealed.position(cursor);
            sealed.storage.read_var(block, offset, 1) as usize == n
        };
        if matches {
            emit(graph, cursor)?;
        }
    }
    for slot in marks.edits.ones(0, marks.edits.slots) {
        let cursor = EDIT | slot as u64;
        let matches = if let Some((map, levels)) = planned {
            let address = map.get(marks.index(cursor));
            address >= levels[n].start && address < levels[n].end
        } else {
            graph.nodes[slot].as_ref().unwrap().level == n
        };
        if matches {
            emit(graph, cursor)?;
        }
    }
    Ok(())
}
fn position(block: &mut u64, offset: &mut usize, length: usize, capacity: usize) -> u64 {
    if *offset != 0 && *offset + length > capacity {
        *block += 1;
        *offset = 0;
    }
    let address = (*block << capacity.trailing_zeros()) | *offset as u64;
    *offset += length;
    address
}
fn remaining(bits: &RankBits, blocks: usize, slots_per_block: usize) -> Vec<usize> {
    (0..blocks)
        .map(|block| bits.rank((block + 1) * slots_per_block) - bits.rank(block * slots_per_block))
        .collect()
}

pub(super) fn run<S: ReclaimingBlockStorage>(
    mut graph: MutableFst<S>,
) -> Result<(PackedFst<S>, BlockCompactionStats)> {
    let domains = graph.dimension_sizes();
    let capacity = graph.block_capacity();
    let height = graph.height();
    let mut stats = BlockCompactionStats {
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
    let visits = u64::try_from(graph.logical_visits)
        .map_err(|_| TreeError::InvalidConfig("FST logical visit count exceeds u64".into()))?;
    graph.register = Vec::new();
    graph.memo = Vec::new();
    graph.base_register = None;
    graph.sampled_register = None;
    graph.free = Vec::new();
    if graph.source().word(graph.root) == 0 {
        drop(graph);
        stats.reachable_states = 1;
        stats.released_blocks = stats.source_blocks;
        return Ok((PackedFst::try_build(Vec::new(), domains, capacity)?, stats));
    }
    let phase = std::time::Instant::now();
    let mut marks = Marks {
        base: RankBits::new(graph.base.block_count() * (capacity / 4)),
        sealed: RankBits::new(
            graph
                .sealed
                .as_ref()
                .map_or(0, |s| s.storage.block_count() * s.capacity),
        ),
        edits: RankBits::new(graph.nodes.len()),
    };
    let mut ranges = vec![None; height];
    {
        let source = graph.source();
        let _guard = source.begin_read();
        mark(
            &graph,
            graph.root,
            0,
            &mut marks,
            &mut ranges,
            &mut stats.reachable_transitions,
        );
    }
    marks.base.finish();
    marks.sealed.finish();
    marks.edits.finish();
    stats.reachable_states = marks.total();
    stats.rank_bytes = marks.bytes();
    stats.mark_ns = phase.elapsed().as_nanos() as u64;
    let phase = std::time::Instant::now();
    let mut levels = vec![Level::default(); height];
    let (mut block, mut offset) = (0, 0);
    for (n, layout) in levels.iter_mut().enumerate() {
        layout.start = (block << capacity.trailing_zeros()) + offset as u64;
        level(&mut graph, &marks, &ranges, n, None, |g, cursor| {
            let children = if n + 1 == height {
                0
            } else {
                g.source().word(cursor).count_ones() as usize
            };
            position(&mut block, &mut offset, 8 + children * 4, capacity);
            layout.nodes += 1;
            Ok(())
        })?;
        layout.end = (block << capacity.trailing_zeros()) + offset as u64;
    }
    let mut map = Relocations::new(marks.total(), levels[height - 1].end <= u32::MAX as u64);
    (block, offset) = (0, 0);
    for n in 0..height {
        level(&mut graph, &marks, &ranges, n, None, |g, cursor| {
            let children = if n + 1 == height {
                0
            } else {
                g.source().word(cursor).count_ones() as usize
            };
            let address = position(&mut block, &mut offset, 8 + children * 4, capacity);
            map.set(marks.index(cursor), address);
            Ok(())
        })?;
    }
    stats.relocation_bytes = map.bytes();
    stats.planning_ns = phase.elapsed().as_nanos() as u64;
    let phase = std::time::Instant::now();
    let mut base_left = remaining(&marks.base, graph.base.block_count(), capacity / 4);
    let mut sealed_left = graph.sealed.as_ref().map_or_else(Vec::new, |s| {
        remaining(&marks.sealed, s.storage.block_count(), s.capacity)
    });
    stats.workspace_bytes = marks.bytes()
        + map.bytes()
        + crate::vec_bytes(&ranges)
        + crate::vec_bytes(&levels)
        + crate::vec_bytes(&base_left)
        + crate::vec_bytes(&sealed_left);
    let mut source_bytes: usize = (0..graph.base.block_count())
        .map(|b| graph.base.block_len(b))
        .sum();
    if let Some(s) = &graph.sealed {
        source_bytes += (0..s.storage.block_count())
            .map(|b| s.storage.block_len(b))
            .sum::<usize>();
    }
    stats.input_uncompressed_block_bytes = source_bytes;
    stats.peak_uncompressed_block_bytes = source_bytes;
    for (b, &left) in base_left.iter().enumerate() {
        if left == 0 {
            source_bytes -= graph.base.block_len(b);
            graph.base.release_compaction_block(b);
            stats.released_blocks += 1;
        }
    }
    if let Some(s) = &mut graph.sealed {
        for (b, &left) in sealed_left.iter().enumerate() {
            if left == 0 {
                source_bytes -= s.storage.block_len(b);
                s.storage.release_block_payload(b);
                stats.released_blocks += 1;
            }
        }
    }
    let mut storage = S::empty();
    for n in 0..height {
        level(
            &mut graph,
            &marks,
            &ranges,
            n,
            Some((&map, &levels)),
            |g, cursor| {
                let source = g.source();
                let word = source.word(cursor);
                let mut record = [0u8; 8 + 64 * 4];
                record[..8].copy_from_slice(&word.to_le_bytes());
                let address = map.get(marks.index(cursor));
                let mut count = 0;
                if n + 1 < height {
                    let mut todo = word;
                    while todo != 0 {
                        let slot = todo.trailing_zeros();
                        todo &= todo - 1;
                        let child = source.follow(cursor, word, slot);
                        let target = map.get(marks.index(child));
                        let delta = target
                            .checked_sub(address)
                            .and_then(|v| u32::try_from(v).ok())
                            .ok_or_else(|| {
                                TreeError::InvalidConfig("FST child delta exceeds 32 bits".into())
                            })?;
                        record[8 + count * 4..12 + count * 4].copy_from_slice(&delta.to_le_bytes());
                        count += 1;
                    }
                }
                let length = 8 + count * 4;
                let pos = storage.append_record(&record[..length], capacity);
                debug_assert_eq!(
                    ((pos.block as u64) << capacity.trailing_zeros()) | pos.offset as u64,
                    address
                );
                stats.output_record_bytes += length;
                stats.peak_uncompressed_block_bytes = stats
                    .peak_uncompressed_block_bytes
                    .max(source_bytes + stats.output_record_bytes);
                if cursor & EDIT != 0 {
                    g.nodes[(cursor & !EDIT) as usize] = None;
                } else if cursor & SEALED != 0 {
                    let s = g.sealed.as_mut().unwrap();
                    let (b, _) = s.position(cursor);
                    sealed_left[b] -= 1;
                    if sealed_left[b] == 0 {
                        source_bytes -= s.storage.block_len(b);
                        s.storage.release_block_payload(b);
                        stats.released_blocks += 1;
                    }
                } else {
                    let b = (cursor >> capacity.trailing_zeros()) as usize;
                    base_left[b] -= 1;
                    if base_left[b] == 0 {
                        source_bytes -= g.base.block_len(b);
                        g.base.release_compaction_block(b);
                        stats.released_blocks += 1;
                    }
                }
                Ok(())
            },
        )?;
    }
    debug_assert_eq!(stats.released_blocks, stats.source_blocks);
    stats.copy_ns = phase.elapsed().as_nanos() as u64;
    storage.finish_append(capacity);
    drop(graph);
    let counts = levels.into_iter().map(|level| level.nodes).collect();
    Ok((
        PackedFst::from_compacted_storage(storage, domains, capacity, counts, visits),
        stats,
    ))
}
