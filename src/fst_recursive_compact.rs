//! Reserve BFS block numbers on descent, fill their payload on return.
use super::*;
use crate::storage::ReservedBlockStorage;
use std::time::Instant;

#[derive(Clone, Copy)]
struct BlockLevels {
    lo: u8,
    hi: u8,
}
impl Default for BlockLevels {
    fn default() -> Self {
        Self { lo: u8::MAX, hi: 0 }
    }
}
impl BlockLevels {
    fn include(&mut self, level: usize) {
        self.lo = self.lo.min(level as u8);
        self.hi = self.hi.max(level as u8);
    }
}

fn mark_live<S: BlockStorage>(
    graph: &MutableFst<S>,
    cursor: u64,
    level: usize,
    marks: &mut Marks,
    ranges: &mut [Option<(usize, usize)>],
    sealed_levels: &mut [BlockLevels],
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
    } else if cursor & EDIT == 0 {
        let (block, _) = graph.sealed.as_ref().unwrap().position(cursor);
        sealed_levels[block].include(level);
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
        mark_live(
            graph,
            source.follow(cursor, word, slot),
            level + 1,
            marks,
            ranges,
            sealed_levels,
            edges,
        );
    }
}

// No level lookup may touch a previously retired block or dropped arena node.
fn records<S: BlockStorage>(
    graph: &mut MutableFst<S>,
    marks: &Marks,
    ranges: &[Option<(usize, usize)>],
    sealed_levels: &[BlockLevels],
    n: usize,
    mut emit: impl FnMut(&mut MutableFst<S>, u64) -> Result<()>,
) -> Result<()> {
    if let Some((lo, hi)) = ranges[n] {
        for slot in marks.base.ones(lo, hi) {
            emit(graph, (slot * 4) as u64)?;
        }
    }
    if let Some(sealed) = &graph.sealed {
        let capacity = sealed.capacity;
        for (block, levels) in sealed_levels.iter().enumerate() {
            if n < levels.lo as usize || n > levels.hi as usize {
                continue;
            }
            if graph.sealed.as_ref().unwrap().storage.block_len(block) == 0 {
                continue;
            }
            let start = block * capacity;
            for slot in marks.sealed.ones(start, start + capacity) {
                let matches = levels.lo == levels.hi
                    || graph
                        .sealed
                        .as_ref()
                        .unwrap()
                        .storage
                        .read_var(block, slot - start, 1) as usize
                        == n;
                if matches {
                    emit(graph, SEALED | slot as u64)?;
                    // Mixed-level blocks can be retired before this bitmap's
                    // end: remaining marked starts may belong to deeper levels
                    // already copied on earlier recursive returns.
                    if levels.lo != levels.hi
                        && graph.sealed.as_ref().unwrap().storage.block_len(block) == 0
                    {
                        break;
                    }
                }
            }
        }
    }
    for slot in marks.edits.ones(0, marks.edits.slots) {
        if graph.nodes[slot]
            .as_ref()
            .is_some_and(|node| node.level == n)
        {
            emit(graph, EDIT | slot as u64)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Layout {
    first: usize,
    blocks: usize,
    nodes: usize,
    end: u64,
}

struct Context<S: ReservedBlockStorage> {
    graph: MutableFst<S>,
    storage: S,
    marks: Marks,
    ranges: Vec<Option<(usize, usize)>>,
    sealed_levels: Vec<BlockLevels>,
    layouts: Vec<Layout>,
    base_left: Vec<usize>,
    sealed_left: Vec<usize>,
    map: Option<Relocations>,
    buffer: Vec<u8>,
    source_bytes: usize,
    stats: BlockCompactionStats,
}

fn compact_level<S: ReservedBlockStorage>(ctx: &mut Context<S>, n: usize) -> Result<()> {
    let capacity = ctx.graph.block_capacity();
    let height = ctx.graph.height();
    let bits = capacity.trailing_zeros();
    let start = Instant::now();
    let (mut block, mut offset, mut nodes) = (0, 0, 0);
    records(
        &mut ctx.graph,
        &ctx.marks,
        &ctx.ranges,
        &ctx.sealed_levels,
        n,
        |g, cursor| {
            let children = if n + 1 == height {
                0
            } else {
                g.source().word(cursor).count_ones() as usize
            };
            position(&mut block, &mut offset, 8 + children * 4, capacity);
            nodes += 1;
            Ok(())
        },
    )?;
    let blocks = if nodes == 0 { 0 } else { block as usize + 1 };
    let first = ctx.storage.reserve_compaction_blocks(blocks, capacity);
    ctx.layouts[n] = Layout {
        first,
        blocks,
        nodes,
        end: ((first as u64 + block) << bits) + offset as u64,
    };
    ctx.stats.planning_ns += start.elapsed().as_nanos() as u64;

    if n + 1 < height {
        compact_level(ctx, n + 1)?;
    } else {
        let start = Instant::now();
        ctx.map = Some(Relocations::new(
            ctx.marks.total(),
            ctx.layouts[n].end <= u32::MAX as u64,
        ));
        ctx.stats.relocation_bytes = ctx.map.as_ref().unwrap().bytes();
        ctx.stats.planning_ns += start.elapsed().as_nanos() as u64;
    }

    let start = Instant::now();
    let Layout { first, blocks, .. } = ctx.layouts[n];
    let Context {
        graph,
        storage,
        marks,
        ranges,
        sealed_levels,
        base_left,
        sealed_left,
        map,
        buffer,
        source_bytes,
        stats,
        ..
    } = ctx;
    let map = map.as_mut().unwrap();
    let mut output_block = first;
    records(graph, marks, ranges, sealed_levels, n, |g, cursor| {
        let source = g.source();
        let word = source.word(cursor);
        let count = if n + 1 == height {
            0
        } else {
            word.count_ones() as usize
        };
        let length = 8 + 4 * count;
        if !buffer.is_empty() && buffer.len() + length > capacity {
            storage.install_compaction_block(output_block, buffer, capacity);
            output_block += 1;
        }
        let address = ((output_block as u64) << bits) | buffer.len() as u64;
        let mut record = [0u8; 8 + 64 * 4];
        record[..8].copy_from_slice(&word.to_le_bytes());
        if count != 0 {
            let mut todo = word;
            for i in 0..count {
                let slot = todo.trailing_zeros();
                todo &= todo - 1;
                let child = source.follow(cursor, word, slot);
                let delta = map
                    .get(marks.index(child))
                    .checked_sub(address)
                    .and_then(|v| u32::try_from(v).ok())
                    .filter(|&v| v != 0)
                    .ok_or_else(|| {
                        TreeError::InvalidConfig("FST child delta exceeds 32 bits".into())
                    })?;
                record[8 + i * 4..12 + i * 4].copy_from_slice(&delta.to_le_bytes());
            }
        }
        if buffer.capacity() == 0 {
            buffer.reserve_exact(capacity);
        }
        buffer.extend_from_slice(&record[..length]);
        map.set(marks.index(cursor), address);
        stats.output_record_bytes += length;
        stats.peak_uncompressed_block_bytes = stats
            .peak_uncompressed_block_bytes
            .max(*source_bytes + stats.output_record_bytes);
        if cursor & EDIT != 0 {
            g.nodes[(cursor & !EDIT) as usize] = None;
        } else if cursor & SEALED != 0 {
            let sealed = g.sealed.as_mut().unwrap();
            let (b, _) = sealed.position(cursor);
            sealed_left[b] -= 1;
            if sealed_left[b] == 0 {
                *source_bytes -= sealed.storage.block_len(b);
                sealed.storage.release_block_payload(b);
                stats.released_blocks += 1;
            }
        } else {
            let b = (cursor >> bits) as usize;
            base_left[b] -= 1;
            if base_left[b] == 0 {
                *source_bytes -= g.base.block_len(b);
                g.base.release_compaction_block(b);
                stats.released_blocks += 1;
            }
        }
        Ok(())
    })?;
    if !buffer.is_empty() {
        storage.install_compaction_block(output_block, buffer, capacity);
        output_block += 1;
    }
    debug_assert_eq!(output_block, first + blocks);
    stats.copy_ns += start.elapsed().as_nanos() as u64;
    Ok(())
}

pub(in crate::mutable_fst) fn run<S: ReservedBlockStorage>(
    mut graph: MutableFst<S>,
) -> Result<(PackedFst<S>, BlockCompactionStats)> {
    let domains = graph.dimension_sizes();
    let capacity = graph.block_capacity();
    let height = graph.height();
    let visits = u64::try_from(graph.logical_visits)
        .map_err(|_| TreeError::InvalidConfig("FST logical visit count exceeds u64".into()))?;
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
    let start = Instant::now();
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
    let mut sealed_levels =
        vec![BlockLevels::default(); graph.sealed.as_ref().map_or(0, |s| s.storage.block_count())];
    {
        let source = graph.source();
        let _guard = source.begin_read();
        mark_live(
            &graph,
            graph.root,
            0,
            &mut marks,
            &mut ranges,
            &mut sealed_levels,
            &mut stats.reachable_transitions,
        );
    }
    marks.base.finish();
    marks.sealed.finish();
    marks.edits.finish();
    stats.reachable_states = marks.total();
    stats.rank_bytes = marks.bytes();
    let base_left = remaining(&marks.base, graph.base.block_count(), capacity / 4);
    let sealed_left = graph.sealed.as_ref().map_or_else(Vec::new, |s| {
        remaining(&marks.sealed, s.storage.block_count(), s.capacity)
    });
    let mut source_bytes = 0;
    for (b, &left) in base_left.iter().enumerate() {
        let len = graph.base.block_len(b);
        stats.input_uncompressed_block_bytes += len;
        if left == 0 {
            graph.base.release_compaction_block(b);
            stats.released_blocks += 1;
        } else {
            source_bytes += len;
        }
    }
    if let Some(sealed) = &mut graph.sealed {
        for (b, &left) in sealed_left.iter().enumerate() {
            let len = sealed.storage.block_len(b);
            stats.input_uncompressed_block_bytes += len;
            if left == 0 {
                sealed.storage.release_block_payload(b);
                stats.released_blocks += 1;
            } else {
                source_bytes += len;
            }
        }
    }
    stats.peak_uncompressed_block_bytes = stats.input_uncompressed_block_bytes;
    stats.mark_ns = start.elapsed().as_nanos() as u64;
    let mut ctx = Context {
        graph,
        storage: S::empty(),
        marks,
        ranges,
        sealed_levels,
        layouts: vec![Layout::default(); height],
        base_left,
        sealed_left,
        map: None,
        buffer: Vec::new(),
        source_bytes,
        stats,
    };
    compact_level(&mut ctx, 0)?;
    ctx.stats.workspace_bytes = ctx.marks.bytes()
        + ctx.map.as_ref().unwrap().bytes()
        + crate::vec_bytes(&ctx.ranges)
        + crate::vec_bytes(&ctx.layouts)
        + crate::vec_bytes(&ctx.sealed_levels)
        + crate::vec_bytes(&ctx.base_left)
        + crate::vec_bytes(&ctx.sealed_left);
    debug_assert_eq!(ctx.stats.released_blocks, ctx.stats.source_blocks);
    ctx.storage.finish_append(capacity);
    let counts = ctx.layouts.iter().map(|l| l.nodes).collect();
    let Context { storage, stats, .. } = ctx;
    Ok((
        PackedFst::from_compacted_storage(storage, domains, capacity, counts, visits),
        stats,
    ))
}
