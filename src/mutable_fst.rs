//! Local updates over a packed, suffix-sharing FST. Edits copy one root-to-leaf
//! path; the packed base is never rewritten. Compact freeze repacks physical
//! states; experimental in-place freeze seals edited states in blocked storage.
use crate::automata_core::{BitmapSource, TripleSource};
use crate::path_fst::{BfsNodeMeta, PackedFst};
use crate::{BlockStorage, OutOfDomain, Result, TreeError, Triple};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[path = "fst_block_compact.rs"]
mod block_compact;
pub use block_compact::BlockCompactionStats;

#[path = "fst_stream_compact.rs"]
mod streaming;

const EDIT: u64 = 1 << 63;
const SEALED: u64 = 1 << 62;
const EMPTY: u64 = u64::MAX;

struct ReadSessions<A, B> {
    _base: A,
    _sealed: Option<B>,
}
impl<A, B> Drop for ReadSessions<A, B> {
    fn drop(&mut self) {}
}

// Append-only blocked records: level:u8, word:u64, absolute children:u64[].
// Tagged absolute cursors link to either base without relocating old records.
struct SealedStorage<S: BlockStorage> {
    storage: S,
    capacity: usize,
    per_level: Vec<usize>,
}
impl<S: BlockStorage> SealedStorage<S> {
    fn new(capacity: usize, height: usize) -> Self {
        Self {
            storage: S::empty(),
            capacity: capacity.max(1024),
            per_level: vec![0; height],
        }
    }
    fn position(&self, cursor: u64) -> (usize, usize) {
        let address = cursor & !SEALED;
        (
            (address >> self.capacity.trailing_zeros()) as usize,
            (address & (self.capacity as u64 - 1)) as usize,
        )
    }
    fn word(&self, cursor: u64) -> u64 {
        let (block, offset) = self.position(cursor);
        self.storage.read_u64(block, offset + 1)
    }
    fn child(&self, cursor: u64, index: usize) -> u64 {
        let (block, offset) = self.position(cursor);
        self.storage.read_var(block, offset + 9 + index * 8, 8)
    }
    fn append(&mut self, level: usize, word: u64, children: &[u64]) -> Result<u64> {
        // Reserve the two high cursor bits, including a possible next block.
        if self.storage.block_count() as u64 >= (SEALED >> self.capacity.trailing_zeros()) - 1 {
            return Err(TreeError::InvalidConfig(
                "sealed FST address space exhausted".into(),
            ));
        }
        // A 64-child absolute-link record needs 521 bytes; sealed blocks use
        // at least 1024 even when the original packed block budget is 512.
        let mut record = [0u8; 9 + 64 * 8];
        record[0] = level as u8;
        record[1..9].copy_from_slice(&word.to_le_bytes());
        for (i, child) in children.iter().enumerate() {
            record[9 + i * 8..17 + i * 8].copy_from_slice(&child.to_le_bytes());
        }
        let pos = self
            .storage
            .append_record(&record[..9 + children.len() * 8], self.capacity);
        self.per_level[level] += 1;
        Ok(SEALED | ((pos.block as u64) << self.capacity.trailing_zeros()) | pos.offset as u64)
    }
    fn visit(&self, mut emit: impl FnMut(usize, u64, u64) -> bool) {
        for block in 0..self.storage.block_count() {
            let mut offset = 0;
            while offset < self.storage.block_len(block) {
                let level = self.storage.read_var(block, offset, 1) as usize;
                let word = self.storage.read_u64(block, offset + 1);
                let cursor =
                    SEALED | ((block as u64) << self.capacity.trailing_zeros()) | offset as u64;
                if !emit(level, cursor, word) {
                    return;
                }
                offset += 9 + if level + 1 == self.per_level.len() {
                    0
                } else {
                    word.count_ones() as usize * 8
                };
            }
        }
    }
    fn mem_size_bytes(&self) -> usize {
        self.storage.mem_size_bytes() + crate::vec_bytes(&self.per_level)
    }
}

/// Experimental time/space alternatives for exact suffix canonicalization.
/// Scanning preserves minimality but trades retained indexes for repeated work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpdateLookup {
    #[default]
    Indexed,
    /// Scan packed records; retain the edited-state lookup table.
    ScanBase,
    /// Scan packed records and live edit slots; retain neither lookup index.
    /// Reference counts and slot-reuse bookkeeping are still maintained.
    ScanAll,
    /// Memoize exact packed-state lookups (including absences) across edits.
    /// Remember packed states on edited paths; scan only uncached signatures.
    /// The edited-state lookup table is retained. Compaction clears the memo.
    Memoized,
}

fn fingerprint(word: u64, children: impl ExactSizeIterator<Item = u64>) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    word.hash(&mut hash);
    children.len().hash(&mut hash);
    for child in children {
        child.hash(&mut hash);
    }
    hash.finish()
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Signature {
    word: u64,
    children: Vec<u64>,
}
struct Node {
    signature: Arc<Signature>,
    references: usize,
    level: usize,
}

/// Graph-owned retained RAM, excluding allocator bookkeeping, process RSS and
/// temporary query/compaction workspaces. Vector capacities are counted;
/// HashMap bucket overhead is estimated, as in `mem_size_bytes`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Original packed backend plus any blocked records sealed by prior batches.
    pub packed_base_bytes: usize,
    /// Fingerprint/cursor pairs and per-level vector headers for packed states.
    pub base_register_bytes: usize,
    /// Cross-edit memo tables, including retained signature child vectors.
    pub memo_register_bytes: usize,
    /// Live edited occupancy words, child vectors and shared signature headers.
    pub edit_payload_bytes: usize,
    /// State slots, including reference counts, level tags and unused capacity.
    pub edit_arena_bytes: usize,
    /// Exact-signature lookup tables used to merge equivalent edited states.
    pub edit_register_bytes: usize,
    pub free_list_bytes: usize,
    pub live_edit_states: usize,
    pub arena_slots: usize,
    pub free_slots: usize,
}
impl MemoryUsage {
    pub fn total_ram_bytes(&self) -> usize {
        self.packed_base_bytes
            + self.base_register_bytes
            + self.memo_register_bytes
            + self.edit_payload_bytes
            + self.edit_arena_bytes
            + self.edit_register_bytes
            + self.free_list_bytes
    }
}

/// Cumulative packed lookup work since construction or the last compaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateLookupStats {
    pub base_scans: u64,
    pub memo_hits: u64,
    pub memo_entries: usize,
    pub cached_absences: usize,
}

/// Physical ordering of records newly written by `freeze_in_place`.
/// Existing packed and previously sealed records keep their addresses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FreezeOrder {
    /// Child-first DFS postorder, retained as an experimental comparison.
    Dfs,
    /// Parent-first BFS levels, each starting in a fresh block.
    #[default]
    Levels,
}

/// Immutable prototype retaining original packed blocks and blocked edit records.
/// Update tables and the RAM edit arena are released. Obsolete stored records
/// remain until explicit compaction. Methods shared with the editable engine
/// are available through an immutable dereference; mutation requires consuming
/// `into_mutable` first.
pub struct FrozenFst<S: BlockStorage = crate::storage::HeapStorage> {
    graph: MutableFst<S>,
}
impl<S: BlockStorage> std::ops::Deref for FrozenFst<S> {
    type Target = MutableFst<S>;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}
impl<S: BlockStorage> FrozenFst<S> {
    /// Resume local edits without copying or scanning retained storage.
    pub fn into_mutable(self) -> MutableFst<S> {
        self.into_mutable_with_lookup(UpdateLookup::Indexed)
    }
    pub fn into_mutable_with_lookup(self, lookup: UpdateLookup) -> MutableFst<S> {
        let mut graph = self.graph;
        graph.lookup = lookup;
        graph.register = if lookup == UpdateLookup::ScanAll {
            Vec::new()
        } else {
            (0..graph.height()).map(|_| HashMap::new()).collect()
        };
        graph.memo = if lookup == UpdateLookup::Memoized {
            (0..graph.height()).map(|_| HashMap::new()).collect()
        } else {
            Vec::new()
        };
        graph
    }
    /// Stored sealed states, including obsolete records from previous batches.
    pub fn sealed_node_count(&self) -> usize {
        self.graph
            .sealed
            .as_ref()
            .map_or(0, |s| s.per_level.iter().sum())
    }
    /// Reclaim obsolete records and produce ordinary SGF1-compatible storage.
    pub fn compact(self) -> Result<PackedFst<S>>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        self.graph.freeze()
    }
    pub fn compact_blocks(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        self.graph.compact_blocks()
    }
    pub fn compact_recursive(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReservedBlockStorage,
    {
        self.graph.compact_recursive()
    }
    pub fn compact_depths(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        self.graph.compact_depths()
    }
    pub fn compact_streaming(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        self.graph.compact_streaming()
    }
}

impl<S: BlockStorage> crate::join::PatternBackend for FrozenFst<S> {
    type Scan<'a>
        = crate::path_ops::PatternIter<'a, MutableFstSource<'a, S>>
    where
        S: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        self.graph.scan(bounds)
    }
}
impl<S: BlockStorage> crate::path::AdjacencyBackend for FrozenFst<S> {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        crate::path::AdjacencyBackend::out_neighbors(&self.graph, node, predicate)
    }
    fn num_nodes(&self) -> usize {
        crate::path::AdjacencyBackend::num_nodes(&self.graph)
    }
}

/// Editable FST with set semantics and fixed coordinate domains. Unchanged
/// suffixes remain in the packed base; edited suffixes are interned per level.
/// The edit arena is RAM-resident for all storage modes. Removed edit states
/// are reclaimed immediately and their slots reused. Base bytes are reclaimed
/// by `compact`/`freeze`, never by rewriting the base on each update.
pub struct MutableFst<S: BlockStorage = crate::storage::HeapStorage> {
    base: PackedFst<S>,
    sealed: Option<SealedStorage<S>>,
    root: u64,
    logical_visits: u128,
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    register: Vec<HashMap<Arc<Signature>, usize>>,
    base_register: Option<Vec<Vec<(u64, u64)>>>,
    // Keys contain only packed child identities. Packed bytes and addresses
    // remain immutable until compaction, so both hits and absences stay valid.
    memo: Vec<HashMap<Signature, Option<u64>>>,
    base_scans: u64,
    memo_hits: u64,
    lookup: UpdateLookup,
    live_nodes: usize,
}

/// Borrowed graph language over both packed and edited FST states.
pub struct MutableFstSource<'a, S: BlockStorage = crate::storage::HeapStorage>(
    pub &'a MutableFst<S>,
);
impl<S: BlockStorage> Copy for MutableFstSource<'_, S> {}
impl<S: BlockStorage> Clone for MutableFstSource<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: BlockStorage> BitmapSource for MutableFstSource<'_, S> {
    fn begin_read(&self) -> impl Drop + '_ {
        ReadSessions {
            _base: self.0.base.read_session(),
            _sealed: self
                .0
                .sealed
                .as_ref()
                .map(|s| s.storage.begin_read_session()),
        }
    }

    type Cursor = u64;
    fn root(&self) -> Option<u64> {
        Some(self.0.root)
    }
    fn height(&self) -> usize {
        self.0.base.height()
    }
    #[inline]
    fn word(&self, cursor: u64) -> u64 {
        if cursor == EMPTY {
            0
        } else if cursor & EDIT == 0 {
            self.0.base_word(cursor)
        } else {
            self.0.node(cursor).signature.word
        }
    }
    #[inline]
    fn follow(&self, cursor: u64, word: u64, slot: u32) -> u64 {
        let index = (word & ((1u64 << slot) - 1)).count_ones() as usize;
        if cursor & EDIT == 0 {
            self.0.base_child(cursor, index)
        } else {
            self.0.node(cursor).signature.children[index]
        }
    }
}
impl<S: BlockStorage> TripleSource for MutableFstSource<'_, S> {
    fn domains(&self) -> [u64; 3] {
        self.0.dimension_sizes()
    }
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        self.0.base.level_schedule(level)
    }
}

impl<S: BlockStorage> PackedFst<S> {
    /// Enable local updates without copying packed graph bytes or compiling a
    /// graph-sized directory. Read-only FST searches retain their original path.
    pub fn into_mutable(self) -> MutableFst<S> {
        self.into_mutable_with_lookup(UpdateLookup::Indexed)
    }
    /// Choose an experimental suffix lookup strategy without changing the
    /// represented language, update semantics or reclamation algorithm.
    pub fn into_mutable_with_lookup(self, lookup: UpdateLookup) -> MutableFst<S> {
        MutableFst {
            root: self.root_addr(),
            logical_visits: self.visited_count() as u128,
            sealed: None,
            register: if lookup == UpdateLookup::ScanAll {
                Vec::new()
            } else {
                (0..self.height()).map(|_| HashMap::new()).collect()
            },
            base_register: None,
            memo: if lookup == UpdateLookup::Memoized {
                (0..self.height()).map(|_| HashMap::new()).collect()
            } else {
                Vec::new()
            },
            base_scans: 0,
            memo_hits: 0,
            lookup,
            base: self,
            nodes: Vec::new(),
            free: Vec::new(),
            live_nodes: 0,
        }
    }
}

impl<S: BlockStorage> MutableFst<S> {
    fn base_word(&self, cursor: u64) -> u64 {
        if cursor & SEALED != 0 {
            self.sealed.as_ref().unwrap().word(cursor)
        } else {
            self.base.read_word(cursor)
        }
    }
    fn base_child(&self, cursor: u64, index: usize) -> u64 {
        if cursor & SEALED != 0 {
            self.sealed.as_ref().unwrap().child(cursor, index)
        } else {
            cursor + self.base.read_delta(cursor, index)
        }
    }
    fn for_each_base_state(&self, mut emit: impl FnMut(usize, u64, u64)) {
        self.base.for_each_physical_state(&mut emit);
        if let Some(sealed) = &self.sealed {
            sealed.visit(|level, cursor, word| {
                emit(level, cursor, word);
                true
            });
        }
    }
    pub fn try_build(
        triples: Vec<Triple>,
        domains: [u64; 3],
        block_capacity: usize,
    ) -> Result<Self> {
        Ok(PackedFst::<S>::try_build(triples, domains, block_capacity)?.into_mutable())
    }
    pub fn build(triples: Vec<Triple>, domains: [u64; 3], block_capacity: usize) -> Self {
        Self::try_build(triples, domains, block_capacity).expect("invalid FST build")
    }
    pub fn build_dynamic(triples: Vec<Triple>, block_capacity: usize) -> Self {
        PackedFst::<S>::build_dynamic(triples, block_capacity).into_mutable()
    }
    pub fn try_build_dynamic(triples: Vec<Triple>, block_capacity: usize) -> Result<Self> {
        Ok(PackedFst::<S>::try_build_dynamic(triples, block_capacity)?.into_mutable())
    }
    pub fn dimension_sizes(&self) -> [u64; 3] {
        self.base.dimension_sizes()
    }
    pub fn domain_size(&self, dimension: usize) -> u64 {
        self.base.domain_size(dimension)
    }
    pub fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        self.base.in_bounds(s, p, o)
    }
    pub fn height(&self) -> usize {
        self.base.height()
    }
    pub fn block_capacity(&self) -> usize {
        self.base.block_capacity()
    }
    /// Reachable edited physical states, excluding the retained packed base.
    pub fn edited_node_count(&self) -> usize {
        self.live_nodes
    }
    /// Retained graph/arena allocation estimate; HashMap bucket overhead is approximate.
    pub fn mem_size_bytes(&self) -> usize {
        self.memory_usage().total_ram_bytes()
    }
    /// Diagnostic work counters and current cross-edit cache population.
    pub fn update_lookup_stats(&self) -> UpdateLookupStats {
        UpdateLookupStats {
            base_scans: self.base_scans,
            memo_hits: self.memo_hits,
            memo_entries: self.memo.iter().map(HashMap::len).sum(),
            cached_absences: self
                .memo
                .iter()
                .flat_map(HashMap::values)
                .filter(|cursor| cursor.is_none())
                .count(),
        }
    }
    /// Separate packed graph data, live edit payload and online-maintenance
    /// allocations. No query execution or graph mutation is performed.
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            packed_base_bytes: self.base.mem_size_bytes()
                + self
                    .sealed
                    .as_ref()
                    .map_or(0, SealedStorage::mem_size_bytes),
            base_register_bytes: self.base_register.as_ref().map_or(0, |levels| {
                crate::vec_bytes(levels) + levels.iter().map(crate::vec_bytes).sum::<usize>()
            }),
            memo_register_bytes: crate::vec_bytes(&self.memo)
                + self
                    .memo
                    .iter()
                    .map(|r| {
                        r.capacity() * (std::mem::size_of::<(Signature, Option<u64>)>() + 1)
                            + r.keys()
                                .map(|s| crate::vec_bytes(&s.children))
                                .sum::<usize>()
                    })
                    .sum::<usize>(),
            edit_arena_bytes: crate::vec_bytes(&self.nodes),
            free_list_bytes: crate::vec_bytes(&self.free),
            edit_register_bytes: crate::vec_bytes(&self.register)
                + self
                    .register
                    .iter()
                    .map(|r| r.capacity() * (std::mem::size_of::<(Arc<Signature>, usize)>() + 1))
                    .sum::<usize>(),
            edit_payload_bytes: self
                .nodes
                .iter()
                .flatten()
                .map(|n| {
                    std::mem::size_of::<Signature>()
                        + 2 * std::mem::size_of::<usize>()
                        + crate::vec_bytes(&n.signature.children)
                })
                .sum::<usize>(),
            live_edit_states: self.live_nodes,
            arena_slots: self.nodes.len(),
            free_slots: self.free.len(),
        }
    }
    pub fn source(&self) -> MutableFstSource<'_, S> {
        MutableFstSource(self)
    }
    pub fn search<'a, A: crate::automata::Compile<MutableFstSource<'a, S>>>(
        &'a self,
        machine: A,
    ) -> Result<A::Compiled> {
        machine.compile(self.source())
    }
    pub fn join<P: Copy + Into<crate::graph::Term>>(
        &self,
        query: &crate::graph::JoinQuery<P>,
    ) -> Result<crate::scheduled_join::ScheduledJoin<MutableFstSource<'_, S>>> {
        crate::scheduled_join::ScheduledJoin::with_source(
            query,
            self.source(),
            crate::scheduled_join::Policy::Adaptive,
        )
    }
    pub fn bgp(
        &self,
        query: &crate::graph::JoinQuery<crate::graph::Term>,
    ) -> Result<crate::query_ops::Bgp<MutableFstSource<'_, S>>> {
        crate::query_ops::Bgp::new(
            query,
            self.source(),
            crate::scheduled_join::Policy::Adaptive,
        )
    }
    pub fn scan(
        &self,
        bounds: [Option<u32>; 3],
    ) -> crate::path_ops::PatternIter<'_, MutableFstSource<'_, S>> {
        crate::path_ops::PatternIter::new(self.source(), &self.base.scan_layout, bounds)
    }
    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        if !self.in_bounds(s, p, o) {
            return false;
        }
        let source = self.source();
        let mut cursor = self.root;
        for level in 0..source.height() {
            let word = source.word(cursor);
            let (bits, shifts, remaining) = source.layout(level);
            let slot = [s, p, o].into_iter().enumerate().fold(0, |slot, (d, v)| {
                slot | if bits[d] == 0 {
                    0
                } else {
                    ((v >> remaining[d]) & ((1u32 << bits[d]) - 1)) << shifts[d]
                }
            });
            if word & (1u64 << slot) == 0 {
                return false;
            }
            if level + 1 < source.height() {
                cursor = source.follow(cursor, word, slot);
            }
        }
        true
    }
    pub fn contains_batch(&self, triples: &[Triple]) -> Vec<bool> {
        triples
            .iter()
            .map(|&(s, p, o)| self.contains(s, p, o))
            .collect()
    }
    pub fn contains_batch_into(&self, triples: &[Triple], out: &mut Vec<bool>) {
        out.clear();
        out.extend(triples.iter().map(|&(s, p, o)| self.contains(s, p, o)));
    }
    /// Counts by physical state and mask, retaining a query-local memo table.
    pub fn count_matching(&self, bounds: [Option<u32>; 3]) -> u64 {
        let scan = self.scan(bounds);
        if scan.done {
            return 0;
        }
        fn count<S: BlockStorage>(
            source: MutableFstSource<'_, S>,
            cursor: u64,
            level: usize,
            masks: &[u64],
            memo: &mut HashMap<u64, u64>,
        ) -> u64 {
            if let Some(&n) = memo.get(&cursor) {
                return n;
            }
            let word = source.word(cursor);
            let mut todo = word & masks[level];
            let mut n = 0u64;
            if level + 1 == source.height() {
                n = todo.count_ones() as u64;
            } else {
                while todo != 0 {
                    let slot = todo.trailing_zeros();
                    todo &= todo - 1;
                    n = n
                        .checked_add(count(
                            source,
                            source.follow(cursor, word, slot),
                            level + 1,
                            masks,
                            memo,
                        ))
                        .expect("FST match count exceeds u64");
                }
            }
            memo.insert(cursor, n);
            n
        }
        count(
            self.source(),
            self.root,
            0,
            &scan.masks,
            &mut HashMap::new(),
        )
    }
    fn node(&self, cursor: u64) -> &Node {
        self.nodes[(cursor & !EDIT) as usize]
            .as_ref()
            .expect("live edit cursor")
    }
    fn acquire(&mut self, cursor: u64) {
        if cursor != EMPTY && cursor & EDIT != 0 {
            self.nodes[(cursor & !EDIT) as usize]
                .as_mut()
                .unwrap()
                .references += 1;
        }
    }
    fn release(&mut self, cursor: u64) {
        if cursor == EMPTY || cursor & EDIT == 0 {
            return;
        }
        let index = (cursor & !EDIT) as usize;
        let node = self.nodes[index].as_mut().unwrap();
        node.references -= 1;
        if node.references != 0 {
            return;
        }
        let node = self.nodes[index].take().unwrap();
        if self.lookup != UpdateLookup::ScanAll {
            self.register[node.level].remove(&node.signature);
        }
        self.free.push(index);
        self.live_nodes -= 1;
        for &child in &node.signature.children {
            self.release(child);
        }
    }
    fn intern(&mut self, level: usize, word: u64, children: Vec<u64>) -> u64 {
        if word == 0 {
            return EMPTY;
        }
        // Bottom-up canonical identities make base comparison possible without
        // expanding right languages. A base state cannot have an edited child.
        let signature = Signature { word, children };
        if signature.children.iter().all(|c| c & EDIT == 0)
            && let Some(cursor) = self.find_base(level, &signature)
        {
            return cursor;
        }
        let existing = if self.lookup == UpdateLookup::ScanAll {
            self.nodes.iter().position(|n| {
                n.as_ref()
                    .is_some_and(|n| n.level == level && *n.signature == signature)
            })
        } else {
            self.register[level].get(&signature).copied()
        };
        if let Some(index) = existing {
            return EDIT | index as u64;
        }
        for &child in &signature.children {
            self.acquire(child);
        }
        let index = self.free.pop().unwrap_or_else(|| {
            self.nodes.push(None);
            self.nodes.len() - 1
        });
        let signature = Arc::new(signature);
        if self.lookup != UpdateLookup::ScanAll {
            self.register[level].insert(signature.clone(), index);
        }
        self.nodes[index] = Some(Node {
            signature,
            references: 0,
            level,
        });
        self.live_nodes += 1;
        EDIT | index as u64
    }
    /// Build the exact base-state register before a write burst. Otherwise it
    /// is built lazily on the first effective edit (no-op writes do not pay).
    /// The scan-based and memoized strategies skip this preparation.
    /// Retains 16 bytes per packed state plus O(height) vector metadata; scans
    /// physical records, not triples. Hash collisions are checked structurally.
    pub fn prepare_updates(&mut self) {
        if self.lookup != UpdateLookup::Indexed || self.base_register.is_some() {
            return;
        }
        let mut levels: Vec<Vec<(u64, u64)>> = self
            .base
            .physical_states_per_level()
            .iter()
            .enumerate()
            .map(|(level, &n)| {
                Vec::with_capacity(n + self.sealed.as_ref().map_or(0, |s| s.per_level[level]))
            })
            .collect();
        self.for_each_base_state(|level, cursor, word| {
            if word == 0 {
                return;
            }
            let count = if level + 1 == self.height() {
                0
            } else {
                word.count_ones() as usize
            };
            let hash = fingerprint(word, (0..count).map(|i| self.base_child(cursor, i)));
            levels[level].push((hash, cursor));
        });
        for states in &mut levels {
            states.sort_unstable();
        }
        self.base_register = Some(levels);
    }
    fn find_base(&mut self, level: usize, signature: &Signature) -> Option<u64> {
        let word = signature.word;
        let children = &signature.children;
        if self.lookup == UpdateLookup::Memoized
            && let Some(&cursor) = self.memo[level].get(signature)
        {
            self.memo_hits += 1;
            return cursor;
        }
        if self.lookup != UpdateLookup::Indexed {
            self.base_scans += 1;
            let mut found = self.base.find_physical_state(level, |cursor, stored_word| {
                stored_word == word
                    && children
                        .iter()
                        .enumerate()
                        .all(|(i, &child)| cursor + self.base.read_delta(cursor, i) == child)
            });
            if found.is_none()
                && let Some(sealed) = &self.sealed
            {
                sealed.visit(|stored_level, cursor, stored_word| {
                    if stored_level == level
                        && stored_word == word
                        && children
                            .iter()
                            .enumerate()
                            .all(|(i, &child)| sealed.child(cursor, i) == child)
                    {
                        found = Some(cursor);
                        false
                    } else {
                        true
                    }
                });
            }
            if self.lookup == UpdateLookup::Memoized {
                self.memo[level].insert(signature.clone(), found);
            }
            return found;
        }
        let states = &self
            .base_register
            .as_ref()
            .expect("prepared update register")[level];
        let hash = fingerprint(word, children.iter().copied());
        let start = states.partition_point(|&(h, _)| h < hash);
        states[start..]
            .iter()
            .take_while(|&&(h, _)| h == hash)
            .map(|&(_, cursor)| cursor)
            .find(|&cursor| {
                self.base_word(cursor) == word
                    && children
                        .iter()
                        .enumerate()
                        .all(|(i, &c)| self.base_child(cursor, i) == c)
            })
    }

    fn update(&mut self, coords: [u32; 3], insert: bool) -> bool {
        let mut path = [(EMPTY, 0u64, 0u32); crate::tree::MAX_HEIGHT];
        let mut cursor = self.root;
        for (level, entry) in path.iter_mut().enumerate().take(self.height()) {
            let source = self.source();
            let word = source.word(cursor);
            let (bits, shifts, remaining) = source.layout(level);
            let slot = (0..3).fold(0, |slot, d| {
                slot | if bits[d] == 0 {
                    0
                } else {
                    ((coords[d] >> remaining[d]) & ((1u32 << bits[d]) - 1)) << shifts[d]
                }
            });
            let present = word & (1u64 << slot) != 0;
            if !insert && !present {
                return false;
            }
            *entry = (cursor, word, slot);
            if level + 1 == self.height() {
                if present == insert {
                    return false;
                }
            } else {
                cursor = if present {
                    source.follow(cursor, word, slot)
                } else {
                    EMPTY
                };
            }
        }
        self.prepare_updates();
        let mut child = EMPTY;
        for (level, &(cursor, old_word, slot)) in path[..self.height()].iter().enumerate().rev() {
            let mut word = old_word;
            let mut children = Vec::new();
            if level + 1 < self.height() {
                let source = self.source();
                children.reserve(word.count_ones() as usize + usize::from(insert));
                let mut todo = word;
                while todo != 0 {
                    let q = todo.trailing_zeros();
                    todo &= todo - 1;
                    children.push(source.follow(cursor, word, q));
                }
            }
            if self.lookup == UpdateLookup::Memoized && cursor != EMPTY && cursor & EDIT == 0 {
                // The traversal already identifies this exact packed state.
                // Seed its signature before changing it, making inverse edits
                // cheap even if no lookup previously requested this state.
                let original = Signature { word, children };
                if !self.memo[level].contains_key(&original) {
                    self.memo[level].insert(original.clone(), Some(cursor));
                }
                children = original.children;
            }
            let bit = 1u64 << slot;
            if level + 1 == self.height() {
                if insert {
                    word |= bit;
                } else {
                    word &= !bit;
                }
            } else {
                let index = (word & (bit - 1)).count_ones() as usize;
                if child == EMPTY {
                    children.remove(index);
                    word &= !bit;
                } else if word & bit != 0 {
                    children[index] = child;
                } else {
                    children.insert(index, child);
                    word |= bit;
                }
            }
            child = self.intern(level, word, children);
            if level != 0 {
                if old_word == 0 && word != 0 {
                    self.logical_visits += 1;
                }
                if old_word != 0 && word == 0 {
                    self.logical_visits -= 1;
                }
            }
        }
        let old = self.root;
        self.acquire(child);
        self.root = child;
        self.release(old);
        true
    }
    fn validate_insert(&self, coords: [u32; 3]) -> Result<()> {
        for (dim, value) in coords.into_iter().enumerate() {
            let domain_size = self.domain_size(dim);
            if value as u64 >= domain_size {
                return Err(TreeError::OutOfDomain(OutOfDomain {
                    dim,
                    value,
                    domain_size,
                }));
            }
        }
        Ok(())
    }
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> Result<bool> {
        self.validate_insert([s, p, o])?;
        Ok(self.update([s, p, o], true))
    }
    pub fn remove(&mut self, s: u32, p: u32, o: u32) -> Result<bool> {
        Ok(self.in_bounds(s, p, o) && self.update([s, p, o], false))
    }
    /// Validate the complete batch before editing; duplicates count once.
    pub fn insert_batch(&mut self, triples: &mut [Triple]) -> Result<u64> {
        for &(s, p, o) in triples.iter() {
            self.validate_insert([s, p, o])?;
        }
        triples.sort_unstable();
        Ok(triples
            .iter()
            .map(|&(s, p, o)| u64::from(self.update([s, p, o], true)))
            .sum())
    }
    pub fn remove_batch(&mut self, triples: &mut [Triple]) -> Result<u64> {
        triples.sort_unstable();
        let mut count = 0;
        for &(s, p, o) in triples.iter() {
            count += u64::from(self.remove(s, p, o)?);
        }
        Ok(count)
    }
    /// Borrowing snapshot/backend conversion: rebuild and pack reachable states.
    /// Preserves this graph and its update metadata. Unlike consuming `freeze`,
    /// this path retains the source storage until the output is complete.
    /// Does not enumerate root-to-leaf paths or materialize graph relations.
    pub fn to_fst<T: BlockStorage>(&self) -> Result<PackedFst<T>> {
        pack_source(self.source(), self.block_capacity())
    }
    /// Consume this graph and repack its already minimal live states in BFS order.
    /// Uses the rank-free streaming compactor: reuses edit metadata and retires
    /// source blocks incrementally, without a global liveness/minimization pass.
    /// All built-in storage backends support this operation. For compaction
    /// diagnostics, use [`Self::compact_streaming`].
    pub fn freeze(self) -> Result<PackedFst<S>>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        self.compact_streaming().map(|(packed, _)| packed)
    }
    /// Consume the graph, repack its already minimal live states, and release
    /// source blocks as soon as their last live record has been copied.
    /// Uses RAM rank bitmaps and 32/64-bit relocations; does not spill to files.
    pub fn compact_blocks(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        block_compact::run(self)
    }
    /// Experimental depth-scoped compaction. Discovers liveness while copying
    /// source records, then resolves the preceding output depth block by block.
    /// Reuses frontier, rank, relocation and block-scratch allocations.
    pub fn compact_depths(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        block_compact::depths::run(self)
    }
    /// The default consuming freeze algorithm, with compaction diagnostics.
    /// Root-first BFS compaction without rank directories.
    /// Reuses edit-arena metadata and maps stored addresses with block runs.
    pub fn compact_streaming(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReclaimingBlockStorage,
    {
        streaming::run(self)
    }
    /// Experimental child-first compaction into reserved BFS block slots.
    /// Preserves minimality; retains a dense relocation map for live states.
    pub fn compact_recursive(self) -> Result<(PackedFst<S>, BlockCompactionStats)>
    where
        S: crate::storage::ReservedBlockStorage,
    {
        block_compact::recursive::run(self)
    }
    /// Seal only live edited states into blocked storage. Original storage is
    /// moved intact; no packed-base traversal, minimization or repack occurs.
    /// Retains obsolete base/sealed records until explicit compaction.
    pub fn freeze_in_place(self) -> Result<FrozenFst<S>> {
        self.freeze_in_place_with_order(FreezeOrder::default())
    }
    pub fn freeze_in_place_with_order(mut self, order: FreezeOrder) -> Result<FrozenFst<S>> {
        // Free graph-sized maintenance allocations before allocating relocation
        // workspace. Edit payload is now owned only by arena slots.
        self.register = Vec::new();
        self.memo = Vec::new();
        self.base_register = None;
        self.free = Vec::new();
        if self.live_nodes != 0 {
            if self.sealed.is_none() {
                self.sealed = Some(SealedStorage::new(self.block_capacity(), self.height()));
            }
            let mut locations = vec![EMPTY; self.nodes.len()];
            fn seal<S: BlockStorage>(
                graph: &mut MutableFst<S>,
                cursor: u64,
                locations: &mut [u64],
            ) -> Result<u64> {
                if cursor == EMPTY || cursor & EDIT == 0 {
                    return Ok(cursor);
                }
                let index = (cursor & !EDIT) as usize;
                if locations[index] != EMPTY {
                    return Ok(locations[index]);
                }
                let node = graph.nodes[index].take().expect("live edit state");
                let mut signature =
                    Arc::try_unwrap(node.signature).unwrap_or_else(|arc| (*arc).clone());
                for child in &mut signature.children {
                    *child = seal(graph, *child, locations)?;
                }
                let address = graph.sealed.as_mut().unwrap().append(
                    node.level,
                    signature.word,
                    &signature.children,
                )?;
                locations[index] = address;
                Ok(address)
            }
            if order == FreezeOrder::Levels {
                // Bucket slots in O(arena slots + height), avoiding a sort.
                let mut counts = vec![0; self.height()];
                for node in self.nodes.iter().flatten() {
                    counts[node.level] += 1;
                }
                let mut levels: Vec<Vec<usize>> =
                    counts.into_iter().map(Vec::with_capacity).collect();
                for (i, node) in self.nodes.iter().enumerate() {
                    if let Some(node) = node {
                        levels[node.level].push(i);
                    }
                }
                let sealed = self.sealed.as_mut().unwrap();
                let bits = sealed.capacity.trailing_zeros();
                let mut block = sealed.storage.block_count();
                // Plan forward references before consuming any arena payload.
                // Isolate levels, including the first level of each new batch.
                for slots in levels.iter().filter(|slots| !slots.is_empty()) {
                    let mut offset = 0;
                    for &i in slots {
                        let node = self.nodes[i].as_ref().unwrap();
                        let len = 9 + 8 * node.signature.children.len();
                        if offset + len > sealed.capacity {
                            block += 1;
                            offset = 0;
                        }
                        if block as u64 >= SEALED >> bits {
                            return Err(TreeError::InvalidConfig(
                                "sealed FST address space exhausted".into(),
                            ));
                        }
                        locations[i] = SEALED | ((block as u64) << bits) | offset as u64;
                        offset += len;
                    }
                    block += 1;
                }
                for slots in levels.iter().filter(|slots| !slots.is_empty()) {
                    if !sealed.storage.start_new_append_block(sealed.capacity) {
                        return Err(TreeError::InvalidConfig(
                            "BFS sealing requires explicit append-block boundaries".into(),
                        ));
                    }
                    for &i in slots {
                        let node = self.nodes[i].take().expect("live edit state");
                        let mut signature =
                            Arc::try_unwrap(node.signature).unwrap_or_else(|arc| (*arc).clone());
                        for child in &mut signature.children {
                            if *child != EMPTY && *child & EDIT != 0 {
                                *child = locations[(*child & !EDIT) as usize];
                                debug_assert_ne!(*child, EMPTY);
                            }
                        }
                        let address =
                            sealed.append(node.level, signature.word, &signature.children)?;
                        debug_assert_eq!(address, locations[i]);
                    }
                }
            }
            let root = self.root;
            self.root = seal(&mut self, root, &mut locations)?;
            // Finish only touches the sealed segment. The original base's
            // backend, block list, file and compressed chunks are untouched.
            let sealed = self.sealed.as_mut().unwrap();
            sealed.storage.finish_append(sealed.capacity);
        }
        self.nodes = Vec::new();
        self.live_nodes = 0;
        self.base_scans = 0;
        self.memo_hits = 0;
        Ok(FrozenFst { graph: self })
    }
    /// Reclaim dead base bytes and reset the edit arena; O(physical states and
    /// transitions). Failure leaves the existing graph intact.
    pub fn compact(&mut self) -> Result<()> {
        let packed = self.to_fst::<S>()?;
        *self = packed.into_mutable_with_lookup(self.lookup);
        Ok(())
    }
    /// Serialize a compact physical-state snapshot; no edit history is stored.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.to_fst::<S>()?.serialize())
    }
    pub fn serialize_into<W: std::io::Write>(&self, out: &mut W) -> Result<()> {
        self.to_fst::<S>()?.serialize_into(out)
    }
    pub fn serialize_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.to_fst::<S>()?.serialize_to_path(path)
    }
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize(bytes)?.into_mutable())
    }
    pub fn deserialize_with_limits(
        bytes: &[u8],
        limits: crate::serialize::DeserializeLimits,
    ) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize_with_limits(bytes, limits)?.into_mutable())
    }
    pub fn deserialize_from<R: std::io::Read>(input: &mut R) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize_from(input)?.into_mutable())
    }
    pub fn deserialize_from_with_limits<R: std::io::Read>(
        input: &mut R,
        limits: crate::serialize::DeserializeLimits,
    ) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize_from_with_limits(input, limits)?.into_mutable())
    }
    pub fn deserialize_from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize_from_path(path)?.into_mutable())
    }
    pub fn deserialize_from_path_with_limits(
        path: impl AsRef<std::path::Path>,
        limits: crate::serialize::DeserializeLimits,
    ) -> Result<Self> {
        Ok(PackedFst::<S>::deserialize_from_path_with_limits(path, limits)?.into_mutable())
    }
}

fn pack_source<S: TripleSource<Cursor = u64>, T: BlockStorage>(
    source: S,
    block_capacity: usize,
) -> Result<PackedFst<T>> {
    let height = source.height();
    let mut pending = vec![source.root().unwrap_or(EMPTY)];
    let mut levels = Vec::with_capacity(height);
    for level in 0..height {
        let mut next = Vec::new();
        let mut index = HashMap::new();
        let mut nodes = Vec::with_capacity(pending.len());
        for cursor in pending {
            let word = source.word(cursor);
            let mut children = Vec::new();
            if level + 1 < height {
                let mut todo = word;
                while todo != 0 {
                    let slot = todo.trailing_zeros();
                    todo &= todo - 1;
                    let child = source.follow(cursor, word, slot);
                    let n = u32::try_from(next.len()).map_err(|_| {
                        TreeError::InvalidConfig("FST physical level exceeds 32-bit indices".into())
                    })?;
                    let id = *index.entry(child).or_insert_with(|| {
                        next.push(child);
                        n
                    });
                    children.push(id);
                }
            }
            nodes.push(BfsNodeMeta {
                existence_word: word,
                child_indices: children,
            });
        }
        levels.push(nodes);
        pending = next;
    }
    let mut canonical: Vec<Vec<BfsNodeMeta>> = (0..height).map(|_| Vec::new()).collect();
    let mut child_map = Vec::new();
    let mut child_visits: Vec<u64> = Vec::new();
    for level in (0..height).rev() {
        let mut register = HashMap::new();
        let mut mapping = Vec::with_capacity(levels[level].len());
        let mut visits = Vec::new();
        for node in &levels[level] {
            let children: Vec<u32> = node
                .child_indices
                .iter()
                .map(|&c| child_map[c as usize])
                .collect();
            let mut count = 1u64;
            for &child in &children {
                count = count
                    .checked_add(child_visits[child as usize])
                    .ok_or_else(|| {
                        TreeError::InvalidConfig("FST logical visit count exceeds u64".into())
                    })?;
            }
            let id = u32::try_from(canonical[level].len()).map_err(|_| {
                TreeError::InvalidConfig("FST physical level exceeds 32-bit indices".into())
            })?;
            let id = *register
                .entry((node.existence_word, children.clone()))
                .or_insert_with(|| {
                    canonical[level].push(BfsNodeMeta {
                        existence_word: node.existence_word,
                        child_indices: children,
                    });
                    visits.push(count);
                    id
                });
            mapping.push(id);
        }
        child_map = mapping;
        child_visits = visits;
    }
    PackedFst::pack_nodes(canonical, source.domains(), block_capacity, child_visits[0])
}

impl<S: BlockStorage> crate::join::PatternBackend for MutableFst<S> {
    type Scan<'a>
        = crate::path_ops::PatternIter<'a, MutableFstSource<'a, S>>
    where
        S: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        self.scan(bounds)
    }
}
impl<S: BlockStorage> crate::path::AdjacencyBackend for MutableFst<S> {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        let mut out: Vec<_> = self
            .scan([Some(node), predicate, None])
            .map(|(_, _, o)| o)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
    fn num_nodes(&self) -> usize {
        usize::try_from(self.domain_size(0).max(self.domain_size(2)))
            .expect("dense graph domains require a 64-bit host")
    }
}

#[cfg(feature = "file")]
impl MutableFst<crate::storage_file::FileStorage> {
    /// Packed base bytes on disk; edits remain in the RAM arena until compacted.
    pub fn disk_bytes(&self) -> u64 {
        self.base.disk_bytes() + self.sealed.as_ref().map_or(0, |s| s.storage.file_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_compaction_preserves_shared_suffixes_and_incremental_visit_counts() {
        let mut graph = crate::Fst::build(vec![(0, 0, 1), (8, 0, 1)], [16, 4, 16], 512)
            .into_mutable_with_lookup(UpdateLookup::Memoized);
        assert!(graph.remove(0, 0, 1).unwrap());
        for i in 0..64 {
            let t = (i % 16, 2, (i * 7) % 16);
            graph.insert(t.0, t.1, t.2).unwrap();
            assert_eq!(
                graph.logical_visits,
                graph
                    .to_fst::<crate::storage::HeapStorage>()
                    .unwrap()
                    .visited_count() as u128
            );
            graph.remove(t.0, t.1, t.2).unwrap();
            assert_eq!(
                graph.logical_visits,
                graph
                    .to_fst::<crate::storage::HeapStorage>()
                    .unwrap()
                    .visited_count() as u128
            );
        }
        let expected = graph.to_fst::<crate::storage::HeapStorage>().unwrap();
        let (packed, stats) = graph.compact_blocks().unwrap();
        assert_eq!(packed.scan([None; 3]).collect::<Vec<_>>(), vec![(8, 0, 1)]);
        assert_eq!(packed.visited_count(), expected.visited_count());
        assert_eq!(stats.released_blocks, stats.source_blocks);
        let mut graph = packed.into_mutable();
        graph.remove(8, 0, 1).unwrap();
        let (packed, _) = graph.compact_blocks().unwrap();
        assert_eq!(packed.node_count(), 1);
        assert_eq!(packed.count_matching([None; 3]), 0);
    }
    #[test]
    fn in_place_freeze_handles_empty_singleton_and_full_width_domains() {
        for domains in [[1, 1, 1], [1 << 32, 1, 1 << 32]] {
            let graph = crate::Fst::build(vec![], domains, 512).into_mutable();
            let frozen = graph.freeze_in_place().unwrap();
            assert_eq!(frozen.sealed_node_count(), 0);
            assert_eq!(frozen.count_matching([None; 3]), 0);
            let t = ((domains[0] - 1) as u32, 0, (domains[2] - 1) as u32);
            let mut graph = frozen.into_mutable_with_lookup(UpdateLookup::Memoized);
            assert!(graph.insert(t.0, t.1, t.2).unwrap());
            let frozen = graph.freeze_in_place().unwrap();
            assert_eq!(frozen.scan([None; 3]).collect::<Vec<_>>(), vec![t]);
            let mut graph = frozen.into_mutable_with_lookup(UpdateLookup::Memoized);
            assert!(graph.remove(t.0, t.1, t.2).unwrap());
            let frozen = graph.freeze_in_place().unwrap();
            assert_eq!(frozen.count_matching([None; 3]), 0);
            let compact = frozen.compact().unwrap();
            assert_eq!(compact.node_count(), 1);
        }
    }
    #[test]
    fn in_place_freeze_retains_base_bytes_and_releases_all_update_allocations() {
        let base = crate::Fst::build(
            (0..4096).map(|s| (s, 0, (s * 197) % 4096)).collect(),
            [4096, 4, 4096],
            512,
        );
        let bytes = base.serialize();
        let mut graph = base.into_mutable_with_lookup(UpdateLookup::Memoized);
        graph.insert(9, 2, 13).unwrap();
        let live = graph.live_nodes;
        let frozen = graph.freeze_in_place().unwrap();
        assert_eq!(frozen.graph.base.serialize(), bytes);
        assert_eq!(frozen.sealed_node_count(), live);
        assert!(frozen.graph.nodes.is_empty());
        assert_eq!(
            frozen.memory_usage().packed_base_bytes,
            frozen.mem_size_bytes()
        );
        assert!(frozen.contains(9, 2, 13));
        let mut graph = frozen.into_mutable_with_lookup(UpdateLookup::Memoized);
        graph.remove(9, 2, 13).unwrap();
        assert_eq!(graph.live_nodes, 0);
        let frozen = graph.freeze_in_place().unwrap();
        assert_eq!(frozen.sealed_node_count(), live); // no new records for inverse edit
        assert_eq!(frozen.graph.base.serialize(), bytes);
        assert!(!frozen.contains(9, 2, 13));
    }
    #[test]
    fn memoized_lookups_survive_inverse_edits_and_reset_on_compaction() {
        let initial: Vec<_> = (0..1024).map(|s| (s, 0, (s * 197 + 73) % 1024)).collect();
        let mut graph = crate::Fst::build(initial, [1024, 4, 1024], 512)
            .into_mutable_with_lookup(UpdateLookup::Memoized);
        let original = graph.root;
        let before = graph.mem_size_bytes();
        graph.prepare_updates();
        assert!(!graph.insert(0, 0, 73).unwrap());
        assert!(!graph.remove(0, 3, 13).unwrap());
        assert_eq!(graph.update_lookup_stats(), UpdateLookupStats::default());
        assert_eq!(graph.mem_size_bytes(), before);

        assert!(graph.insert(9, 2, 13).unwrap());
        assert!(graph.remove(9, 2, 13).unwrap());
        let warm = graph.update_lookup_stats();
        assert!(warm.base_scans > 0);
        assert!(warm.cached_absences > 0);
        assert!(warm.memo_entries > warm.cached_absences);
        for _ in 0..100 {
            assert!(graph.insert(9, 2, 13).unwrap());
            assert!(graph.remove(9, 2, 13).unwrap());
            assert_eq!(graph.root, original);
            assert_eq!(graph.edited_node_count(), 0);
        }
        let after = graph.update_lookup_stats();
        assert_eq!(after.base_scans, warm.base_scans);
        assert_eq!(after.memo_entries, warm.memo_entries);
        assert!(after.memo_hits > warm.memo_hits);
        let usage = graph.memory_usage();
        assert_eq!(usage.base_register_bytes, 0);
        assert!(usage.memo_register_bytes > 0);
        assert_eq!(usage.total_ram_bytes(), graph.mem_size_bytes());
        assert!(graph.mem_size_bytes() > before);
        graph.compact().unwrap();
        assert_eq!(graph.update_lookup_stats(), UpdateLookupStats::default());
        assert!(graph.insert(9, 2, 13).unwrap());
        assert!(graph.update_lookup_stats().base_scans > 0);
        assert_eq!(graph.memory_usage().base_register_bytes, 0);
        assert!(graph.remove(9, 2, 13).unwrap());
        assert_eq!(graph.edited_node_count(), 0);
    }

    #[test]
    fn lookup_strategies_preserve_live_minimality_and_inverse_updates() {
        use std::collections::HashSet;
        for lookup in [
            UpdateLookup::Indexed,
            UpdateLookup::ScanBase,
            UpdateLookup::ScanAll,
            UpdateLookup::Memoized,
        ] {
            let initial: Vec<_> = (0..32).map(|s| (s, 0, s ^ 1)).collect();
            let mut graph =
                crate::Fst::build(initial, [64, 4, 64], 512).into_mutable_with_lookup(lookup);
            for i in 0..32 {
                assert!(graph.insert(i, 2, (i * 7) % 64).unwrap());
                assert!(graph.remove(i, 0, i ^ 1).unwrap());
                let source = graph.source();
                let mut pending = HashSet::from([source.root().unwrap()]);
                let mut count = 0;
                for level in 0..source.height() {
                    count += pending.len();
                    let mut next = HashSet::new();
                    for cursor in pending {
                        let word = source.word(cursor);
                        let mut todo = word;
                        if level + 1 < source.height() {
                            while todo != 0 {
                                let slot = todo.trailing_zeros();
                                todo &= todo - 1;
                                next.insert(source.follow(cursor, word, slot));
                            }
                        }
                    }
                    pending = next;
                }
                let packed = graph.to_fst::<crate::storage::HeapStorage>().unwrap();
                assert_eq!(count as u64, packed.node_count());
                assert_eq!(graph.count_matching([None; 3]), 32);
                assert!(graph.remove(i, 2, (i * 7) % 64).unwrap());
                assert!(graph.insert(i, 0, i ^ 1).unwrap());
                assert_eq!(graph.edited_node_count(), 0);
                if lookup != UpdateLookup::Indexed {
                    assert_eq!(graph.memory_usage().base_register_bytes, 0);
                }
                if lookup == UpdateLookup::ScanAll {
                    assert_eq!(graph.memory_usage().edit_register_bytes, 0);
                }
            }
        }
    }
    #[test]
    fn packed_register_collisions_require_exact_structural_equality() {
        let mut graph = crate::Fst::build(
            vec![(0, 0, 0), (0, 0, 1), (8, 0, 0), (8, 0, 2)],
            [16, 1, 16],
            512,
        )
        .into_mutable();
        graph.prepare_updates();
        let level = graph.height() - 1;
        let states = &mut graph.base_register.as_mut().unwrap()[level];
        assert_eq!(states.len(), 2);
        states.sort_unstable_by_key(|&(_, address)| address);
        let desired = states[1];
        states[0].0 = desired.0; // forced collision, wrong candidate first
        let word = graph.base.read_word(desired.1);
        assert_eq!(
            graph.find_base(
                level,
                &Signature {
                    word,
                    children: vec![]
                }
            ),
            Some(desired.1)
        );
    }
    #[test]
    fn inverse_edits_reuse_base_identity_and_bound_the_arena() {
        let mut graph =
            crate::Fst::build(vec![(0, 0, 1), (1, 0, 2)], [1024, 4, 1024], 512).into_mutable();
        let original = graph.root;
        // Duplicate/absent operations don't construct the graph-sized register.
        assert!(!graph.insert(0, 0, 1).unwrap());
        assert!(!graph.remove(3, 0, 1).unwrap());
        assert!(graph.base_register.is_none());
        for _ in 0..1000 {
            assert!(graph.insert(9, 2, 13).unwrap());
            assert!(graph.remove(9, 2, 13).unwrap());
            assert_eq!(graph.root, original);
            assert_eq!(graph.live_nodes, 0);
            assert!(graph.nodes.len() <= 2 * graph.height());
        }
    }
    #[test]
    fn freezing_walks_physical_states_instead_of_exponentially_many_triples() {
        let empty = crate::Fst::build(vec![], [1 << 24, 1, 1 << 24], 512);
        let height = empty.height();
        let mut levels: Vec<Vec<BfsNodeMeta>> = (0..height).map(|_| Vec::new()).collect();
        let mut visits = 0u64;
        for level in (0..height).rev() {
            let (bits, _, _) = empty.level_schedule(level);
            let slots = 1usize << bits.iter().sum::<u8>();
            let word = if slots == 64 {
                u64::MAX
            } else {
                (1 << slots) - 1
            };
            let children = if level + 1 < height {
                vec![0; slots]
            } else {
                Vec::new()
            };
            visits = 1 + visits * children.len() as u64;
            levels[level].push(BfsNodeMeta {
                existence_word: word,
                child_indices: children,
            });
        }
        let base = crate::Fst::pack_nodes(levels, empty.dimension_sizes(), 512, visits).unwrap();
        assert_eq!(base.node_count(), height as u64);
        let mut graph = base.into_mutable();
        assert_eq!(graph.count_matching([None; 3]), 1 << 48);
        assert!(graph.remove(1, 0, 2).unwrap());
        assert_eq!(graph.count_matching([None; 3]), (1 << 48) - 1);
        let frozen = graph.freeze_in_place().unwrap();
        assert!(frozen.sealed_node_count() <= height);
        assert_eq!(frozen.count_matching([None; 3]), (1 << 48) - 1);
        let graph = frozen.into_mutable();
        let (packed, stats) = graph.compact_blocks().unwrap();
        assert_eq!(stats.released_blocks, stats.source_blocks);
        assert!(packed.node_count() <= 2 * height as u64);
        assert_eq!(packed.count_matching([None; 3]), (1 << 48) - 1);
        assert!(packed.contains(2, 0, 2));
        assert!(!packed.contains(1, 0, 2));
        let restored = crate::Fst::deserialize(&packed.serialize()).unwrap();
        assert_eq!(restored.count_matching([None; 3]), (1 << 48) - 1);
        let (packed, stats) = restored.into_mutable().compact_recursive().unwrap();
        assert_eq!(stats.released_blocks, stats.source_blocks);
        assert_eq!(packed.count_matching([None; 3]), (1 << 48) - 1);
        assert!(packed.contains(2, 0, 2));
        assert!(!packed.contains(1, 0, 2));
        let restored = crate::Fst::deserialize(&packed.serialize()).unwrap();
        assert_eq!(restored.count_matching([None; 3]), (1 << 48) - 1);
        let (packed, stats) = restored.into_mutable().compact_depths().unwrap();
        assert_eq!(stats.released_blocks, stats.source_blocks);
        assert_eq!(packed.count_matching([None; 3]), (1 << 48) - 1);
        assert!(packed.contains(2, 0, 2));
        assert!(!packed.contains(1, 0, 2));
        let restored = crate::Fst::deserialize(&packed.serialize()).unwrap();
        assert_eq!(restored.count_matching([None; 3]), (1 << 48) - 1);
        let mut graph = restored.into_mutable();
        graph.remove(2, 0, 3).unwrap();
        let edit_states = graph.live_nodes;
        let (packed, stats) = graph.compact_streaming().unwrap();
        assert_eq!(stats.rank_lookups, 0);
        assert_eq!(stats.forwarded_edit_states, edit_states);
        assert_eq!(packed.count_matching([None; 3]), (1 << 48) - 2);
        assert!(!packed.contains(1, 0, 2));
        assert!(!packed.contains(2, 0, 3));
    }
}
