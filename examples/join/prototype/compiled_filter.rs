//! Experimental compiled restriction products; injected into an isolated crate.
use crate::level_automaton::{And, DimRestrictionAutomaton, DimState, LevelAutomaton};
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::{DimRestriction, MAX_HEIGHT, advance_lo, restriction_mask};
use std::collections::HashMap;
use std::ops::ControlFlow;

type Query = And<And<DimRestrictionAutomaton, DimRestrictionAutomaton>, DimRestrictionAutomaton>;
type QueryState = ((DimState, DimState), DimState);
const DEAD: u32 = 0;
const ALL: u32 = 1;
const STATE_LIMIT: usize = 65_536;

fn slots(tree: &PathFstDeltaPow2Bfs, level: usize) -> u64 {
    let (bits, _, _) = tree.level_schedule(level);
    let count = 1u32 << bits.iter().sum::<u8>();
    if count == 64 {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}
pub(crate) fn masks(
    tree: &PathFstDeltaPow2Bfs,
    restrictions: &[DimRestriction<'_>; 3],
) -> [u64; MAX_HEIGHT] {
    let mut masks = [0; MAX_HEIGHT];
    let mut lo = [0; 3];
    for (level, mask) in masks.iter_mut().enumerate().take(tree.height()) {
        let (bits, shifts, remaining) = tree.level_schedule(level);
        *mask = restriction_mask(bits, shifts, lo, remaining, restrictions);
        for d in 0..3 {
            if let DimRestriction::Exact(value) = restrictions[d] {
                lo[d] = (value as u64 >> remaining[d]) << remaining[d];
            }
        }
    }
    masks
}
fn invalid(tree: &PathFstDeltaPow2Bfs, restrictions: &[DimRestriction<'_>; 3]) -> bool {
    restrictions.iter().enumerate().any(|(d, r)| match *r {
        DimRestriction::Exact(v) => v as u64 >= tree.dim_sizes()[d],
        DimRestriction::Range(a, b) => a > b || a as u64 >= tree.dim_sizes()[d],
        DimRestriction::Multi(ranges) => !ranges
            .iter()
            .any(|&(a, b)| a <= b && (a as u64) < tree.dim_sizes()[d]),
        _ => false,
    })
}
fn normalize(r: DimRestriction<'_>, domain: u64) -> Vec<(u32, u32)> {
    let max = (domain - 1) as u32;
    let mut ranges = match r {
        DimRestriction::Free => vec![(0, max)],
        DimRestriction::Exact(v) => vec![(v, v)],
        DimRestriction::Range(a, b) => vec![(a, b)],
        DimRestriction::Multi(xs) => xs.to_vec(),
    };
    ranges.retain(|&(a, b)| a <= b && a <= max);
    for (_, b) in &mut ranges {
        *b = (*b).min(max);
    }
    ranges.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (a, b) in ranges {
        if let Some(last) = out.last_mut()
            && (a as u64) <= last.1 as u64 + 1
        {
            last.1 = last.1.max(b);
        } else {
            out.push((a, b));
        }
    }
    out
}
fn query(tree: &PathFstDeltaPow2Bfs, r: [DimRestriction<'_>; 3]) -> Query {
    let (bits, shifts, remaining) = tree.full_schedule();
    And(
        And(
            DimRestrictionAutomaton::new(0, r[0], bits, shifts, remaining),
            DimRestrictionAutomaton::new(1, r[1], bits, shifts, remaining),
        ),
        DimRestrictionAutomaton::new(2, r[2], bits, shifts, remaining),
    )
}
fn normalized_dimensions(
    tree: &PathFstDeltaPow2Bfs,
    r: [DimRestriction<'_>; 3],
) -> [DimRestrictionAutomaton; 3] {
    let ranges: [Vec<(u32, u32)>; 3] =
        std::array::from_fn(|d| normalize(r[d], tree.dim_sizes()[d]));
    let normalized: [DimRestriction<'_>; 3] = std::array::from_fn(|d| {
        let xs = &ranges[d];
        if xs.len() == 1 && xs[0] == (0, (tree.dim_sizes()[d] - 1) as u32) {
            DimRestriction::Free
        } else if xs.len() == 1 && xs[0].0 == xs[0].1 {
            DimRestriction::Exact(xs[0].0)
        } else if xs.len() == 1 {
            DimRestriction::Range(xs[0].0, xs[0].1)
        } else {
            DimRestriction::Multi(xs)
        }
    });
    let (bits, shifts, remaining) = tree.full_schedule();
    std::array::from_fn(|d| DimRestrictionAutomaton::new(d, normalized[d], bits, shifts, remaining))
}

fn normalized_query(tree: &PathFstDeltaPow2Bfs, r: [DimRestriction<'_>; 3]) -> Query {
    let [a, b, c] = normalized_dimensions(tree, r);
    And(And(a, b), c)
}

enum LegacyKernel {
    Empty,
    Masks([u64; MAX_HEIGHT]),
    Composed(Query),
}
pub struct LegacyFilter<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    kernel: LegacyKernel,
}
impl<'a> LegacyFilter<'a> {
    pub fn new(
        tree: &'a PathFstDeltaPow2Bfs,
        r: [DimRestriction<'_>; 3],
        normalized: bool,
    ) -> Self {
        let kernel = if tree.height() == 0 || invalid(tree, &r) {
            LegacyKernel::Empty
        } else if r
            .iter()
            .all(|r| matches!(r, DimRestriction::Exact(_) | DimRestriction::Free))
        {
            LegacyKernel::Masks(masks(tree, &r))
        } else {
            LegacyKernel::Composed(if normalized {
                normalized_query(tree, r)
            } else {
                query(tree, r)
            })
        };
        Self { tree, kernel }
    }
    pub fn visit<T>(&self, mut emit: impl FnMut([u32; 3]) -> ControlFlow<T>) -> ControlFlow<T> {
        match &self.kernel {
            LegacyKernel::Empty => ControlFlow::Continue(()),
            LegacyKernel::Masks(ms) => {
                walk_masks(self.tree, self.tree.root_addr(), 0, [0; 3], ms, &mut emit)
            }
            LegacyKernel::Composed(q) => walk_legacy(
                self.tree,
                self.tree.root_addr(),
                0,
                [0; 3],
                q,
                q.start(),
                &mut emit,
            ),
        }
    }
    /// Persistent query-heap allocations, excluding tree and inline structs.
    pub fn allocated_bytes(&self) -> usize {
        match &self.kernel {
            LegacyKernel::Composed(q) => {
                q.0.0.prototype_bytes() + q.0.1.prototype_bytes() + q.1.prototype_bytes()
            }
            _ => 0,
        }
    }
}
fn walk_legacy<T>(
    tree: &PathFstDeltaPow2Bfs,
    addr: u64,
    level: usize,
    lo: [u64; 3],
    q: &Query,
    state: QueryState,
    emit: &mut impl FnMut([u32; 3]) -> ControlFlow<T>,
) -> ControlFlow<T> {
    if !q.can_match(&state) {
        return ControlFlow::Continue(());
    }
    let word = tree.read_word(addr);
    let mut remaining = word & q.mask(&state, level);
    let (bits, shifts, cell) = tree.level_schedule(level);
    while remaining != 0 {
        let slot = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let next_lo = advance_lo(lo, bits, shifts, slot as usize, cell);
        if level + 1 == tree.height() {
            emit(next_lo.map(|v| v as u32))?;
        } else {
            let next = q.advance(&state, level, slot);
            if !q.can_match(&next) {
                continue;
            }
            let child =
                addr + tree.read_delta(addr, (word & ((1u64 << slot) - 1)).count_ones() as usize);
            walk_legacy(tree, child, level + 1, next_lo, q, next, emit)?;
        }
    }
    ControlFlow::Continue(())
}
fn walk_masks<T>(
    tree: &PathFstDeltaPow2Bfs,
    addr: u64,
    level: usize,
    lo: [u64; 3],
    ms: &[u64],
    emit: &mut impl FnMut([u32; 3]) -> ControlFlow<T>,
) -> ControlFlow<T> {
    let word = tree.read_word(addr);
    let mut remaining = word & ms[level];
    let (bits, shifts, cell) = tree.level_schedule(level);
    while remaining != 0 {
        let slot = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let next_lo = advance_lo(lo, bits, shifts, slot as usize, cell);
        if level + 1 == tree.height() {
            emit(next_lo.map(|v| v as u32))?;
        } else {
            let child =
                addr + tree.read_delta(addr, (word & ((1u64 << slot) - 1)).count_ones() as usize);
            walk_masks(tree, child, level + 1, next_lo, ms, emit)?;
        }
    }
    ControlFlow::Continue(())
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Node {
    level: usize,
    mask: u64,
    next: [u32; 64],
}
enum Kernel {
    Masks([u64; MAX_HEIGHT]),
    States(u32, Vec<Node>),
}
pub struct CompiledFilter<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    kernel: Kernel,
    pub raw_states: usize,
}
struct Compiler<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    q: Query,
    memo: HashMap<(usize, QueryState), u32>,
    started: usize,
    budget: usize,
    unique: HashMap<Node, u32>,
    nodes: Vec<Node>,
}
impl Compiler<'_> {
    fn build(&mut self, level: usize, state: QueryState) -> Result<u32, String> {
        if level == self.tree.height() {
            return Ok(ALL);
        }
        let key = (level, state);
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        if self.started >= self.budget {
            return Err("compiled filter state budget exceeded".into());
        }
        self.started += 1;
        let mut remaining = self.q.mask(&key.1, level) & slots(self.tree, level);
        let mut node = Node {
            level,
            mask: 0,
            next: [DEAD; 64],
        };
        while remaining != 0 {
            let slot = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let next = self.q.advance(&key.1, level, slot);
            let child = self.build(level + 1, next)?;
            if child != DEAD {
                node.mask |= 1u64 << slot;
                node.next[slot as usize] = child;
            }
        }
        let id = if node.mask == 0 {
            DEAD
        } else if node.mask == slots(self.tree, level)
            && (0..64).all(|i| node.mask & (1u64 << i) == 0 || node.next[i] == ALL)
        {
            ALL
        } else if let Some(&id) = self.unique.get(&node) {
            id
        } else {
            let id = self.nodes.len() as u32 + 2;
            self.nodes.push(node.clone());
            self.unique.insert(node, id);
            id
        };
        self.memo.insert(key, id);
        Ok(id)
    }
}

struct DimDfa {
    root: u32,
    nodes: Vec<Node>,
}
struct DimCompiler<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    dim: usize,
    q: DimRestrictionAutomaton,
    memo: HashMap<(usize, DimState), u32>,
    unique: HashMap<Node, u32>,
    nodes: Vec<Node>,
    started: usize,
}
fn intern(node: Node, full: u64, nodes: &mut Vec<Node>, unique: &mut HashMap<Node, u32>) -> u32 {
    if node.mask == 0 {
        DEAD
    } else if node.mask == full
        && (0..64).all(|i| node.mask & (1u64 << i) == 0 || node.next[i] == ALL)
    {
        ALL
    } else if let Some(&id) = unique.get(&node) {
        id
    } else {
        let id = nodes.len() as u32 + 2;
        nodes.push(node.clone());
        unique.insert(node, id);
        id
    }
}
impl DimCompiler<'_> {
    fn build(&mut self, level: usize, state: DimState) -> Result<u32, String> {
        if level == self.tree.height() {
            return Ok(ALL);
        }
        let key = (level, state);
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        if self.started >= STATE_LIMIT {
            return Err("dimension state budget exceeded".into());
        }
        self.started += 1;
        let (bits, shifts, _) = self.tree.level_schedule(level);
        let count = 1u32 << bits[self.dim];
        let full = if count == 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        let mask = self.q.mask(&key.1, level);
        let mut node = Node {
            level,
            mask: 0,
            next: [DEAD; 64],
        };
        for digit in 0..count {
            let slot = digit << shifts[self.dim];
            if mask & (1u64 << slot) == 0 {
                continue;
            }
            let child = self.build(level + 1, self.q.advance(&key.1, level, slot))?;
            if child != DEAD {
                node.mask |= 1u64 << digit;
                node.next[digit as usize] = child;
            }
        }
        let id = intern(node, full, &mut self.nodes, &mut self.unique);
        self.memo.insert(key, id);
        Ok(id)
    }
}
struct ProductCompiler<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    dims: Vec<DimDfa>,
    memo: HashMap<(usize, [u32; 3]), u32>,
    unique: HashMap<Node, u32>,
    nodes: Vec<Node>,
    started: usize,
}
impl ProductCompiler<'_> {
    fn build(&mut self, level: usize, ids: [u32; 3]) -> Result<u32, String> {
        if ids.contains(&DEAD) {
            return Ok(DEAD);
        }
        if ids == [ALL; 3] {
            return Ok(ALL);
        }
        if level == self.tree.height() {
            return Ok(ALL);
        }
        let key = (level, ids);
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        if self.started >= STATE_LIMIT {
            return Err("product state budget exceeded".into());
        }
        self.started += 1;
        let (bits, shifts, _) = self.tree.level_schedule(level);
        let mut remaining = slots(self.tree, level);
        let mut node = Node {
            level,
            mask: 0,
            next: [DEAD; 64],
        };
        while remaining != 0 {
            let slot = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let mut next = [ALL; 3];
            for d in 0..3 {
                if ids[d] != ALL {
                    let digit = (slot >> shifts[d]) & ((1u32 << bits[d]) - 1);
                    next[d] = self.dims[d].nodes[ids[d] as usize - 2].next[digit as usize];
                }
            }
            let child = self.build(level + 1, next)?;
            if child != DEAD {
                node.mask |= 1u64 << slot;
                node.next[slot as usize] = child;
            }
        }
        let id = intern(
            node,
            slots(self.tree, level),
            &mut self.nodes,
            &mut self.unique,
        );
        self.memo.insert(key, id);
        Ok(id)
    }
}
impl<'a> CompiledFilter<'a> {
    pub fn new(tree: &'a PathFstDeltaPow2Bfs, r: [DimRestriction<'_>; 3]) -> Result<Self, String> {
        if tree.height() == 0
            || invalid(tree, &r)
            || r.iter()
                .all(|r| matches!(r, DimRestriction::Exact(_) | DimRestriction::Free))
        {
            return Self::new_monolithic(tree, r);
        }
        let mut dims = Vec::new();
        let mut raw_states = 0;
        for (dim, q) in normalized_dimensions(tree, r).into_iter().enumerate() {
            let start = q.start();
            let mut c = DimCompiler {
                tree,
                dim,
                q,
                memo: HashMap::new(),
                unique: HashMap::new(),
                nodes: Vec::new(),
                started: 0,
            };
            let root = c.build(0, start)?;
            raw_states += c.started;
            dims.push(DimDfa {
                root,
                nodes: c.nodes,
            });
        }
        let start = std::array::from_fn(|d| dims[d].root);
        let mut c = ProductCompiler {
            tree,
            dims,
            memo: HashMap::new(),
            unique: HashMap::new(),
            nodes: Vec::new(),
            started: 0,
        };
        let root = c.build(0, start)?;
        raw_states += c.started;
        Ok(Self {
            tree,
            kernel: Kernel::States(root, c.nodes),
            raw_states,
        })
    }

    pub fn new_monolithic(
        tree: &'a PathFstDeltaPow2Bfs,
        r: [DimRestriction<'_>; 3],
    ) -> Result<Self, String> {
        if tree.height() == 0 || invalid(tree, &r) {
            return Ok(Self {
                tree,
                kernel: Kernel::States(DEAD, Vec::new()),
                raw_states: 0,
            });
        }
        if r.iter()
            .all(|r| matches!(r, DimRestriction::Exact(_) | DimRestriction::Free))
        {
            return Ok(Self {
                tree,
                kernel: Kernel::Masks(masks(tree, &r)),
                raw_states: 0,
            });
        }
        let q = normalized_query(tree, r);
        let state = q.start();
        let mut c = Compiler {
            tree,
            q,
            memo: HashMap::new(),
            started: 0,
            budget: STATE_LIMIT,
            unique: HashMap::new(),
            nodes: Vec::new(),
        };
        let root = c.build(0, state)?;
        let raw_states = c.memo.len();
        Ok(Self {
            tree,
            kernel: Kernel::States(root, c.nodes),
            raw_states,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        match &self.kernel {
            Kernel::Masks(_) => 0,
            Kernel::States(_, nodes) => nodes.capacity() * std::mem::size_of::<Node>(),
        }
    }
    pub fn states(&self) -> usize {
        match &self.kernel {
            Kernel::Masks(_) => 0,
            Kernel::States(_, nodes) => nodes.len(),
        }
    }
    pub fn visit<T>(&self, mut emit: impl FnMut([u32; 3]) -> ControlFlow<T>) -> ControlFlow<T> {
        match &self.kernel {
            Kernel::Masks(ms) => {
                walk_masks(self.tree, self.tree.root_addr(), 0, [0; 3], ms, &mut emit)
            }
            Kernel::States(DEAD, _) => ControlFlow::Continue(()),
            Kernel::States(root, nodes) => {
                self.walk(*root, nodes, self.tree.root_addr(), 0, [0; 3], &mut emit)
            }
        }
    }
    fn walk<T>(
        &self,
        id: u32,
        nodes: &[Node],
        addr: u64,
        level: usize,
        lo: [u64; 3],
        emit: &mut impl FnMut([u32; 3]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if id == DEAD {
            return ControlFlow::Continue(());
        }
        let word = self.tree.read_word(addr);
        let mask = if id == ALL {
            u64::MAX
        } else {
            nodes[id as usize - 2].mask
        };
        let mut remaining = word & mask;
        let (bits, shifts, cell) = self.tree.level_schedule(level);
        while remaining != 0 {
            let slot = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let next_id = if id == ALL {
                ALL
            } else {
                nodes[id as usize - 2].next[slot as usize]
            };
            let next_lo = advance_lo(lo, bits, shifts, slot as usize, cell);
            if level + 1 == self.tree.height() {
                if next_id == ALL {
                    emit(next_lo.map(|v| v as u32))?;
                }
            } else if next_id != DEAD {
                let child = addr
                    + self
                        .tree
                        .read_delta(addr, (word & ((1u64 << slot) - 1)).count_ones() as usize);
                self.walk(next_id, nodes, child, level + 1, next_lo, emit)?;
            }
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_budget_counts_in_progress_states() {
        let tree = PathFstDeltaPow2Bfs::build(
            vec![(5, 1, 7)],
            [33, 5, 129],
            crate::DEFAULT_BLOCK_CAPACITY,
        );
        let q = normalized_query(
            &tree,
            [
                DimRestriction::Exact(5),
                DimRestriction::Exact(1),
                DimRestriction::Range(7, 8),
            ],
        );
        let start = q.start();
        let mut compiler = Compiler {
            tree: &tree,
            q,
            memo: HashMap::new(),
            started: 0,
            budget: 2,
            unique: HashMap::new(),
            nodes: Vec::new(),
        };
        assert!(compiler.build(0, start).is_err());
        assert_eq!(compiler.started, 2);
    }
    fn accepts(r: DimRestriction<'_>, v: u32) -> bool {
        match r {
            DimRestriction::Free => true,
            DimRestriction::Exact(x) => v == x,
            DimRestriction::Range(a, b) => a <= v && v <= b,
            DimRestriction::Multi(xs) => xs.iter().any(|&(a, b)| a <= v && v <= b),
        }
    }
    #[test]
    fn filters_match_source_and_legacy() {
        let dims = [33, 5, 129];
        let triples: Vec<_> = (0..33)
            .flat_map(|s| {
                (0..5).flat_map(move |p| {
                    (0..129)
                        .filter(move |o| (s * 19 + p * 7 + o) % 13 == 0)
                        .map(move |o| (s, p, o))
                })
            })
            .collect();
        let tree = PathFstDeltaPow2Bfs::build(triples.clone(), dims, crate::DEFAULT_BLOCK_CAPACITY);
        let xs = [(8, 15), (0, 10), (14, 20), (25, 32)];
        let ys = [(0, 2), (1, 4)];
        let zs = [(12, 40), (70, 80), (75, 110), (128, 200)];
        for seed in 0..30u32 {
            let r = [
                if seed % 3 == 0 {
                    DimRestriction::Multi(&xs)
                } else {
                    DimRestriction::Range(seed / 2, seed + 1)
                },
                if seed % 2 == 0 {
                    DimRestriction::Multi(&ys)
                } else {
                    DimRestriction::Exact(seed % 7)
                },
                if seed % 4 == 0 {
                    DimRestriction::Multi(&zs)
                } else {
                    DimRestriction::Range(seed, 128 - seed)
                },
            ];
            let mut expected: Vec<_> = triples
                .iter()
                .filter(|&&(s, p, o)| accepts(r[0], s) && accepts(r[1], p) && accepts(r[2], o))
                .map(|&(s, p, o)| [s, p, o])
                .collect();
            expected.sort_unstable();
            let compiled = CompiledFilter::new(&tree, r).unwrap();
            let mut actual = Vec::new();
            let _: ControlFlow<()> = compiled.visit(|t| {
                actual.push(t);
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, expected);
            let legacy = LegacyFilter::new(&tree, r, false);
            let mut old = Vec::new();
            let _: ControlFlow<()> = legacy.visit(|t| {
                old.push(t);
                ControlFlow::Continue(())
            });
            old.sort_unstable();
            assert_eq!(old, expected);
            if !expected.is_empty() {
                let result = compiled.visit(|t| ControlFlow::Break(t));
                assert!(matches!(result, ControlFlow::Break(_)));
            }
        }
    }
    #[test]
    fn empty_universal_and_singleton_filters() {
        let tree = PathFstDeltaPow2Bfs::build(
            vec![(0, 0, 0), (3, 1, 8)],
            [4, 2, 9],
            crate::DEFAULT_BLOCK_CAPACITY,
        );
        for r in [
            [
                DimRestriction::Multi(&[]),
                DimRestriction::Free,
                DimRestriction::Free,
            ],
            [
                DimRestriction::Range(10, 2),
                DimRestriction::Free,
                DimRestriction::Free,
            ],
            [
                DimRestriction::Exact(99),
                DimRestriction::Free,
                DimRestriction::Free,
            ],
        ] {
            let p = CompiledFilter::new(&tree, r).unwrap();
            let _: ControlFlow<()> = p.visit(|_| panic!("empty filter emitted"));
        }
        let p = CompiledFilter::new(
            &tree,
            [
                DimRestriction::Range(0, 3),
                DimRestriction::Multi(&[(1, 1), (0, 0)]),
                DimRestriction::Range(0, 8),
            ],
        )
        .unwrap();
        let mut n = 0;
        let _: ControlFlow<()> = p.visit(|_| {
            n += 1;
            ControlFlow::Continue(())
        });
        assert_eq!(n, 2);
    }
}
