//! Bounded graph traversal over integer IDs. No dictionary or query-language
//! semantics are imposed. The supplied start vertex is considered valid, even
//! if isolated; this makes zero-hop traversal usable with external vertex tables.
//!
//! Reachability uses layer-local visitation (walk semantics). A vertex reached
//! before `min` must be revisited at later depths. Endpoint-only execution keeps
//! two frontiers, independent of the hop bound. Path reconstruction retains
//! predecessor records; enumerating paths is output-sensitive and may be
//! exponential. Every streaming operation supports immediate cancellation.

use crate::join::PatternBackend;
use crate::{Result, TreeError, Triple};
use std::collections::HashSet;
use std::ops::ControlFlow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub direction: Direction,
    pub predicate: Option<u32>,
}

impl Default for Step {
    fn default() -> Self {
        Self { direction: Direction::Outgoing, predicate: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HopRange {
    min: u32,
    max: u32,
}

impl HopRange {
    pub fn new(min: u32, max: u32) -> Result<Self> {
        if min > max {
            return Err(TreeError::InvalidConfig("min_hops exceeds max_hops".into()));
        }
        Ok(Self { min, max })
    }
    pub fn min(self) -> u32 {
        self.min
    }
    pub fn max(self) -> u32 {
        self.max
    }
}

/// A directed stored triple is the edge identity. Identical triples are one
/// edge; a consumer needing parallel edges must encode distinct edge identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathSemantics {
    Walk,
    Trail,
    Simple,
}

/// Optional dense bitset, with sparse overflow for arbitrary u32 IDs. Clearing
/// touches only words written during the previous query/layer.
#[derive(Default)]
struct NodeSet {
    words: Vec<u64>,
    touched: Vec<usize>,
    sparse: HashSet<u32>,
}

impl NodeSet {
    fn dense(nodes: usize) -> Self {
        Self { words: vec![0; nodes.div_ceil(64)], ..Self::default() }
    }
    fn clear(&mut self) {
        for i in self.touched.drain(..) {
            self.words[i] = 0;
        }
        self.sparse.clear();
    }
    fn insert(&mut self, node: u32) -> bool {
        let index = node as usize / 64;
        if index >= self.words.len() {
            return self.sparse.insert(node);
        }
        let word = &mut self.words[index];
        let bit = 1u64 << (node % 64);
        if *word & bit != 0 {
            return false;
        }
        if *word == 0 {
            self.touched.push(index);
        }
        *word |= bit;
        true
    }
}

#[derive(Clone, Copy)]
struct Predecessor {
    node: u32,
    parent: usize,
    edge: Triple,
}

/// Reuses frontier, visited-set and predecessor allocations. Default visitation
/// is sparse and never allocates by maximum node ID. Dense visitation is opt-in.
#[derive(Default)]
pub struct PathWorkspace {
    frontier: Vec<u32>,
    next: Vec<u32>,
    layer: NodeSet,
    emitted: NodeSet,
    records: Vec<Predecessor>,
    record_frontier: Vec<usize>,
    record_next: Vec<usize>,
}

impl PathWorkspace {
    /// Allocate two bitsets for IDs below `nodes`, with sparse overflow.
    /// Intended for repeatedly queried dense domains; default is sparse.
    pub fn dense(nodes: usize) -> Self {
        Self { layer: NodeSet::dense(nodes), emitted: NodeSet::dense(nodes), ..Self::default() }
    }

    /// Drop retained high-water allocations after a large query.
    pub fn release(&mut self) {
        *self = Self::default();
    }
}

pub(crate) struct Edges<'a, B: PatternBackend + 'a> {
    graph: &'a B,
    scan: B::Scan<'a>,
    node: u32,
    step: Step,
    incoming: bool,
}

impl<'a, B: PatternBackend> Edges<'a, B> {
    pub(crate) fn new(graph: &'a B, node: u32, step: Step) -> Self {
        let incoming = step.direction == Direction::Incoming;
        let bounds = if incoming { [None, step.predicate, Some(node)] } else { [Some(node), step.predicate, None] };
        Self { graph, scan: graph.scan_pattern(bounds), node, step, incoming }
    }
}

impl<B: PatternBackend> Iterator for Edges<'_, B> {
    type Item = (u32, Triple);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(edge @ (s, _, o)) = self.scan.next() {
                // A self-loop is already emitted by the outgoing half of Both.
                if self.incoming && self.step.direction == Direction::Both && s == o {
                    continue;
                }
                return Some((if self.incoming { s } else { o }, edge));
            }
            if !self.incoming && self.step.direction == Direction::Both {
                self.incoming = true;
                self.scan = self.graph.scan_pattern([None, self.step.predicate, Some(self.node)]);
            } else {
                return None;
            }
        }
    }
}

/// Emit each reachable endpoint once, with its smallest qualifying walk length.
/// Layer-local deduplication preserves longer routes to previously seen nodes.
pub fn reachable<B: PatternBackend, T>(
    graph: &B,
    start: u32,
    hops: HopRange,
    step: Step,
    ws: &mut PathWorkspace,
    mut emit: impl FnMut(u32, u32) -> ControlFlow<T>,
) -> ControlFlow<T> {
    ws.frontier.clear();
    ws.next.clear();
    ws.emitted.clear();
    ws.frontier.push(start);
    if hops.min == 0 {
        ws.emitted.insert(start);
        emit(start, 0)?;
    }
    for depth in 0..hops.max {
        ws.next.clear();
        ws.layer.clear();
        for &node in &ws.frontier {
            for (next, _) in Edges::new(graph, node, step) {
                if !ws.layer.insert(next) {
                    continue;
                }
                if depth + 1 >= hops.min {
                    // After the lower bound is satisfied, an earlier qualifying
                    // visit dominates every later one (more remaining hop budget).
                    if !ws.emitted.insert(next) {
                        continue;
                    }
                    emit(next, depth + 1)?;
                }
                ws.next.push(next);
            }
        }
        std::mem::swap(&mut ws.frontier, &mut ws.next);
        if ws.frontier.is_empty() {
            break;
        }
    }
    ControlFlow::Continue(())
}

pub fn has_path<B: PatternBackend>(graph: &B, start: u32, target: u32, hops: HopRange, step: Step, ws: &mut PathWorkspace) -> bool {
    reachable(graph, start, hops, step, ws, |node, _| if node == target { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }).is_break()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphPath {
    pub nodes: Vec<u32>,
    /// Stored orientation, even when traversing an edge backwards.
    pub edges: Vec<Triple>,
}

/// The shortest walk whose length lies in `hops`, including reconstruction.
/// For min > 1, retains per-depth predecessors: O(sum of frontier sizes).
/// For min <= 1, global visitation avoids revisiting nodes after first discovery.
pub fn shortest_walk<B: PatternBackend>(graph: &B, start: u32, target: u32, hops: HopRange, step: Step, ws: &mut PathWorkspace) -> Option<GraphPath> {
    if start == target && hops.min == 0 {
        return Some(GraphPath { nodes: vec![start], edges: vec![] });
    }
    ws.records.clear();
    ws.record_frontier.clear();
    ws.layer.clear();
    ws.records.push(Predecessor { node: start, parent: 0, edge: (0, 0, 0) });
    ws.record_frontier.push(0);
    if hops.min == 0 {
        ws.layer.insert(start);
    }
    for depth in 0..hops.max {
        ws.record_next.clear();
        if depth < hops.min {
            ws.layer.clear();
        }
        for &parent in &ws.record_frontier {
            for (node, edge) in Edges::new(graph, ws.records[parent].node, step) {
                if !ws.layer.insert(node) {
                    continue;
                }
                let index = ws.records.len();
                ws.records.push(Predecessor { node, parent, edge });
                if node == target && depth + 1 >= hops.min {
                    let mut path = GraphPath { nodes: Vec::new(), edges: Vec::new() };
                    let mut cursor = index;
                    while cursor != 0 {
                        let record = ws.records[cursor];
                        path.nodes.push(record.node);
                        path.edges.push(record.edge);
                        cursor = record.parent;
                    }
                    path.nodes.push(start);
                    path.nodes.reverse();
                    path.edges.reverse();
                    return Some(path);
                }
                ws.record_next.push(index);
            }
        }
        std::mem::swap(&mut ws.record_frontier, &mut ws.record_next);
        if ws.record_frontier.is_empty() {
            break;
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub struct PathOptions {
    pub hops: HopRange,
    pub step: Step,
    pub semantics: PathSemantics,
    /// None emits paths ending at any node.
    pub target: Option<u32>,
}

/// Stream paths with O(max_hops * backend cursor size) traversal memory and no
/// result collection. Edge-distinct paths are emitted separately. Simple paths
/// cannot repeat a vertex; trails cannot repeat a stored triple in either direction.
pub fn enumerate_paths<B: PatternBackend, T>(
    graph: &B,
    start: u32,
    options: PathOptions,
    mut emit: impl FnMut(&GraphPath) -> ControlFlow<T>,
) -> ControlFlow<T> {
    let mut path = GraphPath { nodes: vec![start], edges: Vec::new() };
    let mut nodes = HashSet::new();
    let mut edges = HashSet::new();
    if options.semantics == PathSemantics::Simple {
        nodes.insert(start);
    }
    if options.hops.min == 0 && options.target.is_none_or(|v| v == start) {
        emit(&path)?;
    }
    if options.hops.max == 0 {
        return ControlFlow::Continue(());
    }
    let mut stack = vec![Edges::new(graph, start, options.step)];
    while let Some(cursor) = stack.last_mut() {
        let Some((node, edge)) = cursor.next() else {
            stack.pop();
            if let Some(edge) = path.edges.pop() {
                let node = path.nodes.pop().unwrap();
                nodes.remove(&node);
                edges.remove(&edge);
            }
            continue;
        };
        if options.semantics == PathSemantics::Simple && nodes.contains(&node) {
            continue;
        }
        if options.semantics == PathSemantics::Trail && edges.contains(&edge) {
            continue;
        }
        path.nodes.push(node);
        path.edges.push(edge);
        if options.semantics == PathSemantics::Simple {
            nodes.insert(node);
        }
        if options.semantics == PathSemantics::Trail {
            edges.insert(edge);
        }
        let depth = path.edges.len() as u32;
        if depth >= options.hops.min && options.target.is_none_or(|v| v == node) {
            emit(&path)?;
        }
        if depth < options.hops.max {
            stack.push(Edges::new(graph, node, options.step));
        } else {
            path.nodes.pop();
            path.edges.pop();
            nodes.remove(&node);
            edges.remove(&edge);
        }
    }
    ControlFlow::Continue(())
}
