//! Regular-path builders, a shared-language adapter, an allocation-free FST
//! PatternBackend, and a binding-plan adapter for distinct endpoint queries.
use crate::automata_core::{Automaton, TripleSource};
use crate::automaton::{AutomatonWorkspace, PathAutomaton, Transition};
use crate::graph::Term;
use crate::join::PatternBackend;
use crate::query_ops::{Binding, Plan, Row};
use crate::traversal::{Direction, HopRange, Step};
use crate::{Result, TreeError, Triple};
use std::ops::ControlFlow;
const MAX_HEIGHT: usize = 64;

#[derive(Clone)]
pub enum PathExpr {
    Epsilon,
    Edge(Step),
    Sequence(Vec<PathExpr>),
    Alternative(Vec<PathExpr>),
    Optional(Box<PathExpr>),
    Star(Box<PathExpr>),
    Plus(Box<PathExpr>),
}
impl PathExpr {
    pub fn label(predicate: u32) -> Self {
        Self::Edge(Step {
            direction: Direction::Outgoing,
            predicate: Some(predicate),
        })
    }
    pub fn inverse(predicate: u32) -> Self {
        Self::Edge(Step {
            direction: Direction::Incoming,
            predicate: Some(predicate),
        })
    }
    pub fn any(direction: Direction) -> Self {
        Self::Edge(Step {
            direction,
            predicate: None,
        })
    }
    pub fn then(self, other: Self) -> Self {
        Self::Sequence(vec![self, other])
    }
    pub fn or(self, other: Self) -> Self {
        Self::Alternative(vec![self, other])
    }
    pub fn optional(self) -> Self {
        Self::Optional(Box::new(self))
    }
    pub fn star(self) -> Self {
        Self::Star(Box::new(self))
    }
    pub fn plus(self) -> Self {
        Self::Plus(Box::new(self))
    }
    /// Reverse both the word order and every traversed edge direction.
    pub fn reversed(self) -> Self {
        match self {
            Self::Epsilon => Self::Epsilon,
            Self::Edge(mut step) => {
                step.direction = match step.direction {
                    Direction::Outgoing => Direction::Incoming,
                    Direction::Incoming => Direction::Outgoing,
                    Direction::Both => Direction::Both,
                };
                Self::Edge(step)
            }
            Self::Sequence(parts) => {
                Self::Sequence(parts.into_iter().rev().map(Self::reversed).collect())
            }
            Self::Alternative(parts) => {
                Self::Alternative(parts.into_iter().map(Self::reversed).collect())
            }
            Self::Optional(p) => p.reversed().optional(),
            Self::Star(p) => p.reversed().star(),
            Self::Plus(p) => p.reversed().plus(),
        }
    }
    pub fn compile(&self) -> Result<PathAutomaton> {
        self.compile_with_limits(4096, 1_000_000)
    }
    /// Bounded Thompson construction followed by epsilon elimination. This is
    /// preparation; runtime keeps packed NFA transitions, not a powerset DFA.
    pub fn compile_with_limits(
        &self,
        state_limit: usize,
        transition_limit: usize,
    ) -> Result<PathAutomaton> {
        fn invalid() -> TreeError {
            TreeError::InvalidConfig("path compilation budget exceeded".into())
        }
        struct Builder {
            edges: Vec<Vec<(usize, Option<Step>)>>,
            limit: usize,
        }
        impl Builder {
            fn pair(&mut self) -> Result<(usize, usize)> {
                if self.edges.len().saturating_add(2) > self.limit {
                    return Err(invalid());
                }
                let a = self.edges.len();
                self.edges.push(Vec::new());
                self.edges.push(Vec::new());
                Ok((a, a + 1))
            }
            fn build(&mut self, p: &PathExpr, depth: usize) -> Result<(usize, usize)> {
                if depth > 256 {
                    return Err(invalid());
                }
                let (a, b) = self.pair()?;
                match p {
                    PathExpr::Epsilon => self.edges[a].push((b, None)),
                    PathExpr::Edge(step) => self.edges[a].push((b, Some(*step))),
                    PathExpr::Sequence(parts) => {
                        let mut previous = a;
                        for p in parts {
                            let (x, y) = self.build(p, depth + 1)?;
                            self.edges[previous].push((x, None));
                            previous = y;
                        }
                        self.edges[previous].push((b, None));
                    }
                    PathExpr::Alternative(parts) => {
                        for p in parts {
                            let (x, y) = self.build(p, depth + 1)?;
                            self.edges[a].push((x, None));
                            self.edges[y].push((b, None));
                        }
                    }
                    PathExpr::Optional(child) | PathExpr::Star(child) | PathExpr::Plus(child) => {
                        let (x, y) = self.build(child, depth + 1)?;
                        self.edges[a].push((x, None));
                        self.edges[y].push((b, None));
                        if matches!(p, PathExpr::Optional(_) | PathExpr::Star(_)) {
                            self.edges[a].push((b, None));
                        }
                        if matches!(p, PathExpr::Star(_) | PathExpr::Plus(_)) {
                            self.edges[y].push((x, None));
                        }
                    }
                }
                Ok((a, b))
            }
        }
        let mut builder = Builder {
            edges: Vec::new(),
            limit: state_limit.min(u32::MAX as usize),
        };
        let (start, end) = builder.build(self, 0)?;
        let n = builder.edges.len();
        let mut accepting = Vec::new();
        let mut transitions = Vec::new();
        let mut seen = vec![false; n];
        let mut stack = Vec::new();
        for state in 0..n {
            seen.fill(false);
            stack.clear();
            stack.push(state);
            seen[state] = true;
            let mut accepts = false;
            let mut outgoing = Vec::new();
            while let Some(v) = stack.pop() {
                accepts |= v == end;
                for &(to, step) in &builder.edges[v] {
                    if let Some(step) = step {
                        outgoing.push(Transition {
                            from: state as u32,
                            to: to as u32,
                            step,
                        });
                    } else if !seen[to] {
                        seen[to] = true;
                        stack.push(to);
                    }
                }
            }
            outgoing.sort_unstable_by_key(|t| (t.to, t.step.direction as u8, t.step.predicate));
            outgoing.dedup_by_key(|t| (t.to, t.step.direction as u8, t.step.predicate));
            if transitions.len().saturating_add(outgoing.len()) > transition_limit {
                return Err(invalid());
            }
            transitions.extend(outgoing);
            if accepts {
                accepting.push(state as u32);
            }
        }
        PathAutomaton::new(n as u32, start as u32, &accepting, transitions)
    }
}
// A path event records the direction actually traversed, not the storage slot.
#[derive(Clone, Copy)]
pub struct PathEvent {
    pub predicate: u32,
    pub direction: Direction,
}
/// Small path languages keep their active NFA states inline. Larger programs
/// use canonical sparse state sets; the specialized NFA runner avoids either
/// powerset representation during graph traversal.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum NfaStateSet {
    Small(u64),
    Large(Vec<u32>),
}
impl Automaton<PathEvent> for PathAutomaton {
    type State = NfaStateSet;
    fn start(&self) -> Self::State {
        if self.prototype_states() <= 64 {
            NfaStateSet::Small(1u64 << self.prototype_start())
        } else {
            NfaStateSet::Large(vec![self.prototype_start()])
        }
    }
    fn step(&self, state: &Self::State, e: &PathEvent, _: &()) -> Option<Self::State> {
        let matches = |t: &Transition| {
            (t.step.direction == Direction::Both || t.step.direction == e.direction)
                && t.step.predicate.is_none_or(|p| p == e.predicate)
        };
        match state {
            NfaStateSet::Small(bits) => {
                let mut bits = *bits;
                let mut next = 0u64;
                while bits != 0 {
                    let s = bits.trailing_zeros();
                    bits &= bits - 1;
                    for t in self.prototype_transitions(s) {
                        if matches(t) {
                            next |= 1u64 << t.to;
                        }
                    }
                }
                (next != 0).then_some(NfaStateSet::Small(next))
            }
            NfaStateSet::Large(states) => {
                let mut next = Vec::new();
                for &s in states {
                    for t in self.prototype_transitions(s) {
                        if matches(t) {
                            next.push(t.to);
                        }
                    }
                }
                next.sort_unstable();
                next.dedup();
                (!next.is_empty()).then_some(NfaStateSet::Large(next))
            }
        }
    }
    fn is_accepting(&self, state: &Self::State, _: &()) -> bool {
        match state {
            NfaStateSet::Small(bits) => {
                let mut bits = *bits;
                while bits != 0 {
                    let s = bits.trailing_zeros();
                    bits &= bits - 1;
                    if self.prototype_accepts(s) {
                        return true;
                    }
                }
                false
            }
            NfaStateSet::Large(states) => states.iter().any(|&s| self.prototype_accepts(s)),
        }
    }
    fn can_match(&self, state: &Self::State, _: &()) -> bool {
        match state {
            NfaStateSet::Small(bits) => *bits != 0,
            NfaStateSet::Large(states) => !states.is_empty(),
        }
    }
}

#[derive(Clone, Copy)]
struct Layout {
    bits: [u32; 3],
    shifts: [u8; 3],
    remaining: [u8; 3],
    valid: u64,
}
pub struct PatternIndex<S: TripleSource> {
    source: S,
    layout: Vec<Layout>,
}
impl<S: TripleSource> PatternIndex<S> {
    pub fn new(source: S) -> Result<Self> {
        if source.height() > MAX_HEIGHT {
            return Err(TreeError::InvalidConfig(
                "path source exceeds 64 levels".into(),
            ));
        }
        let layout = (0..source.height())
            .map(|l| {
                let (bits, shifts, remaining) = source.layout(l);
                let slots = 1u32 << bits.iter().sum::<u8>();
                Layout {
                    bits: bits.map(|b| (1u32 << b) - 1),
                    shifts,
                    remaining,
                    valid: if slots == 64 {
                        u64::MAX
                    } else {
                        (1u64 << slots) - 1
                    },
                }
            })
            .collect();
        Ok(Self { source, layout })
    }
}
#[derive(Clone, Copy, Default)]
struct Frame<C: Copy + Default> {
    cursor: C,
    word: u64,
    todo: u64,
    values: [u32; 3],
}
pub struct PatternIter<'a, S: TripleSource> {
    index: &'a PatternIndex<S>,
    frames: [Frame<S::Cursor>; MAX_HEIGHT],
    masks: [u64; MAX_HEIGHT],
    depth: usize,
    done: bool,
}
impl<S: TripleSource> PatternBackend for PatternIndex<S> {
    type Scan<'a>
        = PatternIter<'a, S>
    where
        S: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        let mut scan = PatternIter {
            index: self,
            frames: [Frame::default(); MAX_HEIGHT],
            masks: [0; MAX_HEIGHT],
            depth: 0,
            done: true,
        };
        if bounds
            .iter()
            .enumerate()
            .any(|(d, x)| x.is_some_and(|x| x as u64 >= self.source.domains()[d]))
        {
            return scan;
        }
        let Some(root) = self.source.root().filter(|_| !self.layout.is_empty()) else {
            return scan;
        };
        for (level, l) in self.layout.iter().enumerate() {
            let mut mask = l.valid;
            for (d, value) in bounds.iter().enumerate() {
                if let Some(value) = value {
                    let mut bits = l.bits[d];
                    while bits != 0 {
                        let bit = bits.trailing_zeros();
                        bits &= bits - 1;
                        let ones = crate::native_join::slot_ones(l.shifts[d] + bit as u8);
                        mask &= if value & (1u32 << (l.remaining[d] + bit as u8)) != 0 {
                            ones
                        } else {
                            !ones
                        };
                    }
                }
            }
            scan.masks[level] = mask;
        }
        let word = self.source.word(root);
        scan.frames[0] = Frame {
            cursor: root,
            word,
            todo: word & scan.masks[0],
            values: [0; 3],
        };
        scan.done = false;
        scan
    }
}
impl<S: TripleSource> Iterator for PatternIter<'_, S> {
    type Item = Triple;
    fn next(&mut self) -> Option<Triple> {
        if self.done {
            return None;
        }
        loop {
            let f = &mut self.frames[self.depth];
            if f.todo == 0 {
                if self.depth == 0 {
                    self.done = true;
                    return None;
                }
                self.depth -= 1;
                continue;
            }
            let slot = f.todo.trailing_zeros();
            f.todo &= f.todo - 1;
            let l = &self.index.layout[self.depth];
            let values: [u32; 3] = std::array::from_fn(|d| {
                f.values[d] | (((slot >> l.shifts[d]) & l.bits[d]) << l.remaining[d])
            });
            if self.depth + 1 == self.index.layout.len() {
                return Some((values[0], values[1], values[2]));
            }
            let cursor = self.index.source.follow(f.cursor, f.word, slot);
            let word = self.index.source.word(cursor);
            self.depth += 1;
            self.frames[self.depth] = Frame {
                cursor,
                word,
                todo: word & self.masks[self.depth],
                values,
            };
        }
    }
}
/// Execution specialization for a path machine. The packed NFA runner pushes
/// transition labels into storage scans; generic composition preserves the
/// same language semantics with conservative outgoing/incoming exploration.
pub trait PathRunner<G: PatternBackend> {
    type Workspace: Default;
    fn reachable<T>(
        &self,
        graph: &G,
        start: u32,
        hops: HopRange,
        ws: &mut Self::Workspace,
        emit: impl FnMut(u32, u32) -> ControlFlow<T>,
    ) -> ControlFlow<T>;
}
impl<G: PatternBackend> PathRunner<G> for PathAutomaton {
    type Workspace = AutomatonWorkspace;
    fn reachable<T>(
        &self,
        graph: &G,
        start: u32,
        hops: HopRange,
        ws: &mut Self::Workspace,
        emit: impl FnMut(u32, u32) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        PathAutomaton::reachable(self, graph, start, hops, ws, emit)
    }
}
pub struct Composed<A>(pub A);
pub struct LanguageWorkspace<Q: Clone + Eq + std::hash::Hash> {
    frontier: Vec<(u32, Q)>,
    next: Vec<(u32, Q)>,
    visited: std::collections::HashSet<(u32, Q)>,
    layer: std::collections::HashSet<(u32, Q)>,
    emitted: std::collections::HashSet<u32>,
}
impl<Q: Clone + Eq + std::hash::Hash> Default for LanguageWorkspace<Q> {
    fn default() -> Self {
        Self {
            frontier: Vec::new(),
            next: Vec::new(),
            visited: Default::default(),
            layer: Default::default(),
            emitted: Default::default(),
        }
    }
}
impl<G: PatternBackend, A: Automaton<PathEvent>> PathRunner<G> for Composed<A>
where
    A::State: Eq + std::hash::Hash,
{
    type Workspace = LanguageWorkspace<A::State>;
    fn reachable<T>(
        &self,
        graph: &G,
        start: u32,
        hops: HopRange,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(u32, u32) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        ws.frontier.clear();
        ws.next.clear();
        ws.visited.clear();
        ws.layer.clear();
        ws.emitted.clear();
        let initial = self.0.start();
        if hops.min() == 0 {
            ws.visited.insert((start, initial.clone()));
            if self.0.is_accepting(&initial, &()) {
                ws.emitted.insert(start);
                emit(start, 0)?;
            }
        }
        ws.frontier.push((start, initial));
        for depth in 0..hops.max() {
            ws.layer.clear();
            ws.next.clear();
            for (node, state) in &ws.frontier {
                for direction in [Direction::Outgoing, Direction::Incoming] {
                    let bounds = if direction == Direction::Outgoing {
                        [Some(*node), None, None]
                    } else {
                        [None, None, Some(*node)]
                    };
                    for (s, p, o) in graph.scan_pattern(bounds) {
                        let next = if direction == Direction::Outgoing {
                            o
                        } else {
                            s
                        };
                        let Some(q) = self.0.step(
                            state,
                            &PathEvent {
                                predicate: p,
                                direction,
                            },
                            &(),
                        ) else {
                            continue;
                        };
                        if !self.0.can_match(&q, &()) {
                            continue;
                        }
                        let product = (next, q);
                        if !ws.layer.insert(product.clone()) {
                            continue;
                        }
                        if depth + 1 >= hops.min() {
                            if !ws.visited.insert(product.clone()) {
                                continue;
                            }
                            if self.0.is_accepting(&product.1, &()) && ws.emitted.insert(next) {
                                emit(next, depth + 1)?;
                            }
                        }
                        ws.next.push(product);
                    }
                }
            }
            std::mem::swap(&mut ws.frontier, &mut ws.next);
            if ws.frontier.is_empty() {
                break;
            }
        }
        ControlFlow::Continue(())
    }
}
/// Distinct endpoint RPQ relation. Unbound starts enumerate a caller-supplied
/// vertex universe (deduplicated once); no allocation by maximum vertex ID.
pub struct PathBindings<'a, G: PatternBackend, M: PathRunner<G> = PathAutomaton> {
    pub graph: &'a G,
    pub machine: &'a M,
    starts: Vec<u32>,
    subject: Term,
    object: Term,
    width: usize,
    hops: HopRange,
    possible: Vec<u64>,
}
pub struct PathBindingWorkspace<W: Default = AutomatonWorkspace> {
    machine: W,
    row: Row,
}
impl<W: Default> Default for PathBindingWorkspace<W> {
    fn default() -> Self {
        Self {
            machine: W::default(),
            row: Row::default(),
        }
    }
}
impl<'a, G: PatternBackend, M: PathRunner<G>> PathBindings<'a, G, M> {
    pub fn new(
        graph: &'a G,
        machine: &'a M,
        mut starts: Vec<u32>,
        subject: Term,
        object: Term,
        width: usize,
        hops: HopRange,
    ) -> Result<Self> {
        for t in [subject, object] {
            if matches!(t,Term::Var(v) if v>=width) {
                return Err(TreeError::InvalidConfig(
                    "path variable outside global schema".into(),
                ));
            }
        }
        starts.sort_unstable();
        starts.dedup();
        let mut possible = vec![0; width.div_ceil(64)];
        for t in [subject, object] {
            if let Term::Var(v) = t {
                possible[v / 64] |= 1u64 << (v % 64);
            }
        }
        Ok(Self {
            graph,
            machine,
            starts,
            subject,
            object,
            width,
            hops,
            possible,
        })
    }
    fn run<T>(
        &self,
        ws: &mut PathBindingWorkspace<M::Workspace>,
        hint: Option<Binding<'_>>,
        emit: &mut impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        let known = |term| match term {
            Term::Const(c) => Some(c),
            Term::Var(v) => hint.and_then(|h| h.get(v)),
        };
        let subject = known(self.subject);
        let target = known(self.object);
        // A lookup hint restricts this relation; it cannot introduce a start
        // outside the independently declared unbound-start universe.
        if matches!(self.subject, Term::Var(_))
            && subject.is_some_and(|s| self.starts.binary_search(&s).is_err())
        {
            return ControlFlow::Continue(());
        }
        let one = [subject.unwrap_or(0)];
        let starts = if subject.is_some() {
            &one[..]
        } else {
            &self.starts
        };
        for &start in starts {
            ws.row.reset(self.width);
            if let Term::Var(v) = self.subject {
                ws.row.set(v, start);
            }
            let result=self.machine.reachable(self.graph,start,self.hops,&mut ws.machine,|end,_| {
                if target.is_some_and(|t|t!=end) || matches!((self.subject,self.object),(Term::Var(a),Term::Var(b)) if a==b && start!=end) {return ControlFlow::Continue(());}
                if let Term::Var(v)=self.object {ws.row.set(v,end);}
                if hint.is_none_or(|h|ws.row.view().compatible(h)) {
                    if let ControlFlow::Break(t)=emit(ws.row.view()) {return ControlFlow::Break(Some(t));}
                    if target.is_some() {return ControlFlow::Break(None);}
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(Some(t)) = result {
                return ControlFlow::Break(t);
            }
        }
        ControlFlow::Continue(())
    }
}
impl<G: PatternBackend, M: PathRunner<G>> Plan for PathBindings<'_, G, M> {
    type Workspace = PathBindingWorkspace<M::Workspace>;
    fn width(&self) -> usize {
        self.width
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        Some(&self.possible)
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.run(ws, None, &mut emit)
    }
    fn visit_matching<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        assert_eq!(hint.values.len(), self.width);
        self.run(ws, Some(hint), &mut emit)
    }
}
#[cfg(test)]
#[path = "path_ops_tests.rs"]
mod tests;
