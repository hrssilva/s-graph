//! Compiled register automata with prefix and whole-atom macro transitions.
//! All schedules share guards, source, binding effects, and result semantics.
use crate::automata_core::{
    And, BindingGuard, BitTest, BitmapAutomaton, FstSource, Mask, TripleSource,
};
use crate::automata_core::{ReversibleAction, Slot};
#[derive(Clone, Copy, Default)]
struct Write {
    variable: usize,
    mask: u32,
    slot_shift: u8,
    value_shift: u8,
}
use crate::graph::{JoinQuery, Term};
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::{Result, TreeError};
use std::ops::ControlFlow;
const ROOT: usize = usize::MAX;
const PROBE_NODES: u64 = 128;
const MAX_FRONTIER_PROBE_STATES: usize = 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Finish each atom before entering the next.
    Nested,
    /// Synchronize this many leading levels, then finish atoms.
    Prefix(usize),
    /// Synchronize every level across all atoms.
    Levels,
    /// Intersect shared-bit supports and advance singleton atom frontiers.
    Frontier,
    /// Choose a schedule with bounded probes; the public default.
    Adaptive,
}
struct Stage {
    guard: And<Mask, BindingGuard>,
    writes: [Write; 3],
    count: u8,
    decode: [u32; 3],
    shifts: [u8; 3],
    remaining: [u8; 3],
    prev: usize,
    level: usize,
}
impl ReversibleAction<Slot, [u32]> for Stage {
    type Undo = ();
    #[inline]
    fn apply(&self, b: &mut [u32], e: &Slot) {
        for w in &self.writes[..self.count as usize] {
            b[w.variable] |= ((e.value >> w.slot_shift) << w.value_shift) & w.mask;
        }
    }
    #[inline]
    fn undo(&self, b: &mut [u32], _: ()) {
        for w in &self.writes[..self.count as usize] {
            b[w.variable] &= !w.mask;
        }
    }
}
#[derive(Clone, Copy)]
struct Output {
    variable: usize,
    mask: u32,
    dim: usize,
}
struct Block {
    first: usize,
    end: usize,
    prev: usize,
    outputs: Vec<Output>,
}
struct Plan {
    stages: Vec<Stage>,
    blocks: Vec<Block>,
    prefix: usize,
    cut: usize,
    order: Vec<usize>,
}
impl Plan {
    fn empty() -> Self {
        Self {
            stages: Vec::new(),
            blocks: Vec::new(),
            prefix: 0,
            cut: 0,
            order: Vec::new(),
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Frame<C: Copy + Default> {
    address: C,
    word: u64,
    remaining: u64,
    child: C,
}
#[derive(Clone, Copy, Default)]
struct CachedMask {
    stamp: u64,
    mask: u64,
}
#[derive(Default)]
pub struct ScheduleWorkspace<C: Copy + Default = u64> {
    frontier: crate::frontier_join::FrontierWorkspace<C>,
    frames: Vec<Frame<C>>,
    bindings: Vec<u32>,
    masks: Vec<CachedMask>,
    stamp: u64,
    parameters: Vec<u64>,
    represented: Vec<u32>,
}
impl<C: Copy + Default> ScheduleWorkspace<C> {
    pub fn allocated_bytes(&self) -> usize {
        self.frontier.allocated_bytes()
            + self.frames.capacity() * std::mem::size_of::<Frame<C>>()
            + self.bindings.capacity() * 4
            + self.masks.capacity() * std::mem::size_of::<CachedMask>()
            + self.parameters.capacity() * 8
            + self.represented.capacity() * 4
    }
}
#[derive(Default, Clone, Debug)]
pub struct ScheduleStats {
    pub nodes: u64,
    pub candidates: u64,
    pub rows: u64,
    pub atomic_entries: u64,
    pub mask_updates: u64,
    pub outer_rows: u64,
    pub terminal_checks: u64,
}
#[derive(Default, Clone, Debug)]
pub struct PlanningStats {
    /// Sum of the hard node limits of attempted probes. Large queries receive
    /// a query-size-aware limit, capped at 32,768 nodes per probe.
    pub node_budget: u64,
    pub nodes: u64,
    pub candidates: u64,
    pub trials: usize,
    pub certified_empty: bool,
    pub selected_cut: usize,
    pub selected_order: Vec<usize>,
    pub selected_frontier: bool,
}
pub(crate) enum Stop<T> {
    User(T),
    Budget,
}
pub struct ScheduledJoin<S: TripleSource> {
    frontier: Option<crate::frontier_join::FrontierPlan>,
    source: S,
    plan: Plan,
    variables: usize,
    empty: bool,
    scalar: bool,
    pub planning: PlanningStats,
}
pub type HybridJoin<'a> = ScheduledJoin<FstSource<'a>>;
impl<'a> HybridJoin<'a> {
    pub fn new<P: Copy + Into<Term>>(
        query: &JoinQuery<P>,
        tree: &'a PathFstDeltaPow2Bfs,
        policy: Policy,
    ) -> Result<Self> {
        Self::with_source(query, FstSource(tree), policy)
    }
}
impl<S: TripleSource> ScheduledJoin<S> {
    pub fn with_source<P: Copy + Into<Term>>(
        query: &JoinQuery<P>,
        source: S,
        policy: Policy,
    ) -> Result<Self> {
        let invalid = |s: &str| TreeError::InvalidConfig(s.into());
        if query.num_vars > query.patterns.len().saturating_mul(3) {
            return Err(invalid("unused variables"));
        }
        let domains = source.domains();
        if source.height() > crate::tree::MAX_HEIGHT
            || !domains.iter().all(|n| (1..=1u64 << 32).contains(n))
        {
            return Err(invalid("invalid triple source domains or height"));
        }
        let mut used = vec![false; query.num_vars];
        let mut zero = vec![0u32; query.num_vars];
        let mut atoms = Vec::new();
        let mut unique = (query.patterns.len() > 64).then(std::collections::BTreeSet::new);
        let mut empty = false;
        let mut scalar = false;
        for p in &query.patterns {
            let a = [p.subject, p.predicate.into(), p.object];
            let mut local = Vec::new();
            for (d, t) in a.iter().enumerate() {
                match *t {
                    Term::Const(c) => empty |= c as u64 >= domains[d],
                    Term::Var(v) => {
                        if v >= query.num_vars {
                            return Err(invalid("variable outside domain"));
                        }
                        used[v] = true;
                        let width = 64 - (domains[d] - 1).leading_zeros();
                        zero[v] |= !((1u64 << width) - 1) as u32;
                        scalar |= local.contains(&v);
                        local.push(v);
                    }
                }
            }
            let fresh = if let Some(unique) = &mut unique {
                unique.insert(a.map(|t| match t {
                    Term::Const(c) => (0u8, c as u64),
                    Term::Var(v) => (1u8, v as u64),
                }))
            } else {
                !atoms.contains(&a)
            };
            if fresh {
                atoms.push(a);
            }
        }
        if used.contains(&false) {
            return Err(invalid("unused variables"));
        }
        // Macro recursion is fast for small BGPs, but its depth grows with
        // atom count. Large compositions use the existing flat register
        // executor with exactly the same compiled schedule and guards.
        scalar |= atoms.len() > 64;
        empty |= !atoms.is_empty() && source.root().is_none();
        let mut order: Vec<_> = (0..atoms.len()).collect();
        let mut bound = vec![false; query.num_vars];
        if atoms.len() > 64 {
            order = large_order(&atoms, query.num_vars);
        } else {
            for i in 0..order.len() {
                let j = (i..order.len())
                    .max_by_key(|&j| {
                        let a = &atoms[order[j]];
                        (
                            a.iter().filter(|t| matches!(t, Term::Const(_))).count(),
                            a.iter()
                                .filter(|t| matches!(t,Term::Var(v) if bound[*v]))
                                .count(),
                            std::cmp::Reverse(j),
                        )
                    })
                    .unwrap();
                order.swap(i, j);
                for t in atoms[order[i]] {
                    if let Term::Var(v) = t {
                        bound[v] = true;
                    }
                }
            }
        }
        let anchored = if policy == Policy::Adaptive {
            atoms.iter().position(|a| {
                matches!(a[0], Term::Const(_))
                    && a.iter().filter(|t| matches!(t, Term::Const(_))).count() >= 2
            })
        } else {
            None
        };
        if let Some(first) = anchored {
            order.retain(|&x| x != first);
            order.insert(0, first);
        }
        let cut = match policy {
            Policy::Prefix(n) => n.min(source.height()),
            Policy::Levels => source.height(),
            Policy::Adaptive if anchored.is_none() && atoms.len() > 1 => source.height(),
            _ => 0,
        };
        let frontier = (policy == Policy::Frontier)
            .then(|| crate::frontier_join::FrontierPlan::compile(&source, &atoms, &zero));
        let plan = if frontier.is_some() {
            Plan {
                stages: Vec::new(),
                blocks: Vec::new(),
                prefix: 0,
                cut: 0,
                order: order.clone(),
            }
        } else {
            compile(&source, &atoms, &order, &zero, cut)
        };
        let mut join = Self {
            frontier,
            source,
            plan,
            variables: query.num_vars,
            empty,
            scalar,
            planning: PlanningStats::default(),
        };
        if policy == Policy::Adaptive && !empty && atoms.len() > 1 && anchored.is_none() {
            // A probe must reach across a large query before lack of rows is
            // meaningful. Still cap each trial independently of graph size.
            let large = atoms.len() > 16;
            let limit = if large {
                (atoms
                    .len()
                    .saturating_mul(join.source.height())
                    .saturating_mul(4) as u64)
                    .clamp(64, 32_768)
            } else {
                64
            };
            let nested_limit = if large { limit } else { PROBE_NODES };
            let (initial, finished) = join.probe(limit);
            if finished && initial.rows == 0 {
                join.empty = true;
                join.planning.certified_empty = true;
            } else if !finished {
                let levels = std::mem::replace(&mut join.plan, Plan::empty());
                let mut best_score = -1f64;
                let mut best_order = order.clone();
                let mut best_stats = ScheduleStats::default();
                let mut best_plan = None;
                let mut explosive = false;
                // Recompiling permutations dominates preparation at thousands
                // of atoms. The incidence heuristic already orders those in
                // O(atoms log atoms); compare schedules on that order.
                let candidates = if large {
                    vec![order.clone()]
                } else {
                    orders(&order)
                };
                for candidate in candidates {
                    join.plan = compile(&join.source, &atoms, &candidate, &zero, 0);
                    let (stats, finished) = join.probe(nested_limit);
                    if finished && stats.rows == 0 {
                        join.empty = true;
                        join.planning.certified_empty = true;
                        break;
                    }
                    let score = if stats.rows > 0 {
                        1e6 * stats.rows as f64 / stats.nodes.max(1) as f64
                    } else {
                        stats.terminal_checks as f64 / stats.nodes.max(1) as f64
                    };
                    let amplified = stats.rows == 0
                        && stats.outer_rows > 0
                        && stats.terminal_checks > 4 * stats.outer_rows;
                    if score > best_score {
                        best_score = score;
                        best_order = candidate;
                        best_stats = stats.clone();
                        best_plan = Some(std::mem::replace(&mut join.plan, Plan::empty()));
                        explosive = amplified;
                    }
                    if explosive {
                        break;
                    }
                }
                if !join.empty {
                    // Compare completed answers per unit of traversal work.
                    // Terminal-check progress still chooses a nested order
                    // when neither schedule has produced any answers.
                    let rate = |s: &ScheduleStats| {
                        s.rows as f64 / (s.nodes as f64 + s.candidates as f64 * 0.125).max(1.)
                    };
                    if initial.rows > 0 && rate(&initial) > rate(&best_stats) * 1.05 {
                        join.plan = levels;
                    } else {
                        join.plan = if explosive {
                            compile(
                                &join.source,
                                &atoms,
                                &best_order,
                                &zero,
                                2.min(join.source.height()),
                            )
                        } else {
                            best_plan.unwrap()
                        };
                    }
                    if initial.rows == 0
                        && best_stats.rows == 0
                        && atoms.len().saturating_mul(join.source.height())
                            <= MAX_FRONTIER_PROBE_STATES
                    {
                        // Shared-bit propagation is a more expensive executor.
                        // Use it only with evidence: a completed bounded probe
                        // while neither cheap schedule has yielded a row.
                        join.frontier = Some(crate::frontier_join::FrontierPlan::compile(
                            &join.source,
                            &atoms,
                            &zero,
                        ));
                        let (frontier_stats, completed) = join.probe(nested_limit);
                        if completed {
                            if frontier_stats.rows == 0 {
                                join.empty = true;
                                join.planning.certified_empty = true;
                            }
                            let order = std::mem::take(&mut join.plan.order);
                            join.plan = Plan {
                                order,
                                ..Plan::empty()
                            };
                        } else {
                            join.frontier = None;
                        }
                    }
                }
            }
        }
        join.planning.selected_cut = join.plan.cut;
        join.planning.selected_order = join.plan.order.clone();
        join.planning.selected_frontier = join.frontier.is_some();
        Ok(join)
    }
    fn probe(&mut self, limit: u64) -> (ScheduleStats, bool) {
        self.planning.node_budget += limit;
        let mut stats = ScheduleStats::default();
        let mut ws = ScheduleWorkspace::default();
        let flow = self.run::<true, true, false, ()>(&mut ws, &mut stats, limit, &mut |_| {
            ControlFlow::Continue(())
        });
        self.planning.nodes += stats.nodes;
        self.planning.candidates += stats.candidates;
        self.planning.trials += 1;
        (stats, matches!(flow, ControlFlow::Continue(())))
    }
    pub fn allocated_bytes(&self) -> usize {
        self.frontier.as_ref().map_or(0, |p| p.allocated_bytes())
            + self.plan.stages.capacity() * std::mem::size_of::<Stage>()
            + self.plan.blocks.capacity() * std::mem::size_of::<Block>()
            + self.plan.order.capacity() * std::mem::size_of::<usize>()
            + self
                .plan
                .blocks
                .iter()
                .map(|b| b.outputs.capacity() * std::mem::size_of::<Output>())
                .sum::<usize>()
            + self.planning.selected_order.capacity() * std::mem::size_of::<usize>()
    }
    pub fn states(&self) -> usize {
        self.frontier
            .as_ref()
            .map_or(self.plan.stages.len(), |p| p.states())
    }
    pub fn visit<T>(
        &self,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        match self.run::<false, false, false, T>(
            ws,
            &mut ScheduleStats::default(),
            u64::MAX,
            &mut emit,
        ) {
            ControlFlow::Continue(()) => ControlFlow::Continue(()),
            ControlFlow::Break(Stop::User(t)) => ControlFlow::Break(t),
            _ => unreachable!(),
        }
    }
    /// Constrain complete variable values without recompiling or changing the
    /// output schema. Parameters restrict first-writer slot bits; normal
    /// register ownership, undo and whole-atom mask caching remain unchanged.
    pub fn visit_matching<T>(
        &self,
        values: &[Option<u32>],
        ws: &mut ScheduleWorkspace<S::Cursor>,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        assert_eq!(values.len(), self.variables);
        if self.frontier.is_some() {
            ws.frontier.parameters.clear();
            ws.frontier.parameters.extend_from_slice(values);
            return match self.run::<false, false, true, T>(
                ws,
                &mut ScheduleStats::default(),
                u64::MAX,
                &mut emit,
            ) {
                ControlFlow::Continue(()) => ControlFlow::Continue(()),
                ControlFlow::Break(Stop::User(t)) => ControlFlow::Break(t),
                _ => unreachable!(),
            };
        }
        if values.iter().all(Option::is_none) {
            return self.visit(ws, emit);
        }
        ws.parameters.resize(self.plan.stages.len(), u64::MAX);
        ws.parameters.fill(u64::MAX);
        // Only bits with a first writer can be nonzero in a valid binding.
        // This also rejects parameters outside unequal dimension widths.
        ws.represented.resize(self.variables, 0);
        ws.represented.fill(0);
        for (i, stage) in self.plan.stages.iter().enumerate() {
            for w in &stage.writes[..stage.count as usize] {
                ws.represented[w.variable] |= w.mask;
                if let Some(value) = values[w.variable] {
                    let mut bits = w.mask;
                    while bits != 0 {
                        let b = bits.trailing_zeros();
                        bits &= bits - 1;
                        let ones = crate::native_join::slot_ones(
                            (b - w.value_shift as u32 + w.slot_shift as u32) as u8,
                        );
                        ws.parameters[i] &= if value & (1u32 << b) != 0 {
                            ones
                        } else {
                            !ones
                        };
                    }
                }
            }
        }
        if values
            .iter()
            .enumerate()
            .any(|(v, x)| x.is_some_and(|x| x & !ws.represented[v] != 0))
        {
            return ControlFlow::Continue(());
        }
        match self.run::<false, false, true, T>(
            ws,
            &mut ScheduleStats::default(),
            u64::MAX,
            &mut emit,
        ) {
            ControlFlow::Continue(()) => ControlFlow::Continue(()),
            ControlFlow::Break(Stop::User(t)) => ControlFlow::Break(t),
            _ => unreachable!(),
        }
    }
    pub fn visit_with_stats<T>(
        &self,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        *stats = ScheduleStats::default();
        match self.run::<true, false, false, T>(ws, stats, u64::MAX, &mut emit) {
            ControlFlow::Continue(()) => ControlFlow::Continue(()),
            ControlFlow::Break(Stop::User(t)) => ControlFlow::Break(t),
            _ => unreachable!(),
        }
    }
    fn run<const TRACK: bool, const LIMIT: bool, const PARAM: bool, T>(
        &self,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<Stop<T>> {
        if self.empty {
            return ControlFlow::Continue(());
        }
        if let Some(plan) = &self.frontier {
            return plan.run::<S, TRACK, LIMIT, PARAM, T>(
                &self.source,
                &mut ws.frontier,
                stats,
                limit,
                emit,
            );
        }
        ws.bindings.resize(self.variables, 0);
        ws.bindings.fill(0);
        if self.plan.stages.is_empty() {
            if self.plan.order.is_empty() {
                if TRACK {
                    stats.rows += 1;
                }
                return emit(&ws.bindings).map_break(Stop::User);
            }
            return ControlFlow::Continue(());
        }
        ws.frames.resize(self.plan.stages.len(), Frame::default());
        ws.masks
            .resize(self.plan.stages.len(), CachedMask::default());
        ws.masks.fill(CachedMask::default());
        ws.stamp = 0;
        let count = if self.scalar {
            self.plan.stages.len()
        } else {
            self.plan.prefix
        };
        if count == 0 {
            return self.block::<TRACK, LIMIT, PARAM, T>(0, ws, stats, limit, emit);
        }
        let mut depth = 0;
        self.open::<TRACK, LIMIT, PARAM, T>(depth, ws, stats, limit)?;
        loop {
            let stage = &self.plan.stages[depth];
            if ws.frames[depth].remaining == 0 {
                if depth == 0 {
                    return ControlFlow::Continue(());
                }
                depth -= 1;
                undo(&self.plan.stages[depth], &mut ws.bindings);
                continue;
            }
            let q = ws.frames[depth].remaining.trailing_zeros();
            ws.frames[depth].remaining &= ws.frames[depth].remaining - 1;
            if TRACK {
                stats.candidates += 1;
            }
            stage.apply(
                &mut ws.bindings[..],
                &Slot {
                    level: stage.level,
                    value: q,
                },
            );
            if stage.level + 1 < self.source.height() {
                let f = &mut ws.frames[depth];
                f.child = self.source.follow(f.address, f.word, q);
            }
            if depth + 1 == count {
                if self.scalar || self.plan.blocks.is_empty() {
                    if TRACK {
                        stats.rows += 1;
                    }
                    emit(&ws.bindings).map_break(Stop::User)?;
                } else {
                    self.block::<TRACK, LIMIT, PARAM, T>(0, ws, stats, limit, emit)?;
                }
                undo(stage, &mut ws.bindings);
            } else {
                depth += 1;
                self.open::<TRACK, LIMIT, PARAM, T>(depth, ws, stats, limit)?;
            }
        }
    }
    #[inline]
    fn open<const TRACK: bool, const LIMIT: bool, const PARAM: bool, T>(
        &self,
        depth: usize,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
    ) -> ControlFlow<Stop<T>> {
        if LIMIT && stats.nodes >= limit {
            return ControlFlow::Break(Stop::Budget);
        }
        let stage = &self.plan.stages[depth];
        let address = if stage.prev == ROOT {
            self.source.root().unwrap()
        } else {
            ws.frames[stage.prev].child
        };
        let word = self.source.word(address);
        let mask = stage.guard.mask(&((), ()), stage.level, &ws.bindings[..]);
        ws.frames[depth] = Frame {
            address,
            word,
            remaining: word
                & mask
                & if PARAM {
                    ws.parameters[depth]
                } else {
                    u64::MAX
                },
            child: S::Cursor::default(),
        };
        if TRACK {
            stats.nodes += 1;
            stats.mask_updates += 1;
        }
        ControlFlow::Continue(())
    }
    fn block<const TRACK: bool, const LIMIT: bool, const PARAM: bool, T>(
        &self,
        index: usize,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<Stop<T>> {
        let block = &self.plan.blocks[index];
        if TRACK {
            stats.atomic_entries += 1;
            if index + 1 == self.plan.blocks.len() {
                stats.terminal_checks += 1;
            }
        }
        let address = if block.prev == ROOT {
            self.source.root().unwrap()
        } else {
            ws.frames[block.prev].child
        };
        ws.stamp = ws.stamp.checked_add(1).expect("mask stamp exhausted");
        let stamp = ws.stamp;
        let mut values = [0; 3];
        for o in &block.outputs {
            values[o.dim] = ws.bindings[o.variable];
        }
        let result = self.descend::<TRACK, LIMIT, PARAM, T>(
            index,
            block.first,
            address,
            values,
            stamp,
            ws,
            stats,
            limit,
            emit,
        );
        // Complete bindings are overwritten at the next leaf. Restore only
        // when returning from this atom, including cancellation/budget exits.
        for o in &block.outputs {
            ws.bindings[o.variable] &= !o.mask;
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn descend<const TRACK: bool, const LIMIT: bool, const PARAM: bool, T>(
        &self,
        index: usize,
        depth: usize,
        address: S::Cursor,
        values: [u32; 3],
        stamp: u64,
        ws: &mut ScheduleWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<Stop<T>> {
        if LIMIT && stats.nodes >= limit {
            return ControlFlow::Break(Stop::Budget);
        }
        let stage = &self.plan.stages[depth];
        let block = &self.plan.blocks[index];
        let word = self.source.word(address);
        if ws.masks[depth].stamp != stamp {
            ws.masks[depth] = CachedMask {
                stamp,
                mask: stage.guard.mask(&((), ()), stage.level, &ws.bindings[..])
                    & if PARAM {
                        ws.parameters[depth]
                    } else {
                        u64::MAX
                    },
            };
            if TRACK {
                stats.mask_updates += 1;
            }
        }
        let mut remaining = word & ws.masks[depth].mask;
        if TRACK {
            stats.nodes += 1;
        }
        while remaining != 0 {
            if TRACK {
                stats.candidates += 1;
            }
            let q = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let mut next = values;
            for d in 0..3 {
                next[d] |= ((q >> stage.shifts[d]) << stage.remaining[d]) & stage.decode[d];
            }
            if depth + 1 == block.end {
                if TRACK && index == 0 {
                    stats.outer_rows += 1;
                }
                for o in &block.outputs {
                    ws.bindings[o.variable] = next[o.dim];
                }
                if index + 1 == self.plan.blocks.len() {
                    if TRACK {
                        stats.rows += 1;
                    }
                    emit(&ws.bindings).map_break(Stop::User)?;
                } else {
                    self.block::<TRACK, LIMIT, PARAM, T>(index + 1, ws, stats, limit, emit)?;
                }
            } else {
                let child = self.source.follow(address, word, q);
                self.descend::<TRACK, LIMIT, PARAM, T>(
                    index,
                    depth + 1,
                    child,
                    next,
                    stamp,
                    ws,
                    stats,
                    limit,
                    emit,
                )?;
            }
        }
        ControlFlow::Continue(())
    }
}
#[inline]
fn undo(s: &Stage, b: &mut [u32]) {
    s.undo(b, ());
}
/// Incremental constants/connectivity heuristic for large BGPs. Each variable
/// becomes bound once; only incident atom priorities change. Original atom
/// ordinal is the deterministic tie break (small plans retain the old order).
fn large_order(atoms: &[[Term; 3]], variables: usize) -> Vec<usize> {
    use std::{cmp::Reverse, collections::BinaryHeap};
    let mut offsets = vec![0; variables + 1];
    let constants: Vec<_> = atoms
        .iter()
        .map(|a| a.iter().filter(|t| matches!(t, Term::Const(_))).count())
        .collect();
    for a in atoms {
        for t in a {
            if let Term::Var(v) = *t {
                offsets[v + 1] += 1;
            }
        }
    }
    for v in 0..variables {
        offsets[v + 1] += offsets[v];
    }
    let mut cursor = offsets.clone();
    let mut incident = vec![0; offsets[variables]];
    for (i, a) in atoms.iter().enumerate() {
        for t in a {
            if let Term::Var(v) = *t {
                incident[cursor[v]] = i;
                cursor[v] += 1;
            }
        }
    }
    let mut heap: BinaryHeap<_> = (0..atoms.len())
        .map(|i| (constants[i], 0, Reverse(i)))
        .collect();
    let mut bound = vec![false; variables];
    let mut chosen = vec![false; atoms.len()];
    let mut shared = vec![0; atoms.len()];
    let mut order = Vec::with_capacity(atoms.len());
    while let Some((_, score, Reverse(i))) = heap.pop() {
        if chosen[i] || shared[i] != score {
            continue;
        }
        chosen[i] = true;
        order.push(i);
        for t in &atoms[i] {
            if let Term::Var(v) = *t {
                if !bound[v] {
                    bound[v] = true;
                    for &a in &incident[offsets[v]..offsets[v + 1]] {
                        if !chosen[a] {
                            shared[a] += 1;
                            heap.push((constants[a], shared[a], Reverse(a)));
                        }
                    }
                }
            }
        }
    }
    order
}
fn orders(first: &[usize]) -> Vec<Vec<usize>> {
    let mut result = vec![first.to_vec()];
    // At most six orders; compilation/probing work stays bounded for large BGPs.
    if first.len() <= 3 {
        fn permute(a: &mut [usize], i: usize, out: &mut Vec<Vec<usize>>) {
            if i == a.len() {
                if !out.iter().any(|x| x == a) {
                    out.push(a.to_vec());
                }
                return;
            }
            for j in i..a.len() {
                a.swap(i, j);
                permute(a, i + 1, out);
                a.swap(i, j);
            }
        }
        permute(&mut first.to_vec(), 0, &mut result);
    } else {
        for i in 1..first.len().min(6) {
            let mut a = first.to_vec();
            a.swap(0, i);
            result.push(a);
        }
    }
    result
}
fn compile<S: TripleSource>(
    source: &S,
    atoms: &[[Term; 3]],
    order: &[usize],
    zero: &[u32],
    cut: usize,
) -> Plan {
    let height = source.height();
    let mut events = Vec::new();
    for l in 0..cut {
        for &a in order {
            events.push((l, a));
        }
    }
    let prefix = events.len();
    for &a in order {
        for l in cut..height {
            events.push((l, a));
        }
    }
    let mut stages = Vec::with_capacity(events.len());
    let mut known = zero.to_vec();
    let mut prev = vec![ROOT; atoms.len()];
    for (level, a) in events {
        let (bits, shifts, remaining) = source.layout(level);
        let slots = 1u32 << bits.iter().sum::<u8>();
        let mut mask = if slots == 64 {
            u64::MAX
        } else {
            (1u64 << slots) - 1
        };
        let mut guard = BindingGuard {
            tests: [BitTest::default(); 6],
            count: 0,
        };
        let mut local = [(usize::MAX, 0u32, 0u64); 6];
        let mut n = 0;
        let mut writes = [Write::default(); 3];
        let mut count = 0;
        let mut decode = [0; 3];
        for d in 0..3 {
            let mut owned = 0;
            for b in 0..bits[d] {
                let bit = 1u32 << (remaining[d] + b);
                let ones = crate::native_join::slot_ones(shifts[d] + b);
                match atoms[a][d] {
                    Term::Const(c) => mask &= if c & bit != 0 { ones } else { !ones },
                    Term::Var(v) => {
                        if zero[v] & bit != 0 {
                            mask &= !ones;
                        } else if let Some(&(_, _, old)) = local[..n]
                            .iter()
                            .find(|&&(var, value, _)| var == v && value == bit)
                        {
                            mask &= !(old ^ ones);
                        } else if known[v] & bit != 0 {
                            guard.tests[guard.count as usize] = BitTest {
                                variable: v,
                                bit,
                                ones,
                            };
                            guard.count += 1;
                        } else {
                            known[v] |= bit;
                            local[n] = (v, bit, ones);
                            n += 1;
                            owned |= bit;
                        }
                    }
                }
            }
            if owned != 0 {
                let Term::Var(variable) = atoms[a][d] else {
                    unreachable!()
                };
                writes[count] = Write {
                    variable,
                    mask: owned,
                    slot_shift: shifts[d],
                    value_shift: remaining[d],
                };
                count += 1;
                decode[d] = owned;
            }
        }
        stages.push(Stage {
            guard: And(Mask(mask), guard),
            writes,
            count: count as u8,
            decode,
            shifts,
            remaining,
            prev: prev[a],
            level,
        });
        prev[a] = stages.len() - 1;
    }
    let mut blocks = Vec::new();
    if cut < height {
        for (i, &a) in order.iter().enumerate() {
            let first = prefix + i * (height - cut);
            let end = first + height - cut;
            let mut owned = [0u32; 3];
            for s in &stages[first..end] {
                for d in 0..3 {
                    owned[d] |= s.decode[d];
                }
            }
            let outputs = (0..3)
                .filter(|&d| owned[d] != 0)
                .map(|d| {
                    let Term::Var(variable) = atoms[a][d] else {
                        unreachable!()
                    };
                    Output {
                        variable,
                        mask: owned[d],
                        dim: d,
                    }
                })
                .collect();
            blocks.push(Block {
                first,
                end,
                prev: stages[first].prev,
                outputs,
            });
        }
    }
    Plan {
        stages,
        blocks,
        prefix,
        cut,
        order: order.to_vec(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fst_join::{FstJoin, FstJoinStats, FstJoinWorkspace};
    pub(crate) fn check(tree: &PathFstDeltaPow2Bfs, q: &JoinQuery<Term>, oracle: &[Vec<u32>]) {
        for policy in [
            Policy::Nested,
            Policy::Prefix(1),
            Policy::Prefix(2),
            Policy::Levels,
            Policy::Frontier,
            Policy::Adaptive,
        ] {
            let p = HybridJoin::new(q, tree, policy).unwrap();
            assert!(p.planning.nodes <= p.planning.node_budget);
            if p.planning.certified_empty {
                assert!(oracle.is_empty());
            }
            let mut ws = ScheduleWorkspace::default();
            let mut rows = Vec::new();
            let mut st = ScheduleStats::default();
            let _: ControlFlow<()> = p.visit_with_stats(&mut ws, &mut st, |r| {
                rows.push(r.to_vec());
                ControlFlow::Continue(())
            });
            rows.sort_unstable();
            assert_eq!(rows, oracle, "policy={policy:?}");
            if policy == Policy::Levels {
                let f = FstJoin::new(q, tree).unwrap();
                let mut fs = FstJoinStats::default();
                let _: ControlFlow<()> =
                    f.visit_with_stats(&mut FstJoinWorkspace::default(), &mut fs, |_| {
                        ControlFlow::Continue(())
                    });
                assert_eq!(st.nodes, fs.node_visits);
                assert_eq!(st.candidates, fs.candidates);
            }
            if !oracle.is_empty() {
                assert_eq!(
                    p.visit(&mut ws, |_| ControlFlow::Break(12)),
                    ControlFlow::Break(12)
                );
            }
            let mut count = 0;
            let _: ControlFlow<()> = p.visit(&mut ws, |_| {
                count += 1;
                ControlFlow::Continue(())
            });
            assert_eq!(count, oracle.len());
        }
    }
    #[test]
    fn multi_level_partial_bindings_and_wide_ids() {
        use crate::graph::TriplePattern;
        use Term::{Const as C, Var as V};
        let triples = vec![(u32::MAX, 0, 1), (1, 0, u32::MAX), (0, 1, 0)];
        let t = PathFstDeltaPow2Bfs::build(
            triples,
            [1u64 << 32, 2, 1u64 << 32],
            crate::DEFAULT_BLOCK_CAPACITY,
        );
        let a = |s, p, o| TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        };
        check(
            &t,
            &JoinQuery {
                num_vars: 2,
                patterns: vec![a(V(0), C(0), V(1)), a(V(1), C(0), V(0))],
            },
            &[vec![1, u32::MAX], vec![u32::MAX, 1]],
        );
        for q in [
            JoinQuery {
                num_vars: 1,
                patterns: vec![],
            },
            JoinQuery {
                num_vars: 1,
                patterns: vec![a(V(1), C(0), C(0))],
            },
        ] {
            assert!(HybridJoin::new(&q, &t, Policy::Adaptive).is_err());
        }
    }
    #[test]
    fn many_atoms_and_deep_asymmetric_domains() {
        use crate::graph::TriplePattern;
        use Term::{Const as C, Var as V};
        let a = |s, p, o| TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        };
        let tree =
            PathFstDeltaPow2Bfs::build(vec![(1, 0, 1)], [3, 1, 3], crate::DEFAULT_BLOCK_CAPACITY);
        let q = JoinQuery {
            num_vars: 72,
            patterns: (0..72).map(|v| a(C(1), C(0), V(v))).collect(),
        };
        check(&tree, &q, &[vec![1; 72]]);
        let triples: Vec<_> = (0..33)
            .flat_map(|s| {
                (0..5).flat_map(move |p| {
                    (0..129)
                        .filter(move |o| (s * 17 + p * 11 + o) % 31 == 0)
                        .map(move |o| (s, p, o))
                })
            })
            .collect();
        let set: std::collections::HashSet<_> = triples.iter().copied().collect();
        let tree = PathFstDeltaPow2Bfs::build(triples, [33, 5, 129], crate::DEFAULT_BLOCK_CAPACITY);
        let q = JoinQuery {
            num_vars: 2,
            patterns: vec![a(V(0), C(1), V(1)), a(V(1), C(2), V(0))],
        };
        let mut oracle = Vec::new();
        for x in 0..129 {
            for y in 0..129 {
                if set.contains(&(x, 1, y)) && set.contains(&(y, 2, x)) {
                    oracle.push(vec![x, y]);
                }
            }
        }
        check(&tree, &q, &oracle);
        let q = JoinQuery {
            num_vars: 2,
            patterns: vec![a(V(0), V(1), V(0)), a(C(1), V(1), V(0))],
        };
        let mut oracle = Vec::new();
        for x in 0..129 {
            for p in 0..5 {
                if set.contains(&(x, p, x)) && set.contains(&(1, p, x)) {
                    oracle.push(vec![x, p]);
                }
            }
        }
        check(&tree, &q, &oracle);
    }
    #[derive(Clone, Copy)]
    struct OneWord(u64);
    impl crate::automata_core::BitmapSource for OneWord {
        type Cursor = usize;
        fn root(&self) -> Option<usize> {
            Some(0)
        }
        fn height(&self) -> usize {
            1
        }
        fn word(&self, _: usize) -> u64 {
            self.0
        }
        fn follow(&self, _: usize, _: u64, _: u32) -> usize {
            unreachable!()
        }
    }
    impl TripleSource for OneWord {
        fn domains(&self) -> [u64; 3] {
            [2, 2, 2]
        }
        fn layout(&self, _: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
            ([1, 1, 1], [2, 1, 0], [0, 0, 0])
        }
    }
    #[test]
    fn independent_storage_backend_and_shared_predicates() {
        use crate::graph::TriplePattern;
        use Term::{Const as C, Var as V};
        let a = |s, p, o| TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        };
        let q = JoinQuery {
            num_vars: 2,
            patterns: vec![a(C(0), V(0), V(1)), a(V(1), V(0), C(0))],
        };
        // Tuples (0,0,1), (1,0,0), (0,1,0).
        for policy in [
            Policy::Nested,
            Policy::Prefix(1),
            Policy::Levels,
            Policy::Frontier,
            Policy::Adaptive,
        ] {
            let p = ScheduledJoin::with_source(&q, OneWord((1 << 1) | (1 << 4) | (1 << 2)), policy)
                .unwrap();
            let mut rows = Vec::new();
            let _: ControlFlow<()> = p.visit(&mut ScheduleWorkspace::default(), |r| {
                rows.push(r.to_vec());
                ControlFlow::Continue(())
            });
            rows.sort_unstable();
            assert_eq!(rows, vec![vec![0, 1], vec![1, 0]]);
        }
    }
}
