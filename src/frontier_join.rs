//! Shared-bit support propagation over the live frontier of a product automaton.
//! Cells retain packed transitions; no graph relation or query DFA is expanded.
use crate::automata_core::{BitSupport, TripleSource};
#[cfg(test)]
use crate::execution::Uncontrolled;
use crate::execution::{Check, StopReason};
use crate::graph::Term;
use crate::scheduled_join::{ScheduleStats, Stop};
use std::ops::ControlFlow;

#[derive(Clone, Copy, Default)]
struct Bit {
    variable: usize,
    value: u32,
    key: usize,
    ones: u64,
}

struct Spec {
    mask: u64,
    bits: [Bit; 6],
    count: usize,
}
#[derive(Clone, Copy)]
struct Occurrence {
    atom: usize,
    level: usize,
}
pub(crate) struct FrontierPlan {
    specs: Vec<Spec>,
    offsets: Vec<usize>,
    incident: Vec<Occurrence>,
    zero: Vec<u32>,
    height: usize,
    atoms: usize,
}
#[derive(Clone, Copy, Default)]
struct Cell<C> {
    cursor: C,
    word: u64,
    allowed: u64,
    level: usize,
}
enum Undo<C> {
    Cell(usize, Cell<C>),
    Bit(Bit),
}
struct Choice {
    bit: Bit,
    checkpoint: usize,
    second: bool,
}
#[derive(Default)]
struct Ready {
    words: Vec<u64>,
    summary: Vec<u64>,
}
impl Ready {
    fn reset(&mut self, n: usize) {
        self.words.resize(n.div_ceil(64), 0);
        self.words.fill(0);
        self.summary.resize(self.words.len().div_ceil(64), 0);
        self.summary.fill(0);
    }
    fn set(&mut self, key: usize, live: bool) {
        let word = key / 64;
        let bit = 1u64 << (key % 64);
        if live {
            self.words[word] |= bit;
        } else {
            self.words[word] &= !bit;
        }
        if self.words[word] != 0 {
            self.summary[word / 64] |= 1u64 << (word % 64);
        } else {
            self.summary[word / 64] &= !(1u64 << (word % 64));
        }
    }
    fn first(&self) -> Option<usize> {
        for (i, &word) in self.summary.iter().enumerate() {
            if word != 0 {
                let j = i * 64 + word.trailing_zeros() as usize;
                return Some(j * 64 + self.words[j].trailing_zeros() as usize);
            }
        }
        None
    }
}
pub(crate) struct FrontierWorkspace<C: Copy + Default> {
    cells: Vec<Cell<C>>,
    values: Vec<u32>,
    known: Vec<u32>,
    references: Vec<u32>,
    ready: Ready,
    queue: Vec<usize>,
    queued: Vec<bool>,
    head: usize,
    completed: usize,
    trail: Vec<Undo<C>>,
    choices: Vec<Choice>,
    pub(crate) parameters: Vec<Option<u32>>,
}
impl<C: Copy + Default> Default for FrontierWorkspace<C> {
    fn default() -> Self {
        Self {
            cells: Vec::new(),
            values: Vec::new(),
            known: Vec::new(),
            references: Vec::new(),
            ready: Ready::default(),
            queue: Vec::new(),
            queued: Vec::new(),
            head: 0,
            completed: 0,
            trail: Vec::new(),
            choices: Vec::new(),
            parameters: Vec::new(),
        }
    }
}
impl<C: Copy + Default> FrontierWorkspace<C> {
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.cells.capacity() * std::mem::size_of::<Cell<C>>()
            + (self.values.capacity() + self.known.capacity() + self.references.capacity()) * 4
            + (self.ready.words.capacity() + self.ready.summary.capacity()) * 8
            + self.queue.capacity() * std::mem::size_of::<usize>()
            + self.queued.capacity()
            + self.trail.capacity() * std::mem::size_of::<Undo<C>>()
            + self.choices.capacity() * std::mem::size_of::<Choice>()
            + self.parameters.capacity() * std::mem::size_of::<Option<u32>>()
    }
    fn enqueue(&mut self, atom: usize) {
        if !self.queued[atom] {
            self.queued[atom] = true;
            self.queue.push(atom);
        }
    }
    fn clear_queue(&mut self) {
        for &a in &self.queue[self.head..] {
            self.queued[a] = false;
        }
        self.queue.clear();
        self.head = 0;
    }
}
impl FrontierPlan {
    #[cfg(test)]
    pub(crate) fn compile<S: TripleSource>(source: &S, atoms: &[[Term; 3]], zero: &[u32]) -> Self {
        Self::compile_checked(source, atoms, zero, &Uncontrolled).unwrap()
    }
    pub(crate) fn compile_checked<S: TripleSource>(
        source: &S,
        atoms: &[[Term; 3]],
        zero: &[u32],
        control: &impl Check,
    ) -> std::result::Result<Self, StopReason> {
        let height = source.height();
        let variables = zero.len();
        let mut specs = Vec::with_capacity(atoms.len() * height);
        let mut offsets = vec![0; variables * 32 + 1];
        for atom in atoms {
            for level in 0..height {
                control.checkpoint()?;
                let (widths, shifts, remaining) = source.layout(level);
                let slots = 1u32 << widths.iter().sum::<u8>();
                let mut spec = Spec {
                    mask: if slots == 64 {
                        u64::MAX
                    } else {
                        (1u64 << slots) - 1
                    },
                    bits: [Bit::default(); 6],
                    count: 0,
                };
                for d in 0..3 {
                    for b in 0..widths[d] {
                        let position = remaining[d] + b;
                        let value = 1u32 << position;
                        let ones = crate::native_join::slot_ones(shifts[d] + b);
                        match atom[d] {
                            Term::Const(x) => {
                                spec.mask &= if x & value == 0 { !ones } else { ones }
                            }
                            Term::Var(v) if zero[v] & value != 0 => spec.mask &= !ones,
                            Term::Var(v) => {
                                if let Some(old) = spec.bits[..spec.count]
                                    .iter()
                                    .find(|x| x.variable == v && x.value == value)
                                {
                                    spec.mask &= !(old.ones ^ ones);
                                } else {
                                    let key = v * 32 + (31 - position as usize);
                                    spec.bits[spec.count] = Bit {
                                        variable: v,
                                        value,
                                        key,
                                        ones,
                                    };
                                    spec.count += 1;
                                    offsets[key + 1] += 1;
                                }
                            }
                        }
                    }
                }
                specs.push(spec);
            }
        }
        for i in 1..offsets.len() {
            control.checkpoint()?;
            offsets[i] += offsets[i - 1];
        }
        let mut positions = offsets.clone();
        let mut incident = vec![Occurrence { atom: 0, level: 0 }; offsets[variables * 32]];
        for (i, spec) in specs.iter().enumerate() {
            control.checkpoint()?;
            for bit in &spec.bits[..spec.count] {
                incident[positions[bit.key]] = Occurrence {
                    atom: i / height,
                    level: i % height,
                };
                positions[bit.key] += 1;
            }
        }
        Ok(Self {
            specs,
            offsets,
            incident,
            zero: zero.to_vec(),
            height,
            atoms: atoms.len(),
        })
    }
    pub(crate) fn states(&self) -> usize {
        self.specs.len()
    }
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.specs.capacity() * std::mem::size_of::<Spec>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.incident.capacity() * std::mem::size_of::<Occurrence>()
            + self.zero.capacity() * 4
    }
    fn reference<C: Copy + Default>(
        &self,
        ws: &mut FrontierWorkspace<C>,
        cell: Cell<C>,
        atom: usize,
        add: bool,
    ) {
        if cell.level >= self.height {
            return;
        }
        let spec = &self.specs[atom * self.height + cell.level];
        for bit in &spec.bits[..spec.count] {
            let n = &mut ws.references[bit.key];
            if add {
                *n += 1;
            } else {
                *n -= 1;
            }
            ws.ready
                .set(bit.key, *n != 0 && ws.known[bit.variable] & bit.value == 0);
        }
    }
    fn replace<C: Copy + Default>(
        &self,
        ws: &mut FrontierWorkspace<C>,
        atom: usize,
        next: Cell<C>,
        trail: bool,
    ) {
        let old = ws.cells[atom];
        if trail {
            ws.trail.push(Undo::Cell(atom, old));
        }
        if old.level != next.level {
            self.reference(ws, old, atom, false);
            self.reference(ws, next, atom, true);
            if old.level == self.height {
                ws.completed -= 1;
            }
            if next.level == self.height {
                ws.completed += 1;
            }
        }
        ws.cells[atom] = next;
    }
    fn bind<C: Copy + Default, T>(
        &self,
        ws: &mut FrontierWorkspace<C>,
        bit: Bit,
        one: bool,
        origin: usize,
        control: &impl Check,
    ) -> ControlFlow<Stop<T>> {
        debug_assert_eq!(ws.known[bit.variable] & bit.value, 0);
        ws.trail.push(Undo::Bit(bit));
        ws.known[bit.variable] |= bit.value;
        if one {
            ws.values[bit.variable] |= bit.value;
        }
        ws.ready.set(bit.key, false);
        for occurrence in &self.incident[self.offsets[bit.key]..self.offsets[bit.key + 1]] {
            control.step()?;
            if occurrence.atom != origin && ws.cells[occurrence.atom].level == occurrence.level {
                ws.enqueue(occurrence.atom);
            }
        }
        ControlFlow::Continue(())
    }
    fn restore<C: Copy + Default, T>(
        &self,
        ws: &mut FrontierWorkspace<C>,
        checkpoint: usize,
        control: &impl Check,
    ) -> ControlFlow<Stop<T>> {
        ws.clear_queue();
        while ws.trail.len() > checkpoint {
            control.step()?;
            match ws.trail.pop().unwrap() {
                Undo::Cell(atom, old) => self.replace(ws, atom, old, false),
                Undo::Bit(bit) => {
                    ws.values[bit.variable] &= !bit.value;
                    ws.known[bit.variable] &= !bit.value;
                    ws.ready.set(bit.key, ws.references[bit.key] != 0);
                }
            }
        }
        ControlFlow::Continue(())
    }
    fn propagate<S: TripleSource, const TRACK: bool, const LIMIT: bool, T>(
        &self,
        source: &S,
        ws: &mut FrontierWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        control: &impl Check,
    ) -> ControlFlow<Stop<T>, bool> {
        while ws.head < ws.queue.len() {
            control.step()?;
            // Cached frontier words can support many branches without another
            // storage read (especially a one-level source). Bound propagation
            // work as well as node reads so a planning probe cannot enumerate
            // a large Cartesian product inside its read budget.
            if LIMIT
                && stats.mask_updates.saturating_add(stats.candidates) >= limit.saturating_mul(8)
            {
                return ControlFlow::Break(Stop::Budget);
            }
            let atom = ws.queue[ws.head];
            ws.head += 1;
            ws.queued[atom] = false;
            let mut cell = ws.cells[atom];
            if cell.level == self.height {
                continue;
            }
            if TRACK {
                stats.mask_updates += 1;
            }
            let spec = &self.specs[atom * self.height + cell.level];
            let mut allowed = cell.allowed;
            for bit in &spec.bits[..spec.count] {
                if ws.known[bit.variable] & bit.value != 0 {
                    allowed &= if ws.values[bit.variable] & bit.value == 0 {
                        !bit.ones
                    } else {
                        bit.ones
                    };
                }
            }
            if allowed == 0 {
                return ControlFlow::Continue(false);
            }
            if cell.allowed != allowed {
                cell.allowed = allowed;
                self.replace(ws, atom, cell, true);
            }
            for bit in &spec.bits[..spec.count] {
                if ws.known[bit.variable] & bit.value == 0 {
                    if let Some(one) = BitSupport::from_mask(allowed, bit.ones).singleton() {
                        self.bind(ws, *bit, one, atom, control)?;
                    }
                }
            }
            if allowed.is_power_of_two() {
                let slot = allowed.trailing_zeros();
                let level = cell.level + 1;
                if level == self.height {
                    self.replace(ws, atom, Cell { level, ..cell }, true);
                } else {
                    if LIMIT && stats.nodes >= limit {
                        return ControlFlow::Break(Stop::Budget);
                    }
                    let cursor = source.follow(cell.cursor, cell.word, slot);
                    let word = source.word(cursor);
                    if TRACK {
                        stats.nodes += 1;
                    }
                    let next = &self.specs[atom * self.height + level];
                    self.replace(
                        ws,
                        atom,
                        Cell {
                            cursor,
                            word,
                            allowed: word & next.mask,
                            level,
                        },
                        true,
                    );
                    ws.enqueue(atom);
                }
            }
        }
        ws.clear_queue();
        ControlFlow::Continue(true)
    }
    #[cfg(test)]
    pub(crate) fn run<
        S: TripleSource,
        const TRACK: bool,
        const LIMIT: bool,
        const PARAM: bool,
        T,
    >(
        &self,
        source: &S,
        ws: &mut FrontierWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<Stop<T>> {
        self.run_checked::<S, TRACK, LIMIT, PARAM, T>(source, ws, stats, limit, &Uncontrolled, emit)
    }
    pub(crate) fn run_checked<
        S: TripleSource,
        const TRACK: bool,
        const LIMIT: bool,
        const PARAM: bool,
        T,
    >(
        &self,
        source: &S,
        ws: &mut FrontierWorkspace<S::Cursor>,
        stats: &mut ScheduleStats,
        limit: u64,
        control: &impl Check,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<Stop<T>> {
        control.step()?;
        let variables = self.zero.len();
        ws.values.resize(variables, 0);
        ws.values.fill(0);
        ws.known.clone_from(&self.zero);
        if PARAM {
            for (v, parameter) in ws.parameters.iter().enumerate() {
                if let Some(x) = *parameter {
                    if x & self.zero[v] != 0 {
                        return ControlFlow::Continue(());
                    }
                    ws.values[v] = x;
                    ws.known[v] = u32::MAX;
                }
            }
        }
        ws.references.resize(variables * 32, 0);
        ws.references.fill(0);
        ws.ready.reset(variables * 32);
        ws.cells.resize(self.atoms, Cell::default());
        ws.queued.resize(self.atoms, false);
        ws.queued.fill(false);
        ws.queue.clear();
        ws.head = 0;
        ws.completed = 0;
        ws.trail.clear();
        ws.choices.clear();
        if self.atoms != 0 {
            let Some(root) = source.root() else {
                return ControlFlow::Continue(());
            };
            for atom in 0..self.atoms {
                control.step()?;
                if LIMIT && stats.nodes >= limit {
                    return ControlFlow::Break(Stop::Budget);
                }
                let word = source.word(root);
                if TRACK {
                    stats.nodes += 1;
                }
                let cell = Cell {
                    cursor: root,
                    word,
                    allowed: word & self.specs[atom * self.height].mask,
                    level: 0,
                };
                ws.cells[atom] = cell;
                self.reference(ws, cell, atom, true);
                ws.enqueue(atom);
            }
        }
        let mut consistent =
            self.propagate::<S, TRACK, LIMIT, T>(source, ws, stats, limit, control)?;
        // The initial fixed point is common to every branch and need not stay
        // on the undo trail. A later run resets the workspace, including after
        // callback cancellation or an exhausted diagnostic budget.
        ws.trail.clear();
        loop {
            control.step()?;
            if consistent {
                if ws.completed == self.atoms {
                    if TRACK {
                        stats.rows += 1;
                    }
                    emit(&ws.values).map_break(Stop::User)?;
                } else {
                    let key = ws
                        .ready
                        .first()
                        .expect("nonterminal frontier has an unbound bit");
                    let variable = key / 32;
                    let value = 1u32 << (31 - key % 32);
                    let bit = Bit {
                        key,
                        variable,
                        value,
                        ones: 0,
                    };
                    ws.choices.push(Choice {
                        bit,
                        checkpoint: ws.trail.len(),
                        second: true,
                    });
                    if TRACK {
                        stats.candidates += 1;
                    }
                    self.bind(ws, bit, false, usize::MAX, control)?;
                    consistent =
                        self.propagate::<S, TRACK, LIMIT, T>(source, ws, stats, limit, control)?;
                    continue;
                }
            }
            loop {
                control.step()?;
                let Some(choice) = ws.choices.last() else {
                    return ControlFlow::Continue(());
                };
                let (bit, checkpoint, second) = (choice.bit, choice.checkpoint, choice.second);
                self.restore(ws, checkpoint, control)?;
                if second {
                    ws.choices.last_mut().unwrap().second = false;
                    if TRACK {
                        stats.candidates += 1;
                    }
                    self.bind(ws, bit, true, usize::MAX, control)?;
                    consistent =
                        self.propagate::<S, TRACK, LIMIT, T>(source, ws, stats, limit, control)?;
                    break;
                }
                ws.choices.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{JoinQuery, Term, TriplePattern};
    use crate::scheduled_join::{HybridJoin, Policy, ScheduleStats, ScheduleWorkspace};
    use crate::{DEFAULT_BLOCK_CAPACITY, Fst};
    use std::ops::ControlFlow;
    #[test]
    fn probe_budget_also_bounds_cached_leaf_cartesian_products() {
        use super::{FrontierPlan, FrontierWorkspace};
        use crate::automata_core::FstSource;
        use crate::scheduled_join::Stop;
        let graph = Fst::build(
            vec![(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
            [2, 1, 2],
            DEFAULT_BLOCK_CAPACITY,
        );
        let atoms = (0..8)
            .map(|i| [Term::Var(i * 2), Term::Const(0), Term::Var(i * 2 + 1)])
            .collect::<Vec<_>>();
        let plan = FrontierPlan::compile(&FstSource(&graph), &atoms, &vec![!1u32; 16]);
        let mut stats = ScheduleStats::default();
        let mut ws = FrontierWorkspace::default();
        let result = plan.run::<_, true, true, false, ()>(
            &FstSource(&graph),
            &mut ws,
            &mut stats,
            128,
            &mut |_| ControlFlow::Continue(()),
        );
        assert!(matches!(result, ControlFlow::Break(Stop::Budget)));
        assert_eq!(stats.nodes, 8); // Reusing the eight leaf words cannot evade the work cap.
        assert!(stats.rows < 65_536);
    }
    #[test]
    fn large_dense_cycles_prune_before_free_bits_and_handle_digit_alignment() {
        for (domain, frontier_expected) in [(257, true), (1024, false)] {
            let mut triples = Vec::new();
            for a in 0..128 {
                for b in 128..256 {
                    triples.extend([(a, 0, b), (b, 0, a)]);
                }
            }
            triples.push((256, 0, 256));
            let graph = Fst::build(triples, [domain, 1, domain], DEFAULT_BLOCK_CAPACITY);
            let query = JoinQuery {
                num_vars: 129,
                patterns: (0..129)
                    .map(|i| TriplePattern {
                        subject: Term::Var(i),
                        predicate: Term::Const(0),
                        object: Term::Var((i + 1) % 129),
                    })
                    .collect(),
            };
            for policy in [Policy::Frontier, Policy::Adaptive] {
                let plan = HybridJoin::new(&query, &graph, policy).unwrap();
                if policy == Policy::Adaptive {
                    assert_eq!(plan.planning.selected_frontier, frontier_expected);
                }
                let mut stats = ScheduleStats::default();
                let mut ws = ScheduleWorkspace::default();
                let mut rows = Vec::new();
                let _: ControlFlow<()> = plan.visit_with_stats(&mut ws, &mut stats, |r| {
                    rows.push(r.to_vec());
                    ControlFlow::Continue(())
                });
                assert_eq!(rows, vec![vec![256; 129]]);
                assert!(stats.nodes < 1024);
                assert_eq!(
                    plan.visit(&mut ws, |_| ControlFlow::Break(3)),
                    ControlFlow::Break(3)
                );
                let mut hint = vec![None; 129];
                hint[3] = Some(0);
                let _: ControlFlow<()> =
                    plan.visit_matching(&hint, &mut ws, |_| panic!("impossible pinned cycle"));
                hint[3] = Some(256);
                let mut count = 0;
                let _: ControlFlow<()> = plan.visit_matching(&hint, &mut ws, |_| {
                    count += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(count, 1);
            }
        }
    }
}
