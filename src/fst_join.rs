//! Conjunctive joins directly over `PathFstDeltaPow2Bfs`'s delta-addressed
//! storage -- see `DESIGN.md`'s "Joins and paths as automatons" entry for
//! the design discussion this is "Phase 2" of.
//!
//! Level-synchronized alternative retained for comparison with the default
//! [`crate::Join`] adaptive automaton. Repeated variables are supported.
//! The design history below describes this executor's fixed schedule.
//!
//! **Not a literal port of `native_join::NativeJoin`, and deliberately not
//! expressed through `level_automaton::LevelAutomaton` either.** Both were
//! considered and rejected for the same reason: `NativeJoin`'s real
//! algorithm is a flat backtracking search over `height * atoms` stages,
//! grouped *by level first, then by atom* -- every atom picks its own
//! candidate at level 0 before any atom moves to level 1. That
//! interleaving is what makes it Leapfrog-Triejoin-shaped (Veldhuizen,
//! ICDT 2014: advance one variable at a time, sharing what's already
//! known, never materializing a full per-atom result before starting the
//! next atom) instead of a naive nested-loop join. A `LevelAutomaton`-
//! per-atom design -- fully resolve atom 0's own top-to-bottom walk,
//! *then* start atom 1's -- would have been easy to express through the
//! walk already built in `level_automaton.rs`, but it throws away exactly
//! that interleaved pruning, which this tree's schedule supports every
//! bit as well as `Heap`'s does (same per-level `dim_bits`/`shifts` shape
//! for every atom, since every atom queries the same physical tree).
//! Chose fidelity to the proven algorithm over fitting the existing
//! trait.
//!
//! What *is* reused, because it's genuinely backend-agnostic: the
//! `Stage`/`Check`/`Write` compiled-plan shape and the atom-reordering
//! heuristic (favor constants, then connectivity to already-selected
//! atoms) from `NativeJoin::new` -- pure query-compilation logic that
//! never touches `PrefixCursor` or any other Heap-specific type. What's
//! genuinely rewritten, because it's the actual backend-specific part:
//! `open`/`run`'s node access. `Heap`'s `PrefixCursor` carries a
//! precomputed "unskipped occupancy" word; this tree instead reads
//! `word`/`real_index`/`delta` directly, the same primitives every other
//! traversal in `path_fst.rs` uses -- and because a stage's own `level`
//! is known at compile time, a leaf-level stage (`level == height - 1`)
//! skips the delta read and child-address computation entirely, since no
//! stage for *this* atom ever reads that child position again. `NativeJoin`
//! has no equivalent of this -- not because `Heap` couldn't support it,
//! but because `PrefixCursor`'s own abstraction doesn't expose the
//! distinction as directly as this tree's own leaf/non-leaf split already
//! does everywhere else in this file.

use crate::graph::{JoinQuery, Term};
use crate::native_join::slot_ones;
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::NUM_DIMS;
use crate::{Result, TreeError};
use std::ops::ControlFlow;

#[derive(Clone, Copy, Default)]
pub(crate) struct Check {
    pub variable: usize,
    pub bit: u32,
    pub ones: u64,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Write {
    pub variable: usize,
    pub mask: u32,
    pub slot_shift: u8,
    pub value_shift: u8,
}

pub(crate) struct Stage {
    pub mask: u64,
    pub checks: [Check; 6],
    pub check_count: u8,
    pub writes: [Write; 3],
    pub write_count: u8,
    pub level: usize,
}

/// Shared compiler for [`FstJoin`] and [`crate::recursive_join::RecursiveJoin`]:
/// identical atom discovery, validation, reordering heuristic and
/// `Stage`/`Check`/`Write` generation -- the two executors differ only
/// in how they *walk* the compiled stages (a flat backtracking array
/// here, synchronized per-atom recursion there), never in what gets
/// compiled. One compiler, two consumers, so a correctness fix to the
/// compilation step (like the `Range`/`Multi`-style bugs found
/// elsewhere in this file's history) can't silently apply to only one
/// of them.
pub(crate) fn compile<P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &PathFstDeltaPow2Bfs) -> Result<(Vec<Stage>, usize, usize, bool)> {
    let invalid = |s: &str| TreeError::InvalidConfig(s.into());
    if query.num_vars > query.patterns.len().saturating_mul(3) {
        return Err(invalid("join has unused variables"));
    }
    let domains = tree.dim_sizes();
    let mut used = vec![false; query.num_vars];
    let mut zero_bits = vec![0u32; query.num_vars];
    let mut atoms = Vec::with_capacity(query.patterns.len());
    let mut empty = false;
    for p in &query.patterns {
        let atom = [p.subject, p.predicate.into(), p.object];
        for (d, &term) in atom.iter().enumerate() {
            match term {
                Term::Const(c) => empty |= c as u64 >= domains[d],
                Term::Var(v) => {
                    if v >= query.num_vars {
                        return Err(invalid("variable index exceeds num_vars"));
                    }
                    used[v] = true;
                    let width = 64 - (domains[d] - 1).leading_zeros();
                    zero_bits[v] |= !((1u64 << width) - 1) as u32;
                }
            }
        }
        // Duplicate atoms do not change a full conjunctive query's answers.
        if !atoms.contains(&atom) {
            atoms.push(atom);
        }
    }
    if used.contains(&false) {
        return Err(invalid("join has unused variables"));
    }

    // Fixed ordering preserves schedule compilation; favor constants, then
    // connectivity to atoms already selected. No data scan for estimates.
    // Identical heuristic to `NativeJoin::new` -- pure query-planning,
    // not backend-specific.
    used.fill(false);
    for i in 0..atoms.len() {
        let best = (i..atoms.len())
            .max_by_key(|&j| {
                let constants = atoms[j].iter().filter(|t| matches!(t, Term::Const(_))).count();
                let shared = atoms[j].iter().filter(|t| matches!(t, Term::Var(v) if used[*v])).count();
                (constants, shared, std::cmp::Reverse(j))
            })
            .unwrap();
        atoms.swap(i, best);
        for &term in &atoms[i] {
            if let Term::Var(v) = term {
                used[v] = true;
            }
        }
    }

    let height = tree.height();
    let atoms_count = atoms.len();
    if empty || atoms.is_empty() {
        return Ok((Vec::new(), atoms_count, query.num_vars, empty));
    }
    let mut stages = Vec::with_capacity(height * atoms_count);
    let mut known = zero_bits.clone();
    for level in 0..height {
        let (bits, shifts, remaining) = tree.level_schedule(level);
        for atom in &atoms {
            let slots = 1u32 << bits.iter().sum::<u8>();
            let mut mask = if slots == 64 { u64::MAX } else { (1u64 << slots) - 1 };
            let mut owners = [(usize::MAX, 0u32, 0u64); 6];
            let mut owner_count = 0;
            let mut checks = [Check::default(); 6];
            let mut check_count = 0u8;
            let mut writes = [Write::default(); 3];
            let mut write_count = 0u8;
            for d in 0..3 {
                let mut write_mask = 0u32;
                for b in 0..bits[d] {
                    let bit = 1u32 << (remaining[d] + b);
                    let slot_bit = shifts[d] + b;
                    let ones = slot_ones(slot_bit);
                    match atom[d] {
                        Term::Const(c) => mask &= if c & bit != 0 { ones } else { !ones },
                        Term::Var(v) => {
                            if zero_bits[v] & bit != 0 {
                                mask &= !ones;
                            } else if let Some(&(_, _, previous)) = owners[..owner_count].iter().find(|&&(var, value_bit, _)| var == v && value_bit == bit) {
                                mask &= !(ones ^ previous);
                            } else if known[v] & bit != 0 {
                                checks[check_count as usize] = Check { variable: v, bit, ones };
                                check_count += 1;
                            } else {
                                owners[owner_count] = (v, bit, ones);
                                owner_count += 1;
                                known[v] |= bit;
                                write_mask |= bit;
                            }
                        }
                    }
                }
                if write_mask != 0 {
                    let Term::Var(variable) = atom[d] else { unreachable!() };
                    writes[write_count as usize] = Write { variable, mask: write_mask, slot_shift: shifts[d], value_shift: remaining[d] };
                    write_count += 1;
                }
            }
            stages.push(Stage { mask, checks, check_count, writes, write_count, level });
        }
    }
    debug_assert!(known.iter().all(|&mask| mask == u32::MAX));
    Ok((stages, atoms_count, query.num_vars, empty))
}

/// One atom's current node, mid-descent. `addr`/`lo`/`word` are this
/// frame's *own* resolved position, set once by `open` and never
/// mutated afterward -- `child_addr`/`child_lo` are a separate, mutable
/// slot for "the position the most recently chosen candidate bit leads
/// to", read by the *next* real level's own `open` call.
///
/// A first version mutated `addr`/`lo` in place to become the chosen
/// child's position, directly. That's wrong the moment backtracking
/// returns to this same frame to try a *different* sibling bit: the
/// parent's own address is gone by then, overwritten by the previous
/// candidate's child address, so the next candidate's delta gets read
/// from the wrong node entirely. Caught immediately: a single-atom,
/// no-join query with three matching rows for the same variable
/// returned only the first one, every time -- not a join-specific bug
/// at all, since nothing about joining was involved yet.
#[derive(Clone, Copy)]
struct Frame {
    addr: u64,
    lo: [u64; NUM_DIMS],
    word: u64,
    remaining: u64,
    child_addr: u64,
    child_lo: [u64; NUM_DIMS],
}

impl Frame {
    const EMPTY: Self = Self { addr: 0, lo: [0; NUM_DIMS], word: 0, remaining: 0, child_addr: 0, child_lo: [0; NUM_DIMS] };
}

/// Reusable `O(atoms * height + variables)` execution storage, same
/// convention as `NativeJoinWorkspace`. Use one workspace per concurrent
/// call.
#[derive(Default)]
pub struct FstJoinWorkspace {
    frames: Vec<Frame>,
    bindings: Vec<u32>,
}

impl FstJoinWorkspace {
    pub fn allocated_bytes(&self) -> usize {
        crate::vec_bytes(&self.frames) + crate::vec_bytes(&self.bindings)
    }

    pub fn release(&mut self) {
        *self = Self::default();
    }
}

/// Optional diagnostics; ordinary `visit` compiles out counter updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FstJoinStats {
    pub node_visits: u64,
    pub candidates: u64,
    pub results: u64,
}

/// A compiled join borrowing a `PathFstDeltaPow2Bfs`. Compilation
/// allocates only `O(atoms * height + variables)` metadata, independently
/// of graph cardinality -- same bound as `NativeJoin`.
pub struct FstJoin<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    stages: Vec<Stage>,
    atoms: usize,
    variables: usize,
    empty: bool,
}

impl<'a> FstJoin<'a> {
    /// Compile constants, shared variables (including predicates),
    /// repeated variables, ground atoms and disconnected patterns --
    /// identical contract to `NativeJoin::new`. Variable identifiers must
    /// cover `0..query.num_vars`. Delegates to `compile`, shared with
    /// [`crate::recursive_join::RecursiveJoin`] -- see that function's
    /// own doc comment for why.
    pub fn new<P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &'a PathFstDeltaPow2Bfs) -> Result<Self> {
        let (stages, atoms, variables, empty) = compile(query, tree)?;
        Ok(Self { tree, stages, atoms, variables, empty })
    }

    /// Allocated plan vector capacity, excluding the borrowed tree and struct.
    pub fn allocated_bytes(&self) -> usize {
        crate::vec_bytes(&self.stages)
    }

    /// Stream full bindings in variable-ID order. After workspace warmup this
    /// executor allocates nothing.
    pub fn visit<T>(&self, workspace: &mut FstJoinWorkspace, emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.run::<false, T>(workspace, emit, &mut FstJoinStats::default())
    }

    /// Like `visit`, with counters reset at entry and retained on cancellation.
    pub fn visit_with_stats<T>(&self, workspace: &mut FstJoinWorkspace, stats: &mut FstJoinStats, emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        *stats = FstJoinStats::default();
        self.run::<true, T>(workspace, emit, stats)
    }

    fn run<const TRACK: bool, T>(&self, ws: &mut FstJoinWorkspace, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>, stats: &mut FstJoinStats) -> ControlFlow<T> {
        if self.empty {
            return ControlFlow::Continue(());
        }
        ws.bindings.resize(self.variables, 0);
        ws.bindings.fill(0);
        if self.atoms == 0 || self.stages.is_empty() {
            if self.atoms == 0 {
                if TRACK {
                    stats.results += 1;
                }
                return emit(&ws.bindings);
            }
            return ControlFlow::Continue(());
        }
        ws.frames.resize(self.stages.len(), Frame::EMPTY);
        let mut depth = 0;
        self.open::<TRACK>(ws, depth, stats);
        loop {
            let stage = &self.stages[depth];
            let frame = &mut ws.frames[depth];
            if frame.remaining == 0 {
                if depth == 0 {
                    return ControlFlow::Continue(());
                }
                depth -= 1;
                self.clear_binding(ws, depth);
                continue;
            }
            let qi = frame.remaining.trailing_zeros();
            frame.remaining &= frame.remaining - 1;
            if TRACK {
                stats.candidates += 1;
            }
            for write in &stage.writes[..stage.write_count as usize] {
                ws.bindings[write.variable] |= ((qi >> write.slot_shift) << write.value_shift) & write.mask;
            }
            // A leaf-level stage (this atom's last real level) never has
            // a stage `depth + self.atoms` after it for the *same* atom
            // -- the next atom (or the final emit) starts fresh at its
            // own root instead. So the child position this bit would
            // lead to is never read again; skip computing it entirely,
            // the same leaf shortcut every other traversal in
            // `path_fst.rs` already takes.
            if stage.level != self.tree.height() - 1 {
                let real_index = (frame.word & ((1u64 << qi) - 1)).count_ones() as usize;
                let delta = self.tree.read_delta(frame.addr, real_index);
                let (dim_bits, shifts, child_cell_bits) = self.tree.level_schedule(stage.level);
                frame.child_addr = frame.addr + delta;
                frame.child_lo = crate::tree::advance_lo(frame.lo, dim_bits, shifts, qi as usize, child_cell_bits);
            }
            if depth + 1 == self.stages.len() {
                if TRACK {
                    stats.results += 1;
                }
                emit(&ws.bindings)?;
                self.clear_binding(ws, depth);
            } else {
                depth += 1;
                self.open::<TRACK>(ws, depth, stats);
            }
        }
    }

    #[inline]
    fn clear_binding(&self, ws: &mut FstJoinWorkspace, depth: usize) {
        let stage = &self.stages[depth];
        for write in &stage.writes[..stage.write_count as usize] {
            ws.bindings[write.variable] &= !write.mask;
        }
    }

    #[inline]
    fn open<const TRACK: bool>(&self, ws: &mut FstJoinWorkspace, depth: usize, stats: &mut FstJoinStats) {
        let stage = &self.stages[depth];
        let (addr, lo) = if depth < self.atoms { (self.tree.root_addr(), [0u64; NUM_DIMS]) } else { (ws.frames[depth - self.atoms].child_addr, ws.frames[depth - self.atoms].child_lo) };
        let word = self.tree.read_word(addr);
        let mut remaining = word & stage.mask;
        for check in &stage.checks[..stage.check_count as usize] {
            remaining &= if ws.bindings[check.variable] & check.bit != 0 { check.ones } else { !check.ones };
        }
        ws.frames[depth] = Frame { addr, lo, word, remaining, child_addr: 0, child_lo: [0; NUM_DIMS] };
        if TRACK {
            stats.node_visits += 1;
        }
    }
}
