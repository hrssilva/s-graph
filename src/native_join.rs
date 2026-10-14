//! Conjunctive joins directly over the joint S/P/O Morton topology.
//!
//! Compilation reads only the global schedule and domains. Each physical level
//! has fixed masks for constants, repeated coordinates and prefix compatibility.
//! Execution traverses surviving child combinations across all atoms before
//! descending to the next level. It never extracts or sorts input relations.
//!
//! Results use set semantics, include every declared variable, and are emitted
//! in an unspecified order. This implementation does not claim a WCO bound:
//! compatible prefix combinations can still require substantial exploration.

use crate::graph::{JoinQuery, Term};
use crate::storage::Pos;
use crate::tree::{PrefixCursor, RadixTreeDfsInterleavedVarSkipUnrolledGeneric as Tree};
use crate::{BlockStorage, Result, TreeError};
use std::ops::ControlFlow;

struct Check {
    variable: usize,
    bit: u32,
    ones: u64,
}

#[derive(Clone, Copy, Default)]
struct Write {
    variable: usize,
    mask: u32,
    slot_shift: u8,
    value_shift: u8,
}

/// A stage touches at most one dimension's variable per `d in 0..3`, so at
/// most 3 writes -- small enough to embed inline rather than slice into a
/// shared `Vec<Write>`. Profiling attributed ~20% of `NativeJoin::run`'s
/// self-time to iterating that shared-Vec slice (`next<Write>`/`eq<Write>`
/// in the profile) once per candidate child, even though almost every stage
/// writes 0 or 1 variable: the slice itself is a second, separately-fetched
/// allocation on top of `Stage`'s own cache line, and bounds-checking a
/// `start..end` range into it is pure overhead on top of that fetch. Inline
/// storage means `stage.writes[..stage.write_count]` is already resident
/// the moment `stage` itself is loaded, and the fixed max length of 3 lets
/// the compiler unroll/bound the loop instead of touching a second buffer.
struct Stage {
    mask: u64,
    check_start: usize,
    check_end: usize,
    writes: [Write; 3],
    write_count: u8,
    level: usize,
}

#[derive(Clone, Copy)]
struct Frame {
    cursor: PrefixCursor,
    remaining: u64,
}

impl Frame {
    const EMPTY: Self = Self { cursor: PrefixCursor::EMPTY, remaining: 0 };
}

/// Reusable O(atoms * height + variables) execution storage. No graph rows or
/// intermediate relations are retained. Use one workspace per concurrent call.
#[derive(Default)]
pub struct NativeJoinWorkspace {
    frames: Vec<Frame>,
    bindings: Vec<u32>,
}

impl NativeJoinWorkspace {
    /// Allocated vector capacity, excluding the workspace struct itself.
    pub fn allocated_bytes(&self) -> usize {
        crate::vec_bytes(&self.frames) + crate::vec_bytes(&self.bindings)
    }

    /// Discard capacity retained from earlier queries.
    pub fn release(&mut self) {
        *self = Self::default();
    }
}

/// Optional diagnostics; ordinary `visit` compiles out counter updates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeJoinStats {
    /// Node cursor openings, including repeated openings of the same node.
    pub node_visits: u64,
    /// Child choices surviving the compiled prefix filters.
    pub candidates: u64,
    pub results: u64,
}

/// A compiled join borrowing a tree. The borrow prevents mutation or stale
/// physical positions while the plan is in use. Compilation allocates only
/// O(atoms * height + variables) metadata, independently of graph cardinality.
pub struct NativeJoin<'a, S: BlockStorage> {
    tree: &'a Tree<S>,
    stages: Vec<Stage>,
    checks: Vec<Check>,
    atoms: usize,
    variables: usize,
    empty: bool,
}

impl<'a, S: BlockStorage> NativeJoin<'a, S> {
    /// Compile constants, shared variables (including predicates), repeated
    /// variables, ground atoms and disconnected patterns. Variable identifiers
    /// must cover `0..query.num_vars`.
    pub fn new<P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &'a Tree<S>) -> Result<Self> {
        let invalid = |s: &str| TreeError::InvalidConfig(s.into());
        if query.num_vars > query.patterns.len().saturating_mul(3) {
            return Err(invalid("join has unused variables"));
        }
        let domains = tree.prefix_domains();
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

        let mut plan = Self { tree, stages: Vec::new(), checks: Vec::new(), atoms: atoms.len(), variables: query.num_vars, empty };
        if empty || atoms.is_empty() {
            return Ok(plan);
        }
        plan.stages.reserve_exact(tree.prefix_schedule().len() * atoms.len());
        let mut known = zero_bits.clone();
        for (level, (bits, shifts, remaining)) in tree.prefix_schedule().enumerate() {
            for atom in &atoms {
                let slots = 1u32 << bits.iter().sum::<u8>();
                let mut mask = if slots == 64 { u64::MAX } else { (1u64 << slots) - 1 };
                let check_start = plan.checks.len();
                let mut owners = [(usize::MAX, 0u32, 0u64); 6];
                let mut owner_count = 0;
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
                                } else if let Some(&(_, _, previous)) = owners[..owner_count].iter().find(|&&(var, value_bit, _)| var == v && value_bit == bit)
                                {
                                    mask &= !(ones ^ previous);
                                } else if known[v] & bit != 0 {
                                    plan.checks.push(Check { variable: v, bit, ones });
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
                plan.stages.push(Stage { mask, check_start, check_end: plan.checks.len(), writes, write_count, level });
            }
        }
        debug_assert!(known.iter().all(|&mask| mask == u32::MAX));
        Ok(plan)
    }

    /// Allocated plan vector capacity, excluding the borrowed tree and struct.
    pub fn allocated_bytes(&self) -> usize {
        crate::vec_bytes(&self.stages) + crate::vec_bytes(&self.checks)
    }

    /// Stream full bindings in variable-ID order. After workspace warmup this
    /// executor allocates nothing; storage backends may manage their own caches.
    pub fn visit<T>(&self, workspace: &mut NativeJoinWorkspace, emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.run::<false, T>(workspace, emit, &mut NativeJoinStats::default())
    }

    /// Like `visit`, with counters reset at entry and retained on cancellation.
    pub fn visit_with_stats<T>(
        &self,
        workspace: &mut NativeJoinWorkspace,
        stats: &mut NativeJoinStats,
        emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        *stats = NativeJoinStats::default();
        self.run::<true, T>(workspace, emit, stats)
    }

    fn run<const TRACK: bool, T>(
        &self,
        ws: &mut NativeJoinWorkspace,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
        stats: &mut NativeJoinStats,
    ) -> ControlFlow<T> {
        let _guard = self.tree.prefix_guard();
        if self.empty {
            return ControlFlow::Continue(());
        }
        ws.bindings.resize(self.variables, 0);
        ws.bindings.fill(0);
        if self.atoms == 0 || self.stages.is_empty() {
            if self.atoms == 0 || self.tree.prefix_single() {
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
            let slot = frame.remaining.trailing_zeros();
            frame.remaining &= frame.remaining - 1;
            self.tree.prefix_child(&mut frame.cursor, stage.level, slot);
            if TRACK {
                stats.candidates += 1;
            }
            for write in &stage.writes[..stage.write_count as usize] {
                ws.bindings[write.variable] |= ((slot >> write.slot_shift) << write.value_shift) & write.mask;
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
    fn clear_binding(&self, ws: &mut NativeJoinWorkspace, depth: usize) {
        let stage = &self.stages[depth];
        for write in &stage.writes[..stage.write_count as usize] {
            ws.bindings[write.variable] &= !write.mask;
        }
    }

    #[inline]
    fn open<const TRACK: bool>(&self, ws: &mut NativeJoinWorkspace, depth: usize, stats: &mut NativeJoinStats) {
        let stage = &self.stages[depth];
        let pos = if depth < self.atoms { Pos::START } else { ws.frames[depth - self.atoms].cursor.position() };
        let cursor = self.tree.prefix_cursor(pos, stage.level);
        let mut remaining = cursor.unskipped & stage.mask;
        for check in &self.checks[stage.check_start..stage.check_end] {
            remaining &= if ws.bindings[check.variable] & check.bit != 0 { check.ones } else { !check.ones };
        }
        ws.frames[depth] = Frame { cursor, remaining };
        if TRACK {
            stats.node_visits += 1;
        }
    }
}

/// Occupancy positions whose child index contains a given radix bit.
fn slot_ones(bit: u8) -> u64 {
    const ONES: [u64; 6] = [0xaaaaaaaaaaaaaaaa, 0xcccccccccccccccc, 0xf0f0f0f0f0f0f0f0, 0xff00ff00ff00ff00, 0xffff0000ffff0000, 0xffffffff00000000];
    ONES[bit as usize]
}
