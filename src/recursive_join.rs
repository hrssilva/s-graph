//! A third join executor for `PathFstDeltaPow2Bfs`: the *same* algorithm
//! `fst_join::FstJoin` runs -- interleaved by level, every atom advances
//! together, synchronized through shared bindings rather than a shared
//! physical position -- expressed as plain recursion (one automaton-shaped
//! step per atom per level) instead of `FstJoin`'s flat `Stage`/`Frame`
//! array with manual `depth`/`atoms` index arithmetic.
//!
//! Proposed directly, in response to a question about whether the
//! per-atom automatons used elsewhere in this crate could be composed
//! into "a single descent where all patterns contribute together"
//! instead of resolving one atom fully before the next
//! (`automaton_join::AutomatonJoin`'s shape). The honest answer: that
//! composition already exists, it's `FstJoin`'s own algorithm (see its
//! module doc and `TOPOLOGY_REVIEW.md`'s "virtual dimension lifting" --
//! Navarro et al.'s Qdag approach) -- but it can't be expressed as one
//! combined `u64` mask applied to one node the way
//! `level_automaton::And` composes restriction kinds, because each atom
//! occupies its *own* physical tree position (reflecting its own
//! variable-to-dimension assignment), not a shared one. What *can*
//! change is whether that per-atom, per-level stepping is written as a
//! flat array walked with index arithmetic, or as synchronized
//! recursion -- a real, open question about whether `FstJoin`'s own
//! overhead included some of that bookkeeping, not just the
//! interleaving itself. This module answers it by building the
//! recursive version and measuring both, not by assuming either shape
//! is faster.
//!
//! Compilation is identical to `FstJoin`'s -- literally the same
//! [`crate::fst_join::compile`] call, same `Stage`/`Check`/`Write`
//! plan. Only `visit`'s own walk differs: `step` recurses once per
//! `(level, atom)` pair, in the same level-major order `FstJoin`'s flat
//! `stages` array is laid out in, carrying each atom's *next* position
//! in a small stack-allocated buffer (bounded by [`MAX_ATOMS`]) instead
//! of a shared `Vec<Frame>` indexed by `depth - atoms`. Variable writes
//! still need the same `clear`-on-backtrack `FstJoin`/`NativeJoin` both
//! use (not `AutomatonJoin`'s write-once-by-assignment shortcut):
//! unlike `AutomatonJoin`, which resolves a whole variable in one shot
//! at one atom's leaf, this walk builds a shared variable's value up
//! bit-by-bit across levels via `|=`, so an abandoned candidate's bits
//! have to be subtracted back out (`&= !write.mask`) before the next
//! candidate's bits go in, or they'd corrupt it.

use crate::fst_join::{compile, Stage};
use crate::graph::{JoinQuery, Term};
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::{advance_lo, NUM_DIMS};
use crate::Result;
use std::ops::ControlFlow;

/// Join atoms beyond this count aren't expected for this crate's own
/// query shapes (every benchmark and example in `DESIGN.md` uses 2-3);
/// kept as a small, `Copy`-friendly stack array bound rather than a
/// `Vec` rebuilt at every level transition. Compilation itself has no
/// such limit -- only this executor's own per-level scratch space does.
const MAX_ATOMS: usize = 16;

#[derive(Clone, Copy)]
struct AtomNode {
    addr: u64,
    lo: [u64; NUM_DIMS],
}

impl AtomNode {
    const EMPTY: Self = Self { addr: 0, lo: [0; NUM_DIMS] };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecursiveJoinStats {
    pub node_visits: u64,
    pub candidates: u64,
    pub results: u64,
}

/// A compiled join borrowing a `PathFstDeltaPow2Bfs`, walked by
/// synchronized per-atom recursion instead of `FstJoin`'s flat stage
/// array. See this module's own doc comment for why it exists
/// alongside `FstJoin` rather than replacing it.
pub struct RecursiveJoin<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    stages: Vec<Stage>,
    atoms: usize,
    variables: usize,
    empty: bool,
}

impl<'a> RecursiveJoin<'a> {
    pub fn new<P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &'a PathFstDeltaPow2Bfs) -> Result<Self> {
        let (stages, atoms, variables, empty) = compile(query, tree)?;
        if atoms > MAX_ATOMS {
            return Err(crate::TreeError::InvalidConfig(format!("RecursiveJoin supports at most {MAX_ATOMS} atoms, got {atoms}")));
        }
        Ok(Self { tree, stages, atoms, variables, empty })
    }

    pub fn allocated_bytes(&self) -> usize {
        crate::vec_bytes(&self.stages)
    }

    pub fn visit<T>(&self, bindings: &mut Vec<u32>, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.visit_with_stats(bindings, &mut RecursiveJoinStats::default(), &mut emit)
    }

    pub fn visit_with_stats<T>(&self, bindings: &mut Vec<u32>, stats: &mut RecursiveJoinStats, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        *stats = RecursiveJoinStats::default();
        if self.empty {
            return ControlFlow::Continue(());
        }
        bindings.resize(self.variables, 0);
        bindings.fill(0);
        if self.atoms == 0 || self.stages.is_empty() {
            if self.atoms == 0 {
                stats.results += 1;
                return emit(bindings);
            }
            return ControlFlow::Continue(());
        }
        let mut nodes = [AtomNode::EMPTY; MAX_ATOMS];
        for node in nodes.iter_mut().take(self.atoms) {
            node.addr = self.tree.root_addr();
        }
        let mut next_nodes = [AtomNode::EMPTY; MAX_ATOMS];
        self.step(0, 0, &nodes[..self.atoms], &mut next_nodes[..self.atoms], bindings, stats, &mut emit)
    }

    #[allow(clippy::too_many_arguments)]
    fn step<T>(
        &self,
        level: usize,
        atom_idx: usize,
        nodes: &[AtomNode],
        next_nodes: &mut [AtomNode],
        bindings: &mut [u32],
        stats: &mut RecursiveJoinStats,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if atom_idx == self.atoms {
            if level == self.tree.height() - 1 {
                stats.results += 1;
                return emit(bindings);
            }
            let mut next_next = [AtomNode::EMPTY; MAX_ATOMS];
            return self.step(level + 1, 0, next_nodes, &mut next_next[..self.atoms], bindings, stats, emit);
        }

        let stage = &self.stages[level * self.atoms + atom_idx];
        let node = nodes[atom_idx];
        let word = self.tree.read_word(node.addr);
        stats.node_visits += 1;
        let mut remaining = word & stage.mask;
        for check in &stage.checks[..stage.check_count as usize] {
            remaining &= if bindings[check.variable] & check.bit != 0 { check.ones } else { !check.ones };
        }

        while remaining != 0 {
            let qi = remaining.trailing_zeros();
            remaining &= remaining - 1;
            stats.candidates += 1;

            for write in &stage.writes[..stage.write_count as usize] {
                bindings[write.variable] |= ((qi >> write.slot_shift) << write.value_shift) & write.mask;
            }
            if level != self.tree.height() - 1 {
                let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
                let delta = self.tree.read_delta(node.addr, real_index);
                let (dim_bits, shifts, child_cell_bits) = self.tree.level_schedule(level);
                next_nodes[atom_idx] = AtomNode { addr: node.addr + delta, lo: advance_lo(node.lo, dim_bits, shifts, qi as usize, child_cell_bits) };
            }

            self.step(level, atom_idx + 1, nodes, next_nodes, bindings, stats, emit)?;

            for write in &stage.writes[..stage.write_count as usize] {
                bindings[write.variable] &= !write.mask;
            }
        }
        ControlFlow::Continue(())
    }
}
