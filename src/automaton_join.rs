//! Historical whole-atom automaton executor, retained for comparisons.
//! Use [`crate::Join`] for the default adaptive FST executor, which supports
//! repeated variables within atoms and addresses skew/prefix regressions.
//! The implementation history below explains this alternative's design.
//!
//! Originally built to be compared against `fst_join::FstJoin`, not to
//! replace it -- see `DESIGN.md`'s "Phase 2" entries. `FstJoin` was
//! built to preserve `NativeJoin`'s own interleaved-by-level
//! backtracking shape rather than go through
//! `level_automaton::LevelAutomaton`, on the reasoning that an
//! automaton-per-atom design would lose that interleaving. This module
//! was that rejected design, actually built, so the reasoning could be
//! measured instead of just argued -- and ended up winning once its
//! own first version's unrelated performance bug (below) was found and
//! fixed.
//!
//! **Shape**: one atom at a time, each one a *complete* top-to-bottom
//! walk, reaching every one of *this* atom's own matches before the
//! *next* atom's walk starts at all -- plain recursion, nested loop join,
//! not `NativeJoin`'s level-major interleaving. A variable already bound
//! by an earlier atom becomes `DimRestriction::Exact(bindings[v])`.
//!
//! **A first version routed each atom's walk through
//! `crate::level_automaton::LevelAutomaton`** (composing three
//! `DimRestrictionAutomaton`s via `And`, exactly like
//! `restricted_descent_via_composed_automaton`) -- straightforward reuse
//! of Phase 1's machinery, and it measured 1.2-1.7x slower than `FstJoin`
//! on star/triangle joins (`DESIGN.md`). `perf record` on that version
//! (2000 star-join queries, 1M triples) found the slowdown wasn't where
//! the design doc guessed ("rebuilding the bound-variable automaton on
//! every atom entry") -- that's only `RangeTable::build` (3.72%) +
//! `restriction_mask` (8.69%), ~12% combined. The real cost (~78%) was
//! the `LevelAutomaton`/`And` abstraction's own per-node overhead:
//! `And::advance` (31.68%), `DimRestrictionAutomaton::advance` (14.93%),
//! `DimRestrictionAutomaton::mask` (8.13%), plus the extra indirection in
//! `walk_leaf_rec` itself (23.25%) -- three independent per-dimension
//! automata (each a real struct, each dispatched through `DimKind`'s
//! enum match), composed through nested tuples, called once per
//! accepted bit at every level.
//!
//! That machinery exists to support genuine `Range`/`Multi` restrictions
//! with real per-level state tracking (`RangeTable`'s boundary-tracking
//! `Tracking`/`Satisfied` states). A join atom never needs that: at the
//! moment any atom is walked, every one of its three dimensions is
//! always `Const`, `Free`, or `Exact` (a bound variable's *current*
//! value) -- never a genuine multi-state automaton, no `Range`/`Multi`
//! ever appears here. That's exactly the degenerate case
//! `precompute_level_masks` already handles with a flat
//! `[u64; MAX_HEIGHT]` array and zero per-node dispatch -- the same
//! mechanism `restricted_descent`/`PrecomputedMaskAutomaton` use. This
//! version computes that flat array once per atom entry
//! (`compute_atom_masks`, a near-verbatim copy of
//! `PathFstDeltaPow2Bfs::precompute_level_masks`) and walks through it
//! directly, with no `LevelAutomaton` involved at all. Measured effect:
//! see `DESIGN.md`.
//!
//! **Deliberately narrower than `FstJoin`**: a variable repeated *within*
//! the same atom (`?x :p ?x`) needs a same-level bit-equality constraint
//! between two dimensions that a single combined mask can't express the
//! way `NativeJoin`'s `owners` mechanism does (not reused here) --
//! `new` returns `InvalidConfig` for that shape rather than silently
//! getting it wrong. Every shape measured against `FstJoin` avoids it.
//!
//! Recursion has one pleasant side effect worth noting, not just a
//! footnote: because each atom's bindings are written only once its own
//! leaf is reached, and a later atom's walk is only ever entered as a
//! direct call from inside that leaf (not resumed later against a flat
//! array the way `NativeJoin`/`FstJoin`'s backtracking frames are),
//! nothing here needs an explicit "undo this write before trying the
//! next candidate" step -- the next sibling's own write simply overwrites
//! the same slot before anything reads it again. `FstJoin`'s `clear_binding`
//! has no equivalent here, not because it was optimized away but because
//! the recursive shape never created the problem it solves.

use crate::graph::{JoinQuery, Term};
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::{advance_lo, restriction_mask, DimRestriction, MAX_HEIGHT, NUM_DIMS};
use crate::{Result, TreeError};
use std::ops::ControlFlow;

struct AtomPlan {
    terms: [Term; NUM_DIMS],
    /// For a `Var` term: true if this is that variable's first
    /// occurrence anywhere in the (already reordered) atom list --
    /// write its resolved value once this atom's own leaf is reached.
    /// False means "already bound by an earlier atom" -- read the
    /// current value from `bindings` and treat it as `Exact`.
    first_occurrence: [bool; NUM_DIMS],
}

/// A compiled join borrowing a `PathFstDeltaPow2Bfs`. Compilation here
/// is deliberately light -- atom reordering and first-occurrence
/// classification only -- since each atom's own level-mask array is
/// computed at walk time, not ahead of it (it depends on bound
/// variables' runtime values).
pub struct AutomatonJoin<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    atoms: Vec<AtomPlan>,
    variables: usize,
    empty: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutomatonJoinStats {
    pub atom_entries: u64,
    pub results: u64,
}

impl<'a> AutomatonJoin<'a> {
    pub fn new<P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &'a PathFstDeltaPow2Bfs) -> Result<Self> {
        let invalid = |s: &str| TreeError::InvalidConfig(s.into());
        if query.num_vars > query.patterns.len().saturating_mul(3) {
            return Err(invalid("join has unused variables"));
        }
        let domains = tree.dim_sizes();
        let mut used = vec![false; query.num_vars];
        let mut atoms_terms: Vec<[Term; NUM_DIMS]> = Vec::with_capacity(query.patterns.len());
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
                    }
                }
            }
            if !atoms_terms.contains(&atom) {
                atoms_terms.push(atom);
            }
        }
        if used.contains(&false) {
            return Err(invalid("join has unused variables"));
        }

        // Same reordering heuristic as `FstJoin`/`NativeJoin` -- pure
        // query planning, independent of how each atom then gets walked.
        used.fill(false);
        for i in 0..atoms_terms.len() {
            let best = (i..atoms_terms.len())
                .max_by_key(|&j| {
                    let constants = atoms_terms[j].iter().filter(|t| matches!(t, Term::Const(_))).count();
                    let shared = atoms_terms[j].iter().filter(|t| matches!(t, Term::Var(v) if used[*v])).count();
                    (constants, shared, std::cmp::Reverse(j))
                })
                .unwrap();
            atoms_terms.swap(i, best);
            for &term in &atoms_terms[i] {
                if let Term::Var(v) = term {
                    used[v] = true;
                }
            }
        }

        let mut seen = vec![false; query.num_vars];
        let mut atoms = Vec::with_capacity(atoms_terms.len());
        for terms in &atoms_terms {
            let mut local_seen: [Option<usize>; NUM_DIMS] = [None; NUM_DIMS];
            let mut local_count = 0;
            let mut first_occurrence = [false; NUM_DIMS];
            for (d, term) in terms.iter().enumerate() {
                if let Term::Var(v) = *term {
                    if local_seen[..local_count].contains(&Some(v)) {
                        return Err(invalid("automaton-shaped join does not support a variable repeated within the same atom"));
                    }
                    local_seen[local_count] = Some(v);
                    local_count += 1;
                    if !seen[v] {
                        seen[v] = true;
                        first_occurrence[d] = true;
                    }
                }
            }
            atoms.push(AtomPlan { terms: *terms, first_occurrence });
        }

        Ok(Self { tree, atoms, variables: query.num_vars, empty })
    }

    pub fn visit<T>(&self, bindings: &mut Vec<u32>, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        self.visit_with_stats(bindings, &mut AutomatonJoinStats::default(), &mut emit)
    }

    pub fn visit_with_stats<T>(&self, bindings: &mut Vec<u32>, stats: &mut AutomatonJoinStats, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        if self.empty {
            return ControlFlow::Continue(());
        }
        bindings.resize(self.variables, 0);
        bindings.fill(0);
        if self.atoms.is_empty() {
            stats.results += 1;
            return emit(bindings);
        }
        self.walk_atom(0, bindings, stats, &mut emit)
    }

    fn walk_atom<T>(&self, atom_idx: usize, bindings: &mut Vec<u32>, stats: &mut AutomatonJoinStats, emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        stats.atom_entries += 1;
        let plan = &self.atoms[atom_idx];
        let mut restrictions = [DimRestriction::Free, DimRestriction::Free, DimRestriction::Free];
        for (d, r) in restrictions.iter_mut().enumerate() {
            *r = match plan.terms[d] {
                Term::Const(c) => DimRestriction::Exact(c),
                Term::Var(v) => {
                    if plan.first_occurrence[d] {
                        DimRestriction::Free
                    } else {
                        DimRestriction::Exact(bindings[v])
                    }
                }
            };
        }
        let masks = compute_atom_masks(self.tree, &restrictions);
        let is_last_atom = atom_idx + 1 == self.atoms.len();

        walk_leaf_rec(self.tree, &masks, self.tree.root_addr(), [0u64; NUM_DIMS], 0, &mut |child_lo: [u64; NUM_DIMS]| -> ControlFlow<T> {
            for (d, term) in plan.terms.iter().enumerate() {
                if plan.first_occurrence[d]
                    && let Term::Var(v) = *term
                {
                    bindings[v] = child_lo[d] as u32;
                }
            }
            if is_last_atom {
                stats.results += 1;
                emit(bindings)
            } else {
                self.walk_atom(atom_idx + 1, bindings, stats, emit)
            }
        })
    }
}

/// Near-verbatim copy of `PathFstDeltaPow2Bfs::precompute_level_masks`
/// -- one flat, zero-allocation array, computed once per atom entry,
/// not a per-node `LevelAutomaton`. See this module's own doc comment
/// for why that's the right level of generality for a join atom's own
/// restriction (always `Const`/`Free`/`Exact`, never genuinely
/// stateful).
fn compute_atom_masks(tree: &PathFstDeltaPow2Bfs, restrictions: &[DimRestriction; NUM_DIMS]) -> [u64; MAX_HEIGHT] {
    let height = tree.height();
    let (level_dim_bits, level_shifts, level_child_cell_bits) = tree.full_schedule();
    let mut lo = [0u64; NUM_DIMS];
    let mut masks = [0u64; MAX_HEIGHT];
    for level in 0..height {
        let dim_bits = level_dim_bits[level];
        let shifts = level_shifts[level];
        let child_cell_bits = level_child_cell_bits[level];
        masks[level] = restriction_mask(dim_bits, shifts, lo, child_cell_bits, restrictions);
        for (d, cell_bits) in child_cell_bits.iter().enumerate() {
            if let DimRestriction::Exact(v) = restrictions[d] {
                let sub = (v as u64 - lo[d]) >> cell_bits;
                lo[d] += sub << cell_bits;
            }
        }
    }
    masks
}

/// Same shape as `path_fst::PathFstDeltaPow2Bfs::visit_rec`, just
/// emitting the *whole* resolved `[S, P, O]` coordinate at the leaf
/// instead of one `emit_dim` -- a join atom can need to write more than
/// one dimension's resolved value into `bindings`, not only one, so a
/// single scalar emit doesn't fit here the way it does for a plain
/// restriction query.
fn walk_leaf_rec<T>(tree: &PathFstDeltaPow2Bfs, masks: &[u64], addr: u64, lo: [u64; NUM_DIMS], level: usize, on_leaf: &mut impl FnMut([u64; NUM_DIMS]) -> ControlFlow<T>) -> ControlFlow<T> {
    let word = tree.read_word(addr);
    let (dim_bits, shifts, child_cell_bits) = tree.level_schedule(level);
    let mut relevant = word & masks[level];
    let is_last = level == tree.height() - 1;

    if is_last {
        while relevant != 0 {
            let qi = relevant.trailing_zeros();
            relevant &= relevant - 1;
            let child_lo = advance_lo(lo, dim_bits, shifts, qi as usize, child_cell_bits);
            on_leaf(child_lo)?;
        }
        return ControlFlow::Continue(());
    }

    while relevant != 0 {
        let qi = relevant.trailing_zeros();
        relevant &= relevant - 1;
        let real_index = (word & ((1u64 << qi) - 1)).count_ones() as usize;
        let delta = tree.read_delta(addr, real_index);
        let child_addr = addr + delta;
        let child_lo = advance_lo(lo, dim_bits, shifts, qi as usize, child_cell_bits);
        walk_leaf_rec(tree, masks, child_addr, child_lo, level + 1, on_leaf)?;
    }
    ControlFlow::Continue(())
}
