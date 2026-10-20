//! Generic walk+automaton split for one bounded tree descent -- see
//! `DESIGN.md`'s "Joins and paths as automatons" entry for the design
//! discussion this implements. [`LevelAutomaton`] generalizes
//! `path_fst::PathFstDeltaPow2Bfs::precompute_level_masks`'s
//! `[u64; MAX_HEIGHT]` (one fixed mask per level, computed once per
//! query) into a trait: a *stateful* per-level mask, so the same walk
//! that drives today's `restricted_descent` can also drive a join atom
//! (mask depends on already-bound variables) or a path hop (mask
//! depends on which automaton states are still live) without a new
//! bespoke recursive descent for each.
//!
//! The alphabet is this crate's own up-to-64-slot child index per level,
//! not a single byte -- the one place this departs from BurntSushi's
//! `fst::Automaton` (whose `accept` consumes one byte at a time), because
//! the tree's real branching factor is the multi-bit group a level
//! allocates across S/P/O, not a byte. Everything else is deliberately
//! shaped like that trait: `can_match` is its dead-state check, under
//! the same name, checked *before* any storage read; [`And`] is its `And`
//! combinator, for the same reason -- composing two independent
//! restriction kinds (or, later, a restriction with a path-step
//! automaton) without writing a new method per combination.
//!
//! [`PrecomputedMaskAutomaton`] is the degenerate case that reproduces
//! `restricted_descent`'s current behavior exactly: one state, `mask`
//! ignores it, `advance` is the identity. [`DimRestrictionAutomaton`]
//! is one dimension's own restriction (`Exact`/`Free`/`Range`/`Multi`),
//! expanded as if every *other* dimension were unconstrained -- composing
//! three of these via [`And`] must equal `tree::restriction_mask`'s
//! combined result exactly, since `shifts` partitions each dimension
//! into disjoint bit ranges within one level's `qi` (every dimension's
//! own allowed sub-values occupy bits no other dimension ever touches),
//! so intersecting three independently-expanded masks is mathematically
//! the same as `restriction_mask`'s sequential fold -- verified by this
//! crate's own correctness probes (`path_fst_bgp_perf.rs`'s
//! `composedautomaton`), not just argued.
//!
//! `Range`/`Multi`, specifically, are precomputed per query rather than
//! recomputed per node -- see [`RangeTable`]'s own doc comment for the
//! boundary-tracking decomposition this relies on, and `DESIGN.md` for
//! the first version's cost (1.6-4.3x over `restricted_descent`) that
//! motivated it.

use crate::tree::{restriction_mask, DimRestriction, NUM_DIMS};

/// One bounded tree descent's walk+automaton contract. `State` is
/// threaded through every recursive call in the walk; implementors
/// should still keep it cheap to clone (a unit, a small `Copy` struct,
/// or a tuple of those via [`And`]) even though `Clone` alone, not
/// `Copy`, is required -- [`DimRestrictionAutomaton`]'s `Multi` case
/// needs a `Vec` (one slot per range in the union), which only `Clone`
/// can express.
pub(crate) trait LevelAutomaton {
    type State: Clone;

    /// The state before any level has been consumed.
    fn start(&self) -> Self::State;

    /// Dead-state check: true if *some* sequence of further levels from
    /// this state could still reach a result. A `false` here lets the
    /// walk skip reading a node's storage at all, not just skip
    /// recursing after the fact -- the same role as `fst::Automaton::
    /// can_match`. Every automaton in this file today returns `true`
    /// unconditionally: none of them has a real notion of "permanently
    /// dead" yet (that only becomes meaningful for a path-step DFA,
    /// future work), but the hook exists now so adding one later doesn't
    /// change this trait's shape.
    fn can_match(&self, state: &Self::State) -> bool;

    /// Which of this level's up to 64 child-slot bits are acceptable
    /// from `state` -- same role as `precompute_level_masks`'s per-level
    /// entry, now a function of state instead of a flat precomputed
    /// array indexed only by level.
    fn mask(&self, state: &Self::State, level: usize) -> u64;

    /// The state reached after consuming accepted bit `qi` at `level`.
    fn advance(&self, state: &Self::State, level: usize, qi: u32) -> Self::State;
}

/// Reproduces `restricted_descent`'s current behavior exactly, through
/// the trait instead of a bespoke method: wraps an already-precomputed
/// `[u64]`-per-level array (whatever `precompute_level_masks` produces),
/// stateless, no pruning.
pub(crate) struct PrecomputedMaskAutomaton<'a> {
    pub masks: &'a [u64],
}

impl LevelAutomaton for PrecomputedMaskAutomaton<'_> {
    type State = ();

    fn start(&self) {}
    fn can_match(&self, _state: &()) -> bool {
        true
    }
    fn mask(&self, _state: &(), level: usize) -> u64 {
        self.masks[level]
    }
    fn advance(&self, _state: &(), _level: usize, _qi: u32) {}
}

/// Composes two automata by intersection: a bit is accepted only where
/// both agree, and `can_match` requires both to still be live. Same role
/// as `fst::Automaton`'s `And` combinator.
pub(crate) struct And<A, B>(pub A, pub B);

impl<A: LevelAutomaton, B: LevelAutomaton> LevelAutomaton for And<A, B> {
    type State = (A::State, B::State);

    fn start(&self) -> Self::State {
        (self.0.start(), self.1.start())
    }
    fn can_match(&self, (a, b): &Self::State) -> bool {
        self.0.can_match(a) && self.1.can_match(b)
    }
    fn mask(&self, (a, b): &Self::State, level: usize) -> u64 {
        self.0.mask(a, level) & self.1.mask(b, level)
    }
    fn advance(&self, (a, b): &Self::State, level: usize, qi: u32) -> Self::State {
        (self.0.advance(a, level, qi), self.1.advance(b, level, qi))
    }
}

/// Whether a boundary-tracking run is still exactly glued to its query
/// endpoint (`Tracking`) or has already diverged from it in the
/// direction that makes the bound permanently true (`Satisfied`, i.e.
/// "free" from here on for *this* bound specifically). See
/// [`RangeTable`]'s own doc comment for the decomposition this
/// implements.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Boundary {
    Tracking,
    Satisfied,
}

#[derive(Clone, Copy)]
pub(crate) struct RangeState {
    lower: Boundary,
    upper: Boundary,
}

/// One dimension's `Range(a, b)` restriction, precomputed per level
/// instead of recomputed per node from a raw `lo`.
///
/// The key fact this relies on: `contiguous_range`'s clip of the query
/// range against a node's window always produces a *contiguous* run of
/// accepted sub-indices. That means at any point during the descent, at
/// most the *leftmost* accepted sub-index can still be touching `a`,
/// and at most the *rightmost* can still be touching `b` -- everything
/// strictly between them is already fully inside `[a, b]`, with no way
/// to fall back out, i.e. unconstrained (`Free`) from there on. So the
/// state needed is not the raw accumulated `lo` (which can take many
/// values depending on which sub-index was actually taken at each
/// level) -- it's just two independent flags, "still tracking the
/// lower boundary" and "still tracking the upper boundary", each only
/// ever `Tracking` or `Satisfied`. And since `a`/`b` come from the
/// query, not the data, their own bit-prefix at every level is known
/// *before* touching the tree at all -- so the whole per-level
/// `(state, level) -> mask` table can be built once per query, the
/// same timing class as today's `precompute_level_masks`, instead of
/// calling `contiguous_range`/`restriction_mask` fresh at every node.
///
/// `Exact(v)` is `Range(v, v)` with `a == b`: the lower and upper
/// boundary trackers coincide and neither can ever reach `Satisfied`
/// (any sub-index that would satisfy one bound necessarily violates
/// the other, since they're the same value), so exactly one sub-index
/// survives at every level -- the same single-path behavior `Exact`
/// has always had, now falling out of the general mechanism instead of
/// a separate code path.
pub(crate) struct RangeTable {
    // No `full_mask` table: "fully free for this dimension" is always
    // `u64::MAX` regardless of level or which dimension this is --
    // other dimensions' own masks (ANDed in by `And`) are what actually
    // bound the result, same identity-element role as `DimKind::Free`'s
    // own mask.
    lower_boundary_mask: Vec<u64>,
    upper_boundary_mask: Vec<u64>,
    a_sub: Vec<u32>,
    b_sub: Vec<u32>,
    shifts: Vec<u8>,
    dim_bit_masks: Vec<u32>,
}

impl RangeTable {
    fn build(dim: usize, a: u32, b: u32, level_dim_bits: &[[u8; NUM_DIMS]], level_shifts: &[[u8; NUM_DIMS]], level_child_cell_bits: &[[u8; NUM_DIMS]]) -> Self {
        let height = level_dim_bits.len();
        let (mut lo_a, mut lo_b) = (0u64, 0u64);
        let mut t = RangeTable {
            lower_boundary_mask: Vec::with_capacity(height),
            upper_boundary_mask: Vec::with_capacity(height),
            a_sub: Vec::with_capacity(height),
            b_sub: Vec::with_capacity(height),
            shifts: Vec::with_capacity(height),
            dim_bit_masks: Vec::with_capacity(height),
        };
        for level in 0..height {
            let shift = level_shifts[level][dim];
            let dim_bits = level_dim_bits[level][dim];
            let child_cell_bits = level_child_cell_bits[level][dim];
            let width = 1u32 << dim_bits;
            // `a`/`b` come from the query, so `a >= lo_a` and `b >= lo_b`
            // hold by construction (`lo_a`/`lo_b` are exactly `a`/`b`'s
            // own bits truncated to what's been resolved so far) -- no
            // underflow, and the resulting sub-value is always `< width`
            // for the same reason `advance_lo`'s own decode always is.
            let a_s = ((a as u64 - lo_a) >> child_cell_bits) as u32;
            let b_s = ((b as u64 - lo_b) >> child_cell_bits) as u32;

            // Delegate to `restriction_mask` for *each* boundary mask,
            // with the other two dimensions pinned to `Free`, instead
            // of hand-expanding `expand_run(1, start, run_len, shift)`
            // in isolation -- that's the exact same cross-dimension
            // mistake `DimRestrictionAutomaton::mask` already made and
            // fixed once (see this module's own history in `DESIGN.md`),
            // reintroduced here by trying to precompute the mask without
            // going through the proven fold. `expand_run(1, ...)` only
            // sets bits where the *other* two dimensions are also zero;
            // calling `restriction_mask` here still only runs once per
            // *level*, at construction time, not once per node -- the
            // actual property this precomputation is for -- it just
            // does the cross-dimension expansion correctly while doing
            // it. Caught directly: even `Exact`-via-`Range(v, v)`
            // regressed to returning far fewer rows than
            // `restricted_descent` on the very first correctness check
            // after this precomputation was introduced, not a subtle
            // case.
            let mut lo = [0u64; NUM_DIMS];
            lo[dim] = lo_a;
            let mut lower_restrictions = [DimRestriction::Free, DimRestriction::Free, DimRestriction::Free];
            lower_restrictions[dim] = DimRestriction::Range(a, u32::MAX);
            t.lower_boundary_mask.push(restriction_mask(level_dim_bits[level], level_shifts[level], lo, level_child_cell_bits[level], &lower_restrictions));

            let mut lo_u = [0u64; NUM_DIMS];
            lo_u[dim] = lo_b;
            let mut upper_restrictions = [DimRestriction::Free, DimRestriction::Free, DimRestriction::Free];
            upper_restrictions[dim] = DimRestriction::Range(0, b);
            t.upper_boundary_mask.push(restriction_mask(level_dim_bits[level], level_shifts[level], lo_u, level_child_cell_bits[level], &upper_restrictions));

            t.a_sub.push(a_s);
            t.b_sub.push(b_s);
            t.shifts.push(shift);
            t.dim_bit_masks.push(width - 1);
            lo_a += (a_s as u64) << child_cell_bits;
            lo_b += (b_s as u64) << child_cell_bits;
        }
        t
    }

    fn start() -> RangeState {
        RangeState { lower: Boundary::Tracking, upper: Boundary::Tracking }
    }

    fn mask(&self, state: RangeState, level: usize) -> u64 {
        let lower = if state.lower == Boundary::Tracking { self.lower_boundary_mask[level] } else { u64::MAX };
        let upper = if state.upper == Boundary::Tracking { self.upper_boundary_mask[level] } else { u64::MAX };
        lower & upper
    }

    fn sub_at(&self, level: usize, qi: u32) -> u32 {
        (qi >> self.shifts[level]) & self.dim_bit_masks[level]
    }

    fn advance(&self, state: RangeState, level: usize, sub: u32) -> RangeState {
        let lower = if state.lower == Boundary::Satisfied || sub > self.a_sub[level] { Boundary::Satisfied } else { Boundary::Tracking };
        let upper = if state.upper == Boundary::Satisfied || sub < self.b_sub[level] { Boundary::Satisfied } else { Boundary::Tracking };
        RangeState { lower, upper }
    }
}

pub(crate) enum DimKind {
    Free,
    Range(RangeTable),
    Multi(Vec<RangeTable>),
}

#[derive(Clone)]
pub(crate) enum DimState {
    Free,
    Range(RangeState),
    Multi(Vec<Option<RangeState>>),
}

/// One dimension's own restriction (`Exact`/`Free`/`Range`/`Multi`),
/// computed as if every *other* dimension were unconstrained -- see
/// this module's own doc comment for why composing three of these via
/// [`And`] is sound. `Exact`/`Range` are precomputed via a single
/// [`RangeTable`] (see its own doc comment); `Multi` holds one
/// `RangeTable` per range in the union, with its own state tracking,
/// independently, which ranges are still alive along the current path
/// (`None` once a range's own mask stops including the real `qi`
/// taken -- permanently dead for that path, not resurrected).
pub(crate) struct DimRestrictionAutomaton {
    kind: DimKind,
}

impl DimRestrictionAutomaton {
    pub fn new(dim: usize, restriction: DimRestriction<'_>, level_dim_bits: &[[u8; NUM_DIMS]], level_shifts: &[[u8; NUM_DIMS]], level_child_cell_bits: &[[u8; NUM_DIMS]]) -> Self {
        let kind = match restriction {
            DimRestriction::Free => DimKind::Free,
            DimRestriction::Exact(v) => DimKind::Range(RangeTable::build(dim, v, v, level_dim_bits, level_shifts, level_child_cell_bits)),
            DimRestriction::Range(a, b) => DimKind::Range(RangeTable::build(dim, a, b, level_dim_bits, level_shifts, level_child_cell_bits)),
            DimRestriction::Multi(ranges) => {
                DimKind::Multi(ranges.iter().map(|&(a, b)| RangeTable::build(dim, a, b, level_dim_bits, level_shifts, level_child_cell_bits)).collect())
            }
        };
        DimRestrictionAutomaton { kind }
    }
}

impl LevelAutomaton for DimRestrictionAutomaton {
    type State = DimState;

    fn start(&self) -> DimState {
        match &self.kind {
            DimKind::Free => DimState::Free,
            DimKind::Range(_) => DimState::Range(RangeTable::start()),
            DimKind::Multi(tables) => DimState::Multi(vec![Some(RangeTable::start()); tables.len()]),
        }
    }

    fn can_match(&self, _state: &DimState) -> bool {
        true
    }

    fn mask(&self, state: &DimState, level: usize) -> u64 {
        match (&self.kind, state) {
            (DimKind::Free, DimState::Free) => self.full_mask_free(level),
            (DimKind::Range(t), DimState::Range(s)) => t.mask(*s, level),
            (DimKind::Multi(tables), DimState::Multi(states)) => tables
                .iter()
                .zip(states.iter())
                .fold(0u64, |acc, (t, s)| if let Some(s) = s { acc | t.mask(*s, level) } else { acc }),
            _ => unreachable!("DimState always matches the DimKind it was started from"),
        }
    }

    fn advance(&self, state: &DimState, level: usize, qi: u32) -> DimState {
        match (&self.kind, state) {
            (DimKind::Free, DimState::Free) => DimState::Free,
            (DimKind::Range(t), DimState::Range(s)) => {
                let sub = t.sub_at(level, qi);
                DimState::Range(t.advance(*s, level, sub))
            }
            (DimKind::Multi(tables), DimState::Multi(states)) => DimState::Multi(
                tables
                    .iter()
                    .zip(states.iter())
                    .map(|(t, s)| {
                        let s = (*s)?;
                        let sub = t.sub_at(level, qi);
                        // Only advance a range that actually accepted this
                        // `qi` at this level -- the overall accepted `qi`
                        // came from the *union* of every still-alive
                        // range's own mask, so an individual range can be
                        // live overall yet not itself contain this bit.
                        if (t.mask(s, level) >> qi) & 1 != 0 {
                            Some(t.advance(s, level, sub))
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            _ => unreachable!("DimState always matches the DimKind it was started from"),
        }
    }
}

impl DimRestrictionAutomaton {
    fn full_mask_free(&self, level: usize) -> u64 {
        // `Free` never needed a table of its own -- any `RangeTable`'s
        // `full_mask` is "every sub-index at this level", independent
        // of which dimension/range it was built for, so building one
        // just to read `full_mask` back out would be redundant. Kept
        // as a small helper here instead of its own `DimKind` field.
        let _ = level;
        u64::MAX
    }
}
