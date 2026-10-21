//! Demand-driven composition of physical FSTs and a compiled binding automaton.
//!
//! A product state holds one physical cursor per atom and the variable bits
//! already chosen. It expands only reached transitions; no lifted relation or
//! answer automaton is materialized. The event alphabet is (level, atom, slot).
//! The existing compiler supplies exactly the same equality/constant semantics
//! and traversal order as FstJoin. This is a factored automaton implementation of
//! that search, not a new join ordering algorithm or a lazy BDD converter.
use crate::Result;
use crate::fst_join::{Check, Write, compile};
use crate::graph::{JoinQuery, Term};
use crate::path_fst::PathFstDeltaPow2Bfs;
use std::ops::ControlFlow;

struct Transition {
    base_mask: u64,
    checks: [Check; 6],
    check_count: u8,
    writes: [Write; 3],
    write_count: u8,
    filter: usize,
    descend: bool,
}
impl Transition {
    #[inline]
    fn allowed<const TABLES: bool>(&self, masks: &[u64], bindings: &[u32]) -> u64 {
        if !TABLES {
            let mut mask = self.base_mask;
            for check in &self.checks[..self.check_count as usize] {
                mask &= if bindings[check.variable] & check.bit != 0 {
                    check.ones
                } else {
                    !check.ones
                };
            }
            return mask;
        }
        let mut key = 0;
        for (i, check) in self.checks[..self.check_count as usize].iter().enumerate() {
            key |= usize::from(bindings[check.variable] & check.bit != 0) << i;
        }
        masks[self.filter + key]
    }
    #[inline]
    fn advance(&self, slot: u32, bindings: &mut [u32]) {
        for write in &self.writes[..self.write_count as usize] {
            bindings[write.variable] |=
                ((slot >> write.slot_shift) << write.value_shift) & write.mask;
        }
    }
    #[inline]
    fn undo(&self, bindings: &mut [u32]) {
        for write in &self.writes[..self.write_count as usize] {
            bindings[write.variable] &= !write.mask;
        }
    }
}
#[derive(Clone, Copy, Default)]
struct State {
    address: u64,
    word: u64,
    remaining: u64,
    child: u64,
}
#[derive(Default)]
pub struct DemandWorkspace {
    states: Vec<State>,
    bindings: Vec<u32>,
}
impl DemandWorkspace {
    pub fn allocated_bytes(&self) -> usize {
        self.states.capacity() * std::mem::size_of::<State>() + self.bindings.capacity() * 4
    }
}
#[derive(Default, Debug)]
pub struct DemandStats {
    pub source_reads: u64,
    pub transitions: u64,
    pub rows: u64,
}
pub struct DemandJoin<'a> {
    tree: &'a PathFstDeltaPow2Bfs,
    transitions: Vec<Transition>,
    masks: Vec<u64>,
    atoms: usize,
    variables: usize,
    empty: bool,
    tables: bool,
}
impl<'a> DemandJoin<'a> {
    pub fn new<P: Copy + Into<Term>>(
        query: &JoinQuery<P>,
        tree: &'a PathFstDeltaPow2Bfs,
    ) -> Result<Self> {
        Self::with_tables(query, tree, false)
    }
    /// Optional mask tables are an experimental control; default guards are evaluated directly.
    pub fn with_tables<P: Copy + Into<Term>>(
        query: &JoinQuery<P>,
        tree: &'a PathFstDeltaPow2Bfs,
        tables: bool,
    ) -> Result<Self> {
        let (stages, atoms, variables, empty) = compile(query, tree)?;
        let mut masks = Vec::new();
        let mut transitions = Vec::with_capacity(stages.len());
        for stage in stages {
            let filter = masks.len();
            // At most six checks (the physical alphabet has at most 64 slots).
            // Only query metadata is compiled: no data node is read here.
            for key in 0..if tables {
                1usize << stage.check_count
            } else {
                0
            } {
                let mut mask = stage.mask;
                for (i, check) in stage.checks[..stage.check_count as usize]
                    .iter()
                    .enumerate()
                {
                    mask &= if key & (1 << i) != 0 {
                        check.ones
                    } else {
                        !check.ones
                    };
                }
                masks.push(mask);
            }
            transitions.push(Transition {
                base_mask: stage.mask,
                checks: stage.checks,
                check_count: stage.check_count,
                writes: stage.writes,
                write_count: stage.write_count,
                filter,
                descend: stage.level + 1 < tree.height(),
            });
        }
        Ok(Self {
            tree,
            transitions,
            masks,
            atoms,
            variables,
            empty,
            tables,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.transitions.capacity() * std::mem::size_of::<Transition>() + self.masks.capacity() * 8
    }
    pub fn states(&self) -> usize {
        self.transitions.len()
    }
    pub fn visit<T>(
        &self,
        ws: &mut DemandWorkspace,
        emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if self.tables {
            self.run::<false, true, T>(ws, &mut DemandStats::default(), emit)
        } else {
            self.run::<false, false, T>(ws, &mut DemandStats::default(), emit)
        }
    }
    pub fn visit_with_stats<T>(
        &self,
        ws: &mut DemandWorkspace,
        stats: &mut DemandStats,
        emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        *stats = DemandStats::default();
        if self.tables {
            self.run::<true, true, T>(ws, stats, emit)
        } else {
            self.run::<true, false, T>(ws, stats, emit)
        }
    }
    fn run<const TRACK: bool, const TABLES: bool, T>(
        &self,
        ws: &mut DemandWorkspace,
        stats: &mut DemandStats,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if self.empty {
            return ControlFlow::Continue(());
        }
        ws.bindings.resize(self.variables, 0);
        ws.bindings.fill(0);
        if self.atoms == 0 {
            if TRACK {
                stats.rows += 1;
            }
            return emit(&ws.bindings);
        }
        if self.transitions.is_empty() {
            return ControlFlow::Continue(());
        }
        ws.states.resize(self.transitions.len(), State::default());
        let mut depth = 0;
        self.open::<TRACK, TABLES>(ws, depth, stats);
        loop {
            let transition = &self.transitions[depth];
            let state = &mut ws.states[depth];
            if state.remaining == 0 {
                if depth == 0 {
                    return ControlFlow::Continue(());
                }
                depth -= 1;
                self.transitions[depth].undo(&mut ws.bindings);
                continue;
            }
            let slot = state.remaining.trailing_zeros();
            state.remaining &= state.remaining - 1;
            if TRACK {
                stats.transitions += 1;
            }
            transition.advance(slot, &mut ws.bindings);
            if transition.descend {
                let rank = (state.word & ((1u64 << slot) - 1)).count_ones() as usize;
                state.child = state.address + self.tree.read_delta(state.address, rank);
            }
            if depth + 1 == self.transitions.len() {
                if TRACK {
                    stats.rows += 1;
                }
                emit(&ws.bindings)?;
                transition.undo(&mut ws.bindings);
            } else {
                depth += 1;
                self.open::<TRACK, TABLES>(ws, depth, stats);
            }
        }
    }
    #[inline]
    fn open<const TRACK: bool, const TABLES: bool>(
        &self,
        ws: &mut DemandWorkspace,
        depth: usize,
        stats: &mut DemandStats,
    ) {
        let address = if depth < self.atoms {
            self.tree.root_addr()
        } else {
            ws.states[depth - self.atoms].child
        };
        let word = self.tree.read_word(address);
        let mask = self.transitions[depth].allowed::<TABLES>(&self.masks, &ws.bindings);
        ws.states[depth] = State {
            address,
            word,
            remaining: word & mask,
            child: 0,
        };
        if TRACK {
            stats.source_reads += 1;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fst_join::{FstJoin, FstJoinStats, FstJoinWorkspace};
    use crate::graph::TriplePattern;
    fn p(a: [Term; 3]) -> TriplePattern<Term> {
        TriplePattern {
            subject: a[0],
            predicate: a[1],
            object: a[2],
        }
    }
    pub(crate) fn check(tree: &PathFstDeltaPow2Bfs, query: &JoinQuery<Term>, oracle: &[Vec<u32>]) {
        for tables in [false, true] {
            let plan = DemandJoin::with_tables(query, tree, tables).unwrap();
            let mut ws = DemandWorkspace::default();
            let mut actual = Vec::new();
            let mut stats = DemandStats::default();
            let _: ControlFlow<()> = plan.visit_with_stats(&mut ws, &mut stats, |r| {
                actual.push(r.to_vec());
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, oracle);
            let baseline = FstJoin::new(query, tree).unwrap();
            let mut fs = FstJoinStats::default();
            let _: ControlFlow<()> =
                baseline.visit_with_stats(&mut FstJoinWorkspace::default(), &mut fs, |_| {
                    ControlFlow::Continue(())
                });
            assert_eq!(stats.source_reads, fs.node_visits);
            assert_eq!(stats.transitions, fs.candidates);
            if !oracle.is_empty() {
                assert_eq!(
                    plan.visit(&mut ws, |_| ControlFlow::Break(42)),
                    ControlFlow::Break(42)
                );
            }
            let mut count = 0;
            let _: ControlFlow<()> = plan.visit(&mut ws, |_| {
                count += 1;
                ControlFlow::Continue(())
            });
            assert_eq!(count, oracle.len());
        }
    }
    #[test]
    fn empty_ground_duplicate_and_invalid_queries() {
        use Term::{Const as C, Var as V};
        let tree =
            PathFstDeltaPow2Bfs::build(vec![(1, 0, 1)], [3, 1, 3], crate::DEFAULT_BLOCK_CAPACITY);
        for (terms, n, out) in [
            (vec![], 0, vec![vec![]]),
            (vec![[C(1), C(0), C(1)]], 0, vec![vec![]]),
            (vec![[C(2), C(0), C(1)]], 0, vec![]),
            (vec![[V(0), C(0), V(0)]; 2], 1, vec![vec![1]]),
            (vec![[V(0), C(8), V(0)]], 1, vec![]),
        ] {
            check(
                &tree,
                &JoinQuery {
                    patterns: terms.into_iter().map(p).collect(),
                    num_vars: n,
                },
                &out,
            );
        }
        for query in [
            JoinQuery {
                patterns: vec![],
                num_vars: 1,
            },
            JoinQuery {
                patterns: vec![p([V(1), C(0), C(1)])],
                num_vars: 1,
            },
        ] {
            assert!(DemandJoin::new(&query, &tree).is_err());
        }
    }
    #[test]
    fn many_atoms_and_singleton_dimensions() {
        use Term::{Const as C, Var as V};
        let tree =
            PathFstDeltaPow2Bfs::build(vec![(1, 0, 1)], [3, 1, 3], crate::DEFAULT_BLOCK_CAPACITY);
        let query = JoinQuery {
            patterns: (0..72).map(|v| p([C(1), C(0), V(v)])).collect(),
            num_vars: 72,
        };
        check(&tree, &query, &[vec![1; 72]]);
        for triples in [vec![], vec![(0, 0, 0)]] {
            let tree = PathFstDeltaPow2Bfs::build(
                triples.clone(),
                [1, 1, 1],
                crate::DEFAULT_BLOCK_CAPACITY,
            );
            let query = JoinQuery {
                patterns: vec![p([V(0), V(0), V(0)])],
                num_vars: 1,
            };
            assert_eq!(tree.contains(0, 0, 0), !triples.is_empty());
            let expected = if triples.is_empty() { vec![] } else { vec![vec![0]] };
            check(&tree, &query, &expected);
        }
    }
    #[test]
    fn wide_ids_and_high_bits_shared_across_dimensions() {
        use Term::{Const as C, Var as V};
        let tree = PathFstDeltaPow2Bfs::build(
            vec![(u32::MAX, 0, 1), (1, 0, u32::MAX), (0, 1, 0)],
            [1u64 << 32, 2, 1u64 << 32],
            crate::DEFAULT_BLOCK_CAPACITY,
        );
        check(
            &tree,
            &JoinQuery {
                patterns: vec![p([V(0), C(0), V(1)]), p([V(1), C(0), V(0)])],
                num_vars: 2,
            },
            &[vec![1, u32::MAX], vec![u32::MAX, 1]],
        );
    }
}
