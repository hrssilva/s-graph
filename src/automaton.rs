//! Regular-path reachability using a caller-supplied epsilon-free NFA.
//! Sequences, alternatives, optional steps and repetition can be represented
//! without introducing a parser or a database query language into the library.
//! Transitions are packed by source state; the search deduplicates (node,state)
//! pairs, not just nodes, and observes the same min/max walk bounds as traversal.

use crate::join::PatternBackend;
use crate::traversal::{Edges, HopRange, Step};
use crate::{Result, TreeError};
use std::collections::HashSet;
use std::ops::ControlFlow;

#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub from: u32,
    pub to: u32,
    pub step: Step,
}

pub struct PathAutomaton {
    start: u32,
    accepting: Vec<bool>,
    offsets: Vec<usize>,
    transitions: Vec<Transition>,
}

#[derive(Default)]
pub struct AutomatonWorkspace {
    frontier: Vec<(u32, u32)>,
    next: Vec<(u32, u32)>,
    layer: HashSet<(u32, u32)>,
    visited: HashSet<(u32, u32)>,
    emitted: HashSet<u32>,
}

impl PathAutomaton {
    /// All transitions consume one edge. Mark the start state accepting to
    /// accept zero hops. Eliminate epsilon transitions in the caller's compiler.
    pub fn new(states: u32, start: u32, accepting: &[u32], mut transitions: Vec<Transition>) -> Result<Self> {
        if start >= states || accepting.iter().any(|&s| s >= states) || transitions.iter().any(|t| t.from >= states || t.to >= states) {
            return Err(TreeError::InvalidConfig("automaton state outside domain".into()));
        }
        let mut flags = vec![false; states as usize];
        for &s in accepting {
            flags[s as usize] = true;
        }
        transitions.sort_unstable_by_key(|t| t.from);
        let mut offsets = vec![0; states as usize + 1];
        for t in &transitions {
            offsets[t.from as usize + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        Ok(Self { start, accepting: flags, offsets, transitions })
    }

    /// Emit distinct endpoints and their first accepted length. Workspace grows
    /// with reached product states rather than maximum graph ID. After min_hops,
    /// global product-state visitation terminates cycles without rescanning them.
    pub fn reachable<B: PatternBackend, T>(
        &self,
        graph: &B,
        start: u32,
        hops: HopRange,
        ws: &mut AutomatonWorkspace,
        mut emit: impl FnMut(u32, u32) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        ws.frontier.clear();
        ws.next.clear();
        ws.visited.clear();
        ws.emitted.clear();
        ws.frontier.push((start, self.start));
        if hops.min() == 0 {
            ws.visited.insert((start, self.start));
            if self.accepting[self.start as usize] {
                ws.emitted.insert(start);
                emit(start, 0)?;
            }
        }
        for depth in 0..hops.max() {
            ws.layer.clear();
            ws.next.clear();
            for &(node, state) in &ws.frontier {
                for transition in &self.transitions[self.offsets[state as usize]..self.offsets[state as usize + 1]] {
                    for (next, _) in Edges::new(graph, node, transition.step) {
                        let product = (next, transition.to);
                        if !ws.layer.insert(product) {
                            continue;
                        }
                        if depth + 1 >= hops.min() {
                            if !ws.visited.insert(product) {
                                continue;
                            }
                            if self.accepting[transition.to as usize] && ws.emitted.insert(next) {
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
