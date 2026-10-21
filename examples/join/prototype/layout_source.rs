//! Prototype adapter: the existing scheduled executor over postorder storage.
//! The disposable crate injects matching accessors in path_fst.rs; production
//! sources and the BFS source adapter remain unchanged.
use crate::automata_core::{BitmapSource, TripleSource};
use crate::path_fst::PathFstDeltaPow2;

#[derive(Clone, Copy)]
pub struct Pow2Source<'a>(pub &'a PathFstDeltaPow2);

impl BitmapSource for Pow2Source<'_> {
    type Cursor = u64;
    #[inline]
    fn root(&self) -> Option<u64> {
        (self.0.height() > 0).then(|| self.0.layout_root_addr())
    }
    #[inline]
    fn height(&self) -> usize {
        self.0.height()
    }
    #[inline]
    fn word(&self, cursor: u64) -> u64 {
        self.0.layout_read_word(cursor)
    }
    #[inline]
    fn follow(&self, cursor: u64, word: u64, slot: u32) -> u64 {
        cursor
            - self
                .0
                .layout_read_delta(cursor, (word & ((1u64 << slot) - 1)).count_ones() as usize)
    }
}
impl TripleSource for Pow2Source<'_> {
    fn domains(&self) -> [u64; 3] {
        self.0.layout_domains()
    }
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        self.0.layout_schedule(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{JoinQuery, Term, TriplePattern};
    use crate::scheduled_join::{Policy, ScheduleWorkspace, ScheduledJoin};
    use crate::{
        automata_core::FstSource, join::PatternBackend, path_fst::PathFstDeltaPow2Bfs,
        path_ops::PatternIndex,
    };
    use std::ops::ControlFlow;

    #[test]
    fn both_layouts_match_the_existing_independent_join_oracle() {
        use Term::{Const as C, Var as V};
        let triples: Vec<_> = (0..7)
            .flat_map(|s| {
                (0..3).flat_map(move |p| {
                    (0..11)
                        .filter(move |o| (s * 7 + p * 11 + o) % 5 == 0)
                        .map(move |o| (s, p, o))
                })
            })
            .collect();
        for cap in [512, 4096, 65536] {
            let a = PathFstDeltaPow2::build(triples.clone(), [7, 3, 11], cap);
            let b = PathFstDeltaPow2Bfs::build(triples.clone(), [7, 3, 11], cap);
            assert_eq!(a.node_count(), b.node_count());
            let scan = PatternIndex::new(Pow2Source(&a)).unwrap();
            let mut all = scan.scan_pattern([None; 3]).collect::<Vec<_>>();
            all.sort_unstable();
            let mut expected = triples.clone();
            expected.sort_unstable();
            assert_eq!(all, expected);
            for terms in [
                vec![[V(0), C(0), V(1)]],
                vec![[V(0), V(1), V(0)]],
                vec![[V(0), C(0), V(1)], [V(1), C(1), V(0)]],
            ] {
                let query = JoinQuery {
                    num_vars: 2,
                    patterns: terms
                        .into_iter()
                        .map(|a| TriplePattern {
                            subject: a[0],
                            predicate: a[1],
                            object: a[2],
                        })
                        .collect(),
                };
                let mut oracle = Vec::new();
                for x in 0..11 {
                    for y in 0..11 {
                        let row = [x, y];
                        let value = |t| match t {
                            C(c) => c,
                            V(v) => row[v],
                        };
                        if query.patterns.iter().all(|p| {
                            triples.contains(&(
                                value(p.subject),
                                value(p.predicate),
                                value(p.object),
                            ))
                        }) {
                            oracle.push(row.to_vec());
                        }
                    }
                }
                for policy in [
                    Policy::Nested,
                    Policy::Prefix(1),
                    Policy::Prefix(2),
                    Policy::Levels,
                    Policy::Adaptive,
                ] {
                    let pa = ScheduledJoin::with_source(&query, Pow2Source(&a), policy).unwrap();
                    let pb = ScheduledJoin::with_source(&query, FstSource(&b), policy).unwrap();
                    assert_eq!(format!("{:?}", pa.planning), format!("{:?}", pb.planning));
                    let mut actual = Vec::new();
                    let mut ws = ScheduleWorkspace::default();
                    let _: ControlFlow<()> = pa.visit(&mut ws, |r| {
                        actual.push(r.to_vec());
                        ControlFlow::Continue(())
                    });
                    actual.sort_unstable();
                    assert_eq!(actual, oracle);
                    if !oracle.is_empty() {
                        assert_eq!(
                            pa.visit(&mut ws, |_| ControlFlow::Break(7)),
                            ControlFlow::Break(7)
                        );
                    }
                }
            }
        }
    }
}
