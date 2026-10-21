//! Thousands of DISTINCT connected atoms, with an analytical answer oracle.
use crate::{
    DEFAULT_BLOCK_CAPACITY,
    graph::{JoinQuery, Term, TriplePattern},
    path_fst::PathFstDeltaPow2Bfs,
};
pub fn fixture(kind: &str, atoms: usize) -> (PathFstDeltaPow2Bfs, JoinQuery<Term>, Vec<Vec<u32>>) {
    use Term::{Const as C, Var as V};
    let n = 16u32;
    let star = kind == "star";
    let cycle = kind == "cycle";
    let carry = kind == "carry-chain";
    let triples = if star {
        (0..n)
            .flat_map(|s| (0..atoms as u32).map(move |p| (s, p, s)))
            .collect()
    } else {
        (0..n)
            .map(|s| (s, 0, if carry { (s + 1) % n } else { s ^ 1 }))
            .collect()
    };
    let tree = PathFstDeltaPow2Bfs::build(
        triples,
        [n as u64, if star { atoms as u64 } else { 1 }, n as u64],
        DEFAULT_BLOCK_CAPACITY,
    );
    let patterns = (0..atoms)
        .map(|i| TriplePattern {
            subject: if star { V(0) } else { V(i) },
            predicate: C(if star { i as u32 } else { 0 }),
            object: if cycle && i + 1 == atoms {
                V(0)
            } else {
                V(i + 1)
            },
        })
        .collect();
    let variables = if cycle { atoms } else { atoms + 1 };
    let mut expected = Vec::new();
    if !cycle || atoms % 2 == 0 {
        for s in 0..n {
            expected.push(
                (0..variables)
                    .map(|v| {
                        if carry {
                            (s + v as u32) % n
                        } else if star {
                            s
                        } else {
                            s ^ (v as u32 % 2)
                        }
                    })
                    .collect(),
            );
        }
    }
    (
        tree,
        JoinQuery {
            num_vars: variables,
            patterns,
        },
        expected,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduled_join::{HybridJoin, Policy, ScheduleWorkspace};
    use std::ops::ControlFlow;
    #[test]
    fn large_deduplication_and_repeated_variable_incidence() {
        use Term::{Const as C, Var as V};
        let (tree, _, _) = fixture("star", 128);
        let patterns: Vec<_> = (0..128)
            .map(|p| TriplePattern {
                subject: V(0),
                predicate: C(p),
                object: V(0),
            })
            .collect();
        let mut q = JoinQuery {
            num_vars: 1,
            patterns,
        };
        q.patterns.extend(q.patterns.clone());
        let p = HybridJoin::new(&q, &tree, Policy::Adaptive).unwrap();
        assert_eq!(p.states(), 128 * tree.height());
        let mut actual = Vec::new();
        let _: ControlFlow<()> = p.visit(&mut ScheduleWorkspace::default(), |r| {
            actual.push(r.to_vec());
            ControlFlow::Continue(())
        });
        actual.sort_unstable();
        assert_eq!(actual, (0..16).map(|v| vec![v]).collect::<Vec<_>>());
    }
    #[test]
    fn huge_connected_bgps_flat_stack_cancellation_and_reuse() {
        // A deliberately small stack proves execution depth does not follow
        // either atom count or the product of atom count and storage height.
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(|| {
                for (kind, n) in [
                    ("chain", 4096),
                    ("star", 1024),
                    ("cycle", 4096),
                    ("cycle", 4097),
                    ("carry-chain", 4096),
                ] {
                    let (tree, q, expected) = fixture(kind, n);
                    assert_eq!(q.patterns.len(), n);
                    for policy in [
                        Policy::Nested,
                        Policy::Prefix(1),
                        Policy::Levels,
                        Policy::Frontier,
                        Policy::Adaptive,
                    ] {
                        // Carry permits exponentially many coarse prefixes
                        // under level synchronization. Exercise that shape
                        // at large size with schedules that bind whole atoms.
                        if kind == "carry-chain"
                            && !matches!(
                                policy,
                                Policy::Nested | Policy::Frontier | Policy::Adaptive
                            )
                        {
                            continue;
                        }
                        let plan = HybridJoin::new(&q, &tree, policy).unwrap();
                        assert!(plan.planning.nodes <= plan.planning.node_budget);
                        assert_eq!(plan.states(), n * tree.height());
                        let mut ws = ScheduleWorkspace::default();
                        if !expected.is_empty() {
                            assert_eq!(
                                plan.visit(&mut ws, |_| ControlFlow::Break(7)),
                                ControlFlow::Break(7)
                            );
                        }
                        for _ in 0..2 {
                            let mut actual = Vec::new();
                            let _: ControlFlow<()> = plan.visit(&mut ws, |r| {
                                actual.push(r.to_vec());
                                ControlFlow::Continue(())
                            });
                            actual.sort_unstable();
                            assert_eq!(actual, expected, "{kind}/{n}/{policy:?}");
                        }
                    }
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
