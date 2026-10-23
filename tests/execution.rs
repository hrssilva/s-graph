#![cfg(feature = "heap")]
use s_graph::{
    DEFAULT_BLOCK_CAPACITY, Fst,
    automata::*,
    graph::{JoinQuery, TriplePattern},
    scheduled_join::ScheduledJoin,
};
use std::{
    cell::Cell,
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

fn cycle(atoms: usize) -> JoinQuery<Term> {
    JoinQuery {
        num_vars: atoms,
        patterns: (0..atoms)
            .map(|i| TriplePattern {
                subject: Term::Var(i),
                predicate: Term::Const(0),
                object: Term::Var((i + 1) % atoms),
            })
            .collect(),
    }
}
fn bipartite() -> Fst {
    Fst::build(
        (0..8)
            .flat_map(|a| (8..16).flat_map(move |b| [(a, 0, b), (b, 0, a)]))
            .collect(),
        [16, 1, 16],
        DEFAULT_BLOCK_CAPACITY,
    )
}

#[test]
fn budgets_stop_before_results_under_every_schedule_and_scratch_is_reusable() {
    let graph = bipartite();
    for schedule in [
        Schedule::Nested,
        Schedule::Prefix(1),
        Schedule::Levels,
        Schedule::Frontier,
        Schedule::Adaptive,
    ] {
        // Even length has answers, but requires traversing many atoms first.
        let plan = ScheduledJoin::new(&cycle(32), &graph, schedule).unwrap();
        let mut ws = Default::default();
        let control = Execution::new().with_budget(12);
        assert_eq!(
            plan.visit_controlled(&mut ws, &control, |_| panic!("early answer")),
            ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::WorkBudget))
        );
        assert_eq!(control.work(), 12);
        assert_eq!(
            plan.visit_matching_controlled(
                &vec![Some(0); 32],
                &mut ws,
                &Execution::new(),
                |_| panic!("impossible binding")
            ),
            ControlFlow::<VisitStop<()>>::Continue(())
        );
        assert_eq!(
            plan.visit_controlled(&mut ws, &Execution::new(), |_| ControlFlow::Break(42)),
            ControlFlow::Break(VisitStop::User(42))
        );
        assert_eq!(
            plan.visit(&mut ws, |_| ControlFlow::Break(43)),
            ControlFlow::Break(43)
        );
    }
}

struct Source<'a> {
    graph: FstSource<'a>,
    reads: &'a Cell<u64>,
    cancelled: &'a AtomicBool,
    active: &'a Cell<u32>,
}
struct Guard<'a>(&'a Cell<u32>);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}
impl BitmapSource for Source<'_> {
    type Cursor = u64;
    fn begin_read(&self) -> impl Drop + '_ {
        self.active.set(self.active.get() + 1);
        Guard(self.active)
    }
    fn root(&self) -> Option<u64> {
        self.graph.root()
    }
    fn height(&self) -> usize {
        self.graph.height()
    }
    fn word(&self, cursor: u64) -> u64 {
        self.reads.set(self.reads.get() + 1);
        if self.reads.get() >= 3 {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        self.graph.word(cursor)
    }
    fn follow(&self, cursor: u64, word: u64, slot: u32) -> u64 {
        self.graph.follow(cursor, word, slot)
    }
}
impl TripleSource for Source<'_> {
    fn domains(&self) -> [u64; 3] {
        self.graph.domains()
    }
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        self.graph.layout(level)
    }
}
#[test]
fn cancellation_inside_a_no_answer_walk_releases_the_read_session() {
    let graph = bipartite();
    let reads = Cell::new(0);
    let active = Cell::new(0);
    let cancelled = AtomicBool::new(false);
    let source = Source {
        graph: graph.source(),
        reads: &reads,
        active: &active,
        cancelled: &cancelled,
    };
    let plan = ScheduledJoin::with_source(&cycle(33), source, Schedule::Levels).unwrap();
    let control = Execution::new().with_cancellation(&cancelled);
    assert_eq!(
        plan.visit_controlled(&mut Default::default(), &control, |_| panic!(
            "odd bipartite cycle"
        )),
        ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::Cancelled))
    );
    assert!(reads.get() >= 3);
    assert!(control.work() <= 256);
    assert_eq!(active.get(), 0);
}

#[test]
fn cancelling_a_planning_probe_cannot_return_an_empty_certificate() {
    let graph = bipartite();
    let reads = Cell::new(0);
    let active = Cell::new(0);
    let cancelled = AtomicBool::new(false);
    let source = Source {
        graph: graph.source(),
        reads: &reads,
        active: &active,
        cancelled: &cancelled,
    };
    let control = Execution::new().with_cancellation(&cancelled);
    let result =
        ScheduledJoin::with_source_controlled(&cycle(33), source, Schedule::Adaptive, &control);
    assert!(matches!(
        result,
        Err(PreparationError::Interrupted(StopReason::Cancelled))
    ));
    assert!(reads.get() >= 3);
    assert_eq!(active.get(), 0);
}

#[test]
fn reused_request_polls_before_values_and_composed_entries() {
    let graph = bipartite();
    for phase in 0..3 {
        let cancelled = AtomicBool::new(false);
        let control = Execution::new().with_cancellation(&cancelled);
        let values = Values::unit(1);
        let compiled = values.compile_controlled(graph.source(), &control).unwrap();
        assert_ne!(control.work() % 256, 0);
        cancelled.store(true, Ordering::Relaxed);
        let before = control.work();
        if phase == 0 {
            assert!(matches!(
                values.compile_controlled(graph.source(), &control),
                Err(PreparationError::Interrupted(StopReason::Cancelled))
            ));
        } else if phase == 1 {
            assert_eq!(
                compiled.visit_controlled(&mut (), &control, |_| panic!("pre-cancelled VALUES")),
                ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::Cancelled))
            );
        } else {
            let plan = s_graph::query_ops::UnionAll {
                left: compiled,
                right: values,
            };
            assert_eq!(
                plan.visit_controlled(&mut Default::default(), &control, |_| panic!(
                    "pre-cancelled union"
                )),
                ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::Cancelled))
            );
        }
        assert_eq!(control.work(), before);
    }
}

#[test]
fn cancellation_during_custom_filter_construction_is_reported() {
    struct CancelOnClone<'a>(&'a AtomicBool);
    impl Clone for CancelOnClone<'_> {
        fn clone(&self) -> Self {
            self.0.store(true, Ordering::Relaxed);
            Self(self.0)
        }
    }
    impl s_graph::query_ops::Condition for CancelOnClone<'_> {
        fn test(&self, _: Binding<'_>) -> bool {
            true
        }
    }
    let graph = bipartite();
    let cancelled = AtomicBool::new(false);
    let control = Execution::new().with_cancellation(&cancelled);
    let builder = Values::unit(1).filter(CancelOnClone(&cancelled));
    assert!(matches!(
        builder.compile_controlled(graph.source(), &control),
        Err(PreparationError::Interrupted(StopReason::Cancelled))
    ));
}

#[test]
fn preparation_limits_and_pre_cancelled_ground_queries_are_explicit() {
    let graph = bipartite();
    let builder = (0..200).fold(Bgp::new(201), |q, i| {
        q.triple([Term::Var(i), Term::Const(0), Term::Var(i + 1)])
    });
    let control = Execution::new().with_budget(10);
    assert!(matches!(
        builder.compile_controlled(graph.source(), &control),
        Err(PreparationError::Interrupted(StopReason::WorkBudget))
    ));
    assert_eq!(control.work(), 10);
    let cancelled = AtomicBool::new(true);
    let machine = Bgp::new(0).triple([Term::Const(0), Term::Const(0), Term::Const(8)]);
    let plan = graph.search(&machine).unwrap();
    for control in [
        Execution::new().with_cancellation(&cancelled),
        Execution::new().with_deadline(Instant::now() - Duration::from_secs(1)),
        Execution::new().with_budget(0),
    ] {
        assert!(matches!(
            plan.visit_controlled(&mut Default::default(), &control, |_| panic!(
                "stopped before ground answer"
            )),
            ControlFlow::Break(VisitStop::<()>::Interrupted(_))
        ));
    }
    assert!(matches!(
        machine.compile_controlled(
            graph.source(),
            &Execution::new().with_cancellation(&cancelled)
        ),
        Err(PreparationError::Interrupted(StopReason::Cancelled))
    ));
}

fn same<P: Search + ControlledSearch>(plan: &P) {
    let mut expected = Vec::new();
    let _: ControlFlow<()> = plan.visit(&mut Default::default(), |r| {
        expected.push(r.to_options());
        ControlFlow::Continue(())
    });
    let mut ws = Default::default();
    let mut actual = Vec::new();
    let result: VisitResult<()> = plan.visit_controlled(&mut ws, &Execution::new(), |r| {
        actual.push(r.to_options());
        ControlFlow::Continue(())
    });
    assert!(result.is_continue());
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    if !expected.is_empty() {
        assert_eq!(
            plan.visit_controlled(&mut ws, &Execution::new(), |_| ControlFlow::Break("stop")),
            ControlFlow::Break(VisitStop::User("stop"))
        );
    }
    let mut again = Vec::new();
    let _: VisitResult<()> = plan.visit_controlled(&mut ws, &Execution::new(), |r| {
        again.push(r.to_options());
        ControlFlow::Continue(())
    });
    again.sort_unstable();
    assert_eq!(again, expected);
}
#[test]
fn controlled_composition_preserves_bags_partial_bindings_and_correlation() {
    use Term::{Const as C, Var as V};
    let graph = Fst::build_dynamic(
        vec![(0, 0, 1), (0, 0, 2), (1, 1, 3)],
        DEFAULT_BLOCK_CAPACITY,
    );
    let a = Bgp::new(3).triple([C(0), C(0), V(0)]);
    let b = Bgp::new(3).triple([V(0), C(1), V(2)]);
    let values = Values::new(
        3,
        vec![
            vec![Some(1), None, None],
            vec![Some(1), None, None],
            vec![None, None, None],
        ],
    )
    .unwrap();
    let control = Execution::new();
    same(
        &a.clone()
            .join(b.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .optional(b.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .optional_if(b.clone(), |r: Binding<'_>| r.get(2) == Some(99))
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .minus(b.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .exists(b.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .not_exists(b.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .filter(|r: Binding<'_>| r.get(0) == Some(1))
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &a.clone()
            .union_all(values.clone())
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    same(
        &values
            .join(a)
            .compile_controlled(graph.source(), &control)
            .unwrap(),
    );
    let left = graph
        .search(Bgp::new(3).triple([C(0), C(0), V(0)]))
        .unwrap();
    let right = s_graph::query_ops::CorrelatedFilter {
        input: graph.search(&b).unwrap(),
        condition: |a: Binding<'_>, b: Binding<'_>| a.get(0) == b.get(0),
    };
    same(&s_graph::query_ops::Exists { left, right });
}

// A custom search opts in without changing Plan or the storage abstraction.
struct NoAnswers;
impl Search for NoAnswers {
    type Workspace = ();
    fn width(&self) -> usize {
        1
    }
    fn visit<T>(&self, _: &mut (), _: impl FnMut(Binding<'_>) -> ControlFlow<T>) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }
}
impl ControlledSearch for NoAnswers {
    fn visit_controlled<T>(
        &self,
        _: &mut (),
        control: &Execution<'_>,
        _: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        for _ in 0..1000 {
            if let Err(reason) = control.checkpoint() {
                return ControlFlow::Break(VisitStop::Interrupted(reason));
            }
        }
        ControlFlow::Continue(())
    }
}
#[test]
fn interrupted_rhs_is_never_an_optional_or_negation_fallback() {
    let values = || Values::new(1, vec![vec![Some(1)]]).unwrap();
    let control = Execution::new().with_budget(20);
    let result = s_graph::query_ops::LeftJoin::new(values(), NoAnswers).visit_controlled(
        &mut Default::default(),
        &control,
        |_| panic!("false optional fallback"),
    );
    assert_eq!(
        result,
        ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::WorkBudget))
    );
    let control = Execution::new().with_budget(20);
    assert_eq!(
        s_graph::query_ops::Minus {
            left: values(),
            right: NoAnswers
        }
        .visit_controlled(&mut Default::default(), &control, |_| panic!(
            "false minus fallback"
        )),
        ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::WorkBudget))
    );
    let control = Execution::new().with_budget(20);
    assert_eq!(
        s_graph::query_ops::NotExists {
            left: values(),
            right: NoAnswers
        }
        .visit_controlled(&mut Default::default(), &control, |_| panic!(
            "false NOT EXISTS"
        )),
        ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::WorkBudget))
    );
}

#[derive(Clone)]
struct Reject;
impl Automaton<Slot> for Reject {
    type State = ();
    fn start(&self) {}
    fn step(&self, _: &(), _: &Slot, _: &()) -> Option<()> {
        Some(())
    }
    fn is_accepting(&self, _: &(), _: &()) -> bool {
        false
    }
}
#[test]
fn user_recognizers_can_cancel_unsuccessful_language_intersections() {
    let graph = bipartite();
    let plan = Language(Unfiltered(Reject))
        .compile_controlled(graph.source(), &Execution::new())
        .unwrap();
    let mut ws = Default::default();
    let control = Execution::new().with_budget(20);
    assert_eq!(
        plan.visit_controlled(&mut ws, &control, |_| panic!("rejecting language")),
        ControlFlow::Break(VisitStop::<()>::Interrupted(StopReason::WorkBudget))
    );
    same(&plan);
    let _: VisitResult<()> =
        plan.visit_controlled(&mut ws, &Execution::new(), |_| panic!("rejecting language"));
    assert!(ws.is_empty());
}

#[test]
fn controlled_searches_use_packed_and_editable_sources_across_storage_modes() {
    fn check<S: s_graph::BlockStorage>(graph: s_graph::PackedFst<S>) {
        let machine = Bgp::new(1).triple([Term::Const(0), Term::Const(0), Term::Var(0)]);
        same(
            &machine
                .compile_controlled(graph.source(), &Execution::new())
                .unwrap(),
        );
        let mut graph = graph.into_mutable();
        assert!(graph.insert(0, 0, 2).unwrap());
        let plan = machine
            .compile_controlled(graph.source(), &Execution::new())
            .unwrap();
        same(&plan);
        let mut values = Vec::new();
        let _: VisitResult<()> =
            plan.visit_controlled(&mut Default::default(), &Execution::new(), |r| {
                values.push(r.get(0).unwrap());
                ControlFlow::Continue(())
            });
        values.sort_unstable();
        assert_eq!(values, [1, 2]);
    }
    check(Fst::build(
        vec![(0, 0, 1)],
        [4, 1, 4],
        DEFAULT_BLOCK_CAPACITY,
    ));
    #[cfg(feature = "file")]
    check(s_graph::FileFst::build(
        vec![(0, 0, 1)],
        [4, 1, 4],
        DEFAULT_BLOCK_CAPACITY,
    ));
    #[cfg(feature = "compressed")]
    check(s_graph::CompressedFst::build(
        vec![(0, 0, 1)],
        [4, 1, 4],
        DEFAULT_BLOCK_CAPACITY,
    ));
}
