//! Binding-aware, bag-preserving query composition. Query widths use one global
//! variable namespace. Probe hints restrict rows; they never add bindings or
//! change the lexical context of a right-hand relation/filter.
use crate::automata_core::TripleSource;
use crate::execution::{Check, Execution, PreparationError, Uncontrolled};
use crate::graph::{JoinQuery, Term, TriplePattern};
use crate::scheduled_join::{Policy, ScheduleWorkspace, ScheduledJoin};
use crate::{Result, TreeError};
use std::ops::ControlFlow;

#[derive(Clone, Copy, Debug)]
pub struct Binding<'a> {
    pub values: &'a [u32],
    pub bound: &'a [u64],
}
impl Binding<'_> {
    pub fn get(self, v: usize) -> Option<u32> {
        assert!(v < self.values.len());
        (self.bound[v / 64] & (1u64 << (v % 64)) != 0).then(|| self.values[v])
    }
    pub fn compatible(self, other: Self) -> bool {
        assert_eq!(self.values.len(), other.values.len());
        self.bound
            .iter()
            .zip(other.bound)
            .enumerate()
            .all(|(word, (&a, &b))| {
                let mut shared = a & b;
                while shared != 0 {
                    let v = word * 64 + shared.trailing_zeros() as usize;
                    shared &= shared - 1;
                    if self.values[v] != other.values[v] {
                        return false;
                    }
                }
                true
            })
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.bound
            .iter()
            .zip(other.bound)
            .any(|(&a, &b)| a & b != 0)
    }
    pub fn to_options(self) -> Vec<Option<u32>> {
        (0..self.values.len()).map(|v| self.get(v)).collect()
    }
}
#[derive(Default, Clone, Debug)]
pub struct Row {
    values: Vec<u32>,
    bound: Vec<u64>,
}
impl Row {
    pub fn empty(width: usize) -> Self {
        let mut r = Self::default();
        r.reset(width);
        r
    }
    pub fn reset(&mut self, width: usize) {
        self.values.resize(width, 0);
        self.values.fill(0);
        self.bound.resize(width.div_ceil(64), 0);
        self.bound.fill(0);
    }
    pub fn set(&mut self, v: usize, value: u32) {
        self.values[v] = value;
        self.bound[v / 64] |= 1u64 << (v % 64);
    }
    pub fn view(&self) -> Binding<'_> {
        Binding {
            values: &self.values,
            bound: &self.bound,
        }
    }
    pub fn merge(&mut self, a: Binding<'_>, b: Binding<'_>) {
        assert_eq!(a.values.len(), b.values.len());
        self.values.resize(a.values.len(), 0);
        self.values.copy_from_slice(a.values);
        self.bound.resize(a.bound.len(), 0);
        for (i, (&x, &y)) in a.bound.iter().zip(b.bound).enumerate() {
            self.bound[i] = x | y;
            let mut todo = y;
            while todo != 0 {
                let v = i * 64 + todo.trailing_zeros() as usize;
                todo &= todo - 1;
                self.values[v] = b.values[v];
            }
        }
    }
}
pub trait Plan {
    type Workspace: Default;
    fn width(&self) -> usize;
    /// Conservative bound-variable superset; None means unknown. Allows
    /// MINUS to skip an RHS that cannot share any currently bound variable.
    fn possible_variables(&self) -> Option<&[u64]> {
        None
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T>;
    /// A physical lookup constraint, not correlated expression evaluation.
    /// Every emitted row remains a row of this plan evaluated independently.
    fn visit_matching<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        assert_eq!(self.width(), hint.values.len());
        self.visit(ws, |r| {
            if r.compatible(hint) {
                emit(r)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
}
impl<P: Plan + ?Sized> Plan for &P {
    type Workspace = P::Workspace;
    fn width(&self) -> usize {
        (**self).width()
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        (**self).possible_variables()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        (**self).visit(ws, emit)
    }
    fn visit_matching<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        (**self).visit_matching(ws, hint, emit)
    }
}
/// VALUES is a multiset. Empty mappings and duplicate rows are retained.
#[derive(Clone)]
pub struct Values {
    width: usize,
    rows: Vec<Row>,
    possible: Vec<u64>,
}
impl Values {
    pub fn new(width: usize, rows: Vec<Vec<Option<u32>>>) -> Result<Self> {
        let mut out = Vec::with_capacity(rows.len());
        for values in rows {
            if values.len() != width {
                return Err(TreeError::InvalidConfig("VALUES width mismatch".into()));
            }
            let mut row = Row::empty(width);
            for (v, x) in values.into_iter().enumerate() {
                if let Some(x) = x {
                    row.set(v, x);
                }
            }
            out.push(row);
        }
        let mut possible = vec![0; width.div_ceil(64)];
        for row in &out {
            for (p, &b) in possible.iter_mut().zip(&row.bound) {
                *p |= b;
            }
        }
        Ok(Self {
            width,
            rows: out,
            possible,
        })
    }
    pub fn unit(width: usize) -> Self {
        Self {
            width,
            rows: vec![Row::empty(width)],
            possible: vec![0; width.div_ceil(64)],
        }
    }
}
impl Plan for Values {
    type Workspace = ();
    fn width(&self) -> usize {
        self.width
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        Some(&self.possible)
    }
    fn visit<T>(
        &self,
        _: &mut (),
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        for r in &self.rows {
            emit(r.view())?;
        }
        ControlFlow::Continue(())
    }
}
/// Adapter for the compiled BGP machine. Leaf schemas may contain only a subset
/// of the global variable namespace; local IDs are densely remapped once.
pub struct Bgp<S: TripleSource> {
    machine: ScheduledJoin<S>,
    width: usize,
    variables: Vec<usize>,
    possible: Vec<u64>,
}
pub struct BgpWorkspace<C: Copy + Default> {
    machine: ScheduleWorkspace<C>,
    row: Row,
    parameters: Vec<Option<u32>>,
}
impl<C: Copy + Default> Default for BgpWorkspace<C> {
    fn default() -> Self {
        Self {
            machine: ScheduleWorkspace::default(),
            row: Row::default(),
            parameters: Vec::new(),
        }
    }
}
impl<S: TripleSource> Bgp<S> {
    pub fn planning(&self) -> &crate::scheduled_join::PlanningStats {
        &self.machine.planning
    }
    pub fn allocated_bytes(&self) -> usize {
        self.machine.allocated_bytes()
            + self.variables.capacity() * std::mem::size_of::<usize>()
            + self.possible.capacity() * 8
    }
    pub fn new(query: &JoinQuery<Term>, source: S, policy: Policy) -> Result<Self> {
        Self::prepare(query, source, policy, &Uncontrolled).map_err(PreparationError::uncontrolled)
    }
    pub fn new_controlled(
        query: &JoinQuery<Term>,
        source: S,
        policy: Policy,
        control: &Execution<'_>,
    ) -> std::result::Result<Self, PreparationError> {
        control.check()?;
        let plan = Self::prepare(query, source, policy, control)?;
        control.check()?;
        Ok(plan)
    }
    fn prepare(
        query: &JoinQuery<Term>,
        source: S,
        policy: Policy,
        control: &impl Check,
    ) -> std::result::Result<Self, PreparationError> {
        control.checkpoint()?;
        let mut used = vec![false; query.num_vars];
        for p in &query.patterns {
            control.checkpoint()?;
            for t in [p.subject, p.predicate, p.object] {
                if let Term::Var(v) = t {
                    if v >= used.len() {
                        return Err(TreeError::InvalidConfig(
                            "BGP variable outside global schema".into(),
                        )
                        .into());
                    }
                    used[v] = true;
                }
            }
        }
        let variables: Vec<_> = used
            .iter()
            .enumerate()
            .filter_map(|(v, &u)| u.then_some(v))
            .collect();
        let mut remap = vec![0; query.num_vars];
        for (local, &v) in variables.iter().enumerate() {
            remap[v] = local;
        }
        let term = |t| match t {
            Term::Var(v) => Term::Var(remap[v]),
            x => x,
        };
        let q = JoinQuery {
            num_vars: variables.len(),
            patterns: query
                .patterns
                .iter()
                .map(|p| TriplePattern {
                    subject: term(p.subject),
                    predicate: term(p.predicate),
                    object: term(p.object),
                })
                .collect(),
        };
        let mut possible = vec![0; query.num_vars.div_ceil(64)];
        for &v in &variables {
            possible[v / 64] |= 1u64 << (v % 64);
        }
        Ok(Self {
            machine: ScheduledJoin::prepare(&q, source, policy, control)?,
            width: query.num_vars,
            variables,
            possible,
        })
    }
}
impl<S: TripleSource> Plan for Bgp<S> {
    type Workspace = BgpWorkspace<S::Cursor>;
    fn width(&self) -> usize {
        self.width
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        Some(&self.possible)
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        // Sorted global IDs are the identity mapping when every slot is used.
        // Keep the register machine's borrowed row rather than copying it into
        // a second full-width row on the primary public BGP search path.
        if self.variables.len() == self.width {
            return self.machine.visit(&mut ws.machine, |values| {
                emit(Binding {
                    values,
                    bound: &self.possible,
                })
            });
        }
        ws.row.reset(self.width);
        for &v in &self.variables {
            ws.row.set(v, 0);
        }
        self.machine.visit(&mut ws.machine, |local| {
            for (i, &v) in self.variables.iter().enumerate() {
                ws.row.values[v] = local[i];
            }
            emit(ws.row.view())
        })
    }
    fn visit_matching<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        assert_eq!(hint.values.len(), self.width);
        ws.parameters.clear();
        ws.parameters
            .extend(self.variables.iter().map(|&v| hint.get(v)));
        if self.variables.len() == self.width {
            return self
                .machine
                .visit_matching(&ws.parameters, &mut ws.machine, |values| {
                    emit(Binding {
                        values,
                        bound: &self.possible,
                    })
                });
        }
        ws.row.reset(self.width);
        for &v in &self.variables {
            ws.row.set(v, 0);
        }
        self.machine
            .visit_matching(&ws.parameters, &mut ws.machine, |local| {
                for (i, &v) in self.variables.iter().enumerate() {
                    ws.row.values[v] = local[i];
                }
                emit(ws.row.view())
            })
    }
}
pub struct BinaryWorkspace<A: Default, B: Default> {
    left: A,
    right: B,
    merged: Row,
}
impl<A: Default, B: Default> Default for BinaryWorkspace<A, B> {
    fn default() -> Self {
        Self {
            left: A::default(),
            right: B::default(),
            merged: Row::default(),
        }
    }
}
pub trait Condition {
    fn test(&self, row: Binding<'_>) -> bool;
}
impl<F: Fn(Binding<'_>) -> bool> Condition for F {
    fn test(&self, row: Binding<'_>) -> bool {
        self(row)
    }
}
#[derive(Clone, Copy)]
pub struct Always;
impl Condition for Always {
    fn test(&self, _: Binding<'_>) -> bool {
        true
    }
}
pub struct Join<A, B> {
    pub left: A,
    pub right: B,
}
pub struct LeftJoin<A, B, F = Always> {
    pub left: A,
    pub right: B,
    pub condition: F,
}
impl<A, B> LeftJoin<A, B> {
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            condition: Always,
        }
    }
}
impl<A: Plan, B: Plan> Plan for Join<A, B> {
    type Workspace = BinaryWorkspace<A::Workspace, B::Workspace>;
    fn width(&self) -> usize {
        assert_eq!(self.left.width(), self.right.width());
        self.left.width()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.width();
        self.left.visit(&mut ws.left, |a| {
            self.right.visit_matching(&mut ws.right, a, |b| {
                ws.merged.merge(a, b);
                emit(ws.merged.view())
            })
        })
    }
}
impl<A: Plan, B: Plan, F: Condition> Plan for LeftJoin<A, B, F> {
    type Workspace = BinaryWorkspace<A::Workspace, B::Workspace>;
    fn width(&self) -> usize {
        assert_eq!(self.left.width(), self.right.width());
        self.left.width()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.width();
        self.left.visit(&mut ws.left, |a| {
            let mut matched = false;
            self.right.visit_matching(&mut ws.right, a, |b| {
                ws.merged.merge(a, b);
                if self.condition.test(ws.merged.view()) {
                    matched = true;
                    emit(ws.merged.view())
                } else {
                    ControlFlow::Continue(())
                }
            })?;
            if matched {
                ControlFlow::Continue(())
            } else {
                emit(a)
            }
        })
    }
}
pub struct UnionAll<A, B> {
    pub left: A,
    pub right: B,
}
impl<A: Plan, B: Plan> Plan for UnionAll<A, B> {
    type Workspace = (A::Workspace, B::Workspace);
    fn width(&self) -> usize {
        assert_eq!(self.left.width(), self.right.width());
        self.left.width()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.width();
        self.left.visit(&mut ws.0, &mut emit)?;
        self.right.visit(&mut ws.1, emit)
    }
}
/// SPARQL MINUS: a matching RHS row must share at least one bound variable.
/// The RHS relation is evaluated independently; hints only index its rows.
pub struct AntiJoin<A, B> {
    pub left: A,
    pub right: B,
}
pub type Minus<A, B> = AntiJoin<A, B>;
impl<A: Plan, B: Plan> Plan for AntiJoin<A, B> {
    type Workspace = (A::Workspace, B::Workspace);
    fn width(&self) -> usize {
        assert_eq!(self.left.width(), self.right.width());
        self.left.width()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.width();
        self.left.visit(&mut ws.0, |a| {
            if self
                .right
                .possible_variables()
                .is_some_and(|vars| !a.bound.iter().zip(vars).any(|(&a, &b)| a & b != 0))
            {
                return emit(a);
            }
            let found = self
                .right
                .visit_matching(&mut ws.1, a, |b| {
                    if a.overlaps(b) {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .is_break();
            if found {
                ControlFlow::Continue(())
            } else {
                emit(a)
            }
        })
    }
}
pub struct Filter<P, F> {
    pub input: P,
    pub condition: F,
}
impl<P: Plan, F: Condition> Plan for Filter<P, F> {
    type Workspace = P::Workspace;
    fn width(&self) -> usize {
        self.input.width()
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        self.input.possible_variables()
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.input.visit(ws, |r| {
            if self.condition.test(r) {
                emit(r)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
    fn visit_matching<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.input.visit_matching(ws, hint, |r| {
            if self.condition.test(r) {
                emit(r)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
}
/// Correlation is explicit and cannot accidentally change MINUS filter scope.
pub trait CorrelatedPlan {
    type Workspace: Default;
    fn correlation_width(&self) -> usize;
    fn probe(&self, ws: &mut Self::Workspace, outer: Binding<'_>) -> bool;
}
impl<P: Plan> CorrelatedPlan for P {
    type Workspace = P::Workspace;
    fn correlation_width(&self) -> usize {
        Plan::width(self)
    }
    fn probe(&self, ws: &mut Self::Workspace, outer: Binding<'_>) -> bool {
        self.visit_matching(ws, outer, |_| ControlFlow::Break(()))
            .is_break()
    }
}
pub struct CorrelatedFilter<P, F> {
    pub input: P,
    pub condition: F,
}
impl<P: Plan, F: Fn(Binding<'_>, Binding<'_>) -> bool> CorrelatedPlan for CorrelatedFilter<P, F> {
    type Workspace = P::Workspace;
    fn correlation_width(&self) -> usize {
        self.input.width()
    }
    fn probe(&self, ws: &mut Self::Workspace, outer: Binding<'_>) -> bool {
        self.input
            .visit_matching(ws, outer, |r| {
                if (self.condition)(outer, r) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .is_break()
    }
}
pub struct Existence<A, B, const NEGATED: bool> {
    pub left: A,
    pub right: B,
}
pub type Exists<A, B> = Existence<A, B, false>;
pub type NotExists<A, B> = Existence<A, B, true>;
impl<A: Plan, B: CorrelatedPlan, const NEGATED: bool> Plan for Existence<A, B, NEGATED> {
    type Workspace = (A::Workspace, B::Workspace);
    fn width(&self) -> usize {
        assert_eq!(
            Plan::width(&self.left),
            CorrelatedPlan::correlation_width(&self.right)
        );
        Plan::width(&self.left)
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.width();
        self.left.visit(&mut ws.0, |a| {
            if self.right.probe(&mut ws.1, a) != NEGATED {
                emit(a)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
}
#[cfg(test)]
#[path = "query_ops_tests.rs"]
mod tests;

#[path = "controlled_query.rs"]
mod controlled;
pub use controlled::{ControlledCorrelatedPlan, ControlledPlan};
