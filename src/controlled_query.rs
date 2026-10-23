//! Opt-in controlled search extension. Existing custom `Plan` implementations
//! remain source compatible; implementing this trait promises internal checks,
//! including work that never reaches a result callback.
use super::*;
use crate::execution::{StopReason, VisitResult, VisitStop};

pub(crate) fn checkpoint<T>(control: &Execution<'_>) -> VisitResult<T> {
    match control.checkpoint() {
        Ok(()) => ControlFlow::Continue(()),
        Err(reason) => ControlFlow::Break(VisitStop::Interrupted(reason)),
    }
}
fn begin<T>(control: &Execution<'_>) -> VisitResult<T> {
    if let Err(reason) = control.check() {
        return ControlFlow::Break(VisitStop::Interrupted(reason));
    }
    checkpoint(control)
}
fn flatten<T>(result: VisitResult<VisitStop<T>>) -> VisitResult<T> {
    result.map_break(|stop| match stop {
        VisitStop::User(stop) => stop,
        VisitStop::Interrupted(reason) => VisitStop::Interrupted(reason),
    })
}

/// Searches that cooperatively check request limits inside their execution.
/// Custom implementations must check at entry and within bounded units of
/// no-output work. `visit_matching_controlled` may use filtering by default;
/// index-backed implementations should override it to push down bindings.
///
/// This is deliberately separate from `Plan`: wrapping result callbacks alone
/// cannot provide cancellation for a custom algorithm's unsuccessful search.
pub trait ControlledPlan: Plan {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T>;

    fn visit_matching_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        assert_eq!(self.width(), hint.values.len());
        self.visit_controlled(ws, control, |r| {
            if r.compatible(hint) {
                emit(r)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
}
impl<P: ControlledPlan + ?Sized> ControlledPlan for &P {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        (**self).visit_controlled(ws, control, emit)
    }
    fn visit_matching_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        control: &Execution<'_>,
        emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        (**self).visit_matching_controlled(ws, hint, control, emit)
    }
}
impl ControlledPlan for Values {
    fn visit_controlled<T>(
        &self,
        _: &mut (),
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        for row in &self.rows {
            checkpoint(control)?;
            emit(row.view()).map_break(VisitStop::User)?;
        }
        match control.check() {
            Ok(()) => ControlFlow::Continue(()),
            Err(reason) => ControlFlow::Break(VisitStop::Interrupted(reason)),
        }
    }
}
impl<S: TripleSource> crate::automata::ControlledCompile<S> for Values {
    fn compile_controlled(
        &self,
        _: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self, PreparationError> {
        control.check()?;
        control.checkpoint()?;
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            control.checkpoint()?;
            rows.push(row.clone());
        }
        let compiled = Self {
            width: self.width,
            rows,
            possible: self.possible.clone(),
        };
        control.check()?;
        Ok(compiled)
    }
}
impl<S: TripleSource> ControlledPlan for Bgp<S> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        if self.variables.len() == self.width {
            return self
                .machine
                .visit_controlled(&mut ws.machine, control, |values| {
                    emit(Binding {
                        values,
                        bound: &self.possible,
                    })
                });
        }
        begin(control)?;
        ws.row.reset(self.width);
        for &v in &self.variables {
            ws.row.set(v, 0);
        }
        self.machine
            .visit_controlled(&mut ws.machine, control, |local| {
                for (i, &v) in self.variables.iter().enumerate() {
                    ws.row.values[v] = local[i];
                }
                emit(ws.row.view())
            })
    }
    fn visit_matching_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        assert_eq!(hint.values.len(), self.width);
        begin(control)?;
        ws.parameters.clear();
        for &v in &self.variables {
            checkpoint(control)?;
            ws.parameters.push(hint.get(v));
        }
        if self.variables.len() == self.width {
            return self.machine.visit_matching_controlled(
                &ws.parameters,
                &mut ws.machine,
                control,
                |values| {
                    emit(Binding {
                        values,
                        bound: &self.possible,
                    })
                },
            );
        }
        ws.row.reset(self.width);
        for &v in &self.variables {
            ws.row.set(v, 0);
        }
        self.machine
            .visit_matching_controlled(&ws.parameters, &mut ws.machine, control, |local| {
                for (i, &v) in self.variables.iter().enumerate() {
                    ws.row.values[v] = local[i];
                }
                emit(ws.row.view())
            })
    }
}
impl<A: ControlledPlan, B: ControlledPlan> ControlledPlan for Join<A, B> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        self.width();
        flatten(self.left.visit_controlled(&mut ws.left, control, |a| {
            self.right
                .visit_matching_controlled(&mut ws.right, a, control, |b| {
                    ws.merged.merge(a, b);
                    emit(ws.merged.view())
                })
        }))
    }
}
impl<A: ControlledPlan, B: ControlledPlan, F: Condition> ControlledPlan for LeftJoin<A, B, F> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        self.width();
        flatten(self.left.visit_controlled(&mut ws.left, control, |a| {
            let mut matched = false;
            self.right
                .visit_matching_controlled(&mut ws.right, a, control, |b| {
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
                emit(a).map_break(VisitStop::User)
            }
        }))
    }
}
impl<A: ControlledPlan, B: ControlledPlan> ControlledPlan for UnionAll<A, B> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        self.width();
        self.left.visit_controlled(&mut ws.0, control, &mut emit)?;
        self.right.visit_controlled(&mut ws.1, control, emit)
    }
}
impl<A: ControlledPlan, B: ControlledPlan> ControlledPlan for AntiJoin<A, B> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        self.width();
        flatten(self.left.visit_controlled(&mut ws.0, control, |a| {
            if self
                .right
                .possible_variables()
                .is_some_and(|vars| !a.bound.iter().zip(vars).any(|(&a, &b)| a & b != 0))
            {
                return emit(a).map_break(VisitStop::User);
            }
            match self
                .right
                .visit_matching_controlled(&mut ws.1, a, control, |b| {
                    if a.overlaps(b) {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                }) {
                ControlFlow::Break(VisitStop::Interrupted(reason)) => {
                    ControlFlow::Break(VisitStop::Interrupted(reason))
                }
                ControlFlow::Break(VisitStop::User(())) => ControlFlow::Continue(()),
                ControlFlow::Continue(()) => emit(a).map_break(VisitStop::User),
            }
        }))
    }
}
impl<P: ControlledPlan, F: Condition> ControlledPlan for Filter<P, F> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        self.input.visit_controlled(ws, control, |r| {
            if self.condition.test(r) {
                emit(r)
            } else {
                ControlFlow::Continue(())
            }
        })
    }
    fn visit_matching_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        hint: Binding<'_>,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        self.input
            .visit_matching_controlled(ws, hint, control, |r| {
                if self.condition.test(r) {
                    emit(r)
                } else {
                    ControlFlow::Continue(())
                }
            })
    }
}

/// Controlled correlation must distinguish an interrupted probe from absence.
pub trait ControlledCorrelatedPlan: CorrelatedPlan {
    fn probe_controlled(
        &self,
        ws: &mut Self::Workspace,
        outer: Binding<'_>,
        control: &Execution<'_>,
    ) -> std::result::Result<bool, StopReason>;
}
fn found(result: VisitResult<()>) -> std::result::Result<bool, StopReason> {
    match result {
        ControlFlow::Continue(()) => Ok(false),
        ControlFlow::Break(VisitStop::User(())) => Ok(true),
        ControlFlow::Break(VisitStop::Interrupted(reason)) => Err(reason),
    }
}
impl<P: ControlledPlan> ControlledCorrelatedPlan for P {
    fn probe_controlled(
        &self,
        ws: &mut Self::Workspace,
        outer: Binding<'_>,
        control: &Execution<'_>,
    ) -> std::result::Result<bool, StopReason> {
        found(self.visit_matching_controlled(ws, outer, control, |_| ControlFlow::Break(())))
    }
}
impl<P: ControlledPlan, F: Fn(Binding<'_>, Binding<'_>) -> bool> ControlledCorrelatedPlan
    for CorrelatedFilter<P, F>
{
    fn probe_controlled(
        &self,
        ws: &mut Self::Workspace,
        outer: Binding<'_>,
        control: &Execution<'_>,
    ) -> std::result::Result<bool, StopReason> {
        found(
            self.input
                .visit_matching_controlled(ws, outer, control, |r| {
                    if (self.condition)(outer, r) {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                }),
        )
    }
}
impl<A: ControlledPlan, B: ControlledCorrelatedPlan, const N: bool> ControlledPlan
    for Existence<A, B, N>
{
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        begin(control)?;
        self.width();
        flatten(self.left.visit_controlled(&mut ws.0, control, |a| {
            match self.right.probe_controlled(&mut ws.1, a, control) {
                Err(reason) => ControlFlow::Break(VisitStop::Interrupted(reason)),
                Ok(found) if found != N => emit(a).map_break(VisitStop::User),
                Ok(_) => ControlFlow::Continue(()),
            }
        }))
    }
}
