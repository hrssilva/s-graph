//! Build automata independently of storage, compose them, then `Fst::search`.
//! BGPs are register automata; OPTIONAL/MINUS compose binding relations, while
//! intersection/union/complement compose recognizers over one typed alphabet.
pub use crate::automata_core::{
    And, Automaton, BitSupport, BitmapAutomaton, BitmapSource, FstSource, GraphAutomaton, MapInput,
    Mask, Not, Or, Project, Reducer, ReversibleAction, Slot, Transducer, TripleSource, Unfiltered,
};
pub use crate::automaton::PathAutomaton;
pub use crate::execution::{Execution, PreparationError, StopReason, VisitResult, VisitStop};
pub use crate::graph::Term;
use crate::graph::{JoinQuery, TriplePattern};
pub use crate::mutable_fst::MutableFstSource;
pub use crate::path_ops::{Composed, PathBindings, PathEvent, PathExpr, PathRunner};
use crate::query_ops::{self as ops, Condition, Plan};
pub use crate::query_ops::{Binding, Plan as Search, Row, Values};
pub use crate::query_ops::{ControlledCorrelatedPlan, ControlledPlan as ControlledSearch};
pub use crate::scheduled_join::{Policy as Schedule, ScheduleStats, ScheduleWorkspace};
pub use crate::tree_source::{TreeCursor, TreeSource};
use crate::{Result, TreeError};
use std::ops::ControlFlow;

/// Compile a reusable search. The source and any borrowed constituent
/// automata determine the resulting plan's lifetimes.
pub trait Compile<S: TripleSource> {
    type Compiled: Search;
    fn compile(&self, source: S) -> Result<Self::Compiled>;
}
/// Opt-in compilation with request checkpoints, including adaptive probes.
/// Custom builders can implement this without changing their ordinary Compile
/// implementation. Check inside long construction loops, not only on return.
pub trait ControlledCompile<S: TripleSource>: Compile<S> {
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError>;
}
impl<S: TripleSource, Q: ControlledCompile<S> + ?Sized> ControlledCompile<S> for &Q {
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        (**self).compile_controlled(source, control)
    }
}
impl<S: TripleSource, Q: Compile<S> + ?Sized> Compile<S> for &Q {
    type Compiled = Q::Compiled;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        (**self).compile(source)
    }
}

/// A conjunction of triple languages in one global binding namespace.
/// Add all conjunctive patterns here so compilation can fuse their constraints.
#[derive(Clone)]
pub struct Bgp {
    width: usize,
    patterns: Vec<TriplePattern<Term>>,
    schedule: Schedule,
    compatible: bool,
}
impl Bgp {
    pub fn new(variables: usize) -> Self {
        Self {
            width: variables,
            patterns: Vec::new(),
            schedule: Schedule::Adaptive,
            compatible: true,
        }
    }
    pub fn triple(mut self, terms: [Term; 3]) -> Self {
        self.patterns.push(TriplePattern {
            subject: terms[0],
            predicate: terms[1],
            object: terms[2],
        });
        self
    }
    /// Fuse conjunction before compiling. The left schedule governs the merged
    /// automaton; call `schedule` after composition to override it explicitly.
    pub fn and(mut self, other: Self) -> Self {
        self.compatible &= other.compatible && self.width == other.width;
        self.patterns.extend(other.patterns);
        self
    }
    pub fn schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = schedule;
        self
    }
    pub fn patterns(&self) -> &[TriplePattern<Term>] {
        &self.patterns
    }
}
impl<S: TripleSource> Compile<S> for Bgp {
    type Compiled = ops::Bgp<S>;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        if !self.compatible {
            return Err(TreeError::InvalidConfig(
                "automata binding widths differ".into(),
            ));
        }
        ops::Bgp::new(
            &JoinQuery {
                num_vars: self.width,
                patterns: self.patterns.clone(),
            },
            source,
            self.schedule,
        )
    }
}
impl<S: TripleSource> ControlledCompile<S> for Bgp {
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        control.check()?;
        if !self.compatible {
            return Err(TreeError::InvalidConfig("automata binding widths differ".into()).into());
        }
        let mut patterns = Vec::with_capacity(self.patterns.len());
        for pattern in &self.patterns {
            control.checkpoint()?;
            patterns.push(pattern.clone());
        }
        ops::Bgp::new_controlled(
            &JoinQuery {
                num_vars: self.width,
                patterns,
            },
            source,
            self.schedule,
            control,
        )
    }
}
/// Fluent composition of binding-producing automata. All operands use the same
/// global width; OPTIONAL can leave slots unbound and union preserves bags.
pub trait Query: Sized {
    fn binding_width(&self) -> usize;
    fn join<B: Query>(self, right: B) -> ops::Join<Self, B> {
        ops::Join { left: self, right }
    }
    fn optional<B: Query>(self, right: B) -> ops::LeftJoin<Self, B> {
        ops::LeftJoin::new(self, right)
    }
    fn optional_if<B: Query, F: Condition>(
        self,
        right: B,
        condition: F,
    ) -> ops::LeftJoin<Self, B, F> {
        ops::LeftJoin {
            left: self,
            right,
            condition,
        }
    }
    fn minus<B: Query>(self, right: B) -> ops::Minus<Self, B> {
        ops::Minus { left: self, right }
    }
    fn union_all<B: Query>(self, right: B) -> ops::UnionAll<Self, B> {
        ops::UnionAll { left: self, right }
    }
    fn exists<B: Query>(self, right: B) -> ops::Exists<Self, B> {
        ops::Exists { left: self, right }
    }
    fn not_exists<B: Query>(self, right: B) -> ops::NotExists<Self, B> {
        ops::NotExists { left: self, right }
    }
    fn filter<F: Condition>(self, condition: F) -> ops::Filter<Self, F> {
        ops::Filter {
            input: self,
            condition,
        }
    }
}
impl Query for Bgp {
    fn binding_width(&self) -> usize {
        self.width
    }
}
impl Query for Values {
    fn binding_width(&self) -> usize {
        Plan::width(self)
    }
}
impl<S: TripleSource> Compile<S> for Values {
    type Compiled = Self;
    fn compile(&self, _: S) -> Result<Self> {
        Ok(self.clone())
    }
}
impl<Q: Query> Query for &Q {
    fn binding_width(&self) -> usize {
        (**self).binding_width()
    }
}
fn same_width<A: Query, B: Query>(a: &A, b: &B) -> Result<()> {
    if a.binding_width() == b.binding_width() {
        Ok(())
    } else {
        Err(TreeError::InvalidConfig(
            "automata binding widths differ".into(),
        ))
    }
}
macro_rules! binary {
    ($name:ident) => {
        impl<A: Query, B: Query> Query for ops::$name<A, B> {
            fn binding_width(&self) -> usize {
                self.left.binding_width()
            }
        }
        impl<S: TripleSource + Copy, A: Query + Compile<S>, B: Query + Compile<S>> Compile<S>
            for ops::$name<A, B>
        {
            type Compiled = ops::$name<A::Compiled, B::Compiled>;
            fn compile(&self, source: S) -> Result<Self::Compiled> {
                same_width(&self.left, &self.right)?;
                Ok(ops::$name {
                    left: self.left.compile(source)?,
                    right: self.right.compile(source)?,
                })
            }
        }
        impl<
            S: TripleSource + Copy,
            A: Query + ControlledCompile<S>,
            B: Query + ControlledCompile<S>,
        > ControlledCompile<S> for ops::$name<A, B>
        {
            fn compile_controlled(
                &self,
                source: S,
                control: &Execution<'_>,
            ) -> std::result::Result<Self::Compiled, PreparationError> {
                control.check()?;
                control.checkpoint()?;
                same_width(&self.left, &self.right)?;
                let compiled = ops::$name {
                    left: self.left.compile_controlled(source, control)?,
                    right: self.right.compile_controlled(source, control)?,
                };
                control.check()?;
                Ok(compiled)
            }
        }
    };
}
binary!(Join);
binary!(AntiJoin);
binary!(UnionAll);
impl<A: Query, B: Query, F> Query for ops::LeftJoin<A, B, F> {
    fn binding_width(&self) -> usize {
        self.left.binding_width()
    }
}
impl<S: TripleSource + Copy, A: Query + Compile<S>, B: Query + Compile<S>, F: Condition + Clone>
    Compile<S> for ops::LeftJoin<A, B, F>
{
    type Compiled = ops::LeftJoin<A::Compiled, B::Compiled, F>;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        same_width(&self.left, &self.right)?;
        Ok(ops::LeftJoin {
            left: self.left.compile(source)?,
            right: self.right.compile(source)?,
            condition: self.condition.clone(),
        })
    }
}
impl<
    S: TripleSource + Copy,
    A: Query + ControlledCompile<S>,
    B: Query + ControlledCompile<S>,
    F: Condition + Clone,
> ControlledCompile<S> for ops::LeftJoin<A, B, F>
{
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        control.check()?;
        control.checkpoint()?;
        same_width(&self.left, &self.right)?;
        let compiled = ops::LeftJoin {
            left: self.left.compile_controlled(source, control)?,
            right: self.right.compile_controlled(source, control)?,
            condition: self.condition.clone(),
        };
        control.check()?;
        Ok(compiled)
    }
}
impl<A: Query, B: Query, const N: bool> Query for ops::Existence<A, B, N> {
    fn binding_width(&self) -> usize {
        self.left.binding_width()
    }
}
impl<S: TripleSource + Copy, A: Query + Compile<S>, B: Query + Compile<S>, const N: bool> Compile<S>
    for ops::Existence<A, B, N>
{
    type Compiled = ops::Existence<A::Compiled, B::Compiled, N>;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        same_width(&self.left, &self.right)?;
        Ok(ops::Existence {
            left: self.left.compile(source)?,
            right: self.right.compile(source)?,
        })
    }
}
impl<
    S: TripleSource + Copy,
    A: Query + ControlledCompile<S>,
    B: Query + ControlledCompile<S>,
    const N: bool,
> ControlledCompile<S> for ops::Existence<A, B, N>
{
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        control.check()?;
        control.checkpoint()?;
        same_width(&self.left, &self.right)?;
        let compiled = ops::Existence {
            left: self.left.compile_controlled(source, control)?,
            right: self.right.compile_controlled(source, control)?,
        };
        control.check()?;
        Ok(compiled)
    }
}
impl<P: Query, F> Query for ops::Filter<P, F> {
    fn binding_width(&self) -> usize {
        self.input.binding_width()
    }
}
impl<S: TripleSource, P: Compile<S>, F: Condition + Clone> Compile<S> for ops::Filter<P, F> {
    type Compiled = ops::Filter<P::Compiled, F>;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        Ok(ops::Filter {
            input: self.input.compile(source)?,
            condition: self.condition.clone(),
        })
    }
}
impl<S: TripleSource, P: ControlledCompile<S>, F: Condition + Clone> ControlledCompile<S>
    for ops::Filter<P, F>
{
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        control.check()?;
        control.checkpoint()?;
        let compiled = ops::Filter {
            input: self.input.compile_controlled(source, control)?,
            condition: self.condition.clone(),
        };
        control.check()?;
        Ok(compiled)
    }
}

/// Adapt a packed-slot recognizer into a relation of `[subject,predicate,object]`.
/// Wrap ordinary `Automaton<Slot>` implementations in `Unfiltered` if they do
/// not provide a conservative bitmap mask. Complement must use that adapter.
#[derive(Clone)]
pub struct Language<A>(pub A);
pub struct LanguageSearch<S, A> {
    source: S,
    machine: A,
}
impl<A> Query for Language<A> {
    fn binding_width(&self) -> usize {
        3
    }
}
impl<S: TripleSource, A: BitmapAutomaton + Clone> Compile<S> for Language<A> {
    type Compiled = LanguageSearch<S, A>;
    fn compile(&self, source: S) -> Result<Self::Compiled> {
        if source.height() > crate::tree::MAX_HEIGHT {
            return Err(TreeError::InvalidConfig(
                "source exceeds automaton traversal height".into(),
            ));
        }
        Ok(LanguageSearch {
            source,
            machine: self.0.clone(),
        })
    }
}
impl<S: TripleSource, A: BitmapAutomaton + Clone> ControlledCompile<S> for Language<A> {
    fn compile_controlled(
        &self,
        source: S,
        control: &Execution<'_>,
    ) -> std::result::Result<Self::Compiled, PreparationError> {
        control.check()?;
        control.checkpoint()?;
        let search = self.compile(source)?;
        control.check()?;
        Ok(search)
    }
}
impl<S: TripleSource, A: BitmapAutomaton> Plan for LanguageSearch<S, A> {
    type Workspace = Vec<Slot>;
    fn width(&self) -> usize {
        3
    }
    fn possible_variables(&self) -> Option<&[u64]> {
        Some(&[7])
    }
    fn visit<T>(
        &self,
        ws: &mut Self::Workspace,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        crate::automata_core::visit_language_with_workspace(
            &self.source,
            &self.machine,
            &(),
            ws,
            |path| {
                let mut values = [0u32; 3];
                for slot in path {
                    let (bits, shifts, remaining) = self.source.layout(slot.level);
                    for d in 0..3 {
                        values[d] |=
                            ((slot.value >> shifts[d]) & ((1u32 << bits[d]) - 1)) << remaining[d];
                    }
                }
                emit(Binding {
                    values: &values,
                    bound: &[7],
                })
            },
        )
    }
}
impl<S: TripleSource, A: BitmapAutomaton> ControlledSearch for LanguageSearch<S, A> {
    fn visit_controlled<T>(
        &self,
        ws: &mut Self::Workspace,
        control: &Execution<'_>,
        mut emit: impl FnMut(Binding<'_>) -> ControlFlow<T>,
    ) -> VisitResult<T> {
        crate::automata_core::visit_language_controlled(
            &self.source,
            &self.machine,
            &(),
            ws,
            control,
            |path| {
                let mut values = [0u32; 3];
                for slot in path {
                    let (bits, shifts, remaining) = self.source.layout(slot.level);
                    for d in 0..3 {
                        values[d] |=
                            ((slot.value >> shifts[d]) & ((1u32 << bits[d]) - 1)) << remaining[d];
                    }
                }
                emit(Binding {
                    values: &values,
                    bound: &[7],
                })
            },
        )
    }
}

/// Construction-only conveniences; execution remains statically dispatched.
pub trait AutomatonExt: Sized {
    fn intersection<B>(self, other: B) -> And<Self, B> {
        And(self, other)
    }
    fn union<B>(self, other: B) -> Or<Self, B> {
        Or(self, other)
    }
    fn complement(self) -> Not<Self> {
        Not(self)
    }
}
impl<A> AutomatonExt for A {}
