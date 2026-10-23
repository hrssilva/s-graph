//! Opt-in cooperative execution limits, shared by planning and nested searches.
//!
//! One work unit is an executor checkpoint, not a node read or a time estimate.
//! Budgets are exact; cancellation/deadline polling is amortized across 256
//! checkpoints. Custom algorithms choose their own bounded checkpoint locations.
use crate::TreeError;
use std::{
    cell::Cell,
    fmt,
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Cancelled,
    Deadline,
    WorkBudget,
}

/// A caller's callback break and an interrupted, incomplete search are distinct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisitStop<T> {
    User(T),
    Interrupted(StopReason),
}
pub type VisitResult<T> = ControlFlow<VisitStop<T>>;

pub(crate) enum Stop<T> {
    User(T),
    Budget,
    Interrupted(StopReason),
}

/// Per-request cooperative limits. Share by reference across nested operations;
/// the work allowance is cumulative, including controlled compilation/probes.
/// Use a fresh value for a new request. A stop is sticky for this request.
///
/// Cancellation may come from another thread through an `AtomicBool` (including
/// one inside an `Arc`). This object and its work accounting stay on the worker.
/// Checkpoints cannot interrupt blocking backend I/O or user callbacks.
#[derive(Default)]
pub struct Execution<'a> {
    budget: Option<u64>,
    deadline: Option<Instant>,
    cancelled: Option<&'a AtomicBool>,
    work: Cell<u64>,
    stopped: Cell<Option<StopReason>>,
}
impl<'a> Execution<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_budget(mut self, work: u64) -> Self {
        self.budget = Some(work);
        self
    }
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }
    pub fn with_cancellation(mut self, cancelled: &'a AtomicBool) -> Self {
        self.cancelled = Some(cancelled);
        self
    }
    pub fn work(&self) -> u64 {
        self.work.get()
    }
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stopped.get()
    }

    /// Poll cancellation/deadline immediately without spending work. Work
    /// budgets are checked by `checkpoint`, before the next unit is performed.
    #[inline]
    pub fn check(&self) -> std::result::Result<(), StopReason> {
        let reason = self.stopped.get().or_else(|| {
            if self.cancelled.is_some_and(|c| c.load(Ordering::Relaxed)) {
                Some(StopReason::Cancelled)
            } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
                Some(StopReason::Deadline)
            } else {
                None
            }
        });
        self.stopped.set(reason);
        reason.map_or(Ok(()), Err)
    }

    /// Spend one work unit, polling external cancellation/deadline every 256
    /// units. Call inside loops even when they produce no answers or reads.
    #[inline]
    pub fn checkpoint(&self) -> std::result::Result<(), StopReason> {
        if let Some(reason) = self.stopped.get() {
            return Err(reason);
        }
        let work = self.work.get();
        if work & 255 == 0 {
            self.check()?;
        }
        if self.budget.is_some_and(|b| work >= b) {
            self.stopped.set(Some(StopReason::WorkBudget));
            return Err(StopReason::WorkBudget);
        }
        self.work.set(work.saturating_add(1));
        Ok(())
    }
}

#[derive(Debug)]
pub enum PreparationError {
    Invalid(TreeError),
    Interrupted(StopReason),
}
impl From<TreeError> for PreparationError {
    fn from(error: TreeError) -> Self {
        Self::Invalid(error)
    }
}
impl From<StopReason> for PreparationError {
    fn from(reason: StopReason) -> Self {
        Self::Interrupted(reason)
    }
}
impl fmt::Display for PreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => error.fmt(f),
            Self::Interrupted(reason) => write!(f, "query preparation interrupted: {reason:?}"),
        }
    }
}
impl std::error::Error for PreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid(error) => Some(error),
            _ => None,
        }
    }
}
impl PreparationError {
    pub(crate) fn uncontrolled(self) -> TreeError {
        match self {
            Self::Invalid(error) => error,
            Self::Interrupted(_) => unreachable!(),
        }
    }
}

// Static specialization keeps checkpoints out of the existing fast path.
pub(crate) trait Check {
    fn checkpoint(&self) -> std::result::Result<(), StopReason>;
    #[inline(always)]
    fn step<T>(&self) -> ControlFlow<Stop<T>> {
        match self.checkpoint() {
            Ok(()) => ControlFlow::Continue(()),
            Err(reason) => ControlFlow::Break(Stop::Interrupted(reason)),
        }
    }
}
impl Check for Execution<'_> {
    #[inline]
    fn checkpoint(&self) -> std::result::Result<(), StopReason> {
        Execution::checkpoint(self)
    }
}
pub(crate) struct Uncontrolled;
impl Check for Uncontrolled {
    #[inline(always)]
    fn checkpoint(&self) -> std::result::Result<(), StopReason> {
        Ok(())
    }
}

pub(crate) fn visited<T>(control: &Execution<'_>, result: ControlFlow<Stop<T>>) -> VisitResult<T> {
    let result = result.map_break(|stop| match stop {
        Stop::User(value) => VisitStop::User(value),
        Stop::Interrupted(reason) => VisitStop::Interrupted(reason),
        Stop::Budget => {
            unreachable!("internal planning budget escaped traversal")
        }
    });
    if result.is_continue() {
        if let Err(reason) = control.check() {
            return ControlFlow::Break(VisitStop::Interrupted(reason));
        }
    }
    result
}
