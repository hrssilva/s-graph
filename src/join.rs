//! Adaptive FST joins, the explicit mutable-tree native executor, and the
//! scan interface used by path algorithms. No WCO guarantee is provided.

use crate::Triple;
pub use crate::native_join::{NativeJoin, NativeJoinStats, NativeJoinWorkspace};

/// Default adaptive automaton join over the immutable FST.
#[cfg(feature = "heap")]
pub struct Join<'a>(crate::scheduled_join::HybridJoin<'a>);

#[cfg(feature = "heap")]
pub type JoinWorkspace = crate::scheduled_join::ScheduleWorkspace;

#[cfg(feature = "heap")]
impl<'a> Join<'a> {
    pub fn new<P: Copy + Into<crate::graph::Term>>(
        query: &crate::graph::JoinQuery<P>,
        graph: &'a crate::Fst,
    ) -> crate::Result<Self> {
        Ok(Self(crate::scheduled_join::HybridJoin::new(
            query,
            graph,
            crate::scheduled_join::Policy::Adaptive,
        )?))
    }
}

#[cfg(feature = "heap")]
impl<'a> std::ops::Deref for Join<'a> {
    type Target = crate::scheduled_join::HybridJoin<'a>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Monomorphized, allocation-free triple scans with optional coordinate bounds.
pub trait PatternBackend {
    type Scan<'a>: Iterator<Item = Triple>
    where
        Self: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_>;
}

impl<S: crate::BlockStorage> PatternBackend
    for crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>
{
    type Scan<'a>
        = crate::tree::Scan<'a, S>
    where
        S: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        self.scan(bounds)
    }
}
