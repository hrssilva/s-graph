//! Native graph joins and the scan interface used by path algorithms.
//!
//! [`NativeJoin`] compiles the global Morton schedule and executes directly over
//! the tree. No materialized-relation join engine or WCO guarantee is provided.

use crate::Triple;
pub use crate::native_join::{NativeJoin, NativeJoinStats, NativeJoinWorkspace};

/// Monomorphized, allocation-free triple scans with optional coordinate bounds.
pub trait PatternBackend {
    type Scan<'a>: Iterator<Item = Triple>
    where
        Self: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_>;
}

impl<S: crate::BlockStorage> PatternBackend for crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    type Scan<'a>
        = crate::tree::Scan<'a, S>
    where
        S: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        self.scan(bounds)
    }
}
