//! Error types for the crate's fallible boundaries: out-of-domain
//! coordinates, corrupt/truncated serialized input, and I/O failures.
//! Everything else (arithmetic on values already known to be in-domain,
//! internal storage-format invariants guaranteed by the build path) stays
//! `debug_assert!`-checked, not wrapped in `Result` -- see the crate-level
//! docs for the boundary this draws.

use std::fmt;

/// A coordinate value outside the domain size established when the tree
/// was built (or explicitly declared via [`crate::Heap::build_dynamic_empty`]
/// and friends).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfDomain {
    /// Which dimension (0 = subject, 1 = predicate, 2 = object).
    pub dim: usize,
    pub value: u32,
    pub domain_size: u64,
}

impl fmt::Display for OutOfDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "coordinate {} for dimension {} is outside the domain [0, {})", self.value, self.dim, self.domain_size)
    }
}

/// Everything that can go wrong at a public API boundary.
#[derive(Debug)]
pub enum TreeError {
    /// `insert`/`insert_batch` called with a coordinate outside the domain
    /// established at build time. Check with [`crate::Heap::in_bounds`]
    /// (or the `File`/`Compressed` equivalent) up front to avoid
    /// constructing this for input you can filter yourself.
    OutOfDomain(OutOfDomain),
    /// A read/write against the backing file (`file`/`compressed` feature
    /// paths only) failed.
    Io(std::io::Error),
    /// Serialized bytes were truncated, had a bad magic number, or
    /// otherwise didn't parse as a tree of this format.
    CorruptData(String),
    /// The serialized data declares a format version this build doesn't
    /// know how to read.
    UnsupportedVersion(u8),
    /// A build-time configuration parameter (e.g. [`crate::Heap::build_with_order`]'s
    /// `split`) is invalid on its own terms -- not a matter of the data.
    InvalidConfig(String),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::OutOfDomain(e) => write!(f, "{e}"),
            TreeError::Io(e) => write!(f, "I/O error: {e}"),
            TreeError::CorruptData(msg) => write!(f, "corrupt serialized data: {msg}"),
            TreeError::UnsupportedVersion(v) => write!(f, "unsupported serialization format version: {v}"),
            TreeError::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for TreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TreeError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TreeError {
    fn from(e: std::io::Error) -> Self {
        TreeError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, TreeError>;
