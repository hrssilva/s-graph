//! Automaton access to the existing mutable Heap/File/Compressed radix trees.
use crate::automata_core::{BitmapSource, TripleSource};
use crate::storage::Pos;
use crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric as Tree;
use crate::{BlockStorage, Result};

/// Opaque position and level. No borrowed cache buffers survive a read.
#[derive(Clone, Copy)]
pub struct TreeCursor {
    pos: Pos,
    level: usize,
}
impl Default for TreeCursor {
    fn default() -> Self {
        Self {
            pos: Pos::START,
            level: 0,
        }
    }
}
pub struct TreeSource<'a, S: BlockStorage>(pub &'a Tree<S>);
impl<S: BlockStorage> Copy for TreeSource<'_, S> {}
impl<S: BlockStorage> Clone for TreeSource<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: BlockStorage> BitmapSource for TreeSource<'_, S> {
    fn begin_read(&self) -> impl Drop + '_ {
        self.0.automata_read_session()
    }

    type Cursor = TreeCursor;
    fn root(&self) -> Option<TreeCursor> {
        Some(TreeCursor::default())
    }
    fn height(&self) -> usize {
        (self.0.height() as usize).max(1)
    }
    fn word(&self, cursor: TreeCursor) -> u64 {
        self.0.automata_word(cursor.pos)
    }
    fn follow(&self, cursor: TreeCursor, word: u64, slot: u32) -> TreeCursor {
        TreeCursor {
            pos: self.0.automata_child(cursor.pos, cursor.level, word, slot),
            level: cursor.level + 1,
        }
    }
}
impl<S: BlockStorage> TripleSource for TreeSource<'_, S> {
    fn domains(&self) -> [u64; 3] {
        self.0.prefix_domains()
    }
    fn layout(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        self.0.automata_layout(level)
    }
}
impl<S: BlockStorage> Tree<S> {
    pub fn source(&self) -> TreeSource<'_, S> {
        TreeSource(self)
    }
    /// Compile against this immutable borrow; drop the search before updating
    /// the tree, then compile the same expression again against the new state.
    pub fn search<'a, A: crate::automata::Compile<TreeSource<'a, S>>>(
        &'a self,
        machine: A,
    ) -> Result<A::Compiled> {
        machine.compile(self.source())
    }
}
