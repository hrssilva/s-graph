//! Prepared, general conjunctive joins with packed lexicographic relations.
//!
//! Preparation scans each atom once, applies constants and repeated-variable
//! equalities, sorts and deduplicates its projected relation. This explicit
//! O(sum N_i log N_i) cost creates an immutable snapshot, not a live view.
//! Execution is leapfrog triejoin over those relations (worst-case optimal up
//! to logarithmic factors for full conjunctive queries). Morton traversal
//! itself is not claimed to provide the ordered-seek complexity of a trie.
//!
//! No permanent index permutations are added to the graph. Relation columns are
//! bit-packed relative to their minimum value by default; plain u32 rows are
//! optional. Workspaces are O(query size), reuse allocations, and results are
//! borrowed through a cancellable callback.

use crate::graph::{JoinQuery, Term};
use crate::{Result, TreeError, Triple};
use std::ops::ControlFlow;

/// Monomorphized, allocation-free triple scans with optional coordinate bounds.
pub trait PatternBackend {
    type Scan<'a>: Iterator<Item = Triple>
    where
        Self: 'a;
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_>;
}

#[cfg(test)]
mod packing_tests {
    use super::*;

    #[test]
    fn packed_rows_preserve_full_width_keys_across_word_boundaries() {
        let rows: Vec<_> = (0..1000u32).map(|i| [i & 1, 0xffff0000 + (i * 73 % 65536), i.wrapping_mul(0x9e3779b9)]).collect();
        for arity in 1..=3 {
            for layout in [RelationLayout::Packed, RelationLayout::Plain] {
                let relation = Relation::new(rows.clone(), arity, layout);
                assert_eq!(relation.len(), rows.len());
                for (i, row) in rows.iter().enumerate() {
                    for (column, &value) in row.iter().enumerate().take(arity) {
                        assert_eq!(relation.key(i, column), value);
                    }
                }
                let singleton = Relation::new(vec![[u32::MAX; 3]], arity, layout);
                for column in 0..arity {
                    assert_eq!(singleton.key(0, column), u32::MAX);
                }
            }
        }
    }

    #[test]
    fn galloping_seek_matches_linear_projection_search() {
        let rows: Vec<_> = (0..150u32).map(|i| [i / 3, i, 0]).collect();
        for layout in [RelationLayout::Packed, RelationLayout::Plain] {
            let relation = Relation::new(rows.clone(), 2, layout);
            for start in 0..=rows.len() {
                for target in [0, 1, 7, 49, 50, u32::MAX] {
                    for inclusive in [false, true] {
                        let expected = start + rows[start..].iter().take_while(|r| r[0] < target || (inclusive && r[0] == target)).count();
                        assert_eq!(relation.partition(start, rows.len(), 0, target, inclusive), expected);
                    }
                }
            }
        }
    }
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

/// Preparation controls. The row budget includes all matching atom rows before
/// deduplication; it bounds temporary row payloads to at most 12 bytes per row
/// (Vec capacity growth and query metadata are additional).
#[derive(Clone, Debug)]
pub struct JoinOptions {
    pub variable_order: Option<Vec<usize>>,
    pub max_rows: usize,
    pub layout: RelationLayout,
}

/// Packed minimizes relation payload; Plain avoids bit extraction on each seek.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RelationLayout {
    #[default]
    Packed,
    Plain,
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self { variable_order: None, max_rows: usize::MAX, layout: RelationLayout::Packed }
    }
}

struct Relation {
    rows: RelationRows,
    arity: usize,
    len: usize,
}

enum RelationRows {
    Plain(Box<[u32]>),
    Packed { words: Box<[u64]>, bases: [u32; 3], widths: [u8; 3], offsets: [u8; 3], stride: u8 },
}

impl Relation {
    fn new(rows: Vec<[u32; 3]>, arity: usize, layout: RelationLayout) -> Self {
        let len = rows.len();
        let rows = if layout == RelationLayout::Plain {
            let mut packed = rows.into_flattened();
            if arity < 3 {
                for row in 0..len {
                    for column in 0..arity {
                        packed[row * arity + column] = packed[row * 3 + column];
                    }
                }
                packed.truncate(len * arity);
            }
            RelationRows::Plain(packed.into_boxed_slice())
        } else {
            let mut bases = [u32::MAX; 3];
            let mut maxima = [0; 3];
            for row in &rows {
                for c in 0..arity {
                    bases[c] = bases[c].min(row[c]);
                    maxima[c] = maxima[c].max(row[c]);
                }
            }
            let mut widths = [0; 3];
            let mut offsets = [0; 3];
            let mut stride = 0u8;
            for c in 0..arity {
                offsets[c] = stride;
                widths[c] = (32 - (maxima[c] - bases[c]).leading_zeros()) as u8;
                stride += widths[c];
            }
            // Compute in words to avoid overflowing usize in len * stride.
            let word_count = (len / 64) * stride as usize + ((len % 64) * stride as usize).div_ceil(64);
            let mut words = vec![0u64; word_count];
            for (i, row) in rows.iter().enumerate() {
                for c in 0..arity {
                    if widths[c] == 0 {
                        continue;
                    }
                    let base_word = (i / 64) * stride as usize;
                    let bit = (i % 64) * stride as usize + offsets[c] as usize;
                    let word = base_word + bit / 64;
                    let shift = bit % 64;
                    let value = (row[c] - bases[c]) as u64;
                    words[word] |= value << shift;
                    if shift + widths[c] as usize > 64 {
                        words[word + 1] |= value >> (64 - shift);
                    }
                }
            }
            RelationRows::Packed { words: words.into_boxed_slice(), bases, widths, offsets, stride }
        };
        Self { rows, arity, len }
    }

    fn len(&self) -> usize {
        self.len
    }
    #[inline]
    fn key(&self, row: usize, column: usize) -> u32 {
        match &self.rows {
            RelationRows::Plain(rows) => rows[row * self.arity + column],
            RelationRows::Packed { words, bases, widths, offsets, stride } => {
                let width = widths[column] as usize;
                if width == 0 {
                    return bases[column];
                }
                let bit = (row % 64) * *stride as usize + offsets[column] as usize;
                let index = (row / 64) * *stride as usize + bit / 64;
                let shift = bit % 64;
                let mut value = words[index] >> shift;
                if shift + width > 64 {
                    value |= words[index + 1] << (64 - shift);
                }
                bases[column] + (value & ((1u64 << width) - 1)) as u32
            }
        }
    }
    // Binary search rows without creating row slices or allocating cursors.
    fn partition(&self, mut lo: usize, mut hi: usize, col: usize, value: u32, inclusive: bool) -> usize {
        let before = |row| {
            let key = self.key(row, col);
            key < value || (inclusive && key == value)
        };
        if lo == hi || !before(lo) {
            return lo;
        }
        let start = lo;
        lo += 1;
        let mut step = 1;
        // Gallop from the current cursor: nearby seeks avoid log(whole relation)
        // probes, while long seeks retain logarithmic worst-case cost.
        while step < hi - start {
            let probe = start + step;
            if !before(probe) {
                hi = probe;
                break;
            }
            lo = probe + 1;
            step = step.saturating_mul(2);
        }
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if before(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

#[derive(Clone, Copy)]
struct Occurrence {
    relation: usize,
    column: usize,
    parent: Option<usize>,
}

/// Immutable prepared snapshot. Rebuild after graph changes when fresh results
/// are required. Share across threads, each with its own `JoinWorkspace`.
pub struct PreparedJoin {
    relations: Vec<Relation>,
    order: Vec<usize>,
    offsets: Vec<usize>,
    occurrences: Vec<Occurrence>,
    empty: bool,
}

#[derive(Clone, Copy, Default)]
struct Cursor {
    pos: usize,
    end: usize,
}

/// Reusable execution memory; no allocation after warming up for a query size.
#[derive(Default)]
pub struct JoinWorkspace {
    cursors: Vec<Cursor>,
    bindings: Vec<u32>,
}

impl PreparedJoin {
    pub fn new<B: PatternBackend, P: Copy + Into<Term>>(query: &JoinQuery<P>, graph: &B) -> Result<Self> {
        Self::with_options(query, graph, JoinOptions::default())
    }

    pub fn with_options<B: PatternBackend, P: Copy + Into<Term>>(query: &JoinQuery<P>, graph: &B, options: JoinOptions) -> Result<Self> {
        let invalid = |s: &str| TreeError::InvalidConfig(s.into());
        // Check the syntactic upper bound before allocating by user-supplied num_vars.
        if query.num_vars > query.patterns.len().saturating_mul(3) {
            return Err(invalid("join has unused variables"));
        }
        let mut used = vec![false; query.num_vars];
        let mut terms: Vec<[Term; 3]> = query.patterns.iter().map(|p| [p.subject, p.predicate.into(), p.object]).collect();
        for atom in &terms {
            for &term in atom {
                if let Term::Var(v) = term {
                    if v >= query.num_vars {
                        return Err(invalid("variable index exceeds num_vars"));
                    }
                    used[v] = true;
                }
            }
        }
        if used.contains(&false) {
            return Err(invalid("join has unused variables"));
        }
        let order = options.variable_order.unwrap_or_else(|| {
            // Prefer variables in constant-filtered atoms, then highly shared variables.
            let mut score = vec![0usize; query.num_vars];
            for atom in &terms {
                let constants = atom.iter().filter(|t| matches!(t, Term::Const(_))).count();
                for &term in atom {
                    if let Term::Var(v) = term {
                        score[v] += 1 + 3 * constants;
                    }
                }
            }
            let mut order: Vec<_> = (0..query.num_vars).collect();
            order.sort_unstable_by_key(|&v| (std::cmp::Reverse(score[v]), v));
            order
        });
        let mut sorted = order.clone();
        sorted.sort_unstable();
        if sorted.len() != query.num_vars || sorted.iter().copied().ne(0..query.num_vars) {
            return Err(invalid("variable_order must be a permutation of all variables"));
        }
        let mut rank = vec![0; query.num_vars];
        for (i, &v) in order.iter().enumerate() {
            rank[v] = i;
        }
        let mut relations = Vec::new();
        let mut atom_vars = Vec::new();
        let mut consumed = 0usize;
        let mut singletons = vec![None; query.num_vars];
        // Prepare selective constant-bound atoms first. A singleton projection
        // constrains every solution, so push it into later tree scans immediately.
        terms.sort_by_key(|atom| std::cmp::Reverse(atom.iter().filter(|t| matches!(t, Term::Const(_))).count()));
        for atom in terms {
            let bounds = atom.map(|t| match t {
                Term::Const(c) => Some(c),
                Term::Var(v) => singletons[v],
            });
            let mut vars: Vec<usize> = atom.iter().filter_map(|t| if let Term::Var(v) = t { Some(*v) } else { None }).collect();
            vars.sort_unstable_by_key(|&v| rank[v]);
            vars.dedup();
            if vars.is_empty() {
                if graph.scan_pattern(bounds).next().is_none() {
                    return Ok(Self { relations: Vec::new(), order, offsets: Vec::new(), occurrences: Vec::new(), empty: true });
                }
                continue;
            }
            let mut columns = [0usize; 3];
            for (i, &v) in vars.iter().enumerate() {
                columns[i] = atom.iter().position(|t| *t == Term::Var(v)).unwrap();
            }
            let mut rows = Vec::new();
            for (s, p, o) in graph.scan_pattern(bounds) {
                let values = [s, p, o];
                if (0..3).any(|i| (i + 1..3).any(|j| matches!(atom[i], Term::Var(_)) && atom[i] == atom[j] && values[i] != values[j])) {
                    continue;
                }
                if consumed == options.max_rows {
                    return Err(invalid("join preparation row budget exceeded"));
                }
                consumed += 1;
                let mut row = [0; 3];
                for i in 0..vars.len() {
                    row[i] = values[columns[i]];
                }
                rows.push(row);
            }
            rows.sort_unstable();
            rows.dedup();
            if rows.is_empty() {
                return Ok(Self { relations: Vec::new(), order, offsets: Vec::new(), occurrences: Vec::new(), empty: true });
            }
            let arity = vars.len();
            if let Some(first) = rows.first() {
                for column in 0..arity {
                    if rows.iter().all(|row| row[column] == first[column]) {
                        singletons[vars[column]] = Some(first[column]);
                    }
                }
            }
            relations.push(Relation::new(rows, arity, options.layout));
            atom_vars.push(vars);
        }
        let mut offsets = Vec::with_capacity(order.len() + 1);
        let mut occurrences = Vec::new();
        let mut previous = vec![None; relations.len()];
        for &v in &order {
            offsets.push(occurrences.len());
            for (r, vars) in atom_vars.iter().enumerate() {
                if let Some(column) = vars.iter().position(|&x| x == v) {
                    occurrences.push(Occurrence { relation: r, column, parent: previous[r] });
                    previous[r] = Some(occurrences.len() - 1);
                }
            }
        }
        offsets.push(occurrences.len());
        Ok(Self { relations, order, offsets, occurrences, empty: false })
    }

    pub fn variable_order(&self) -> &[usize] {
        &self.order
    }

    /// Ordered relation payload only, excluding small plan/workspace metadata.
    pub fn relation_bytes(&self) -> usize {
        self.relations
            .iter()
            .map(|r| match &r.rows {
                RelationRows::Plain(rows) => rows.len() * 4,
                RelationRows::Packed { words, .. } => words.len() * 8,
            })
            .sum()
    }

    fn open(&self, depth: usize, ws: &mut JoinWorkspace) {
        for i in self.offsets[depth]..self.offsets[depth + 1] {
            let occurrence = self.occurrences[i];
            let r = &self.relations[occurrence.relation];
            ws.cursors[i] = if let Some(parent) = occurrence.parent {
                let c = ws.cursors[parent];
                let col = occurrence.column - 1;
                Cursor { pos: c.pos, end: r.partition(c.pos, c.end, col, r.key(c.pos, col), true) }
            } else {
                Cursor { pos: 0, end: r.len() }
            };
        }
    }

    fn align(&self, depth: usize, ws: &mut JoinWorkspace) -> Option<u32> {
        let range = self.offsets[depth]..self.offsets[depth + 1];
        let mut target = 0;
        loop {
            let mut equal = true;
            for i in range.clone() {
                let o = self.occurrences[i];
                let r = &self.relations[o.relation];
                let c = &mut ws.cursors[i];
                c.pos = r.partition(c.pos, c.end, o.column, target, false);
                if c.pos == c.end {
                    return None;
                }
                let value = r.key(c.pos, o.column);
                if value > target {
                    target = value;
                    equal = false;
                }
            }
            if equal {
                return Some(target);
            }
        }
    }

    fn advance(&self, depth: usize, ws: &mut JoinWorkspace) {
        // Advancing one stream suffices; align seeks the rest to catch up.
        let i = self.offsets[depth];
        let o = self.occurrences[i];
        let r = &self.relations[o.relation];
        let c = &mut ws.cursors[i];
        c.pos = r.partition(c.pos, c.end, o.column, r.key(c.pos, o.column), true);
    }

    /// Enumerate full bindings indexed by variable ID. Break stops immediately.
    /// Iterative backtracking avoids a call-stack frame per query variable.
    pub fn visit<T>(&self, ws: &mut JoinWorkspace, mut emit: impl FnMut(&[u32]) -> ControlFlow<T>) -> ControlFlow<T> {
        if self.empty {
            return ControlFlow::Continue(());
        }
        ws.bindings.resize(self.order.len(), 0);
        ws.cursors.resize(self.occurrences.len(), Cursor::default());
        if self.order.is_empty() {
            return emit(&[]);
        }
        let mut depth = 0;
        self.open(depth, ws);
        loop {
            if let Some(value) = self.align(depth, ws) {
                ws.bindings[self.order[depth]] = value;
                if depth + 1 == self.order.len() {
                    emit(&ws.bindings)?;
                    self.advance(depth, ws);
                } else {
                    depth += 1;
                    self.open(depth, ws);
                }
            } else if depth == 0 {
                return ControlFlow::Continue(());
            } else {
                depth -= 1;
                self.advance(depth, ws);
            }
        }
    }
}
