//! The base k²-tree (Brisaboa, Ladra, Navarro): a binary `n x n` adjacency
//! matrix recursively split into `k x k` submatrices, with one existence bit
//! per submatrix, recursing into set bits down to individual cells.
//!
//! Storage is exactly two bitmaps (Structure of Arrays, not a node graph):
//! `t` holds every level's existence bits except the last, `l` holds the
//! last level. A set bit's children begin at `rank1(t, bit_pos + 1) * k*k`
//! -- an *inclusive* rank at `bit_pos` -- in the (virtual) concatenation of
//! `t` then `l`. The `+1` matters: level 0 always occupies a fixed `k*k`
//! slots in `t` regardless of how many of them are set, so a bit's own
//! set-ness must count towards its own children's starting offset, not just
//! the bits strictly before it. See [`K2Tree::contains`].
//!
//! This module is the shared skeleton reused, unmodified, by IK²-tree's
//! existence layer (`ik2tree.rs`): "does any predicate have a triple here"
//! is exactly a plain k²-tree over the union `(S,O)` matrix.

use crate::common::bitvector::BitVector;
use crate::common::metrics::MemSize;

pub struct K2Tree {
    t: BitVector,
    l: BitVector,
    n: u64,
    k: u32,
    height: u32,
}

struct Frame {
    base: u64, // bit offset of this node's k*k children slots within t (or l, if is_last)
    row0: u64,
    col0: u64,
    cell: u64, // size of THIS node's own box (children each have cell/k)
    next_dc: u32,
    level: u32, // depth of this node; is_last = (level == height-1)
}

/// Smallest `dc` (0-indexed child column-band) whose covered range
/// `[coord0+dc*child_cell, coord0+(dc+1)*child_cell)` could still contain a
/// value `>= target`. Used both for the root frame and for every freshly
/// pushed child frame so a `seek` target set mid-iteration is honored by
/// subtrees not yet visited.
fn initial_dc(coord0: u64, child_cell: u64, target: u64) -> u32 {
    if target <= coord0 {
        0
    } else {
        ((target - coord0) / child_cell) as u32
    }
}

impl K2Tree {
    /// Builds a k²-tree over the given (row, col) points. `points` is
    /// reordered in place (level-order counting-sort partition, no
    /// recursion). The matrix side `n` is the smallest `k^h` (`h >= 1`)
    /// covering every coordinate.
    pub fn build(k: u32, points: &mut [(u32, u32)]) -> Self {
        let max_coord = points
            .iter()
            .map(|&(r, c)| (r as u64).max(c as u64))
            .max()
            .unwrap_or(0);
        let mut n: u64 = k as u64;
        let mut height: u32 = 1;
        while n < max_coord + 1 {
            n *= k as u64;
            height += 1;
        }

        let mut t = BitVector::new();
        let mut l = BitVector::new();
        let kk = (k * k) as usize;

        struct FrontierEntry {
            start: usize,
            end: usize,
            row0: u64,
            col0: u64,
            cell: u64,
        }
        let mut frontier = vec![FrontierEntry {
            start: 0,
            end: points.len(),
            row0: 0,
            col0: 0,
            cell: n,
        }];

        for level in 0..height {
            let is_last = level == height - 1;
            let mut next_frontier = Vec::new();
            for entry in &frontier {
                let child_cell = entry.cell / k as u64;
                let slice = &mut points[entry.start..entry.end];

                let mut counts = vec![0usize; kk];
                for &(r, c) in slice.iter() {
                    let dr = ((r as u64 - entry.row0) / child_cell) as usize;
                    let dc = ((c as u64 - entry.col0) / child_cell) as usize;
                    counts[dr * k as usize + dc] += 1;
                }
                let mut offsets = vec![0usize; kk + 1];
                for i in 0..kk {
                    offsets[i + 1] = offsets[i] + counts[i];
                }
                let mut buf = vec![(0u32, 0u32); slice.len()];
                let mut cursor = offsets.clone();
                for &(r, c) in slice.iter() {
                    let dr = ((r as u64 - entry.row0) / child_cell) as usize;
                    let dc = ((c as u64 - entry.col0) / child_cell) as usize;
                    let qi = dr * k as usize + dc;
                    buf[cursor[qi]] = (r, c);
                    cursor[qi] += 1;
                }
                slice.copy_from_slice(&buf);

                for qi in 0..kk {
                    let qstart = entry.start + offsets[qi];
                    let qend = entry.start + offsets[qi + 1];
                    let non_empty = qstart < qend;
                    if is_last {
                        l.push(non_empty);
                    } else {
                        t.push(non_empty);
                    }
                    if non_empty && !is_last {
                        let dr = qi / k as usize;
                        let dc = qi % k as usize;
                        next_frontier.push(FrontierEntry {
                            start: qstart,
                            end: qend,
                            row0: entry.row0 + dr as u64 * child_cell,
                            col0: entry.col0 + dc as u64 * child_cell,
                            cell: child_cell,
                        });
                    }
                }
            }
            frontier = next_frontier;
        }

        t.finalize();
        l.finalize();
        K2Tree { t, l, n, k, height }
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    /// Appends this tree's serialized bytes to `out`: `n`/`k`/`height`,
    /// then `t`, then `l` (see `crate::common::k2_serialize`'s doc
    /// comment for the bitvector format). Used directly by `k2tree.rs`'s
    /// own consumers (`Mk2Forest`, one call per predicate tree) rather
    /// than as a `serialize()`/`deserialize()` pair of its own, since a
    /// bare `K2Tree` is never itself a top-level persisted structure in
    /// this project.
    pub(crate) fn write_into(&self, out: &mut Vec<u8>) {
        crate::common::tree_serialize::write_u64(out, self.n);
        crate::common::tree_serialize::write_u32(out, self.k);
        crate::common::tree_serialize::write_u32(out, self.height);
        crate::common::k2_serialize::write_bitvector(out, &self.t);
        crate::common::k2_serialize::write_bitvector(out, &self.l);
    }

    /// Reads one tree written by [`Self::write_into`].
    pub(crate) fn read_from<R: std::io::Read>(r: &mut R) -> Self {
        let n = crate::common::tree_serialize::read_u64_from(r);
        let k = crate::common::tree_serialize::read_u32_from(r);
        let height = crate::common::tree_serialize::read_u32_from(r);
        let t = crate::common::k2_serialize::read_bitvector_from(r);
        let l = crate::common::k2_serialize::read_bitvector_from(r);
        K2Tree { t, l, n, k, height }
    }

    /// Starting bit offset, within whichever of `t`/`l` the *next* level
    /// lives in, of the k*k children of the set bit at `bit_pos` (which
    /// itself lives in `t`, since only non-final-level bits have children).
    ///
    /// `t.rank1(bit_pos + 1) * k*k` (an *inclusive* rank at `bit_pos`) gives
    /// the child block's start in the *virtual* concatenation of `t` then
    /// `l` -- level 0 always occupies a fixed k*k slots in `t` regardless of
    /// how many are set, and inclusive rank happens to account for exactly
    /// that fixed prefix. But `t` and `l` are two separate arrays, not
    /// actually concatenated: when the children live in `l` (i.e. `bit_pos`
    /// is on the last `t`-level), that virtual offset must be translated
    /// into an `l`-relative index by subtracting `t.len()`.
    #[inline]
    fn child_offset(&self, bit_pos: u64, next_is_last: bool) -> u64 {
        let kk = self.k as u64 * self.k as u64;
        let virtual_base = self.t.rank1(bit_pos + 1) * kk;
        if next_is_last {
            virtual_base - self.t.len()
        } else {
            virtual_base
        }
    }

    pub fn contains(&self, s: u32, o: u32) -> bool {
        let (s, o) = (s as u64, o as u64);
        if s >= self.n || o >= self.n {
            return false;
        }
        let (mut row0, mut col0, mut cell, mut base) = (0u64, 0u64, self.n, 0u64);
        for level in 0..self.height {
            let is_last = level == self.height - 1;
            let child_cell = cell / self.k as u64;
            let dr = (s - row0) / child_cell;
            let dc = (o - col0) / child_cell;
            let bit_pos = base + dr * self.k as u64 + dc;
            let bv = if is_last { &self.l } else { &self.t };
            if !bv.get(bit_pos) {
                return false;
            }
            if is_last {
                return true;
            }
            base = self.child_offset(bit_pos, level + 1 == self.height - 1);
            row0 += dr * child_cell;
            col0 += dc * child_cell;
            cell = child_cell;
        }
        unreachable!("height is always >= 1")
    }

    /// Ascending iterator over objects `o` such that `(s, o)` is present.
    /// Explicit-stack `Iterator`, supporting `seek` -- needed by
    /// [`crate::common::sorted_neighbors`] for leapfrog joins. For a
    /// one-shot "just collect everything" call, prefer [`Self::row_vec`]:
    /// the seek-supporting stack machinery (computing `initial_dc`, a
    /// division, on every frame push) has fixed overhead that only pays for
    /// itself when `seek` is actually used.
    pub fn row_iter(&self, s: u32) -> RowIter<'_> {
        RowIter::new(self, s as u64)
    }

    /// Ascending iterator over subjects `s` such that `(s, o)` is present.
    /// See [`Self::row_iter`]'s doc comment -- prefer [`Self::col_vec`] for
    /// one-shot collection.
    pub fn col_iter(&self, o: u32) -> ColIter<'_> {
        ColIter::new(self, o as u64)
    }

    /// Ascending objects `o` such that `(s, o)` is present, collected via a
    /// plain recursive descent (native call stack, no `Iterator`
    /// state-machine/seek-support overhead) -- faster than
    /// `row_iter(s).collect()` when seeking isn't needed.
    pub fn row_vec(&self, s: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let s = s as u64;
        if s < self.n {
            self.row_vec_rec(0, 0, 0, 0, self.n, s, &mut out);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn row_vec_rec(&self, level: u32, base: u64, row0: u64, col0: u64, cell: u64, fixed_row: u64, out: &mut Vec<u32>) {
        let is_last = level == self.height - 1;
        let child_cell = cell / self.k as u64;
        let dr = (fixed_row - row0) / child_cell;
        let bv = if is_last { &self.l } else { &self.t };
        for dc in 0..self.k as u64 {
            let bit_pos = base + dr * self.k as u64 + dc;
            if !bv.get(bit_pos) {
                continue;
            }
            let child_col0 = col0 + dc * child_cell;
            if is_last {
                out.push(child_col0 as u32);
                continue;
            }
            let child_row0 = row0 + dr * child_cell;
            let child_base = self.child_offset(bit_pos, level + 1 == self.height - 1);
            self.row_vec_rec(level + 1, child_base, child_row0, child_col0, child_cell, fixed_row, out);
        }
    }

    /// Ascending subjects `s` such that `(s, o)` is present -- see
    /// [`Self::row_vec`].
    pub fn col_vec(&self, o: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let o = o as u64;
        if o < self.n {
            self.col_vec_rec(0, 0, 0, 0, self.n, o, &mut out);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn col_vec_rec(&self, level: u32, base: u64, row0: u64, col0: u64, cell: u64, fixed_col: u64, out: &mut Vec<u32>) {
        let is_last = level == self.height - 1;
        let child_cell = cell / self.k as u64;
        let dc = (fixed_col - col0) / child_cell;
        let bv = if is_last { &self.l } else { &self.t };
        for dr in 0..self.k as u64 {
            let bit_pos = base + dr * self.k as u64 + dc;
            if !bv.get(bit_pos) {
                continue;
            }
            let child_row0 = row0 + dr * child_cell;
            if is_last {
                out.push(child_row0 as u32);
                continue;
            }
            let child_col0 = col0 + dc * child_cell;
            let child_base = self.child_offset(bit_pos, level + 1 == self.height - 1);
            self.col_vec_rec(level + 1, child_base, child_row0, child_col0, child_cell, fixed_col, out);
        }
    }
}

impl MemSize for K2Tree {
    fn mem_size_bytes(&self) -> usize {
        self.t.mem_size_bytes() + self.l.mem_size_bytes()
    }
}

impl K2Tree {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![("t", self.t.mem_size_bytes()), ("l", self.l.mem_size_bytes())]
    }
}

pub struct RowIter<'a> {
    tree: &'a K2Tree,
    fixed_row: u64,
    stack: Vec<Frame>,
    seek_target: u64,
}

impl<'a> RowIter<'a> {
    fn new(tree: &'a K2Tree, fixed_row: u64) -> Self {
        let mut stack = Vec::with_capacity(tree.height as usize);
        if fixed_row < tree.n {
            stack.push(Frame {
                base: 0,
                row0: 0,
                col0: 0,
                cell: tree.n,
                next_dc: initial_dc(0, tree.n / tree.k as u64, 0),
                level: 0,
            });
        }
        RowIter {
            tree,
            fixed_row,
            stack,
            seek_target: 0,
        }
    }

    /// Advances the iterator so the next yielded value (if any) is `>=
    /// target`, without visiting whole subtrees that fall entirely below
    /// `target`.
    pub fn seek(&mut self, target: u32) {
        let target = target as u64;
        if target <= self.seek_target {
            return;
        }
        self.seek_target = target;
        while let Some(frame) = self.stack.last_mut() {
            if frame.col0 + frame.cell <= target {
                self.stack.pop();
                continue;
            }
            let child_cell = frame.cell / self.tree.k as u64;
            let min_dc = initial_dc(frame.col0, child_cell, target);
            if min_dc > frame.next_dc {
                frame.next_dc = min_dc;
            }
            break;
        }
    }
}

impl<'a> Iterator for RowIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let k = self.tree.k as u64;
        loop {
            let frame = self.stack.last_mut()?;
            if frame.next_dc as u64 >= k {
                self.stack.pop();
                continue;
            }
            let dc = frame.next_dc;
            frame.next_dc += 1;
            let child_cell = frame.cell / k;
            let dr = ((self.fixed_row - frame.row0) / child_cell) as u32;
            let bit_pos = frame.base + dr as u64 * k + dc as u64;
            let is_last = frame.level == self.tree.height - 1;
            let bv = if is_last { &self.tree.l } else { &self.tree.t };
            if !bv.get(bit_pos) {
                continue;
            }
            let child_row0 = frame.row0 + dr as u64 * child_cell;
            let child_col0 = frame.col0 + dc as u64 * child_cell;
            let child_level = frame.level + 1;
            if is_last {
                return Some(child_col0 as u32);
            }
            let child_base = self.tree.child_offset(bit_pos, child_level == self.tree.height - 1);
            let grandchild_cell = child_cell / k;
            self.stack.push(Frame {
                base: child_base,
                row0: child_row0,
                col0: child_col0,
                cell: child_cell,
                next_dc: initial_dc(child_col0, grandchild_cell, self.seek_target),
                level: child_level,
            });
        }
    }
}

pub struct ColIter<'a> {
    tree: &'a K2Tree,
    fixed_col: u64,
    stack: Vec<ColFrame>,
    seek_target: u64,
}

struct ColFrame {
    base: u64,
    row0: u64,
    col0: u64,
    cell: u64,
    next_dr: u32,
    level: u32,
}

impl<'a> ColIter<'a> {
    fn new(tree: &'a K2Tree, fixed_col: u64) -> Self {
        let mut stack = Vec::with_capacity(tree.height as usize);
        if fixed_col < tree.n {
            stack.push(ColFrame {
                base: 0,
                row0: 0,
                col0: 0,
                cell: tree.n,
                next_dr: initial_dc(0, tree.n / tree.k as u64, 0),
                level: 0,
            });
        }
        ColIter {
            tree,
            fixed_col,
            stack,
            seek_target: 0,
        }
    }

    pub fn seek(&mut self, target: u32) {
        let target = target as u64;
        if target <= self.seek_target {
            return;
        }
        self.seek_target = target;
        while let Some(frame) = self.stack.last_mut() {
            if frame.row0 + frame.cell <= target {
                self.stack.pop();
                continue;
            }
            let child_cell = frame.cell / self.tree.k as u64;
            let min_dr = initial_dc(frame.row0, child_cell, target);
            if min_dr > frame.next_dr {
                frame.next_dr = min_dr;
            }
            break;
        }
    }
}

impl<'a> Iterator for ColIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let k = self.tree.k as u64;
        loop {
            let frame = self.stack.last_mut()?;
            if frame.next_dr as u64 >= k {
                self.stack.pop();
                continue;
            }
            let dr = frame.next_dr;
            frame.next_dr += 1;
            let child_cell = frame.cell / k;
            let dc = ((self.fixed_col - frame.col0) / child_cell) as u32;
            let bit_pos = frame.base + dr as u64 * k + dc as u64;
            let is_last = frame.level == self.tree.height - 1;
            let bv = if is_last { &self.tree.l } else { &self.tree.t };
            if !bv.get(bit_pos) {
                continue;
            }
            let child_row0 = frame.row0 + dr as u64 * child_cell;
            let child_col0 = frame.col0 + dc as u64 * child_cell;
            let child_level = frame.level + 1;
            if is_last {
                return Some(child_row0 as u32);
            }
            let child_base = self.tree.child_offset(bit_pos, child_level == self.tree.height - 1);
            let grandchild_cell = child_cell / k;
            self.stack.push(ColFrame {
                base: child_base,
                row0: child_row0,
                col0: child_col0,
                cell: child_cell,
                next_dr: initial_dc(child_row0, grandchild_cell, self.seek_target),
                level: child_level,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct Naive {
        set: HashSet<(u32, u32)>,
    }
    impl Naive {
        fn row(&self, s: u32) -> Vec<u32> {
            let mut v: Vec<u32> = self.set.iter().filter(|&&(r, _)| r == s).map(|&(_, c)| c).collect();
            v.sort_unstable();
            v
        }
        fn col(&self, o: u32) -> Vec<u32> {
            let mut v: Vec<u32> = self.set.iter().filter(|&&(_, c)| c == o).map(|&(r, _)| r).collect();
            v.sort_unstable();
            v
        }
    }

    fn rand_points(n_points: usize, coord_max: u32, seed: u64) -> Vec<(u32, u32)> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n_points)
            .map(|_| ((next() % (coord_max as u64 + 1)) as u32, (next() % (coord_max as u64 + 1)) as u32))
            .collect()
    }

    fn check(k: u32, coord_max: u32, n_points: usize, seed: u64) {
        let mut points = rand_points(n_points, coord_max, seed);
        let naive = Naive {
            set: points.iter().copied().collect(),
        };
        let tree = K2Tree::build(k, &mut points);

        for r in 0..=coord_max {
            for c in 0..=coord_max {
                let expect = naive.set.contains(&(r, c));
                assert_eq!(tree.contains(r, c), expect, "contains({r},{c}) k={k} seed={seed}");
            }
        }
        for s in 0..=coord_max {
            let got: Vec<u32> = tree.row_iter(s).collect();
            assert_eq!(got, naive.row(s), "row_iter({s}) k={k} seed={seed}");
            assert_eq!(tree.row_vec(s), naive.row(s), "row_vec({s}) k={k} seed={seed}");
        }
        for o in 0..=coord_max {
            let got: Vec<u32> = tree.col_iter(o).collect();
            assert_eq!(got, naive.col(o), "col_iter({o}) k={k} seed={seed}");
            assert_eq!(tree.col_vec(o), naive.col(o), "col_vec({o}) k={k} seed={seed}");
        }
    }

    #[test]
    fn cross_check_against_naive() {
        for &k in &[2u32, 4, 8] {
            for &coord_max in &[7u32, 15, 63] {
                for seed in 1..6u64 {
                    check(k, coord_max, (coord_max as usize + 1) * 3, seed * 7919);
                }
            }
        }
    }

    #[test]
    fn empty_tree() {
        let mut points: Vec<(u32, u32)> = vec![];
        let tree = K2Tree::build(2, &mut points);
        assert!(!tree.contains(0, 0));
        assert_eq!(tree.row_iter(0).count(), 0);
    }

    #[test]
    fn seek_matches_linear_scan() {
        let mut points = rand_points(400, 63, 12345);
        let tree = K2Tree::build(2, &mut points);
        for s in [0u32, 5, 31, 63] {
            let all: Vec<u32> = tree.row_iter(s).collect();
            for &target in &[0u32, 1, 10, 32, 50, 63, 64] {
                let expected: Vec<u32> = all.iter().copied().filter(|&v| v >= target).collect();
                let mut it = tree.row_iter(s);
                it.seek(target);
                let got: Vec<u32> = it.collect();
                assert_eq!(got, expected, "seek({target}) on row {s} mismatch");
            }
        }
    }
}
