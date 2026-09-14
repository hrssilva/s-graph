//! IK²-tree (Interleaved k²-tree): a *single* k²-tree over the union
//! `(subject, object)` matrix, where each node's children carry a
//! variable-width bitmap (one bit per still-"active" predicate inherited
//! from the parent) instead of a single existence bit.
//!
//! Source: García, Brisaboa, de Bernardo, Navarro, "Interleaved K²-tree:
//! Indexing and Navigating Ternary Relations" (DCC 2014), Section 3.1
//! (verified directly against the paper's own worked numeric example, p.5:
//! `rank1(T,5)=3`, child region at `3*4+12=24`, matching their stated
//! `adjust = |Y|*K²` for the root's own transition).
//!
//! Model: node `v` has an inherited active-predicate list `A_v` (root:
//! identity `[0..num_predicates)`). Each of `v`'s k² children carries a
//! bitmap of width `m_v = |A_v|`: bit `j` says whether predicate `A_v[j]`
//! has >=1 triple in that child's submatrix. A child `c` exists iff its
//! bitmap is non-zero; `A_c` is the subsequence of `A_v` at `c`'s 1-bit
//! positions.
//!
//! **Physical layout, matching the paper (no separate existence skeleton,
//! no per-node offset table)**: exactly two bitvectors, `t` and `l`
//! (T/L-split by level, same convention as base k²-tree), holding *only*
//! the variable-width presence groups -- existence of a node is just
//! "popcount of its own w-bit slice is nonzero", the identical computation
//! already needed for its width, so no separate skeleton is stored.
//!
//! Locating an arbitrary node's own k²·w-bit block still needs *some*
//! auxiliary structure, since node widths vary and a plain `rank1(x)*k²`
//! (which works for base k²-tree only because every node has the same
//! fixed k² children) doesn't by itself account for that variation. The
//! resolution -- confirmed by re-deriving and checking it against a
//! concrete non-uniform-width worked example -- is that this only needs
//! **O(height)** bookkeeping, not O(existing nodes):
//!
//! `region_start_t[d]` is the bit offset in `t` where all depth-`d` nodes'
//! own w-bit identifying slices begin (written, as one contiguous region,
//! during their parents' construction pass). Given a node's own block start
//! `x` (relative to `region_start_t[depth]`) and width `w`, its child at
//! quadrant `qi` has identifying slice `slice_start = region_start_t[depth]
//! + x + qi*w`; that child's own subtree data is:
//! - width: `rank1(slice_start+w) - rank1(slice_start)` (its own popcount)
//! - block start (relative to `region_start_t[depth+1]`):
//!   `(rank1(slice_start) - rank1(region_start_t[depth])) * k²`
//!
//! Both are plain rank1 differences over `t` -- no stored per-node data.
//! The `* k²` term is the crux: each unit of *popcount* found so far
//! within the current depth's region corresponds to exactly `k²` bits of
//! space reserved for that one node's own children in the *next* depth's
//! region, regardless of what any other node's own width happens to be --
//! which is what lets a running popcount (i.e. `rank1`) stand in for a sum
//! of variable-width contributions.
//!
//! Two query modes for an unbound predicate (`(s,?,o)`, `(s,?,?)`, ...):
//! - *Eager*: materialize each visited node's active-predicate list
//!   explicitly, O(m) per visited node (including dead ends).
//! - *Lazy* (the structure's headline trick): descend without ever
//!   materializing predicate lists, only recording the path taken; on a
//!   leaf hit, walk the path bottom-up via `select1` to recover the global
//!   predicate id -- O(depth) per **hit**, not per visited node.

use crate::common::bitvector::BitVector;
use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

pub struct Ik2Tree {
    t: BitVector,
    l: BitVector,
    /// `region_start_t[d]`: bit offset in `t` where depth-`d` nodes' own
    /// identifying slices begin, for `d` in `1..height`. Index 0 is
    /// unused (the root has no identifying slice -- it's the implicit,
    /// always-present origin). O(height) total, not O(existing nodes).
    region_start_t: Vec<u64>,
    n: u64,
    k: u32,
    height: u32,
    num_predicates: u32,
}

struct FrontierEntry {
    start: usize,
    end: usize,
    row0: u64,
    col0: u64,
    cell: u64,
    active: Vec<u32>, // dense, ascending global predicate ids active at this node
}

impl Ik2Tree {
    pub fn build(k: u32, triples: &[Triple]) -> Self {
        let num_predicates = triples.iter().map(|&(_, p, _)| p).max().map(|m| m + 1).unwrap_or(0);
        let max_coord = triples.iter().map(|&(s, _, o)| (s as u64).max(o as u64)).max().unwrap_or(0);
        let mut n: u64 = k as u64;
        let mut height: u32 = 1;
        while n < max_coord + 1 {
            n *= k as u64;
            height += 1;
        }
        let kk = (k * k) as usize;

        // points: (row, col, local_pred_pos relative to the CURRENT node's
        // active list). At the root, active = identity, so local_pred_pos
        // starts out equal to the triple's actual global predicate id.
        let mut points: Vec<(u32, u32, u32)> = triples.iter().map(|&(s, p, o)| (s, o, p)).collect();

        let mut t = BitVector::new();
        let mut l = BitVector::new();
        let mut region_start_t = vec![0u64; height as usize + 1];

        let root_active: Vec<u32> = (0..num_predicates).collect();
        let mut frontier = vec![FrontierEntry {
            start: 0,
            end: points.len(),
            row0: 0,
            col0: 0,
            cell: n,
            active: root_active,
        }];

        for level in 0..height {
            let is_last = level == height - 1;
            // depth (level+1)'s own identifying-slice region begins right
            // here, before level's entries write their children's data.
            region_start_t[(level + 1) as usize] = t.len();

            let mut next_frontier = Vec::new();
            for entry in &frontier {
                let m = entry.active.len();
                let child_cell = entry.cell / k as u64;
                let slice = &mut points[entry.start..entry.end];

                let mut counts = vec![0usize; kk];
                for &(r, c, _) in slice.iter() {
                    let dr = ((r as u64 - entry.row0) / child_cell) as usize;
                    let dc = ((c as u64 - entry.col0) / child_cell) as usize;
                    counts[dr * k as usize + dc] += 1;
                }
                let mut offsets = vec![0usize; kk + 1];
                for i in 0..kk {
                    offsets[i + 1] = offsets[i] + counts[i];
                }
                let mut buf = vec![(0u32, 0u32, 0u32); slice.len()];
                let mut cursor = offsets.clone();
                for &(r, c, lp) in slice.iter() {
                    let dr = ((r as u64 - entry.row0) / child_cell) as usize;
                    let dc = ((c as u64 - entry.col0) / child_cell) as usize;
                    let qi = dr * k as usize + dc;
                    buf[cursor[qi]] = (r, c, lp);
                    cursor[qi] += 1;
                }
                slice.copy_from_slice(&buf);

                // This node's own block: for each of its k*k quadrants, an
                // m-bit "present" bitmap over the node's active list.
                let bv = if is_last { &mut l } else { &mut t };
                for qi in 0..kk {
                    let qstart = entry.start + offsets[qi];
                    let qend = entry.start + offsets[qi + 1];
                    let non_empty = qstart < qend;

                    let mut present = vec![false; m];
                    for &(_, _, lp) in points[qstart..qend].iter() {
                        present[lp as usize] = true;
                    }
                    for &p in present.iter() {
                        bv.push(p);
                    }

                    if non_empty && !is_last {
                        let dr = qi / k as usize;
                        let dc = qi % k as usize;
                        // Child's active list + remap old->new local pos.
                        let mut remap = vec![0u32; m];
                        let mut child_active = Vec::new();
                        for (old_pos, &present_bit) in present.iter().enumerate() {
                            if present_bit {
                                remap[old_pos] = child_active.len() as u32;
                                child_active.push(entry.active[old_pos]);
                            }
                        }
                        for pt in points[qstart..qend].iter_mut() {
                            pt.2 = remap[pt.2 as usize];
                        }
                        next_frontier.push(FrontierEntry {
                            start: qstart,
                            end: qend,
                            row0: entry.row0 + dr as u64 * child_cell,
                            col0: entry.col0 + dc as u64 * child_cell,
                            cell: child_cell,
                            active: child_active,
                        });
                    }
                }
            }
            frontier = next_frontier;
        }

        t.finalize();
        l.finalize();

        Ik2Tree {
            t,
            l,
            region_start_t,
            n,
            k,
            height,
            num_predicates,
        }
    }

    pub fn n(&self) -> u64 {
        self.n
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn num_predicates(&self) -> u32 {
        self.num_predicates
    }

    #[inline]
    fn kk(&self) -> u64 {
        self.k as u64 * self.k as u64
    }

    /// `(s,p,o)` membership.
    pub fn locate(&self, p: u32, s: u32, o: u32) -> bool {
        if p >= self.num_predicates || (s as u64) >= self.n || (o as u64) >= self.n {
            return false;
        }
        let (s, o) = (s as u64, o as u64);
        let (mut row0, mut col0, mut cell) = (0u64, 0u64, self.n);
        let (mut x, mut w, mut local_pred_pos) = (0u64, self.num_predicates as u64, p as u64);
        for depth in 0..self.height {
            let is_last = depth == self.height - 1;
            let region_bv = if is_last { &self.l } else { &self.t };
            let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
            let block_start = region_base + x;

            let child_cell = cell / self.k as u64;
            let dr = (s - row0) / child_cell;
            let dc = (o - col0) / child_cell;
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;
            let bit_pos = slice_start + local_pred_pos;
            if !region_bv.get(bit_pos) {
                return false;
            }
            if is_last {
                return true;
            }
            local_pred_pos = region_bv.rank1(bit_pos) - region_bv.rank1(slice_start);
            w = region_bv.rank1(slice_start + w) - region_bv.rank1(slice_start);
            x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            row0 += dr * child_cell;
            col0 += dc * child_cell;
            cell = child_cell;
        }
        unreachable!("height is always >= 1")
    }

    /// Ascending objects `o` such that `(s,p,o)` is present.
    pub fn row_bound(&self, p: u32, s: u32) -> Vec<u32> {
        let mut out = Vec::new();
        if p >= self.num_predicates || (s as u64) >= self.n {
            return out;
        }
        self.row_bound_rec(0, 0, self.num_predicates as u64, p as u64, 0, 0, self.n, s as u64, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn row_bound_rec(&self, depth: u32, x: u64, w: u64, local_pred_pos: u64, row0: u64, col0: u64, cell: u64, fixed_row: u64, out: &mut Vec<u32>) {
        let is_last = depth == self.height - 1;
        let region_bv = if is_last { &self.l } else { &self.t };
        let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
        let block_start = region_base + x;
        let child_cell = cell / self.k as u64;
        let dr = (fixed_row - row0) / child_cell;

        for dc in 0..self.k as u64 {
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;
            let bit_pos = slice_start + local_pred_pos;
            if !region_bv.get(bit_pos) {
                continue;
            }
            let child_col0 = col0 + dc * child_cell;
            if is_last {
                out.push(child_col0 as u32);
                continue;
            }
            let child_local_pred_pos = region_bv.rank1(bit_pos) - region_bv.rank1(slice_start);
            let child_w = region_bv.rank1(slice_start + w) - region_bv.rank1(slice_start);
            let child_x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            let child_row0 = row0 + dr * child_cell;
            self.row_bound_rec(depth + 1, child_x, child_w, child_local_pred_pos, child_row0, child_col0, child_cell, fixed_row, out);
        }
    }

    /// Ascending subjects `s` such that `(s,p,o)` is present. Mirrors
    /// [`Self::row_bound`] with rows/columns swapped.
    pub fn col_bound(&self, p: u32, o: u32) -> Vec<u32> {
        let mut out = Vec::new();
        if p >= self.num_predicates || (o as u64) >= self.n {
            return out;
        }
        self.col_bound_rec(0, 0, self.num_predicates as u64, p as u64, 0, 0, self.n, o as u64, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn col_bound_rec(&self, depth: u32, x: u64, w: u64, local_pred_pos: u64, row0: u64, col0: u64, cell: u64, fixed_col: u64, out: &mut Vec<u32>) {
        let is_last = depth == self.height - 1;
        let region_bv = if is_last { &self.l } else { &self.t };
        let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
        let block_start = region_base + x;
        let child_cell = cell / self.k as u64;
        let dc = (fixed_col - col0) / child_cell;

        for dr in 0..self.k as u64 {
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;
            let bit_pos = slice_start + local_pred_pos;
            if !region_bv.get(bit_pos) {
                continue;
            }
            let child_row0 = row0 + dr * child_cell;
            if is_last {
                out.push(child_row0 as u32);
                continue;
            }
            let child_local_pred_pos = region_bv.rank1(bit_pos) - region_bv.rank1(slice_start);
            let child_w = region_bv.rank1(slice_start + w) - region_bv.rank1(slice_start);
            let child_x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            let child_col0 = col0 + dc * child_cell;
            self.col_bound_rec(depth + 1, child_x, child_w, child_local_pred_pos, child_row0, child_col0, child_cell, fixed_col, out);
        }
    }

    /// Eager unbound-predicate row query: `(predicate, object)` pairs for a
    /// fixed subject, materializing each visited node's active-predicate
    /// list explicitly.
    pub fn row_unbound_eager(&self, s: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        if (s as u64) >= self.n {
            return out;
        }
        let root_active: Vec<u32> = (0..self.num_predicates).collect();
        self.row_unbound_eager_rec(0, 0, &root_active, 0, 0, self.n, s as u64, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn row_unbound_eager_rec(&self, depth: u32, x: u64, active: &[u32], row0: u64, col0: u64, cell: u64, fixed_row: u64, out: &mut Vec<(u32, u32)>) {
        let is_last = depth == self.height - 1;
        let w = active.len() as u64;
        let region_bv = if is_last { &self.l } else { &self.t };
        let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
        let block_start = region_base + x;
        let child_cell = cell / self.k as u64;
        let dr = (fixed_row - row0) / child_cell;

        for dc in 0..self.k as u64 {
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;
            let mut child_active = Vec::new();
            for (j, &pid) in active.iter().enumerate() {
                if region_bv.get(slice_start + j as u64) {
                    child_active.push(pid);
                }
            }
            if child_active.is_empty() {
                continue;
            }
            let child_col0 = col0 + dc * child_cell;
            if is_last {
                for &pid in &child_active {
                    out.push((pid, child_col0 as u32));
                }
                continue;
            }
            let child_x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            let child_row0 = row0 + dr * child_cell;
            self.row_unbound_eager_rec(depth + 1, child_x, &child_active, child_row0, child_col0, child_cell, fixed_row, out);
        }
    }

    /// Lazy unbound-predicate row query: same result as
    /// [`Self::row_unbound_eager`], but predicate ids are recovered via
    /// `select1` only at leaf hits, not materialized at every visited node.
    /// Ascending predicates linking a specific (subject, object) pair.
    /// Unlike [`Self::row_unbound_lazy`] (which must branch over every
    /// object reachable from a fixed subject), both S and O are already
    /// fully bound here, so there's exactly one path down the tree -- no
    /// recursion/branching needed, just [`Self::locate`]'s descent loop
    /// with the target predicate left unresolved until the leaf, then
    /// lazy `select1`-based decode for whichever bits are actually set
    /// there. O(height) descent plus O(hits) `select1` calls, the same
    /// complexity shape as `row_unbound_lazy`'s headline trick -- not
    /// O(num_predicates) the way `Mk2Forest::predicates_for_seq` is forced
    /// to be, since there's no notion of "which independent tree" here.
    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        let mut out = Vec::new();
        if (s as u64) >= self.n || (o as u64) >= self.n {
            return out;
        }
        let (s, o) = (s as u64, o as u64);
        let (mut row0, mut col0, mut cell) = (0u64, 0u64, self.n);
        let (mut x, mut w) = (0u64, self.num_predicates as u64);
        let mut path: Vec<u64> = Vec::new();
        for depth in 0..self.height {
            let is_last = depth == self.height - 1;
            let region_bv = if is_last { &self.l } else { &self.t };
            let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
            let block_start = region_base + x;
            let child_cell = cell / self.k as u64;
            let dr = (s - row0) / child_cell;
            let dc = (o - col0) / child_cell;
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;

            if is_last {
                for j in 0..w {
                    if region_bv.get(slice_start + j) {
                        out.push(self.resolve_predicate_id(&path, j));
                    }
                }
                break;
            }
            let child_w = region_bv.rank1(slice_start + w) - region_bv.rank1(slice_start);
            if child_w == 0 {
                break; // no predicate connects s and o at all
            }
            let child_x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            path.push(slice_start);
            x = child_x;
            w = child_w;
            row0 += dr * child_cell;
            col0 += dc * child_cell;
            cell = child_cell;
        }
        out.sort_unstable();
        out
    }

    pub fn row_unbound_lazy(&self, s: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        if (s as u64) >= self.n {
            return out;
        }
        let mut path = Vec::new();
        self.row_unbound_lazy_rec(0, 0, self.num_predicates as u64, 0, 0, self.n, s as u64, &mut path, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn row_unbound_lazy_rec(&self, depth: u32, x: u64, w: u64, row0: u64, col0: u64, cell: u64, fixed_row: u64, path: &mut Vec<u64>, out: &mut Vec<(u32, u32)>) {
        let is_last = depth == self.height - 1;
        let region_bv = if is_last { &self.l } else { &self.t };
        let region_base = if is_last { 0 } else { self.region_start_t[(depth + 1) as usize] };
        let block_start = region_base + x;
        let child_cell = cell / self.k as u64;
        let dr = (fixed_row - row0) / child_cell;

        for dc in 0..self.k as u64 {
            let slice_start = block_start + (dr * self.k as u64 + dc) * w;
            let child_col0 = col0 + dc * child_cell;
            if is_last {
                // `j` is already expressed relative to *this* node's own
                // active list (w) -- no inversion step is needed for
                // locating a specific child within it, so `path` (which
                // only records the T-level transitions taken to get here)
                // is used as-is.
                for j in 0..w {
                    if region_bv.get(slice_start + j) {
                        let pid = self.resolve_predicate_id(path, j);
                        out.push((pid, child_col0 as u32));
                    }
                }
                continue;
            }
            let child_w = region_bv.rank1(slice_start + w) - region_bv.rank1(slice_start);
            if child_w == 0 {
                continue;
            }
            let child_x = (region_bv.rank1(slice_start) - region_bv.rank1(region_base)) * self.kk();
            let child_row0 = row0 + dr * child_cell;
            path.push(slice_start);
            self.row_unbound_lazy_rec(depth + 1, child_x, child_w, child_row0, child_col0, child_cell, fixed_row, path, out);
            path.pop();
        }
    }

    /// Walks `path` bottom-up (leaf to root), converting a leaf-local
    /// predicate bit position back into a global predicate id via
    /// `select1` -- the inverse of the `rank1`-based compaction used while
    /// descending. Every entry in `path` is a `t`-resident slice start
    /// (the compaction step that produces a child's local predicate
    /// position always happens at a non-final depth, i.e. in `t`).
    fn resolve_predicate_id(&self, path: &[u64], leaf_local_pos: u64) -> u32 {
        let mut pos = leaf_local_pos;
        for &slice_start in path.iter().rev() {
            let absolute = self.t.select1(self.t.rank1(slice_start) + pos).expect("lazy decode: bit must exist, it was found present during descent");
            pos = absolute - slice_start;
        }
        pos as u32
    }
}

impl MemSize for Ik2Tree {
    fn mem_size_bytes(&self) -> usize {
        self.t.mem_size_bytes() + self.l.mem_size_bytes() + crate::common::metrics::vec_bytes(&self.region_start_t)
    }
}

impl Ik2Tree {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("t (unified topology + presence bits)", self.t.mem_size_bytes()),
            ("l (unified topology + presence bits)", self.l.mem_size_bytes()),
            ("region_start_t (O(height))", crate::common::metrics::vec_bytes(&self.region_start_t)),
        ]
    }
}

const IK2_MAGIC: &[u8; 4] = b"IK2A";

impl Ik2Tree {
    /// Serializes this tree to a self-contained byte buffer: a magic
    /// tag, `n`/`k`/`height`/`num_predicates`, `region_start_t` (length
    /// then its `u64`s), then `t`, then `l`. See
    /// `crate::common::k2_serialize`'s doc comment for why writes still
    /// buffer into a `Vec<u8>` while reads (below) stream directly.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(IK2_MAGIC);
        crate::common::tree_serialize::write_u64(&mut out, self.n);
        crate::common::tree_serialize::write_u32(&mut out, self.k);
        crate::common::tree_serialize::write_u32(&mut out, self.height);
        crate::common::tree_serialize::write_u32(&mut out, self.num_predicates);
        crate::common::tree_serialize::write_u64(&mut out, self.region_start_t.len() as u64);
        for &v in &self.region_start_t {
            crate::common::tree_serialize::write_u64(&mut out, v);
        }
        crate::common::k2_serialize::write_bitvector(&mut out, &self.t);
        crate::common::k2_serialize::write_bitvector(&mut out, &self.l);
        out
    }

    /// Reconstructs a tree from bytes already in memory (written by
    /// [`Self::serialize`]) -- a thin wrapper over
    /// [`Self::deserialize_from_reader`] via [`std::io::Cursor`]. Prefer
    /// [`Self::deserialize_from_path`] when reading a file.
    pub fn deserialize(buf: &[u8]) -> Self {
        Self::deserialize_from_reader(&mut std::io::Cursor::new(buf))
    }

    /// Reconstructs a tree by reading directly from a file written by
    /// [`Self::serialize`] -- never materializes the whole serialized
    /// content as one in-memory buffer.
    pub fn deserialize_from_path(path: &std::path::Path) -> Self {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path:?}: {e}"));
        Self::deserialize_from_reader(&mut std::io::BufReader::new(file))
    }

    fn deserialize_from_reader<R: std::io::Read>(r: &mut R) -> Self {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).expect("unexpected EOF reading Ik2Tree magic -- truncated or corrupt file");
        assert_eq!(&magic, IK2_MAGIC, "not a serialized Ik2Tree (bad magic)");
        let n = crate::common::tree_serialize::read_u64_from(r);
        let k = crate::common::tree_serialize::read_u32_from(r);
        let height = crate::common::tree_serialize::read_u32_from(r);
        let num_predicates = crate::common::tree_serialize::read_u32_from(r);
        let region_len = crate::common::tree_serialize::read_u64_from(r) as usize;
        let mut region_start_t = Vec::with_capacity(region_len);
        for _ in 0..region_len {
            region_start_t.push(crate::common::tree_serialize::read_u64_from(r));
        }
        let t = crate::common::k2_serialize::read_bitvector_from(r);
        let l = crate::common::k2_serialize::read_bitvector_from(r);
        Ik2Tree { t, l, region_start_t, n, k, height, num_predicates }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::mk2tree::Mk2Forest;
    use crate::common::rdf_gen::{uniform, zipfian_predicates};
    use crate::common::reference::NaiveIndex;

    fn cross_check(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        cross_check_k(triples, num_s, num_p, num_o, 2);
    }

    fn cross_check_k(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, k: u32) {
        let naive = NaiveIndex::build(triples);
        let mk2 = Mk2Forest::build(triples);
        let tree = Ik2Tree::build(k, triples);

        for s in 0..num_s {
            for p in 0..num_p {
                for o in 0..num_o {
                    assert_eq!(tree.locate(p, s, o), naive.contains(s, p, o), "locate({p},{s},{o})");
                }
                assert_eq!(tree.row_bound(p, s), naive.row(p, s), "row_bound(p={p},s={s})");
                assert_eq!(tree.row_bound(p, s), mk2.row(p, s), "row_bound vs Mk2Forest disagree p={p},s={s}");
            }
            for o in 0..num_o {
                for p in 0..num_p {
                    assert_eq!(tree.col_bound(p, o), naive.col(p, o), "col_bound(p={p},o={o})");
                }
                assert_eq!(tree.predicates_for(s, o), naive.predicates_for(s, o), "predicates_for(s={s},o={o})");
                assert_eq!(tree.predicates_for(s, o), mk2.predicates_for_seq(s, o), "predicates_for vs Mk2Forest disagree s={s},o={o}");
            }

            let mut expect_po: Vec<(u32, u32)> = Vec::new();
            for p in 0..num_p {
                for o in naive.row(p, s) {
                    expect_po.push((p, o));
                }
            }
            expect_po.sort_unstable();

            let mut eager = tree.row_unbound_eager(s);
            let mut lazy = tree.row_unbound_lazy(s);
            eager.sort_unstable();
            lazy.sort_unstable();
            assert_eq!(eager, expect_po, "eager unbound row(s={s}) vs NaiveIndex");
            assert_eq!(lazy, expect_po, "lazy unbound row(s={s}) vs NaiveIndex");
            assert_eq!(eager, lazy, "eager vs lazy decode disagree at s={s}");
        }
    }

    #[test]
    fn tiny_hand_traceable_case() {
        // A small, fully worked case: 2 predicates, 4x4 domain, k=2.
        // p=0: (0,0). p=1: (0,2), (3,3).
        let triples = vec![(0u32, 0u32, 0u32), (0, 1, 2), (3, 1, 3)];
        cross_check(&triples, 4, 2, 4);
    }

    #[test]
    fn cross_check_uniform_and_zipfian() {
        for seed in 1..5u64 {
            let triples = uniform(10, 5, 10, 150, seed * 37);
            cross_check(&triples, 10, 5, 10);
            let triples = zipfian_predicates(10, 8, 10, 150, 1.3, seed * 53);
            cross_check(&triples, 10, 8, 10);
        }
    }

    #[test]
    fn cross_check_k4_and_larger_domains() {
        for seed in 1..6u64 {
            let triples = uniform(30, 6, 30, 400, seed * 911);
            cross_check_k(&triples, 30, 6, 30, 4);
            let triples = uniform(17, 9, 17, 300, seed * 613);
            cross_check_k(&triples, 17, 9, 17, 2);
        }
    }

    #[test]
    fn cross_check_k8() {
        // k=8 (64-way branching, one u64 word per node) is the value used
        // throughout the compare benchmark for a fair rank-mechanism
        // comparison against RadixTree.
        for seed in 1..6u64 {
            let triples = uniform(30, 6, 30, 400, seed * 4111);
            cross_check_k(&triples, 30, 6, 30, 8);
            let triples = zipfian_predicates(40, 10, 40, 600, 1.2, seed * 2027);
            cross_check_k(&triples, 40, 10, 40, 8);
        }
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = Ik2Tree::build(2, &triples);
        assert!(!tree.locate(0, 0, 0));
        assert_eq!(tree.row_unbound_eager(0), Vec::new());
        assert_eq!(tree.row_unbound_lazy(0), Vec::new());
    }

    #[test]
    fn serialize_round_trip() {
        for seed in 1..6u64 {
            let triples = uniform(17, 9, 17, 300, seed * 613);
            let naive = NaiveIndex::build(&triples);
            let tree = Ik2Tree::build(8, &triples);

            let bytes = tree.serialize();
            let restored = Ik2Tree::deserialize(&bytes);
            assert_eq!(restored.n(), tree.n());
            assert_eq!(restored.height(), tree.height());
            assert_eq!(restored.num_predicates(), tree.num_predicates());

            for s in 0..17u32 {
                for p in 0..9u32 {
                    for o in 0..17u32 {
                        assert_eq!(restored.locate(p, s, o), naive.contains(s, p, o), "locate({p},{s},{o}) after round-trip");
                    }
                    assert_eq!(restored.row_bound(p, s), naive.row(p, s), "row_bound(p={p},s={s}) after round-trip");
                }
                let mut eager = restored.row_unbound_eager(s);
                let mut lazy = restored.row_unbound_lazy(s);
                eager.sort_unstable();
                lazy.sort_unstable();
                assert_eq!(eager, lazy, "eager vs lazy decode disagree after round-trip at s={s}");
            }

            // Also round-trip via a real file, exercising
            // `deserialize_from_path`'s streaming read path specifically.
            let path = std::env::temp_dir().join(format!("ik2_round_trip_test_{seed}.bin"));
            std::fs::write(&path, &bytes).unwrap();
            let from_path = Ik2Tree::deserialize_from_path(&path);
            std::fs::remove_file(&path).ok();
            for s in 0..17u32 {
                for p in 0..9u32 {
                    for o in 0..17u32 {
                        assert_eq!(from_path.locate(p, s, o), naive.contains(s, p, o), "locate({p},{s},{o}) after deserialize_from_path");
                    }
                }
            }
        }
    }
}
