//! Omnitree (Pollinger, Ishii & Domke): a d-dimensional generalization of
//! k^d-trees where each *node* picks its own subset of dimensions to split,
//! rather than a fixed per-depth schedule. Here d=3 (subject, predicate,
//! object).
//!
//! Source: Pollinger, Ishii, Domke, "The Beauty of Anisotropic Mesh
//! Refinement: Omnitrees for Efficient Dyadic Discretizations", accepted to
//! Mathematics and Computers in Simulation (arXiv:2508.06316).
//!
//! Model: a node `v` has a d-bit label `b(v)`, bit `j` set iff dimension
//! `j` is split at `v`. A non-leaf has `2^|active|` *potential* child
//! slots, indexed by the active dimensions' binary tuple (MSB-first by
//! ascending dimension index); an unsplit dimension contributes no bit at
//! that node, unlike octrees (always split all d) or k^d-trees (always
//! split exactly 1) -- this per-node adaptivity is the structural
//! differentiator from `radix64.rs`'s own fixed-schedule design.
//!
//! **Split policy (an implementation choice, not from the paper)**: the
//! paper's own refinement criterion is external/adaptive to a geometric
//! problem, not a rule for building a tree from a static point set. Here we
//! use *greedy-maximal*: split every dimension that still has more than one
//! distinct value among the node's remaining points; stop (leaf) when none
//! do, or when the point count drops to [`MIN_BUCKET_SIZE`].
//!
//! **Storage**: preorder DFS of `labels` (one byte per node, low 3 bits =
//! `b(v)`; `label == 0` doubles as the leaf marker, valid because
//! greedy-maximal never produces an all-zero label at an internal node)
//! plus `subtree_size` (classic succinct-ordinal-tree skip pointers, one
//! `u32` per node: `descend_to_child` jumps directly past whole preceding
//! sibling subtrees in O(#preceding siblings) instead of scanning their
//! contents). Payload (leaf triples) lives in a separate array, T/L-style,
//! addressed via `leaf_start` prefix sums over `is_leaf_bits`.
//!
//! One thing the paper's abstract model doesn't need but a concrete
//! encoding does: unlike k²-tree, an Omnitree's `2^|active|` child slots
//! are *not* stored as a fixed-size array with empty slots marked -- empty
//! buckets are skipped entirely during construction (no bit spent on them).
//! So each internal node also carries a small `child_mask` byte: bit `i`
//! set iff combinatorial bucket `i` (of up to `2^3 = 8`) is actually
//! present. `descend_to_child` needs a bucket's *presence rank* (0-indexed
//! among only the present buckets, matching DFS order), recovered via
//! `popcount(child_mask & ((1 << bucket) - 1))`.

use crate::common::bitvector::BitVector;
use crate::common::metrics::MemSize;
use crate::common::rdf_gen::Triple;

const MIN_BUCKET_SIZE: usize = 1;
const NUM_DIMS: usize = 3; // S, P, O

pub struct Omnitree {
    labels: Vec<u8>,
    child_mask: Vec<u8>,
    subtree_size: Vec<u32>,
    is_leaf_bits: BitVector,
    leaf_start: Vec<u32>,
    payload: Vec<[u32; 3]>,
    domain_sizes: [u64; 3],
}

impl Omnitree {
    pub fn build(triples: &[Triple]) -> Self {
        let mut domain_sizes = [1u64; 3];
        for &(s, p, o) in triples {
            domain_sizes[0] = domain_sizes[0].max(s as u64 + 1);
            domain_sizes[1] = domain_sizes[1].max(p as u64 + 1);
            domain_sizes[2] = domain_sizes[2].max(o as u64 + 1);
        }
        let points: Vec<[u32; 3]> = triples.iter().map(|&(s, p, o)| [s, p, o]).collect();

        let mut labels = Vec::new();
        let mut child_mask = Vec::new();
        let mut subtree_size = Vec::new();
        let mut payload = Vec::new();
        let mut leaf_sizes: Vec<u32> = Vec::new();

        build_rec(&points, [0, 0, 0], domain_sizes, &mut labels, &mut child_mask, &mut subtree_size, &mut payload, &mut leaf_sizes);
        // `labels`/`child_mask`/`subtree_size` grow one push at a time
        // during the recursive build (final node count is data-adaptive,
        // not known up front), and `payload` grows via repeated
        // `extend_from_slice` across leaf buckets -- all via amortized
        // doubling, with no upfront `with_capacity`. Since `MemSize`
        // reports actual allocated bytes (`.capacity()`), that leftover
        // doubling slack would otherwise inflate every memory comparison
        // this structure feeds.
        labels.shrink_to_fit();
        child_mask.shrink_to_fit();
        subtree_size.shrink_to_fit();
        payload.shrink_to_fit();

        let mut is_leaf_bits = BitVector::with_capacity(labels.len());
        for &l in &labels {
            is_leaf_bits.push(l == 0);
        }
        is_leaf_bits.finalize();

        let mut leaf_start = Vec::with_capacity(leaf_sizes.len() + 1);
        let mut cum = 0u32;
        leaf_start.push(0);
        for sz in leaf_sizes {
            cum += sz;
            leaf_start.push(cum);
        }

        Omnitree {
            labels,
            child_mask,
            subtree_size,
            is_leaf_bits,
            leaf_start,
            payload,
            domain_sizes,
        }
    }

    pub fn node_count(&self) -> usize {
        self.labels.len()
    }

    pub fn domain_size(&self, dim: usize) -> u64 {
        self.domain_sizes[dim]
    }

    /// Ascending, deduplicated objects reachable from `s` regardless of
    /// predicate -- a one-bound-dimension scan (unlike
    /// [`Self::query_two_bound`]'s two), needed for predicate-unbound path
    /// algorithms.
    pub fn objects_from_subject(&self, s: u32) -> Vec<u32> {
        let mut out = Vec::new();
        if (s as u64) >= self.domain_sizes[0] {
            return out;
        }
        self.scan_one_bound_rec(0, [0, 0, 0], self.domain_sizes, 0, s, 2, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_one_bound_rec(&self, pos: usize, lo: [u64; 3], hi: [u64; 3], bound_dim: usize, bound_val: u32, target_dim: usize, out: &mut Vec<u32>) {
        let label = self.labels[pos];
        if label == 0 {
            let leaf_rank = self.is_leaf_bits.rank1(pos as u64) as usize;
            let (start, end) = (self.leaf_start[leaf_rank] as usize, self.leaf_start[leaf_rank + 1] as usize);
            for pt in &self.payload[start..end] {
                if pt[bound_dim] == bound_val {
                    out.push(pt[target_dim]);
                }
            }
            return;
        }
        let (active_dims, m) = active_dims_of(label);
        let mut mids = [0u64; NUM_DIMS];
        for i in 0..m {
            mids[i] = (lo[active_dims[i]] + hi[active_dims[i]]) / 2;
        }

        // Only bound_dim constrains branching; every other active dim
        // (including target_dim, and any third unrelated dim) branches
        // both ways since we don't care about its value distribution here.
        // Fixed-capacity (max 2^3=8), no heap allocation on this hot path.
        let mut candidates = [0usize; 8];
        let mut num_candidates = 1;
        for i in 0..m {
            let d = active_dims[i];
            let shift = m - 1 - i;
            if d == bound_dim {
                let bit = if bound_val as u64 >= mids[i] { 1 } else { 0 };
                for c in candidates.iter_mut().take(num_candidates) {
                    *c |= bit << shift;
                }
            } else {
                for j in 0..num_candidates {
                    candidates[num_candidates + j] = candidates[j] | (1 << shift);
                }
                num_candidates *= 2;
            }
        }

        for &bucket in candidates.iter().take(num_candidates) {
            let mask = self.child_mask[pos];
            if (mask >> bucket) & 1 == 0 {
                continue;
            }
            let rank = (mask & ((1u8 << bucket) - 1)).count_ones() as usize; // bucket <= 7, no overflow
            let child_pos = descend_to_child(&self.subtree_size, pos, rank);
            let mut child_lo = lo;
            let mut child_hi = hi;
            for i in 0..m {
                let d = active_dims[i];
                let shift = m - 1 - i;
                let bit = (bucket >> shift) & 1;
                if bit == 1 {
                    child_lo[d] = mids[i];
                } else {
                    child_hi[d] = mids[i];
                }
            }
            self.scan_one_bound_rec(child_pos, child_lo, child_hi, bound_dim, bound_val, target_dim, out);
        }
    }

    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        let point = [s, p, o];
        for d in 0..3 {
            if point[d] as u64 >= self.domain_sizes[d] {
                return false;
            }
        }
        let mut pos = 0usize;
        let mut lo = [0u64; 3];
        let mut hi = self.domain_sizes;
        loop {
            let label = self.labels[pos];
            if label == 0 {
                let leaf_rank = self.is_leaf_bits.rank1(pos as u64) as usize;
                let (start, end) = (self.leaf_start[leaf_rank] as usize, self.leaf_start[leaf_rank + 1] as usize);
                return self.payload[start..end].contains(&point);
            }
            let (active_dims, m) = active_dims_of(label);
            let mut bucket = 0usize;
            for i in 0..m {
                let d = active_dims[i];
                let mid = (lo[d] + hi[d]) / 2;
                let bit = if point[d] as u64 >= mid { 1 } else { 0 };
                bucket |= bit << (m - 1 - i);
                if bit == 1 {
                    lo[d] = mid;
                } else {
                    hi[d] = mid;
                }
            }
            let mask = self.child_mask[pos];
            if (mask >> bucket) & 1 == 0 {
                return false;
            }
            let rank = (mask & ((1u8 << bucket) - 1)).count_ones() as usize; // bucket <= 7, no overflow
            pos = descend_to_child(&self.subtree_size, pos, rank);
        }
    }

    /// Ascending predicates linking a specific (subject, object) pair --
    /// binds S and O, leaves P free. Omnitree's dimensions are symmetric
    /// (its adaptive per-node split doesn't privilege any axis), so this is
    /// just [`Self::query_two_bound`] with S,O bound instead of the usual
    /// row/col shapes -- no special-casing needed.
    pub fn predicates_for(&self, s: u32, o: u32) -> Vec<u32> {
        self.query_two_bound([0, 2], [s, o], 1)
    }

    /// Ascending values of `free_dim` such that a stored triple matches
    /// both bound dimensions. `bound_dims`/`bound_vals` give the two fixed
    /// dimensions (order-independent); `free_dim` is the third.
    pub fn query_two_bound(&self, bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for i in 0..2 {
            if bound_vals[i] as u64 >= self.domain_sizes[bound_dims[i]] {
                return out;
            }
        }
        self.query_rec(0, [0, 0, 0], self.domain_sizes, bound_dims, bound_vals, free_dim, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn query_rec(&self, pos: usize, lo: [u64; 3], hi: [u64; 3], bound_dims: [usize; 2], bound_vals: [u32; 2], free_dim: usize, out: &mut Vec<u32>) {
        let label = self.labels[pos];
        if label == 0 {
            let leaf_rank = self.is_leaf_bits.rank1(pos as u64) as usize;
            let (start, end) = (self.leaf_start[leaf_rank] as usize, self.leaf_start[leaf_rank + 1] as usize);
            for pt in &self.payload[start..end] {
                if pt[bound_dims[0]] == bound_vals[0] && pt[bound_dims[1]] == bound_vals[1] {
                    out.push(pt[free_dim]);
                }
            }
            return;
        }
        let (active_dims, m) = active_dims_of(label);
        let mut mids = [0u64; NUM_DIMS];
        for i in 0..m {
            mids[i] = (lo[active_dims[i]] + hi[active_dims[i]]) / 2;
        }

        // At most one active dim is the free one (dims are disjoint), so
        // there are either 1 (no free dim active here) or 2 (free dim
        // active) candidate buckets; pushing 0 before 1 keeps ascending
        // order, matching domain-halving's guarantee that the 0-branch's
        // values are all smaller than the 1-branch's. Fixed-capacity (max
        // 2^3=8), no heap allocation on this hot path.
        let mut candidates = [0usize; 8];
        let mut num_candidates = 1;
        for i in 0..m {
            let d = active_dims[i];
            let shift = m - 1 - i;
            if d == free_dim {
                for j in 0..num_candidates {
                    candidates[num_candidates + j] = candidates[j] | (1 << shift);
                }
                num_candidates *= 2;
            } else {
                let bound_val = if d == bound_dims[0] { bound_vals[0] } else { bound_vals[1] };
                let bit = if bound_val as u64 >= mids[i] { 1 } else { 0 };
                for c in candidates.iter_mut().take(num_candidates) {
                    *c |= bit << shift;
                }
            }
        }

        for &bucket in candidates.iter().take(num_candidates) {
            let mask = self.child_mask[pos];
            if (mask >> bucket) & 1 == 0 {
                continue;
            }
            let rank = (mask & ((1u8 << bucket) - 1)).count_ones() as usize; // bucket <= 7, no overflow
            let child_pos = descend_to_child(&self.subtree_size, pos, rank);
            let mut child_lo = lo;
            let mut child_hi = hi;
            for i in 0..m {
                let d = active_dims[i];
                let shift = m - 1 - i;
                let bit = (bucket >> shift) & 1;
                if bit == 1 {
                    child_lo[d] = mids[i];
                } else {
                    child_hi[d] = mids[i];
                }
            }
            self.query_rec(child_pos, child_lo, child_hi, bound_dims, bound_vals, free_dim, out);
        }
    }
}

/// Which dimensions a node's label splits, as a fixed-capacity `[usize; 3]`
/// (only the first `m` entries meaningful) instead of a heap-allocated
/// `Vec` -- called on every node visited during traversal, so avoiding an
/// allocation here matters.
#[inline]
fn active_dims_of(label: u8) -> ([usize; NUM_DIMS], usize) {
    let mut dims = [0usize; NUM_DIMS];
    let mut m = 0;
    for d in 0..NUM_DIMS {
        if (label >> d) & 1 == 1 {
            dims[m] = d;
            m += 1;
        }
    }
    (dims, m)
}

fn descend_to_child(subtree_size: &[u32], node_pos: usize, child_rank: usize) -> usize {
    let mut pos = node_pos + 1;
    for _ in 0..child_rank {
        pos += subtree_size[pos] as usize;
    }
    pos
}

#[allow(clippy::too_many_arguments)]
fn build_rec(
    points: &[[u32; 3]],
    lo: [u64; 3],
    hi: [u64; 3],
    labels: &mut Vec<u8>,
    child_mask: &mut Vec<u8>,
    subtree_size: &mut Vec<u32>,
    payload: &mut Vec<[u32; 3]>,
    leaf_sizes: &mut Vec<u32>,
) {
    let my_pos = labels.len();

    let mut active = [false; 3];
    for d in 0..3 {
        let mut it = points.iter().map(|p| p[d]);
        if let Some(first) = it.next() {
            if it.any(|v| v != first) {
                active[d] = true;
            }
        }
    }
    let any_active = active.iter().any(|&a| a);

    if !any_active || points.len() <= MIN_BUCKET_SIZE {
        labels.push(0);
        child_mask.push(0);
        subtree_size.push(1);
        leaf_sizes.push(points.len() as u32);
        payload.extend_from_slice(points);
        return;
    }

    let label = (active[0] as u8) | ((active[1] as u8) << 1) | ((active[2] as u8) << 2);
    labels.push(label);
    child_mask.push(0); // patched below
    subtree_size.push(0); // patched below

    let (active_dims, m) = active_dims_of(label);
    let mut mids = [0u64; NUM_DIMS];
    for i in 0..m {
        mids[i] = (lo[active_dims[i]] + hi[active_dims[i]]) / 2;
    }
    let num_buckets = 1usize << m;

    let mut buckets: Vec<Vec<[u32; 3]>> = vec![Vec::new(); num_buckets];
    for &pt in points {
        let mut idx = 0usize;
        for i in 0..m {
            let d = active_dims[i];
            let bit = if pt[d] as u64 >= mids[i] { 1 } else { 0 };
            idx |= bit << (m - 1 - i);
        }
        buckets[idx].push(pt);
    }

    let mut mask = 0u8;
    for (bidx, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        mask |= 1 << bidx;
        let mut child_lo = lo;
        let mut child_hi = hi;
        for i in 0..m {
            let d = active_dims[i];
            let bit = (bidx >> (m - 1 - i)) & 1;
            if bit == 1 {
                child_lo[d] = mids[i];
            } else {
                child_hi[d] = mids[i];
            }
        }
        build_rec(&bucket, child_lo, child_hi, labels, child_mask, subtree_size, payload, leaf_sizes);
    }
    child_mask[my_pos] = mask;
    subtree_size[my_pos] = (labels.len() - my_pos) as u32;
}

impl MemSize for Omnitree {
    fn mem_size_bytes(&self) -> usize {
        crate::common::metrics::vec_bytes(&self.labels)
            + crate::common::metrics::vec_bytes(&self.child_mask)
            + crate::common::metrics::vec_bytes(&self.subtree_size)
            + self.is_leaf_bits.mem_size_bytes()
            + crate::common::metrics::vec_bytes(&self.leaf_start)
            + crate::common::metrics::vec_bytes(&self.payload)
    }
}

impl Omnitree {
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("labels", crate::common::metrics::vec_bytes(&self.labels)),
            ("child_mask", crate::common::metrics::vec_bytes(&self.child_mask)),
            ("subtree_size", crate::common::metrics::vec_bytes(&self.subtree_size)),
            ("is_leaf_bits", self.is_leaf_bits.mem_size_bytes()),
            ("leaf_start", crate::common::metrics::vec_bytes(&self.leaf_start)),
            ("payload (explicit s,p,o per leaf point)", crate::common::metrics::vec_bytes(&self.payload)),
        ]
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_start.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::{uniform, zipfian_predicates};
    use crate::common::reference::NaiveIndex;

    fn cross_check(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32) {
        let naive = NaiveIndex::build(triples);
        let tree = Omnitree::build(triples);

        let mut leaf_total = 0u32;
        for i in 0..tree.leaf_start.len() - 1 {
            leaf_total += tree.leaf_start[i + 1] - tree.leaf_start[i];
        }
        assert_eq!(leaf_total as usize, tree.payload.len(), "leaf bucket sizes must sum to total payload");
        assert_eq!(tree.subtree_size[0] as usize, tree.node_count(), "root subtree_size must equal total node count");

        for s in 0..num_s {
            for p in 0..num_p {
                for o in 0..num_o {
                    assert_eq!(tree.contains(s, p, o), naive.contains(s, p, o), "contains({s},{p},{o})");
                }
                // (S,P) bound, O free -- the "row" query shape.
                let mut got = tree.query_two_bound([0, 1], [s, p], 2);
                let mut expect = naive.row(p, s);
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "query_two_bound(S={s},P={p},free=O)");
            }
        }
        for p in 0..num_p {
            for o in 0..num_o {
                let mut got = tree.query_two_bound([1, 2], [p, o], 0);
                let mut expect = naive.col(p, o);
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "query_two_bound(P={p},O={o},free=S)");
            }
        }
        // S,O bound, P free -- exercises the structure's genuine 3D
        // symmetry (unlike row/col, which mirror what a 2D structure can
        // also do).
        for s in 0..num_s {
            for o in 0..num_o {
                assert_eq!(tree.predicates_for(s, o), naive.predicates_for(s, o), "predicates_for(S={s},O={o})");
            }
        }
    }

    #[test]
    fn empty_tree() {
        let triples: Vec<Triple> = vec![];
        let tree = Omnitree::build(&triples);
        assert!(!tree.contains(0, 0, 0));
        assert_eq!(tree.query_two_bound([0, 1], [0, 0], 2), Vec::<u32>::new());
    }

    #[test]
    fn tiny_hand_traceable_case() {
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
    fn cross_check_larger_domain() {
        for seed in 1..4u64 {
            let triples = uniform(30, 6, 30, 400, seed * 911);
            cross_check(&triples, 30, 6, 30);
        }
    }

    #[test]
    fn duplicate_heavy_dataset() {
        // Many identical triples should collapse into one leaf bucket
        // without infinite recursion (any_active becomes false quickly).
        let mut triples = vec![(1u32, 1u32, 1u32); 20];
        triples.push((5, 5, 5));
        cross_check(&triples, 6, 6, 6);
    }
}
