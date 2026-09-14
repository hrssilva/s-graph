//! BFS reachability / unweighted shortest-path (hop count), sequential and
//! parallel, running over any [`AdjacencyBackend`] -- the same algorithm
//! executes unmodified over Mk²-tree, IK²-tree, Omnitree, and RadixTree.

use crate::common::ik2tree::Ik2Tree;
use crate::common::mk2tree::Mk2Forest;
use crate::common::omnitree::Omnitree;
use crate::common::radixtree::RadixTree;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};

/// Out-neighbors of `node`, optionally restricted to one predicate.
/// Returns a plain `Vec` (not a lazy stream): BFS just needs neighbor
/// lists, not seekable streams, so this stays simple.
pub trait AdjacencyBackend {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32>;
    fn num_nodes(&self) -> usize;
}

impl AdjacencyBackend for Mk2Forest {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.row(p, node),
            None => self.scan_all_seq(Some(node), None).into_iter().map(|(_, o)| o).collect(),
        }
    }
    fn num_nodes(&self) -> usize {
        self.node_upper_bound()
    }
}

impl AdjacencyBackend for Ik2Tree {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.row_bound(p, node),
            None => {
                let mut v: Vec<u32> = self.row_unbound_eager(node).into_iter().map(|(_, o)| o).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
        }
    }
    fn num_nodes(&self) -> usize {
        self.n() as usize
    }
}

impl AdjacencyBackend for Omnitree {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.objects_from_subject(node),
        }
    }
    fn num_nodes(&self) -> usize {
        // A node that only ever appears as an object (e.g. a chain's
        // terminal node) doesn't count towards the subject dimension's
        // domain size, so the distance array must be sized against
        // whichever dimension's max is larger.
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for RadixTree {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

/// `radixtree_veb`/`radixtree_dfs`/`radixtree_bp` share the exact same
/// `query_two_bound`/`query_one_bound`/`domain_size` surface as the
/// baseline (via [`crate::common::radixtree::RadixTreeApi`]), so this is
/// the identical body three times over, just against different concrete
/// types -- one blanket `impl<T: RadixTreeApi>` isn't possible here since
/// that would conflict with the (also-generic-friendly) baseline impl
/// above without specialization.
impl AdjacencyBackend for crate::common::radixtree_veb::RadixTreeVeb {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for crate::common::radixtree_dfs::RadixTreeDfs {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for crate::common::radixtree_bp::RadixTreeBp {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for crate::common::radixtree_dfs_bp::RadixTreeDfsBp {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

impl AdjacencyBackend for crate::common::radixtree_rankdelta::RadixTreeRankDelta {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn num_nodes(&self) -> usize {
        self.domain_size(0).max(self.domain_size(2)) as usize
    }
}

/// `dist[i]` is the hop count from `start` to node `i`, or `-1` if
/// unreachable.
pub fn bfs_sequential<B: AdjacencyBackend>(g: &B, start: u32, predicate: Option<u32>) -> Vec<i32> {
    let n = g.num_nodes();
    let mut dist = vec![-1i32; n];
    if (start as usize) >= n {
        return dist;
    }
    dist[start as usize] = 0;
    let mut frontier = vec![start];
    let mut level = 0i32;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for u in frontier {
            for v in g.out_neighbors(u, predicate) {
                if dist[v as usize] == -1 {
                    dist[v as usize] = level + 1;
                    next.push(v);
                }
            }
        }
        frontier = next;
        level += 1;
    }
    dist
}

/// Level-synchronous parallel BFS: each frontier level's neighbor
/// expansion runs via `rayon::par_iter`, with newly-discovered nodes
/// claimed through a single `compare_exchange` on an atomic distance array
/// so only one thread ever "wins" a given node, avoiding duplicate work and
/// races on the visited set.
pub fn bfs_parallel<B: AdjacencyBackend + Sync>(g: &B, start: u32, predicate: Option<u32>) -> Vec<i32> {
    let n = g.num_nodes();
    let dist: Vec<AtomicI32> = (0..n).map(|_| AtomicI32::new(-1)).collect();
    if (start as usize) >= n {
        return dist.into_iter().map(|a| a.into_inner()).collect();
    }
    dist[start as usize].store(0, Ordering::Relaxed);
    let mut frontier = vec![start];
    let mut level = 0i32;
    while !frontier.is_empty() {
        let next: Vec<u32> = frontier
            .par_iter()
            .flat_map_iter(|&u| g.out_neighbors(u, predicate).into_iter())
            .filter(|&v| {
                dist[v as usize]
                    .compare_exchange(-1, level + 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            })
            .collect();
        frontier = next;
        level += 1;
    }
    dist.into_iter().map(|a| a.into_inner()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rdf_gen::chain;

    #[test]
    fn sequential_and_parallel_agree_across_backends() {
        for seed in 1..4u64 {
            let triples = chain(5, 8, 3, seed * 271);
            // Each backend's internal domain is padded differently (k²-tree
            // rounds up to a power of k; Omnitree doesn't), so distance
            // vectors differ in trailing length even when they agree on
            // every node that actually appears in the data -- compare only
            // that meaningful prefix.
            let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0) as usize;
            let trim = |v: Vec<i32>| -> Vec<i32> { v[..=max_node].to_vec() };

            let mk2 = Mk2Forest::build(&triples);
            let ik2 = Ik2Tree::build(2, &triples);
            let omni = Omnitree::build(&triples);
            let radix = RadixTree::build(&triples, RadixTree::default_split());

            for start in [0u32, 1, 4, 9] {
                for predicate in [None, Some(0u32), Some(1)] {
                    let seq = trim(bfs_sequential(&mk2, start, predicate));
                    let par = trim(bfs_parallel(&mk2, start, predicate));
                    assert_eq!(seq, par, "Mk2Forest seq/par disagree start={start} pred={predicate:?}");

                    let seq_ik2 = trim(bfs_sequential(&ik2, start, predicate));
                    assert_eq!(seq_ik2, seq, "Ik2Tree disagrees with Mk2Forest start={start} pred={predicate:?}");
                    let par_ik2 = trim(bfs_parallel(&ik2, start, predicate));
                    assert_eq!(par_ik2, seq, "Ik2Tree seq/par disagree start={start} pred={predicate:?}");

                    let seq_omni = trim(bfs_sequential(&omni, start, predicate));
                    assert_eq!(seq_omni, seq, "Omnitree disagrees with Mk2Forest start={start} pred={predicate:?}");
                    let par_omni = trim(bfs_parallel(&omni, start, predicate));
                    assert_eq!(par_omni, seq, "Omnitree seq/par disagree start={start} pred={predicate:?}");

                    let seq_radix = trim(bfs_sequential(&radix, start, predicate));
                    assert_eq!(seq_radix, seq, "RadixTree disagrees with Mk2Forest start={start} pred={predicate:?}");
                    let par_radix = trim(bfs_parallel(&radix, start, predicate));
                    assert_eq!(par_radix, seq, "RadixTree seq/par disagree start={start} pred={predicate:?}");
                }
            }
        }
    }

    #[test]
    fn chain_distances_are_correct() {
        // A single chain of length 6: node i is reachable from node 0 in
        // exactly i hops (ignoring predicate).
        let triples = chain(1, 6, 1, 42);
        let mk2 = Mk2Forest::build(&triples);
        let dist = bfs_sequential(&mk2, 0, None);
        for i in 0..=6 {
            assert_eq!(dist[i], i as i32, "node {i} should be {i} hops from 0");
        }
    }
}
