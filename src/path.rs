//! BFS reachability / unweighted shortest-path (hop count), sequential
//! and (behind the `parallel` feature) `rayon`-parallel.
//!
//! For sparse IDs, bounded walks, path reconstruction and streamed path
//! enumeration, see [`crate::traversal`].

pub use crate::traversal::*;

use crate::storage::BlockStorage;
use crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicI32, Ordering};

/// Out-neighbors of `node`, optionally restricted to one predicate.
/// Returns a plain `Vec` (not a lazy stream): BFS just needs neighbor
/// lists, not seekable streams.
pub trait AdjacencyBackend {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32>;
    fn num_nodes(&self) -> usize;
    fn out_neighbors_into(&self, node: u32, predicate: Option<u32>, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.out_neighbors(node, predicate));
    }
}

impl<S: BlockStorage> AdjacencyBackend for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        match predicate {
            Some(p) => self.query_two_bound([0, 1], [node, p], 2),
            None => self.query_one_bound(0, node, 2),
        }
    }
    fn out_neighbors_into(&self, node: u32, predicate: Option<u32>, out: &mut Vec<u32>) {
        match predicate {
            Some(p) => self.query_two_bound_into([0, 1], [node, p], 2, out),
            None => self.query_one_bound_into(0, node, 2, out),
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
    let mut next = Vec::new();
    let mut neighbors = Vec::new();
    while !frontier.is_empty() {
        next.clear();
        for &u in &frontier {
            g.out_neighbors_into(u, predicate, &mut neighbors);
            for &v in &neighbors {
                if dist[v as usize] == -1 {
                    dist[v as usize] = level + 1;
                    next.push(v);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
        level += 1;
    }
    dist
}

/// Level-synchronous parallel BFS: each frontier level's neighbor
/// expansion runs via `rayon::par_iter`, with newly-discovered nodes
/// claimed through a single `compare_exchange` on an atomic distance
/// array so only one thread ever "wins" a given node.
#[cfg(feature = "parallel")]
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
            .filter(|&v| dist[v as usize].compare_exchange(-1, level + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok())
            .collect();
        frontier = next;
        level += 1;
    }
    dist.into_iter().map(|a| a.into_inner()).collect()
}

#[cfg(all(test, feature = "heap"))]
mod tests {
    use super::*;
    use crate::test_support::NaiveIndex;
    use crate::Triple;

    fn chain(num_chains: u32, chain_len: u32, num_predicates: u32, seed: u64) -> Vec<Triple> {
        let mut rng = crate::test_support::SplitMix64::new(seed);
        let mut out = Vec::new();
        for c in 0..num_chains {
            let base = c * (chain_len + 1);
            for i in 0..chain_len {
                out.push((base + i, rng.next_range(num_predicates as u64) as u32, base + i + 1));
            }
        }
        out
    }

    #[test]
    fn chain_distances_are_correct() {
        // A single chain of length 6: node i is reachable from node 0 in
        // exactly i hops (ignoring predicate).
        let triples = chain(1, 6, 1, 42);
        let heap = crate::Heap::build_dynamic(triples, 4096);
        let dist = bfs_sequential(&heap, 0, None);
        for i in 0..=6 {
            assert_eq!(dist[i], i as i32, "node {i} should be {i} hops from 0");
        }
    }

    #[test]
    fn sequential_matches_naive_bfs() {
        for seed in 1..4u64 {
            let triples = chain(5, 8, 3, seed * 271);
            let naive = NaiveIndex::build(&triples);
            let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0) as usize;
            let heap = crate::Heap::build_dynamic(triples, 4096);

            for start in [0u32, 1, 4, 9] {
                for predicate in [None, Some(0u32), Some(1)] {
                    let got = bfs_sequential(&heap, start, predicate);
                    // Naive reference via plain repeated out_neighbors expansion.
                    let mut expect = vec![-1i32; max_node + 1];
                    if (start as usize) <= max_node {
                        expect[start as usize] = 0;
                        let mut frontier = vec![start];
                        let mut level = 0i32;
                        while !frontier.is_empty() {
                            let mut next = Vec::new();
                            for u in frontier {
                                for v in naive.out_neighbors(u, predicate) {
                                    if expect[v as usize] == -1 {
                                        expect[v as usize] = level + 1;
                                        next.push(v);
                                    }
                                }
                            }
                            frontier = next;
                            level += 1;
                        }
                    }
                    assert_eq!(got[..=max_node], expect[..], "start={start} pred={predicate:?}");
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn sequential_and_parallel_agree() {
        let triples = chain(5, 8, 3, 271);
        let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0) as usize;
        let heap = crate::Heap::build_dynamic(triples, 4096);
        for start in [0u32, 1, 4, 9] {
            for predicate in [None, Some(0u32), Some(1)] {
                let seq = bfs_sequential(&heap, start, predicate);
                let par = bfs_parallel(&heap, start, predicate);
                assert_eq!(seq[..=max_node], par[..=max_node], "start={start} pred={predicate:?}");
            }
        }
    }
}
