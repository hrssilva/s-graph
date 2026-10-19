//! Diagnostic prototype, follow-up to `subtree_dedup_proto.rs` -- see
//! `DESIGN.md`'s "subtree deduplication" entry. That prototype measures
//! state-merging potential at this crate's *existing* per-level
//! granularity (each level groups up to `BASE`=6 bits, across up to all
//! three dimensions, into one combined 64-way digit before ever testing
//! for a match). This one tests a specific objection to that result: a
//! bit-level trie (1-bit branching) is a strict refinement of the 64-way
//! one, and refinement can only ever reveal *more or equal* mergeable
//! state, since two different prefixes can share a long run of
//! continuation bits that straddles a level boundary or an uneven
//! per-dimension split within a level -- invisible to a test that only
//! ever compares whole pre-grouped 64-way digits in one shot.
//!
//! The merge criterion has to be the real Myhill-Nerode one: two
//! *prefixes* (reached states) merge only if their entire *sets* of
//! remaining continuations are identical -- not just "some trailing bit
//! value happens to recur somewhere in the dataset" (an earlier version
//! of this file computed exactly that simpler, wrong thing via a flat
//! `HashSet` over trailing-bit values, which silently assumes every
//! prefix has at most one key below it; it does not, so that number did
//! not mean what it looked like it meant and is not used here). The
//! right computation needs an actual bit-level trie, built from the
//! sorted key list via recursive partitioning (mirrors `build_rec`'s own
//! partition-by-digit recursion, with a 2-way digit instead of up to
//! 64-way), with the same bottom-up Merkle-style hash merge
//! `subtree_dedup_rec` already uses on the real tree.
//!
//! This duplicates (not reuses -- `dynamic_schedule`/`total_bits_needed`
//! are `pub(crate)`, not reachable from an example binary) the exact
//! per-level, per-dimension bit-consumption policy `src/tree.rs` already
//! uses, then un-groups it to individual bits in the same MSB-first,
//! dimension-major order, so depths at real level boundaries are
//! directly comparable to `subtree_dedup_report`'s per-level counts.
//!
//! Kept in the repository (not deleted after measuring), same as
//! `subtree_dedup_proto.rs`, for comparison against further ideas.
//! Run with: cargo run --release --example bitlevel_dedup_proto

use std::collections::HashMap;
use std::hash::Hasher;

#[path = "common/mod.rs"]
mod common;
use common::rdf_gen::uniform;

const NUM_DIMS: usize = 3;
const BASE: u8 = 6; // K_SQUARE = 64, see src/tree.rs

/// Duplicated from `tree::total_bits_needed`.
fn total_bits_needed(dimension_sizes: [u64; NUM_DIMS]) -> [u8; NUM_DIMS] {
    let mut bits = [0u8; NUM_DIMS];
    for d in 0..NUM_DIMS {
        let mut cap = 1u64;
        while cap < dimension_sizes[d] {
            cap *= 2;
            bits[d] += 1;
        }
    }
    bits
}

/// Duplicated from `tree::dynamic_schedule` -- the real per-level,
/// even-split-across-active-dimensions policy.
fn dynamic_schedule(mut remaining: [u8; NUM_DIMS]) -> Vec<[u8; NUM_DIMS]> {
    let mut level_dim_bits = Vec::new();
    while remaining.iter().any(|&r| r > 0) {
        let mut this_level = [0u8; NUM_DIMS];
        let mut budget = BASE;
        loop {
            let active: Vec<usize> = (0..NUM_DIMS).filter(|&d| remaining[d] > 0).collect();
            if active.is_empty() || budget == 0 {
                break;
            }
            let share = (budget / active.len() as u8).max(1);
            let mut progressed = false;
            for &d in &active {
                if budget == 0 {
                    break;
                }
                let take = share.min(remaining[d]).min(budget);
                if take > 0 {
                    this_level[d] += take;
                    remaining[d] -= take;
                    budget -= take;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        level_dim_bits.push(this_level);
    }
    level_dim_bits
}

/// Un-groups the real schedule's per-level digits into the same bits,
/// individually, in MSB-first/dimension-major order -- same information,
/// finer granularity. Returns (per-triple full key, total_bits, cumulative
/// bit offset at the START of each real level -- i.e. the depth at which
/// a bit-level state count is directly comparable to
/// `subtree_dedup_report`'s per-level `distinct_count`/`node_count`).
fn bitlevel_keys(triples: &[(u32, u32, u32)], level_dim_bits: &[[u8; NUM_DIMS]], total_bits: [u8; NUM_DIMS]) -> (Vec<u64>, usize, Vec<usize>) {
    let grand_total: usize = total_bits.iter().map(|&b| b as usize).sum();
    assert!(grand_total <= 64, "test domain too wide for a u64 key");
    let mut cumulative_at_level_start = Vec::with_capacity(level_dim_bits.len());
    let mut acc = 0usize;
    for lvl in level_dim_bits {
        cumulative_at_level_start.push(acc);
        acc += lvl.iter().map(|&b| b as usize).sum::<usize>();
    }
    let keys = triples
        .iter()
        .map(|&(s, p, o)| {
            let coords = [s as u64, p as u64, o as u64];
            let mut remaining = total_bits;
            let mut key = 0u64;
            for lvl in level_dim_bits {
                for d in 0..NUM_DIMS {
                    let take = lvl[d];
                    for bit_i in 0..take {
                        let shift = remaining[d] - 1 - bit_i;
                        let bit = (coords[d] >> shift) & 1;
                        key = (key << 1) | bit;
                    }
                    remaining[d] -= take;
                }
            }
            key
        })
        .collect();
    (keys, grand_total, cumulative_at_level_start)
}

/// Recursive, correct (Myhill-Nerode: whole continuation-*set* equality,
/// not element overlap) bit-trie build + bottom-up hash-merge, directly
/// over a sorted key slice -- same technique `subtree_dedup_rec` uses on
/// the real tree, generalized to 2-way branching. `keys` must be sorted;
/// `depth` is bits already consumed (0-indexed from the MSB); `total_bits`
/// is this key's fixed width. Tallies, per depth, the raw (unmerged) node
/// count and the merged (distinct-hash) state count.
fn bit_trie_rec(keys: &[u64], depth: usize, total_bits: usize, raw_nodes: &mut [u64], merged: &mut [HashMap<u64, u32>]) -> u64 {
    raw_nodes[depth] += 1;
    if depth == total_bits {
        return 0; // the single shared "accept, nothing left" state
    }
    let shift = total_bits - 1 - depth;
    let split = keys.partition_point(|&k| (k >> shift) & 1 == 0);
    let (left, right) = keys.split_at(split);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write_u8(!left.is_empty() as u8);
    hasher.write_u8(!right.is_empty() as u8);
    if !left.is_empty() {
        hasher.write_u64(bit_trie_rec(left, depth + 1, total_bits, raw_nodes, merged));
    }
    if !right.is_empty() {
        hasher.write_u64(bit_trie_rec(right, depth + 1, total_bits, raw_nodes, merged));
    }
    let hash = hasher.finish();
    *merged[depth].entry(hash).or_insert(0) += 1;
    hash
}

fn report(name: &str, num_s: u32, num_p: u32, num_o: u32, num_triples: usize) {
    let triples = uniform(num_s, num_p, num_o, num_triples, 42);
    let domain = [num_s as u64, num_p as u64, num_o as u64];
    let total_bits = total_bits_needed(domain);
    let level_dim_bits = dynamic_schedule(total_bits);
    let (mut keys, grand_total, level_start_depth) = bitlevel_keys(&triples, &level_dim_bits, total_bits);
    keys.sort_unstable();

    let mut raw_nodes = vec![0u64; grand_total + 1];
    let mut merged: Vec<HashMap<u64, u32>> = (0..=grand_total).map(|_| HashMap::new()).collect();
    bit_trie_rec(&keys, 0, grand_total, &mut raw_nodes, &mut merged);

    println!("=== {name} (domain={num_s}x{num_p}x{num_o}, {num_triples} triples, {grand_total} total bits, {} real levels) ===", level_dim_bits.len());
    println!("  {:>6} {:>10} {:>14} {:>14} {:>8}  {:>10}", "level", "depth", "unmin_nodes", "bitlvl_merged", "ratio", "boundary?");

    for depth in 0..=grand_total {
        let level = level_start_depth.iter().position(|&d| d == depth);
        let unmin = raw_nodes[depth];
        let merged_count = merged[depth].len() as u64;
        let marker = match level {
            Some(l) => format!("level {l}"),
            None => String::new(),
        };
        println!(
            "  {:>6} {:>10} {:>14} {:>14} {:>7.3}  {:>10}",
            level.map(|l| l.to_string()).unwrap_or_default(),
            depth,
            unmin,
            merged_count,
            merged_count as f64 / unmin.max(1) as f64,
            marker
        );
    }
    println!();
}

fn main() {
    // Same small-scale shape already measured at 64-way/level granularity
    // via subtree_dedup_proto.rs (see DESIGN.md), for a direct comparison.
    report("uniform (small, symmetric)", 5_000, 200, 5_000, 100_000);
}
