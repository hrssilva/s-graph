//! Integer graph patterns and native prefix-join collection helpers.
//! For streaming results and reusable execution memory, use
//! [`crate::join::NativeJoin`].

use crate::storage::BlockStorage;
use crate::tree::RadixTreeDfsInterleavedVarSkipUnrolledGeneric;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    Var(usize),
    Const(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct TriplePattern<P = u32> {
    pub subject: Term,
    pub predicate: P,
    pub object: Term,
}

pub struct JoinQuery<P = u32> {
    pub num_vars: usize,
    pub patterns: Vec<TriplePattern<P>>,
}

impl From<u32> for Term {
    fn from(value: u32) -> Self { Self::Const(value) }
}

/// Compile and collect a native prefix join. For repeated execution or bounded
/// output memory, use [`crate::join::NativeJoin`] and its workspace directly.
/// Invalid queries panic; [`try_join_run`] returns validation errors instead.
/// Bindings are indexed by variable ID; result order is unspecified.
pub fn join_run<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> Vec<Vec<u32>> {
    try_join_run(query, tree).expect("invalid join query")
}

/// Fallible native prefix-join collection helper.
pub fn try_join_run<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> crate::Result<Vec<Vec<u32>>> {
    let plan = crate::join::NativeJoin::new(query, tree)?;
    let mut workspace = crate::join::NativeJoinWorkspace::default();
    let mut out = Vec::new();
    let _: std::ops::ControlFlow<()> = plan.visit(&mut workspace, |row| {
        out.push(row.to_vec());
        std::ops::ControlFlow::Continue(())
    });
    Ok(out)
}

/// Historical name for [`join_run`]; executes the native prefix algorithm.
pub use join_run as ltj_run;
/// Historical name for [`try_join_run`]; executes the native prefix algorithm.
pub use try_join_run as try_ltj_run;

/// Above this many candidate bits, [`apply_minus`] falls back to a
/// `HashSet` instead of a flat direct-indexed bitmap -- 8 MiB, picked so
/// the common case (a shared variable or two, each a few thousand to a
/// few million wide) stays on the fast, allocation-light bitmap path,
/// while a pattern combining several wide/unbounded variables doesn't
/// try to allocate a bitmap sized to their full cross product.
const MINUS_BITMAP_MAX_BITS: u128 = 1 << 26;

/// Like [`join_run`], additionally suppressing any row for which
/// substituting its bindings into one of `minus`'s patterns yields a
/// triple that actually exists (`MINUS`/`FILTER NOT EXISTS` over one
/// triple pattern each). Unlike a hand-rolled "materialize both sides as
/// bitmaps and subtract" store-side implementation, this never descends
/// once per row: each `minus` pattern is checked with exactly one
/// [`RadixTreeDfsInterleavedVarSkipUnrolledGeneric::contains_batch`]
/// call, after deduplicating the (often far smaller) set of *distinct*
/// ground triples the positive rows actually produce for it. Invalid
/// queries panic; use [`try_join_run_minus`] for a `Result` instead.
pub fn join_run_minus<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, minus: &[TriplePattern<P>], tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> Vec<Vec<u32>> {
    try_join_run_minus(query, minus, tree).expect("invalid join query")
}

/// Fallible version of [`join_run_minus`]. A `minus` pattern may use any
/// of `query.num_vars`'s variables (every one of them is already bound
/// by the time a full row exists); an out-of-range variable index is a
/// [`TreeError::InvalidConfig`].
pub fn try_join_run_minus<S: BlockStorage, P: Copy + Into<Term>>(query: &JoinQuery<P>, minus: &[TriplePattern<P>], tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>) -> crate::Result<Vec<Vec<u32>>> {
    let mut rows = try_join_run(query, tree)?;
    for pattern in minus {
        let atom = [pattern.subject, pattern.predicate.into(), pattern.object];
        for &term in &atom {
            if let Term::Var(v) = term
                && v >= query.num_vars
            {
                return Err(crate::TreeError::InvalidConfig("minus pattern variable index exceeds num_vars".into()));
            }
        }
        rows = apply_minus(tree, atom, rows);
        if rows.is_empty() {
            break;
        }
    }
    Ok(rows)
}

/// Removes every row whose `atom` (subject/predicate/object, each either
/// a constant or one of the row's own variables) resolves to a triple
/// that exists in `tree`. Dedupes the rows' *distinct* resolved ground
/// triples first -- a positive join commonly produces many rows sharing
/// the same minus-relevant values (e.g. one more-bound variable fanning
/// out over others the minus pattern doesn't use) -- then checks that
/// deduplicated set with one `contains_batch` call, and finally maps the
/// per-triple hit/miss result back across every original row, instead of
/// re-descending once per row (what a plain per-row `contains` loop, or
/// an external store-side bitmap-subtraction, would do).
fn apply_minus<S: BlockStorage>(tree: &RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S>, atom: [Term; 3], rows: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    if rows.is_empty() {
        return rows;
    }
    let resolve = |row: &[u32], t: Term| match t {
        Term::Const(c) => c,
        Term::Var(v) => row[v],
    };
    let ground = |row: &[u32]| -> crate::Triple { (resolve(row, atom[0]), resolve(row, atom[1]), resolve(row, atom[2])) };
    let grounds: Vec<crate::Triple> = rows.iter().map(|r| ground(r)).collect();
    let domains = tree.prefix_domains();
    let combined: u128 = (0..3).filter(|&d| matches!(atom[d], Term::Var(_))).map(|d| domains[d] as u128).product();

    if combined <= MINUS_BITMAP_MAX_BITS {
        let index = |g: crate::Triple| -> usize {
            let vals = [g.0 as u64, g.1 as u64, g.2 as u64];
            let mut idx = 0u64;
            for d in 0..3 {
                if matches!(atom[d], Term::Var(_)) {
                    idx = idx * domains[d] + vals[d];
                }
            }
            idx as usize
        };
        let words = (combined as usize).div_ceil(64).max(1);
        let mut seen = vec![0u64; words];
        let mut unique = Vec::new();
        for &g in &grounds {
            let i = index(g);
            if seen[i / 64] & (1 << (i % 64)) == 0 {
                seen[i / 64] |= 1 << (i % 64);
                unique.push(g);
            }
        }
        let hits = tree.contains_batch(&unique);
        let mut hit_bits = vec![0u64; words];
        for (&g, &found) in unique.iter().zip(&hits) {
            if found {
                let i = index(g);
                hit_bits[i / 64] |= 1 << (i % 64);
            }
        }
        rows.into_iter().zip(grounds).filter(|&(_, g)| { let i = index(g); hit_bits[i / 64] & (1 << (i % 64)) == 0 }).map(|(row, _)| row).collect()
    } else {
        let mut seen = std::collections::HashSet::new();
        let mut unique = Vec::new();
        for &g in &grounds {
            if seen.insert(g) {
                unique.push(g);
            }
        }
        let hits = tree.contains_batch(&unique);
        let hit_set: std::collections::HashSet<crate::Triple> = unique.into_iter().zip(hits).filter(|&(_, found)| found).map(|(g, _)| g).collect();
        rows.into_iter().zip(grounds).filter(|(_, g)| !hit_set.contains(g)).map(|(row, _)| row).collect()
    }
}

#[cfg(all(test, feature = "heap"))]
mod tests {
    use super::*;
    use crate::test_support::{uniform, NaiveIndex};
    use crate::Triple;

    fn naive_join(query: &JoinQuery, naive: &NaiveIndex, all_nodes: &[u32]) -> Vec<Vec<u32>> {
        fn rec(query: &JoinQuery, naive: &NaiveIndex, all_nodes: &[u32], var_idx: usize, bindings: &mut Vec<Option<u32>>, out: &mut Vec<Vec<u32>>) {
            if var_idx == query.num_vars {
                let candidate: Vec<u32> = bindings.iter().map(|b| b.unwrap()).collect();
                let ok = query.patterns.iter().all(|pat| {
                    let s = match pat.subject {
                        Term::Const(c) => c,
                        Term::Var(v) => candidate[v],
                    };
                    let o = match pat.object {
                        Term::Const(c) => c,
                        Term::Var(v) => candidate[v],
                    };
                    naive.contains(s, pat.predicate, o)
                });
                if ok {
                    out.push(candidate);
                }
                return;
            }
            for &node in all_nodes {
                bindings[var_idx] = Some(node);
                rec(query, naive, all_nodes, var_idx + 1, bindings, out);
            }
            bindings[var_idx] = None;
        }
        let mut out = Vec::new();
        let mut bindings = vec![None; query.num_vars];
        rec(query, naive, all_nodes, 0, &mut bindings, &mut out);
        out
    }

    fn sorted(mut v: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        v.sort_unstable();
        v
    }

    fn check_heap(triples: &[Triple], query: &JoinQuery, all_nodes: &[u32]) {
        let naive = NaiveIndex::build(triples);
        let expected = sorted(naive_join(query, &naive, all_nodes));
        assert!(!expected.is_empty(), "test dataset/query produced zero matches -- not exercising real join logic");

        let heap = crate::Heap::build_dynamic(triples.to_vec(), 4096);
        assert_eq!(sorted(ltj_run(query, &heap)), expected, "Heap join mismatch");
    }

    // Anchored compatibility cases; general patterns are tested in tests/native_join.rs.

    #[test]
    fn two_hop_path_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (0, 0, 3), (1, 1, 10), (1, 1, 11), (2, 2, 20)];
        let all_nodes: Vec<u32> = (0..30).collect();
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }, TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) }] };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn triangle_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 1, 4), (3, 2, 0)];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 2,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
                TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Const(0) },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn four_cycle_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 5), (1, 1, 2), (5, 1, 6), (2, 2, 3), (6, 2, 7), (3, 3, 0)];
        let all_nodes: Vec<u32> = (0..10).collect();
        let query = JoinQuery {
            num_vars: 3,
            patterns: vec![
                TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
                TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
                TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Var(2) },
                TriplePattern { subject: Term::Var(2), predicate: 3, object: Term::Const(0) },
            ],
        };
        check_heap(&triples, &query, &all_nodes);
    }

    #[test]
    fn star_pattern_with_bound_center() {
        let triples = uniform(1, 3, 12, 30, 1717);
        let mut triples: Vec<Triple> = triples.into_iter().map(|(_, p, o)| (0, p, o)).collect();
        triples.push((0, 0, 99));
        let max_node = triples.iter().flat_map(|&(s, _, o)| [s, o]).max().unwrap_or(0);
        let all_nodes: Vec<u32> = (0..=max_node).collect();
        let query = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }] };
        check_heap(&triples, &query, &all_nodes);
    }

    fn naive_minus(rows: Vec<Vec<u32>>, naive: &NaiveIndex, minus: &[TriplePattern]) -> Vec<Vec<u32>> {
        rows.into_iter()
            .filter(|row| {
                !minus.iter().any(|pat| {
                    let resolve = |t: Term| match t {
                        Term::Const(c) => c,
                        Term::Var(v) => row[v],
                    };
                    naive.contains(resolve(pat.subject), pat.predicate, resolve(pat.object))
                })
            })
            .collect()
    }

    fn check_heap_minus(triples: &[Triple], query: &JoinQuery, minus: &[TriplePattern], all_nodes: &[u32]) {
        let naive = NaiveIndex::build(triples);
        let expected = sorted(naive_minus(naive_join(query, &naive, all_nodes), &naive, minus));
        assert!(!expected.is_empty(), "test dataset/query/minus produced zero surviving rows -- not exercising real minus logic");

        let heap = crate::Heap::build_dynamic(triples.to_vec(), 4096);
        assert_eq!(sorted(try_join_run_minus(query, minus, &heap).unwrap()), expected, "Heap join_run_minus mismatch");
    }

    #[test]
    fn minus_removes_rows_with_a_matching_shared_variable_pattern() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (1, 0, 2), (1, 1, 2), (2, 0, 3), (0, 1, 2)];
        let all_nodes: Vec<u32> = (0..4).collect();
        // ?s --0--> ?o minus ?s --1--> ?o -- drops (s=0,o=2) since (0,1,2) exists.
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Var(1) }] };
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) }];
        check_heap_minus(&triples, &query, &minus, &all_nodes);
    }

    #[test]
    fn minus_pattern_that_never_matches_removes_nothing() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (1, 0, 2), (2, 0, 3)];
        let all_nodes: Vec<u32> = (0..4).collect();
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Var(1) }] };
        // Predicate 9 never occurs.
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 9, object: Term::Var(1) }];
        check_heap_minus(&triples, &query, &minus, &all_nodes);
    }

    #[test]
    fn minus_pattern_that_always_matches_removes_everything() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 9, 9), (1, 0, 2), (1, 9, 9), (2, 0, 3), (2, 9, 9)];
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Var(1) }] };
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 9, object: Term::Const(9) }];
        let heap = crate::Heap::build_dynamic(triples, 4096);
        assert!(try_join_run_minus(&query, &minus, &heap).unwrap().is_empty());
    }

    #[test]
    fn multiple_minus_patterns_combine_as_independent_not_exists_checks() {
        let triples: Vec<Triple> = vec![(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 1), (0, 2, 3)];
        let all_nodes: Vec<u32> = (0..4).collect();
        let query = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }] };
        // o=1 excluded by the first pattern, o=3 by the second; o=2 survives.
        let minus = [TriplePattern { subject: Term::Const(0), predicate: 1, object: Term::Var(0) }, TriplePattern { subject: Term::Const(0), predicate: 2, object: Term::Var(0) }];
        check_heap_minus(&triples, &query, &minus, &all_nodes);
    }

    /// The exact shape that made the earlier in-descent fusion slow: a
    /// variable (`?o1`) that many rows share because a *different*
    /// variable (`?o2`) fans out underneath it. Correctness here is what
    /// matters -- the dedup step must collapse those duplicates before
    /// `contains_batch`, not just happen to be fast.
    #[test]
    fn minus_dedupes_a_variable_many_rows_share_due_to_fanout() {
        let mut triples: Vec<Triple> = Vec::new();
        for s in 0..20u32 {
            triples.push((s, 0, s + 1)); // ?s --0--> ?o1, one edge per subject
        }
        for o1 in 0..21u32 {
            for o2 in 0..5u32 {
                triples.push((o1, 1, o2)); // ?o1 --1--> ?o2, dense fan-out
            }
        }
        for s in (0..20u32).step_by(2) {
            triples.push((s, 2, s + 1)); // half of the (s, o1) pairs get excluded
        }
        let all_nodes: Vec<u32> = (0..25).collect();
        let query = JoinQuery {
            num_vars: 3,
            patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Var(1) }, TriplePattern { subject: Term::Var(1), predicate: 1, object: Term::Var(2) }],
        };
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 2, object: Term::Var(1) }];
        check_heap_minus(&triples, &query, &minus, &all_nodes);
    }

    #[test]
    fn minus_falls_back_past_the_bitmap_threshold_and_still_matches() {
        // Subject/object domains of 1<<20 each (comfortably within u32)
        // push `combined` (their product, for a pattern using both as
        // variables) to 1<<40 -- well over `MINUS_BITMAP_MAX_BITS`,
        // forcing the `HashSet` branch. `build_dynamic_empty` declares
        // that domain explicitly, since `build_dynamic` would otherwise
        // infer a tiny one straight from the handful of actual triples
        // below.
        let mut heap = crate::Heap::build_dynamic_empty([1u64 << 20, 4, 1u64 << 20], 64);
        heap.insert(0, 0, 1).unwrap(); // row (s=0, o=1) -- excluded below
        heap.insert(1, 0, 2).unwrap(); // row (s=1, o=2) -- survives
        heap.insert(0, 1, 1).unwrap(); // matches minus's (s=0, p=1, o=1) ground check
        let query = JoinQuery { num_vars: 2, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Var(1) }] };
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) }];
        let rows = try_join_run_minus(&query, &minus, &heap).unwrap();
        assert_eq!(rows, vec![vec![1, 2]]);
    }

    #[test]
    fn minus_rejects_out_of_range_variable_index() {
        let heap = crate::Heap::build_dynamic(vec![(0u32, 0u32, 0u32)], 64);
        let query = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Const(0) }] };
        let minus = [TriplePattern { subject: Term::Var(5), predicate: 0, object: Term::Const(0) }];
        assert!(try_join_run_minus(&query, &minus, &heap).is_err());
    }

    #[test]
    fn minus_empty_query_results_short_circuit_without_checking() {
        let heap = crate::Heap::build_dynamic(vec![(0u32, 0u32, 0u32)], 64);
        let query = JoinQuery { num_vars: 1, patterns: vec![TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Const(99) }] };
        let minus = [TriplePattern { subject: Term::Var(0), predicate: 0, object: Term::Const(0) }];
        assert!(try_join_run_minus(&query, &minus, &heap).unwrap().is_empty());
    }
}
