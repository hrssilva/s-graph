use crate::graph::{JoinQuery, Term, TriplePattern};
use crate::path_fst::PathFstDeltaPow2Bfs;
#[test]
fn randomized_patterns_and_orders_match_enumeration() {
    use Term::{Const as C, Var as V};
    let triples: Vec<_> = (0..5)
        .flat_map(|s| {
            (0..3).flat_map(move |p| {
                (0..9)
                    .filter(move |o| (s * 17 + p * 5 + o) % 4 == 0)
                    .map(move |o| (s, p, o))
            })
        })
        .collect();
    let mut rng = 42u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng >> 32
    };
    for _ in 0..48 {
        let count = 1 + next() as usize % 3;
        let mut terms = Vec::new();
        for _ in 0..count {
            terms.push(std::array::from_fn::<_, 3, _>(|d| {
                if next() % 4 == 0 {
                    C((next() % [5, 3, 9][d]) as u32)
                } else {
                    V(next() as usize % 3)
                }
            }));
        }
        let used: Vec<_> = (0..3)
            .filter(|&v| terms.iter().any(|a| a.contains(&V(v))))
            .collect();
        let patterns = terms
            .into_iter()
            .map(|a| {
                let a = a.map(|t| match t {
                    V(v) => V(used.iter().position(|&x| x == v).unwrap()),
                    x => x,
                });
                p(a[0], a[1], a[2])
            })
            .collect::<Vec<_>>();
        check(
            &triples,
            [5, 3, 9],
            JoinQuery {
                num_vars: used.len(),
                patterns: patterns.clone(),
            },
        );
        check(
            &triples,
            [5, 3, 9],
            JoinQuery {
                num_vars: used.len(),
                patterns: patterns.into_iter().rev().collect(),
            },
        );
    }
}
fn p(s: Term, p: Term, o: Term) -> TriplePattern<Term> {
    TriplePattern {
        subject: s,
        predicate: p,
        object: o,
    }
}
fn check(triples: &[(u32, u32, u32)], dims: [u64; 3], query: JoinQuery<Term>) {
    let tree = PathFstDeltaPow2Bfs::build(triples.to_vec(), dims, crate::DEFAULT_BLOCK_CAPACITY);
    let mut oracle = Vec::new();
    let domain = *dims.iter().max().unwrap() as u32;
    fn brute(
        v: usize,
        b: &mut [u32],
        domain: u32,
        q: &JoinQuery<Term>,
        triples: &[(u32, u32, u32)],
        out: &mut Vec<Vec<u32>>,
    ) {
        if v == b.len() {
            let value = |t: Term| match t {
                Term::Const(x) => x,
                Term::Var(v) => b[v],
            };
            if q.patterns
                .iter()
                .all(|p| triples.contains(&(value(p.subject), value(p.predicate), value(p.object))))
            {
                out.push(b.to_vec());
            }
            return;
        }
        for x in 0..domain {
            b[v] = x;
            brute(v + 1, b, domain, q, triples, out);
        }
    }
    brute(
        0,
        &mut vec![0; query.num_vars],
        domain,
        &query,
        triples,
        &mut oracle,
    );
    oracle.sort_unstable();
    crate::scheduled_join::tests::check(&tree, &query, &oracle);
}
