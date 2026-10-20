//! Experimental relational automata: lift shared physical FST states to ordered
//! variable-bit decision DAGs, then lazily intersect or merge those DAGs.
use crate::compiled_filter::masks;
use crate::graph::{JoinQuery, Term};
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::DimRestriction;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::ops::ControlFlow;

// Compile-time tables hold internal integer tuples. Equality still verifies keys.
#[derive(Clone, Copy)]
struct KeyBuildHasher {
    fast: bool,
}
struct KeyHasher {
    fast: bool,
    state: u64,
    standard: std::collections::hash_map::DefaultHasher,
}
impl KeyHasher {
    fn mix(&mut self, value: u64) {
        self.state ^= value
            .wrapping_add(0x9e3779b97f4a7c15)
            .wrapping_mul(0xbf58476d1ce4e5b9);
        self.state = self.state.rotate_left(27).wrapping_mul(0x94d049bb133111eb);
    }
}
impl BuildHasher for KeyBuildHasher {
    type Hasher = KeyHasher;
    fn build_hasher(&self) -> KeyHasher {
        KeyHasher {
            fast: self.fast,
            state: 0x243f6a8885a308d3,
            standard: std::collections::hash_map::DefaultHasher::new(),
        }
    }
}
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        if !self.fast {
            return self.standard.finish();
        }
        let mut x = self.state;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        if !self.fast {
            self.standard.write(bytes);
            return;
        }
        for &byte in bytes {
            self.mix(byte as u64);
        }
        self.mix(bytes.len() as u64);
    }
    fn write_u32(&mut self, value: u32) {
        if self.fast {
            self.mix(value as u64);
        } else {
            self.standard.write_u32(value);
        }
    }
    fn write_u64(&mut self, value: u64) {
        if self.fast {
            self.mix(value);
        } else {
            self.standard.write_u64(value);
        }
    }
    fn write_usize(&mut self, value: usize) {
        if self.fast {
            self.mix(value as u64);
        } else {
            self.standard.write_usize(value);
        }
    }
}
type KeyMap<K, V> = HashMap<K, V, KeyBuildHasher>;

const MAX_ROOTS: usize = 32;
const NODE_BUDGET: usize = 2_000_000;
const APPLY_BUDGET: usize = 4_000_000;
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct Node {
    position: usize,
    low: u32,
    high: u32,
}
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Op {
    And,
    Or,
}
#[derive(Default, Clone, Debug)]
pub struct BuildStats {
    pub source_reads: u64,
    pub source_reuses: u64,
    pub nodes_built: usize,
    pub apply_calls: u64,
    pub apply_hits: u64,
    pub cached_products: usize,
}
struct Builder {
    nodes: Vec<Node>,
    unique: KeyMap<Node, u32>,
    computed: KeyMap<(Op, u32, u32), u32>,
    stats: BuildStats,
    node_limit: usize,
    apply_limit: usize,
    started_products: usize,
}
impl Builder {
    #[cfg(test)]
    fn new() -> Self {
        Self::with_hash(true)
    }
    fn with_hash(fast: bool) -> Self {
        Self {
            nodes: Vec::new(),
            unique: HashMap::with_hasher(KeyBuildHasher { fast }),
            computed: HashMap::with_hasher(KeyBuildHasher { fast }),
            stats: BuildStats::default(),
            node_limit: NODE_BUDGET,
            apply_limit: APPLY_BUDGET,
            started_products: 0,
        }
    }
    fn mk(&mut self, position: usize, low: u32, high: u32) -> Result<u32, String> {
        if low == high {
            return Ok(low);
        }
        let key = Node {
            position,
            low,
            high,
        };
        if let Some(&id) = self.unique.get(&key) {
            return Ok(id);
        }
        if self.nodes.len() >= self.node_limit {
            return Err("lifted decision-DAG node budget exceeded".into());
        }
        let id = self.nodes.len() as u32 + 2;
        self.nodes.push(key);
        self.unique.insert(key, id);
        Ok(id)
    }
    fn position(&self, id: u32) -> usize {
        if id < 2 {
            usize::MAX
        } else {
            self.nodes[id as usize - 2].position
        }
    }
    fn apply(&mut self, op: Op, mut a: u32, mut b: u32) -> Result<u32, String> {
        self.stats.apply_calls += 1;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        match op {
            Op::And => {
                if a == 0 {
                    return Ok(0);
                }
                if a == 1 || a == b {
                    return Ok(b);
                }
            }
            Op::Or => {
                if a == 0 || a == b {
                    return Ok(b);
                }
                if a == 1 {
                    return Ok(1);
                }
            }
        }
        let key = (op, a, b);
        if let Some(&id) = self.computed.get(&key) {
            self.stats.apply_hits += 1;
            return Ok(id);
        }
        if self.started_products >= self.apply_limit {
            return Err("lifted product budget exceeded".into());
        }
        self.started_products += 1;
        let position = self.position(a).min(self.position(b));
        let (a0, a1) = if self.position(a) == position {
            let n = self.nodes[a as usize - 2];
            (n.low, n.high)
        } else {
            (a, a)
        };
        let (b0, b1) = if self.position(b) == position {
            let n = self.nodes[b as usize - 2];
            (n.low, n.high)
        } else {
            (b, b)
        };
        let low = self.apply(op, a0, b0)?;
        let high = self.apply(op, a1, b1)?;
        let id = self.mk(position, low, high)?;
        self.computed.insert(key, id);
        Ok(id)
    }
}
#[derive(Clone, Copy)]
struct Test {
    variable: usize,
    bit: u8,
}
pub struct LiftedJoin {
    nodes: Vec<Node>,
    roots: Vec<u32>,
    tests: Vec<Test>,
    variables: usize,
    pub stats: BuildStats,
    merged: bool,
}
#[derive(Default, Debug)]
pub struct VisitStats {
    pub decisions: u64,
    pub rows: u64,
}
impl LiftedJoin {
    pub fn new(
        query: &JoinQuery<Term>,
        tree: &PathFstDeltaPow2Bfs,
        merged: bool,
    ) -> Result<Self, String> {
        Self::with_order(query, tree, merged, true)
    }
    pub fn with_order(
        query: &JoinQuery<Term>,
        tree: &PathFstDeltaPow2Bfs,
        merged: bool,
        shared_first: bool,
    ) -> Result<Self, String> {
        Self::with_order_and_hash(query, tree, merged, shared_first, true)
    }
    pub fn with_order_and_hash(
        query: &JoinQuery<Term>,
        tree: &PathFstDeltaPow2Bfs,
        merged: bool,
        shared_first: bool,
        fast: bool,
    ) -> Result<Self, String> {
        if query.num_vars > query.patterns.len().saturating_mul(3) {
            return Err("unused variables".into());
        }
        let mut domains = vec![1u64 << 32; query.num_vars];
        let mut used = vec![false; query.num_vars];
        let mut atoms = Vec::new();
        let mut empty = false;
        for pattern in &query.patterns {
            let atom = [pattern.subject, pattern.predicate, pattern.object];
            for d in 0..3 {
                match atom[d] {
                    Term::Const(v) => empty |= v as u64 >= tree.dim_sizes()[d],
                    Term::Var(v) => {
                        if v >= query.num_vars {
                            return Err("variable index outside query".into());
                        }
                        used[v] = true;
                        domains[v] = domains[v].min(tree.dim_sizes()[d]);
                    }
                }
            }
            if !atoms.contains(&atom) {
                atoms.push(atom);
            }
        }
        if used.contains(&false) {
            return Err("unused variables".into());
        }
        let widths: Vec<u8> = domains
            .iter()
            .map(|&d| (64 - (d - 1).leading_zeros()) as u8)
            .collect();
        let mut tests = Vec::new();
        let mut positions = vec![[usize::MAX; 32]; query.num_vars];
        if shared_first {
            let mut variables: Vec<usize> = (0..query.num_vars).collect();
            variables.sort_by_key(|&v| {
                let degree = atoms.iter().filter(|a| a.contains(&Term::Var(v))).count();
                let first_pair = atoms.len() >= 2
                    && atoms[0].contains(&Term::Var(v))
                    && atoms[1].contains(&Term::Var(v));
                (std::cmp::Reverse(degree), std::cmp::Reverse(first_pair), v)
            });
            for v in variables {
                for bit in (0..widths[v]).rev() {
                    positions[v][bit as usize] = tests.len();
                    tests.push(Test { variable: v, bit });
                }
            }
        } else {
            for bit in (0..widths.iter().copied().max().unwrap_or(0)).rev() {
                for v in 0..query.num_vars {
                    if bit < widths[v] {
                        positions[v][bit as usize] = tests.len();
                        tests.push(Test { variable: v, bit });
                    }
                }
            }
        }
        let mut b = Builder::with_hash(fast);
        let mut roots = Vec::new();
        if empty || (!atoms.is_empty() && tree.height() == 0) {
            roots.push(0);
        } else {
            // Enforce non-power-of-two domains in the common assignment universe.
            let mut bounds = 1;
            for v in 0..domains.len() {
                let limit = domains[v] - 1;
                let mut root = 1;
                for bit in 0..widths[v] {
                    let pos = positions[v][bit as usize];
                    root = if limit & (1u64 << bit) != 0 {
                        b.mk(pos, 1, root)?
                    } else {
                        b.mk(pos, root, 0)?
                    };
                }
                bounds = b.apply(Op::And, bounds, root)?;
            }
            roots.push(bounds);
            for atom in atoms {
                let restrictions = atom.map(|t| match t {
                    Term::Const(v) => DimRestriction::Exact(v),
                    _ => DimRestriction::Free,
                });
                let ms = masks(tree, &restrictions);
                let mut memo = HashMap::new();
                let root = lift_node(
                    &mut b,
                    tree,
                    tree.root_addr(),
                    0,
                    &atom,
                    &positions,
                    &ms,
                    &mut memo,
                )?;
                roots.push(root);
            }
        }
        if merged {
            let mut root = 1;
            for &r in &roots {
                root = b.apply(Op::And, root, r)?;
                if root == 0 {
                    break;
                }
            }
            roots = vec![root];
        } else {
            roots.retain(|&r| r != 1);
            if roots.contains(&0) {
                roots = vec![0];
            }
        }
        if roots.len() > MAX_ROOTS {
            return Err(format!(
                "lazy prototype supports at most {MAX_ROOTS} residual roots"
            ));
        }
        b.stats.nodes_built = b.nodes.len();
        b.stats.cached_products = b.computed.len();
        // Keep only nodes reachable from the final merged result or lazy roots.
        let mut compact = Vec::new();
        let mut remap = vec![u32::MAX; b.nodes.len() + 2];
        remap[0] = 0;
        remap[1] = 1;
        fn copy(id: u32, old: &[Node], new: &mut Vec<Node>, map: &mut [u32]) -> u32 {
            if map[id as usize] != u32::MAX {
                return map[id as usize];
            }
            let n = old[id as usize - 2];
            let low = copy(n.low, old, new, map);
            let high = copy(n.high, old, new, map);
            let next = new.len() as u32 + 2;
            new.push(Node {
                position: n.position,
                low,
                high,
            });
            map[id as usize] = next;
            next
        }
        for root in &mut roots {
            *root = copy(*root, &b.nodes, &mut compact, &mut remap);
        }
        Ok(Self {
            nodes: compact,
            roots,
            tests,
            variables: query.num_vars,
            stats: b.stats,
            merged,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.roots.capacity() * 4
            + self.tests.capacity() * std::mem::size_of::<Test>()
    }
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn visit<T>(
        &self,
        bindings: &mut Vec<u32>,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.run::<false, T>(bindings, &mut VisitStats::default(), &mut emit)
    }
    pub fn visit_with_stats<T>(
        &self,
        bindings: &mut Vec<u32>,
        stats: &mut VisitStats,
        mut emit: impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.run::<true, T>(bindings, stats, &mut emit)
    }
    fn run<const TRACK: bool, T>(
        &self,
        bindings: &mut Vec<u32>,
        stats: &mut VisitStats,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        *stats = VisitStats::default();
        bindings.resize(self.variables, 0);
        bindings.fill(0);
        let mut roots = [1; MAX_ROOTS];
        roots[..self.roots.len()].copy_from_slice(&self.roots);
        if self.roots.contains(&0) {
            return ControlFlow::Continue(());
        }
        self.step::<TRACK, T>(0, &roots, self.roots.len(), bindings, stats, emit)
    }
    fn step<const TRACK: bool, T>(
        &self,
        position: usize,
        roots: &[u32; MAX_ROOTS],
        count: usize,
        bindings: &mut [u32],
        stats: &mut VisitStats,
        emit: &mut impl FnMut(&[u32]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if position == self.tests.len() {
            if TRACK {
                stats.rows += 1;
            }
            return emit(bindings);
        }
        if TRACK {
            stats.decisions += 1;
        }
        let test = self.tests[position];
        let bit = 1u32 << test.bit;
        for high in [false, true] {
            let mut next = *roots;
            let mut live = true;
            for i in 0..count {
                let id = roots[i];
                if id >= 2 {
                    let node = self.nodes[id as usize - 2];
                    debug_assert!(node.position >= position);
                    if node.position == position {
                        next[i] = if high { node.high } else { node.low };
                    }
                }
                if next[i] == 0 {
                    live = false;
                    break;
                }
            }
            if live {
                if high {
                    bindings[test.variable] |= bit;
                } else {
                    bindings[test.variable] &= !bit;
                }
                self.step::<TRACK, T>(position + 1, &next, count, bindings, stats, emit)?;
            }
        }
        bindings[test.variable] &= !bit;
        ControlFlow::Continue(())
    }
    pub fn count(&self) -> Result<u128, String> {
        if !self.merged {
            return Err("count uses the merged result automaton".into());
        }
        fn scale(value: u128, gap: usize) -> Result<u128, String> {
            if value == 0 {
                return Ok(0);
            }
            let factor = 1u128.checked_shl(gap as u32).ok_or("count overflow")?;
            value.checked_mul(factor).ok_or("count overflow".into())
        }
        fn go(
            id: u32,
            nodes: &[Node],
            end: usize,
            memo: &mut [Option<u128>],
        ) -> Result<u128, String> {
            if id < 2 {
                return Ok(id as u128);
            }
            if let Some(value) = memo[id as usize] {
                return Ok(value);
            }
            let n = nodes[id as usize - 2];
            let mut total = 0u128;
            for child in [n.low, n.high] {
                let pos = if child < 2 {
                    end
                } else {
                    nodes[child as usize - 2].position
                };
                let amount = scale(go(child, nodes, end, memo)?, pos - n.position - 1)?;
                total = total.checked_add(amount).ok_or("count overflow")?;
            }
            memo[id as usize] = Some(total);
            Ok(total)
        }
        let root = self.roots[0];
        let mut memo = vec![None; self.nodes.len() + 2];
        let pos = if root < 2 {
            self.tests.len()
        } else {
            self.nodes[root as usize - 2].position
        };
        scale(go(root, &self.nodes, self.tests.len(), &mut memo)?, pos)
    }
}
#[allow(clippy::too_many_arguments)]
fn lift_node(
    b: &mut Builder,
    tree: &PathFstDeltaPow2Bfs,
    addr: u64,
    level: usize,
    atom: &[Term; 3],
    positions: &[[usize; 32]],
    ms: &[u64],
    memo: &mut HashMap<(u64, usize), u32>,
) -> Result<u32, String> {
    if let Some(&id) = memo.get(&(addr, level)) {
        b.stats.source_reuses += 1;
        return Ok(id);
    }
    b.stats.source_reads += 1;
    let word = tree.read_word(addr);
    let mut candidates = word & ms[level];
    let (bits, shifts, remaining) = tree.level_schedule(level);
    let mut result = 0;
    while candidates != 0 {
        let slot = candidates.trailing_zeros();
        candidates &= candidates - 1;
        let mut literals = [(0usize, false); 6];
        let mut n = 0;
        let mut valid = true;
        for d in 0..3 {
            if let Term::Var(v) = atom[d] {
                for offset in 0..bits[d] {
                    let value = (slot >> (shifts[d] + offset)) & 1 != 0;
                    let bit = remaining[d] + offset;
                    let pos = positions[v][bit as usize];
                    if pos == usize::MAX {
                        if value {
                            valid = false;
                        }
                        continue;
                    }
                    if let Some(&(_, old)) = literals[..n].iter().find(|&&(p, _)| p == pos) {
                        if old != value {
                            valid = false;
                        }
                    } else {
                        literals[n] = (pos, value);
                        n += 1;
                    }
                }
            }
        }
        if !valid {
            continue;
        }
        let child = if level + 1 == tree.height() {
            1
        } else {
            let addr =
                addr + tree.read_delta(addr, (word & ((1u64 << slot) - 1)).count_ones() as usize);
            lift_node(b, tree, addr, level + 1, atom, positions, ms, memo)?
        };
        if child == 0 {
            continue;
        }
        literals[..n].sort_unstable_by_key(|x| x.0);
        let mut cube = 1;
        for &(pos, high) in literals[..n].iter().rev() {
            cube = if high {
                b.mk(pos, 0, cube)?
            } else {
                b.mk(pos, cube, 0)?
            };
        }
        let branch = b.apply(Op::And, cube, child)?;
        result = b.apply(Op::Or, result, branch)?;
    }
    memo.insert((addr, level), result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TriplePattern;

    #[test]
    fn construction_budgets_return_errors() {
        let mut b = Builder::new();
        b.node_limit = 1;
        b.mk(0, 0, 1).unwrap();
        assert!(b.mk(1, 0, 1).is_err());
        let mut b = Builder::new();
        let left = b.mk(0, 0, 1).unwrap();
        let right = b.mk(1, 0, 1).unwrap();
        b.apply_limit = 0;
        assert!(b.apply(Op::And, left, right).is_err());
    }

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
        let tree =
            PathFstDeltaPow2Bfs::build(triples.to_vec(), dims, crate::DEFAULT_BLOCK_CAPACITY);
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
                if q.patterns.iter().all(|p| {
                    triples.contains(&(value(p.subject), value(p.predicate), value(p.object)))
                }) {
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
        for (merged, shared_first) in [(false, false), (true, false), (false, true), (true, true)] {
            let plan = LiftedJoin::with_order(&query, &tree, merged, shared_first).unwrap();
            let mut b = Vec::new();
            let mut actual = Vec::new();
            let _: ControlFlow<()> = plan.visit(&mut b, |r| {
                actual.push(r.to_vec());
                ControlFlow::Continue(())
            });
            actual.sort_unstable();
            assert_eq!(actual, oracle);
            if merged {
                assert_eq!(plan.count().unwrap(), oracle.len() as u128);
            }
            if !oracle.is_empty() {
                assert!(matches!(
                    plan.visit(&mut b, |_| ControlFlow::Break(())),
                    ControlFlow::Break(())
                ));
                let mut n = 0;
                let _: ControlFlow<()> = plan.visit(&mut b, |_| {
                    n += 1;
                    ControlFlow::Continue(())
                });
                assert_eq!(n, oracle.len());
            }
        }
    }
    #[test]
    fn lifted_atoms_and_joins_match_brute_force() {
        let triples: Vec<_> = (0..7)
            .flat_map(|s| {
                (0..3).flat_map(move |p| {
                    (0..11)
                        .filter(move |o| (s * 7 + p * 11 + o) % 5 == 0)
                        .map(move |o| (s, p, o))
                })
            })
            .collect();
        use Term::{Const as C, Var as V};
        let queries = vec![
            JoinQuery {
                num_vars: 2,
                patterns: vec![p(V(0), C(1), V(1))],
            },
            JoinQuery {
                num_vars: 2,
                patterns: vec![p(V(0), C(0), V(1)), p(V(1), C(1), V(0))],
            },
            JoinQuery {
                num_vars: 3,
                patterns: vec![
                    p(V(0), C(0), V(1)),
                    p(V(1), C(1), V(2)),
                    p(V(2), C(2), V(0)),
                ],
            },
            JoinQuery {
                num_vars: 1,
                patterns: vec![p(V(0), C(0), V(0))],
            },
            JoinQuery {
                num_vars: 2,
                patterns: vec![p(V(0), V(1), V(0)), p(C(1), V(1), V(0))],
            },
            JoinQuery {
                num_vars: 2,
                patterns: vec![p(V(0), C(0), C(1)), p(C(2), C(1), V(1))],
            },
            JoinQuery {
                num_vars: 0,
                patterns: vec![],
            },
            JoinQuery {
                num_vars: 0,
                patterns: vec![p(C(0), C(0), C(0))],
            },
            JoinQuery {
                num_vars: 1,
                patterns: vec![p(V(0), C(9), C(0))],
            },
        ];
        for query in queries {
            check(&triples, [7, 3, 11], query);
        }
    }
    #[test]
    fn sharing_preserves_enumeration_multiplicity() {
        use Term::{Const as C, Var as V};
        let triples: Vec<_> = (0..8)
            .flat_map(|s| (0..8).map(move |o| (s, 0, o)))
            .collect();
        check(
            &triples,
            [8, 1, 8],
            JoinQuery {
                num_vars: 3,
                patterns: vec![p(V(0), C(0), V(1)), p(V(1), C(0), V(2))],
            },
        );
        check(
            &[],
            [8, 1, 8],
            JoinQuery {
                num_vars: 2,
                patterns: vec![p(V(0), C(0), V(1))],
            },
        );
    }
    #[test]
    fn wide_ids_cross_dimensions_and_degenerate_domains() {
        use Term::{Const as C, Var as V};
        let triples = vec![(u32::MAX, 0, 1), (1, 1, 2), (2, 0, u32::MAX), (0, 0, 0)];
        let tree = PathFstDeltaPow2Bfs::build(
            triples,
            [1u64 << 32, 2, 1u64 << 32],
            crate::DEFAULT_BLOCK_CAPACITY,
        );
        let q = JoinQuery {
            num_vars: 2,
            patterns: vec![p(V(0), C(0), V(1))],
        };
        for merged in [false, true] {
            let plan = LiftedJoin::new(&q, &tree, merged).unwrap();
            let mut b = Vec::new();
            let mut rows = Vec::new();
            let _: ControlFlow<()> = plan.visit(&mut b, |r| {
                rows.push(r.to_vec());
                ControlFlow::Continue(())
            });
            rows.sort_unstable();
            assert_eq!(rows, vec![vec![0, 0], vec![2, u32::MAX], vec![u32::MAX, 1]]);
        }
        check(
            &[(0, 0, 1), (0, 0, 3)],
            [1, 1, 4],
            JoinQuery {
                num_vars: 1,
                patterns: vec![p(C(0), C(0), V(0))],
            },
        );
    }
}
