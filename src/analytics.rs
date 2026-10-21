//! Compressed-edge dataflow: typed automata, contributions,
//! reducers and synchronous iteration. No analytics state is keyed by FST node.
use crate::automata_core::{
    And, Automaton, BitmapAutomaton, Mask, Project, Reducer, Transducer, TripleSource,
};
use crate::{Result, TreeError, Triple};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub source: u32,
    pub predicate: u32,
    pub target: u32,
}
/// Each edge is a one-event word; cross-edge effects belong to the reducer.
pub trait EdgeSource {
    type Workspace: Default;
    fn vertices(&self) -> usize;
    fn edges(&self, ws: &mut Self::Workspace, emit: impl FnMut(Edge));
}
#[derive(Clone, Copy)]
pub struct Predicate(pub Option<u32>);
impl<C: ?Sized> Automaton<Edge, C> for Predicate {
    type State = ();
    fn start(&self) {}
    fn step(&self, _: &(), e: &Edge, _: &C) -> Option<()> {
        self.0.is_none_or(|p| p == e.predicate).then_some(())
    }
    fn is_accepting(&self, _: &(), _: &C) -> bool {
        true
    }
}
#[derive(Clone, Copy)]
pub struct NonLoop;
impl<C: ?Sized> Automaton<Edge, C> for NonLoop {
    type State = ();
    fn start(&self) {}
    fn step(&self, _: &(), e: &Edge, _: &C) -> Option<()> {
        (e.source != e.target).then_some(())
    }
    fn is_accepting(&self, _: &(), _: &C) -> bool {
        true
    }
}
/// Statically composed one-edge transduction. The scan never materializes edges.
pub fn fold<S: EdgeSource, C: ?Sized, A: Transducer<Edge, C>, R: Reducer<A::Output>>(
    source: &S,
    ws: &mut S::Workspace,
    machine: &A,
    context: &C,
    reducer: &mut R,
) {
    let initial = machine.start();
    source.edges(ws, |e| {
        if let Some(state) = machine.step(&initial, &e, context)
            && machine.is_accepting(&state, context)
        {
            reducer.push(machine.output(&state, &e, context));
        }
    });
}
struct ScanStage {
    mask: u64,
    shifts: [u8; 3],
    remaining: [u8; 3],
    bits: [u32; 3],
}
#[derive(Clone, Copy, Default)]
struct Frame<C: Copy + Default> {
    cursor: C,
    word: u64,
    todo: u64,
    values: [u32; 3],
}
pub struct ScanWorkspace<C: Copy + Default> {
    frames: Vec<Frame<C>>,
}
impl<C: Copy + Default> Default for ScanWorkspace<C> {
    fn default() -> Self {
        Self { frames: Vec::new() }
    }
}
impl<C: Copy + Default> ScanWorkspace<C> {
    pub fn allocated_bytes(&self) -> usize {
        self.frames.capacity() * std::mem::size_of::<Frame<C>>()
    }
}
/// Exact predicate restriction is lowered to prefix-safe bitmap guards once.
/// Coordinate decoding is retained per occurrence, even for shared FST nodes.
pub struct CompressedEdges<S: TripleSource> {
    source: S,
    stages: Vec<ScanStage>,
    vertices: usize,
}
impl<S: TripleSource> CompressedEdges<S> {
    pub fn new(source: S, vertices: usize, predicate: Option<u32>) -> Result<Self> {
        if source.root().is_some_and(|root| source.word(root) != 0)
            && source.domains()[0].max(source.domains()[2]) > vertices as u64
        {
            return Err(TreeError::InvalidConfig(
                "vertex count smaller than endpoint domains".into(),
            ));
        }
        let invalid = predicate.is_some_and(|p| p as u64 >= source.domains()[1]);
        let stages = (0..source.height())
            .map(|l| {
                let (bits, shifts, remaining) = source.layout(l);
                let slots = 1u32 << bits.iter().sum::<u8>();
                let mut mask = if slots == 64 {
                    u64::MAX
                } else {
                    (1u64 << slots) - 1
                };
                if invalid {
                    mask = 0;
                }
                if let Some(p) = predicate {
                    for b in 0..bits[1] {
                        let ones = crate::native_join::slot_ones(shifts[1] + b);
                        mask &= if p & (1u32 << (remaining[1] + b)) != 0 {
                            ones
                        } else {
                            !ones
                        };
                    }
                }
                // This is the same Mask automaton used by the join guards.
                let mask = Mask(mask).mask(&(), l, &());
                ScanStage {
                    mask,
                    shifts,
                    remaining,
                    bits: bits.map(|b| (1u32 << b) - 1),
                }
            })
            .collect();
        Ok(Self {
            source,
            stages,
            vertices,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.stages.capacity() * std::mem::size_of::<ScanStage>()
    }
}
impl<S: TripleSource> EdgeSource for CompressedEdges<S> {
    type Workspace = ScanWorkspace<S::Cursor>;
    fn vertices(&self) -> usize {
        self.vertices
    }
    fn edges(&self, ws: &mut Self::Workspace, mut emit: impl FnMut(Edge)) {
        let height = self.stages.len();
        let Some(root) = self.source.root().filter(|_| height > 0) else {
            return;
        };
        ws.frames.resize(height, Frame::default());
        let word = self.source.word(root);
        ws.frames[0] = Frame {
            cursor: root,
            word,
            todo: word & self.stages[0].mask,
            values: [0; 3],
        };
        let mut depth = 0;
        loop {
            let f = &mut ws.frames[depth];
            if f.todo == 0 {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                continue;
            }
            let slot = f.todo.trailing_zeros();
            f.todo &= f.todo - 1;
            let stage = &self.stages[depth];
            let values = std::array::from_fn(|d| {
                f.values[d] | (((slot >> stage.shifts[d]) & stage.bits[d]) << stage.remaining[d])
            });
            if depth + 1 == height {
                emit(Edge {
                    source: values[0],
                    predicate: values[1],
                    target: values[2],
                });
            } else {
                let cursor = self.source.follow(f.cursor, f.word, slot);
                let word = self.source.word(cursor);
                depth += 1;
                ws.frames[depth] = Frame {
                    cursor,
                    word,
                    todo: word & self.stages[depth].mask,
                    values,
                };
            }
        }
    }
}
/// Comparison backend; never required by the compressed execution path.
/// Keeps predicates so projection=None retains distinct labelled parallel edges.
pub struct Csr {
    pub offsets: Vec<usize>,
    pub targets: Vec<u32>,
    predicates: Option<Vec<u32>>,
    selected_predicate: u32,
}
impl Csr {
    /// Build a transient projection from the engine's actual edge source.
    /// Two compressed passes; no uncompressed input relation is assumed.
    pub fn from_source<S: EdgeSource>(source: &S, predicate: Option<u32>) -> Self {
        let n = source.vertices();
        let mut ws = S::Workspace::default();
        let mut offsets = vec![0usize; n + 1];
        source.edges(&mut ws, |e| {
            if predicate.is_none_or(|p| p == e.predicate) {
                offsets[e.source as usize + 1] += 1;
            }
        });
        for v in 0..n {
            offsets[v + 1] += offsets[v];
        }
        let mut targets = vec![0; offsets[n]];
        let mut predicates = predicate.is_none().then(|| vec![0; offsets[n]]);
        let mut positions = offsets[..n].to_vec();
        source.edges(&mut ws, |e| {
            if predicate.is_none_or(|p| p == e.predicate) {
                let i = positions[e.source as usize];
                positions[e.source as usize] += 1;
                targets[i] = e.target;
                if let Some(ps) = &mut predicates {
                    ps[i] = e.predicate;
                }
            }
        });
        Self {
            offsets,
            targets,
            predicates,
            selected_predicate: predicate.unwrap_or(0),
        }
    }
    pub fn new(triples: &[Triple], vertices: usize, predicate: Option<u32>) -> Self {
        let mut triples: Vec<_> = triples
            .iter()
            .copied()
            .filter(|&(_, p, _)| predicate.is_none_or(|x| x == p))
            .collect();
        triples.sort_unstable();
        triples.dedup();
        let mut offsets = vec![0; vertices + 1];
        for &(s, _, o) in &triples {
            assert!((s as usize) < vertices && (o as usize) < vertices);
            offsets[s as usize + 1] += 1;
        }
        for v in 0..vertices {
            offsets[v + 1] += offsets[v];
        }
        let mut targets = Vec::with_capacity(triples.len());
        let mut predicates = predicate
            .is_none()
            .then(|| Vec::with_capacity(triples.len()));
        for (_, p, o) in triples {
            targets.push(o);
            if let Some(ps) = &mut predicates {
                ps.push(p);
            }
        }
        Self {
            offsets,
            targets,
            predicates,
            selected_predicate: predicate.unwrap_or(0),
        }
    }
    pub fn allocated_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.targets.capacity() * 4
            + self.predicates.as_ref().map_or(0, |ps| ps.capacity() * 4)
    }
}
impl EdgeSource for Csr {
    type Workspace = ();
    fn vertices(&self) -> usize {
        self.offsets.len() - 1
    }
    fn edges(&self, _: &mut (), mut emit: impl FnMut(Edge)) {
        for s in 0..self.vertices() {
            for i in self.offsets[s]..self.offsets[s + 1] {
                emit(Edge {
                    source: s as u32,
                    predicate: self
                        .predicates
                        .as_ref()
                        .map_or(self.selected_predicate, |ps| ps[i]),
                    target: self.targets[i],
                });
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct Iteration<T> {
    pub values: Vec<T>,
    pub rounds: usize,
    pub converged: bool,
    pub residual: f64,
}
pub struct Sum<'a>(pub &'a mut [f64]);
impl Reducer<(usize, f64)> for Sum<'_> {
    #[inline]
    fn push(&mut self, (v, value): (usize, f64)) {
        self.0[v] += value;
    }
}
/// Persistent vectors and scan frames are allocated once and reused each run.
pub struct PageRank<W> {
    scan: W,
    degree: Vec<usize>,
    pub rank: Vec<f64>,
    next: Vec<f64>,
    contribution: Vec<f64>,
}
impl<W: Default> Default for PageRank<W> {
    fn default() -> Self {
        Self {
            scan: W::default(),
            degree: Vec::new(),
            rank: Vec::new(),
            next: Vec::new(),
            contribution: Vec::new(),
        }
    }
}
impl<W: Default> PageRank<W> {
    pub fn allocated_bytes(&self) -> usize {
        self.degree.capacity() * std::mem::size_of::<usize>()
            + (self.rank.capacity() + self.next.capacity() + self.contribution.capacity()) * 8
    }
    pub fn run<S: EdgeSource<Workspace = W>>(
        &mut self,
        source: &S,
        damping: f64,
        tolerance: f64,
        max_rounds: usize,
    ) -> (usize, bool, f64) {
        assert!(damping.is_finite() && (0.0..1.0).contains(&damping));
        assert!(tolerance.is_finite() && tolerance >= 0.0);
        let n = source.vertices();
        self.degree.resize(n, 0);
        self.degree.fill(0);
        let degree_machine = Project(Predicate(None), |e: &Edge, _: &()| e.source as usize);
        fold(source, &mut self.scan, &degree_machine, &(), &mut |s| {
            self.degree[s] += 1
        });
        self.rank.resize(n, 0.0);
        self.rank.fill(if n == 0 { 0.0 } else { 1.0 / n as f64 });
        self.next.resize(n, 0.0);
        self.contribution.resize(n, 0.0);
        if n == 0 {
            return (0, true, 0.0);
        }
        let machine = Project(Predicate(None), |e: &Edge, c: &[f64]| {
            (e.target as usize, c[e.source as usize])
        });
        let mut residual = f64::INFINITY;
        for round in 1..=max_rounds {
            let mut dangling = 0.0;
            for v in 0..n {
                if self.degree[v] == 0 {
                    dangling += self.rank[v];
                    self.contribution[v] = 0.0;
                } else {
                    self.contribution[v] = damping * self.rank[v] / self.degree[v] as f64;
                }
            }
            self.next
                .fill((1.0 - damping + damping * dangling) / n as f64);
            fold(
                source,
                &mut self.scan,
                &machine,
                &self.contribution[..],
                &mut Sum(&mut self.next),
            );
            residual = self
                .rank
                .iter()
                .zip(&self.next)
                .map(|(a, b)| (a - b).abs())
                .sum();
            std::mem::swap(&mut self.rank, &mut self.next);
            if residual <= tolerance {
                return (round, true, residual);
            }
        }
        (max_rounds, false, residual)
    }
}
struct Votes<'a> {
    positions: &'a mut [usize],
    labels: &'a mut [u32],
}
impl Reducer<[(usize, u32); 2]> for Votes<'_> {
    #[inline]
    fn push(&mut self, contributions: [(usize, u32); 2]) {
        for (v, label) in contributions {
            self.labels[self.positions[v]] = label;
            self.positions[v] += 1;
        }
    }
}
/// Synchronous undirected projection, smallest-label tie break; self loops
/// excluded. Isolates retain their labels. May oscillate: max_rounds is required.
pub struct LabelPropagation<W> {
    scan: W,
    offsets: Vec<usize>,
    positions: Vec<usize>,
    votes: Vec<u32>,
    pub labels: Vec<u32>,
    next: Vec<u32>,
}
impl<W: Default> Default for LabelPropagation<W> {
    fn default() -> Self {
        Self {
            scan: W::default(),
            offsets: Vec::new(),
            positions: Vec::new(),
            votes: Vec::new(),
            labels: Vec::new(),
            next: Vec::new(),
        }
    }
}
impl<W: Default> LabelPropagation<W> {
    pub fn allocated_bytes(&self) -> usize {
        (self.offsets.capacity() + self.positions.capacity()) * std::mem::size_of::<usize>()
            + (self.votes.capacity() + self.labels.capacity() + self.next.capacity()) * 4
    }
    pub fn run<S: EdgeSource<Workspace = W>>(
        &mut self,
        source: &S,
        max_rounds: usize,
    ) -> (usize, bool) {
        let n = source.vertices();
        assert!(n <= u32::MAX as usize);
        self.offsets.resize(n + 1, 0);
        self.offsets.fill(0);
        let machine = Project(And(Predicate(None), NonLoop), |e: &Edge, _: &()| {
            [e.source as usize, e.target as usize]
        });
        fold(
            source,
            &mut self.scan,
            &machine,
            &(),
            &mut |ends: [usize; 2]| {
                for v in ends {
                    self.offsets[v + 1] += 1;
                }
            },
        );
        for v in 0..n {
            self.offsets[v + 1] += self.offsets[v];
        }
        self.positions.resize(n, 0);
        self.votes.resize(self.offsets[n], 0);
        self.labels.resize(n, 0);
        for (v, x) in self.labels.iter_mut().enumerate() {
            *x = v as u32;
        }
        self.next.resize(n, 0);
        if n == 0 {
            return (0, true);
        }
        let machine = Project(And(Predicate(None), NonLoop), |e: &Edge, labels: &[u32]| {
            [
                (e.source as usize, labels[e.target as usize]),
                (e.target as usize, labels[e.source as usize]),
            ]
        });
        for round in 1..=max_rounds {
            self.positions.copy_from_slice(&self.offsets[..n]);
            fold(
                source,
                &mut self.scan,
                &machine,
                &self.labels[..],
                &mut Votes {
                    positions: &mut self.positions,
                    labels: &mut self.votes,
                },
            );
            let mut changed = false;
            for v in 0..n {
                let votes = &mut self.votes[self.offsets[v]..self.offsets[v + 1]];
                votes.sort_unstable();
                let (mut best, mut count, mut i) = (self.labels[v], 0, 0);
                while i < votes.len() {
                    let label = votes[i];
                    let start = i;
                    while i < votes.len() && votes[i] == label {
                        i += 1;
                    }
                    if i - start > count {
                        best = label;
                        count = i - start;
                    }
                }
                self.next[v] = best;
                changed |= best != self.labels[v];
            }
            std::mem::swap(&mut self.labels, &mut self.next);
            if !changed {
                return (round, true);
            }
        }
        (max_rounds, false)
    }
}

// Deliberately independent reference loops (tuple input, no EdgeSource/fold).
pub fn reference_pagerank(
    triples: &[Triple],
    n: usize,
    damping: f64,
    tolerance: f64,
    rounds: usize,
) -> Iteration<f64> {
    let mut degree = vec![0; n];
    for &(s, _, _) in triples {
        degree[s as usize] += 1;
    }
    let mut rank = vec![if n == 0 { 0.0 } else { 1.0 / n as f64 }; n];
    if n == 0 {
        return Iteration {
            values: rank,
            rounds: 0,
            converged: true,
            residual: 0.0,
        };
    }
    let mut residual = f64::INFINITY;
    for round in 1..=rounds {
        let dangling: f64 = (0..n).filter(|&v| degree[v] == 0).map(|v| rank[v]).sum();
        let mut next = vec![(1.0 - damping + damping * dangling) / n as f64; n];
        for &(s, _, o) in triples {
            next[o as usize] += damping * rank[s as usize] / degree[s as usize] as f64;
        }
        residual = rank.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
        rank = next;
        if residual <= tolerance {
            return Iteration {
                values: rank,
                rounds: round,
                converged: true,
                residual,
            };
        }
    }
    Iteration {
        values: rank,
        rounds,
        converged: false,
        residual,
    }
}
pub fn reference_labels(triples: &[Triple], n: usize, rounds: usize) -> Iteration<u32> {
    use std::collections::BTreeMap;
    let mut labels: Vec<_> = (0..n as u32).collect();
    if n == 0 {
        return Iteration {
            values: labels,
            rounds: 0,
            converged: true,
            residual: 0.0,
        };
    }
    for round in 1..=rounds {
        let mut counts = vec![BTreeMap::<u32, usize>::new(); n];
        for &(s, _, o) in triples {
            if s != o {
                *counts[s as usize].entry(labels[o as usize]).or_default() += 1;
                *counts[o as usize].entry(labels[s as usize]).or_default() += 1;
            }
        }
        let next: Vec<_> = counts
            .iter()
            .enumerate()
            .map(|(v, c)| {
                c.iter()
                    .max_by_key(|&(label, count)| (*count, std::cmp::Reverse(*label)))
                    .map_or(labels[v], |(&label, _)| label)
            })
            .collect();
        if next == labels {
            return Iteration {
                values: next,
                rounds: round,
                converged: true,
                residual: 0.0,
            };
        }
        labels = next;
    }
    Iteration {
        values: labels,
        rounds,
        converged: false,
        residual: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_BLOCK_CAPACITY, automata_core::FstSource, path_fst::PathFstDeltaPow2Bfs};
    fn check(mut triples: Vec<Triple>, n: usize) {
        triples.sort_unstable();
        triples.dedup();
        let t = PathFstDeltaPow2Bfs::build(
            triples.clone(),
            [n.max(1) as u64, 3, n.max(1) as u64],
            DEFAULT_BLOCK_CAPACITY,
        );
        // Source domains include a dummy vertex when graph has no vertices.
        let scan = CompressedEdges::new(FstSource(&t), n.max(1), None).unwrap();
        let mut actual = Vec::new();
        scan.edges(&mut ScanWorkspace::default(), |e| {
            actual.push((e.source, e.predicate, e.target))
        });
        actual.sort_unstable();
        assert_eq!(actual, triples);
        for predicate in [None, Some(0), Some(1), Some(2), Some(3)] {
            let selected: Vec<_> = triples
                .iter()
                .copied()
                .filter(|&(_, p, _)| predicate.is_none_or(|x| x == p))
                .collect();
            let scan = CompressedEdges::new(FstSource(&t), n.max(1), predicate).unwrap();
            let csr = Csr::new(&triples, n.max(1), predicate);
            let projected = Csr::from_source(&scan, predicate);
            let mut projected_rows = Vec::new();
            projected.edges(&mut (), |e| {
                projected_rows.push((e.source, e.predicate, e.target))
            });
            projected_rows.sort_unstable();
            assert_eq!(projected_rows, selected);
            for rounds in [0, 1, 2, 30] {
                let reference = reference_pagerank(&selected, n.max(1), 0.85, 1e-12, rounds);
                let mut pr = PageRank::default();
                let status = pr.run(&scan, 0.85, 1e-12, rounds);
                let mut csr_pr = PageRank::default();
                let csr_status = csr_pr.run(&csr, 0.85, 1e-12, rounds);
                assert_eq!(status.0, reference.rounds);
                assert_eq!(status.1, reference.converged);
                assert_eq!(status.0, csr_status.0);
                for ((a, b), c) in pr.rank.iter().zip(&reference.values).zip(&csr_pr.rank) {
                    assert!((a - b).abs() < 1e-12 && (a - c).abs() < 1e-12);
                }
                assert!((pr.rank.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                let reference = reference_labels(&selected, n.max(1), rounds);
                let mut lp = LabelPropagation::default();
                let status = lp.run(&scan, rounds);
                assert_eq!(lp.labels, reference.values);
                assert_eq!(status, (reference.rounds, reference.converged));
                // Reuse an already-filled workspace and labels.
                assert_eq!(lp.run(&scan, rounds), status);
                assert_eq!(lp.labels, reference.values);
            }
        }
    }
    #[test]
    fn fixtures_and_oscillation() {
        check(vec![], 1);
        check(vec![(0, 0, 0)], 1);
        check(vec![(0, 0, 1)], 4); // directed sink plus isolates; LPA period two
        check(
            vec![
                (0, 0, 1),
                (1, 0, 2),
                (2, 0, 0),
                (0, 1, 1),
                (0, 0, 1),
                (3, 2, 3),
            ],
            6,
        );
        let s = Csr::new(&[(0, 0, 1)], 2, None);
        let mut lp = LabelPropagation::default();
        assert_eq!(lp.run(&s, 9), (9, false));
        assert_eq!(lp.labels, vec![1, 0]);
        let empty = Csr::new(&[], 0, None);
        assert_eq!(
            PageRank::default().run(&empty, 0.85, 1e-12, 10),
            (0, true, 0.0)
        );
        assert_eq!(LabelPropagation::default().run(&empty, 10), (0, true));
    }
    #[test]
    fn deterministic_random_graphs_and_shared_suffixes() {
        let mut seed = 15u64;
        for n in [2, 3, 7, 17, 65] {
            let mut triples = Vec::new();
            for s in 0..n {
                for o in 0..n {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if seed % 11 < 3 {
                        triples.push((s, (seed % 3) as u32, o));
                    }
                }
            }
            check(triples, n as usize);
        }
    }
    #[test]
    fn output_composition_and_reduction() {
        let source = Csr::new(&[(0, 0, 1), (0, 1, 2), (2, 0, 2), (1, 0, 2)], 3, None);
        let machine = Project(And(Predicate(Some(0)), NonLoop), |e: &Edge, _: &()| {
            (e.target as usize, 2.0)
        });
        let mut totals = vec![0.0; 3];
        fold(&source, &mut (), &machine, &(), &mut Sum(&mut totals));
        assert_eq!(totals, vec![0.0, 2.0, 2.0]);
    }
    #[test]
    fn scan_full_width_coordinates_and_empty_vertex_universe() {
        let triples = vec![(0, 1, u32::MAX), (u32::MAX, 2, 1)];
        let tree = PathFstDeltaPow2Bfs::build(
            triples.clone(),
            [1u64 << 32, 3, 1u64 << 32],
            DEFAULT_BLOCK_CAPACITY,
        );
        let scan = CompressedEdges::new(FstSource(&tree), 1usize << 32, None).unwrap();
        let mut actual = Vec::new();
        scan.edges(&mut ScanWorkspace::default(), |e| {
            actual.push((e.source, e.predicate, e.target))
        });
        actual.sort_unstable();
        assert_eq!(actual, triples);
        assert!(CompressedEdges::new(FstSource(&tree), 16, None).is_err());
        let empty = PathFstDeltaPow2Bfs::build(vec![], [1, 1, 1], DEFAULT_BLOCK_CAPACITY);
        let scan = CompressedEdges::new(FstSource(&empty), 0, None).unwrap();
        assert_eq!(
            PageRank::default().run(&scan, 0.85, 1e-12, 10),
            (0, true, 0.0)
        );
        assert_eq!(LabelPropagation::default().run(&scan, 10), (0, true));
    }
}
