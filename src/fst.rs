//! Default immutable graph index and allocation-free pattern scans.
use crate::path_fst::PathFstDeltaPow2Bfs;
use crate::tree::DimRestriction;
use crate::{Result, Triple};

/// Immutable, suffix-sharing graph index. Rebuild to apply updates; use
/// [`crate::Heap`] when local insertion and removal are required.
pub type Fst = PathFstDeltaPow2Bfs;

#[derive(Clone, Copy)]
pub(crate) struct ScanLayout {
    pub(crate) bits: [u32; 3],
    pub(crate) shifts: [u8; 3],
    pub(crate) remaining: [u8; 3],
    pub(crate) valid: u64,
}
impl ScanLayout {
    pub(crate) fn new(bits: [u8; 3], shifts: [u8; 3], remaining: [u8; 3]) -> Self {
        let slots = 1u32 << bits.iter().sum::<u8>();
        Self {
            bits: bits.map(|b| (1u32 << b) - 1),
            shifts,
            remaining,
            valid: if slots == 64 {
                u64::MAX
            } else {
                (1u64 << slots) - 1
            },
        }
    }
}

/// Allocation-free pattern iterator over the default FST.
pub type Scan<'a> = crate::path_ops::PatternIter<'a, crate::automata_core::FstSource<'a>>;

impl Fst {
    /// Compile an explicitly constructed automaton against this immutable graph.
    /// The expression can be reused against other graphs; execution memory is
    /// separate and reusable through the returned `automata::Search` plan.
    pub fn search<'a, A: crate::automata::Compile<crate::automata_core::FstSource<'a>>>(
        &'a self,
        machine: A,
    ) -> Result<A::Compiled> {
        machine.compile(crate::automata_core::FstSource(self))
    }
    /// Infer each coordinate domain from the largest ID (empty domains use 1).
    pub fn build_dynamic(triples: Vec<Triple>, block_capacity: usize) -> Self {
        Self::try_build_dynamic(triples, block_capacity).expect("invalid FST build")
    }

    pub fn try_build_dynamic(triples: Vec<Triple>, block_capacity: usize) -> Result<Self> {
        let mut domains = [1; 3];
        for &(s, p, o) in &triples {
            for (d, value) in [s, p, o].into_iter().enumerate() {
                domains[d] = domains[d].max(value as u64 + 1);
            }
        }
        Self::try_build(triples, domains, block_capacity)
    }

    pub fn domain_size(&self, dimension: usize) -> u64 {
        self.dim_sizes()[dimension]
    }
    pub fn dimension_sizes(&self) -> [u64; 3] {
        self.dim_sizes()
    }
    pub fn in_bounds(&self, s: u32, p: u32, o: u32) -> bool {
        [s, p, o]
            .into_iter()
            .zip(self.dim_sizes())
            .all(|(v, n)| (v as u64) < n)
    }

    /// Compile the default adaptive automaton join; reuse its workspace.
    pub fn join<P: Copy + Into<crate::graph::Term>>(
        &self,
        query: &crate::graph::JoinQuery<P>,
    ) -> Result<crate::Join<'_>> {
        crate::Join::new(query, self)
    }

    /// Compile a composable BGP in a global variable namespace. Variables not
    /// mentioned in this leaf remain unbound, so leaves can share one width.
    pub fn bgp(
        &self,
        query: &crate::graph::JoinQuery<crate::graph::Term>,
    ) -> Result<crate::query_ops::Bgp<crate::automata_core::FstSource<'_>>> {
        crate::query_ops::Bgp::new(
            query,
            crate::automata_core::FstSource(self),
            crate::scheduled_join::Policy::Adaptive,
        )
    }

    /// Exact bounds or `None` wildcards; iteration allocates no heap memory.
    #[inline]
    pub fn scan(&self, bounds: [Option<u32>; 3]) -> Scan<'_> {
        Scan::new(
            crate::automata_core::FstSource(self),
            &self.scan_layout,
            bounds,
        )
    }

    /// Count without decoding triples. Shared suffix results are memoized for
    /// this bound pattern; each matching logical prefix still contributes.
    pub fn count_matching(&self, bounds: [Option<u32>; 3]) -> u64 {
        let scan = self.scan(bounds);
        if scan.done {
            return 0;
        }
        fn count(
            graph: &Fst,
            addr: u64,
            level: usize,
            masks: &[u64],
            memo: &mut std::collections::HashMap<u64, u64>,
        ) -> u64 {
            if let Some(&n) = memo.get(&addr) {
                return n;
            }
            let word = graph.read_word(addr);
            let mut todo = word & masks[level];
            let mut n = 0;
            if level + 1 == graph.height() {
                n = todo.count_ones() as u64;
            } else {
                while todo != 0 {
                    let slot = todo.trailing_zeros();
                    todo &= todo - 1;
                    let i = (word & ((1u64 << slot) - 1)).count_ones() as usize;
                    n += count(
                        graph,
                        addr + graph.read_delta(addr, i),
                        level + 1,
                        masks,
                        memo,
                    );
                }
            }
            memo.insert(addr, n);
            n
        }
        count(
            self,
            self.root_addr(),
            0,
            &scan.masks,
            &mut std::collections::HashMap::new(),
        )
    }

    /// Independent point lookups in input order (no sorting or relation copy).
    pub fn contains_batch(&self, triples: &[Triple]) -> Vec<bool> {
        triples
            .iter()
            .map(|&(s, p, o)| self.contains(s, p, o))
            .collect()
    }
    pub fn contains_batch_into(&self, triples: &[Triple], out: &mut Vec<bool>) {
        out.clear();
        out.extend(triples.iter().map(|&(s, p, o)| self.contains(s, p, o)));
    }

    /// Sorted, distinct projected coordinate values, with exact/range/set bounds.
    pub fn select(&self, restrictions: [DimRestriction<'_>; 3], emit_dimension: usize) -> Vec<u32> {
        assert!(emit_dimension < 3);
        self.restricted_descent_via_composed_automaton(restrictions, emit_dimension)
    }
}

impl crate::join::PatternBackend for Fst {
    type Scan<'a> = Scan<'a>;
    #[inline]
    fn scan_pattern(&self, bounds: [Option<u32>; 3]) -> Self::Scan<'_> {
        self.scan(bounds)
    }
}

impl crate::path::AdjacencyBackend for Fst {
    fn out_neighbors(&self, node: u32, predicate: Option<u32>) -> Vec<u32> {
        let mut out = Vec::new();
        self.out_neighbors_into(node, predicate, &mut out);
        out
    }
    fn out_neighbors_into(&self, node: u32, predicate: Option<u32>, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.scan([Some(node), predicate, None]).map(|(_, _, o)| o));
        out.sort_unstable();
        out.dedup();
    }
    fn num_nodes(&self) -> usize {
        usize::try_from(self.domain_size(0).max(self.domain_size(2)))
            .expect("dense BFS domains require a 64-bit host")
    }
}
