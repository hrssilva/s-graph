//! Synthetic RDF triple generators used across every demo and the
//! comparison benchmark, plus a small hand-rolled PRNG (splitmix64) so
//! dataset generation doesn't need an external `rand` dependency.
//!
//! Four dataset "shapes", each stressing a different part of the
//! comparison:
//! - [`uniform`]: pure random triples, the baseline case.
//! - [`zipfian_predicates`]: subjects/objects uniform, but predicates drawn
//!   from a Zipfian distribution -- mirrors real RDF's heavy predicate skew
//!   (e.g. DBpedia), which is exactly what makes Mk²-tree's
//!   scan-every-predicate-tree strategy weak on unbound-predicate queries
//!   and IK²-tree's interleaving strong.
//! - [`chain`]: disjoint linear paths, for BFS/path-algorithm benchmarks.
//! - [`star_triangle`]: star patterns plus explicit 3-cycles, for
//!   graph-pattern / WCOJ join benchmarks.

pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)`.
    pub fn next_range(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub type Triple = (u32, u32, u32);

/// Pure uniform random triples over `[0,num_s) x [0,num_p) x [0,num_o)`.
/// May contain duplicates (harmless -- every structure here treats the
/// triple set as a set).
pub fn uniform(num_s: u32, num_p: u32, num_o: u32, num_triples: usize, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    (0..num_triples)
        .map(|_| {
            (
                rng.next_range(num_s as u64) as u32,
                rng.next_range(num_p as u64) as u32,
                rng.next_range(num_o as u64) as u32,
            )
        })
        .collect()
}

/// Builds a Zipfian (`weight(rank) = 1/rank^exponent`) cumulative
/// distribution over `n` ranks, for [`sample_zipfian`] to draw from --
/// shared by [`zipfian_predicates`] and [`zipfian_entities`] instead of
/// each rebuilding the same table inline.
fn zipfian_cumulative(n: u32, exponent: f64) -> (Vec<f64>, f64) {
    let mut cumulative = Vec::with_capacity(n as usize);
    let mut total = 0.0f64;
    for rank in 1..=n as u64 {
        total += 1.0 / (rank as f64).powf(exponent);
        cumulative.push(total);
    }
    (cumulative, total)
}

fn sample_zipfian(rng: &mut SplitMix64, cumulative: &[f64], total: f64, n: u32) -> u32 {
    let target = rng.next_f64() * total;
    match cumulative.binary_search_by(|probe| probe.partial_cmp(&target).unwrap()) {
        Ok(i) | Err(i) => i.min(n as usize - 1) as u32,
    }
}

/// Subjects/objects uniform; predicate drawn from a Zipfian distribution
/// (`weight(rank) = 1/rank^exponent`) over `num_p` predicate ids, so a small
/// number of predicates account for most triples.
pub fn zipfian_predicates(num_s: u32, num_p: u32, num_o: u32, num_triples: usize, exponent: f64, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let (cumulative, total) = zipfian_cumulative(num_p, exponent);
    (0..num_triples)
        .map(|_| {
            (
                rng.next_range(num_s as u64) as u32,
                sample_zipfian(&mut rng, &cumulative, total, num_p),
                rng.next_range(num_o as u64) as u32,
            )
        })
        .collect()
}

/// Subjects, predicates, *and* objects each drawn from independent
/// Zipfian distributions -- unlike [`zipfian_predicates`] (which only
/// skews predicate selection, leaving subjects/objects uniform), this also
/// skews *entity* selection, closer to a real RDF graph's degree
/// distribution: a small number of subjects/objects account for a
/// disproportionate share of triples (e.g. LUBM's University/Department
/// entities vs. its many low-degree Student/Publication entities), not
/// just a small number of predicates. Separate exponents per dimension
/// since real graphs typically skew predicate frequency far more sharply
/// than entity degree (a handful of predicates like `rdf:type` can
/// dominate a graph, while even a "hub" entity's degree is usually a much
/// smaller multiple of the median).
pub fn zipfian_entities(num_s: u32, num_p: u32, num_o: u32, num_triples: usize, s_exponent: f64, p_exponent: f64, o_exponent: f64, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let (s_cum, s_total) = zipfian_cumulative(num_s, s_exponent);
    let (p_cum, p_total) = zipfian_cumulative(num_p, p_exponent);
    let (o_cum, o_total) = zipfian_cumulative(num_o, o_exponent);
    (0..num_triples)
        .map(|_| {
            (
                sample_zipfian(&mut rng, &s_cum, s_total, num_s),
                sample_zipfian(&mut rng, &p_cum, p_total, num_p),
                sample_zipfian(&mut rng, &o_cum, o_total, num_o),
            )
        })
        .collect()
}

/// `num_chains` disjoint linear paths of length `chain_len` (number of
/// edges), each edge's predicate drawn uniformly from `num_predicates`.
/// Node ids are assigned so chains don't share nodes: chain `c`'s nodes are
/// `c*(chain_len+1) .. c*(chain_len+1)+chain_len`.
pub fn chain(num_chains: u32, chain_len: u32, num_predicates: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let mut triples = Vec::with_capacity((num_chains * chain_len) as usize);
    for c in 0..num_chains {
        let base = c * (chain_len + 1);
        for i in 0..chain_len {
            let p = rng.next_range(num_predicates as u64) as u32;
            triples.push((base + i, p, base + i + 1));
        }
    }
    triples
}

/// `num_centers` star hubs, each with `fanout` outgoing spokes to fresh leaf
/// nodes, plus explicit 3-cycles among a `triangle_density` fraction of the
/// leaves of each star (closing leaf[i] -> leaf[i+1] -> leaf[i+2] -> hub).
/// Predicates are drawn from a small fixed palette so 2-hop/triangle/star
/// join patterns have real matches to find.
pub fn star_triangle(num_centers: u32, fanout: u32, triangle_density: f64, num_predicates: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let mut triples = Vec::new();
    let mut next_id = num_centers; // leaves start right after the hub ids [0, num_centers)
    for hub in 0..num_centers {
        let mut leaves = Vec::with_capacity(fanout as usize);
        for _ in 0..fanout {
            let leaf = next_id;
            next_id += 1;
            let p = rng.next_range(num_predicates as u64) as u32;
            triples.push((hub, p, leaf));
            leaves.push(leaf);
        }
        let mut i = 0;
        while i + 2 < leaves.len() {
            if rng.next_f64() < triangle_density {
                let p1 = rng.next_range(num_predicates as u64) as u32;
                let p2 = rng.next_range(num_predicates as u64) as u32;
                let p3 = rng.next_range(num_predicates as u64) as u32;
                triples.push((leaves[i], p1, leaves[i + 1]));
                triples.push((leaves[i + 1], p2, leaves[i + 2]));
                triples.push((leaves[i + 2], p3, hub));
            }
            i += 3;
        }
    }
    triples
}

/// A fixed-schema predicate palette for [`graph_patterns_dataset`]: unlike
/// [`star_triangle`] (which assigns triangle-closing predicates randomly
/// per instance, fine for a one-off correctness check but not for a
/// benchmark query with a consistent schema), every hub's star/triangle/
/// 4-cycle here uses the *same* predicate id at the same position, so a
/// single `JoinQuery` with these predicates as constants finds a match at
/// literally every hub.
pub const GP_STAR: u32 = 0;
pub const GP_TRI_1: u32 = 1;
pub const GP_TRI_2: u32 = 2;
pub const GP_TRI_3: u32 = 3; // closes the triangle back to the hub
pub const GP_CYC_1: u32 = 4;
pub const GP_CYC_2: u32 = 5;
pub const GP_CYC_3: u32 = 6;
pub const GP_CYC_4: u32 = 7; // closes the 4-cycle back to the hub

/// `num_hubs` hubs, each with: `~fanout` star spokes (`hub -GP_STAR-> leaf`,
/// fanout jittered per hub in `[fanout/2, 3*fanout/2]` via `seed`), exactly
/// one closed triangle (`hub -GP_TRI_1-> a -GP_TRI_2-> b -GP_TRI_3-> hub`),
/// and exactly one closed 4-cycle (`hub -GP_CYC_1-> w -GP_CYC_2-> x
/// -GP_CYC_3-> y -GP_CYC_4-> hub`) -- every non-hub node is freshly
/// allocated and touched by exactly one hub, so hubs never interfere with
/// each other's patterns. Built for `crate::common::join`'s `ltj_run`
/// benchmarks: anchoring any of these three query shapes at any hub id in
/// `0..num_hubs` is guaranteed to find exactly that hub's own instance
/// (plus nothing else), giving a controllable, non-trivial, and
/// cross-structure-comparable join workload -- unlike this project's
/// `large_scale_*` scenarios (`crate::common::insertable_bench`), which
/// are unstructured random triples with no guaranteed join matches at all.
pub fn graph_patterns_dataset(num_hubs: u32, fanout: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let mut triples = Vec::with_capacity(num_hubs as usize * (fanout as usize + 7));
    let mut next_id = num_hubs;
    for hub in 0..num_hubs {
        let this_fanout = fanout / 2 + rng.next_range(fanout as u64 + 1) as u32;
        for _ in 0..this_fanout {
            let leaf = next_id;
            next_id += 1;
            triples.push((hub, GP_STAR, leaf));
        }

        let a = next_id;
        let b = next_id + 1;
        next_id += 2;
        triples.push((hub, GP_TRI_1, a));
        triples.push((a, GP_TRI_2, b));
        triples.push((b, GP_TRI_3, hub));

        let w = next_id;
        let x = next_id + 1;
        let y = next_id + 2;
        next_id += 3;
        triples.push((hub, GP_CYC_1, w));
        triples.push((w, GP_CYC_2, x));
        triples.push((x, GP_CYC_3, y));
        triples.push((y, GP_CYC_4, hub));
    }
    triples
}

// Deliberately *not* numbered to avoid colliding with `GP_STAR..GP_CYC_4`
// (0..7): this dataset never mixes with `graph_patterns_dataset`'s own
// triples (separate function, separate binary), so there's no actual
// collision risk, and low ids keep `Mk2Forest`/`Ik2Tree`'s own
// `num_predicates = max(predicate id) + 1` computation honest (2, not
// 10) -- see the `phase_wide_join` investigation this constant's
// numbering was originally an accidental confound in.
pub const GP_WIDE: u32 = 0;
pub const GP_MARK: u32 = 1;

/// Stresses the one structural asymmetry `crate::common::sorted_neighbors`'s
/// own doc comment calls out: only `Mk2Forest` (via `K2Tree`'s row/col
/// iterators) has a genuine lazy, subtree-pruning `seek` -- every other
/// backend's `SortedNeighbors` stream is `SortedVecIter`, which
/// materializes the *whole* row/col result eagerly and can only seek
/// *within* that already-computed list, never skip decoding any of it.
/// `graph_patterns_dataset`'s streams are all small (a handful to a few
/// dozen results), so that asymmetry never actually costs anything
/// there -- materializing something tiny is cheap regardless of
/// approach. This dataset is built specifically to make it cost
/// something: each hub gets a *wide* relation (`hub -GP_WIDE-> x`, one
/// triple per `x` in a `wide_fanout`-sized freshly-allocated block) and
/// a *sparse* one reusing `mark_count` values drawn from that same
/// block (`hub -GP_MARK-> x`). Joining the two on the shared variable
/// (see `examples/phase_wide_join.rs`) has a true answer of size
/// `mark_count` regardless of how large `wide_fanout` is -- a real
/// seek-based stream (leapfrogging the wide stream forward to each rare
/// mark value) never needs to decode most of the wide relation to find
/// it; an eager backend pays to materialize the entire wide relation
/// first no matter how selective the join turns out to be.
pub fn wide_intersection_dataset(num_hubs: u32, wide_fanout: u32, mark_count: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let mut triples = Vec::with_capacity(num_hubs as usize * (wide_fanout as usize + mark_count as usize));
    let mut next_id = num_hubs;
    for hub in 0..num_hubs {
        let wide_start = next_id;
        for _ in 0..wide_fanout {
            triples.push((hub, GP_WIDE, next_id));
            next_id += 1;
        }
        let mut marked: Vec<u32> = Vec::with_capacity(mark_count as usize);
        while marked.len() < mark_count as usize {
            let candidate = wide_start + rng.next_range(wide_fanout as u64) as u32;
            if !marked.contains(&candidate) {
                marked.push(candidate);
            }
        }
        for m in marked {
            triples.push((hub, GP_MARK, m));
        }
    }
    triples
}

/// Same shape as [`wide_intersection_dataset`], but each hub's own id
/// is spread out to `i * wide_fanout` instead of packed into `[0,
/// num_hubs)` -- built to test a specific hypothesis about *why*
/// `Mk2Forest` measured so much slower than `heap`/`file` on the packed
/// version (see `examples/phase_wide_join.rs`'s own investigation):
/// `K2Tree` is a *square* matrix, sized to the larger of its two
/// dimensions -- here, the wide relation's object domain
/// (`num_hubs * wide_fanout`). With hub ids packed into `[0, num_hubs)`,
/// that's minuscule next to the object domain, so the row discriminator
/// (`dr = (hub - row0) / child_cell`) computes to the *same* value for
/// every hub through most of the descent -- different hubs' data stays
/// aliased together, indistinguishable, until the very last couple of
/// levels, forcing a search through data that turns out to belong to
/// other hubs before finding this one's. Spreading hub ids out to the
/// same magnitude as the object domain should let `dr` separate
/// different hubs from the first few levels instead, the way `heap`/
/// `file`'s independent per-dimension bit budgets already do. Leaf ids
/// are allocated starting safely above the highest possible spread hub
/// id, so the two id spaces never collide.
pub fn wide_intersection_dataset_spread_s(num_hubs: u32, wide_fanout: u32, mark_count: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    let mut triples = Vec::with_capacity(num_hubs as usize * (wide_fanout as usize + mark_count as usize));
    let stride = wide_fanout.max(1);
    let mut next_id = num_hubs * stride + stride;
    for i in 0..num_hubs {
        let hub = i * stride;
        let wide_start = next_id;
        for _ in 0..wide_fanout {
            triples.push((hub, GP_WIDE, next_id));
            next_id += 1;
        }
        let mut marked: Vec<u32> = Vec::with_capacity(mark_count as usize);
        while marked.len() < mark_count as usize {
            let candidate = wide_start + rng.next_range(wide_fanout as u64) as u32;
            if !marked.contains(&candidate) {
                marked.push(candidate);
            }
        }
        for m in marked {
            triples.push((hub, GP_MARK, m));
        }
    }
    triples
}
