//! One function per benchmark category, each building (or reusing) all
//! four structures over the same dataset and pushing timing/size rows
//! into the shared [`Report`].

use crate::common::ik2tree::Ik2Tree;
use crate::common::join::{ltj_run, JoinQuery, Term, TriplePattern};
use crate::common::metrics::{human_bytes, peak_rss_kb, time_it, MemSize};
use crate::common::mk2tree::Mk2Forest;
use crate::common::omnitree::Omnitree;
use crate::common::path::{bfs_parallel, bfs_sequential, AdjacencyBackend};
use crate::common::radixtree::{RadixTree, RadixTreeApi};
use crate::common::rdf_gen::{SplitMix64, Triple};
use crate::common::sorted_neighbors::GraphBackend;
use crate::report::Report;
use std::hint::black_box;

const BUILD_REPS: usize = 3;
const QUERY_REPS: usize = 15;
const NUM_PROBES: usize = 300;
/// `k=8` gives 64-way branching (one `u64` word per node), matching
/// `RadixTree`'s node granularity for a fair rank-mechanism comparison
/// (Mk2Forest's own predicate trees use this same `k` internally).
const IK2_K: u32 = 8;

#[allow(clippy::type_complexity)]
pub fn bench_build_and_size(name: &str, triples: &[Triple], report: &mut Report) -> (Mk2Forest, Ik2Tree, Omnitree, RadixTree, RadixTree) {
    let d = time_it(BUILD_REPS, || {
        black_box(Mk2Forest::build(triples));
    });
    let mk2 = Mk2Forest::build(triples);
    report.push(name, "Mk2Forest", "build_time", format!("{d:?}"));
    report.push(name, "Mk2Forest", "mem_size", human_bytes(mk2.mem_size_bytes()));

    let d = time_it(BUILD_REPS, || {
        black_box(Ik2Tree::build(IK2_K, triples));
    });
    let ik2 = Ik2Tree::build(IK2_K, triples);
    report.push(name, "Ik2Tree", "build_time", format!("{d:?}"));
    report.push(name, "Ik2Tree", "mem_size", human_bytes(ik2.mem_size_bytes()));

    let d = time_it(BUILD_REPS, || {
        black_box(Omnitree::build(triples));
    });
    let omni = Omnitree::build(triples);
    report.push(name, "Omnitree", "build_time", format!("{d:?}"));
    report.push(name, "Omnitree", "mem_size", human_bytes(omni.mem_size_bytes()));

    let d = time_it(BUILD_REPS, || {
        black_box(RadixTree::build(triples, RadixTree::default_split()));
    });
    let radix_static = RadixTree::build(triples, RadixTree::default_split());
    report.push(name, "RadixTree(static)", "build_time", format!("{d:?}"));
    report.push(name, "RadixTree(static)", "mem_size", human_bytes(radix_static.mem_size_bytes()));

    let d = time_it(BUILD_REPS, || {
        black_box(RadixTree::build_dynamic(triples));
    });
    let radix_dynamic = RadixTree::build_dynamic(triples);
    report.push(name, "RadixTree(dynamic)", "build_time", format!("{d:?}"));
    report.push(name, "RadixTree(dynamic)", "mem_size", human_bytes(radix_dynamic.mem_size_bytes()));

    if let Some(rss) = peak_rss_kb() {
        report.push(name, "(process)", "peak_rss_so_far", format!("{rss} KiB"));
    }

    (mk2, ik2, omni, radix_static, radix_dynamic)
}

/// Build-time/size comparison for large (millions-of-triples), realistic-
/// skew datasets, across all eight structures. Unlike
/// [`bench_build_and_size`] (which times `BUILD_REPS` repeated builds plus
/// one more to keep, `BUILD_REPS+1` total -- fine at thousands of triples,
/// prohibitively slow at tens of millions, e.g. Ik2Tree alone takes ~60s
/// for a *single* 20M-triple build), this builds each structure exactly
/// once, records its time/size, and drops it before building the next --
/// bounding peak memory to roughly one large structure at a time instead
/// of holding all eight alive simultaneously, matching how this was
/// originally investigated via ad hoc debug binaries.
/// Build + memory (as before) plus *single-pass* `contains`/row-query
/// timing -- `QUERY_REPS`-style repetition (meant for the small, ~5k-triple
/// scenarios) would be far too slow at 20M-triple scale, so this measures
/// once per structure per scenario instead, over a fixed probe set drawn
/// from the actual triples (so every structure sees identical queries).
/// One structure is built, measured, and dropped before the next begins,
/// keeping peak memory bounded to roughly one structure's own footprint at
/// a time.
pub fn bench_large_scale_build_and_size(name: &str, triples: &[Triple], report: &mut Report) {
    let contains_probes: Vec<Triple> = triples.iter().step_by((triples.len() / 10_000).max(1)).copied().collect();
    let row_probes: Vec<(u32, u32)> = triples.iter().step_by((triples.len() / 2_000).max(1)).map(|&(s, p, _)| (s, p)).collect();

    #[allow(clippy::too_many_arguments)]
    fn once<T: MemSize>(
        name: &str,
        label: &str,
        triples: &[Triple],
        contains_probes: &[Triple],
        row_probes: &[(u32, u32)],
        build: impl FnOnce(&[Triple]) -> T,
        contains: impl Fn(&T, u32, u32, u32) -> bool,
        row: impl Fn(&T, u32, u32) -> usize,
        report: &mut Report,
    ) {
        let t0 = std::time::Instant::now();
        let tree = build(triples);
        let d = t0.elapsed();
        report.push(name, label, "build_time", format!("{d:?}"));
        report.push(name, label, "mem_size", human_bytes(tree.mem_size_bytes()));

        let t0 = std::time::Instant::now();
        let mut hits = 0u64;
        for &(s, p, o) in contains_probes {
            if contains(&tree, s, p, o) {
                hits += 1;
            }
        }
        let d = t0.elapsed();
        report.push(name, label, "contains_avg", format!("{:?} ({hits} hits)", d / contains_probes.len() as u32));

        let t0 = std::time::Instant::now();
        let mut total_out = 0u64;
        for &(s, p) in row_probes {
            total_out += row(&tree, s, p) as u64;
        }
        let d = t0.elapsed();
        report.push(name, label, "row_query_avg", format!("{:?} ({total_out} total_out)", d / row_probes.len() as u32));

        drop(tree);
    }

    once(name, "Mk2Forest", triples, &contains_probes, &row_probes, Mk2Forest::build, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.row(p, s).len(), report);
    once(name, "Ik2Tree", triples, &contains_probes, &row_probes, |t| Ik2Tree::build(IK2_K, t), |t, s, p, o| t.locate(p, s, o), |t, s, p| t.row_bound(p, s).len(), report);
    once(name, "Omnitree", triples, &contains_probes, &row_probes, Omnitree::build, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(static)", triples, &contains_probes, &row_probes, |t| RadixTree::build(t, RadixTree::default_split()), |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dynamic)", triples, &contains_probes, &row_probes, RadixTree::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(veb)", triples, &contains_probes, &row_probes, crate::common::radixtree_veb::RadixTreeVeb::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dfs)", triples, &contains_probes, &row_probes, crate::common::radixtree_dfs::RadixTreeDfs::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dfs_dac)", triples, &contains_probes, &row_probes, crate::common::radixtree_dfs_dac::RadixTreeDfsDac::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(bp)", triples, &contains_probes, &row_probes, crate::common::radixtree_bp::RadixTreeBp::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dfs_bp)", triples, &contains_probes, &row_probes, crate::common::radixtree_dfs_bp::RadixTreeDfsBp::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dfs_interleaved)", triples, &contains_probes, &row_probes, crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);
    once(name, "RadixTree(dfs_interleaved_varskip)", triples, &contains_probes, &row_probes, crate::common::radixtree_dfs_interleaved_varskip::RadixTreeDfsInterleavedVarSkip::build_dynamic, |t, s, p, o| t.contains(s, p, o), |t, s, p| t.query_two_bound([0, 1], [s, p], 2).len(), report);

    if let Some(rss) = peak_rss_kb() {
        report.push(name, "(process)", "peak_rss_so_far", format!("{rss} KiB"));
    }
}

/// Insertion cost -- single and batch -- for every structure benchmarked
/// in [`bench_large_scale_build_and_size`], on the same large-scale
/// datasets, run right after that function so the "full store" it just
/// measured is exactly what these insertions are landing on.
///
/// None of these structures support incremental mutation -- every one is
/// built once from a full triple slice and is otherwise read-only (no
/// `insert` method exists anywhere in `examples/common`). So "inserting"
/// here can only mean what it actually costs today: extend the triple
/// set and rebuild from scratch. That makes `single_insert_avg`
/// numerically close to `build_time` (adding one triple to millions
/// changes essentially nothing about the rebuild) -- not a redundant
/// measurement, but the honest answer to "what does adding one record
/// cost right now," stated explicitly rather than left for the reader to
/// infer from `build_time` alone. `batch_insert_per_triple` is where the
/// real, useful signal is: it shows how much of that fixed per-rebuild
/// cost gets amortized away once insertions are batched instead of
/// applied one at a time.
///
/// Trial/batch counts are deliberately small (`SINGLE_INSERT_TRIALS`,
/// `BATCH_INSERT_SIZE`) -- each is a full rebuild of the whole dataset,
/// and `Ik2Tree` alone costs ~60s/rebuild on the largest scenario, so
/// this starts cheap-and-fast rather than 1%-of-20M-sized until there's
/// a reason to want finer amortization resolution than a 10-triple batch
/// already shows.
pub fn bench_insertion(name: &str, triples: &[Triple], report: &mut Report) {
    const SINGLE_INSERT_TRIALS: usize = 10;
    const BATCH_INSERT_SIZE: usize = 10;

    let (max_s, max_p, max_o) = triples.iter().fold((0u32, 0u32, 0u32), |(ms, mp, mo), &(s, p, o)| (ms.max(s), mp.max(p), mo.max(o)));
    let mut rng = SplitMix64::new(0xC0FFEE ^ triples.len() as u64);
    let new_triple = |rng: &mut SplitMix64| -> Triple { (rng.next_range(max_s as u64 + 1) as u32, rng.next_range(max_p as u64 + 1) as u32, rng.next_range(max_o as u64 + 1) as u32) };

    // Built once, up front, and shared across every structure below --
    // rebuilding these extended triple sets per structure (instead of
    // per trial) would multiply a ~20M-triple `Vec` clone by the number
    // of structures for no reason, since none of the `build` calls
    // mutate their input.
    let with_singles: Vec<Vec<Triple>> = (0..SINGLE_INSERT_TRIALS)
        .map(|_| {
            let mut v = triples.to_vec();
            v.push(new_triple(&mut rng));
            v
        })
        .collect();
    let with_batch: Vec<Triple> = {
        let mut v = triples.to_vec();
        v.extend((0..BATCH_INSERT_SIZE).map(|_| new_triple(&mut rng)));
        v
    };

    // Printed eagerly (not buffered until the structure finishes) so a
    // stall shows up as "started X, never finished" in the terminal
    // instead of the whole scenario going silent for however long that
    // structure's rebuild takes -- `Ik2Tree` alone is ~60s/rebuild on the
    // largest scenario, so without this a stuck run and a merely-slow one
    // look identical from the outside.
    fn once<T>(name: &str, label: &str, with_singles: &[Vec<Triple>], with_batch: &[Triple], batch_size: usize, build: impl Fn(&[Triple]) -> T, report: &mut Report) {
        use std::io::Write;
        print!("    [insertion:{name}] {label}: single inserts ");
        std::io::stdout().flush().ok();
        let mut total_single = std::time::Duration::ZERO;
        for (i, extended) in with_singles.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let tree = build(extended);
            let d = t0.elapsed();
            total_single += d;
            drop(tree);
            print!("{}/{} ", i + 1, with_singles.len());
            std::io::stdout().flush().ok();
        }
        println!("done");
        report.push(name, label, "single_insert_avg", format!("{:?} (full rebuild, {} trials)", total_single / with_singles.len() as u32, with_singles.len()));

        print!("    [insertion:{name}] {label}: batch insert ({batch_size} triples)... ");
        std::io::stdout().flush().ok();
        let t0 = std::time::Instant::now();
        let tree = build(with_batch);
        let batch_elapsed = t0.elapsed();
        drop(tree);
        println!("done ({batch_elapsed:?})");
        report.push(name, label, "batch_insert_total", format!("{batch_elapsed:?} ({batch_size} triples, full rebuild)"));
        report.push(name, label, "batch_insert_per_triple", format!("{:?}", batch_elapsed / batch_size as u32));
    }

    once(name, "Mk2Forest", &with_singles, &with_batch, BATCH_INSERT_SIZE, Mk2Forest::build, report);
    once(name, "Ik2Tree", &with_singles, &with_batch, BATCH_INSERT_SIZE, |t| Ik2Tree::build(IK2_K, t), report);
    once(name, "Omnitree", &with_singles, &with_batch, BATCH_INSERT_SIZE, Omnitree::build, report);
    once(name, "RadixTree(static)", &with_singles, &with_batch, BATCH_INSERT_SIZE, |t| RadixTree::build(t, RadixTree::default_split()), report);
    once(name, "RadixTree(dynamic)", &with_singles, &with_batch, BATCH_INSERT_SIZE, RadixTree::build_dynamic, report);
    once(name, "RadixTree(veb)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_veb::RadixTreeVeb::build_dynamic, report);
    once(name, "RadixTree(dfs)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_dfs::RadixTreeDfs::build_dynamic, report);
    once(name, "RadixTree(dfs_dac)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_dfs_dac::RadixTreeDfsDac::build_dynamic, report);
    once(name, "RadixTree(bp)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_bp::RadixTreeBp::build_dynamic, report);
    once(name, "RadixTree(dfs_bp)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_dfs_bp::RadixTreeDfsBp::build_dynamic, report);
    once(name, "RadixTree(dfs_interleaved)", &with_singles, &with_batch, BATCH_INSERT_SIZE, crate::common::radixtree_dfs_interleaved::RadixTreeDfsInterleaved::build_dynamic, report);
    once(
        name,
        "RadixTree(dfs_interleaved_varskip)",
        &with_singles,
        &with_batch,
        BATCH_INSERT_SIZE,
        crate::common::radixtree_dfs_interleaved_varskip::RadixTreeDfsInterleavedVarSkip::build_dynamic,
        report,
    );

    if let Some(rss) = peak_rss_kb() {
        report.push(name, "(process)", "peak_rss_so_far", format!("{rss} KiB"));
    }
}

/// Compares RadixTree's static `[2,2,2]` split against the dynamic
/// water-filling split (which redistributes an exhausted dimension's
/// unused bit budget to the still-active dimensions each level, so a node
/// keeps using its full 64-bit word for longer). Node-word count is the
/// more informative number here: it isolates the structural effect from
/// `Vec` capacity-rounding noise in the raw byte count.
pub fn bench_radix_split_comparison(name: &str, triples: &[Triple], report: &mut Report) {
    let static_tree = RadixTree::build(triples, RadixTree::default_split());
    let dynamic_tree = RadixTree::build_dynamic(triples);
    report.push(name, "RadixTree(static)", "node_words", static_tree.node_word_count().to_string());
    report.push(name, "RadixTree(static)", "mem_size", human_bytes(static_tree.mem_size_bytes()));
    report.push(name, "RadixTree(dynamic)", "node_words", dynamic_tree.node_word_count().to_string());
    report.push(name, "RadixTree(dynamic)", "mem_size", human_bytes(dynamic_tree.mem_size_bytes()));
}

fn random_probes(triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, seed: u64) -> Vec<Triple> {
    let mut rng = SplitMix64::new(seed);
    (0..NUM_PROBES)
        .map(|_| {
            if rng.next_f64() < 0.5 && !triples.is_empty() {
                triples[rng.next_range(triples.len() as u64) as usize]
            } else {
                (rng.next_range(num_s as u64) as u32, rng.next_range(num_p as u64) as u32, rng.next_range(num_o as u64) as u32)
            }
        })
        .collect()
}

pub fn bench_point_queries(name: &str, triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, mk2: &Mk2Forest, ik2: &Ik2Tree, omni: &Omnitree, report: &mut Report) {
    let probes = random_probes(triples, num_s, num_p, num_o, 111);

    let d = time_it(QUERY_REPS, || {
        for &(s, p, o) in &probes {
            black_box(mk2.contains(s, p, o));
        }
    });
    report.push(name, "Mk2Forest", "point_query_avg", format!("{:?}", d / probes.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &(s, p, o) in &probes {
            black_box(ik2.locate(p, s, o));
        }
    });
    report.push(name, "Ik2Tree", "point_query_avg", format!("{:?}", d / probes.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &(s, p, o) in &probes {
            black_box(omni.contains(s, p, o));
        }
    });
    report.push(name, "Omnitree", "point_query_avg", format!("{:?}", d / probes.len() as u32));
}

pub fn bench_point_queries_radix(name: &str, triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, radix: &RadixTree, radix_label: &str, report: &mut Report) {
    let probes = random_probes(triples, num_s, num_p, num_o, 111);
    let d = time_it(QUERY_REPS, || {
        for &(s, p, o) in &probes {
            black_box(radix.contains(s, p, o));
        }
    });
    report.push(name, radix_label, "point_query_avg", format!("{:?}", d / probes.len() as u32));
}

/// The literature's actual claim (Álvarez-García et al. / the DCC'14
/// paper's own benchmarks) is that Mk2Forest wins fixed-predicate queries
/// because its per-predicate tree is a plain k²-tree descent with no
/// compaction bookkeeping, while IK2Tree pays extra `rank1`/`select1` calls
/// per level (tracking local_pred_pos/width/offset) even when the
/// predicate is bound. [`bench_point_queries`]/[`bench_row_col_queries`]
/// don't isolate that cleanly: they interleave queries across *all*
/// predicates, so Mk2Forest also pays a cache-locality penalty (25 small
/// trees scattered across the heap, jumped between on every query) that
/// has nothing to do with the algorithmic cost the literature is
/// describing. This benchmark repeats queries against a *single* fixed
/// predicate, so Mk2Forest's one relevant tree stays hot in cache --
/// matching how the literature's own benchmarks isolate this comparison.
pub fn bench_fixed_predicate_query(name: &str, triples: &[Triple], fixed_p: u32, mk2: &Mk2Forest, ik2: &Ik2Tree, report: &mut Report) {
    let subjects: Vec<u32> = triples.iter().filter(|&&(_, p, _)| p == fixed_p).map(|&(s, _, _)| s).collect();
    if subjects.is_empty() {
        return;
    }
    let objects: Vec<u32> = triples.iter().filter(|&&(_, p, _)| p == fixed_p).map(|&(_, _, o)| o).collect();

    let d = time_it(QUERY_REPS, || {
        for (&s, &o) in subjects.iter().zip(&objects) {
            black_box(mk2.contains(s, fixed_p, o));
        }
    });
    report.push(name, "Mk2Forest", "fixed_pred_point_avg", format!("{:?}", d / subjects.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for (&s, &o) in subjects.iter().zip(&objects) {
            black_box(ik2.locate(fixed_p, s, o));
        }
    });
    report.push(name, "Ik2Tree", "fixed_pred_point_avg", format!("{:?}", d / subjects.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(mk2.row(fixed_p, s));
        }
    });
    report.push(name, "Mk2Forest", "fixed_pred_row_avg", format!("{:?}", d / subjects.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(ik2.row_bound(fixed_p, s));
        }
    });
    report.push(name, "Ik2Tree", "fixed_pred_row_avg", format!("{:?}", d / subjects.len() as u32));
}

/// RadixTree's analogue of [`bench_fixed_predicate_query`]: unlike
/// IK2Tree, predicate here is just more bits packed into the same word as
/// S,O -- `contains`/`query_two_bound` never do active-list compaction
/// bookkeeping, bound or not, so there's no reason to expect the same
/// "extra overhead even when bound" cost IK2Tree pays.
pub fn bench_fixed_predicate_query_radix(name: &str, triples: &[Triple], fixed_p: u32, radix: &RadixTree, radix_label: &str, report: &mut Report) {
    let subjects: Vec<u32> = triples.iter().filter(|&&(_, p, _)| p == fixed_p).map(|&(s, _, _)| s).collect();
    if subjects.is_empty() {
        return;
    }
    let objects: Vec<u32> = triples.iter().filter(|&&(_, p, _)| p == fixed_p).map(|&(_, _, o)| o).collect();

    let d = time_it(QUERY_REPS, || {
        for (&s, &o) in subjects.iter().zip(&objects) {
            black_box(radix.contains(s, fixed_p, o));
        }
    });
    report.push(name, radix_label, "fixed_pred_point_avg", format!("{:?}", d / subjects.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(radix.query_two_bound([0, 1], [s, fixed_p], 2));
        }
    });
    report.push(name, radix_label, "fixed_pred_row_avg", format!("{:?}", d / subjects.len() as u32));
}

pub fn bench_row_col_queries(name: &str, triples: &[Triple], mk2: &Mk2Forest, ik2: &Ik2Tree, omni: &Omnitree, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();

    let d = time_it(QUERY_REPS, || {
        for &(s, p, _) in &sample {
            black_box(mk2.row(p, s));
        }
    });
    report.push(name, "Mk2Forest", "row_query_avg", format!("{:?}", d / sample.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &(s, p, _) in &sample {
            black_box(ik2.row_bound(p, s));
        }
    });
    report.push(name, "Ik2Tree", "row_query_avg", format!("{:?}", d / sample.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &(s, p, _) in &sample {
            black_box(omni.query_two_bound([0, 1], [s, p], 2));
        }
    });
    report.push(name, "Omnitree", "row_query_avg", format!("{:?}", d / sample.len() as u32));
}

pub fn bench_row_col_queries_radix(name: &str, triples: &[Triple], radix: &RadixTree, radix_label: &str, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();
    let d = time_it(QUERY_REPS, || {
        for &(s, p, _) in &sample {
            black_box(radix.query_two_bound([0, 1], [s, p], 2));
        }
    });
    report.push(name, radix_label, "row_query_avg", format!("{:?}", d / sample.len() as u32));
}

/// Unbound-predicate query comparison: Mk2Forest's sequential vs.
/// rayon-parallel per-predicate-tree scan; IK2Tree's eager vs. lazy
/// predicate decode; Omnitree's one-bound-dimension scan.
pub fn bench_unbound_predicate(name: &str, num_s: u32, mk2: &Mk2Forest, ik2: &Ik2Tree, omni: &Omnitree, report: &mut Report) {
    let mut rng = SplitMix64::new(222);
    let subjects: Vec<u32> = (0..50).map(|_| rng.next_range(num_s as u64) as u32).collect();

    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(mk2.scan_all_seq(Some(s), None));
        }
    });
    report.push(name, "Mk2Forest", "unbound_pred_seq_avg", format!("{:?}", d / subjects.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(mk2.scan_all_parallel(Some(s), None));
        }
    });
    report.push(name, "Mk2Forest", "unbound_pred_parallel_avg", format!("{:?}", d / subjects.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(ik2.row_unbound_eager(s));
        }
    });
    report.push(name, "Ik2Tree", "unbound_pred_eager_avg", format!("{:?}", d / subjects.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(ik2.row_unbound_lazy(s));
        }
    });
    report.push(name, "Ik2Tree", "unbound_pred_lazy_avg", format!("{:?}", d / subjects.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(omni.objects_from_subject(s));
        }
    });
    report.push(name, "Omnitree", "unbound_pred_scan_avg", format!("{:?}", d / subjects.len() as u32));
}

pub fn bench_unbound_predicate_radix(name: &str, num_s: u32, radix: &RadixTree, radix_label: &str, report: &mut Report) {
    let mut rng = SplitMix64::new(222);
    let subjects: Vec<u32> = (0..50).map(|_| rng.next_range(num_s as u64) as u32).collect();
    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(radix.query_one_bound(0, s, 2));
        }
    });
    report.push(name, radix_label, "unbound_pred_scan_avg", format!("{:?}", d / subjects.len() as u32));
}

/// "What predicate(s) connect this specific subject and object?" -- binds
/// S and O, leaves P free. RadixTree and Omnitree answer this natively
/// (genuinely 3D-symmetric: same query_two_bound machinery as row/col, just
/// with a different pair of dimensions bound). Mk2Forest has no such axis
/// at all -- predicate selects *which* independent tree, not a third
/// traversal dimension -- so it's forced into an O(num_predicates) scan
/// across every tree. IK2Tree sits in between: its single unified tree
/// already descends on (S,O) alone, so once both are bound there's exactly
/// one path down with no branching, and reusing the lazy select1-based
/// decode gives an O(height + hits) answer -- no O(num_predicates) scan
/// needed despite predicate not being a first-class traversal axis there
/// either.
pub fn bench_predicate_lookup(name: &str, triples: &[Triple], mk2: &Mk2Forest, ik2: &Ik2Tree, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();

    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(mk2.predicates_for_seq(s, o));
        }
    });
    report.push(name, "Mk2Forest", "predicate_lookup_seq_avg", format!("{:?}", d / sample.len() as u32));
    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(mk2.predicates_for_parallel(s, o));
        }
    });
    report.push(name, "Mk2Forest", "predicate_lookup_parallel_avg", format!("{:?}", d / sample.len() as u32));

    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(ik2.predicates_for(s, o));
        }
    });
    report.push(name, "Ik2Tree", "predicate_lookup_avg", format!("{:?}", d / sample.len() as u32));
}

pub fn bench_predicate_lookup_radix(name: &str, triples: &[Triple], radix: &RadixTree, radix_label: &str, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();
    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(radix.predicates_for(s, o));
        }
    });
    report.push(name, radix_label, "predicate_lookup_avg", format!("{:?}", d / sample.len() as u32));
}

pub fn bench_predicate_lookup_omni(name: &str, triples: &[Triple], omni: &Omnitree, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();
    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(omni.predicates_for(s, o));
        }
    });
    report.push(name, "Omnitree", "predicate_lookup_avg", format!("{:?}", d / sample.len() as u32));
}

/// Sequential vs. level-synchronous parallel BFS from a fixed start node,
/// across all four backends. Meant to be called on the chain-shaped
/// dataset, where reachability is non-trivial.
pub fn bench_bfs(name: &str, mk2: &Mk2Forest, ik2: &Ik2Tree, omni: &Omnitree, report: &mut Report) {
    let start = 0u32;
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_sequential(mk2, start, None));
    });
    report.push(name, "Mk2Forest", "bfs_seq", format!("{d:?}"));
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_parallel(mk2, start, None));
    });
    report.push(name, "Mk2Forest", "bfs_parallel", format!("{d:?}"));

    let d = time_it(QUERY_REPS, || {
        black_box(bfs_sequential(ik2, start, None));
    });
    report.push(name, "Ik2Tree", "bfs_seq", format!("{d:?}"));
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_parallel(ik2, start, None));
    });
    report.push(name, "Ik2Tree", "bfs_parallel", format!("{d:?}"));

    let d = time_it(QUERY_REPS, || {
        black_box(bfs_sequential(omni, start, None));
    });
    report.push(name, "Omnitree", "bfs_seq", format!("{d:?}"));
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_parallel(omni, start, None));
    });
    report.push(name, "Omnitree", "bfs_parallel", format!("{d:?}"));
}

pub fn bench_bfs_radix(name: &str, radix: &RadixTree, radix_label: &str, report: &mut Report) {
    let start = 0u32;
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_sequential(radix, start, None));
    });
    report.push(name, radix_label, "bfs_seq", format!("{d:?}"));
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_parallel(radix, start, None));
    });
    report.push(name, radix_label, "bfs_parallel", format!("{d:?}"));
}

/// Runs the same worst-case-optimal join (2-hop path, triangle, star)
/// unmodified over all four backends. Meant to be called on the
/// star/triangle-shaped dataset.
fn join_queries() -> [(&'static str, JoinQuery); 3] {
    let two_hop = JoinQuery {
        num_vars: 2,
        patterns: vec![
            TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
            TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
        ],
    };
    let triangle = JoinQuery {
        num_vars: 2,
        patterns: vec![
            TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) },
            TriplePattern { subject: Term::Var(0), predicate: 1, object: Term::Var(1) },
            TriplePattern { subject: Term::Var(1), predicate: 2, object: Term::Const(0) },
        ],
    };
    let star = JoinQuery {
        num_vars: 1,
        patterns: vec![TriplePattern { subject: Term::Const(0), predicate: 0, object: Term::Var(0) }],
    };
    [("join_2hop", two_hop), ("join_triangle", triangle), ("join_star", star)]
}

pub fn bench_join(name: &str, mk2: &Mk2Forest, ik2: &Ik2Tree, omni: &Omnitree, report: &mut Report) {
    for (qname, query) in &join_queries() {
        let d = time_it(QUERY_REPS, || {
            black_box(ltj_run(query, mk2));
        });
        let matches = ltj_run(query, mk2).len();
        report.push(name, "Mk2Forest", qname, format!("{d:?} ({matches} matches)"));

        let d = time_it(QUERY_REPS, || {
            black_box(ltj_run(query, ik2));
        });
        let matches = ltj_run(query, ik2).len();
        report.push(name, "Ik2Tree", qname, format!("{d:?} ({matches} matches)"));

        let d = time_it(QUERY_REPS, || {
            black_box(ltj_run(query, omni));
        });
        let matches = ltj_run(query, omni).len();
        report.push(name, "Omnitree", qname, format!("{d:?} ({matches} matches)"));
    }
}

pub fn bench_join_radix(name: &str, radix: &RadixTree, radix_label: &str, report: &mut Report) {
    for (qname, query) in &join_queries() {
        let d = time_it(QUERY_REPS, || {
            black_box(ltj_run(query, radix));
        });
        let matches = ltj_run(query, radix).len();
        report.push(name, radix_label, qname, format!("{d:?} ({matches} matches)"));
    }
}

// --- Generic variants, run against radixtree_veb/radixtree_dfs/radixtree_bp
// (the layout/encoding-only variants investigated per the topological/
// geometric literature pass) -- written once against `RadixTreeApi` +
// `MemSize`/`AdjacencyBackend`/`GraphBackend` instead of duplicated three
// times over, since all three variants deliberately keep the exact same
// public API as the baseline (only internal storage/navigation differs).
// The baseline's own `_radix` functions above stay untouched/unrisked;
// these are purely additive.

pub fn bench_build_and_size_variant<T: MemSize>(name: &str, label: &str, triples: &[Triple], build: impl Fn(&[Triple]) -> T, report: &mut Report) -> T {
    let d = time_it(BUILD_REPS, || {
        black_box(build(triples));
    });
    let tree = build(triples);
    report.push(name, label, "build_time", format!("{d:?}"));
    report.push(name, label, "mem_size", human_bytes(tree.mem_size_bytes()));
    tree
}

pub fn bench_point_queries_variant<T: RadixTreeApi>(name: &str, triples: &[Triple], num_s: u32, num_p: u32, num_o: u32, tree: &T, label: &str, report: &mut Report) {
    let probes = random_probes(triples, num_s, num_p, num_o, 111);
    let d = time_it(QUERY_REPS, || {
        for &(s, p, o) in &probes {
            black_box(tree.contains(s, p, o));
        }
    });
    report.push(name, label, "point_query_avg", format!("{:?}", d / probes.len() as u32));
}

pub fn bench_row_col_queries_variant<T: RadixTreeApi>(name: &str, triples: &[Triple], tree: &T, label: &str, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();
    let d = time_it(QUERY_REPS, || {
        for &(s, p, _) in &sample {
            black_box(tree.query_two_bound([0, 1], [s, p], 2));
        }
    });
    report.push(name, label, "row_query_avg", format!("{:?}", d / sample.len() as u32));
}

pub fn bench_predicate_lookup_variant<T: RadixTreeApi>(name: &str, triples: &[Triple], tree: &T, label: &str, report: &mut Report) {
    if triples.is_empty() {
        return;
    }
    let sample: Vec<Triple> = triples.iter().step_by((triples.len() / 200).max(1)).copied().collect();
    let d = time_it(QUERY_REPS, || {
        for &(s, _, o) in &sample {
            black_box(tree.predicates_for(s, o));
        }
    });
    report.push(name, label, "predicate_lookup_avg", format!("{:?}", d / sample.len() as u32));
}

pub fn bench_unbound_predicate_variant<T: RadixTreeApi>(name: &str, num_s: u32, tree: &T, label: &str, report: &mut Report) {
    let mut rng = SplitMix64::new(222);
    let subjects: Vec<u32> = (0..50).map(|_| rng.next_range(num_s as u64) as u32).collect();
    let d = time_it(QUERY_REPS, || {
        for &s in &subjects {
            black_box(tree.query_one_bound(0, s, 2));
        }
    });
    report.push(name, label, "unbound_pred_scan_avg", format!("{:?}", d / subjects.len() as u32));
}

pub fn bench_bfs_variant<T: AdjacencyBackend + Sync>(name: &str, tree: &T, label: &str, report: &mut Report) {
    let start = 0u32;
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_sequential(tree, start, None));
    });
    report.push(name, label, "bfs_seq", format!("{d:?}"));
    let d = time_it(QUERY_REPS, || {
        black_box(bfs_parallel(tree, start, None));
    });
    report.push(name, label, "bfs_parallel", format!("{d:?}"));
}

pub fn bench_join_variant<T: GraphBackend>(name: &str, tree: &T, label: &str, report: &mut Report) {
    for (qname, query) in &join_queries() {
        let d = time_it(QUERY_REPS, || {
            black_box(ltj_run(query, tree));
        });
        let matches = ltj_run(query, tree).len();
        report.push(name, label, qname, format!("{d:?} ({matches} matches)"));
    }
}
