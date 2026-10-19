# FST representation and traversal: literature review

Review date: 2026-10-19. Written during the `PathFstDeltaPow2`/`PathFstDeltaPow2Bfs`
restricted-descent investigation (`DESIGN.md`'s "BGP-style queries on the FST"
entries and everything under them). Scoped to the FST/trie *representation* and
*traversal* literature -- physical layout, deduplication, succinct encodings,
and batched/vectorized descent. Join and multi-dimensional-partitioning
literature (kⁿ-trees, Omnitrees, WCO joins) is tracked separately in
[`TOPOLOGY_REVIEW.md`](TOPOLOGY_REVIEW.md) and is not repeated here except
where it bears directly on layout.

The headline finding: this project's own empirical arc this session --
post-order (DFS-discovery) layout with hash-based dedup, versus a BFS/level-order
layout with the same dedup, versus window-batched traversal on top of either --
independently rediscovered a tension the k²-tree literature already has a name
for, from the opposite direction. The rest of this review maps specific papers
to specific pieces of `src/path_fst.rs`.

## Literature mapped to the implementation

| Work | Relevant idea | Current implementation / status |
|---|---|---|
| Brisaboa, Ladra & Navarro, *k²-trees for Compact Web Graph Representation* (SPIRE 2009) | Recursive k²-ary partitioning of a sparse binary relation; one bit per node, height `⌈log_k n⌉`. | `PathFst*`'s `K_SQUARE`-branching Morton partitioning is the 3-dimensional analogue (`compute_qi`/`dynamic_schedule`, `src/tree.rs`). `TOPOLOGY_REVIEW.md` already tracks the formal kⁿ generalization (de Bernardo's thesis) for the join side; not duplicated here. |
| Álvarez-García et al., *k2-triples* (per-predicate k²-tree RDF compression) | Vertically partition a triple store by predicate into one sparse S×O bit-matrix per predicate, each a separate k²-tree; reports 5-7 orders of magnitude speedup over naive representations on triple-pattern queries. | Not implemented or benchmarked against. `PathFst` interleaves S/P/O bits in one structure rather than partitioning by P first. No direct comparison has been run against this as an external reference point, only against `Heap`. |
| Carmona & Manzini, *Depth First Representations of k²-trees* (arXiv:2505.11302, SPIRE 2025) | The standard (level-order/LOUDS-style) k²-tree layout "prevents compressing identical subtrees and is not cache-friendly when accessing individual subtrees"; their fix is a depth-first (BP) layout plus a linear-time suffix-array/LCP-based method to detect and dedup identical subtrees within it. | Already cited in `TOPOLOGY_REVIEW.md` re: structural sharing. This session rediscovered the *same* tension from the opposite starting point: `PathFstDeltaPow2` (post-order/DFS-discovery layout, hash-register dedup) trades locality for easy dedup; `PathFstDeltaPow2Bfs` (level-order/LOUDS-style layout, same dedup mechanism) trades some of that dedup-friendliness back for level-batched locality. The actual measured trade (not assumed) is in `DESIGN.md`'s "pairing a BFS-ordered layout with the BFS algorithm" entry. |
| Arroyuelo, Cánovas, Navarro, Sadakane, *Succinct Trees in Practice* (ALENEX 2010) | Controlled comparison of LOUDS (breadth-first) vs. DFUDS/BP (depth-first) succinct tree encodings for general ordinal-tree navigation. Conclusion: BP/DFUDS win on space, time, *and* functionality in general; LOUDS "supports only a very limited set of operations" and is a niche choice. | Appears to contradict this project's own measured result (the BFS/LOUDS-style layout won for the window-batched traversal) -- but their comparison is about general-purpose `parent`/`child`/`sibling`/`subtree-size` navigation, not this project's specific access pattern (one precomputed mask per level, visited in width-batches). Flagged as a scope caveat on reusing their conclusion, not a refutation of this project's measurement. |
| Zhang, Zhao, Xu, *C²: Cache-Conscious Succinct Tries with Adaptive Unary Path Compression* (arXiv:2606.16104, 2026 preprint) | Two independent, combinable levers on a succinct trie: `C₁`, a more cache-friendly bitvector layout (reports 1.12-1.58x speedup across FST/CoCo-trie/Marisa baselines), and `C₂`, compressing long unary (single-child) chains (~1.3x smaller). | Not implemented. The closest unexplored lever specifically for the plain-DFS traversal path (`restricted_descent_visit`) -- `C₁`'s bitvector-layout tuning targets exactly that traversal shape, not the batched ones already explored here. |
| Leis, Kemper, Neumann, *The Adaptive Radix Tree* (ART, ICDE 2013) | Give up succinctness entirely; adapt each node's physical layout to its local fanout (4 node sizes), plus path compression and lazy expansion, to minimize cache misses per lookup. | Not directly applicable -- this project's entire point is staying smaller than `Heap`, the opposite trade ART makes. Kept here as the "spend space, buy cache behavior" reference point on the other end of the spectrum from everything else in this table. |
| Bender, Demaine, Farach-Colton; Prokop (cache-oblivious / van Emde Boas layout) | Recursive top-tree/bottom-trees split lays out a tree so *some* subtree fits every cache level at every scale, with no tuning parameters (`O(log_B N)` memory transfers, matching the lower bound). | Does not transfer directly: vEB's recursive split assumes one parent per node. This project's DAG dedup means a node can have many real parents, which breaks the premise that one contiguous address range "belongs" to one parent's subtree -- reasoned through live in-session when the question of a vEB-style relayout came up, not separately re-derived from the paper. |
| Beamer, Asanović, Patterson, *Direction-Optimizing Breadth-First Search* (SC12) | Level-synchronous frontier processing, switching between top-down and bottom-up traversal based on the frontier's current size (adopted by 32 of the top 37 Graph500 entries at the time). | `restricted_descent_frontier_visit`/`restricted_descent_bfs_visit` (both `PathFst*` structs) implement the frontier-batching half of this. The size-based *direction switch* -- falling back to plain DFS when the live frontier is narrow, since that's exactly where the window-batched variants showed a `small`/`zipf` regression -- is not implemented. |
| FlashTrie (Microsoft/NVIDIA, arXiv:2607.10044, 2026 preprint) | GPU-resident LOUDS-encoded trie traversed with a warp-cooperative bit-scan: 32 lanes check 32 words of the LOUDS bitvector at once (`__ballot_sync`) instead of one lane scanning sequentially and leaving the other 31 idle. 24x speedup over a tuned CPU baseline at scale. | Not implemented (no GPU path in this project). Cited as confirmation that `restricted_descent_bfs_visit`'s window-batching -- "many frontier nodes' cheap, latency-bound bit-scans are worth doing together" -- is current, real practice elsewhere, not a local curiosity. |
| Wang et al., Hyperscan/Teddy (NSDI 2019); simdjson | SIMD filter-then-confirm: vectorize a cheap filtering pass over many independent lanes (bytes of input), then run a slower confirm/decode pass only on survivors. | `restricted_descent_bfs_masked_visit` tried the identical shape over *nodes* instead of *bytes* -- confirmed via `objdump` to actually vectorize (SSE2 `pand`), but measured as a net loss (`DESIGN.md`). The likely reason: Teddy/simdjson batch thousands of independent byte-lanes per call; this project's window batches at most 256 nodes, usually far fewer, with the AND never having been the bottleneck (flat cache-miss count, confirmed by `perf`). Same technique, two orders of magnitude different batch density. |
| Fernández, Martínez-Prieto, Gutierrez, HDT; WaterFowl | Succinct dictionary + bitmap-triples encoding for a *complete* RDF store (strings included), not just the S/P/O index. | Out of scope for `PathFst` alone -- it doesn't own a string dictionary, only the integer-ID index. Relevant only if this project ever grows an end-to-end storage layer rather than staying an index structure. |
| Veldhuizen, *Leapfrog Triejoin* (ICDT 2014); Arroyuelo, Navarro, Hogan et al., *The Ring* (ACM TODS 2024) | Trie-based worst-case-optimal multi-way join (repeated seek-to-max across sorted trie iterators); succinct generalization needing one wavelet-tree-based structure instead of one index per attribute order. | Belongs with `native_join`/`TOPOLOGY_REVIEW.md`'s WCO-join thread, not this traversal-layout thread. Cross-referenced here, not duplicated. |

## Evidence in the code

- `src/path_fst.rs`'s `PathFstDeltaPow2` (`build_rec_pow2`, post-order/DFS-discovery
  address assignment) and `PathFstDeltaPow2Bfs` (`collect_rec_bfs` + the two-phase
  top-down address/delta assignment in `build`) are the two layout families the
  Carmona & Manzini and Arroyuelo-et-al. rows above describe, built and measured
  independently in this project rather than assumed correct from the papers.
- `restricted_descent_frontier_visit` and `restricted_descent_bfs_visit` (present
  on both structs) are the frontier-batching half of the direction-optimizing-BFS
  row; there is no frontier-size-based dispatch between them and the plain
  `restricted_descent_visit` DFS yet.
- `restricted_descent_bfs_masked_visit` (`PathFstDeltaPow2Bfs` only) is the
  SIMD-filter experiment from the Teddy/simdjson row, kept as a documented
  negative result per this project's convention of not deleting measured
  prototypes.
- `K_SQUARE`, `dynamic_schedule`, `compute_qi` (`src/tree.rs`) are this project's
  k²-tree generalization; see `TOPOLOGY_REVIEW.md` for the formal kⁿ-generalization
  angle (de Bernardo), tracked there rather than here to avoid duplication.
- The sibling repository `../s-dict` (`PackedAutomaton`, referenced in `DESIGN.md`'s
  "Path FST prototype" entry) is independent local prior art for the back-reference/delta
  record format `PathFst*` uses -- not literature, but worth knowing it predates this
  session's work and already worked through the mutation-cost problem for a
  register-deduplicated automaton.

## Possible next steps, not yet started

- Benchmark against k2-triples' per-predicate-partitioned approach directly, as
  an external reference point -- everything measured so far is relative to
  `Heap`, not to a published alternative design for the same problem.
- Try C²'s `C₁` cache-layout tuning on the plain-DFS traversal path specifically;
  its reported 1.12-1.58x applies most directly to exactly that traversal shape,
  not the batched ones already explored.
- Implement the frontier-size-based direction dispatch direction-optimizing BFS
  describes, to address the `small`/`zipf` regression already measured on every
  window-batched variant (`restricted_descent_bfs_visit`,
  `restricted_descent_bfs_masked_visit`) without giving up their large-domain wins.
- Read Carmona & Manzini's actual suffix-array/LCP subtree-detection algorithm in
  full; this project's register/hashmap dedup is a different, simpler mechanism
  that may not generalize as well at larger scales or under incremental updates.

Sources:
- [k²-trees for Compact Web Graph Representation (SPIRE09)](https://users.dcc.uchile.cl/%7Egnavarro/ps/spire09.1.pdf)
- [k²-trees survey](https://users.dcc.uchile.cl/~gnavarro/abstracts/is17.html)
- [Compressed k²-Triples for Full-In-Memory RDF Engines](https://arxiv.org/pdf/1105.4004)
- [Depth First Representations of k²-trees (Carmona & Manzini, 2025)](https://arxiv.org/pdf/2505.11302)
- [Extended Depth-First Representations of k²-trees](https://arxiv.org/pdf/2607.28136)
- [Succinct Trees in Practice (Arroyuelo et al., ALENEX 2010)](https://users.dcc.uchile.cl/~gnavarro/ps/alenex10.pdf)
- [C²: Cache-Conscious Succinct Tries with Adaptive Unary Path Compression](https://arxiv.org/pdf/2606.16104)
- [The Adaptive Radix Tree (ART)](https://db.in.tum.de/~leis/papers/ART.pdf)
- [Cache-Oblivious B-trees (Bender, Demaine, Farach-Colton)](https://www.cs.au.dk/~gerth/papers/soda02.pdf)
- [Direction-Optimizing Breadth-First Search (SC12)](https://sc12.supercomputing.org/schedule/event_detail.php-evid=pap525.html)
- [FlashTrie: A GPU-Accelerated Constrained Beam Search](https://arxiv.org/pdf/2607.10044)
- [Hyperscan: A Fast Multi-pattern Regex Matcher (NSDI 2019)](https://www.usenix.org/conference/nsdi19/presentation/wang-xiang)
- [Compact Representation of Large RDF Data Sets (HDT)](https://files.ifi.uzh.ch/ddis/iswc_archive/iswc/ab/2010/iswc2010.semanticweb.org/accepted-papers/61.html)
- [Leapfrog Triejoin: A Simple, Worst-Case Optimal Join Algorithm](https://arxiv.org/pdf/1210.0481)
- [The Ring: Worst-Case Optimal Joins Using (Almost) No Extra Space](https://users.dcc.uchile.cl/~gnavarro/ps/tods24.pdf)
