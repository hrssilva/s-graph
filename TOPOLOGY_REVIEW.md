# Topology and graph execution: literature review

Review date: 2026-09-17. Updated after implementing `native_join::NativeJoin`.
This distinguishes the implemented prefix executor from further research work.

The storage uses predicates as a full geometric coordinate. The materialized
prepared engine has been removed. The native executor keeps predicates structural
throughout the join, using precompiled prefix checks and DFS child cursors. Batched frontier
traversal for paths remains future work.

## Literature mapped to the implementation

| Work | Relevant idea | Current implementation |
|---|---|---|
| de Bernardo, doctoral thesis (2014), Chapter 6 | Generalize k² partitioning to kⁿ; constrain any combination of coordinates; clustering and branching factor govern compression. | Three physical coordinates with globally scheduled, possibly unequal bit widths. Not an arbitrary-dimensional public tree. |
| Arroyuelo et al., *Faster Dynamic Compressed d-ary Relations* | View the tree as a trie of Morton codes, permitting different dynamic representations from a rank/select bitvector. | Consistent with the packed DFS radix representation. The paper's operation bounds do not automatically apply to our block directory and vector splits. |
| Carmona and Manzini, *Depth first representations of k²-trees* | Contiguous subtree layouts, navigation enrichment and compression of identical subtrees. | DFS layout and subtree spans are present. Structural sharing of repeated subtrees is absent. |
| Pollinger et al., *Omnitrees* | Choose which dimensions to subdivide at individual nodes. | The schedule depends on global coordinate widths and level, not local occupancy. |
| Navarro et al., *Optimal Joins Using Compact Data Structures* | Virtual dimension lifting and synchronized intersection of compressed relations. | Native prefix compatibility is implemented; a Qdag-equivalent traversal and WCO analysis are not. |

Sources: [doctoral thesis, §§6.1–6.3](https://users.dcc.uchile.cl/~gnavarro/mem/tesisGuillermo.pdf),
[Morton trie paper](https://arxiv.org/abs/1911.08971),
[DFS paper](https://arxiv.org/abs/2505.11302),
[Omnitree paper](https://arxiv.org/abs/2508.06316),
[compressed join paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICDT.2020.21).

The thesis distinguishes kⁿ-trees from both per-value collections of k²-trees and
interleaved k²-trees. The latter already indexes all three coordinates, but places
one dimension in variable-length node bitmaps. Our predicate bits instead
participate in child selection alongside subject and object bits. Neither
approach is universally superior: locality in the supplied IDs matters.

## Evidence in the code

- `tree.rs::compute_qi` combines coordinate digits; `bound_mask` restricts children
  across all coordinates. A fixed predicate prunes cells during navigation;
  `(s, ?p, o)` is answered within the same tree.
- `tree.rs::dynamic_schedule` spends at most six coordinate bits per level and
  stops splitting globally exhausted dimensions. It does not choose a different
  split for each subtree.
- `native_join.rs::NativeJoin::new` compiles prefix masks from schedule metadata.
  Execution uses tree child cursors directly, without copied input relations.
  `graph.rs` collection helpers also route through this executor.
- `traversal.rs::Edges` starts a pattern scan for each expanded vertex.
  `automaton.rs` also scans separately for applicable transitions. These paths
  prune within each scan but do not share a traversal across a whole frontier.

## Implemented execution architecture and remaining work

Internal child cursors provide occupancy, child navigation and subtree skipping.
Each atom maps S/P/O coordinate bits to query variables, constants and equality
constraints. Flat cursor arrays, reusable stacks and word-sized masks support the
hot loops; no public general-purpose geometric cursor API was needed.

For `T(x,p,y) AND T(y,p,z)`, both atom views belong to a virtual `(x,p,y,z)`
space. Refine their shared coordinate prefixes together, rejecting a region when
either view is empty. Query dimensionality can exceed three without physically
rebuilding the stored ternary relation into an n-dimensional tree.

Alignment should exploit the globally fixed schedule. For each depth and atom,
precompute how physical child bits map to query-variable bit positions, including
leading zeros implied by smaller domains. The hot loop compares prefixes using
these descriptors; it does not discover subdivision rules at each node.

S and O may have different domain widths, P may stop splitting earlier, and a
variable may occur in different coordinate roles. Compare the bits constrained
by both prefixes; disagreement rejects the pair, while agreement leaves a
candidate for further refinement. Direct occupancy-word intersection applies
when the slots denote the same logical cells. Other mappings require compiled
compatibility masks or grouping. Preserve the joint coordinate constraints of
each atom: independent per-variable projections alone lose correlations. No
general per-node Omnitree alignment machinery is required for the current format.

The revised journal treatment establishes compressed-quadtree WCO joins, but
does not establish WCO evaluation for unrestricted relational algebra. Transferring
join bounds to our mixed-radix cursors requires a separate analysis of navigation
and refinement costs. See [Arroyuelo et al., TODS 2022](https://doi.org/10.1145/3514231).

Avoid unbounded lookup tables or child masks exponential in query variables.
Rivas's 2026 master's thesis studies higher-arity Qdags and generalized hypertree
decompositions; these are relevant alternatives to benchmark, with their own
space and runtime costs. See [*Enhancing Qdags for Higher-Dimensional Queries*](https://users.dcc.uchile.cl/~gnavarro/mem/tesisMatilde.pdf).

For endpoint and automaton reachability, investigate joint frontier/tree
traversals and shared predicate filtering. Preserve hop layers and automaton
states; a shorter visit before the minimum hop count cannot suppress a later
qualifying walk. Trail/simple-path enumeration requires its additional history.

The native executor is the sole production join implementation and preserves
the physical format. No proven WCO executor is currently provided. Local Omnitree splits or
repeated-subtree sharing would add metadata and update complexity and need
evidence of a net benefit.

## Acceptance measurements

Compare end-to-end cold queries, repeated queries, first-result and total latency,
peak temporary memory, bytes read and tree nodes visited. Include variable
predicates, cyclic and disconnected patterns, unequal domains, repeated variables,
clustered and shuffled IDs, predicate skew and increasing query dimensionality.
Differential correctness against exhaustive small relations and an independent
test-only triple enumerator is covered by `tests/native_join.rs`. The algorithm benchmark now
measures native compilation, first result, full execution, fresh-workspace
end-to-end execution, retained metadata, node openings and child choices. These
synthetic measurements do not establish a general speedup or a WCO bound.
