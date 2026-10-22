# Recorded benchmark evidence

The measurements under `prototype/results/` are original experimental records. They are retained unchanged
for provenance, marked generated to reduce review noise, and excluded from Cargo
packages. Start with the reports and aggregate results below; individual trial
logs, perf captures, commands and hashes remain in each experiment directory.
Reproduction runners live in `prototype/` and use current production
sources unless a saved before-change binary is explicitly required.

| Area | Report | Aggregate evidence |
|---|---|---|
| Current default update index | [Compact index](FST_COMPACT_INDEX.md) | [Default-index comparison](prototype/results/fst-default-index/summary.json) |
| Current default freeze | [Rank-free BFS](FST_RANK_FREE_COMPACTION.md) | [Rank-free comparison](prototype/results/fst-rank-free/summary.json) |
| Update semantics/backends | [Updates](FST_UPDATES.md) | [Prepared update comparison](prototype/results/fst-updates-final/summary.json) |
| Memoized lookup | [Memoization](FST_LAZY_INDEX.md) | [Initial working-set experiment](prototype/results/fst-lazy/summary.json) |
| Query API promotion | [Promotion](PROMOTION.md) | [Join controls](prototype/results/promotion-joins.jsonl) |
| Automaton schedules | [Frontiers](AUTOMATA_FRONTIERS.md) | [Final join comparison](prototype/results/automata-final-joins/summary.json) |
| Large graph patterns | [Large cycles](LARGE_CYCLES.md) | [Final cycles](prototype/results/automata-final-cycles/summary.json) |
| Storage layout | [Layout comparison](LAYOUT_COMPARISON.md) | [Paired layout runs](prototype/results/layout-comparison/summary.json) |

Earlier sampled indexes, removed occupancy filters and alternative freeze
algorithms remain historical comparisons. Their API labels and representations
may differ from today's default `Sampled(1)` and rank-free freeze. Use each
report's status note and manifest when interpreting results. In particular,
[fst-level-metadata](prototype/results/fst-level-metadata/summary.json) includes the removed filters;
it is not a description of the current engine.

The larger default-index run uses verified systemd address-space limits. Its
initial timing outlier remains in
[fst-default-index/initial-large-outlier](prototype/results/fst-default-index/initial-large-outlier/README.md);
the reported aggregate uses the subsequent repeated runs. Retained allocation
figures are not RSS, and omitted or unavailable counters must not be read as zero.
