# Comparison harness

Production automata live in `src/`. Historical copies of promoted `.rs` files in
this directory are retained as experiment records; edit the production files.
`prepare.py` copies current production sources into a disposable crate, then
adds the compiled-filter, lifted and demand alternatives, a postorder storage
adapter for layout comparisons, and benchmark entry points. It does not overwrite promoted modules with historical copies.

```sh
mkdir -p target/automata-tmp
TMPDIR="$PWD/target/automata-tmp" python3 examples/join/prototype/prepare.py
# Set this to the disposable path printed above:
sgraph_snapshot="/printed/disposable/path"
cargo test --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --lib --tests
cargo run --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example join_profile -- default skew visit 20
cargo run --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example query_ops_bench
cargo run --offline --profile profiling --manifest-path "$sgraph_snapshot/Cargo.toml" --example query_ops_bench -- --paths
```

`join_profile` engines include `default`, `scheduled-adaptive`, `automaton` and
`fst`; fixtures include `uniform-star`, `skew`, `mixed-triangle` and
`parity-triangle-128`. Reduce repeat counts for slow engines. Run timings
sequentially without competing builds/tests. The `stats` phase reports traversal
counters instead of timing. Use the existing profiling scripts for gated perf
runs that exclude setup.

`PROTO_DIAGNOSTICS=1` enables the historical AutomatonJoin counter patch in the
disposable copy only. That patch uses 16-atom diagnostic arrays; keep diagnostics
off for huge-BGP tests. Its Git invocation stops repository discovery at the
snapshot parent so temporary crates under `target/` cannot patch the main tree.

Current promotion results and source provenance are linked from
[PROMOTION.md](../PROMOTION.md). Earlier experiment reports remain alongside it.

The [layout comparison](../LAYOUT_COMPARISON.md) runs the same scheduled
executor over `PathFstDeltaPow2` and `PathFstDeltaPow2Bfs`. `layout_bench` checks
results and work counters before timing; `layout_compare.py` runs paired orders
and `layout_perf.py` collects gated hardware counters.

The [forced Levels comparison](../FORCED_LEVELS.md) uses `levels_compare.py`
for all regular and huge BGP shapes, and `levels_stress.py` with
`schedule_stress` for bounded carry-chain diagnostics and timed full traversal.
The latter's node-budget method is injected into the disposable crate only.

The [large-cycle investigation](../LARGE_CYCLES.md) adds `cycle_bench` and
`cycle_compare.py`: XOR, carry and shuffled rings on 4,096 vertices, dense odd
cycles with a planted answer, and controls for storage bit allocation. An
exhausted node budget is reported explicitly; partial traversals are not timed
as complete queries. Use `--cases kind:vertices:atoms` to select controls and
`--budget` to adjust the diagnostic limit. Dense fixtures interpret `vertices`
as the size of each bipartite group.

The [current automaton/frontier design](../AUTOMATA_FRONTIERS.md) uses
`evolution_compare.py` for paired pre-change/current adaptive measurements.
Use separate Cargo target directories for the two source snapshots; verify
the executables before running the matrix. `frontier` is also an explicit
policy accepted by `layout_bench`, `cycle_bench` and `schedule_stress`.

The comparison-only compaction APIs and reserved-block storage trait require the
`experimental` feature. Use `--all-features` for compaction comparisons. Normal
rank-free freeze and its diagnostics remain part of the default library. Raw
records are listed in [the results index](../RESULTS.md), marked generated
for review and excluded from Cargo packages. Run this harness from a repository
checkout when reproducing experiments; packaged crates omit the raw lab records.

The [parallel FST construction prototype](../FST_PARALLEL_BUILD.md) uses
`fst_parallel_build.py` for isolated, memory-limited build comparisons.
