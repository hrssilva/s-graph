# Parallel FST construction prototype

`PackedFst::try_build_parallel(triples, domains, block_capacity)` is available
with **both** `experimental` and `parallel`. The ordinary builder remains
sequential. Heap, file and compressed FSTs use the same construction algorithm;
backend writes happen during sequential packing.

```rust,ignore
use s_graph::Fst;

let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
let graph = pool.install(|| {
    Fst::try_build_parallel(triples, [subjects, predicates, objects], 4096)
})?;
```

Use a direct call to use the current/global Rayon pool. The prototype falls back
to ordinary construction with one worker or fewer than 65,536 input triples.
Validation and graph set semantics are shared with the sequential builder.

## Algorithm

```text
validate input
parallel sort triples; remove duplicates
frontier = [whole input at depth 0]
while frontier has fewer than 2 * workers partitions:
    select largest splittable partition with at least 32,768 triples
    if none: stop
    partition its point/scratch slices by the next radix symbol
    retain a prefix node referencing the resulting children

in parallel, for each frontier partition:
    build a sequential minimized DAG with private per-depth registers
    return distinct states, local root ID and visited count
    free private registers

for each returned DAG in prefix order:
    for depth from leaves to its root:
        translate local child IDs to global canonical IDs
        intern each exact (existence word, translated children) signature
        retain local-to-global IDs only for adjacent depths
        consume local states as they are merged

rebuild prefix nodes child-first using the same global registers
free registers, input triples and partition scratch
pack the globally minimized DAG in the existing BFS format
```

The slices are disjoint; workers share neither mutable registers nor storage.
Exact signature comparisons handle hash collisions. Merging after child ID
translation preserves equivalent suffixes across partitions, including suffixes
that acquired different local IDs. Merging only root IDs or concatenating local
outputs would lose that property.

The frontier can overshoot its target by at most 62 partitions because a radix
split has up to 64 nonempty children. Thus at most `2 * workers + 62` private DAGs
are retained, independently of graph size. This is a bound on the number of
partitions, **not** a fixed byte budget. Completed DAGs remain resident until the
merge starts; duplicated states and the global merge register can increase peak
RSS. Final graph state counts match sequential global minimization.

Splitting the largest eligible prefix also handles long unary prefixes. The
frontier cutoff limits scheduling overhead; it does not guarantee balanced tasks
when one split produces a dominant child and many small siblings. Sorting is
also parallelized, so measured speedups include sorting and minimization.

## Reproduce

```sh
CARGO_TARGET_DIR=target/automata-prototype cargo build --release --all-features \
  --example fst_build_perf
python3 examples/join/prototype/fst_parallel_build.py
```

The runner uses independent processes, three trials per configuration, alternating
configuration order, deterministic shuffled input, one million distinct triples,
and 1.5 GiB systemd memory limits with swap disabled. Construction timing includes
validation, sorting, scratch allocation, local construction, merging, and packing.
Fixture generation, pool creation and verification are excluded from that timer.
Peak RSS is process-wide `/usr/bin/time` high water, including generation and
verification; retained bytes describe the finished graph and are not RSS.

The runner records commands, source/binary hashes, individual logs and summaries.
Use `--backends heap file compressed` to measure other backends. Tests separately
check sharing, scans, counts, snapshots and subsequent updates on all three.

## Million-triple results

Three-trial medians on an AMD Ryzen 5 PRO 4650U (6 cores, 12 logical CPUs),
release build, heap backend. Times are milliseconds:

| Shape | Sequential | 2 workers | 4 workers | 8 workers | 8-worker speedup | Peak RSS, sequential → 8 workers |
|---|---:|---:|---:|---:|---:|---:|
| Functional | 150.3 | 115.1 | 70.9 | 52.4 | 2.87× | 25.4 → 25.4 MiB |
| Shared blocks | 101.4 | 66.6 | 51.9 | 42.0 | 2.42× | 25.4 → 25.5 MiB |
| Irregular | 1776.3 | 1908.7 | 1449.3 | 1482.4 | 1.20× | 213.7 → 302.3 MiB |
| Skewed | 187.8 | 134.5 | 80.4 | 67.1 | 2.80× | 25.5 → 25.9 MiB |

Final state counts and retained graph bytes match the sequential builder in all
trials: functional 41 states/1,642 bytes, shared blocks 9/1,620, irregular
1,060,277/16,822,080, skewed 273/34,514. These generated shapes test distinct
compression regimes; they are not measurements on RDF or web datasets.

The two-worker irregular case regresses by about 7%. Four workers perform better
than eight there, while eight workers perform better on the compressible shapes.
Consequently this prototype does not justify automatically enabling parallel FST
construction for everyone. Highly compressible graphs get substantial speedups
with little peak-memory change. Irregular graphs get a modest win with roughly
41% more peak RSS at eight workers. The next target is the global merge, which
currently performs a second exact interning pass sequentially while local DAGs
are still resident. Its contribution has not yet been isolated with phase
profiling; the table measures the complete build.

[Final trials and summary](prototype/results/fst-parallel-build/final/summary.json)
and their adjacent manifest/logs identify the reviewed source and binary. The
initial exploratory run remains in the parent directory; only source formatting
changed before the final run. Timings vary across runs on this shared laptop, so
use medians and avoid interpreting small worker-count differences as precise
scaling estimates.
