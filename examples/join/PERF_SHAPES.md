# Perf: where AutomatonJoin wins and loses

Implemented follow-up: [scheduled compositional automata](SCHEDULED_AUTOMATA.md)
combines whole-atom and prefix execution, bounded planning, and lazy compiled masks.

AutomatonJoin's whole-atom traversal and the on-demand product have complementary
pruning behavior. Completing a binding before probing the next atom can save
work; synchronizing shared prefixes can save much more when intermediate
bindings become large. These profiles support keeping both schedules inside an
automaton-based design.

## Measurements

These are **prepared visits**, not total query times. The graph, query plan,
correctness check, and warmup are outside the perf region. Each new process
checks complete sorted bindings against FstJoin, then performs enough repeated
visits for roughly two seconds of measurements. Timings below are medians of
three `perf stat` runs; recording uses a separate run.

| Shape | AutomatonJoin visit | On-demand visit | Automaton node reads | Product node reads |
|---|---:|---:|---:|---:|
| uniform-star | 21.04 ms | 30.25 ms | 1,220,372 | 1,700,957 |
| uniform-triangle | 16.39 ms | 27.97 ms | 913,655 | 1,326,494 |
| uniform-anchored | 2.34 µs | 77.95 µs | 175 | 4,709 |
| skew | 115.62 ms | 10.84 ms | 8,192,603 | 539,301 |
| parity-triangle-128 | 260.26 ms | 0.44 µs | 4,817,417 | 25 |

All primary profiles use unchanged production AutomatonJoin traversal. Counts
come from a separate disposable diagnostic build. Production library files are
unchanged. FstJoin remains the correctness oracle; the product's node/candidate
counts agree with it in the earlier [on-demand comparison](ON_DEMAND_JOIN.md).

## 1. AutomatonJoin often wins by doing less search

On the uniform star, whole-atom execution reads 28% fewer nodes. It fully resolves
the shared object before starting the second pattern, so every second-atom
level can use that object's complete value. Product execution shares only the
bits resolved so far. Prefix synchronization does not universally minimize work.

The instruction count is **250 million versus 406 million** per star visit.
Normalized by node reads, that is 205 versus 239 instructions. Cycles per node
are closer: 66.9 versus 69.5. Most of the star's cycle advantage is explained by
fewer visited nodes, with a smaller per-node advantage. Node-normalized costs
also include mask compilation, candidate enumeration, and callbacks; they are
not isolated node-access microbenchmarks.

On the anchored query, the first atom produces one binding. AutomatonJoin then
performs one exact-subject probe of the second atom: **84 + 91 node reads**.
The product instead visits 4,709 nodes while interleaving partial bindings.
Instructions are **36,799 versus 1,062,299**, and visits take 2.34 versus 77.95 µs.
Completing a selective atom early has a direct pruning benefit.

On the uniform triangle, AutomatonJoin similarly visits 913,655 nodes versus
1,326,494. The first atom produces 9,907 bindings, the second produces 4,925,
and the last rejects them all. The intermediate search remains manageable.
This is a cyclic query with zero results where whole-atom execution wins.

## 2. Skew loses through repeated expensive probes

Class-first AutomatonJoin produces 100,043 subject bindings, then starts the
rare-edge pattern from its root **100,043 times**. Those probes account for
7,948,758 of its 8,192,603 node reads: **97% of its traversal work**. Only 1,053
probes produce a binding. The product reads 539,301 nodes.

The instruction count grows to **1.717 billion versus 128 million**, even though
AutomatonJoin has the *lower* normalized instruction cost: 210 versus 238
instructions per visited node. Its kernel executes efficiently; the chosen
schedule asks it to do too much work.

The sampled self-cycle breakdown for class-first is:

| Function/work | Share |
|---|---:|
| Whole-atom walk | 54.69% |
| `read_word` | 19.31% |
| `read_delta` | 16.32% |
| `restriction_mask` | 6.04% |
| `compute_atom_masks` | 2.33% |
| Array zeroing (`memset`) | 0.92% |

Eliminating mask construction would remove only about 8% of this profile. It
cannot account for an approximately 11× traversal regression.

A paired ordering control uses the **same diagnostic binary** with traversal
counters disabled during perf, preserving explicit input order. Class-first
takes **116.90 ms**; rare-first takes **6.63 ms**, with the same 1,053 bindings.
Rare-first reads **216,250 nodes**, enters atoms 19,940 times, and executes
**74 million instructions versus 1.721 billion** class-first. This is roughly
18× faster than class-first and faster than the product's 10.84 ms prepared
visit. The live planner does not yet select that order automatically.

After ordering is fixed, restriction-mask construction accounts for about 26%
of the shorter run. Compiling constant masks and cheap bound-value updates is
then a useful second optimization. The planner needs conditional probe cost and
fanout, not only a count of constants or relation cardinality; the earlier
[investigation](INVESTIGATION.md) includes cases where the smaller first
relation is still the slower choice.

## 3. The contradictory triangle needs shared-prefix pruning

The contradictory-group fixture requires variables to belong simultaneously to
the same and opposite groups. Its entire FST is **430 bytes**. AutomatonJoin
enters its three atoms this many times:

```text
first atom:            1 entry       ->    32,768 matches
second atom:      32,768 entries     -> 4,194,304 matches
third atom:    4,194,304 entries     ->         0 matches
```

Every last-atom probe fails at its first node. The executor discovers the same
high-bit incompatibility separately for every complete intermediate binding.
The product propagates those high-bit constraints across all three atoms and
rejects after **25 node reads and 24 accepted slot candidates**.

AutomatonJoin executes **3.793 billion instructions**; the product executes
about **6,163**, including repeated-visit loop overhead. The prepared visit
comparison is 260 ms versus 0.44 µs. The earlier 1.32 µs product result includes
fresh query/workspace construction, which is excluded here.

The AutomatonJoin profile changes substantially:

| Function/work | Share |
|---|---:|
| `restriction_mask` | 34.09% |
| `compute_atom_masks` | 20.20% |
| Whole-atom walk | 14.42% |
| Array zeroing (`memset`) | 14.17% |
| Atom setup/dispatch | 8.98% |
| `read_word` | 7.57% |

`compute_atom_masks` initializes a 64-element mask array (512 bytes) at each
atom entry, although this fixture has only three levels. Millions of immediate
root failures expose both that initialization and unnecessary suffix-mask work.
Computing only reached masks and reusing exact-sized scratch storage would help.
They still leave millions of unnecessary intermediate bindings and root probes.
The previously tested atom orders all retain the large intermediate search;
changing order alone does not solve this fixture.

The uniform triangle and this triangle have the same query shape and both
return zero rows. Their intermediate distributions and shared-prefix
correlations determine which schedule wins. Query shape or final cardinality
alone cannot decide the execution strategy.

## 4. Branch and cache metrics refine the interpretation

On the star, AutomatonJoin executes 26.6 million branches versus the product's
69.0 million. However, their absolute branch misses are similar: approximately
530,000 versus 492,000. Generic cache misses are also similar: roughly 73,500
versus 69,000. The product's extra instructions and visited states explain more
than a claim that AutomatonJoin has fewer mispredictions or cache misses.

On the losing skew/cycle fixtures, AutomatonJoin's branch-miss rates are only
0.42%/0.14%, with about 3.8/3.7 instructions per cycle. The counters show large
amounts of mostly predictable repeated work. They do not establish a general
memory-bandwidth or storage-decoding bottleneck. Generic cache events are not a
precise classification of stalls or DRAM traffic.

## Implications for the next implementation

1. **Keep whole-atom execution.** It propagates complete values early and has a
   compact Exact/Free walk. Synchronizing every level can increase search work.
2. **Fix planning on skew.** Estimate intermediate fanout and the cost of the
   resulting bound/free probe. The ordering control demonstrates a practical fix.
3. **Use product pruning where intermediate bindings explode.** For correlated
   cycles, expose shared-prefix contradictions before enumerating full pairs.
4. **Compile restrictions and defer unused work.** Precompile static masks;
   update bound-variable digits cheaply; avoid initializing 64 masks for a
   three-level probe that fails at its root. These improve the nested kernel
   after choosing an effective schedule.
5. **Investigate a scheduler with both granularities.** A future automaton engine
   could synchronize shared prefixes when needed and complete selective atoms
   when that is cheaper. This is a proposed experiment, not an implemented or
   measured adaptive strategy. It needs explicit tracking of which variable bits
   are known; changing execution order cannot reuse the current fixed ownership
   plan blindly.

The profiles argue for a common automaton representation with different
transition schedules. They do not justify replacing every nested probe with a
synchronized product or treating mask compilation as the sole regression.

## Reproduction and raw evidence

Rust 1.98.1, profiling optimization with debug symbols, default CPU target,
single-threaded, Ryzen 5 PRO 4650U. Runs execute sequentially without CPU pinning
or frequency control. `perf stat` records cycles, instructions, branches, branch
misses, cache references, and cache misses. The six events run about 83% of the
time and are scaled by perf; counters below are normalized per visit. Medians of
counters/ratios need not form exact arithmetic identities. `perf record` samples
user cycles at 999 Hz with DWARF stacks; displayed percentages are self samples
from a separate run. FIFO control excludes setup and teardown. Raw recordings
are discarded after saving summaries to limit disk usage.

```sh
join_snapshot=$(python3 examples/join/prototype/prepare.py)
CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo build --offline --profile profiling --manifest-path "$join_snapshot/Cargo.toml" --example join_profile
cp /tmp/sgraph-composition-target/profiling/examples/join_profile /tmp/sgraph-join-profile-baseline
python3 examples/join/prototype/shape_perf.py /tmp/sgraph-join-profile-baseline

join_diagnostic_snapshot=$(PROTO_DIAGNOSTICS=1 python3 examples/join/prototype/prepare.py)
CARGO_TARGET_DIR=/tmp/sgraph-composition-target cargo build --offline --profile profiling --manifest-path "$join_diagnostic_snapshot/Cargo.toml" --example join_profile
cp /tmp/sgraph-composition-target/profiling/examples/join_profile /tmp/sgraph-join-profile-diagnostics
python3 examples/join/prototype/shape_perf.py /tmp/sgraph-join-profile-diagnostics --mode stats
python3 examples/join/prototype/shape_perf.py /tmp/sgraph-join-profile-diagnostics --mode rare-first
JOIN_INPUT_ORDER=1 /tmp/sgraph-join-profile-diagnostics automaton-rare-first skew stats 1
```

The diagnostic patch lives only in the disposable crate; its per-atom arrays
cover up to 16 atoms and are used here only for two/three-atom fixtures. Node and
candidate counters compile out of ordinary visits. The ordering control is a
separate patched-binary measurement; its class-first run agrees closely with
the primary production-traversal measurement.

Evidence: [combined CSV](prototype/results/shape-perf/comparison.csv),
[work counts](prototype/results/shape-perf/work.json),
[primary hardware-counter medians](prototype/results/shape-perf/perf-summary.json),
[ordering control](prototype/results/shape-perf/rare-first-summary.json),
[source/binary manifest](prototype/results/shape-perf/manifest.txt).
Per-run counter CSVs, timings, and self profiles are alongside these files.
