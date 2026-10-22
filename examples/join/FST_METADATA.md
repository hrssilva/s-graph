# Replacing maintenance indexes with runtime work

The quick experiment keeps the update algorithm and minimality invariant, and
changes exact suffix lookup:

| Strategy | Packed suffixes | Edited suffixes |
|---|---|---|
| `Indexed` (default) | Fingerprint index, checked structurally | Exact hash table |
| `ScanBase` | Scan physical packed records | Exact hash table |
| `ScanAll` | Scan physical packed records | Scan live arena slots |

`ScanAll` retains **no suffix lookup tables**. It still retains live edited graph
states, reference counts, a reusable slot arena and a free list. This is not a
zero-metadata or zero-RAM graph. Reference counts could themselves be replaced
with reachability tracing/recomputed incoming-edge counts; that different
reclamation experiment was not measured here.

The scans compare full signatures and preserve minimality after each update;
minimization is not postponed to compaction. Packed scans parse the earlier
physical levels to locate the target level, then inspect candidate words and
ordered child identities. No per-state directory is retained. This is a simple
exact space-saving baseline, not a claim that exhaustive scans are the best
possible low-memory maintenance algorithm.

```sh
cargo run --release --all-features --example fst_metadata_perf -- 4096 32 3
cargo run --release --all-features --example fst_metadata_perf -- 65536 16 3 heap
```

Select it explicitly:

```rust
use s_graph::mutable_fst::UpdateLookup;
let mut graph = packed.into_mutable_with_lookup(UpdateLookup::ScanAll);
```

`compact()` preserves the selection. Snapshots contain only the graph language;
loading and ordinary `into_mutable()` still choose indexed updates. The default
was not changed.

## Results

Two fixtures have two initial triples per vertex: a very small, highly shared
functional FST and a mixed pseudo-random endpoint fixture. The larger mixed
heap FST contains 72,726 physical states and 131,072 initial triples. Times below
are **microseconds per changed triple**, except first insertion, which includes
lazy initialization and is one complete operation.

| Vertices | Shape | Backend | Strategy | New insert | Undo insertion | Delete original | Restore original | First insertion | Lookup RAM, KiB | Total RAM, KiB |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | functional | heap | Indexed | 1.53 | 1.33 | 1.62 | 1.65 | 7.12 | 2.63 | 11.98 |
| 4096 | functional | heap | ScanBase | 2.49 | 1.88 | 2.52 | 2.43 | 4.32 | 1.91 | 11.26 |
| 4096 | functional | heap | ScanAll | 2.83 | 1.95 | 1.57 | 1.63 | 2.74 | 0.00 | 9.35 |
| 4096 | functional | file | Indexed | 1.00 | 0.82 | 1.02 | 1.02 | 5.16 | 2.63 | 11.99 |
| 4096 | functional | file | ScanBase | 1.04 | 0.81 | 1.05 | 1.10 | 2.71 | 2.38 | 11.73 |
| 4096 | functional | file | ScanAll | 1.39 | 0.92 | 0.84 | 0.79 | 6.42 | 0.00 | 9.36 |
| 4096 | functional | compressed | Indexed | 0.95 | 0.82 | 1.02 | 1.04 | 3.67 | 2.63 | 12.09 |
| 4096 | functional | compressed | ScanBase | 0.99 | 0.82 | 1.05 | 1.09 | 1.99 | 1.91 | 11.37 |
| 4096 | functional | compressed | ScanAll | 1.26 | 0.87 | 0.76 | 0.78 | 1.37 | 0.00 | 9.46 |
| 4096 | mixed | heap | Indexed | 1.37 | 1.21 | 2.00 | 2.15 | 204.61 | 41.29 | 107.26 |
| 4096 | mixed | heap | ScanBase | 55.54 | 6.33 | 18.58 | 47.19 | 58.11 | 0.92 | 66.88 |
| 4096 | mixed | heap | ScanAll | 54.57 | 6.11 | 17.70 | 46.38 | 56.46 | 0.00 | 65.97 |
| 4096 | mixed | file | Indexed | 3.06 | 3.19 | 5.30 | 4.39 | 334.02 | 40.80 | 56.63 |
| 4096 | mixed | file | ScanBase | 128.14 | 15.47 | 42.31 | 100.19 | 141.32 | 0.88 | 16.72 |
| 4096 | mixed | file | ScanAll | 91.61 | 11.43 | 32.27 | 77.38 | 139.39 | 0.00 | 15.84 |
| 4096 | mixed | compressed | Indexed | 5.30 | 5.91 | 9.43 | 6.73 | 310.17 | 41.29 | 110.13 |
| 4096 | mixed | compressed | ScanBase | 121.55 | 17.85 | 47.25 | 103.61 | 133.68 | 1.38 | 70.21 |
| 4096 | mixed | compressed | ScanAll | 124.07 | 17.64 | 45.58 | 102.25 | 132.02 | 0.00 | 68.83 |
| 65536 | functional | heap | Indexed | 1.23 | 0.98 | 1.28 | 1.28 | 6.16 | 2.15 | 11.38 |
| 65536 | functional | heap | ScanBase | 1.12 | 0.97 | 1.28 | 1.32 | 4.17 | 1.84 | 11.07 |
| 65536 | functional | heap | ScanAll | 1.53 | 0.90 | 0.96 | 0.90 | 3.34 | 0.00 | 9.23 |
| 65536 | mixed | heap | Indexed | 1.50 | 1.07 | 2.13 | 2.42 | 4,592.63 | 1,137.40 | 2,513.41 |
| 65536 | mixed | heap | ScanBase | 1,594.44 | 1.27 | 314.27 | 1,242.36 | 1,803.99 | 0.91 | 1,376.92 |
| 65536 | mixed | heap | ScanAll | 1,599.01 | 1.11 | 314.12 | 1,243.38 | 1,792.25 | 0.00 | 1,376.01 |

The large mixed graph saves about 1.11 MiB of suffix-index RAM. New insertions
become roughly 1,000 times slower and deleting original edges about 150 times
slower. Undoing the new insertions remains fast in this particular case: the
restored signatures are found early in packed scans, or a still-edited child
proves that no packed state can match. This cheap inverse operation should not
be mistaken for general removal cost.

The first single insertion without indexes is faster on the mixed fixture,
because constructing the complete index costs more than a few scans. Subsequent
arbitrary updates quickly favor the index. On the functional fixture the base
has only 11 states at 65,536 vertices, so a scan is small; both strategies stay
within a few microseconds. Removing the edited-state table adds little cost
at these small edit batch sizes; that result does not extrapolate to a large
live edit arena.

This suggests keeping indexed updates for sustained arbitrary writes. Runtime
scans are an option for tiny physical FSTs or extremely sparse writes under a
strict RAM budget. Compacting and dropping update metadata after a write burst
is the existing alternative for read-mostly use. No adaptive policy or bounded
cache was added based on this small experiment.

A subsequent [cross-edit memoization prototype](FST_LAZY_INDEX.md) adds the
explicit `UpdateLookup::Memoized` alternative, without changing the default.

## Protocol and validation

- 4,096 vertices / 32 updates on heap, file and compressed storage; 65,536
  vertices / 16 updates on heap. All use 4,096-byte blocks.
- Two fresh-graph passes, reversing strategy order in the second pass; three
  batch samples per phase per pass. Reported steady medians aggregate all six
  samples. First-insertion numbers aggregate two samples only.
- New inserts, undoing them, deletion of original packed triples and restoration
  are timed separately. Build, validation, memory accounting and compaction are
  outside update timing. Samples are wall time, sequential and unpinned.
- Memory is the final pass's graph-owned estimate after the new-insertion
  phase. HashMap overhead is approximate; neither process RSS nor transient
  workspaces are included. Index-free strategies assert zero corresponding
  retained lookup bytes.
- Every batch asserts the number of changes; inverse edits must leave no edited
  states. Each strategy is checked against the complete original triple set.
  A unit test checks the live state's count against independently minimized
  packing during edits for every strategy. Four mutable-FST unit tests and
  the existing five update/source integration tests passed.

[Benchmark source](../fst_metadata_perf.rs) ·
[Raw timings and memory](prototype/results/fst-metadata/summary.json)
