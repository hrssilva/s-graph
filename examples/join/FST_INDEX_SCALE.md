# Suffix-index scaling: 1M, 10M and 100M triples

The `Indexed` label in historical results refers to the former variant. The
current default is `Sampled(1)`; the redundant `Indexed` API was removed.

These are historical measurements of the 16-byte index. The current default
uses narrow offsets and 32-bit prefixes; see [FST_COMPACT_INDEX.md](FST_COMPACT_INDEX.md).
The example now checks the compact representation bound instead of the historical
16-byte formula.

Measured on the current machine on 21 October 2026 using exact **distinct**
triple counts, the heap packed FST and indexed mutable preparation. Both
synthetic fixtures have two predicates per subject; two more predicate IDs
remain reserved for edits. The mixed fixture hashes endpoints; the functional
fixture uses XOR-related endpoints with explicit domain bounds. These are
synthetic shapes, not measurements on an RDF dataset.

## Results

MiB means 1,048,576 bytes. Packed RAM includes the graph's retained layout
allocations. Index size is the packed suffix register alone, before edits.

| Triples | Shape | Physical states | Packed RAM | Suffix index | Index preparation |
|---:|---|---:|---:|---:|---:|
| 1,000,000 | mixed | 248,607 | 6.70 MiB | 3.79 MiB | 30.020 ms |
| 1,000,000 | functional | 21 | 1,010 B | 504 B | 0.004 ms |
| 10,000,000 | mixed | 3,443,790 | 78.01 MiB | 52.55 MiB | 266.155 ms |
| 10,000,000 | functional | 25 | 1,192 B | 592 B | 0.005 ms |
| 100,000,000 | mixed | — | Build allocation refused | Not reached | — |
| 100,000,000 | functional | 29 | 1,446 B | 680 B | 0.003 ms |

The index retains **16 bytes per physical state + 24 bytes per radix level**
on this 64-bit host. Each measurement asserts that exact formula. It scales
with suffix diversity, rather than directly with the number of triples. The
mixed case grows from 248,607 to 3,443,790 states between 1M and 10M triples.
The functional case still has only 29 states at 100M triples.

The 100M mixed index was **not measured**. It is not reported as zero or
extrapolated from 10M. The builder failed an 85,983,248-byte allocation under
its enforced limit after generating all 100M triples. The current builder
holds an input vector and an equally sized scratch vector: those alone need
2,400,000,000 bytes at 100M, before state registers and packed output. The
constraint encountered here is build-time memory, not demonstrated exhaustion
by the final suffix index.

## Actual process memory and containment

All dataset processes run sequentially in transient systemd user services.
Each process verifies its hard and soft address-space limits before allocation.
The helper uses `exec`, replacing itself with the Rust executable; the benchmark
does not spawn child processes. All threads share its bounded address space.
Core-file storage is disabled with `LimitCORE=0`; a Rust allocation failure
still appears as `Result=core-dump`/SIGABRT in systemd's termination status.

The machine had roughly 2.9–3.1 GiB available, with swap effectively full.
The first pass capped each process at half current available RAM, at most
2 GiB. Both 100M fixtures reached generation but failed the 1.2 GB scratch
allocation at that limit. A supplementary 100M pass allowed 80% of available
RAM, still reserving at least 512 MiB. Its actual caps were 2.375 GiB for
functional and 2.453 GiB for mixed. The functional run then completed; mixed
still failed during construction. Bounds are rounded down in 16 MiB units.

**The enforced mechanism is systemd `LimitAS` / kernel `RLIMIT_AS`, not a cgroup
RAM cap.** The user manager accepted `MemoryMax` and `MemorySwapMax`, but this
machine has not delegated its memory controller: the corresponding cgroup
files are absent. The initial cgroup-only verifier refused to run. Creating a
system-manager service requires administrator authentication, which was not
available. We used the verified per-process address-space bound instead; it
also bounds that single process's RSS. Requested cgroup/swap properties are
recorded but must not be interpreted as enforced. No controller configuration
was changed. The build is also a limited systemd job, with a single-threaded
linker; `LimitAS` applies separately to its compiler/linker children, not as an
aggregate build cgroup limit.

Service properties queried after a successful transient unit is unloaded can
show `infinity` or `[not set]`. The actual evidence is the numeric
`getrlimit(RLIMIT_AS)` record inside each service, plus the Rust executable's
independent `/proc/self/limits` assertions before generation. RSS and high-water
RSS come from `/proc/self/status`, not unavailable cgroup memory accounting.

Completed process peaks and retained graph allocations differ:

| Triples | Shape | Peak RSS, MiB | Retained packed graph + index, MiB |
|---:|---|---:|---:|
| 1,000,000 | mixed | 81.53 | 10.4932 |
| 1,000,000 | functional | 25.23 | 0.0018 |
| 10,000,000 | mixed | 884.11 | 130.5631 |
| 10,000,000 | functional | 231.12 | 0.0021 |
| 100,000,000 | functional | 2,290.88 | 0.0024 |

Peak RSS includes building and validation. Even after those temporaries are
freed, the allocator can retain pages, so process RSS need not equal live graph
allocations. For failed runs only the last reported RSS is available; it is not
claimed to be the failure's true peak. Index preparation timings are single
samples, not a timing benchmark with repetitions.

All completed cases validate the full logical count using physical-state
memoization, plus one insertion/removal recovering the original packed root.
No tuple enumeration is used to validate the compressed functional graph.

## Reproduction

```sh
python3 examples/join/prototype/fst_index_scale.py
# Supplemental pass; reuses the existing binary, retains >=20% RAM headroom:
python3 examples/join/prototype/fst_index_scale.py --skip-build --sizes 100000000 --fraction 0.8 --max-gib 3 --output examples/join/prototype/results/fst-index-scale-100m-80pct
```

The driver builds with `cargo rustc --offline --profile profiling --jobs 1`,
using one linker thread to fit the address-space bound, and keeps temporary
files on the repository filesystem. Each job has a 600-second runtime limit.
It never retries a dataset without a limit. User systemd bus access is required.

[Summary](prototype/results/fst-index-scale/summary.json) ·
[First pass, commands and in-process limit verification](prototype/results/fst-index-scale/runs.json) ·
[100M retry](prototype/results/fst-index-scale-100m-80pct/runs.json) ·
[Benchmark source](../fst_index_scale.rs) ·
[Runner](prototype/fst_index_scale.py)
