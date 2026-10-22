# Freeze profiles at 1M and 4M triples

The measurements below record the original experiment. Subsequent runs of
the benchmark runner select packed inputs only and exclude BFS-sealed cases.

This profiles the existing `compact_blocks()` implementation, plus an ordinary
`freeze()` heap control. The recursive compactor is not run. Production
compaction code is unchanged; the new work is a gated profiling driver.

These are the original 1024-bit-checkpoint measurements. The subsequent
[64-bit checkpoint experiment](FST_RANK64.md) changes rank lookup and compares
fresh before/after timings and temporary memory.

## Method

All graphs are synthetic mixed relations with exactly 1,000,000 or 4,000,000
distinct triples. Each subject initially has two distinct predicates and a
hashed object. Before freezing, 1024 triples are removed and 1024 triples with
a different predicate are inserted, preserving the exact triple count.
Coordinates are bounded by the declared domains. This is a controlled workload,
not a real RDF corpus measurement.

`packed` starts from normal packed BFS storage. `bfs` constructs the relation
entirely in the edit arena and seals it in BFS order before applying the final
edits. Their logical graphs and final physical-state counts are identical.
Updates use the indexed lookup strategy to keep fixture preparation bounded;
index destruction is included in consuming freeze, as it is in the API.

FIFO enable/acknowledgment and disable/acknowledgment gates restrict `perf`
measurement to the consuming freeze call. Construction, deserialization,
update-index preparation, edits, input/output counting and correctness probes
are outside the gates. The tiny control handshake is included in counters but
outside the driver's wall-clock timer. No unrelated build/test workload ran
concurrently.

Three separate core-counter runs collect user cycles, instructions, branches
and branch misses. A fourth run measures cycles/instructions and generic cache
references/misses. All counters report 100% running time, without multiplexing.
Reported wall times are medians of the three core-counter runs; cache-pass wall
times are not substituted. For example, the 1M heap cache pass took 95.5 ms
versus 55.5 ms in the core runs, so cache-pass timing is deliberately separate.
Cache event meaning is CPU-specific; its ratio is not an L1 miss rate.

`perf record` uses user cycles, 1999 Hz, DWARF stacks and up to four fresh
fixture iterations, enabling measurement separately for each freeze. Reports
use self sample weights. No recording reports lost samples. Inlining/identical
code folding and incomplete stack reconstruction limit exact callsite
attribution. File recordings also contain unresolved kernel-looking sample
addresses despite the user-only event configuration; no exact syscall/kernel
percentage is inferred from those addresses.

Machine: AMD Ryzen 5 PRO 4650U, 6 cores / 12 threads, perf 7.2.5. Build:
profiling profile (release optimization plus debug information), all features,
default portable CPU target, 4096-byte graph blocks. Runs are sequential and
unpinned; hardware counters and phase timings complement the wall-clock data.

Every large job runs through a user systemd service with a requested memory
cgroup limit and a verified inherited hard/soft `RLIMIT_AS`. Effective caps
were 1424..1488 MiB, calculated as at most half the available memory. Memory
cgroup controls are not delegated on this system, so the enforced fallback is
the address-space limit, not `MemoryMax`. Each job verifies its limits before
exec and the Rust process checks its inherited limit. Unit results and limit
verification are retained. All 50 services completed successfully.

## Timings and scale

| Input | Triples | Physical states after edits | Heap | File | Compressed |
|---|---:|---:|---:|---:|---:|
| Packed | 1,000,000 | 248,878 | 55.46 ms | 298.45 ms | 360.38 ms |
| Packed | 4,000,000 | 1,840,091 | 374.85 ms | 2041.99 ms | 3841.78 ms |
| BFS-sealed | 1,000,000 | 248,878 | 124.94 ms | 431.79 ms | 729.80 ms |

Ordinary `freeze()` on the packed 1M heap control takes 212.62 ms, versus
55.46 ms for `compact_blocks()` (3.83x). It performs the old physical graph
rebuild and exact minimization; this profile is not the in-place sealing API.

Physical sharing changes with scale: the 4M relation has 7.39x as many physical
states as the 1M relation, despite only 4x as many triples. Triple count alone
does not explain the timing growth. Backend timings compare the same physical
graph at each size.

## Where the work goes

### Heap: relocation rank queries, not copying bytes

`RankBits::rank` accounts for 41.08% of self sample weight at 1M and 33.80% at
4M on packed input. On 1M BFS-sealed input it accounts for 33.62%. Ordinary
memory-move routines have approximately 1% self weight on packed heap input;
heap `append_record` has 0.90% at 1M and 1.58% at 4M. The copying phase also
translates every outgoing address, so a long copying phase does not imply
that copying payload bytes is expensive.

The rank directory stores one prefix count per 1024 address slots. Each lookup
recounts up to 15 complete bitmap words plus the partial target word. It is
called repeatedly for parent/child relocation, and for classifying sealed/edit
states by planned output level. Annotating the 4M heap recording confirms SIMD
and scalar software-popcount loops, not hardware POPCNT, in this build.

Packed heap has 3.07 instructions per user cycle at 1M, 2.55 at 4M; branch miss
rates are 2.80% / 2.63%. The independent cache-pass miss ratios are 9.29% /
18.47%. This supports investigating rank work and relocation locality before
optimizing the byte-copy primitive.

### Compressed: repeated decompression during reachability marking

LZ4 decompression has 68.88% self weight at 1M and 81.20% at 4M. LZ4 compression
has only 2.50% / 2.30%. Recompressing the output is not the dominant cost.

The wall-clock phase timers agree:

| Packed input | Backend | Mark live states | Plan addresses | Copy/retire |
|---|---|---:|---:|---:|
| 1M | Heap | 11.96 ms | 6.43 ms | 36.87 ms |
| 4M | Heap | 72.90 ms | 45.48 ms | 255.25 ms |
| 1M | File | 211.00 ms | 12.15 ms | 74.20 ms |
| 4M | File | 1498.94 ms | 82.54 ms | 452.22 ms |
| 1M | Compressed | 291.84 ms | 11.81 ms | 56.70 ms |
| 4M | Compressed | 3333.07 ms | 93.64 ms | 404.39 ms |

Median phase times need not sum exactly to median total time, and phase scopes
exclude maintenance destruction before marking and final backend construction.

The initial marker is DFS over a physically BFS graph. It follows children
immediately, jumping between levels/chunks while the backend retains a bounded
stack cache. The interpretation is that access order causes repeated source
chunk decoding; this run measures decompressor cost, not an exact decompression
count. The next experiment should test that interpretation directly.

### File: source access dominates marking

Marking takes roughly 71% of total at 1M and 73% at 4M. User sample hotspots
include `read_var`, `read_u64` and `ensure_stack_entry`, alongside rank lookup.
This points to source-read locality rather than output packing as the first
backend-specific target. The data does not establish exact syscall counts or
separate page-cache hits from physical disk reads.

### Default freeze: hashing and allocation

The ordinary `freeze()` control has substantial hashing and allocator self
weight: the largest `DefaultHasher::write` entry is 19.22%, with additional
hash-table/hasher entries and allocator/free routines. Its IPC is 1.42, versus
3.07 for the block compactor. This is consistent with constructing temporary
physical graphs and re-minimizing, work the block compactor already avoids.

## Prioritized follow-ups

1. Make rank lookup constant-time with a finer two-level rank directory. A
   per-word `u16` prefix within each 1024-bit group can replace the repeated
   complete-word popcounts, leaving one masked-word popcount. This adds only
   temporary compaction metadata; it does not add a permanent update index.
   Also compare a CPU-specific build to quantify the software-popcount cost
   without confusing that build change with an algorithm improvement.
2. Test level-by-level, address-ordered live marking. Use the existing live
   bitmaps as a frontier: mark children while scanning the current level, then
   visit the next level in block order. This preserves shared-state visitation
   and targets paged/chunk cache reuse. It does not require incoming-edge
   indexes or a different output layout. A bounded compaction-specific chunk
   cache is another control worth comparing.
3. Remove repeated per-level rank classifications of sealed states using
   temporary block/level information. The 1M BFS-sealed heap input needs
   26.59 ms planning versus 6.43 ms on packed input, and 69.90 ms copying versus
   36.87 ms. Output contents and physical state counts are identical.

These were recommendations from this profiling task. The later
[word-level rank experiment](FST_RANK64.md) implements a full prefix per word
and measures its memory/time tradeoff; the other suggestions remain separate.

## Reproduce

```sh
CARGO_TARGET_DIR=target/automata-prototype cargo rustc --offline --all-features \
  --profile profiling --jobs 1 --example fst_freeze_profile -- \
  -C link-arg=-Wl,--threads=1
python3 examples/join/prototype/fst_freeze_profile.py --default-control
```

The Rust driver accepts triple count, backend, layout (`packed` / `bfs`), method
(`blocks` / `freeze`), repetition count and optional shape. It rejects graphs
below 1M triples. The Python runner adds FIFO gating, systemd containment,
core/cache counter passes, sampling, reports and source/binary hashes. Counts
and inserted/deleted probes are validated outside each measured interval.

[Raw counters, reports, summaries and manifest](prototype/results/fst-freeze-profile/)
include the rank disassembly annotation. Raw `.data` recordings are retained
locally but ignored by Git because of their size.

[Profiling driver](../fst_freeze_profile.rs) ·
[Runner](prototype/fst_freeze_profile.py) ·
[Compaction algorithm](FST_BLOCK_COMPACTION.md)
