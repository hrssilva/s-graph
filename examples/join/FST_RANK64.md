# Word-level rank checkpoints during FST compaction

The measurements below record the completed experiment. The benchmark runner
now selects packed inputs only; BFS-sealed cases are excluded from future runs.

`RankBits` now stores an absolute `usize` population prefix before every
64-bit bitmap word, plus a terminal total. The previous version stored a
prefix every 16 words (1024 bits).

```text
finish(bitmap):
    total = 0
    for word in bitmap:
        prefix.push(total)
        total += popcount(word)
    prefix.push(total)

rank(slot):                         # marked positions strictly before slot
    word = slot / 64
    offset = slot % 64
    result = prefix[word]
    if offset != 0:
        result += popcount(bitmap[word] & ((1 << offset) - 1))
    return result
```

A lookup no longer recounts up to 15 complete words. On a 64-bit target the
prefix array costs eight bytes per bitmap word, plus eight bytes for the
terminal count. Previously it cost eight bytes per 16 words, rounded up,
plus the terminal count. Bitmap sizes and the four/eight-byte relocation
entry per live state are unchanged. These allocations are temporary; the
returned packed FST contains none of them. The shared rank implementation is
also used by `compact_recursive()`, although this experiment runs only
`compact_blocks()`.

## Scope of metadata

This section describes `compact_blocks()`. The subsequent
[depth-scoped compactor](FST_DEPTH_COMPACTION.md) implements reusable frontiers
and fused source discovery/copy through `compact_depths()`.

Current implementation: live bitmaps, rank prefixes and relocation entries
cover all depths together. There is one relocation entry per live physical
state, not per transition. Original packed storage uses one bitmap bit per
four virtual address bytes, sealed storage one per virtual byte, and the edit
arena one per slot. Prefix counts are per bitmap word, not per graph state.
Each edge consults the table for its child; it does not allocate another entry.

This global lifetime is a property of the current passes, not a requirement
of suffix sharing. Every edge goes from depth d to d+1. A forward compactor
could plan relocations for d+1, rewrite d using those child addresses, then
discard d and advance. At most two adjacent depths need relocation data at
once. The current state's new address could also come from the sequential
writer rather than a retained relocation entry.

Doing that requires changing how live records are enumerated and classified.
Today sealed/edit records are filtered at every depth using their globally
planned destination addresses, including after source records have been
freed. They need an independent depth enumeration instead. Reachability could
likewise use two deduplicating frontiers with address-ordered traversal rather
than the current recursive global marking. Merely slicing the relocation
array would leave the global bitmaps allocated. Address-indexed frontiers
must also be narrowed to each depth's address ranges to realize the expected
memory saving; sealed batches and the edit arena can scatter a depth's states
throughout their address spaces.

Per-source-block retirement counters still span the source blocks until
retired; they are not per-transition metadata. Planning can retain O(height)
output boundaries without retaining all per-state relocation tables. The
bound for an actual depth-streaming redesign would depend on the widest
adjacent depths, which may themselves contain much of the graph. It would
not be a one-block total-workspace guarantee.

## Measurement

Fresh before/after binaries use the same portable profiling build and exact
1M/4M synthetic graphs as [the earlier perf investigation](FST_FREEZE_PROFILE.md).
Three core-counter runs per case gate only consuming compaction; construction,
edits and result checks remain outside the timer. The runner verifies inherited
systemd-service address-space limits. No CPU-specific code generation was added.

Rank memory includes both bitmap words and their prefix directories. Workspace
includes rank memory, relocation arrays, block live counters and level metadata;
it excludes backend caches, allocator overhead, output block buffers and output
descriptors. It measures explicit allocation capacities rather than RSS.

Results: [before](prototype/results/fst-rank64/before/summary.json),
[after](prototype/results/fst-rank64/after/summary.json). Manifests include
source and binary hashes; each trial retains counter CSV and containment logs.

## Results

Medians of three freeze-only trials per case; 54 contained services passed.

| Input | Triples | Backend | Before | Word prefixes | Speedup |
|---|---:|---|---:|---:|---:|
| packed | 1,000,000 | heap | 55.12 ms | 35.53 ms | 1.55x |
| packed | 1,000,000 | file | 291.96 ms | 276.84 ms | 1.05x |
| packed | 1,000,000 | compressed | 359.83 ms | 337.97 ms | 1.06x |
| packed | 4,000,000 | heap | 362.56 ms | 266.87 ms | 1.36x |
| packed | 4,000,000 | file | 2080.84 ms | 1981.65 ms | 1.05x |
| packed | 4,000,000 | compressed | 3846.22 ms | 3769.31 ms | 1.02x |
| bfs | 1,000,000 | heap | 121.77 ms | 90.28 ms | 1.35x |
| bfs | 1,000,000 | file | 442.15 ms | 397.98 ms | 1.11x |
| bfs | 1,000,000 | compressed | 734.55 ms | 700.40 ms | 1.05x |

Explicit temporary memory is identical across backends for the same input.

| Input | Rank bitmap + prefixes, before → after | Total workspace, before → after | Extra workspace |
|---|---:|---:|---:|
| 1,000,000 packed | 233,488 → 439,480 B | 1,243,064 → 1,449,056 B | 201.2 KiB (16.6%) |
| 4,000,000 packed | 1,272,144 → 2,394,600 B | 8,707,716 → 9,830,172 B | 1096.1 KiB (12.9%) |
| 1,000,000 bfs | 1,651,832 → 3,109,304 B | 2,671,976 → 4,129,448 B | 1423.3 KiB (54.5%) |

All result state counts, retained output bytes, relocation bytes and source/output block-content peaks match before/after. All perf counters ran without multiplexing. Heap instruction counts fell by about 32–35%; file/compressed gains are limited by source access and decompression, as in the earlier profiles. The small 4M compressed timing change should be treated as a modest gain, not a decisive backend improvement.

The 4M heap timing samples were 362.56/368.94/353.47 ms before and 288.07/256.23/266.87 ms after. Runs are unpinned and show timing variability; instruction reductions provide supporting evidence independent of wall time.

[Machine-readable comparison](prototype/results/fst-rank64/comparison.csv).

Validation: all 134 library/integration tests and seven documentation tests
passed with all features enabled. The existing sparse rank test now also checks
empty, fully populated, exact-word-end and partial-word-end bitmaps. All large
benchmark trials checked triple counts, edited-triple probes and complete source
block retirement.
