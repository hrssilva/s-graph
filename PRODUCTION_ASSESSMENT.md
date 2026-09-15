# Assessment: productionizing the radix-tree triple index

Deliverable #1 from `PRODUCTION_HANDOFF_PROMPT.md`. This is my own
read of the design after going through `examples/README.md`, the full
~4,160-line `radixtree_dfs_interleaved_varskip_unrolled_generic.rs`
(structure, `BlockStorage` trait, all three backends, the bulk-build
and bounded-build paths, the 17-test suite), `reference.rs`,
`tree_serialize.rs`, and the current `src/` stubs. Where I disagree
with the handoff doc or found something it didn't mention, I say so
explicitly below rather than restating its framing.

## Verdict

The architecture is sound and the measurements back the claims I
checked. I'm porting it, not redesigning it. The real work is: (1) a
genuine `Result`-based error boundary, because the current code has
exactly one *silent, release-mode data-corruption* gap, not just
missing polish (§3 below — this is the one item I'd call a blocker,
not a nice-to-have); (2) feature-gating; (3) deleting the `src/`
stubs; (4) closing a stale-documentation gap I found between the
README and the actual code.

## 1. Design principles (§4 of the handoff) — agree, verified directly

I read the code, not just the doc comments, for each of these:

- **SoA / flat byte buffers, no node graph.** Confirmed — `BlockStorage`
  is `Vec<u8>` blocks + `Pos{block, offset}` throughout; zero
  `Box`/`Rc` node types anywhere in the trie itself.
- **Pre-reserved hot-path buffers.** Confirmed in `HeapStorage`'s
  `append_record`/`insert_bytes` and both other backends' pending-block
  buffers.
- **Batched I/O/codec calls.** Confirmed — one `pwrite` per finished
  block in `FileStorage`, chunk-granularity (not per-field) LZ4 calls
  in `CompressedStorage`.
- **Monomorphization over dyn dispatch.** Confirmed — `BlockStorage` is
  a type parameter everywhere; the one `Box<dyn SortedNeighbors>`
  exception is real and, on inspection, actually justified (it unifies
  genuinely different concrete iterator types behind one interface for
  a driver that isn't itself the hot loop).
- **Bit-iteration via `trailing_zeros`/`count_ones`.** Confirmed
  throughout `query_rec`/`build_rec`.
- **Disjoint-slice data parallelism, no shared mutable state.**
  Confirmed in `build_rec`'s `rayon` fan-out (`split_at_mut`, private
  `FlatBuf` per branch, stitched back sequentially).
- **`StackCacheEntry`'s three-reverted-designs history.** Read in full.
  The write-through-only design is the right one to keep; I'm not
  reopening the write-back question without a new, narrow, specifically
  motivated reason to.

No changes to any of the above. This is the part of the brief I'm
taking as given.

## 2. A documentation/code mismatch I found — worth correcting now

`examples/README.md` §6 lists "Not yet wired into the shared join/BFS
harness" as an honest weakness of the generic structure. That's stale:
`examples/common/sorted_neighbors.rs:253` already has

```rust
impl<S: BlockStorage> GraphBackend for RadixTreeDfsInterleavedVarSkipUnrolledGeneric<S> { ... }
```

— a real, generic `GraphBackend` impl, already wired for every
backend via monomorphization. The handoff doc's own §3 claim ("one
`GraphBackend` impl... wires all three backends") is the correct,
current state; the README just wasn't updated after that landed. I'll
port this impl into `src/` alongside the tree itself — there's no
reason to leave a working, already-written integration out of the
initial cut. I'd also flag the stale README line for a fix, separate
from this porting work.

## 3. The one item I'm treating as a blocker, not a style call

`insert()`'s only domain check is:

```rust
debug_assert!(self.in_bounds(s, p, o), "insert({s},{p},{o}) is outside the domain established at build time");
```

`debug_assert!` compiles out entirely in `--release` — the build mode
every benchmark in this project runs in, and the only mode a real
deployment would ship. Downstream, `compute_qi` does
`(coords[d] - lo[d]) >> child_cell_bits[d]` on `u64`s with no bounds
check. An out-of-domain coordinate can underflow that subtraction, and
the resulting `sub << shifts[d]` can shift by more than fits the
node's bit budget. Rust doesn't panic on this in release; a
too-large shift amount is silently masked mod 64. So today, calling
`insert()` on a `--release` build with one bad coordinate doesn't
error and doesn't panic — it silently writes into the wrong bit(s) of
a node word, i.e. real, silent tree corruption, with no signal to the
caller at all.

This isn't a hypothetical: it's the exact "malformed input at a
boundary" case §7 of the handoff already asks for `Result`-based
handling of, just not called out by name. I'm treating this as the
first thing to fix in the port — `insert`/`insert_batch` get a real
runtime domain check that returns `Result<_, TreeError>` (or an
`assert!` that isn't compiled away, if the team prefers panic-on-misuse
for this API — see below) instead of inheriting the `debug_assert!`.

## 4. Explicit calls on the handoff's open questions

**`src/radix64.rs` / `src/ksquare.rs`: delete, don't evolve.**
Confirmed `examples/common/` never imports either (README says so;
`grep` confirms it). `radix64.rs` is an incomplete sketch — `encode()`
allocates a `Vec` and returns it empty; `ksquare.rs` is a 9-line struct
with no method bodies at all. Neither shares a type, trait, or
function with the generic structure. Keeping them in `src/` next to
the real implementation would misrepresent them as part of the
production API surface. `radix64.rs`'s header comment is genuinely
good prior-art/lineage writing (Omnitree, k^d-trees, UB-trees, Morton
tries) — I'll fold the substance of it into the new module's
crate-level docs rather than lose it, then delete both files.

**Error handling: `Result` at real boundaries, panic on internal
invariants.** Concretely:
- A `TreeError` enum (hand-rolled `Display`/`std::error::Error` impls
  — no new dependency needed for this) covering: `Io(std::io::Error)`,
  `CorruptData(String)` (bad magic, truncated buffer, unreadable block
  table — everywhere `tree_serialize.rs` currently does
  `assert_eq!(&magic, MAGIC, ...)` or equivalent), and
  `OutOfDomain { dim, value, bound }` (§3 above).
- `deserialize`/`deserialize_from_path`/`deserialize_from_reader`
  return `Result<Self, TreeError>` instead of panicking on untrusted
  bytes.
- `insert`/`insert_batch` return `Result` (or a documented, real
  `assert!`) for domain violations — not silently degrade.
- Internal invariants that are unreachable through any correct public
  API call (e.g. `bytes.len() <= block_capacity` inside
  `append_record`, which build-time code already guarantees) stay
  `debug_assert!`/panic — turning *every* assertion in this file into
  a `Result` would bury real programmer-error signals in the same
  bucket as recoverable I/O/input problems, which the handoff itself
  warns against conflating.

**`unsafe` policy: keep "no unsafe" for the initial cut.** None of
`Heap`/`File`/`Compressed` use `unsafe` today, and the one concrete
data point gathered (lz4_flex's unsafe fast path: -18% to -25% latency
across all three query shapes on a single, admittedly-noisy run) is a
real cost against a policy the project has otherwise held to
consistently. I'm not spending the production port's first pass on an
unproven, single-run number. Recommendation: ship safe-only now; if a
real deployment profiles LZ4 decode as its bottleneck later, gate the
unsafe feature set behind an opt-in Cargo feature and re-measure
several runs before defaulting to it. Mmap/Arena backends stay out of
this pass entirely (explicitly scoped out already, not reassessed) —
natural `BlockStorage` extensions for later, not a redesign.

**Feature gating — one gap the handoff doc didn't mention.**
`Cargo.toml` currently has `rayon` as `[dev-dependencies]`-only, but
`build_rec`'s parallel fan-out (a real, measured part of the ~4x build
speedup in §5.1) is exactly the kind of thing a *production* bulk
build needs, not just examples/benchmarks. It has to move to a real,
optional dependency. My plan:
- `heap` (default): the tree + `HeapStorage`, zero new deps.
- `file`: adds `FileStorage`.
- `compressed`: adds `CompressedStorage` + `lz4_flex`.
- `parallel`: adds `rayon`, gates `build_rec`'s fan-out (falls back to
  the existing sequential path when off — the code already has that
  path, it's just always-parallel-eligible today).

**Serialization versioning.** Agree with the handoff: add a version
byte after `MAGIC`, and make an unrecognized version a `TreeError`
variant (`UnsupportedVersion(u8)`) instead of folding version checking
into the same `assert_eq!` as magic-mismatch, so a future format change
has somewhere to branch.

**Test layout.** Agree the ~26x duplication is an `examples/`-only
artifact of one test module getting pulled in by 26 separate binaries
via `#[path]`; it doesn't survive the move by construction, since
`src/` is one crate with one copy of each module. I'll carry the
existing coverage (cross-backend serialize round-trip, `NaiveIndex`
cross-checks at multiple block capacities, parallel-fan-out and
bounded-build correctness) over as a single `#[cfg(test)]` suite (or
`tests/`), not re-architect the tests themselves.

## 5. What I'm explicitly not doing in this pass

- Not touching `MmapStorage`/`ArenaStorage`, SIMD bitmap batching,
  prefetching, or `#[repr]` tuning — real backlog items per the
  handoff's own framing (§4's "open opportunities"), none measured
  yet, none blocking a correct production `Heap`/`File`/`Compressed`
  release.
- Not fixing the bounded-build "hot spine" re-bucketing cost — flagged
  as known, not regressed by anything I'm doing.
- Not adding domain-growing `insert()` — out of scope per the handoff,
  and a materially larger design problem than this pass.

## 6. Implementation report

The port is done. `src/` module layout:

| Module | Contents |
|---|---|
| `error.rs` | `TreeError`, `OutOfDomain`, `Result` alias |
| `storage.rs` | `BlockStorage` trait, `Pos`, `HeapStorage` (feature `heap`) |
| `storage_file.rs` | `FileStorage` (feature `file`) |
| `storage_compressed.rs` | `CompressedStorage` (feature `compressed`) |
| `serialize.rs` | Wire format, `Result`-based, versioned |
| `tree.rs` | Schedule/build/bounded-build internals + the generic tree + its test suite |
| `graph.rs` | `SortedNeighbors`/`GraphBackend` + `ltj_run` (leapfrog join) |
| `path.rs` | `AdjacencyBackend` + `bfs_sequential`/`bfs_parallel` |
| `test_support.rs` | `#[cfg(test)]`-only dataset generation + `NaiveIndex` oracle |
| `lib.rs` | Crate docs, feature declarations, public re-exports |

`src/radix64.rs`/`src/ksquare.rs` are deleted, as decided in §4.

**Feature matrix, each built and tested independently**: `--features
heap` (default), `--no-default-features --features file`,
`--no-default-features --features compressed`,
`--no-default-features --features heap,parallel`, and
`--all-features` — all compile clean and pass their full test suite
(28 tests on default features, 32 with `--all-features`; the extra 4
are the `file`/`compressed`-specific stress test and the
cross-backend serialize round-trip, which need both features
present). `cargo build --no-default-features` (zero backends) also
compiles, since the bounded/external-build machinery and the shared
tree algorithm don't depend on any one backend. `examples/` was left
untouched and still builds unmodified — its `rayon`/`libc`/`lz4_flex`
dev-dependencies are separate from the new optional `[dependencies]`
entries `src/` uses.

**§3's finding, now a regression test.** `insert_rejects_out_of_domain_coordinate`
reconstructs the exact repro from that section (a tiny tree, an
out-of-domain `s`) and asserts both that `insert()` now returns
`Err(TreeError::OutOfDomain)` and that every existing query result is
byte-for-byte unchanged afterward — the second assertion is the one
that would have caught the original bug (returning an error alone
doesn't prove nothing was corrupted first).
`insert_batch_rejects_out_of_domain_atomically` covers the batch path's
all-or-nothing contract, and `build_with_order_rejects_oversized_split`
covers the second instance of the same bug class found during the
port (§ below).

**A second instance of the same bug class, found while porting.**
`build`/`build_with_order` take a caller-supplied `split: [u8; 3]`
array with the same property `insert`'s domain check had: a
`debug_assert!` was the only guard against `split` summing to more
than 6 bits, which — compiled out in release — would silently produce
a schedule that overflows a node's 64-way branching word the same way
the domain bug did. Fixed the same way: a real check, a new
`TreeError::InvalidConfig` variant, and a test
(`build_with_order_rejects_oversized_split`). `build_dynamic*`'s
schedule is computed automatically and can't hit this, so those stay
infallible.

**What stayed a panic, and why (§3 of the original handoff's ask).**
Beyond deserialization and the two `Result`-worthy cases above,
everything else keeps panicking, on purpose:
- Mid-traversal I/O failures against an already-built `File`/
  `Compressed` tree (a `pread`/`pwrite` genuinely failing on disk
  full, EIO, revoked permissions). There's no valid partial state to
  hand back from inside a tree traversal, and `Heap` — the default
  backend — can never hit this class at all, so threading `Result`
  through `contains`/`query_two_bound`/`query_one_bound`/`insert`'s
  hot paths for an error only two of three backends can produce
  seemed like the wrong trade. Documented in `storage_file.rs`'s own
  module doc comment.
- Internal invariants the build path already guarantees by
  construction (a node record never exceeding `block_capacity`, a
  `build_pending` block existing when `patch_appended` looks for it).
  These are unreachable through any correct call sequence, so they
  stay `debug_assert!`/`.expect()`, the same class as before — turning
  every one of these into `Result` would bury the two real,
  user-triggerable cases in the same bucket as programmer-error
  signals.
- `block_capacity` too small (`< WORD_RECORD_BYTES + MAX_SKIP_RECORD_BYTES`)
  stays a `debug_assert!`, deliberately not promoted like `split` was:
  unlike `split`'s bit-overflow (which corrupts a *different* triple's
  bit), a too-small capacity is absorbed by `Vec`'s own reallocation
  — worse block packing, not wrong query results. Flagging this
  explicitly rather than silently leaving it inconsistent with the
  `split` fix.

**Performance: no measured regression, one measured win.** Built a
scratch benchmark (`s_graph::Heap` directly, 5M synthetic triples,
2M×50×2M domain, not committed — mirrors the `examples/` harness'
own methodology) rather than assume parity:

| | this port (no `parallel`) | this port (`parallel` on) |
|---|---|---|
| Build time | 3.52s | 0.95s (3.7x) |
| `mem_size_bytes` | 172.85 MiB | 172.85 MiB (byte-identical) |
| `contains_avg` | 4.68µs | 4.56µs |
| `insert_avg` | 24.7µs | 22.4µs |

The `parallel`-feature speedup (3.7x at 5M triples) lines up with the
handoff's own measured ~4x at 20M; `mem_size_bytes` being exactly
identical with the feature on/off is a real correctness signal (same
tree regardless of fan-out), not just a performance one.  `contains`/
`insert` latency are consistent with the `examples/` prototype's own
numbers at comparable scale (§5.1 of the original assessment measured
7-8µs/17-27µs at 20M triples; this port's 5M-triple numbers land in
the same neighborhood, scaling as expected).

**Test coverage.** 28 tests (32 with `--all-features`) in one
`#[cfg(test)] mod tests` per module — no `examples/`-style ~26x
duplication, since this is one crate now, not 26 example binaries
pulling in a shared file. Carries over the original suite's coverage
(cross-backend serialize round-trip, `NaiveIndex` cross-checks across
block capacities, small-capacity stress tests, parallel-fan-out and
bounded-build correctness) plus new tests for every `Result`-returning
boundary this pass added (domain rejection + non-corruption, batch
atomicity, bad `split`, bad magic, truncated input, unsupported
version).

**Known gaps in this pass, not silently dropped:**
- The join (`graph.rs`)/BFS (`path.rs`) test suites are smaller than
  the originals' — they check the tree against `NaiveIndex` directly
  rather than against three other structure implementations (which
  don't exist in `src/`, only in `examples/`), so the "do independent
  implementations agree" cross-check the original suite had is
  necessarily narrower here.
- `cargo clippy --lib --all-features` has 7 remaining warnings
  (`needless_range_loop`/`explicit_counter_loop` on per-level array
  indexing, one `chunks_exact` suggestion, one complex-type
  suggestion) — all pre-existing style choices carried over verbatim
  from the exploratory code's own hot loops. Left alone rather than
  "clippy-fixed" without a measurement, per this project's own
  optimization-claims-need-a-benchmark standard.
- `cargo doc` has ~19 warnings, all either "public docs link to a
  private item" (renders as plain text, not broken) or a redundant
  explicit link target — cosmetic, not fixed given the time budget.

## 7. Migration notes for anyone using `examples/common/radixtree_dfs_interleaved_varskip_unrolled_generic.rs` directly

- **Crate, not module**: `use s_graph::{Heap, File, Compressed}` behind
  `heap`/`file`/`compressed` features (`heap` on by default) instead of
  `#[path] mod common;` plus `common::radixtree_dfs_interleaved_varskip_unrolled_generic::{Heap, File, Compressed}`.
- **`insert`/`insert_batch` now return `Result`**: `tree.insert(s, p, o)?`
  or `.unwrap()` instead of a bare `bool`/`u64`. `Ok(true)`/`Ok(false)`
  mean exactly what the old `true`/`false` did (newly inserted /
  already existed); `Err` is new and means the input was rejected, not
  "no-op."
- **`deserialize`/`deserialize_from_path` now return `Result`**: `?` or
  `.unwrap()` instead of a bare `Self`.
- **`build`/`build_with_order` now return `Result`** (the `split`-sum
  validation in §6); `build_dynamic*` are unchanged (still infallible).
- **`rayon`-based parallel bulk-build fan-out is opt-in** via the
  `parallel` feature now, not always-on. A caller who wants the
  measured ~4x build speedup at large N needs to enable it explicitly.
- **`RadixTreeApi`/`MemSize` trait wrappers are gone**: `contains`,
  `query_two_bound`, `query_one_bound`, `predicates_for`,
  `node_word_count`, `domain_size`, `height`, `mem_size_bytes` are all
  plain inherent methods now (they always were the real implementation
  behind those trait impls; only the redundant delegating wrapper was
  dropped — those traits existed to let `examples/compare` dispatch
  generically over many different structure types, which `src/` has no
  need for, shipping only one).
- **New**: `in_bounds(s, p, o) -> bool` is now `pub`, for callers who
  want to pre-filter a batch rather than handle a `Result` per
  candidate.
- **Serialization format gained a version byte** (right after the
  4-byte magic). Bytes produced by the `examples/` prototype (no
  version byte) will not deserialize with this crate — there was no
  installed base to preserve compatibility with, so this wasn't a
  live concern, but worth stating plainly rather than leaving implicit.
