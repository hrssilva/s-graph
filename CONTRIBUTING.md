# Development and merge checks

Use the production modules in `src/` for engine changes. Historical prototype
sources and measured outputs live under `examples/join/`; the comparison harness
copies current production code into a disposable crate.

The default update strategy is `UpdateLookup::Sampled(1)`. Consuming `freeze()`
and `compact_streaming()` use rank-free BFS compaction. Comparison-only
`compact_blocks()`, `compact_depths()` and `compact_recursive()`, plus the
reserved-block interface, require the opt-in `experimental` feature.

Run feature checks before merging:

```sh
cargo check --locked --no-default-features --all-targets
cargo check --locked --all-targets
cargo check --locked --no-default-features --features file --all-targets
cargo check --locked --no-default-features --features compressed --all-targets
cargo check --locked --all-features --all-targets
```

Run the production tests and README examples in default and all-feature modes:

```sh
cargo test --locked --lib --tests
cargo test --locked --doc
cargo test --locked --all-features --lib --tests
cargo test --locked --all-features --doc
```

Use `--offline` when the dependencies are already cached. `cargo check
--all-targets` compiles examples without executing benchmark programs, some of
which build large fixtures. Run performance measurements separately from builds
and tests; record the binary/source hashes, feature flags and containment limits.

Verify the distributable crate:

```sh
cargo package --locked --list
cargo package --locked
```

During uncommitted development, add `--allow-dirty`; review `git status` before
committing. Raw benchmark
results and lab captures are excluded from the package; runtime sources, examples,
tests, license and documentation remain included.

Benchmark evidence is indexed in
[examples/join/RESULTS.md](examples/join/RESULTS.md).
Keep recorded output unchanged so manifests remain meaningful. Generated data is
collapsed in supporting review tools; whitespace inside raw captures and patch
context is preserved. Source/test/documentation whitespace remains checked.
Historical API labels in old measurements are retained and explained by the
current guides. Summaries should distinguish retained allocations from RSS,
preparation from traversal, and completed work from budget-limited probes.
