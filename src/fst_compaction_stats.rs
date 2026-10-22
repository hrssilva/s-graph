//! Diagnostics shared by the default compactor and experimental comparisons.

#[derive(Clone, Copy, Debug, Default)]
pub struct BlockCompactionStats {
    pub input_states: usize,
    pub reachable_states: usize,
    pub reachable_transitions: usize,
    pub source_blocks: usize,
    pub released_blocks: usize,
    pub relocation_bytes: usize,
    pub rank_bytes: usize,
    /// Explicit temporary heap allocations, excluding output block buffers,
    /// backend caches,
    /// allocator bookkeeping, recursion stack and output block descriptors.
    pub workspace_bytes: usize,
    pub input_uncompressed_block_bytes: usize,
    /// Used source-block bytes plus written output bytes; excludes arena,
    /// capacities and compressed-vs-uncompressed differences. Not process RSS.
    pub peak_uncompressed_block_bytes: usize,
    pub output_record_bytes: usize,
    /// Phase timings for local comparisons; excludes final graph construction.
    pub mark_ns: u64,
    pub planning_ns: u64,
    pub copy_ns: u64,
    /// Depth compactor diagnostics (zero for older algorithms).
    pub frontier_bytes: usize,
    pub scratch_bytes: usize,
    pub peak_frontier_states: usize,
    pub peak_relocation_states: usize,
    pub source_block_reads: usize,
    pub output_fixup_blocks: usize,
    pub frontier_buffer_growths: usize,
    pub relocation_buffer_growths: usize,
    pub fixup_ns: u64,
    /// Rank lookup counts for the depth/streaming comparison.
    pub rank_lookups: usize,
    /// Maximum retained block relocation runs (streaming compactor).
    pub relocation_runs: usize,
    /// Live edit states forwarded using the existing arena (streaming).
    pub forwarded_edit_states: usize,
}
