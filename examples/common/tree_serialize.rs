//! Shared binary serialization format for the insertable block-list
//! variants (`radixtree_dfs_interleaved_varskip_unrolled{,_mmap,_file,_file_mmap}`).
//!
//! Unlike this project's usual per-variant self-containment (each
//! algorithm file duplicates its own small helpers so it can evolve
//! independently -- see e.g. `read_u64`/`push_var` copied across every
//! variant), the serialization *format* genuinely needs to be one shared
//! contract: it exists so a structure built in one process can be
//! written to a file and reconstructed, byte-for-byte equivalently, in a
//! completely different process -- see `examples/phase_*.rs`, which use
//! this to separate a structure's build/query/insert phases into
//! isolated processes so each phase's peak RSS reflects *only* that
//! phase, not the others (building peak RSS, in this project's own
//! `rss_*.rs` measurements, is otherwise dominated by build-time
//! transients that have nothing to do with query or insert behavior).
//!
//! # Format
//!
//! ```text
//! magic: 4 bytes "SGZ1"
//! height: u32
//! node_count: u64
//! block_capacity: u64
//! dimension_sizes: 3x u64
//! dim_order: 3x u8
//! single_point_exists: u8 (0 or 1)
//! num_levels: u32
//!   per level: level_dim_bits (3x u8), level_skip_bytes (u8),
//!              level_shifts_cache (3x u8), level_child_cell_bits_cache (3x u8)
//! num_blocks: u32
//!   per block: capacity: u32, used_len: u32
//! content: concatenation of each block's `used_len` bytes, in order
//! ```
//!
//! Every variant's own storage backend (`Vec<Vec<u8>>`, `Vec<MmapBlock>`,
//! on-disk slots, mmap'd slots) is opaque to this module -- each
//! variant's own `serialize`/`deserialize` methods extract/reconstruct
//! their own block representation from the `(capacity, used_len)` table
//! and the flat content this module reads/writes; this module only
//! handles the format-level plumbing common to all of them.
//!
//! # Reading is streamed, not buffered
//!
//! [`read_header_from`]/[`read_block_table_from`] take `impl Read`, not a
//! `&[u8]`, and every variant's own `deserialize_from_path` opens the
//! serialized file and reads directly from it (`BufReader<File>`) -- the
//! first version of this format instead had every variant's
//! `examples/phase_*.rs` binary do one `std::fs::read` of the *whole*
//! serialized file into a `Vec<u8>` before calling a buffer-based
//! `deserialize(&buf)`. That was measured to hold the whole serialized
//! structure resident *twice* during deserialize for every variant
//! (`buf`, plus whatever each variant's own `deserialize` built from it)
//! -- for the RAM-backed variants (heap, anonymous mmap) that meant
//! deserialize's own peak RSS was ~2x the structure's real size for no
//! reason, and for the disk-backed variants (file, file-mmap) it meant
//! paying a full in-memory copy of content that was never supposed to be
//! RAM-resident in the first place. Streaming directly from the file
//! removes that buffer entirely: each variant reads only as much as it
//! needs to place one block at a time (a `Vec<u8>` no bigger than
//! `block_capacity`, reused block to block), so deserialize's own peak
//! RSS now actually reflects what that variant's storage model costs,
//! not an accidental doubling. `deserialize(buf: &[u8])` (the buffer-based
//! entry point, kept for tests and for anyone who already has the bytes
//! in memory) is now a thin wrapper over the same streaming logic via
//! `std::io::Cursor`, so there is exactly one implementation to keep
//! correct, not two.

pub const MAGIC: &[u8; 4] = b"SGZ1";

pub struct Header {
    pub height: u32,
    pub node_count: u64,
    pub block_capacity: usize,
    pub dimension_sizes: [u64; 3],
    pub dim_order: [usize; 3],
    pub single_point_exists: bool,
    pub level_dim_bits: Vec<[u8; 3]>,
    pub level_skip_bytes: Vec<u8>,
    pub level_shifts_cache: Vec<[u8; 3]>,
    pub level_child_cell_bits_cache: Vec<[u8; 3]>,
}

pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_bytes<R: std::io::Read, const N: usize>(r: &mut R) -> [u8; N] {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).expect("unexpected EOF while reading a serialized tree -- truncated or corrupt file");
    buf
}

pub fn read_u32_from<R: std::io::Read>(r: &mut R) -> u32 {
    u32::from_le_bytes(read_bytes(r))
}

pub fn read_u64_from<R: std::io::Read>(r: &mut R) -> u64 {
    u64::from_le_bytes(read_bytes(r))
}

pub fn write_header(out: &mut Vec<u8>, h: &Header) {
    out.extend_from_slice(MAGIC);
    write_u32(out, h.height);
    write_u64(out, h.node_count);
    write_u64(out, h.block_capacity as u64);
    for d in h.dimension_sizes {
        write_u64(out, d);
    }
    for d in h.dim_order {
        out.push(d as u8);
    }
    out.push(h.single_point_exists as u8);
    write_u32(out, h.level_dim_bits.len() as u32);
    for i in 0..h.level_dim_bits.len() {
        out.extend_from_slice(&h.level_dim_bits[i]);
        out.push(h.level_skip_bytes[i]);
        out.extend_from_slice(&h.level_shifts_cache[i]);
        out.extend_from_slice(&h.level_child_cell_bits_cache[i]);
    }
}

pub fn read_header_from<R: std::io::Read>(r: &mut R) -> Header {
    let magic: [u8; 4] = read_bytes(r);
    assert_eq!(&magic, MAGIC, "bad magic: not a serialized radix tree, or wrong format version");
    let height = read_u32_from(r);
    let node_count = read_u64_from(r);
    let block_capacity = read_u64_from(r) as usize;
    let mut dimension_sizes = [0u64; 3];
    for d in &mut dimension_sizes {
        *d = read_u64_from(r);
    }
    let mut dim_order = [0usize; 3];
    for d in &mut dim_order {
        *d = read_bytes::<R, 1>(r)[0] as usize;
    }
    let single_point_exists = read_bytes::<R, 1>(r)[0] != 0;
    let num_levels = read_u32_from(r) as usize;
    let mut level_dim_bits = Vec::with_capacity(num_levels);
    let mut level_skip_bytes = Vec::with_capacity(num_levels);
    let mut level_shifts_cache = Vec::with_capacity(num_levels);
    let mut level_child_cell_bits_cache = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        level_dim_bits.push(read_bytes::<R, 3>(r));
        level_skip_bytes.push(read_bytes::<R, 1>(r)[0]);
        level_shifts_cache.push(read_bytes::<R, 3>(r));
        level_child_cell_bits_cache.push(read_bytes::<R, 3>(r));
    }
    Header { height, node_count, block_capacity, dimension_sizes, dim_order, single_point_exists, level_dim_bits, level_skip_bytes, level_shifts_cache, level_child_cell_bits_cache }
}

/// Writes the per-block `(capacity, used_len)` table. Capacity is
/// preserved (not just length) so a deserialized structure keeps the
/// same insert-time splitting behavior as the original -- reconstructing
/// with zero slack would force every post-deserialize insert into an
/// immediate split.
pub fn write_block_table(out: &mut Vec<u8>, capacities: &[usize], used_lens: &[usize]) {
    assert_eq!(capacities.len(), used_lens.len());
    write_u32(out, capacities.len() as u32);
    for i in 0..capacities.len() {
        write_u32(out, capacities[i] as u32);
        write_u32(out, used_lens[i] as u32);
    }
}

pub fn read_block_table_from<R: std::io::Read>(r: &mut R) -> Vec<(usize, usize)> {
    let n = read_u32_from(r) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let cap = read_u32_from(r) as usize;
        let len = read_u32_from(r) as usize;
        out.push((cap, len));
    }
    out
}
