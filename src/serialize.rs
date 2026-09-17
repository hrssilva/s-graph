//! Shared binary serialization format for every [`crate::storage::BlockStorage`]
//! backend: a structure built via any one backend deserializes correctly
//! into any other, since the format round-trips through plain
//! uncompressed bytes regardless of which backend produced or consumes
//! it.
//!
//! # Format
//!
//! ```text
//! magic: 4 bytes "SGZ1"
//! version: u8
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
//! # This is an untrusted-input boundary
//!
//! Unlike the rest of this crate's read/write API (see the crate-level
//! docs' "Error handling" section), every function here that reads from
//! an `impl Read` returns [`crate::error::Result`] rather than panicking:
//! the bytes may come from a truncated file, a corrupted transfer, or
//! simply the wrong format entirely, and none of that is a programmer
//! error the caller could have prevented by construction.
//!
//! # Reading is streamed, not buffered
//!
//! [`read_header_from`]/[`read_block_table_from`] take `impl Read`, not a
//! `&[u8]`, and each backend's own `deserialize_from_path` opens the
//! serialized file and reads directly from it -- avoiding holding the
//! whole serialized structure resident twice during deserialize (once as
//! a read buffer, once as whatever the backend builds from it).
//! `deserialize(buf: &[u8])` is a thin wrapper over the same streaming
//! logic via `std::io::Cursor`.

use crate::error::{Result, TreeError};

pub const MAGIC: &[u8; 4] = b"SGZ1";
/// Bumped whenever the wire format changes incompatibly. A version this
/// build doesn't recognize is a [`TreeError::UnsupportedVersion`], not a
/// silent misparse.
pub const FORMAT_VERSION: u8 = 1;

/// Resource limits for untrusted snapshots. Increase explicitly for larger graphs.
#[derive(Clone, Copy, Debug)]
pub struct DeserializeLimits {
    pub max_blocks: usize,
    pub max_block_bytes: usize,
    /// Includes reserved block capacity, not only used bytes.
    pub max_bytes: u64,
}

impl Default for DeserializeLimits {
    fn default() -> Self {
        Self { max_blocks: 1_000_000, max_block_bytes: 16 * 1024 * 1024, max_bytes: 1024 * 1024 * 1024 }
    }
}

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

fn read_bytes<R: std::io::Read, const N: usize>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            TreeError::CorruptData("unexpected end of input while reading a serialized tree -- truncated or corrupt".to_string())
        } else {
            TreeError::Io(e)
        }
    })?;
    Ok(buf)
}

pub fn read_u32_from<R: std::io::Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(r)?))
}

pub fn read_u64_from<R: std::io::Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_bytes(r)?))
}

pub fn write_header(out: &mut Vec<u8>, h: &Header) {
    out.extend_from_slice(MAGIC);
    out.push(FORMAT_VERSION);
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

pub fn read_header_from<R: std::io::Read>(r: &mut R) -> Result<Header> {
    let magic: [u8; 4] = read_bytes(r)?;
    if &magic != MAGIC {
        return Err(TreeError::CorruptData("bad magic: not a serialized s-graph tree".to_string()));
    }
    let version = read_bytes::<R, 1>(r)?[0];
    if version != FORMAT_VERSION {
        return Err(TreeError::UnsupportedVersion(version));
    }
    let height = read_u32_from(r)?;
    let node_count = read_u64_from(r)?;
    let block_capacity = usize::try_from(read_u64_from(r)?).map_err(|_| TreeError::CorruptData("block capacity overflows usize".into()))?;
    let mut dimension_sizes = [0u64; 3];
    for d in &mut dimension_sizes {
        *d = read_u64_from(r)?;
    }
    let mut dim_order = [0usize; 3];
    for d in &mut dim_order {
        *d = read_bytes::<R, 1>(r)?[0] as usize;
    }
    let single_point_exists = match read_bytes::<R, 1>(r)?[0] {
        0 => false,
        1 => true,
        _ => return Err(TreeError::CorruptData("invalid point flag".into())),
    };
    let num_levels = read_u32_from(r)? as usize;
    if num_levels > crate::tree::MAX_HEIGHT || height as usize != num_levels {
        return Err(TreeError::CorruptData("invalid level count or height".into()));
    }
    let mut level_dim_bits = Vec::with_capacity(num_levels);
    let mut level_skip_bytes = Vec::with_capacity(num_levels);
    let mut level_shifts_cache = Vec::with_capacity(num_levels);
    let mut level_child_cell_bits_cache = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        level_dim_bits.push(read_bytes::<R, 3>(r)?);
        level_skip_bytes.push(read_bytes::<R, 1>(r)?[0]);
        level_shifts_cache.push(read_bytes::<R, 3>(r)?);
        level_child_cell_bits_cache.push(read_bytes::<R, 3>(r)?);
    }
    Ok(Header { height, node_count, block_capacity, dimension_sizes, dim_order, single_point_exists, level_dim_bits, level_skip_bytes, level_shifts_cache, level_child_cell_bits_cache })
}

/// Writes the per-block `(capacity, used_len)` table. Capacity is
/// preserved (not just length) so a deserialized structure keeps the
/// same insert-time splitting behavior as the original.
pub fn write_block_table(out: &mut Vec<u8>, capacities: &[usize], used_lens: &[usize]) {
    debug_assert_eq!(capacities.len(), used_lens.len());
    write_u32(out, capacities.len() as u32);
    for i in 0..capacities.len() {
        write_u32(out, capacities[i] as u32);
        write_u32(out, used_lens[i] as u32);
    }
}

pub fn read_block_table_from<R: std::io::Read>(r: &mut R) -> Result<Vec<(usize, usize)>> {
    read_block_table_limited(r, DeserializeLimits::default())
}

pub(crate) fn read_block_table_limited<R: std::io::Read>(r: &mut R, limits: DeserializeLimits) -> Result<Vec<(usize, usize)>> {
    let n = read_u32_from(r)? as usize;
    if n > limits.max_blocks { return Err(TreeError::CorruptData("block count exceeds limit".into())); }
    // Do not reserve an attacker-controlled count before reading its entries.
    let mut out = Vec::new();
    let mut total = 0u64;
    for _ in 0..n {
        let cap = read_u32_from(r)? as usize;
        let len = read_u32_from(r)? as usize;
        total = total.checked_add(cap as u64).ok_or_else(|| TreeError::CorruptData("capacity overflow".into()))?;
        if cap == 0 || cap < len || cap > limits.max_block_bytes || total > limits.max_bytes {
            return Err(TreeError::CorruptData("invalid block length/capacity or resource limit exceeded".into()));
        }
        out.push((cap, len));
    }
    Ok(out)
}
