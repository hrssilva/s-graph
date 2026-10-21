//! FST snapshot v1: SGF1, version:u8, domains:3*u64, block_capacity:u64,
//! visited_count:u64, node_count:u64, height:u32, level_counts:height*u64,
//! block_table:(count:u32, (used:u32,used:u32)*), then packed block contents.
//! Integers are little endian; schedules and addresses are reconstructed.
//! Validation is over physical states, never the exponentially larger set of
//! possible root-to-leaf walks. No address directory is retained after loading.
use super::*;
use crate::serialize::{DeserializeLimits, read_u32_from, read_u64_from};
use crate::{Result, TreeError};
use std::io::{Read, Write};

fn corrupt(message: &str) -> TreeError {
    TreeError::CorruptData(message.into())
}

impl PathFstDeltaPow2Bfs {
    /// Versioned FST snapshot. Mutable-tree SGZ1 snapshots have a separate format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.serialize_into(&mut bytes)
            .expect("writing into Vec cannot fail");
        bytes
    }

    /// Stream a snapshot without buffering a second copy of graph storage.
    pub fn serialize_into<W: Write>(&self, out: &mut W) -> Result<()> {
        out.write_all(b"SGF1\x01")?;
        for n in self.dimension_sizes {
            out.write_all(&n.to_le_bytes())?;
        }
        for n in [
            self.block_capacity as u64,
            self.visited_count,
            self.node_count,
        ] {
            out.write_all(&n.to_le_bytes())?;
        }
        out.write_all(&(self.height as u32).to_le_bytes())?;
        for &count in &self.level_node_counts {
            out.write_all(&(count as u64).to_le_bytes())?;
        }
        let table = self.storage.block_table();
        out.write_all(&(table.len() as u32).to_le_bytes())?;
        for &(_, len) in &table {
            out.write_all(&(len as u32).to_le_bytes())?;
            out.write_all(&(len as u32).to_le_bytes())?;
        }
        for (block, &(_, len)) in table.iter().enumerate() {
            out.write_all(&self.storage.bulk_read(block, 0, len))?;
        }
        Ok(())
    }

    pub fn serialize_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.serialize_into(&mut out)?;
        out.flush()?;
        Ok(())
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        Self::deserialize_with_limits(bytes, DeserializeLimits::default())
    }

    pub fn deserialize_with_limits(bytes: &[u8], limits: DeserializeLimits) -> Result<Self> {
        let mut input = std::io::Cursor::new(bytes);
        let graph = Self::deserialize_from_with_limits(&mut input, limits)?;
        if input.position() != bytes.len() as u64 {
            return Err(corrupt("trailing bytes after FST snapshot"));
        }
        Ok(graph)
    }

    /// Read one snapshot, leaving subsequent stream data to the caller.
    pub fn deserialize_from<R: Read>(input: &mut R) -> Result<Self> {
        Self::deserialize_from_with_limits(input, DeserializeLimits::default())
    }

    /// Load with explicit limits on reserved block bytes, block size and count.
    /// Validation uses additional temporary memory proportional to physical
    /// states and transitions; limits bound stored bytes, not total process RAM.
    pub fn deserialize_from_with_limits<R: Read>(
        input: &mut R,
        limits: DeserializeLimits,
    ) -> Result<Self> {
        let mut magic = [0; 4];
        input.read_exact(&mut magic)?;
        if &magic != b"SGF1" {
            return Err(corrupt("bad FST magic"));
        }
        let mut version = [0];
        input.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(TreeError::UnsupportedVersion(version[0]));
        }
        let mut dimension_sizes = [0; 3];
        for n in &mut dimension_sizes {
            *n = read_u64_from(input)?;
        }
        if !dimension_sizes.iter().all(|n| (1..=1u64 << 32).contains(n)) {
            return Err(corrupt("invalid FST domains"));
        }
        let capacity = read_u64_from(input)?;
        let block_capacity =
            usize::try_from(capacity).map_err(|_| corrupt("FST capacity overflow"))?;
        if !block_capacity.is_power_of_two() || !(512..=u32::MAX as usize).contains(&block_capacity)
        {
            return Err(corrupt("invalid FST block capacity"));
        }
        let visited_count = read_u64_from(input)?;
        let node_count = read_u64_from(input)?;
        let total_bits = total_bits_needed(dimension_sizes);
        let mut level_dim_bits = dynamic_schedule(total_bits);
        if level_dim_bits.is_empty() {
            level_dim_bits.push([0; 3]);
        }
        let height = read_u32_from(input)? as usize;
        if height != level_dim_bits.len() {
            return Err(corrupt("FST height disagrees with domains"));
        }
        let mut counts = Vec::with_capacity(height);
        let mut total = 0u64;
        for _ in 0..height {
            let count = read_u64_from(input)?;
            total = total
                .checked_add(count)
                .ok_or_else(|| corrupt("FST node count overflow"))?;
            counts.push(usize::try_from(count).map_err(|_| corrupt("FST level count overflow"))?);
        }
        if total != node_count || counts[0] != 1 {
            return Err(corrupt("invalid FST state counts"));
        }
        let table = crate::serialize::read_block_table_limited(input, limits)?;
        let stored_bytes: u64 = table.iter().map(|&(_, len)| len as u64).sum();
        if table.is_empty()
            || table
                .iter()
                .any(|&(cap, len)| len == 0 || cap != len || len > block_capacity)
            || node_count > stored_bytes / 8
            || visited_count < node_count
        {
            return Err(corrupt("invalid FST blocks or node count"));
        }
        let storage = HeapStorage::read_blocks_from(input, &table)?;
        let level_shifts_cache: Vec<[u8; 3]> = level_dim_bits
            .iter()
            .map(|&bits| level_shifts(bits, [0, 1, 2]))
            .collect();
        let mut remaining = total_bits;
        let level_child_cell_bits_cache: Vec<[u8; 3]> = level_dim_bits
            .iter()
            .map(|bits| {
                for d in 0..3 {
                    remaining[d] -= bits[d];
                }
                remaining
            })
            .collect();
        let scan_layout = (0..height)
            .map(|l| {
                crate::fst::ScanLayout::new(
                    level_dim_bits[l],
                    level_shifts_cache[l],
                    level_child_cell_bits_cache[l],
                )
            })
            .collect();
        let graph = Self {
            storage,
            root_virtual_addr: 0,
            height,
            level_dim_bits,
            level_shifts_cache,
            level_child_cell_bits_cache,
            dimension_sizes,
            block_capacity,
            capacity_bits: block_capacity.trailing_zeros(),
            capacity_mask: (block_capacity - 1) as u64,
            node_count,
            level_node_counts: counts,
            scan_layout,
            visited_count,
            level_addrs: std::sync::OnceLock::new(),
        };
        graph.validate_snapshot(&graph.level_node_counts)?;
        Ok(graph)
    }

    pub fn deserialize_from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::deserialize_from_path_with_limits(path, DeserializeLimits::default())
    }
    pub fn deserialize_from_path_with_limits(
        path: impl AsRef<std::path::Path>,
        limits: DeserializeLimits,
    ) -> Result<Self> {
        Self::deserialize_from_with_limits(
            &mut std::io::BufReader::new(std::fs::File::open(path)?),
            limits,
        )
    }

    fn validate_snapshot(&self, counts: &[usize]) -> Result<()> {
        let mut levels = Vec::with_capacity(self.height);
        let (mut block, mut offset) = (0, 0);
        // Parse every record boundary before dereferencing any delta.
        for (level, &count) in counts.iter().enumerate() {
            let mut addresses = Vec::new();
            let slots = 1u32 << self.level_dim_bits[level].iter().sum::<u8>();
            let valid = if slots == 64 {
                u64::MAX
            } else {
                (1u64 << slots) - 1
            };
            for _ in 0..count {
                if block < self.storage.block_count() && offset == self.storage.block_len(block) {
                    block += 1;
                    offset = 0;
                }
                if block >= self.storage.block_count() || self.storage.block_len(block) - offset < 8
                {
                    return Err(corrupt("truncated FST record"));
                }
                let word = self.storage.read_u64(block, offset);
                if word & !valid != 0 || (word == 0 && (level != 0 || self.node_count != 1)) {
                    return Err(corrupt("invalid FST occupancy"));
                }
                let len = 8 + if level + 1 < self.height {
                    4 * word.count_ones() as usize
                } else {
                    0
                };
                if len > self.storage.block_len(block) - offset {
                    return Err(corrupt("FST record crosses a block boundary"));
                }
                addresses.push(((block as u64) << self.capacity_bits) | offset as u64);
                offset += len;
            }
            levels.push(addresses);
        }
        if block + 1 != self.storage.block_count() || offset != self.storage.block_len(block) {
            return Err(corrupt("unparsed FST record bytes"));
        }
        // Bottom-up maxima prove every decoded coordinate lies in its domain.
        // Also count logical node visits and require all physical states reachable.
        let mut tails: Vec<([u64; 3], u64, u64)> = Vec::new();
        for level in (0..self.height).rev() {
            let mut seen = vec![false; tails.len()];
            let mut current = Vec::with_capacity(levels[level].len());
            for &addr in &levels[level] {
                let mut word = self.read_word(addr);
                let (bits, shifts, remaining) = self.level_schedule(level);
                let (mut maxima, mut visits, mut matches, mut i) = ([0u64; 3], 1u64, 0u64, 0);
                while word != 0 {
                    let slot = word.trailing_zeros();
                    word &= word - 1;
                    let suffix = if level + 1 < self.height {
                        let target = addr
                            .checked_add(self.read_delta(addr, i))
                            .ok_or_else(|| corrupt("FST delta overflow"))?;
                        let index = levels[level + 1]
                            .binary_search(&target)
                            .map_err(|_| corrupt("FST child is not a next-level record"))?;
                        seen[index] = true;
                        visits = visits
                            .checked_add(tails[index].1)
                            .ok_or_else(|| corrupt("FST logical size overflow"))?;
                        matches = matches
                            .checked_add(tails[index].2)
                            .ok_or_else(|| corrupt("FST triple count overflow"))?;
                        tails[index].0
                    } else {
                        matches += 1;
                        [0; 3]
                    };
                    for d in 0..3 {
                        let value = ((((slot >> shifts[d]) & ((1u32 << bits[d]) - 1)) as u64)
                            << remaining[d])
                            | suffix[d];
                        maxima[d] = maxima[d].max(value);
                    }
                    i += 1;
                }
                current.push((maxima, visits, matches));
            }
            if seen.contains(&false) {
                return Err(corrupt("unreachable FST state"));
            }
            tails = current;
        }
        if tails[0].1 != self.visited_count
            || (0..3).any(|d| tails[0].0[d] >= self.dimension_sizes[d])
        {
            return Err(corrupt(
                "invalid FST logical count or out-of-domain coordinate",
            ));
        }
        Ok(())
    }
}
