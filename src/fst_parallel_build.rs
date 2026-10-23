//! Bounded parallel prefix construction; global exact minimization before packing.
use super::*;
use rayon::prelude::*;

type Register = Vec<HashMap<(u64, Vec<u32>), u32>>;
const MIN_PARTITION: usize = 32_768;

struct Schedule {
    bits: Vec<[u8; 3]>,
    shifts: Vec<[u8; 3]>,
    remaining: Vec<[u8; 3]>,
}

struct Partition<'a> {
    points: &'a mut [Triple],
    scratch: &'a mut [Triple],
    lo: [u64; 3],
}

struct Entry<'a> {
    partition: Option<Partition<'a>>,
    level: usize,
    word: u64,
    children: Vec<usize>,
}

struct Local {
    nodes: Vec<Vec<BfsNodeMeta>>,
    root: u32,
    visited: u64,
}

impl<S: BlockStorage> PackedFst<S> {
    /// Experimental parallel bulk construction. Uses the current Rayon pool.
    /// Private prefix DAGs are merged exactly, preserving cross-prefix sharing.
    /// Packing remains sequential. Requires `parallel` and `experimental`.
    pub fn try_build_parallel(
        mut triples: Vec<Triple>,
        dimension_sizes: [u64; 3],
        block_capacity: usize,
    ) -> crate::Result<Self> {
        validate_build(&triples, dimension_sizes, block_capacity)?;
        let threads = rayon::current_num_threads();
        if threads == 1 || triples.len() < MIN_PARTITION * 2 {
            return Self::try_build(triples, dimension_sizes, block_capacity);
        }
        triples.par_sort_unstable();
        triples.dedup();
        let mut bits = dynamic_schedule(total_bits_needed(dimension_sizes));
        if bits.is_empty() {
            bits.push([0; 3]);
        }
        let shifts = bits.iter().map(|&b| level_shifts(b, [0, 1, 2])).collect();
        let mut remaining = total_bits_needed(dimension_sizes);
        let remaining = bits
            .iter()
            .map(|b| {
                for d in 0..3 {
                    remaining[d] -= b[d];
                }
                remaining
            })
            .collect();
        let schedule = Schedule {
            bits,
            shifts,
            remaining,
        };
        let height = schedule.bits.len();
        let mut scratch = vec![(0, 0, 0); triples.len()];
        let mut entries = vec![Entry {
            partition: Some(Partition {
                points: &mut triples,
                scratch: &mut scratch,
                lo: [0; 3],
            }),
            level: 0,
            word: 0,
            children: Vec::new(),
        }];
        // A radix split can expose up to 64 children. Bound retained private
        // DAGs independently of graph size, and keep small jobs sequential.
        let target = threads.saturating_mul(2);
        let mut frontier = 1;
        while frontier < target {
            let candidate = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.level + 1 < height)
                .filter_map(|(i, e)| e.partition.as_ref().map(|p| (i, p.points.len())))
                .filter(|&(_, n)| n >= MIN_PARTITION)
                .max_by_key(|&(_, n)| n);
            let Some((index, _)) = candidate else { break };
            let level = entries[index].level;
            let part = entries[index].partition.take().unwrap();
            let children = split(part, level, &schedule);
            frontier += children.len() - 1;
            for (slot, partition) in children {
                entries[index].word |= 1u64 << slot;
                let child = entries.len();
                entries[index].children.push(child);
                entries.push(Entry {
                    partition: Some(partition),
                    level: level + 1,
                    word: 0,
                    children: Vec::new(),
                });
            }
        }
        // Traverse the frontier in prefix order for reproducible merging,
        // regardless of which large partition was split first or finishes first.
        fn order(i: usize, entries: &[Entry<'_>], out: &mut Vec<usize>) {
            if entries[i].partition.is_some() {
                out.push(i);
            } else {
                for &child in &entries[i].children {
                    order(child, entries, out);
                }
            }
        }
        let mut indices = Vec::new();
        order(0, &entries, &mut indices);
        let jobs: Vec<_> = indices
            .iter()
            .map(|&i| (i, entries[i].level, entries[i].partition.take().unwrap()))
            .collect();
        let locals: Vec<_> = jobs
            .into_par_iter()
            .map(|(i, level, p)| {
                let mut register: Register = (0..height).map(|_| HashMap::new()).collect();
                let mut nodes: Vec<Vec<BfsNodeMeta>> = (0..height).map(|_| Vec::new()).collect();
                let mut visited = 0;
                let root = collect_rec_bfs(
                    p.points,
                    p.scratch,
                    p.lo,
                    level,
                    height,
                    &schedule.bits,
                    &schedule.shifts,
                    &schedule.remaining,
                    &mut register,
                    &mut nodes,
                    &mut visited,
                );
                (
                    i,
                    level,
                    Local {
                        nodes,
                        root,
                        visited,
                    },
                )
            })
            .collect();
        let mut register: Register = (0..height).map(|_| HashMap::new()).collect();
        let mut nodes: Vec<Vec<BfsNodeMeta>> = (0..height).map(|_| Vec::new()).collect();
        let mut roots = vec![0; entries.len()];
        let mut visited = 0;
        // Consume local records as they are merged. Only adjacent depths need
        // remapping arrays; private hash registers were freed by each worker.
        for (i, level, local) in locals {
            visited += local.visited;
            let mut children = Vec::new();
            for (depth, local_nodes) in local.nodes.into_iter().enumerate().rev() {
                let mut remap = Vec::with_capacity(local_nodes.len());
                for mut node in local_nodes {
                    for child in &mut node.child_indices {
                        *child = children[*child as usize];
                    }
                    remap.push(intern(node, &mut register[depth], &mut nodes[depth]));
                }
                children = remap;
                if depth == level {
                    roots[i] = children[local.root as usize];
                    break;
                }
            }
        }
        // Prefix skeleton is a tree; reverse creation order is child-first.
        for i in (0..entries.len()).rev() {
            let entry = &entries[i];
            if !entry.children.is_empty() {
                visited += 1;
                let node = BfsNodeMeta {
                    existence_word: entry.word,
                    child_indices: entry.children.iter().map(|&c| roots[c]).collect(),
                };
                roots[i] = intern(node, &mut register[entry.level], &mut nodes[entry.level]);
            }
        }
        drop(register);
        drop(entries);
        drop(scratch);
        drop(triples);
        Self::pack_nodes(nodes, dimension_sizes, block_capacity, visited)
    }
}

fn intern(
    node: BfsNodeMeta,
    register: &mut HashMap<(u64, Vec<u32>), u32>,
    nodes: &mut Vec<BfsNodeMeta>,
) -> u32 {
    let key = (node.existence_word, node.child_indices.clone());
    if let Some(&index) = register.get(&key) {
        return index;
    }
    let index = nodes.len() as u32;
    register.insert(key, index);
    nodes.push(node);
    index
}

fn split<'a>(part: Partition<'a>, level: usize, s: &Schedule) -> Vec<(usize, Partition<'a>)> {
    let qi = |p: &Triple| {
        compute_qi(
            [p.0 as u64, p.1 as u64, p.2 as u64],
            part.lo,
            s.shifts[level],
            s.remaining[level],
        )
    };
    let mut counts = [0; 64];
    for p in part.points.iter() {
        counts[qi(p)] += 1;
    }
    let mut offsets = [0; 64];
    for i in 1..64 {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }
    for p in part.points.iter() {
        let slot = qi(p);
        part.scratch[offsets[slot]] = *p;
        offsets[slot] += 1;
    }
    part.points.copy_from_slice(part.scratch);
    let (mut points, mut scratch) = (part.points, part.scratch);
    let mut result = Vec::new();
    for (slot, count) in counts.into_iter().enumerate() {
        if count == 0 {
            continue;
        }
        let (p, rest) = points.split_at_mut(count);
        let (tmp, rest_tmp) = scratch.split_at_mut(count);
        result.push((
            slot,
            Partition {
                points: p,
                scratch: tmp,
                lo: advance_lo(
                    part.lo,
                    s.bits[level],
                    s.shifts[level],
                    slot,
                    s.remaining[level],
                ),
            },
        ));
        points = rest;
        scratch = rest_tmp;
    }
    result
}
