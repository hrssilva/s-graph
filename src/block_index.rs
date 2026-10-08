//! Compact Fenwick directory for logical block lengths. Same-block navigation
//! remains direct; cross-block spans cost O(log blocks), independent of span.

use crate::storage::{BlockStorage, Pos};

#[derive(Default)]
pub(crate) struct BlockIndex {
    sums: Vec<u64>,
    total: u64,
}

impl BlockIndex {
    pub(crate) fn build<S: BlockStorage>(storage: &S) -> Self {
        let mut index = Self::default();
        index.rebuild(storage);
        index
    }

    pub(crate) fn rebuild<S: BlockStorage>(&mut self, storage: &S) {
        let n = storage.block_count();
        self.sums.clear();
        self.sums.resize(n, 0);
        self.total = 0;
        for i in 0..n {
            let len = storage.block_len(i) as u64;
            self.total += len;
            self.sums[i] += len;
            let parent = i | (i + 1);
            if parent < n {
                self.sums[parent] += self.sums[i];
            }
        }
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.sums.shrink_to_fit();
    }

    pub(crate) fn grow(&mut self, mut block: usize, bytes: usize) {
        self.total += bytes as u64;
        while block < self.sums.len() {
            self.sums[block] += bytes as u64;
            block |= block + 1;
        }
    }

    pub(crate) fn shrink(&mut self, mut block: usize, bytes: usize) {
        self.total -= bytes as u64;
        while block < self.sums.len() {
            self.sums[block] -= bytes as u64;
            block |= block + 1;
        }
    }

    pub(crate) fn prefix(&self, mut blocks: usize) -> u64 {
        let mut sum = 0;
        while blocks != 0 {
            sum += self.sums[blocks - 1];
            blocks &= blocks - 1;
        }
        sum
    }

    pub(crate) fn locate(&self, target: u64) -> Pos {
        assert!(target <= self.total && !self.sums.is_empty(), "block span exceeds storage");
        if target == self.total {
            let block = self.sums.len() - 1;
            return Pos { block, offset: (target - self.prefix(block)) as usize };
        }
        let mut count = 0;
        let mut sum = 0;
        let mut step = 1usize << (usize::BITS - 1 - self.sums.len().leading_zeros());
        while step != 0 {
            let next = count + step;
            if next <= self.sums.len() && sum + self.sums[next - 1] <= target {
                sum += self.sums[next - 1];
                count = next;
            }
            step >>= 1;
        }
        Pos { block: count, offset: (target - sum) as usize }
    }

    pub(crate) fn mem_size_bytes(&self) -> usize {
        crate::vec_bytes(&self.sums)
    }
}
