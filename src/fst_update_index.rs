//! Compact per-depth suffix indexes: relative addresses and fingerprint prefixes.

enum Offsets {
    Small(Vec<u16>),
    Medium(Vec<u32>),
    Wide(Vec<u64>),
}
struct Addresses {
    origin: u64,
    offsets: Offsets,
}
impl Addresses {
    fn from_sorted(states: Vec<(u64, u64)>) -> Self {
        let origin = states
            .iter()
            .map(|&(_, address)| address)
            .min()
            .unwrap_or(0);
        let span = states
            .iter()
            .map(|&(_, address)| address - origin)
            .max()
            .unwrap_or(0);
        let offsets = if u16::try_from(span).is_ok() {
            let mut values: Vec<u16> = states
                .into_iter()
                .map(|(_, a)| (a - origin) as u16)
                .collect();
            values.shrink_to_fit();
            Offsets::Small(values)
        } else if u32::try_from(span).is_ok() {
            let mut values: Vec<u32> = states
                .into_iter()
                .map(|(_, a)| (a - origin) as u32)
                .collect();
            values.shrink_to_fit();
            Offsets::Medium(values)
        } else {
            let mut values: Vec<u64> = states.into_iter().map(|(_, a)| a - origin).collect();
            values.shrink_to_fit();
            Offsets::Wide(values)
        };
        Self { origin, offsets }
    }
    fn len(&self) -> usize {
        match &self.offsets {
            Offsets::Small(v) => v.len(),
            Offsets::Medium(v) => v.len(),
            Offsets::Wide(v) => v.len(),
        }
    }
    fn bytes(&self) -> usize {
        match &self.offsets {
            Offsets::Small(v) => crate::vec_bytes(v),
            Offsets::Medium(v) => crate::vec_bytes(v),
            Offsets::Wide(v) => crate::vec_bytes(v),
        }
    }
    fn find(
        &self,
        range: std::ops::Range<usize>,
        mut matches: impl FnMut(u64) -> bool,
    ) -> Option<u64> {
        fn scan<T: Copy + Into<u64>>(
            values: &[T],
            origin: u64,
            matches: &mut impl FnMut(u64) -> bool,
        ) -> Option<u64> {
            values
                .iter()
                .map(|&v| origin + v.into())
                .find(|&a| matches(a))
        }
        // Dispatch once per lookup, rather than branching for each candidate.
        match &self.offsets {
            Offsets::Small(v) => scan(&v[range], self.origin, &mut matches),
            Offsets::Medium(v) => scan(&v[range], self.origin, &mut matches),
            Offsets::Wide(v) => scan(&v[range], self.origin, &mut matches),
        }
    }
}

pub(crate) struct SuffixIndexLevel {
    addresses: Addresses,
    fingerprints: Vec<u32>,
}
impl SuffixIndexLevel {
    pub(crate) fn from_sorted(states: Vec<(u64, u64)>, rate: usize) -> Self {
        let fingerprints = states
            .iter()
            .step_by(rate)
            .map(|&(h, _)| (h >> 32) as u32)
            .collect();
        Self {
            addresses: Addresses::from_sorted(states),
            fingerprints,
        }
    }
    pub(crate) fn candidates(&self, hash: u64, rate: usize) -> std::ops::Range<usize> {
        // Truncation is monotone. Include every interval sharing the prefix,
        // including the unsampled tail preceding the first equal sample.
        let hash = (hash >> 32) as u32;
        let lower = self.fingerprints.partition_point(|&h| h < hash);
        let upper = if self.fingerprints.get(lower) == Some(&hash) {
            // Unique prefixes are the common case, especially at rate 1.
            // Avoid a second binary search unless the adjacent sample collides.
            if self.fingerprints.get(lower + 1) == Some(&hash) {
                lower + 2 + self.fingerprints[lower + 2..].partition_point(|&h| h == hash)
            } else {
                lower + 1
            }
        } else {
            lower
        };
        let start = if lower == 0 {
            0
        } else {
            (lower - 1).saturating_mul(rate).saturating_add(1)
        };
        let end = upper.saturating_mul(rate).min(self.addresses.len());
        start..end
    }
    pub(crate) fn find(
        &self,
        hash: u64,
        rate: usize,
        matches: impl FnMut(u64) -> bool,
    ) -> Option<u64> {
        self.addresses.find(self.candidates(hash, rate), matches)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.addresses.bytes() + crate::vec_bytes(&self.fingerprints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_addresses_round_trip_width_boundaries_and_tagged_ranges() {
        for values in [
            vec![],
            vec![u64::MAX],
            vec![70000, 135535],
            vec![70000, 135536],
            vec![10, 10 + u32::MAX as u64],
            vec![10, 11 + u32::MAX as u64],
            vec![12, (1 << 62) + 24],
            vec![u64::MAX - 500, u64::MAX - 1],
        ] {
            let pairs = values
                .iter()
                .rev()
                .enumerate()
                .map(|(i, &a)| (i as u64, a))
                .collect();
            let addresses = Addresses::from_sorted(pairs);
            let expected_width = if values.is_empty()
                || values.iter().max().unwrap() - values.iter().min().unwrap() <= u16::MAX as u64
            {
                2
            } else if values.iter().max().unwrap() - values.iter().min().unwrap() <= u32::MAX as u64
            {
                4
            } else {
                8
            };
            assert_eq!(addresses.bytes(), values.len() * expected_width);
            for (i, &target) in values.iter().rev().enumerate() {
                assert_eq!(addresses.find(i..i + 1, |_| true), Some(target));
                assert_eq!(
                    addresses.find(0..values.len(), |a| a == target),
                    Some(target)
                );
            }
            assert_eq!(addresses.find(0..values.len(), |_| false), None);
        }
    }

    #[test]
    fn truncated_fingerprint_collisions_keep_every_exact_candidate() {
        let hashes = [
            0,
            1,
            (1 << 32) - 1,
            1 << 32,
            (1 << 32) + 1,
            (1 << 32) + 9,
            2 << 32,
            u64::MAX,
        ];
        for rate in [1, 2, 3, 4, 8, 32, usize::MAX] {
            let index = SuffixIndexLevel::from_sorted(
                hashes
                    .iter()
                    .enumerate()
                    .map(|(i, &h)| (h, 100 + i as u64))
                    .collect(),
                rate,
            );
            for (i, &h) in hashes.iter().enumerate() {
                assert_eq!(
                    index.find(h, rate, |a| a == 100 + i as u64),
                    Some(100 + i as u64)
                );
            }
            assert_eq!(index.find((1 << 32) + 2, rate, |_| false), None);
            assert_eq!(index.fingerprints.capacity(), hashes.len().div_ceil(rate));
        }
    }
}
