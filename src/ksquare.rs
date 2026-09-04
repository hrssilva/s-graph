pub struct KSquare<const B: usize> {
    nodes_t: Vec<u64>,
    nodes_l: Vec<u64>,
    height: usize,
}

impl<const B: usize> KSquare<B> {
    const K: u8 = 1u8 << B;
    const K2: u8 = Self::K * Self::K;
}