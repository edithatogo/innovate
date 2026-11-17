//! Controls for the vendored faer sparse index/value sort.

#[test]
fn sparse_sort_preserves_indices_and_values_across_block_boundaries() {
    for len in [
        0_usize, 1, 2, 3, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 511, 512,
        513, 1024, 2049,
    ] {
        for pattern in 0..5 {
            for seed in 0..4_u64 {
                let mut state = seed + 1;
                let input: Vec<usize> = (0..len)
                    .map(|i| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        match pattern {
                            0 => i,
                            1 => len - i,
                            2 => i % 7,
                            3 => (state >> 32) as usize % (len + 1),
                            _ if i % 2 == 0 => i / 2,
                            _ => len - i / 2,
                        }
                    })
                    .collect();
                let mut expected: Vec<_> = input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, k)| (k, i))
                    .collect();
                expected.sort_unstable();
                let mut indices = input;
                let mut values: Vec<_> = (0..len).collect();
                faer::sparse::utils::sort_indices(&[0_usize, len], None, &mut indices, &mut values);
                assert!(indices.windows(2).all(|pair| pair[0] <= pair[1]));
                let mut actual: Vec<_> = indices.into_iter().zip(values).collect();
                actual.sort_unstable();
                assert_eq!(
                    actual, expected,
                    "len={len}, pattern={pattern}, seed={seed}"
                );
            }
        }
    }
}

#[test]
#[should_panic]
fn sparse_sort_rejects_an_out_of_range_column() {
    faer::sparse::utils::sort_indices(&[0_usize, 3], None, &mut [1, 0], &mut [2, 3]);
}
