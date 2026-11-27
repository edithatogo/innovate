use num_bigint::BigInt;

fn dyadic_parts(value: f64) -> (i64, i32) {
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1075)
    };
    if significand == 0 {
        return (0, 0);
    }
    let trailing = significand.trailing_zeros();
    let magnitude = (significand >> trailing) as i64;
    let integer = if value.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    };
    (integer, exponent + trailing as i32)
}

fn common_exponent(values: &[f64]) -> i32 {
    values
        .iter()
        .copied()
        .filter(|&value| value != 0.)
        .map(|value| dyadic_parts(value).1)
        .min()
        .unwrap_or(0)
}

fn scaled_integer(value: f64, scale_exponent: i32) -> BigInt {
    let (integer, exponent) = dyadic_parts(value);
    if integer == 0 {
        BigInt::ZERO
    } else {
        BigInt::from(integer) << (exponent - scale_exponent) as usize
    }
}

/// Assign historical adopter categories from aligned sample weights.
///
/// `time` must be strictly increasing in one consistent unit. `rates` are
/// nonnegative sample weights; they are not multiplied by time-step widths.
/// Weighted time mean/standard deviation define the five historical categories,
/// with boundary ties assigned to the earlier category. One label is returned
/// for every input time, including zero-weight samples.
///
/// # Errors
/// Rejects empty/misaligned arrays, nonfinite values, non-increasing time,
/// negative weights and a zero total weight. Signed net adoption rates are
/// therefore not admissible as gross adoption weights.
///
/// # Examples
/// ```
/// let categories = innovate_core::classify_adopters(
///     &[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 3.0, 3.0, 1.0, 1.0],
/// )?;
/// assert_eq!(categories, [
///     "Innovators", "Early Adopters", "Early Majority", "Late Majority", "Laggards",
/// ]);
/// # Ok::<(), &'static str>(())
/// ```
pub fn classify_adopters(time: &[f64], rates: &[f64]) -> Result<Vec<&'static str>, &'static str> {
    if time.is_empty()
        || time.len() != rates.len()
        || time.iter().chain(rates).any(|v| !v.is_finite())
    {
        return Err("times and rates must be nonempty, finite and aligned");
    }
    if time.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("time must be strictly increasing");
    }
    if rates.iter().all(|&rate| rate == 0.) || rates.iter().any(|&rate| rate < 0.) {
        return Err("adoption rates must be nonnegative with positive total");
    }

    // Interpret the finite binary floats exactly. The common time and weight
    // scales cancel when comparing distances to the weighted mean and
    // standard deviation. Integer cutoffs retain the inclusive ties.
    let time_exponent = common_exponent(time);
    let rate_exponent = common_exponent(rates);
    let mut total_weight = BigInt::ZERO;
    let mut first_moment = BigInt::ZERO;
    let mut second_moment = BigInt::ZERO;
    for (&sample, &rate) in time.iter().zip(rates).filter(|&(_, &rate)| rate > 0.) {
        let sample = scaled_integer(sample, time_exponent);
        let weight = scaled_integer(rate, rate_exponent);
        let weighted_sample = &sample * &weight;
        second_moment += &sample * &weighted_sample;
        first_moment += weighted_sample;
        total_weight += weight;
    }
    let variance_numerator = &total_weight * second_moment - &first_moment * &first_moment;
    let sigma_floor = variance_numerator.sqrt();
    let sigma_ceil = &sigma_floor + u8::from(&sigma_floor * &sigma_floor < variance_numerator);
    let four_variances = variance_numerator << 2_usize;
    let twice_sigma_floor = four_variances.sqrt();
    let twice_sigma_ceil =
        &twice_sigma_floor + u8::from(&twice_sigma_floor * &twice_sigma_floor < four_variances);

    // The time grid is sorted, so find each inclusive cutoff once. Round the
    // subtracted deviations up and the added deviation down for exact integer
    // comparisons, including when the standard deviation is irrational.
    let boundaries = [
        (&first_moment - twice_sigma_ceil, "Innovators"),
        (&first_moment - sigma_ceil, "Early Adopters"),
        (first_moment.clone(), "Early Majority"),
        (first_moment + sigma_floor, "Late Majority"),
    ];
    let mut categories = Vec::with_capacity(time.len());
    for (cutoff, category) in boundaries {
        let end = time.partition_point(|&sample| {
            scaled_integer(sample, time_exponent) * &total_weight <= cutoff
        });
        categories.resize(end, category);
    }
    categories.resize(time.len(), "Laggards");
    Ok(categories)
}

#[cfg(test)]
mod tests {
    use super::classify_adopters;

    #[test]
    fn nonbinary_rate_maximum_preserves_exact_category_ties() {
        assert_eq!(
            classify_adopters(&[0., 1., 2., 3., 4.], &[0., 3., 3., 1., 1.]).unwrap(),
            [
                "Innovators",
                "Early Adopters",
                "Early Majority",
                "Late Majority",
                "Laggards"
            ]
        );
    }

    #[test]
    fn dominant_weight_keeps_the_direction_of_the_mean_shift() {
        let tail = 2_f64.powi(-120);
        let time = [0., 1., 2.];
        // The heavier tail shifts the exact mean to that side of 1. The shift
        // is smaller than the standard deviation but larger than zero.
        assert_eq!(
            classify_adopters(&time, &[tail, 1., tail / 4.]).unwrap(),
            ["Innovators", "Late Majority", "Laggards"]
        );
        assert_eq!(
            classify_adopters(&time, &[tail / 4., 1., tail]).unwrap(),
            ["Innovators", "Early Majority", "Laggards"]
        );
    }

    #[test]
    fn zero_weight_extremes_do_not_compress_weighted_time_grid() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            classify_adopters(
                &[-f64::MAX, 0., tiny, 2. * tiny, f64::MAX],
                &[0., 1., 0., 1., 0.]
            )
            .unwrap(),
            [
                "Innovators",
                "Early Adopters",
                "Early Majority",
                "Late Majority",
                "Laggards"
            ]
        );
    }

    #[test]
    fn categories_preserve_variance_for_extreme_rate_ratios() {
        assert_eq!(
            classify_adopters(&[0., 1e-317, 1.], &[f64::MAX, 0., f64::from_bits(1)]).unwrap(),
            ["Early Majority", "Late Majority", "Laggards"]
        );
    }

    #[test]
    fn dominant_weight_does_not_turn_a_late_sample_into_an_innovator() {
        let smallest = f64::from_bits(1);
        for dominant in [1., f64::MAX] {
            assert_eq!(
                classify_adopters(&[-3., 0., 1., 4.], &[0., 0., smallest, dominant]).unwrap(),
                ["Innovators", "Innovators", "Innovators", "Late Majority"]
            );
        }
    }

    #[test]
    fn categories_include_exact_boundaries_on_nonbinary_span() {
        let expected = [
            "Innovators",
            "Early Adopters",
            "Early Majority",
            "Late Majority",
        ];
        for time in [[0., 1., 2., 3.], [1024., 1025., 1026., 1027.]] {
            assert_eq!(
                classify_adopters(&time, &[0., 1., 0., 1.]).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn categories_preserve_irregular_grid_at_large_origin() {
        let origin = 1.5_f64;
        let next = origin.next_up();
        let fourth = next.next_up().next_up().next_up();
        let rates = [1., 2., 1.];
        let expected = classify_adopters(&[0., 1., 4.], &rates).unwrap();
        assert_eq!(
            classify_adopters(&[origin, next, fourth], &rates).unwrap(),
            expected
        );
        assert_eq!(
            classify_adopters(&[-f64::MAX, -f64::MAX / 2., f64::MAX], &rates).unwrap(),
            expected
        );
    }

    #[test]
    fn irregular_grid_matches_integer_moments_across_rate_scales() {
        for time in [[-1., 0., 2., 5., 9., 12.], [-8., 0., 2., 3., 9., 16.]] {
            for a in 1..=7 {
                for b in 1..=7 {
                    for c in 1..=7 {
                        let weight = f64::from(a + b + c);
                        let moment = f64::from(2 * b + 9 * c);
                        let variance = f64::from(4 * b + 81 * c) * weight - moment * moment;
                        let expected = time.map(|sample| {
                            let delta = sample * weight - moment;
                            let squared = delta * delta;
                            if delta < 0. && squared >= 4. * variance {
                                "Innovators"
                            } else if delta < 0. && squared >= variance {
                                "Early Adopters"
                            } else if delta <= 0. {
                                "Early Majority"
                            } else if squared <= variance {
                                "Late Majority"
                            } else {
                                "Laggards"
                            }
                        });
                        let rates = [0., f64::from(a), f64::from(b), 0., f64::from(c), 0.];
                        for scale in [1., f64::from_bits(1), f64::MIN_POSITIVE, 2_f64.powi(1021)] {
                            assert_eq!(
                                classify_adopters(&time, &rates.map(|rate| rate * scale)).unwrap(),
                                expected,
                                "weights {a}, {b}, {c}, scale {scale}"
                            );
                        }
                    }
                }
            }
        }
    }
}
