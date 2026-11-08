/// Assign historical adopter categories from aligned sample weights.
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
    let max_rate = rates.iter().copied().fold(0., f64::max);
    if max_rate == 0. || rates.iter().any(|rate| *rate < 0.) {
        return Err("adoption rates must be nonnegative with positive total");
    }
    let weighted_samples = time.iter().zip(rates).filter(|&(_, rate)| *rate > 0.);
    let first = *weighted_samples.clone().next().unwrap().0;
    let last = *weighted_samples.clone().next_back().unwrap().0;
    // Halve before subtraction when the grid spans both extremes.
    let prescale = if (last - first).is_finite() { 1. } else { 0.5 };
    let origin = first * prescale;
    let span = last * prescale - origin;
    // For finite spans, use a power of two to avoid rounding exact boundary ties.
    let exponent = span.to_bits() & f64::INFINITY.to_bits();
    let scale = if prescale == 1. {
        f64::from_bits(exponent).max(f64::MIN_POSITIVE)
    } else {
        span
    };
    let scaled_time = |t: f64| (t * prescale - origin) / scale;
    // Binary scaling avoids rounding exact weighted-moment boundaries
    // merely because the largest rate is not a power of two.
    let rate_exponent = max_rate.to_bits() & f64::INFINITY.to_bits();
    let rate_scale = f64::from_bits(rate_exponent).max(f64::MIN_POSITIVE);
    let points = weighted_samples
        .clone()
        .map(|(&t, rate)| (scaled_time(t), rate / rate_scale));
    let total: f64 = points.clone().map(|(_, weight)| weight).sum();
    let mean = points.clone().map(|(t, weight)| t * weight).sum::<f64>() / total;
    let variance = points
        .clone()
        .map(|(t, weight)| (t - mean).powi(2) * weight)
        .sum::<f64>()
        / total;
    let sd = if variance < f64::MIN_POSITIVE {
        // Square-root weights retain dispersion even when variance underflows.
        let root_rate_scale = rate_scale.sqrt();
        weighted_samples
            .map(|(&t, rate)| (scaled_time(t) - mean) * (rate.sqrt() / root_rate_scale))
            .fold(0.0_f64, f64::hypot)
            / total.sqrt()
    } else {
        variance.sqrt()
    };
    let categories: Vec<_> = time
        .iter()
        .map(|&sample| {
            let t = scaled_time(sample);
            if t <= mean - 2. * sd {
                "Innovators"
            } else if t <= mean - sd {
                "Early Adopters"
            } else if t <= mean {
                "Early Majority"
            } else if t <= mean + sd {
                "Late Majority"
            } else {
                "Laggards"
            }
        })
        .collect();
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
        let time = [-1., 0., 2., 5., 9., 12.];
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
