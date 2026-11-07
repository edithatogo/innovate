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
    let first = time[0];
    let last = time[time.len() - 1];
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
    let points = time
        .iter()
        .zip(rates)
        .map(|(&t, rate)| (scaled_time(t), rate / max_rate));
    let total: f64 = points.clone().map(|(_, weight)| weight).sum();
    let mean = points.clone().map(|(t, weight)| t * weight).sum::<f64>() / total;
    let sd = (points
        .clone()
        .map(|(t, weight)| (t - mean).powi(2) * weight)
        .sum::<f64>()
        / total)
        .sqrt();
    let categories: Vec<_> = points
        .map(|(t, _)| {
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
}
