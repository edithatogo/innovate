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
    let time_scale = time.iter().copied().map(f64::abs).fold(1., f64::max);
    let points = time
        .iter()
        .zip(rates)
        .map(|(t, rate)| (t / time_scale, rate / max_rate));
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
