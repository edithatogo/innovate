/// Local linear regression on an evenly spaced sample grid.
pub(crate) fn loess_trend(values: &[f64], span: usize) -> Result<Vec<f64>, &'static str> {
    if span < 3 || span > values.len() {
        return Err("LOESS span must be between three and the series length");
    }
    let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
    if scale == 0. {
        return Ok(vec![0.; values.len()]);
    }
    (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(span / 2).min(values.len() - span);
            let radius = index.abs_diff(start).max(index.abs_diff(start + span - 1)) as f64;
            let (mut weight, mut x_sum, mut y_sum, mut xx, mut xy) = (0., 0., 0., 0., 0.);
            for (offset, value) in values[start..start + span].iter().enumerate() {
                let x = ((start + offset) as f64 - index as f64) / radius;
                let y = value / scale;
                let w = (1. - x.abs().powi(3)).powi(3);
                weight += w;
                x_sum += w * x;
                y_sum += w * y;
                xx += w * x * x;
                xy += w * x * y;
            }
            let determinant = weight * xx - x_sum * x_sum;
            let intercept = if determinant > 0. {
                (xx * y_sum - x_sum * xy) / determinant
            } else {
                y_sum / weight
            };
            let estimate = intercept * scale;
            estimate
                .is_finite()
                .then_some(estimate)
                .ok_or("LOESS estimate exceeded finite numerical domain")
        })
        .collect()
}
