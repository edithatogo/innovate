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

pub(crate) fn decompose(
    values: &[f64],
    period: usize,
    trend_span: usize,
) -> Result<serde_json::Value, &'static str> {
    if period < 2 || period > values.len() / 3 {
        return Err("seasonal period must leave at least three complete cycles");
    }
    let mut trend = vec![0.; values.len()];
    let mut seasonal = vec![0.; values.len()];
    for _ in 0..5 {
        let mut cycle = vec![0.; values.len() + 2 * period];
        for phase in 0..period {
            let samples: Vec<_> = (phase..values.len())
                .step_by(period)
                .map(|i| values[i] - trend[i])
                .collect();
            let smooth = loess_trend(&samples, samples.len().min(7))?;
            // Extend one cycle at each end so the centered low-pass filter
            // can return a value at every original sample.
            cycle[phase] = 2. * smooth[0] - smooth[1];
            for (j, &value) in smooth.iter().enumerate() {
                cycle[phase + (j + 1) * period] = value;
            }
            let n = smooth.len();
            cycle[phase + (n + 1) * period] = 2. * smooth[n - 1] - smooth[n - 2];
        }
        let moving_average = |data: &[f64], span: usize| -> Vec<f64> {
            data.windows(span)
                .map(|window| window.iter().map(|v| v / span as f64).sum())
                .collect()
        };
        let low = moving_average(&moving_average(&moving_average(&cycle, period), period), 3);
        let low_span = (period + 1) | 1;
        let low = loess_trend(&low, low_span)?;
        for (i, (season, low)) in seasonal.iter_mut().zip(low).enumerate() {
            *season = cycle[i + period] - low;
        }
        let adjusted: Vec<_> = values.iter().zip(&seasonal).map(|(y, s)| y - s).collect();
        trend = loess_trend(&adjusted, trend_span)?;
    }
    let residual: Vec<_> = values
        .iter()
        .zip(&trend)
        .zip(&seasonal)
        .map(|((y, t), s)| y - t - s)
        .collect();
    if trend
        .iter()
        .chain(&seasonal)
        .chain(&residual)
        .any(|value| !value.is_finite())
    {
        return Err("seasonal decomposition exceeded finite numerical domain");
    }
    Ok(serde_json::json!({"trend":trend,"seasonal":seasonal,"residual":residual}))
}
