/// Local linear regression on an evenly spaced sample grid.
pub(crate) fn loess_trend(values: &[f64], span: usize) -> Result<Vec<f64>, &'static str> {
    weighted_loess(values, span, None, false)
}

fn weighted_loess(
    values: &[f64],
    span: usize,
    robustness: Option<&[f64]>,
    extend: bool,
) -> Result<Vec<f64>, &'static str> {
    if span < 3 || span > values.len() {
        return Err("LOESS span must be between three and the series length");
    }
    let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
    if scale == 0. {
        return Ok(vec![0.; values.len() + 2 * usize::from(extend)]);
    }
    let extension = isize::from(extend);
    (-extension..values.len() as isize + extension)
        .map(|index| {
            let start = (index - (span / 2) as isize).max(0) as usize;
            let start = start.min(values.len() - span);
            let radius = index
                .abs_diff(start as isize)
                .max(index.abs_diff((start + span - 1) as isize)) as f64;
            let (mut weight, mut x_sum, mut y_sum, mut xx, mut xy) = (0., 0., 0., 0., 0.);
            for (offset, value) in values[start..start + span].iter().enumerate() {
                let x = ((start + offset) as f64 - index as f64) / radius;
                let y = value / scale;
                let w =
                    (1. - x.abs().powi(3)).powi(3) * robustness.map_or(1., |w| w[start + offset]);
                weight += w;
                x_sum += w * x;
                y_sum += w * y;
                xx += w * x * x;
                xy += w * x * y;
            }
            let determinant = weight * xx - x_sum * x_sum;
            let intercept = if determinant > 0. {
                (xx * y_sum - x_sum * xy) / determinant
            } else if weight == 0. {
                let nearest = robustness
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .filter(|(_, w)| **w > 0.)
                    .min_by_key(|(i, _)| (*i as isize).abs_diff(index))
                    .map_or(
                        index.clamp(0, values.len() as isize - 1) as usize,
                        |(i, _)| i,
                    );
                return Ok(values[nearest]);
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
    robust: bool,
) -> Result<serde_json::Value, &'static str> {
    if period < 2 || period > values.len() / 3 {
        return Err("seasonal period must leave at least three complete cycles");
    }
    let mut trend = vec![0.; values.len()];
    let mut seasonal = vec![0.; values.len()];
    let mut weights = vec![1.; values.len()];
    for pass in 0..if robust { 6 } else { 1 } {
        for _ in 0..5 {
            let mut cycle = vec![0.; values.len() + 2 * period];
            for phase in 0..period {
                let samples: Vec<_> = (phase..values.len())
                    .step_by(period)
                    .map(|i| values[i] - trend[i])
                    .collect();
                let phase_weights: Vec<_> = weights
                    .iter()
                    .skip(phase)
                    .step_by(period)
                    .copied()
                    .collect();
                let smooth = weighted_loess(
                    &samples,
                    samples.len().min(7),
                    Some(&phase_weights),
                    !robust,
                )?;
                let offset = usize::from(robust);
                if robust {
                    // Keep the robust endpoint convention while reweighting
                    // behavior is qualified against independent references.
                    cycle[phase] = 2. * smooth[0] - smooth[1];
                    let n = smooth.len();
                    cycle[phase + (n + 1) * period] = 2. * smooth[n - 1] - smooth[n - 2];
                }
                for (j, value) in smooth.into_iter().enumerate() {
                    cycle[phase + (j + offset) * period] = value;
                }
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
            trend = weighted_loess(&adjusted, trend_span, Some(&weights), false)?;
        }
        if robust && pass < 5 {
            let absolute: Vec<_> = values
                .iter()
                .zip(&trend)
                .zip(&seasonal)
                .map(|((y, t), s)| (y - t - s).abs())
                .collect();
            let mut ordered = absolute.clone();
            ordered.sort_by(f64::total_cmp);
            let middle = ordered.len() / 2;
            let median = ordered[(ordered.len() - 1) / 2].midpoint(ordered[middle]);
            let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
            let cutoff = (6. * median).max(32. * f64::EPSILON * scale);
            if cutoff == 0. {
                break;
            }
            for (weight, residual) in weights.iter_mut().zip(absolute) {
                *weight = (1. - (residual / cutoff).min(1.).powi(2)).powi(2);
            }
        }
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
