//! Stationary multiplicative SARIMA innovations filter.

fn multiply_lags(ordinary: &[f64], seasonal: &[f64], period: usize, sign: f64) -> Vec<f64> {
    let mut polynomial = vec![0.; 1 + ordinary.len() + seasonal.len() * period];
    polynomial[0] = 1.;
    for (i, &coefficient) in ordinary.iter().enumerate() {
        polynomial[i + 1] += sign * coefficient;
    }
    for (j, &coefficient) in seasonal.iter().enumerate() {
        let lag = (j + 1) * period;
        polynomial[lag] += sign * coefficient;
        for (i, &ordinary) in ordinary.iter().enumerate() {
            polynomial[lag + i + 1] += ordinary * coefficient;
        }
    }
    polynomial
}

fn stationary(coefficients: &[f64]) -> bool {
    let mut coefficients = coefficients.to_vec();
    while let Some(reflection) = coefficients.pop() {
        if !reflection.is_finite() || reflection.abs() >= 1. {
            return false;
        }
        let denominator = 1. - reflection * reflection;
        let previous = coefficients.clone();
        for (coefficient, (left, right)) in coefficients
            .iter_mut()
            .zip(previous.iter().zip(previous.iter().rev()))
        {
            *coefficient = (left + reflection * right) / denominator;
        }
    }
    true
}

pub(crate) fn filter(
    values: &[f64],
    order: [usize; 3],
    seasonal_order: [usize; 4],
    coefficients: Option<&[f64]>,
    historical: bool,
) -> Result<serde_json::Value, &'static str> {
    let [p, d, q] = order;
    let [sp, sd, sq, period] = seasonal_order;
    if sd > 0 && period < 2 {
        return Err("seasonal differencing requires a period of at least two");
    }
    let warmup = sd
        .checked_mul(period)
        .and_then(|lag| lag.checked_add(d))
        .filter(|lag| *lag <= 64 && *lag < values.len())
        .ok_or("SARIMA differencing requires observations beyond a warmup of at most 64 samples")?;
    let mut differences = values.to_vec();
    for lag in std::iter::repeat_n(1, d).chain(std::iter::repeat_n(period, sd)) {
        for i in 0..differences.len() - lag {
            differences[i] = differences[i + lag] - differences[i];
        }
        differences.truncate(differences.len() - lag);
    }
    let order = [p, 0, q];
    let seasonal_order = [sp, 0, sq, period];
    let estimated;
    let coefficients = match coefficients {
        Some(coefficients) => coefficients,
        None => {
            estimated = if historical {
                estimate(values, [p, d, q], [sp, sd, sq, period], warmup)?
            } else {
                estimate(&differences, order, seasonal_order, 0)?
            };
            &estimated
        }
    };
    if historical && warmup > 0 {
        let output = state_filter(values, [p, d, q], [sp, sd, sq, period], coefficients)?;
        return Ok(serde_json::json!({
            "fitted": output.predictions, "residual": output.innovations,
            "log_likelihood": output.log_likelihood, "coefficients": coefficients
        }));
    }
    let FilterOutput {
        predictions,
        innovations,
        log_likelihood,
    } = stationary_filter(&differences, order, seasonal_order, coefficients)?;
    let mut fitted = vec![None; warmup];
    let mut residual = vec![None; warmup];
    for ((&observed, differenced), (prediction, innovation)) in values[warmup..]
        .iter()
        .zip(differences)
        .zip(predictions.into_iter().zip(innovations))
    {
        let prediction = prediction + (observed - differenced);
        if !prediction.is_finite() {
            return Err("SARIMA reconstructed prediction exceeded its finite numerical domain");
        }
        fitted.push(Some(prediction));
        residual.push(Some(innovation));
    }
    Ok(
        serde_json::json!({"fitted": fitted, "residual": residual, "log_likelihood": log_likelihood, "coefficients": coefficients}),
    )
}

fn estimate(
    values: &[f64],
    order: [usize; 3],
    seasonal_order: [usize; 4],
    burn: usize,
) -> Result<Vec<f64>, &'static str> {
    let [p, _, q] = order;
    let [sp, _, sq, _] = seasonal_order;
    let count = [p, q, sp, sq].into_iter().try_fold(0usize, usize::checked_add)
        .filter(|count| *count <= 64 && *count < values.len() - burn)
        .ok_or("SARIMA estimation requires more observations than coefficients and at most 64 coefficients")?;
    let score = |coefficients: &[f64]| {
        let invertible = |terms: &[f64]| stationary(&terms.iter().map(|v| -v).collect::<Vec<_>>());
        if !invertible(&coefficients[p..p + q]) || !invertible(&coefficients[p + q + sp..]) {
            return None;
        }
        state_filter(values, order, seasonal_order, coefficients)
            .ok()?
            .log_likelihood
    };
    let mut best = vec![0.; count];
    let initial = state_filter(values, order, seasonal_order, &best)?;
    let initial_likelihood = initial
        .log_likelihood
        .ok_or("SARIMA likelihood is degenerate")?;
    let mut likelihood = initial_likelihood;
    let retry_cancellation = p == 1 && q == 1 && sp == 0 && sq == 0;
    let search_all_pairs = sp > 0 && sq > 0 && count <= 8;
    let joint_moves: Vec<_> = if search_all_pairs {
        (0..count)
            .flat_map(|left| (left + 1..count).map(move |right| (left, right)))
            .collect()
    } else {
        [p..p + q, p + q + sp..count]
            .into_iter()
            .flat_map(|terms| (terms.start..terms.end.saturating_sub(1)).map(|i| (i, i + 1)))
            .collect()
    };
    let mut step = 0.25;
    // Refine a deterministic coordinate search over stationary and invertible models.
    let sweeps = 2048;
    for iteration in 0..(sweeps * if retry_cancellation { 2 } else { 1 }) {
        let retry = iteration >= sweeps;
        if iteration == sweeps {
            // Restart with coupled moves for difficult AR/MA cancellation.
            best.fill(0.);
            likelihood = initial_likelihood;
            step = 0.25;
        }
        let base = best.clone();
        for index in 0..count {
            let original = best[index];
            let mut selected = original;
            for direction in [-1., 1.] {
                best[index] = original + direction * step;
                if let Some(candidate) = score(&best).filter(|v| *v > likelihood) {
                    likelihood = candidate;
                    selected = best[index];
                }
            }
            best[index] = selected;
        }
        if best == base {
            // Search joint moves when correlated coefficients stall the coordinate sweep.
            let mut trial = base.clone();
            let pairs = if retry { &[(0, 1)][..] } else { &joint_moves };
            for &(left, right) in pairs {
                let ratios: &[(f64, f64)] = if search_all_pairs || retry {
                    &[(1., 1.), (1., 0.5), (0.5, 1.)]
                } else {
                    &[(1., 1.)]
                };
                for &(left_scale, right_scale) in ratios {
                    for left_direction in [-1., 1.] {
                        for right_direction in [-1., 1.] {
                            trial[left] = base[left] + left_direction * left_scale * step;
                            trial[right] = base[right] + right_direction * right_scale * step;
                            if let Some(candidate) = score(&trial).filter(|v| *v > likelihood) {
                                likelihood = candidate;
                                best.clone_from(&trial);
                            }
                        }
                    }
                }
                trial[left] = base[left];
                trial[right] = base[right];
            }
            if best != base {
                continue;
            }
            step *= 0.5;
            if step < 1e-6 {
                return Ok(best);
            }
        } else {
            // Follow the net direction of the sweep across correlated terms.
            let candidate: Vec<_> = best.iter().zip(base).map(|(a, b)| a + (a - b)).collect();
            if let Some(candidate_score) = score(&candidate).filter(|v| *v > likelihood) {
                best = candidate;
                likelihood = candidate_score;
            }
        }
    }
    Err("SARIMA parameter estimation did not converge")
}

fn stationary_covariance(phi: &[f64], noise: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = phi.len();
    let mut power = vec![0.; n * n];
    let mut covariance = vec![0.; n * n];
    for i in 0..n {
        power[i * n] = phi[i];
        if i + 1 < n {
            power[i * n + i + 1] = 1.;
        }
        for j in 0..n {
            covariance[i * n + j] = noise[i] * noise[j];
        }
    }
    let multiply = |left: &[f64], right: &[f64]| {
        let mut product = vec![0.; n * n];
        for i in 0..n {
            for k in 0..n {
                let scale = left[i * n + k];
                for j in 0..n {
                    product[i * n + j] += scale * right[k * n + j];
                }
            }
        }
        product
    };
    // Each step doubles the number of innovation terms in the covariance sum.
    for _ in 0..64 {
        let transpose: Vec<_> = (0..n * n).map(|i| power[(i % n) * n + i / n]).collect();
        let increment = multiply(&multiply(&power, &covariance), &transpose);
        let converged = increment.iter().all(|value| value.abs() <= 1e-14);
        for (value, addition) in covariance.iter_mut().zip(increment) {
            *value += addition;
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err("SARIMA stationary covariance diverged");
        }
        if converged {
            return Ok(covariance);
        }
        power = multiply(&power, &power);
    }
    Err("SARIMA stationary covariance did not converge")
}

struct FilterOutput {
    predictions: Vec<f64>,
    innovations: Vec<f64>,
    log_likelihood: Option<f64>,
}

fn stationary_filter(
    values: &[f64],
    order: [usize; 3],
    seasonal_order: [usize; 4],
    coefficients: &[f64],
) -> Result<FilterOutput, &'static str> {
    state_filter(values, order, seasonal_order, coefficients)
}

fn state_filter(
    values: &[f64],
    order: [usize; 3],
    seasonal_order: [usize; 4],
    coefficients: &[f64],
) -> Result<FilterOutput, &'static str> {
    let [p, d, q] = order;
    let [sp, sd, sq, period] = seasonal_order;
    if (sp != 0 || sq != 0) && period < 2 {
        return Err("seasonal AR or MA terms require a period of at least two");
    }
    let lag =
        |ordinary: usize, seasonal: usize| seasonal.checked_mul(period)?.checked_add(ordinary);
    let ar_lag = lag(p, sp).ok_or("SARIMA lag overflow")?;
    let ma_lag = lag(q, sq).ok_or("SARIMA lag overflow")?;
    if ar_lag > 64 || ma_lag >= 64 {
        return Err("SARIMA filter supports at most 64 state coordinates");
    }
    if coefficients.len() != p + q + sp + sq || coefficients.iter().any(|v| !v.is_finite()) {
        return Err(
            "SARIMA coefficients must contain finite AR, MA, seasonal AR and seasonal MA terms",
        );
    }
    if !stationary(&coefficients[..p]) || !stationary(&coefficients[p + q..p + q + sp]) {
        return Err("SARIMA autoregressive coefficients must be stationary");
    }
    let ar = multiply_lags(
        &coefficients[..p],
        &coefficients[p + q..p + q + sp],
        period,
        -1.,
    );
    let ma = multiply_lags(
        &coefficients[p..p + q],
        &coefficients[p + q + sp..],
        period,
        1.,
    );
    let n = ar_lag.max(ma_lag + 1);
    let phi: Vec<_> = (0..n)
        .map(|i| -ar.get(i + 1).copied().unwrap_or(0.))
        .collect();
    let noise: Vec<_> = (0..n).map(|i| ma.get(i).copied().unwrap_or(0.)).collect();

    let covariance = stationary_covariance(&phi, &noise)?;

    let burn = d + sd * period;
    let stationary_size = n;
    let n = n + burn;
    let mut transition = vec![Vec::new(); n];
    let mut design: Vec<_> = (0..d).collect();
    design.extend((1..=sd).map(|season| d + season * period - 1));
    for (row, terms) in transition.iter_mut().take(d).enumerate() {
        terms.extend((row..d).map(|column| (column, 1.)));
        terms.extend(design[d..].iter().map(|&column| (column, 1.)));
        terms.push((burn, 1.));
    }
    for season in 0..sd {
        let start = d + season * period;
        transition[start].push((start + period - 1, 1.));
        transition[start].push((burn, 1.));
        if season + 1 < sd {
            transition[start].push((start + 2 * period - 1, 1.));
        }
        for (offset, terms) in transition[start + 1..start + period].iter_mut().enumerate() {
            terms.push((start + offset, 1.));
        }
    }
    design.push(burn);
    for (i, &coefficient) in phi.iter().enumerate() {
        transition[burn + i].push((burn, coefficient));
        if i + 1 < stationary_size {
            transition[burn + i].push((burn + i + 1, 1.));
        }
    }
    let mut initial = vec![0.; n * n];
    for i in 0..burn {
        initial[i * n + i] = 1e6;
    }
    for i in 0..stationary_size {
        for j in 0..stationary_size {
            initial[(burn + i) * n + burn + j] = covariance[i * stationary_size + j];
        }
    }
    let mut covariance = initial;
    let mut next = vec![0.; n * n];
    let noise: Vec<_> = std::iter::repeat_n(0., burn).chain(noise).collect();
    let mut projected = vec![0.; n];
    let mut state = vec![0.; n];
    let mut fitted = Vec::with_capacity(values.len());
    let mut residual = Vec::with_capacity(values.len());
    let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
    let mut standardized_sum = 0.;
    let mut log_variance_sum = 0.;
    for (index, &value) in values.iter().enumerate() {
        let prediction: f64 = design.iter().map(|&i| state[i]).sum();
        let innovation = value - prediction;
        for (row, projection) in projected.iter_mut().enumerate() {
            *projection = design.iter().map(|&j| covariance[row * n + j]).sum();
        }
        let variance: f64 = design.iter().map(|&i| projected[i]).sum();
        if !innovation.is_finite() || !variance.is_finite() || variance <= 0. {
            return Err("SARIMA filter exceeded its finite numerical domain");
        }
        if index >= burn {
            if scale > 0. {
                standardized_sum += (innovation / scale).powi(2) / variance;
            }
            log_variance_sum += variance.ln();
        }
        fitted.push(prediction);
        residual.push(innovation);
        for i in 0..n {
            state[i] += (projected[i] / variance) * innovation;
            for j in 0..n {
                covariance[i * n + j] -= (projected[i] / variance) * projected[j];
            }
        }
        let previous = state.clone();
        for (i, left) in transition.iter().enumerate() {
            state[i] = left.iter().map(|&(k, weight)| weight * previous[k]).sum();
            for (j, right) in transition.iter().enumerate() {
                let mut value = noise[i] * noise[j];
                for &(k, a) in left {
                    for &(l, b) in right {
                        value += a * b * covariance[k * n + l];
                    }
                }
                next[i * n + j] = value;
            }
        }
        std::mem::swap(&mut covariance, &mut next);
    }
    // Concentrate out the common innovation variance.
    let count = (values.len() - burn) as f64;
    let log_variance = (standardized_sum / count).ln() + 2. * scale.ln();
    let likelihood =
        -0.5 * (count * (1. + std::f64::consts::TAU.ln() + log_variance) + log_variance_sum);
    Ok(FilterOutput {
        predictions: fitted,
        innovations: residual,
        log_likelihood: likelihood.is_finite().then_some(likelihood),
    })
}
