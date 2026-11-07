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
            estimated = estimate(&differences, order, seasonal_order)?;
            &estimated
        }
    };
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
) -> Result<Vec<f64>, &'static str> {
    let [p, _, q] = order;
    let [sp, _, sq, _] = seasonal_order;
    let count = [p, q, sp, sq].into_iter().try_fold(0usize, usize::checked_add)
        .filter(|count| *count <= 64 && *count < values.len())
        .ok_or("SARIMA estimation requires more observations than coefficients and at most 64 coefficients")?;
    let score = |coefficients: &[f64]| {
        let invertible = |terms: &[f64]| stationary(&terms.iter().map(|v| -v).collect::<Vec<_>>());
        if !invertible(&coefficients[p..p + q]) || !invertible(&coefficients[p + q + sp..]) {
            return None;
        }
        stationary_filter(values, order, seasonal_order, coefficients)
            .ok()?
            .log_likelihood
    };
    let mut best = vec![0.; count];
    let initial = stationary_filter(values, order, seasonal_order, &best)?;
    let mut likelihood = initial
        .log_likelihood
        .ok_or("SARIMA likelihood is degenerate")?;
    let mut step = 0.25;
    // Refine a deterministic coordinate search over stationary and invertible models.
    for _ in 0..512 {
        let mut improved = false;
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
            improved |= selected != original;
            best[index] = selected;
        }
        if !improved {
            step *= 0.5;
            if step < 1e-6 {
                return Ok(best);
            }
        }
    }
    Err("SARIMA parameter estimation did not converge")
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
    let [p, _, q] = order;
    let [sp, _, sq, period] = seasonal_order;
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

    // T has phi in its first column and ones on the superdiagonal.
    let propagate = |cov: &[f64], next: &mut [f64]| {
        let at = |i, j| if i < n && j < n { cov[i * n + j] } else { 0. };
        for i in 0..n {
            for j in 0..n {
                next[i * n + j] = phi[i] * phi[j] * cov[0]
                    + phi[i] * at(0, j + 1)
                    + phi[j] * at(i + 1, 0)
                    + at(i + 1, j + 1)
                    + noise[i] * noise[j];
            }
        }
    };
    let mut covariance = vec![0.; n * n];
    let mut next = covariance.clone();
    let mut converged = false;
    for _ in 0..10_000 {
        propagate(&covariance, &mut next);
        if next.iter().any(|v| !v.is_finite()) {
            return Err("SARIMA stationary covariance diverged");
        }
        converged = next
            .iter()
            .zip(&covariance)
            .all(|(a, b)| (a - b).abs() <= 1e-14);
        std::mem::swap(&mut covariance, &mut next);
        if converged {
            break;
        }
    }
    if !converged {
        return Err("SARIMA stationary covariance did not converge");
    }

    let mut state = vec![0.; n];
    let mut fitted = Vec::with_capacity(values.len());
    let mut residual = Vec::with_capacity(values.len());
    let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
    let mut standardized_sum = 0.;
    let mut log_variance_sum = 0.;
    for &value in values {
        let prediction = state[0];
        let innovation = value - prediction;
        let variance = covariance[0];
        if !innovation.is_finite() || !variance.is_finite() || variance <= 0. {
            return Err("SARIMA filter exceeded its finite numerical domain");
        }
        if scale > 0. {
            standardized_sum += (innovation / scale).powi(2) / variance;
        }
        log_variance_sum += variance.ln();
        fitted.push(prediction);
        residual.push(innovation);
        for i in 0..n {
            state[i] += (covariance[i * n] / variance) * innovation;
        }
        for i in 0..n {
            for j in (i..n).rev() {
                let updated =
                    covariance[i * n + j] - (covariance[i * n] / variance) * covariance[j * n];
                covariance[i * n + j] = updated;
            }
        }
        for i in 0..n {
            for j in 0..i {
                covariance[i * n + j] = covariance[j * n + i];
            }
        }
        let observed_state = state[0];
        for i in 0..n {
            state[i] = phi[i] * observed_state + state.get(i + 1).copied().unwrap_or(0.);
        }
        propagate(&covariance, &mut next);
        std::mem::swap(&mut covariance, &mut next);
    }
    // Concentrate out the common innovation variance.
    let count = values.len() as f64;
    let log_variance = (standardized_sum / count).ln() + 2. * scale.ln();
    let likelihood =
        -0.5 * (count * (1. + std::f64::consts::TAU.ln() + log_variance) + log_variance_sum);
    Ok(FilterOutput {
        predictions: fitted,
        innovations: residual,
        log_likelihood: likelihood.is_finite().then_some(likelihood),
    })
}
