use serde_json::{Value, json};

pub(crate) fn residual_series(payload: &Value) -> Result<Value, &'static str> {
    let series = |key: &str| {
        serde_json::from_value::<Vec<f64>>(payload[key].clone())
            .map_err(|_| "residual diagnostics require numeric time and residual arrays")
    };
    let time = series("time")?;
    let residuals = series("residuals")?;
    if time.is_empty() || time.len() != residuals.len() {
        return Err("time and residuals must be nonempty and aligned");
    }
    let lags = if payload["lags"].is_null() {
        30
    } else {
        payload["lags"]
            .as_u64()
            .filter(|lag| *lag <= 1024)
            .ok_or("lags must be an integer between 0 and 1024")? as usize
    };
    let lags = lags.min(residuals.len() - 1);
    let scale = residuals.iter().map(|v| v.abs()).fold(0., f64::max);
    let count = residuals.len() as f64;
    let scaled: Vec<_> = residuals
        .iter()
        .map(|v| if scale == 0. { 0. } else { v / scale })
        .collect();
    let mean = scaled.iter().sum::<f64>() / count;
    let centered: Vec<_> = scaled.iter().map(|v| v - mean).collect();
    let variance: f64 = centered.iter().map(|v| v * v).sum();
    let mut acf = vec![Some(1.)];
    let mut pacf = vec![Some(1.)];
    let mut coefficients: Vec<f64> = Vec::new();
    let mut prediction_variance = 1.;
    for lag in 1..=lags {
        let correlation = (variance > 0.).then(|| {
            centered
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / variance
        });
        acf.push(correlation);
        let partial = correlation
            .filter(|_| prediction_variance > f64::EPSILON)
            .map(|correlation| {
                let explained = coefficients
                    .iter()
                    .zip(acf[1..lag].iter().rev())
                    .map(|(coefficient, correlation)| coefficient * correlation.unwrap_or_default())
                    .sum::<f64>();
                ((correlation - explained) / prediction_variance).clamp(-1., 1.)
            });
        pacf.push(partial);
        if let Some(partial) = partial {
            coefficients = coefficients
                .iter()
                .zip(coefficients.iter().rev())
                .map(|(forward, reverse)| forward - partial * reverse)
                .chain(std::iter::once(partial))
                .collect();
            prediction_variance *= 1. - partial * partial;
        }
    }
    Ok(
        json!({"time":time,"residuals":residuals,"lags":(0..=lags).collect::<Vec<_>>(),
        "acf":acf,"pacf":pacf,"pacf_method":"durbin_levinson_biased_autocovariance"}),
    )
}
