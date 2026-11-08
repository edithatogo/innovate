use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Method {
    #[default]
    Empirical,
    SplitConformal,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    time: Vec<f64>,
    observed: Vec<f64>,
    predicted: Vec<f64>,
    #[serde(default = "default_confidence")]
    confidence: f64,
    holdout: Option<Vec<f64>>,
    #[serde(default)]
    method: Method,
    #[serde(default)]
    assumptions: Vec<String>,
}

fn default_confidence() -> f64 {
    0.8
}

pub(crate) fn intervals(payload: &Value) -> Result<Value, &'static str> {
    let r: Request = serde_json::from_value(payload.clone())
        .map_err(|_| "invalid interval calibration payload")?;
    if !(0.0 < r.confidence && r.confidence < 1.0) {
        return Err("confidence must be between 0 and 1");
    }
    let n = r.time.len();
    if n == 0
        || r.observed.len() != n
        || r.predicted.len() != n
        || r.holdout.as_ref().is_some_and(|h| h.len() != n)
    {
        return Err("calibration arrays must have matching nonzero lengths");
    }
    if r.time.iter().any(|time| !time.is_finite())
        || r.time.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("time must be finite and strictly increasing");
    }
    let residuals: Vec<_> = r
        .observed
        .iter()
        .zip(&r.predicted)
        .map(|(actual, forecast)| actual - forecast)
        .collect();
    if residuals.iter().any(|v| !v.is_finite()) {
        return Err("calibration residual exceeds finite numerical domain");
    }
    let mut absolute: Vec<_> = residuals.iter().map(|v| v.abs()).collect();
    absolute.sort_unstable_by(f64::total_cmp);
    let holdout: Vec<_> = (0..n)
        .filter(|&i| r.holdout.as_ref().is_none_or(|h| h[i] != 0.0))
        .collect();
    let (width, method, scope) = match r.method {
        Method::Empirical => {
            let index = (r.confidence * (n - 1) as f64).round_ties_even() as usize;
            let width = holdout
                .iter()
                .fold(absolute[index], |width, &i| width.max(residuals[i].abs()));
            (
                width,
                "empirical",
                "descriptive empirical intervals; holdout rows may influence interval width",
            )
        }
        Method::SplitConformal => {
            let mask = r
                .holdout
                .as_ref()
                .ok_or("split conformal requires an explicit holdout mask")?;
            if holdout.is_empty() || holdout.len() == n {
                return Err("split conformal requires calibration and evaluation rows");
            }
            let mut scores: Vec<_> = residuals
                .iter()
                .zip(mask)
                .filter(|&(_, flag)| *flag == 0.0)
                .map(|(residual, _)| residual.abs())
                .collect();
            scores.sort_unstable_by(f64::total_cmp);
            let rank = ((scores.len() + 1) as f64 * r.confidence).ceil() as usize;
            if rank > scores.len() {
                return Err("too few calibration rows for a finite interval at this confidence");
            }
            (
                scores[rank - 1],
                "split_conformal",
                "evaluation rows do not determine width; marginal coverage requires exchangeability and an independently fitted predictor",
            )
        }
    };
    let lower: Vec<_> = r.predicted.iter().map(|v| v - width).collect();
    let upper: Vec<_> = r.predicted.iter().map(|v| v + width).collect();
    if lower.iter().chain(&upper).any(|v| !v.is_finite()) {
        return Err("calibration interval exceeds finite numerical domain");
    }
    let covered: Vec<_> = r
        .observed
        .iter()
        .enumerate()
        .map(|(i, v)| lower[i] <= *v && *v <= upper[i])
        .collect();
    let coverage = covered.iter().filter(|&&v| v).count() as f64 / n as f64;
    let holdout_coverage = if holdout.is_empty() {
        coverage
    } else {
        holdout.iter().filter(|&&i| covered[i]).count() as f64 / holdout.len() as f64
    };
    let scale = absolute[n - 1].max(f64::MIN_POSITIVE);
    let residual_mean = (residuals.iter().map(|v| v / scale).sum::<f64>() / n as f64) * scale;
    let residual_rmse =
        (residuals.iter().map(|v| (v / scale).powi(2)).sum::<f64>() / n as f64).sqrt() * scale;
    Ok(
        json!({"workflow":"uncertainty_calibration","backend":"cpu","time":r.time,
        "mean":r.predicted,"lower":lower,"upper":upper,
        "metadata":{"confidence":r.confidence,"interval_half_width":width,"assumptions":r.assumptions,
            "method":method,"scope":scope},
        "diagnostics":{"coverage":coverage,"holdout_coverage":holdout_coverage,
            "residual_mean":residual_mean,"residual_rmse":residual_rmse}}),
    )
}
