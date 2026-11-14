use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn compose(payload: &Value) -> Result<Value, &'static str> {
    let time: Vec<f64> = serde_json::from_value(payload["time"].clone())
        .map_err(|_| "time must be a numeric array")?;
    let predictions: BTreeMap<String, Vec<f64>> =
        serde_json::from_value(payload["predictions"].clone())
            .map_err(|_| "predictions must map regimes to numeric arrays")?;
    if time.is_empty()
        || predictions.is_empty()
        || predictions.values().any(|p| p.len() != time.len())
    {
        return Err("regime predictions must align with a nonempty time grid");
    }
    if time.iter().any(|value| !value.is_finite()) || time.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("time must be finite and strictly increasing");
    }
    let mut weights: BTreeMap<String, f64> = if payload["weights"].is_null() {
        predictions.keys().map(|key| (key.clone(), 1.)).collect()
    } else {
        serde_json::from_value(payload["weights"].clone())
            .map_err(|_| "weights must map regimes to numbers")?
    };
    if weights.keys().ne(predictions.keys()) {
        return Err("weights must match prediction regime keys");
    }
    let total: f64 = weights.values().sum();
    if !total.is_finite() || total <= 0. || weights.values().any(|weight| *weight < 0.) {
        return Err("weights must be nonnegative and sum to a finite positive value");
    }
    for weight in weights.values_mut() {
        *weight /= total;
    }
    let mut mean = vec![0.; time.len()];
    for (path, weight) in predictions.values().zip(weights.values()) {
        for (average, value) in mean.iter_mut().zip(path) {
            *average += value * weight;
        }
    }
    if mean.iter().any(|value| !value.is_finite()) {
        return Err("ensemble predictions exceed the finite numerical domain");
    }
    let diagnostics = if payload["observed"].is_null() {
        json!({})
    } else {
        let observed: Vec<f64> = serde_json::from_value(payload["observed"].clone())
            .map_err(|_| "observed must be a numeric array")?;
        if observed.len() != time.len() {
            return Err("observed and predicted values must have matching lengths");
        }
        let residuals = observed
            .iter()
            .zip(&mean)
            .map(|(actual, forecast)| actual - forecast);
        let count = time.len() as f64;
        let mae = residuals
            .clone()
            .map(|error| error.abs() / count)
            .sum::<f64>();
        let rmse = residuals
            .map(|error| error / count.sqrt())
            .fold(0_f64, f64::hypot);
        if !mae.is_finite() || !rmse.is_finite() {
            return Err("ensemble diagnostics exceed the finite numerical domain");
        }
        json!({"mae":mae,"rmse":rmse})
    };
    let assumptions: Vec<String> = if payload["assumptions"].is_null() {
        Vec::new()
    } else {
        serde_json::from_value(payload["assumptions"].clone())
            .map_err(|_| "assumptions must be an array of strings")?
    };
    Ok(
        json!({"workflow":"regime_ensemble","stability":"provisional","backend":"rust_native",
        "time":time,"mean":mean,"diagnostics":diagnostics,
        "metadata":{"weights":weights,"regimes":predictions.keys().collect::<Vec<_>>(),"assumptions":assumptions}}),
    )
}
