//! One-dimensional posterior quadrature using the native predictor.
use serde_json::{Value, json};

pub(crate) fn grid(payload: &Value) -> Result<Value, &'static str> {
    let model = payload["forecast_model"]
        .as_str()
        .ok_or("forecast_model is required")?;
    let name = payload["parameter"]
        .as_str()
        .ok_or("parameter is required")?;
    let mut prediction = payload["prediction"].clone();
    let parameters = prediction["parameters"]
        .as_object()
        .ok_or("explicit parameters are required")?;
    if !parameters.contains_key(name) {
        return Err("unknown posterior parameter");
    }
    let [lower, upper]: [f64; 2] = serde_json::from_value(payload["bounds"].clone())
        .map_err(|_| "bounds must contain two finite numbers")?;
    let width = upper - lower;
    if !width.is_finite() || width <= 0. {
        return Err("posterior bounds must be finite and increasing");
    }
    let sigma = payload["noise_sd"]
        .as_f64()
        .filter(|sd| sd.is_finite() && *sd > 0.)
        .ok_or("noise_sd must be finite and positive")?;
    let points = payload["points"]
        .as_u64()
        .filter(|n| (3..=1001).contains(n))
        .ok_or("points must be between 3 and 1001")? as usize;
    let observed: Vec<f64> = serde_json::from_value(payload["observed"].clone())
        .map_err(|_| "observed must be a finite numeric array")?;
    if observed.is_empty() || observed.len() > 500 || observed.iter().any(|y| !y.is_finite()) {
        return Err("posterior requires 1 to 500 finite observations");
    }
    let kernel = crate::KernelBinding::new();
    let mut grid = Vec::with_capacity(points);
    let mut log_weights = Vec::with_capacity(points);
    for index in 0..points {
        let value = lower + width * ((index as f64 + 0.5) / points as f64);
        prediction["parameters"][name] = json!(value);
        let forecast = kernel
            .predict_model(model, prediction.clone())
            .map_err(|_| "prediction failed at a posterior grid point")?
            .result
            .ok_or("posterior prediction returned no result")?;
        let values = forecast["values"]
            .as_array()
            .filter(|x| x.len() == observed.len())
            .ok_or("posterior requires aligned scalar predictions")?;
        let mut log_weight = 0.;
        for (value, y) in values.iter().zip(&observed) {
            let value = value
                .as_f64()
                .ok_or("posterior prediction must be numeric")?;
            log_weight -= 0.5 * ((value - y) / sigma).powi(2);
        }
        grid.push(value);
        log_weights.push(log_weight);
    }
    let peak = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return Err("all posterior grid likelihoods are nonfinite");
    }
    let mut weights: Vec<_> = log_weights
        .iter()
        .map(|log_weight| (log_weight - peak).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= total);
    let mean: f64 = grid
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value * weight)
        .sum();
    if !mean.is_finite() {
        return Err("posterior mean exceeds finite numerical domain");
    }
    Ok(
        json!({"parameter":name,"grid":grid,"posterior_mass":weights,"posterior_mean":mean,
        "method":"midpoint_grid_quadrature","prior":"uniform_within_bounds","likelihood":"independent_gaussian_known_sd"}),
    )
}
