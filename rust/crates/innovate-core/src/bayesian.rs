//! Bounded posterior quadrature using the native predictor.
use serde_json::{Value, json};

pub(crate) fn grid(payload: &Value) -> Result<Value, &'static str> {
    let model = payload["forecast_model"]
        .as_str()
        .ok_or("forecast_model is required")?;
    let mut prediction = payload["prediction"].clone();
    let parameters = prediction["parameters"]
        .as_object()
        .ok_or("explicit parameters are required")?;
    let axes = match payload.get("axes") {
        Some(value) => {
            if ["parameter", "bounds", "points"]
                .iter()
                .any(|key| payload.get(key).is_some())
            {
                return Err("provide axes or a single parameter grid, not both");
            }
            let axes = value
                .as_array()
                .filter(|axes| axes.len() == 2)
                .ok_or("joint posterior requires two axes")?;
            axes.iter().map(axis).collect::<Result<Vec<_>, _>>()?
        }
        None => vec![axis(payload)?],
    };
    if axes.iter().any(|(name, _)| !parameters.contains_key(*name)) {
        return Err("unknown posterior parameter");
    }
    if axes.len() == 2 && axes[0].0 == axes[1].0 {
        return Err("posterior axes must name distinct parameters");
    }
    let cells = axes.iter().map(|(_, grid)| grid.len()).product::<usize>();
    if cells > 10_000 {
        return Err("posterior grid exceeds 10000 cells");
    }
    let sigma = payload["noise_sd"]
        .as_f64()
        .filter(|sd| sd.is_finite() && *sd > 0.)
        .ok_or("noise_sd must be finite and positive")?;
    let observed: Vec<f64> = serde_json::from_value(payload["observed"].clone())
        .map_err(|_| "observed must be a finite numeric array")?;
    if observed.is_empty() || observed.len() > 500 || observed.iter().any(|y| !y.is_finite()) {
        return Err("posterior requires 1 to 500 finite observations");
    }
    let kernel = crate::KernelBinding::new();
    let mut log_weights = Vec::with_capacity(cells);
    let stride = axes.get(1).map_or(1, |(_, grid)| grid.len());
    for cell in 0..cells {
        let (name, grid) = &axes[0];
        prediction["parameters"][*name] = json!(grid[cell / stride]);
        if let Some((name, grid)) = axes.get(1) {
            prediction["parameters"][*name] = json!(grid[cell % stride]);
        }
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
    let mut result = json!({"posterior_mass":weights,
        "method":"midpoint_grid_quadrature","prior":"uniform_within_bounds","likelihood":"independent_gaussian_known_sd"});
    let means: Vec<_> = axes
        .iter()
        .enumerate()
        .map(|(axis, (_, grid))| {
            weights
                .iter()
                .enumerate()
                .map(|(cell, weight)| {
                    let index = if axis == 0 {
                        cell / stride
                    } else {
                        cell % stride
                    };
                    grid[index] * weight
                })
                .sum::<f64>()
        })
        .collect();
    if means.iter().any(|mean| !mean.is_finite()) {
        return Err("posterior mean exceeds finite numerical domain");
    }
    if axes.len() == 1 {
        result["parameter"] = json!(axes[0].0);
        result["grid"] = json!(axes[0].1);
        result["posterior_mean"] = json!(means[0]);
    } else {
        result["axes"] = json!(
            axes.iter()
                .map(|(name, grid)| json!({"parameter":name,"grid":grid}))
                .collect::<Vec<_>>()
        );
        result["shape"] = json!([axes[0].1.len(), stride]);
        result["posterior_means"] = json!(means);
    }
    Ok(result)
}

fn axis(payload: &Value) -> Result<(&str, Vec<f64>), &'static str> {
    let name = payload["parameter"]
        .as_str()
        .ok_or("parameter is required")?;
    let [lower, upper]: [f64; 2] = serde_json::from_value(payload["bounds"].clone())
        .map_err(|_| "bounds must contain two finite numbers")?;
    let width = upper - lower;
    if !width.is_finite() || width <= 0. {
        return Err("posterior bounds must be finite and increasing");
    }
    let points = payload["points"]
        .as_u64()
        .filter(|n| (3..=1001).contains(n))
        .ok_or("points must be between 3 and 1001")? as usize;
    let grid: Vec<_> = (0..points)
        .map(|index| lower + width * ((index as f64 + 0.5) / points as f64))
        .collect();
    if grid.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("posterior grid points must be distinct");
    }
    Ok((name, grid))
}
