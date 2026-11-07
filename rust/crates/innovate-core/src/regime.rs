//! Two-regime scalar diffusion with explicit historical and time-based semantics.
use crate::validation::{numeric_array, numeric_array_from_aliases};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBinding, KernelBindingError, KernelOperation, KernelRequest,
    KernelResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default = "base")]
    base_model: String,
    #[serde(default = "mode")]
    prediction_mode: String,
    #[serde(default)]
    split_index: Option<usize>,
    #[serde(default)]
    split_time: Option<f64>,
    #[serde(default = "two")]
    regime_count: usize,
    #[serde(default = "penalty")]
    detector_penalty: f64,
    #[serde(default = "jump")]
    detector_jump: usize,
    #[serde(default)]
    detected_breakpoints: Vec<usize>,
    #[serde(default)]
    residual_noise: f64,
}
fn base() -> String {
    "bass".into()
}
fn mode() -> String {
    "historical_index".into()
}
fn two() -> usize {
    2
}
fn penalty() -> f64 {
    3.
}
fn jump() -> usize {
    5
}
// Independently written exhaustive dynamic programming over a subsampled L2 grid.
// Each candidate segment uses Welford's centered sum of squares, not raw moments.
fn detect(y: &[f64], penalty: f64, jump: usize) -> Result<Vec<usize>, String> {
    if y.len() > 2000 {
        return Err("automatic detection supports at most 2000 observations; provide split_index for larger fits".into());
    }
    let n = y.len();
    let mut grid = vec![0];
    grid.extend((jump..n).step_by(jump).filter(|x| *x >= 2));
    grid.push(n);
    let mut scores = vec![f64::INFINITY; grid.len()];
    let mut previous = vec![0; grid.len()];
    scores[0] = 0.;
    for i in 1..grid.len() {
        for j in 0..i {
            if grid[i] - grid[j] < 2 || !scores[j].is_finite() {
                continue;
            }
            let mut mean = 0.;
            let mut sse = 0.;
            for (k, v) in y[grid[j]..grid[i]].iter().enumerate() {
                let delta = v - mean;
                mean += delta / (k + 1) as f64;
                sse += delta * (v - mean);
            }
            let cost = scores[j] + sse.max(0.) + penalty;
            if !cost.is_finite() {
                return Err("changepoint objective overflow".into());
            }
            if cost < scores[i] {
                scores[i] = cost;
                previous[i] = j;
            }
        }
    }
    if !scores[grid.len() - 1].is_finite() {
        return Err("no valid changepoint partition".into());
    }
    let mut out = vec![];
    let mut i = grid.len() - 1;
    while i > 0 {
        out.push(grid[i]);
        i = previous[i];
    }
    out.reverse();
    Ok(out)
}
fn curve(
    base: &str,
    parameters: &Map<String, Value>,
    time: &[f64],
    op: KernelOperation,
) -> Result<Vec<f64>, KernelBindingError> {
    let result = KernelBinding::new()
        .predict_model(
            base,
            json!({"inputs":{"time":time},"parameters":parameters}),
        )?
        .result
        .ok_or_else(|| KernelBindingError::invalid_request(op, "missing regime prediction"))?;
    numeric_array(&result["values"], "regime prediction", op)
}
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    let p = r
        .payload
        .as_object()
        .ok_or_else(|| invalid("payload must be an object"))?;
    if p.keys().any(|k| {
        ![
            "inputs",
            "constructor_kwargs",
            "model_kwargs",
            "state",
            "parameters",
            "fit_options",
            "base_fit_options",
            "regime_fit_options",
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid("unknown regime payload option"));
    }
    for v in [
        p.get("predict_kwargs"),
        p.get("fitter_options"),
        p.get("state").and_then(|s| s.get("predict_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        if !v.as_object().is_some_and(Map::is_empty) {
            return Err(invalid("custom prediction/fitter options unavailable"));
        }
    }
    if p.get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != "regime_switching")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let mut o: Options = serde_json::from_value(
        p.get("constructor_kwargs")
            .or_else(|| p.get("model_kwargs"))
            .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let names: &[&str] = match o.base_model.as_str() {
        "bass" => &["p", "q", "m"],
        "logistic" => &["L", "k", "x0"],
        "gompertz" => &["a", "b", "c"],
        "fisher_pry" => &["alpha", "t0"],
        _ => return Err(invalid("unsupported regime base curve")),
    };
    if !["historical_index", "fixed_time"].contains(&o.prediction_mode.as_str())
        || !(1..=2).contains(&o.regime_count)
        || !o.detector_penalty.is_finite()
        || o.detector_penalty <= 0.
        || o.detector_jump == 0
        || o.detector_jump > 2000
        || o.split_time.is_some_and(|t| !t.is_finite())
        || !o.residual_noise.is_finite()
        || o.residual_noise < 0.
    {
        return Err(invalid(
            "invalid regime mode/count/detector/boundary/noise options",
        ));
    }
    let empty = Map::new();
    let input = p
        .get("inputs")
        .map(|v| {
            v.as_object()
                .ok_or_else(|| invalid("inputs must be an object"))
        })
        .transpose()?
        .unwrap_or(&empty);
    if input.contains_key("covariates") {
        return Err(invalid("regime base covariates not ported"));
    }
    let time = if op == KernelOperation::SummarizeModel
        && !input.contains_key("time")
        && !input.contains_key("t")
    {
        vec![]
    } else {
        numeric_array_from_aliases(input, &["time", "t"], op)?
    };
    if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("time must be finite and strictly increasing"));
    }
    let observed = if op == KernelOperation::FitModel
        || op == KernelOperation::DiagnoseModel
        || input.contains_key("observed")
        || input.contains_key("y")
    {
        let y = numeric_array_from_aliases(input, &["observed", "y"], op)?;
        if y.len() != time.len() || y.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(invalid(
                "observations must be finite nonnegative and aligned",
            ));
        }
        Some(y)
    } else {
        None
    };
    let mut parameters = Map::new();
    let mut segment_diagnostics = vec![];
    let index;
    if op == KernelOperation::FitModel {
        if time.len() < 2 {
            return Err(invalid("regime fit requires at least two observations"));
        }
        if p.get("fit_options")
            .is_some_and(|v| !v.as_object().is_some_and(Map::is_empty))
        {
            return Err(invalid(
                "use base_fit_options or regime_fit_options; split is selected separately",
            ));
        }
        let y = observed.as_ref().unwrap();
        if o.split_index.is_some() && o.split_time.is_some() {
            return Err(invalid("provide split_index or split_time, not both"));
        }
        index = if let Some(boundary) = o.split_time {
            let i = time.partition_point(|t| *t < boundary);
            if i == 0 || i == time.len() {
                return Err(invalid("split_time must leave observations on both sides"));
            }
            o.detected_breakpoints.clear();
            i
        } else if let Some(i) = o.split_index {
            if i == 0 || i >= time.len() {
                return Err(invalid("split_index must lie strictly inside fitted data"));
            }
            o.detected_breakpoints.clear();
            i
        } else {
            o.detected_breakpoints =
                detect(y, o.detector_penalty, o.detector_jump).map_err(|e| invalid(&e))?;
            let detected = if o.detected_breakpoints.len() > 1 {
                o.detected_breakpoints[0]
            } else {
                0
            };
            if detected == 0 || detected >= time.len() - 1 {
                (time.len() / 2).max(1)
            } else {
                detected
            }
        };
        let mut segments: Vec<_> = [(0, index), (index, time.len())]
            .into_iter()
            .filter(|(s, e)| e - s >= 2)
            .collect();
        if segments.is_empty() {
            segments.push((0, time.len()));
        }
        o.regime_count = segments.len();
        let individual = p
            .get("regime_fit_options")
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| invalid("regime_fit_options must be an array"))
            })
            .transpose()?;
        if individual.is_some_and(|v| v.len() != segments.len())
            || (individual.is_some() && p.contains_key("base_fit_options"))
        {
            return Err(invalid(
                "provide one fit option object per retained regime, or common base_fit_options",
            ));
        }
        for (regime, (start, end)) in segments.into_iter().enumerate() {
            let fit_options = individual
                .map(|a| a[regime].clone())
                .or_else(|| p.get("base_fit_options").cloned())
                .unwrap_or_else(|| json!({}));
            let result=KernelBinding::new().fit_model(&o.base_model,json!({"inputs":{"time":time[start..end],"observed":y[start..end]},"fit_options":fit_options}))?.result.ok_or_else(||invalid("missing regime fit"))?;
            for name in names {
                parameters.insert(
                    format!("regime_{regime}_{name}"),
                    result["parameters"][name].clone(),
                );
            }
            segment_diagnostics.push(result["diagnostics"].clone());
        }
        parameters.insert("changepoint_index".into(), json!(index));
        o.split_time = o.split_time.or(Some(time[index]));
    } else {
        parameters = p
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| p.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?
            .clone();
        let i = parameters
            .get("changepoint_index")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("changepoint_index required"))?;
        if !i.is_finite() || i < 1. || i.fract() != 0. || i > 1000000000. {
            return Err(invalid(
                "changepoint_index must be integer in 1..=1000000000",
            ));
        }
        index = i as usize;
        if parameters.len() != o.regime_count * names.len() + 1 {
            return Err(invalid("regime parameter names/count mismatch"));
        }
    }
    let mut regimes = vec![];
    for i in 0..o.regime_count {
        let mut params = Map::new();
        for name in names {
            params.insert(
                name.to_string(),
                parameters
                    .get(&format!("regime_{i}_{name}"))
                    .cloned()
                    .ok_or_else(|| invalid("missing regime parameter"))?,
            );
        }
        regimes.push(params);
    }
    let evaluation = if time.is_empty() {
        vec![0.]
    } else {
        time.clone()
    };
    let mut values = curve(&o.base_model, &regimes[0], &evaluation, op)?;
    if o.regime_count == 2 {
        // Always validate the second curve, even if this batch never uses it.
        let second = curve(&o.base_model, &regimes[1], &evaluation, op)?;
        if o.prediction_mode == "historical_index" {
            if index < time.len().saturating_sub(1) {
                let offset = values[index - 1] - second[index];
                for i in index..time.len() {
                    values[i] = second[i] + offset;
                }
            }
        } else {
            let boundary = o
                .split_time
                .ok_or_else(|| invalid("fixed_time mode requires split_time"))?;
            let first_at = curve(&o.base_model, &regimes[0], &[boundary], op)?[0];
            let second_at = curve(&o.base_model, &regimes[1], &[boundary], op)?[0];
            let offset = first_at - second_at;
            for (i, t) in time.iter().enumerate() {
                if *t >= boundary {
                    values[i] = second[i] + offset;
                }
            }
        }
    }
    if time.is_empty() {
        values.clear();
    }
    let mut maximum = 0_f64;
    for v in &mut values {
        if !v.is_finite() {
            return Err(invalid("regime adjustment overflow"));
        }
        maximum = maximum.max(*v);
        *v = maximum;
    }
    let increments: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if values.len() == 1 {
                *v
            } else if i == 0 {
                0.
            } else {
                v - values[i - 1]
            }
        })
        .collect();
    if op == KernelOperation::FitModel {
        let residuals: Vec<_> = observed
            .as_ref()
            .unwrap()
            .iter()
            .zip(&values)
            .map(|(y, v)| y - v)
            .collect();
        let mean = residuals
            .iter()
            .map(|r| r / residuals.len() as f64)
            .sum::<f64>();
        o.residual_noise = (residuals
            .iter()
            .map(|r| (r - mean).powi(2) / (residuals.len() - 1) as f64)
            .sum::<f64>())
        .sqrt();
        if !o.residual_noise.is_finite() {
            return Err(invalid("regime residual noise overflow"));
        }
    }
    let predictions = json!({"values":values,"increments":increments,"shape":[time.len()],"dtype":"float64","metadata":{"prediction_mode":o.prediction_mode,"rate_semantics":"historical sample differences; singleton returns level, not derivative","batch_semantics":"nonnegative cumulative maximum; historical_index also splits by each batch position","simulation_semantics":"deterministic point path; stochastic draws unavailable"}});
    let state = json!({"model_key":"regime_switching","parameters":parameters,"constructor_kwargs":o,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut out = json!({"model_key":"regime_switching","model_name":"RegimeSwitchingDiffusionModel","parameters":parameters,"state":state,"predictions":predictions,"segment_diagnostics":segment_diagnostics,"details":{"changepoint_index":index,"regime_count":o.regime_count,"split_time":o.split_time,"prediction_mode":o.prediction_mode,"detected_breakpoints":o.detected_breakpoints,"detector":"unpruned_penalized_l2_grid; min_size=2; ascending-predecessor tie order","residual_noise":o.residual_noise}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some(y) = observed {
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &values,
                "RegimeSwitchingDiffusionModel",
                parameters.len() + 1,
            );
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"split-selection degrees of freedom are not qualified"});
            d["warnings"].as_array_mut().unwrap().push(json!("Regimes are fitted independently before continuity adjustment; adjusted predictions need not fit observations. Split detection is a heuristic and residual deviation is not calibrated uncertainty."));
            out["diagnostics"] = d;
        }
        out
    };
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        result: Some(result),
        error: None,
        metadata: json!({"runtime":"rust_native","family":"regime_switching"}),
    })
}
#[cfg(test)]
mod tests {
    use super::detect;
    #[test]
    fn native_grid_matches_independent_ruptures_fixtures() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/regime_ruptures.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let y: Vec<_> = case["observed"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert_eq!(
                serde_json::json!(detect(&y, 3., 5).unwrap()),
                case["breakpoints"],
                "{}",
                case["name"]
            );
        }
    }
}
