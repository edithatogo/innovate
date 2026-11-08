//! Historical exponentially smoothed residual proxy, with explicit saved state.
use crate::validation::{numeric_array, numeric_array_from_aliases};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBinding, KernelBindingError, KernelOperation, KernelRequest,
    KernelResponse,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default = "base")]
    base_model: String,
    #[serde(default = "smoothing")]
    smoothing: f64,
    #[serde(default)]
    latent_time: Vec<f64>,
    #[serde(default)]
    latent_values: Vec<f64>,
}
fn base() -> String {
    "bass".into()
}
fn smoothing() -> f64 {
    0.3
}
fn interpolate(t: f64, time: &[f64], values: &[f64]) -> f64 {
    if time.is_empty() {
        return 0.;
    }
    let i = time.partition_point(|x| *x <= t);
    if i == 0 {
        values[0]
    } else if i == time.len() {
        values[i - 1]
    } else {
        let fraction = (t - time[i - 1]) / (time[i] - time[i - 1]);
        (1. - fraction) * values[i - 1] + fraction * values[i]
    }
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
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid("unknown latent payload option"));
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
        .is_some_and(|m| m != "latent_process")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let mut options: Options = serde_json::from_value(
        p.get("constructor_kwargs")
            .or_else(|| p.get("model_kwargs"))
            .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let names: &[&str] = match options.base_model.as_str() {
        "bass" => &["p", "q", "m"],
        "logistic" => &["L", "k", "x0"],
        "gompertz" => &["a", "b", "c"],
        "fisher_pry" => &["alpha", "t0"],
        _ => return Err(invalid("unsupported latent base curve")),
    };
    if !options.smoothing.is_finite() || !(0. ..=1.).contains(&options.smoothing) {
        return Err(invalid("smoothing must be in [0,1]"));
    }
    if options.latent_time.len() != options.latent_values.len()
        || options
            .latent_time
            .iter()
            .chain(&options.latent_values)
            .any(|x| !x.is_finite())
        || options
            .latent_time
            .windows(2)
            .any(|w| w[0] >= w[1] || !(w[1] - w[0]).is_finite())
    {
        return Err(invalid(
            "latent arrays must be finite, aligned and have strictly increasing finite-span times",
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
        return Err(invalid("latent base covariates not ported"));
    }
    let time = if op == KernelOperation::SummarizeModel
        && !input.contains_key("time")
        && !input.contains_key("t")
    {
        vec![]
    } else {
        numeric_array_from_aliases(input, &["time", "t"], op)?
    };
    if time.iter().any(|x| !x.is_finite()) || time.windows(2).any(|w| w[0] >= w[1]) {
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
    let mut params = Map::new();
    let mut base_diagnostics = Value::Null;
    if op == KernelOperation::FitModel {
        if p.get("fit_options")
            .is_some_and(|v| !v.as_object().is_some_and(Map::is_empty))
        {
            return Err(invalid(
                "use base_fit_options; latent smoothing is held fixed",
            ));
        }
        if !options.latent_time.is_empty() {
            return Err(invalid(
                "fit estimates latent state; omit saved latent arrays",
            ));
        }
        let result=KernelBinding::new().fit_model(&options.base_model,json!({"inputs":{"time":time,"observed":observed},"fit_options":p.get("base_fit_options").cloned().unwrap_or_else(||json!({}))}))?.result.ok_or_else(||invalid("missing base fit"))?;
        for name in names {
            params.insert(format!("base_{name}"), result["parameters"][name].clone());
        }
        params.insert("latent_smoothing".into(), json!(options.smoothing));
        base_diagnostics = result["diagnostics"].clone();
    } else {
        params = p
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| p.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?
            .clone();
        if params.len() != names.len() + 2 {
            return Err(invalid(
                "latent parameter names must match base curve, smoothing and noise",
            ));
        }
        options.smoothing = params
            .get("latent_smoothing")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("latent_smoothing required"))?;
        let noise = params
            .get("latent_noise")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("latent_noise required"))?;
        if !noise.is_finite()
            || noise < 0.
            || !options.smoothing.is_finite()
            || !(0. ..=1.).contains(&options.smoothing)
        {
            return Err(invalid(
                "latent smoothing must be in [0,1] and noise finite nonnegative",
            ));
        }
    }
    let mut base_params = Map::new();
    for name in names {
        base_params.insert(
            name.to_string(),
            params
                .get(&format!("base_{name}"))
                .cloned()
                .ok_or_else(|| invalid("missing base parameter"))?,
        );
    }
    let eval = if time.is_empty() {
        vec![0.]
    } else {
        time.clone()
    };
    let b = KernelBinding::new()
        .predict_model(
            &options.base_model,
            json!({"inputs":{"time":eval},"parameters":base_params}),
        )?
        .result
        .ok_or_else(|| invalid("missing base predictions"))?;
    let baseline = numeric_array(&b["values"], "base predictions", op)?;
    if op == KernelOperation::FitModel {
        let residuals: Vec<_> = observed
            .as_ref()
            .unwrap()
            .iter()
            .zip(&baseline)
            .map(|(y, b)| y - b)
            .collect();
        let mut previous = 0.;
        for (i, residual) in residuals.iter().enumerate() {
            previous = if i == 0 {
                *residual
            } else {
                options.smoothing * residual + (1. - options.smoothing) * previous
            };
            options.latent_values.push(previous);
        }
        options.latent_time = time.clone();
        let mean = residuals
            .iter()
            .map(|r| r / residuals.len() as f64)
            .sum::<f64>();
        let noise = if residuals.len() > 1 {
            (residuals
                .iter()
                .map(|r| (r - mean).powi(2) / (residuals.len() - 1) as f64)
                .sum::<f64>())
            .sqrt()
        } else {
            0.
        };
        if !noise.is_finite() || options.latent_values.iter().any(|x| !x.is_finite()) {
            return Err(invalid("latent residual estimation overflow"));
        }
        params.insert("latent_noise".into(), json!(noise));
    }
    let mut values = Vec::new();
    let mut maximum = 0_f64;
    for (i, t) in time.iter().enumerate() {
        let value = baseline[i] + interpolate(*t, &options.latent_time, &options.latent_values);
        if !value.is_finite() {
            return Err(invalid("latent prediction overflow"));
        }
        maximum = maximum.max(value);
        values.push(maximum);
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
    let predictions = json!({"values":values,"increments":increments,"shape":[time.len()],"dtype":"float64","metadata":{"latent_semantics":"exponentially smoothed residual proxy, not probabilistic state-space inference","rate_semantics":"historical sample increments, singleton returns level; not derivatives","batch_semantics":"nonnegative cumulative maximum within requested batch","simulation_semantics":"deterministic point path; stochastic draws unavailable"}});
    let kwargs = json!({"base_model":options.base_model,"smoothing":options.smoothing,"latent_time":options.latent_time,"latent_values":options.latent_values});
    let state = json!({"model_key":"latent_process","parameters":params,"constructor_kwargs":kwargs,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut out = json!({"model_key":"latent_process","model_name":"LatentProcessDiffusionModel","parameters":params,"state":state,"predictions":predictions,"base_diagnostics":base_diagnostics,"details":{"base_model":options.base_model,"latent_state_length":options.latent_time.len(),"latent_smoothing":options.smoothing,"latent_noise":params["latent_noise"]}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some(y) = observed {
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &values,
                "LatentProcessDiffusionModel",
                names.len() + 2,
            );
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"effective degrees of freedom for fitted residual smoothing are not qualified"});
            d["warnings"].as_array_mut().unwrap().push(json!("Latent residual state is estimated in-sample after baseline fitting with fixed smoothing; noise is residual sample deviation, not calibrated forecast uncertainty"));
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
        metadata: json!({"runtime":"rust_native","family":"latent_process"}),
    })
}
