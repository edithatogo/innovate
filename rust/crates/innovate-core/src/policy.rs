//! Historical post-hoc policy level multipliers; not calibrated survival hazards.
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
    #[serde(default)]
    event_times: Vec<f64>,
    #[serde(default)]
    event_effects: Vec<f64>,
    #[serde(default)]
    event_labels: Vec<String>,
    #[serde(default = "decay")]
    decay: f64,
}
fn base() -> String {
    "bass".into()
}
fn decay() -> f64 {
    0.5
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
        return Err(invalid("unknown policy payload option"));
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
        .is_some_and(|m| m != "policy_hazard")
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
        _ => return Err(invalid("unsupported policy base curve")),
    };
    let n = options.event_times.len();
    if options.event_effects.len() != n
        || options
            .event_times
            .iter()
            .chain(&options.event_effects)
            .any(|x| !x.is_finite())
    {
        return Err(invalid("event times/effects must be finite aligned arrays"));
    }
    if options.event_labels.is_empty() {
        options.event_labels = (0..n).map(|i| format!("event_{i}")).collect();
    }
    if options.event_labels.len() != n {
        return Err(invalid("event labels must match event times"));
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
        return Err(invalid("policy base covariates not ported"));
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
    let mut params = Map::new();
    let mut base_diagnostics = Value::Null;
    if op == KernelOperation::FitModel {
        if p.get("fit_options")
            .is_some_and(|v| !v.as_object().is_some_and(Map::is_empty))
        {
            return Err(invalid("use base_fit_options; policy decay is held fixed"));
        }
        let observed = numeric_array_from_aliases(input, &["observed", "y"], op)?;
        let result=KernelBinding::new().fit_model(&options.base_model,json!({"inputs":{"time":time,"observed":observed},"fit_options":p.get("base_fit_options").cloned().unwrap_or_else(||json!({}))}))?.result.ok_or_else(||invalid("missing base fit"))?;
        for name in names {
            params.insert(format!("base_{name}"), result["parameters"][name].clone());
        }
        params.insert("decay".into(), json!(options.decay));
        base_diagnostics = result["diagnostics"].clone();
    } else {
        params = p
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| p.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?
            .clone();
        if params.len() != names.len() + 1 {
            return Err(invalid(
                "policy parameter names must match base curve and decay",
            ));
        }
        options.decay = params
            .get("decay")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("decay required"))?;
    }
    if !options.decay.is_finite() || options.decay < 0. {
        return Err(invalid("decay must be finite and nonnegative"));
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
    let mut values = Vec::new();
    let mut maximum = 0_f64;
    for (i, t) in time.iter().enumerate() {
        let effect: f64 = options
            .event_times
            .iter()
            .zip(&options.event_effects)
            .filter(|(at, _)| *at <= t)
            .map(|(at, e)| {
                e * if options.decay == 0. {
                    1.
                } else {
                    (-options.decay * (t - at)).exp()
                }
            })
            .sum();
        let value = baseline[i] * (1. + effect);
        if !value.is_finite() {
            return Err(invalid("policy adjustment overflow"));
        }
        maximum = maximum.max(value);
        values.push(maximum);
    }
    let increments: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, v)| if i == 0 { 0. } else { v - values[i - 1] })
        .collect();
    let predictions = json!({"values":values,"increments":increments,"shape":[time.len()],"dtype":"float64","metadata":{"rate_semantics":"historical sample increments, not derivatives","policy_semantics":"post-hoc decaying level multiplier, not calibrated survival hazard or causal effect","batch_semantics":"cumulative maximum within requested batch"}});
    let kwargs = json!({"base_model":options.base_model,"event_times":options.event_times,"event_effects":options.event_effects,"event_labels":options.event_labels,"decay":options.decay});
    let state = json!({"model_key":"policy_hazard","parameters":params,"constructor_kwargs":kwargs,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let average_effect = if n == 0 {
            0.
        } else {
            options.event_effects.iter().map(|x| x / n as f64).sum()
        };
        if !average_effect.is_finite() {
            return Err(invalid("event summary overflow"));
        }
        let mut out = json!({"model_key":"policy_hazard","model_name":"PolicyHazardDiffusionModel","parameters":params,"state":state,"predictions":predictions,"base_diagnostics":base_diagnostics,"details":{"event_count":n,"event_labels":options.event_labels,"event_times":options.event_times,"average_effect":average_effect,"decay":options.decay}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if op != KernelOperation::SummarizeModel
            || input.contains_key("observed")
            || input.contains_key("y")
        {
            let y = numeric_array_from_aliases(input, &["observed", "y"], op)?;
            if y.len() != time.len() || y.iter().any(|x| !x.is_finite() || *x < 0.) {
                return Err(invalid(
                    "observations must be finite nonnegative and aligned",
                ));
            }
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &values,
                "PolicyHazardDiffusionModel",
                names.len() + 1,
            );
            d["warnings"].as_array_mut().unwrap().push(json!("Baseline fitting ignores policy multipliers and holds decay fixed; adjusted predictions need not fit observations"));
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
        metadata: json!({"runtime":"rust_native","family":"policy_hazard"}),
    })
}
