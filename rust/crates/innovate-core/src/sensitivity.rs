//! Deterministic sensitivity summaries for decision reports.
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdInput {
    name: String,
    metric: String,
    thresholds: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdRequest {
    outcome_values: Vec<f64>,
    inputs: Vec<ThresholdInput>,
}

pub(crate) fn thresholds(payload: &Value) -> Result<Value, &'static str> {
    let request = ThresholdRequest::deserialize(payload).map_err(
        |_| "threshold sensitivity requires numeric outcome_values and named threshold inputs",
    )?;
    let mut outcomes = request.outcome_values;
    if outcomes.is_empty() {
        return Err("outcome_values must be nonempty");
    }
    outcomes.sort_unstable_by(f64::total_cmp);
    let min = outcomes[0];
    let max = outcomes[outcomes.len() - 1];
    let scale = min.abs().max(max.abs());
    let mean = if scale == 0. {
        0.
    } else {
        (outcomes.iter().map(|value| value / scale).sum::<f64>() / outcomes.len() as f64) * scale
    };
    let rows: Vec<_> = request
        .inputs
        .iter()
        .flat_map(|input| {
            input.thresholds.iter().map(|threshold| {
                let total = outcomes.len();
                let meet = total - outcomes.partition_point(|value| value < threshold);
                json!({"name":input.name,"metric":input.metric,"threshold":threshold,
                "n_meet":meet,"n_total":total,"share_meet":meet as f64 / total as f64})
            })
        })
        .collect();
    Ok(
        json!({"kind":"threshold_sensitivity","rows":rows,"deterministic":true,
        "outcome_summary":{"min":min,"max":max,"mean":mean}}),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PerturbationRequest {
    model_key: String,
    prediction: Value,
    inputs: Vec<Perturbation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Perturbation {
    name: String,
    deltas: Vec<f64>,
}

pub(crate) fn perturb_parameters(payload: &Value) -> Result<Value, &'static str> {
    let mut request = PerturbationRequest::deserialize(payload).map_err(
        |_| "parameter sensitivity requires model_key, prediction and named delta lists",
    )?;
    if request.prediction.get("state").is_some() {
        return Err("parameter sensitivity requires explicit parameters");
    }
    let parameters = request.prediction["parameters"]
        .as_object()
        .ok_or("parameter sensitivity requires explicit parameters")?;
    for input in &request.inputs {
        if !parameters.get(&input.name).is_some_and(Value::is_number) {
            return Err("parameter sensitivity requires an existing numeric parameter");
        }
    }
    let kernel = crate::KernelBinding::new();
    let outcome = |prediction: &Value| {
        kernel
            .predict_model(&request.model_key, prediction.clone())
            .map_err(|_| "parameter sensitivity model prediction failed")?
            .result
            .and_then(|result| result["values"].as_array()?.last()?.as_f64())
            .ok_or("parameter sensitivity requires a scalar prediction path")
    };
    let baseline_outcome = outcome(&request.prediction)?;
    let mut rows = Vec::new();
    for input in request.inputs {
        let original = request.prediction["parameters"][&input.name].clone();
        let baseline = original.as_f64().unwrap();
        for delta in input.deltas {
            let value = baseline + delta;
            if !value.is_finite() {
                return Err("perturbed parameter must be finite");
            }
            request.prediction["parameters"][&input.name] = json!(value);
            let alternative = outcome(&request.prediction)?;
            let change = alternative - baseline_outcome;
            if !change.is_finite() {
                return Err("absolute outcome change must be finite");
            }
            let relative = (baseline_outcome != 0.).then(|| change / baseline_outcome);
            let elasticity = relative
                .filter(|_| baseline != 0. && delta != 0.)
                .map(|relative| relative / (delta / baseline));
            rows.push(
                json!({"parameter":input.name,"baseline":baseline,"delta":delta,"value":value,
                "baseline_outcome":baseline_outcome,"outcome":alternative,"absolute_change":change,
                "relative_change":relative.filter(|value| value.is_finite()),
                "elasticity":elasticity.filter(|value| value.is_finite())}),
            );
        }
        request.prediction["parameters"][&input.name] = original;
    }
    Ok(
        json!({"kind":"parameter_perturbation","rows":rows,"deterministic":true,
        "outcome":"final_scalar_prediction"}),
    )
}
