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
