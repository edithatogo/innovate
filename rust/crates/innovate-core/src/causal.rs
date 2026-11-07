//! Descriptive summaries of supplied outcomes and effect estimates.

use serde_json::{Value, json};

fn series(payload: &Value, key: &str) -> Result<Vec<f64>, &'static str> {
    serde_json::from_value(payload[key].clone()).map_err(|_| "expected a numeric array")
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| {
        let scale = values.iter().copied().map(f64::abs).fold(0., f64::max);
        if scale == 0. {
            return 0.;
        }
        let average = values.iter().map(|v| v / scale).sum::<f64>() / values.len() as f64;
        average.clamp(-1., 1.) * scale
    })
}

fn outcome_mean(payload: &Value, key: &str) -> Result<f64, &'static str> {
    mean(&series(payload, key)?).ok_or("outcome arrays must be nonempty")
}

fn finite(value: f64) -> Result<f64, &'static str> {
    value
        .is_finite()
        .then_some(value)
        .ok_or("causal summary exceeded finite numerical domain")
}

pub(crate) fn summarize(payload: &Value) -> Result<Value, &'static str> {
    match payload["method"].as_str() {
        Some("pre_post") => {
            let pre_treated = outcome_mean(payload, "pre_treated")?;
            let post_treated = outcome_mean(payload, "post_treated")?;
            let pre_control = outcome_mean(payload, "pre_control")?;
            let post_control = outcome_mean(payload, "post_control")?;
            let treated_effect = finite(post_treated - pre_treated)?;
            let control_trend = finite(post_control - pre_control)?;
            Ok(json!({
                "pre_treated_mean":pre_treated,"post_treated_mean":post_treated,
                "pre_control_mean":pre_control,"post_control_mean":post_control,
                "treated_effect":treated_effect,"control_trend":control_trend,
                "diff_in_diff":finite(treated_effect - control_trend)?
            }))
        }
        Some("counterfactual") => {
            let actual = outcome_mean(payload, "actual")?;
            let counterfactual = outcome_mean(payload, "counterfactual")?;
            let effect = finite(actual - counterfactual)?;
            let percent_change = if counterfactual == 0. {
                0.
            } else {
                finite(100. * (effect / counterfactual))?
            };
            Ok(
                json!({"actual_mean":actual,"counterfactual_mean":counterfactual,
                      "effect":effect,"percent_change":percent_change}),
            )
        }
        Some("event_study") => {
            let periods = series(payload, "periods")?;
            let effects = series(payload, "coefficients")?;
            let errors = series(payload, "standard_errors")?;
            if periods.len() != effects.len()
                || effects.len() != errors.len()
                || errors.iter().any(|&error| error < 0.)
            {
                return Err(
                    "event-study arrays must align and standard errors must be nonnegative",
                );
            }
            let mut pre = Vec::new();
            let mut post = Vec::new();
            let mut pre_errors = Vec::new();
            let mut event = None;
            for ((period, effect), error) in periods.into_iter().zip(effects).zip(errors) {
                if period < 0. {
                    pre.push(effect);
                    pre_errors.push(error);
                } else if period > 0. {
                    post.push(effect);
                } else {
                    event.get_or_insert(effect);
                }
            }
            Ok(
                json!({"pre_trend":mean(&pre),"pre_trend_se":mean(&pre_errors),
                      "post_effects":mean(&post),"event_period_coef":event,
                      "n_pre_periods":pre.len(),"n_post_periods":post.len()}),
            )
        }
        Some("heterogeneous_effects") => {
            let labels: Vec<String> = serde_json::from_value(payload["group_labels"].clone())
                .map_err(|_| "group_labels must be an array of strings")?;
            let effects = series(payload, "effects")?;
            let errors = series(payload, "standard_errors")?;
            if effects.is_empty()
                || labels.len() != effects.len()
                || effects.len() != errors.len()
                || errors.iter().any(|&error| error < 0.)
            {
                return Err(
                    "group arrays must be nonempty and aligned with nonnegative standard errors",
                );
            }
            let mut groups = serde_json::Map::new();
            for ((label, &effect), error) in labels.into_iter().zip(&effects).zip(errors) {
                let interval = json!({"effect":effect,"se":error,
                    "ci_lower":finite(effect - 1.96 * error)?,
                    "ci_upper":finite(effect + 1.96 * error)?});
                if label.is_empty() || groups.insert(label, interval).is_some() {
                    return Err("group labels must be nonempty and unique");
                }
            }
            let min = effects.iter().copied().fold(f64::INFINITY, f64::min);
            let max = effects.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Ok(
                json!({"n_groups":groups.len(),"by_group":groups,"overall_effect":mean(&effects),
                      "max_effect":max,"min_effect":min,"effect_range":finite(max-min)?}),
            )
        }
        Some("e_value") => {
            let risk_ratio = payload["point_estimate"]
                .as_f64()
                .filter(|&v| v > 0.)
                .ok_or("E-value requires a positive risk ratio")?;
            let ratio = risk_ratio.max(risk_ratio.recip());
            let e_value = finite(ratio + ratio.sqrt() * (ratio - 1.).sqrt())?;
            Ok(json!({"e_value":e_value}))
        }
        Some("sensitivity_scenario") => {
            let point = payload["point_estimate"]
                .as_f64()
                .ok_or("point_estimate must be numeric")?;
            let gamma = payload["gamma"]
                .as_f64()
                .filter(|&v| v >= 1.)
                .ok_or("gamma must be at least one")?;
            let margin = finite(point.abs() * 0.1 * (gamma - 1.))?;
            let (lower, upper) = match payload.get("direction").map_or(Some("both"), Value::as_str)
            {
                Some("both") => (point - margin, point + margin),
                Some("positive") => (point, point + margin),
                Some("negative") => (point - margin, point),
                _ => return Err("direction must be both, positive or negative"),
            };
            Ok(
                json!({"gamma":gamma,"point_estimate":point,"lower_bound":finite(lower)?,
                "upper_bound":finite(upper)?,"robustness_margin":margin,"method":"historical_heuristic"}),
            )
        }
        Some("covariate_balance") => covariate_balance(payload),
        _ => Err("unknown causal summary method"),
    }
}

fn covariate_balance(payload: &Value) -> Result<Value, &'static str> {
    let treated = payload["treated_covariates"]
        .as_object()
        .ok_or("treated_covariates must be an object")?;
    let control = payload["control_covariates"]
        .as_object()
        .ok_or("control_covariates must be an object")?;
    let threshold = payload
        .get("threshold")
        .map_or(Some(0.1), Value::as_f64)
        .filter(|v| (0. ..=1.).contains(v))
        .ok_or("balance threshold must be between zero and one")?;
    let mut scores = std::collections::BTreeMap::new();
    for key in treated.keys().filter(|key| control.contains_key(*key)) {
        let mut a = series(&payload["treated_covariates"], key)?;
        let mut b = series(&payload["control_covariates"], key)?;
        if a.len() < 2 || b.len() < 2 {
            return Err("covariate balance requires at least two observations in each group");
        }
        let scale = a.iter().chain(&b).copied().map(f64::abs).fold(0., f64::max);
        if scale > 0. {
            for value in a.iter_mut().chain(&mut b) {
                *value /= scale;
            }
        }
        let mean_a = mean(&a).expect("validated group length");
        let mean_b = mean(&b).expect("validated group length");
        let squared = |values: &[f64], average: f64| {
            values
                .iter()
                .map(|&value| (value - average).powi(2))
                .sum::<f64>()
        };
        let pooled =
            ((squared(&a, mean_a) + squared(&b, mean_b)) / (a.len() + b.len() - 2) as f64).sqrt();
        let score = if pooled > 0. {
            ((mean_a - mean_b).abs() / pooled).min(1.)
        } else {
            0.
        };
        scores.insert(key, score);
    }
    let values: Vec<_> = scores.values().copied().collect();
    let balanced = values.iter().filter(|&&score| score <= threshold).count();
    Ok(
        json!({"smd_by_covariate":scores,"n_balanced":balanced,"n_total":values.len(),
        "all_balanced":balanced == values.len(),
        "max_smd":values.iter().copied().fold(0.,f64::max),
        "mean_smd":mean(&values).unwrap_or(0.)}),
    )
}
