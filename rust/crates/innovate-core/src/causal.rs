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
        _ => Err("unknown causal summary method"),
    }
}
