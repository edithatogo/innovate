//! Descriptive summaries of supplied outcomes and effect estimates.

use serde::{Deserialize, Serialize};
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
        Some("uncertainty_interval") => {
            let number = |key: &str| {
                payload[key]
                    .as_f64()
                    .ok_or("interval fields must be numeric")
            };
            let lower = number("ci_lower")?;
            let upper = number("ci_upper")?;
            let point = number("point_estimate")?;
            if lower > upper {
                return Err("confidence interval bounds must be ordered");
            }
            let estimand = payload["estimand"]
                .as_str()
                .filter(|v| !v.is_empty())
                .ok_or("estimand must be a nonempty string")?;
            let method = payload
                .get("interval_method")
                .map_or(Some("bootstrap"), Value::as_str)
                .filter(|v| !v.is_empty())
                .ok_or("interval_method must be a nonempty string")?;
            let n_bootstrap = payload
                .get("n_bootstrap")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64()
                        .filter(|&n| n > 0)
                        .ok_or("n_bootstrap must be a positive integer")
                })
                .transpose()?;
            let width = finite(upper - lower)?;
            Ok(
                json!({"estimand":estimand,"point_estimate":point,"ci_lower":lower,
                "ci_upper":upper,"method":method,"n_bootstrap":n_bootstrap,
                "ci_width":width,"se":width / 3.92,"excludes_zero":lower > 0. || upper < 0.}),
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
            let margin = sensitivity_margin(point, gamma)?;
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
        Some("sensitivity_grid") => {
            let point = payload["point_estimate"]
                .as_f64()
                .ok_or("point_estimate must be numeric")?;
            let results = series(payload, "gamma_range")?
                .into_iter()
                .map(|gamma| {
                    Ok(json!({"gamma":gamma,"robustness":sensitivity_margin(point, gamma)?}))
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            Ok(json!({"results":results,"method":"historical_heuristic"}))
        }
        Some("paired_sensitivity") => {
            let treated = series(payload, "treated_outcomes")?;
            let control = series(payload, "control_outcomes")?;
            if treated.is_empty() || treated.len() != control.len() {
                return Err("paired sensitivity requires aligned nonempty outcomes");
            }
            let gamma = payload["gamma"]
                .as_f64()
                .filter(|&value| value >= 1.)
                .ok_or("gamma must be at least one")?;
            let differences = treated
                .into_iter()
                .zip(control)
                .map(|(treated, control)| finite(treated - control))
                .collect::<Result<Vec<_>, _>>()?;
            let point = mean(&differences).expect("validated pair count");
            let scale = differences.iter().copied().map(f64::abs).fold(0., f64::max);
            let margin = if scale == 0. {
                0.
            } else {
                let squares = differences
                    .iter()
                    .map(|value| (value / scale - point / scale).powi(2))
                    .sum::<f64>();
                finite(scale * (2. * gamma.ln() * squares.sqrt() / differences.len() as f64))?
            };
            Ok(
                json!({"point_estimate":point,"gamma":gamma,"matched_pairs":differences.len(),
                "lower_bound":finite(point - margin)?,"upper_bound":finite(point + margin)?,
                "method":"historical_paired_heuristic"}),
            )
        }
        Some("counterfactual_forecast") => counterfactual_forecast(payload),
        Some("model_card") => {
            let card: ModelCard = serde_json::from_value(payload.clone())
                .map_err(|_| "invalid causal model card fields")?;
            if [&card.name, &card.description, &card.estimand, &card.version]
                .iter()
                .any(|v| v.trim().is_empty())
            {
                return Err("model card name, description, estimand and version must be nonempty");
            }
            Ok(json!(card))
        }
        Some("assumption_document") => {
            let assumption: AssumptionDocument = serde_json::from_value(payload.clone())
                .map_err(|_| "invalid causal assumption fields")?;
            if [
                &assumption.assumption_name,
                &assumption.mathematical_statement,
                &assumption.intuitive_explanation,
            ]
            .iter()
            .any(|v| v.trim().is_empty())
            {
                return Err("assumption name, statement and explanation must be nonempty");
            }
            let mut markdown = format!("### {}\n\n", assumption.assumption_name);
            for (label, value) in [
                (
                    "Mathematical statement",
                    Some(assumption.mathematical_statement.as_str()),
                ),
                (
                    "Intuitive explanation",
                    Some(assumption.intuitive_explanation.as_str()),
                ),
                ("How checked", assumption.how_checked.as_deref()),
                (
                    "Sensitivity to violation",
                    assumption.sensitivity_to_violation.as_deref(),
                ),
            ] {
                if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
                    markdown.push_str(&format!("**{label}**\n\n{value}\n\n"));
                }
            }
            Ok(json!({"assumption":assumption,"markdown":markdown}))
        }
        Some("treatment_contrast") => treatment_contrast(payload),
        Some("release_evidence") => release_evidence(payload),
        Some("covariate_balance") => covariate_balance(payload),
        _ => Err("unknown causal summary method"),
    }
}

fn sensitivity_margin(point: f64, gamma: f64) -> Result<f64, &'static str> {
    if gamma < 1. {
        return Err("gamma must be at least one");
    }
    finite(point.abs() * 0.1 * (gamma - 1.))
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

fn counterfactual_forecast(payload: &Value) -> Result<Value, &'static str> {
    let model = payload
        .get("forecast_model")
        .unwrap_or(&payload["model"])
        .as_str()
        .ok_or("counterfactual model must be a string")?;
    let baseline = payload["baseline"]
        .as_object()
        .ok_or("baseline must be a prediction request payload")?;
    let parameters = baseline
        .get("parameters")
        .and_then(Value::as_object)
        .ok_or("baseline requires explicit model parameters")?;
    let mut alternative = baseline.clone();
    if let Some(overrides) = payload.get("parameter_overrides") {
        let mut changed = parameters.clone();
        apply_overrides(
            &mut changed,
            overrides,
            "counterfactual override names an unknown parameter",
        )?;
        alternative.insert("parameters".into(), Value::Object(changed));
    }
    if let Some(overrides) = payload.get("covariate_overrides") {
        let inputs = alternative
            .get_mut("inputs")
            .and_then(Value::as_object_mut)
            .ok_or("baseline requires prediction inputs")?;
        let covariates = inputs
            .get_mut("covariates")
            .and_then(Value::as_object_mut)
            .ok_or("baseline requires named covariates for a covariate override")?;
        apply_overrides(
            covariates,
            overrides,
            "counterfactual override names an unknown covariate",
        )?;
    }
    let binding = crate::KernelBinding::new();
    let predict = |request| {
        binding
            .predict_model(model, request)
            .map_err(|_| "counterfactual model prediction failed validation")?
            .result
            .ok_or("counterfactual prediction returned no result")
    };
    let baseline = predict(payload["baseline"].clone())?;
    let counterfactual = predict(Value::Object(alternative))?;
    if baseline.get("rows").is_some() {
        let rows = |result: &Value| {
            serde_json::from_value::<Vec<Vec<f64>>>(result["rows"].clone())
                .map_err(|_| "counterfactual prediction rows must be numeric arrays")
        };
        let before = rows(&baseline)?;
        let after = rows(&counterfactual)?;
        if baseline["columns"] != counterfactual["columns"] || before.len() != after.len() {
            return Err("counterfactual predictions must have matching rows and columns");
        }
        let (difference, percentage): (Vec<_>, Vec<_>) = before
            .iter()
            .zip(&after)
            .map(|(a, b)| compare_values(a, b))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        return Ok(
            json!({"columns":baseline["columns"],"baseline":before,"counterfactual":after,
            "difference":difference,"percentage_difference":percentage}),
        );
    }
    let baseline = series(&baseline, "values")?;
    let counterfactual = series(&counterfactual, "values")?;
    let (difference, percentage) = compare_values(&baseline, &counterfactual)?;
    Ok(json!({"baseline":baseline,"counterfactual":counterfactual,
        "difference":difference,"percentage_difference":percentage}))
}

type ForecastDifference = (Vec<f64>, Vec<Option<f64>>);

fn compare_values(
    baseline: &[f64],
    counterfactual: &[f64],
) -> Result<ForecastDifference, &'static str> {
    if baseline.len() != counterfactual.len() {
        return Err("counterfactual predictions must have the same length as baseline");
    }
    let difference: Vec<_> = baseline
        .iter()
        .zip(counterfactual)
        .map(|(before, after)| finite(after - before))
        .collect::<Result<_, _>>()?;
    let percentage: Vec<_> = baseline
        .iter()
        .zip(&difference)
        .map(|(&before, &delta)| {
            if before == 0. {
                Ok(None)
            } else {
                finite(100. * (delta / before)).map(Some)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((difference, percentage))
}

fn apply_overrides(
    baseline: &mut serde_json::Map<String, Value>,
    overrides: &Value,
    unknown: &'static str,
) -> Result<(), &'static str> {
    for (key, value) in overrides
        .as_object()
        .ok_or("counterfactual overrides must be an object")?
    {
        *baseline.get_mut(key).ok_or(unknown)? = value.clone();
    }
    Ok(())
}

fn release_evidence(payload: &Value) -> Result<Value, &'static str> {
    let claim = payload["claim"]
        .as_str()
        .filter(|v| !v.trim().is_empty())
        .ok_or("claim must be a nonempty string")?;
    let entries = |key: &str| {
        payload.get(key).map_or(Ok(Vec::new()), |value| {
            serde_json::from_value::<Vec<String>>(value.clone())
                .map_err(|_| "evidence and caveats must be arrays of strings")
        })
    };
    let evidence = entries("supporting_evidence")?;
    let caveats = entries("caveats")?;
    let level = payload
        .get("evidence_level")
        .map_or(Some("medium"), Value::as_str)
        .filter(|v| matches!(*v, "low" | "medium" | "high"))
        .ok_or("evidence_level must be low, medium or high")?;
    let sensitivity = payload
        .get("sensitivity_analysis_conducted")
        .map_or(Some(false), Value::as_bool)
        .ok_or("sensitivity_analysis_conducted must be boolean")?;
    let mut issues = Vec::new();
    if evidence.iter().all(|v| v.trim().is_empty()) {
        issues.push("No supporting evidence provided");
    }
    if caveats.iter().all(|v| v.trim().is_empty()) {
        issues.push("No caveats documented");
    }
    if level == "low" {
        issues.push("Evidence level is low - requires external review");
    }
    if level == "medium" && !sensitivity {
        issues.push("Sensitivity analysis recommended for medium-confidence claims");
    }
    Ok(
        json!({"claim":claim,"supporting_evidence":evidence,"caveats":caveats,
        "sensitivity_analysis_conducted":sensitivity,"evidence_level":level,
        "checks_passed":issues.is_empty(),"issues":issues}),
    )
}

#[derive(Deserialize, Serialize)]
struct ModelCard {
    name: String,
    description: String,
    estimand: String,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
    #[serde(default)]
    data_sources: Vec<String>,
    date_created: Option<String>,
    #[serde(default = "model_card_version")]
    version: String,
}

fn model_card_version() -> String {
    "1.0".into()
}

#[derive(Deserialize, Serialize)]
struct AssumptionDocument {
    assumption_name: String,
    mathematical_statement: String,
    intuitive_explanation: String,
    how_checked: Option<String>,
    sensitivity_to_violation: Option<String>,
}

fn treatment_contrast(payload: &Value) -> Result<Value, &'static str> {
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    if outcomes.len() != treatment.len() || treatment.iter().any(|&v| v != 0. && v != 1.) {
        return Err("outcomes must align with binary treatment indicators");
    }
    let summarize = |treated: &[f64], control: &[f64]| -> Result<Value, &'static str> {
        let estimate = match (mean(treated), mean(control)) {
            (Some(a), Some(b)) => Some(finite(a - b)?),
            _ => None,
        };
        Ok(json!({"estimate":estimate,"n_treated":treated.len(),"n_control":control.len()}))
    };
    let (treated, control): (Vec<_>, Vec<_>) = outcomes
        .iter()
        .copied()
        .zip(&treatment)
        .partition(|(_, flag)| **flag == 1.);
    let treated: Vec<_> = treated.into_iter().map(|(outcome, _)| outcome).collect();
    let control: Vec<_> = control.into_iter().map(|(outcome, _)| outcome).collect();
    if treated.is_empty() || control.is_empty() {
        return Err("treatment contrast requires treated and control observations");
    }
    let overall = summarize(&treated, &control)?;
    let bootstrap = payload
        .get("bootstrap")
        .map(|options| bootstrap_contrast(&treated, &control, options))
        .transpose()?;
    let mut modifiers = serde_json::Map::new();
    if let Some(fields) = payload.get("effect_modifiers") {
        for (name, labels) in fields
            .as_object()
            .ok_or("effect_modifiers must be an object")?
        {
            let labels: Vec<String> = serde_json::from_value(labels.clone())
                .map_err(|_| "effect modifiers must contain string labels")?;
            if labels.len() != outcomes.len() {
                return Err("effect modifier labels must align with outcomes");
            }
            let mut groups = std::collections::BTreeMap::<String, (Vec<f64>, Vec<f64>)>::new();
            for ((label, &outcome), &flag) in labels.into_iter().zip(&outcomes).zip(&treatment) {
                let (treated, control) = groups.entry(label).or_default();
                if flag == 1. {
                    treated.push(outcome);
                } else {
                    control.push(outcome);
                }
            }
            let summaries = groups
                .into_iter()
                .map(|(label, (treated, control))| Ok((label, summarize(&treated, &control)?)))
                .collect::<Result<serde_json::Map<_, _>, &'static str>>()?;
            modifiers.insert(name.clone(), Value::Object(summaries));
        }
    }
    Ok(
        json!({"method":"unadjusted_difference_in_means","overall":overall,"by_modifier":modifiers,"bootstrap":bootstrap}),
    )
}

fn bootstrap_contrast(
    treated: &[f64],
    control: &[f64],
    options: &Value,
) -> Result<Value, &'static str> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};

    let draws = options["draws"]
        .as_u64()
        .filter(|n| (2..=10_000).contains(n))
        .ok_or("bootstrap draws must be between 2 and 10000")? as usize;
    let seed = options["seed"]
        .as_u64()
        .ok_or("bootstrap requires a nonnegative integer seed")?;
    let confidence = options
        .get("confidence")
        .map_or(Some(0.95), Value::as_f64)
        .filter(|&level| level > 0. && level < 1.)
        .ok_or("bootstrap confidence must be between zero and one")?;
    if (treated.len() + control.len()) > 10_000_000 / draws {
        return Err("bootstrap workload exceeds 10000000 sampled outcomes");
    }
    let mut rng = DeterministicStream::from_run_seed(seed);
    let treated_index = Uniform::new(0, treated.len()).map_err(|_| "empty treated arm")?;
    let control_index = Uniform::new(0, control.len()).map_err(|_| "empty control arm")?;
    let mut sample_mean = |values: &[f64], index| {
        let sample: Vec<_> = (0..values.len())
            .map(|_| values[rng.sample(index)])
            .collect();
        mean(&sample).expect("bootstrap arm is nonempty")
    };
    let mut estimates = Vec::with_capacity(draws);
    for _ in 0..draws {
        estimates.push(finite(
            sample_mean(treated, treated_index) - sample_mean(control, control_index),
        )?);
    }
    estimates.sort_by(f64::total_cmp);
    let percentile = |probability: f64| {
        let index = probability * (draws - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let fraction = index.fract();
        (1. - fraction) * estimates[lower] + fraction * estimates[upper]
    };
    let tail = (1. - confidence) / 2.;
    Ok(
        json!({"method":"stratified_percentile","seed":seed,"n_bootstrap":draws,
        "confidence":confidence,"ci_lower":finite(percentile(tail))?,
        "ci_upper":finite(percentile(1. - tail))?}),
    )
}
