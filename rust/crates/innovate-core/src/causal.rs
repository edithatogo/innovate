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
        let mut sum: f64 = 0.;
        let mut correction = 0.;
        for value in values.iter().map(|v| v / scale) {
            let next = sum + value;
            correction += if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
            sum = next;
        }
        let average = (sum + correction) / values.len() as f64;
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
        Some("describe_treatment_contrasts") => Ok(json!({
            "contrasts":[
                {"method":"treatment_contrast","adjustment":"none",
                 "required_fields":["outcomes","treatment"],"optional_fields":["bootstrap","effect_modifiers"]},
                {"method":"stratified_treatment_contrast","adjustment":"caller_supplied_strata",
                 "required_fields":["outcomes","treatment","strata"],"optional_fields":["estimand"],
                 "estimands":["ate","att"],"default_estimand":"ate",
                 "uncertainty":"conditional_standard_error"},
                {"method":"propensity_weighted_contrast","adjustment":"caller_supplied_propensities",
                 "required_fields":["outcomes","treatment","propensities"],"optional_fields":["estimand"],
                 "estimands":["ate","att"],"default_estimand":"ate",
                 "uncertainty":"fixed_propensity_standard_error"},
                {"method":"matched_treatment_contrast","adjustment":"nearest_score_with_replacement",
                 "required_fields":["outcomes","treatment","scores","caliper"],"optional_fields":["allow_partial","outcome_variances","balance_covariates","control_predictions","independent_control_prediction_covariance_factors","outcome_covariance_factors","fit_controls","score_draws","score_refit_omit_indices","score_bootstrap"],
                 "estimands":["att"],"default_estimand":"att",
                 "partial_estimand":"matched_subset_att",
                 "uncertainty":"conditional_standard_error",
                 "uncertainty_requires":["outcome_variances"],
                 "uncertainty_assumptions":["fixed_matched_pairs","independent_outcomes"],
                 "correlated_uncertainty_option":"outcome_covariance_factors",
                 "alternative_to_scores":"fit_scores",
                 "fitted_score_uncertainty":"conditional_on_fitted_scores_and_pairs"},
                {"method":"augmented_treatment_contrast","adjustment":"caller_supplied_nuisance_predictions",
                 "required_fields":["outcomes","treatment","propensities","control_predictions"],
                 "required_fields_by_estimand":{"ate":["treated_predictions"],"att":[]},
                 "optional_fields":["estimand","clusters"],"estimands":["ate","att"],"default_estimand":"ate",
                 "uncertainty":"influence_standard_error"},
                {"method":"cross_fitted_stratum_ate","adjustment":"cross_fitted_strata",
                 "required_fields":["outcomes","treatment","strata","folds"],"optional_fields":["clusters","minimum_training_arm_count"],
                 "overlap_diagnostics":["min_training_arm_count","min_training_arm_clusters","min_propensity","max_propensity"],
                 "estimands":["ate"],"default_estimand":"ate"},
                {"method":"cross_fitted_stratum_att","adjustment":"cross_fitted_strata",
                 "required_fields":["outcomes","treatment","strata","folds"],"optional_fields":["clusters","minimum_training_arm_count"],
                 "overlap_diagnostics":["min_training_arm_count","min_training_arm_clusters","min_propensity","max_propensity"],
                 "estimands":["att"],"default_estimand":"att"}
            ]
        })),
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
                finite(percentage(effect, counterfactual))?
            };
            Ok(
                json!({"actual_mean":actual,"counterfactual_mean":counterfactual,
                      "effect":effect,"percent_change":percent_change}),
            )
        }
        Some("panel_event_study") => panel_event_study(payload),
        Some("staggered_panel_att") => staggered_panel_att(payload),
        Some("staggered_long_panel_att") => staggered_long_panel_att(payload),
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
        Some("stratified_treatment_contrast") => stratified_treatment_contrast(payload),
        Some("propensity_weighted_contrast") => propensity_weighted_contrast(payload),
        Some("augmented_treatment_contrast") => augmented_treatment_contrast(payload),
        Some("cross_fitted_stratum_ate") => cross_fitted_stratum_contrast(payload, "ate"),
        Some("cross_fitted_stratum_att") => cross_fitted_stratum_contrast(payload, "att"),
        Some("estimate_stratum_propensities") => estimate_stratum_propensities(payload),
        Some("fit_logistic_propensities") => fit_logistic_propensities(payload),
        Some("predict_logistic_propensities") => predict_logistic_propensities(payload),
        Some("cross_fit_logistic_propensities") => cross_fit_logistic_propensities(payload),
        Some("score_propensities") => score_propensities(payload),
        Some("matched_treatment_contrast") => matched_treatment_contrast(payload),
        Some("intervention_contract") => intervention_contract(payload),
        Some("policy_specification") => policy_specification(payload),
        Some("policy_evidence") => policy_evidence(payload),
        Some("causal_model_contract") => causal_model_contract(payload),
        Some("release_evidence") => release_evidence(payload),
        Some("covariate_balance") => covariate_balance(payload),
        _ => Err("unknown causal summary method"),
    }
}

fn independent_change_contrast(
    treated: &[f64],
    control: &[f64],
) -> Result<(Option<f64>, Option<f64>), &'static str> {
    let Some((treated_mean, control_mean)) = mean(treated).zip(mean(control)) else {
        return Ok((None, None));
    };
    let estimate = finite(treated_mean - control_mean)?;
    let arm_error = |changes: &[f64], average: f64| {
        let n = changes.len() as f64;
        let scale = 1. / (n * (n - 1.)).sqrt();
        changes.iter().fold(0_f64, |error, &change| {
            error.hypot(scale * change - scale * average)
        })
    };
    let error = if treated.len() > 1 && control.len() > 1 {
        Some(finite(
            arm_error(treated, treated_mean).hypot(arm_error(control, control_mean)),
        )?)
    } else {
        None
    };
    Ok((Some(estimate), error))
}

fn staggered_long_panel_att(payload: &Value) -> Result<Value, &'static str> {
    let unit_ids: Vec<String> = serde_json::from_value(payload["unit_ids"].clone())
        .map_err(|_| "unit_ids must contain string labels")?;
    let time = series(payload, "time")?;
    let outcomes = series(payload, "outcomes")?;
    let adoption = payload["adoption_times"]
        .as_object()
        .ok_or("adoption_times must map unit labels to periods or null")?;
    if time.is_empty()
        || time.len() != unit_ids.len()
        || time.len() != outcomes.len()
        || unit_ids.iter().any(|label| label.trim().is_empty())
    {
        return Err("long panel columns must be nonempty and aligned with nonblank unit labels");
    }
    let mut periods = time.clone();
    periods.sort_by(|a, b| a.partial_cmp(b).unwrap());
    periods.dedup();
    let unit_count = unit_ids
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if unit_count.checked_mul(periods.len()) != Some(time.len()) {
        return Err("long panel requires every unit at every period");
    }
    let index_of =
        |period: f64| periods.binary_search_by(|value| value.partial_cmp(&period).unwrap());
    let mut units = std::collections::BTreeMap::<String, Vec<Option<f64>>>::new();
    for ((unit, period), outcome) in unit_ids.into_iter().zip(time).zip(outcomes) {
        let row = units
            .entry(unit)
            .or_insert_with(|| vec![None; periods.len()]);
        if row[index_of(period).unwrap()].replace(outcome).is_some() {
            return Err("duplicate unit and period in long panel");
        }
    }
    if adoption.len() != units.len() {
        return Err("adoption_times must cover exactly the panel units");
    }
    let mut matrix = Vec::new();
    let mut indices = Vec::new();
    for (unit, row) in &units {
        matrix.push(
            row.iter()
                .copied()
                .collect::<Option<Vec<_>>>()
                .ok_or("long panel requires every unit at every period")?,
        );
        let value = adoption
            .get(unit)
            .ok_or("adoption_times must cover exactly the panel units")?;
        indices.push(if value.is_null() {
            None
        } else {
            Some(
                index_of(
                    value
                        .as_f64()
                        .ok_or("adoption times must be numeric or null")?,
                )
                .map_err(|_| "adoption time must identify an observed period")?,
            )
        });
    }
    let mut request = payload.clone();
    request["periods"] = json!(periods);
    request["outcomes"] = json!(matrix);
    request["adoption_indices"] = json!(indices);
    let mut result = staggered_panel_att(&request)?;
    result["unit_order"] = json!(units.keys().collect::<Vec<_>>());
    Ok(result)
}

fn staggered_panel_att(payload: &Value) -> Result<Value, &'static str> {
    let never_treated_only = match payload.get("control_group").map(Value::as_str) {
        None | Some(Some("not_yet_treated")) => false,
        Some(Some("never_treated")) => true,
        _ => return Err("control_group must be not_yet_treated or never_treated"),
    };
    let eligible_control = |index: Option<usize>, period| {
        index.is_none_or(|index| !never_treated_only && index > period)
    };
    let include_pre_trends = payload
        .get("include_pre_trends")
        .map_or(Some(false), Value::as_bool)
        .ok_or("include_pre_trends must be boolean")?;
    let bands = payload.get("simultaneous_bands");
    let include_covariance = payload
        .get("include_event_time_covariance")
        .map_or(Some(false), Value::as_bool)
        .ok_or("include_event_time_covariance must be boolean")?;
    let periods = series(payload, "periods")?;
    let outcomes: Vec<Vec<f64>> = serde_json::from_value(payload["outcomes"].clone())
        .map_err(|_| "outcomes must be a numeric unit-by-period matrix")?;
    let adoption: Vec<Option<usize>> = serde_json::from_value(payload["adoption_indices"].clone())
        .map_err(|_| "adoption indices must be integers or null")?;
    if periods.len() < 2
        || periods.windows(2).any(|pair| pair[0] >= pair[1])
        || outcomes.len() != adoption.len()
        || outcomes.iter().any(|row| row.len() != periods.len())
        || adoption
            .iter()
            .flatten()
            .any(|&index| index == 0 || index >= periods.len())
    {
        return Err(
            "staggered ATT requires aligned panels and adoption indices with an observed baseline",
        );
    }
    let cohorts: std::collections::BTreeSet<_> = adoption.iter().flatten().copied().collect();
    if cohorts.is_empty() {
        return Err("staggered ATT requires at least one adoption cohort");
    }
    let balanced_horizon = payload
        .get("balanced_horizon")
        .map(|value| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n < periods.len() - 1)
                .ok_or("balanced_horizon must be an observed nonnegative event-time offset")
        })
        .transpose()?;
    let cohort_count = cohorts.len();
    let mut contrasts = Vec::new();
    let mut pre_trends = Vec::new();
    let mut by_offset = std::collections::BTreeMap::<usize, Vec<_>>::new();
    for cohort in cohorts {
        let baseline = cohort - 1;
        let start = if include_pre_trends { 0 } else { cohort };
        for period in (start..periods.len()).filter(|&period| period != baseline) {
            let mut treated = Vec::new();
            let mut control = Vec::new();
            for (row, &index) in outcomes.iter().zip(&adoption) {
                if index == Some(cohort) {
                    treated.push(finite(row[period] - row[baseline])?);
                } else if eligible_control(index, period.max(baseline)) {
                    control.push(finite(row[period] - row[baseline])?);
                }
            }
            let (estimate, error) = independent_change_contrast(&treated, &control)?;
            let row = json!({"cohort_period":periods[cohort],"period":periods[period],
                "baseline_period":periods[baseline],"estimate":estimate,"standard_error":error,
                "n_treated":treated.len(),"n_control":control.len()});
            if period < cohort {
                pre_trends.push(row);
                continue;
            }
            by_offset.entry(period - cohort).or_default().push((
                estimate,
                treated.len(),
                cohort,
                control.len(),
            ));
            contrasts.push(row);
        }
    }
    let balanced_cohorts = if let Some(horizon) = balanced_horizon {
        let eligible: std::collections::BTreeSet<_> = by_offset
            .get(&horizon)
            .into_iter()
            .flatten()
            .filter_map(|&(estimate, _, cohort, _)| estimate.map(|_| cohort))
            .collect();
        if eligible.is_empty() {
            return Err("no cohort has eligible controls through balanced_horizon");
        }
        by_offset.retain(|&offset, rows| {
            rows.retain(|&(_, _, cohort, _)| eligible.contains(&cohort));
            offset <= horizon
        });
        Some(json!({"horizon":horizon,"included_cohorts":eligible.len(),
            "excluded_cohorts":cohort_count-eligible.len()}))
    } else {
        None
    };
    let mut event_time = Vec::with_capacity(by_offset.len());
    let mut influence_rows = Vec::new();
    if (include_covariance || bands.is_some()) && by_offset.len() > 128 {
        return Err("joint event-time inference supports at most 128 offsets");
    }
    for (period_offset, rows) in by_offset {
        let supported_units: usize = rows
            .iter()
            .filter(|(estimate, ..)| estimate.is_some())
            .map(|(_, count, ..)| count)
            .sum();
        let supported_cohorts = rows
            .iter()
            .filter(|(estimate, ..)| estimate.is_some())
            .count();
        let estimate = (supported_units > 0)
            .then(|| {
                finite(
                    rows.iter()
                        .filter_map(|(estimate, count, ..)| {
                            estimate.map(|value| value * (*count as f64 / supported_units as f64))
                        })
                        .sum(),
                )
            })
            .transpose()?;
        let influence = if supported_cohorts > 0
            && rows.iter().all(|&(estimate, treated, _, control)| {
                estimate.is_none() || (treated > 1 && control > 1)
            }) {
            let mut influence = vec![0.; outcomes.len()];
            for &(_, treated, cohort, control) in
                rows.iter().filter(|(estimate, ..)| estimate.is_some())
            {
                let period = cohort + period_offset;
                let baseline = cohort - 1;
                for (is_treated, count, sign) in [(true, treated, 1.), (false, control, -1.)] {
                    let members: Vec<_> = adoption
                        .iter()
                        .enumerate()
                        .filter(|&(_, &index)| {
                            if is_treated {
                                index == Some(cohort)
                            } else {
                                eligible_control(index, period)
                            }
                        })
                        .map(|(i, _)| i)
                        .collect();
                    let changes: Vec<_> = members
                        .iter()
                        .map(|&i| outcomes[i][period] - outcomes[i][baseline])
                        .collect();
                    let center = mean(&changes).unwrap();
                    let count = count as f64;
                    let scale = sign * (treated as f64 / supported_units as f64)
                        / (count * (count - 1.)).sqrt();
                    for (&i, change) in members.iter().zip(changes) {
                        influence[i] += scale * change - scale * center;
                    }
                }
            }
            Some(influence)
        } else {
            None
        };
        let standard_error = influence
            .as_ref()
            .map(|row| finite(row.iter().copied().fold(0_f64, f64::hypot)))
            .transpose()?;
        if include_covariance || bands.is_some() {
            influence_rows.push(influence);
        }
        event_time.push(json!({"period_offset":period_offset,"supported_cohort_att":estimate,
            "standard_error":standard_error,
            "supported_cohorts":supported_cohorts,"unsupported_cohorts":rows.len()-supported_cohorts,
            "supported_treated_units":supported_units}));
    }
    let method = if never_treated_only {
        "never_treated_group_time_difference_in_differences"
    } else {
        "not_yet_treated_group_time_difference_in_differences"
    };
    let mut result = json!({"method":method,
        "contrasts":contrasts,"event_time":event_time});
    if include_pre_trends {
        result["pre_trends"] = json!(pre_trends);
    }
    if let Some(balance) = balanced_cohorts {
        result["balanced_event_time"] = balance;
    }
    if let Some(options) = bands {
        result["simultaneous_bands"] =
            event_time_bands(options, &event_time, &influence_rows, outcomes.len())?;
    }
    if include_covariance {
        let mut covariance = vec![vec![None; influence_rows.len()]; influence_rows.len()];
        for (i, left) in influence_rows.iter().enumerate() {
            for (j, right) in influence_rows.iter().enumerate().take(i + 1) {
                if let Some((left, right)) = left.as_ref().zip(right.as_ref()) {
                    let value = finite(left.iter().zip(right).map(|(a, b)| a * b).sum())?;
                    covariance[i][j] = Some(value);
                    covariance[j][i] = Some(value);
                }
            }
        }
        result["event_time_covariance"] = json!(covariance);
    }
    Ok(result)
}

fn event_time_bands(
    options: &Value,
    event_time: &[Value],
    influence: &[Option<Vec<f64>>],
    units: usize,
) -> Result<Value, &'static str> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
    let draws = options["draws"]
        .as_u64()
        .filter(|n| (100..=10_000).contains(n))
        .ok_or("simultaneous bands require 100 to 10000 draws")? as usize;
    let seed = options["seed"]
        .as_u64()
        .ok_or("simultaneous bands require an integer seed")?;
    let confidence = options
        .get("confidence")
        .map_or(Some(0.95), Value::as_f64)
        .filter(|&value| 0. < value && value < 1.)
        .ok_or("confidence must be between zero and one")?;
    let usable: Vec<_> = event_time
        .iter()
        .zip(influence)
        .filter_map(|(row, values)| {
            let error = row["standard_error"].as_f64()?;
            (error > 0.).then_some((values.as_deref()?, error))
        })
        .collect();
    if usable.is_empty() {
        return Err("simultaneous bands require an offset with positive standard error");
    }
    if units > 10_000_000 / draws / usable.len() {
        return Err("simultaneous band workload exceeds 10000000 contributions");
    }
    let mut rng = DeterministicStream::from_run_seed(seed);
    let coin = Uniform::new(0_u32, 2).expect("valid binary distribution");
    let mut signs = vec![0.; units];
    let mut maxima = Vec::with_capacity(draws);
    for _ in 0..draws {
        for sign in &mut signs {
            *sign = if rng.sample(coin) == 0 { -1. } else { 1. };
        }
        let maximum = usable
            .iter()
            .map(|&(values, error)| {
                values
                    .iter()
                    .zip(&signs)
                    .map(|(value, sign)| sign * (value / error))
                    .sum::<f64>()
                    .abs()
            })
            .fold(0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    let critical = maxima[((draws as f64 * confidence).ceil() as usize).clamp(1, draws) - 1];
    let intervals = event_time
        .iter()
        .map(|row| {
            let interval = row["supported_cohort_att"]
                .as_f64()
                .zip(row["standard_error"].as_f64())
                .filter(|&(_, error)| error > 0.)
                .map(|(estimate, error)| {
                    Ok::<_, &'static str>(json!({"lower":finite(estimate-critical*error)?,
                    "upper":finite(estimate+critical*error)?}))
                })
                .transpose()?;
            Ok(json!({"period_offset":row["period_offset"],"interval":interval}))
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    Ok(
        json!({"method":"rademacher_multiplier_max_t","confidence":confidence,"draws":draws,
        "seed":seed,"critical_value":critical,"intervals":intervals}),
    )
}

fn panel_event_study(payload: &Value) -> Result<Value, &'static str> {
    let periods = series(payload, "periods")?;
    let baseline = payload["baseline_index"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&index| index < periods.len())
        .ok_or("baseline_index must identify an observed period")?;
    if periods.len() < 2 || periods.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(
            "event-study periods must be strictly increasing and contain at least two values",
        );
    }
    let parse_arm = |key: &str| -> Result<Vec<Vec<f64>>, &'static str> {
        let units: Vec<Vec<f64>> = serde_json::from_value(payload[key].clone())
            .map_err(|_| "event-study arms must contain numeric unit rows")?;
        if units.is_empty() || units.iter().any(|unit| unit.len() != periods.len()) {
            return Err("each event-study arm requires nonempty unit rows aligned with periods");
        }
        Ok(units)
    };
    let treated = parse_arm("treated")?;
    let control = parse_arm("control")?;
    let changes = |units: &[Vec<f64>], period: usize| {
        units
            .iter()
            .map(|unit| finite(unit[period] - unit[baseline]))
            .collect::<Result<Vec<_>, _>>()
    };
    let mut contrasts = Vec::with_capacity(periods.len());
    for (index, period) in periods.into_iter().enumerate() {
        let (estimate, error) =
            independent_change_contrast(&changes(&treated, index)?, &changes(&control, index)?)?;
        contrasts.push(json!({"period":period,"estimate":estimate,
            "standard_error":error,"reference":index == baseline}));
    }
    Ok(
        json!({"method":"balanced_panel_difference_in_differences","contrasts":contrasts,
        "n_treated":treated.len(),"n_control":control.len()}),
    )
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
    if payload.as_object().is_some_and(|fields| {
        fields.keys().any(|key| {
            !matches!(
                key.as_str(),
                "method"
                    | "forecast_model"
                    | "model"
                    | "baseline"
                    | "parameter_overrides"
                    | "covariate_overrides"
                    | "covariate_history_overrides"
                    | "parameter_event_override"
            )
        })
    }) {
        return Err("unknown counterfactual forecast field");
    }
    if let (Some(model), Some(alias)) = (payload.get("forecast_model"), payload.get("model"))
        && model != alias
    {
        return Err("forecast_model and model must agree when both are supplied");
    }
    let model = payload
        .get("forecast_model")
        .unwrap_or(&payload["model"])
        .as_str()
        .ok_or("counterfactual model must be a string")?;
    let baseline = payload["baseline"]
        .as_object()
        .ok_or("baseline must be a prediction request payload")?;
    let mut alternative = baseline.clone();
    if let Some(overrides) = payload.get("parameter_overrides") {
        let parameters = if let Some(state) = alternative.get_mut("state") {
            state.get_mut("parameters")
        } else {
            alternative.get_mut("parameters")
        };
        let parameters = parameters
            .and_then(Value::as_object_mut)
            .ok_or("baseline requires model parameters or fitted state")?;
        apply_overrides(
            parameters,
            overrides,
            "counterfactual override names an unknown parameter",
        )?;
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
    let mut options = if let Some(state) = alternative.get_mut("state") {
        state.get_mut("constructor_kwargs")
    } else {
        alternative.get_mut("constructor_kwargs")
    };
    if let Some(overrides) = payload.get("covariate_history_overrides") {
        let history = options
            .as_deref_mut()
            .and_then(|options| options.pointer_mut("/covariate_history/values"))
            .and_then(Value::as_object_mut)
            .ok_or("baseline requires named covariate history")?;
        apply_overrides(
            history,
            overrides,
            "counterfactual override names an unknown historical covariate",
        )?;
    }
    if let Some(events) = payload.get("parameter_event_override") {
        if !events.is_array() {
            return Err("parameter_event_override must be an array");
        }
        let options = options
            .and_then(Value::as_object_mut)
            .ok_or("baseline requires constructor options for a parameter event override")?;
        options.insert("parameter_events".into(), events.clone());
    }
    let binding = crate::KernelBinding::new();
    let predict = |request| {
        binding
            .predict_model(model, request)
            .map_err(|_| "counterfactual model prediction failed validation")?
            .result
            .ok_or("counterfactual prediction returned no result")
    };
    let time = series(&payload["baseline"]["inputs"], "time")?;
    let baseline = predict(payload["baseline"].clone())?;
    let counterfactual = predict(Value::Object(alternative))?;
    if baseline.get("rows").is_some() {
        let rows = |result: &Value| {
            serde_json::from_value::<Vec<Vec<f64>>>(result["rows"].clone())
                .map_err(|_| "counterfactual prediction rows must be numeric arrays")
        };
        let before = rows(&baseline)?;
        let after = rows(&counterfactual)?;
        if baseline["columns"] != counterfactual["columns"]
            || before.len() != after.len()
            || before.len() != time.len()
        {
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
            json!({"time":time,"columns":baseline["columns"],"baseline":before,"counterfactual":after,
            "difference":difference,"percentage_difference":percentage}),
        );
    }
    let baseline = series(&baseline, "values")?;
    let counterfactual = series(&counterfactual, "values")?;
    if baseline.len() != time.len() {
        return Err("counterfactual predictions must align with the prediction grid");
    }
    let (difference, percentage) = compare_values(&baseline, &counterfactual)?;
    Ok(
        json!({"time":time,"baseline":baseline,"counterfactual":counterfactual,
        "difference":difference,"percentage_difference":percentage}),
    )
}

fn percentage(delta: f64, baseline: f64) -> f64 {
    let ratio = delta / baseline;
    if ratio.abs() < f64::MIN_POSITIVE {
        // Scale first when division would lose subnormal precision.

        (100. * delta) / baseline
    } else {
        100. * ratio
    }
}

#[cfg(test)]
mod percentage_tests {
    use super::percentage;

    #[test]
    fn preserves_representable_percentages_when_the_ratio_underflows() {
        let smallest = f64::from_bits(1);
        assert_eq!(percentage(smallest, 4.), 25. * smallest);
        assert_eq!(percentage(-smallest, 4.), -25. * smallest);
        assert_eq!(percentage(f64::MAX, f64::MAX), 100.);
    }
}

type ForecastDifference = (Vec<f64>, Vec<Option<f64>>);

pub(crate) fn compare_values(
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
                finite(percentage(delta, before)).map(Some)
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

fn stratified_treatment_contrast(payload: &Value) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("expected a stratified contrast object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method" | "outcomes" | "treatment" | "strata" | "estimand"
        )
    }) {
        return Err(
            "stratified treatment contrast accepts only outcomes, treatment, strata and estimand",
        );
    }
    let att = match fields.get("estimand").map(Value::as_str) {
        None | Some(Some("ate")) => false,
        Some(Some("att")) => true,
        _ => return Err("stratified contrast estimand must be ate or att"),
    };
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    let strata: Vec<String> = serde_json::from_value(payload["strata"].clone())
        .map_err(|_| "strata must contain string labels")?;
    if outcomes.is_empty() || outcomes.len() != treatment.len() || outcomes.len() != strata.len() {
        return Err("outcomes, treatment and strata must be nonempty and aligned");
    }
    let population = if att {
        treatment.iter().filter(|&&flag| flag == 1.).count()
    } else {
        outcomes.len()
    };
    let mut groups = std::collections::BTreeMap::<String, (Vec<f64>, Vec<f64>)>::new();
    for ((label, outcome), flag) in strata.into_iter().zip(outcomes).zip(treatment) {
        if label.trim().is_empty() {
            return Err("stratum labels must be nonblank");
        }
        let (treated, control) = groups.entry(label).or_default();
        match flag {
            1. => treated.push(outcome),
            0. => control.push(outcome),
            _ => return Err("treatment indicators must be binary"),
        }
    }
    let mut estimate = 0.;
    let mut conditional_standard_error = Some(0_f64);
    let mut by_stratum = serde_json::Map::new();
    let mut excluded_control_observations = 0;
    if population == 0 {
        return Err("ATT requires treated observations");
    }
    for (label, (treated, control)) in groups {
        if att && treated.is_empty() {
            excluded_control_observations += control.len();
            continue;
        }
        let (Some(treated_mean), Some(control_mean)) = (mean(&treated), mean(&control)) else {
            return Err("each stratum requires treated and control observations");
        };
        let difference = finite(treated_mean - control_mean)?;
        let count = if att {
            treated.len()
        } else {
            treated.len() + control.len()
        };
        let weight = count as f64 / population as f64;
        estimate += weight * difference;
        for (arm, arm_mean) in [(&treated, treated_mean), (&control, control_mean)] {
            if arm.len() < 2 {
                conditional_standard_error = None;
            }
            if let Some(error) = &mut conditional_standard_error {
                let count = arm.len() as f64;
                let scale = weight / (count * (count - 1.)).sqrt();
                for &outcome in arm {
                    *error = error.hypot(scale * outcome - scale * arm_mean);
                }
            }
        }
        by_stratum.insert(
            label,
            json!({"estimate":difference,"weight":weight,
            "n_treated":treated.len(),"n_control":control.len(),
            "treated_fraction":treated.len() as f64 / (treated.len() + control.len()) as f64}),
        );
    }
    Ok(
        json!({"method":"empirical_stratum_standardization","estimate":finite(estimate)?,
        "conditional_standard_error":conditional_standard_error.map(finite).transpose()?,
        "excluded_control_observations":excluded_control_observations,
        "estimand":if att { "att" } else { "ate" },"by_stratum":by_stratum}),
    )
}

fn scaled_balance_difference(treated: &[f64], control: &[f64], scale: f64) -> f64 {
    if scale == 0. {
        return 0.;
    }
    let anchor = control[0];
    let centered_mean = |values: &[f64]| {
        let offsets: Vec<_> = values
            .iter()
            .map(|&value| {
                let difference = value - anchor;
                if difference.is_finite() {
                    difference / scale
                } else {
                    value / scale - anchor / scale
                }
            })
            .collect();
        mean(&offsets).unwrap()
    };
    centered_mean(treated) - centered_mean(control)
}

fn pooled_scaled_deviation(treated: &[f64], control: &[f64], scale: f64) -> Option<f64> {
    if scale == 0. || treated.len() < 2 || control.len() < 2 {
        return None;
    }
    let squared_deviations = |values: &[f64]| {
        let anchor = values[0];
        let offset = |&value: &f64| {
            let difference = value - anchor;
            if difference.is_finite() {
                difference / scale
            } else {
                value / scale - anchor / scale
            }
        };
        let center = values.iter().map(offset).sum::<f64>() / values.len() as f64;
        values
            .iter()
            .map(|value| (offset(value) - center).powi(2))
            .sum::<f64>()
    };
    let deviation = ((squared_deviations(treated) + squared_deviations(control))
        / (treated.len() + control.len() - 2) as f64)
        .sqrt();
    (deviation > 0.).then_some(deviation)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchingControlFit {
    strata: Vec<String>,
    folds: Vec<u32>,
}

fn fitted_matching_weights(
    fit: &MatchingControlFit,
    treatment: &[f64],
    weights: &[(usize, f64)],
) -> Result<Vec<(usize, f64)>, &'static str> {
    let n = treatment.len();
    if fit.strata.len() != n
        || fit.folds.len() != n
        || fit.strata.iter().any(|label| label.trim().is_empty())
    {
        return Err("control fit strata and folds must align with outcomes");
    }
    let fold_count = fit
        .folds
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if !(2..=32).contains(&fold_count) {
        return Err("control fitting requires two to 32 folds");
    }
    let mut controls = std::collections::BTreeMap::<&str, Vec<usize>>::new();
    for (index, &indicator) in treatment.iter().enumerate() {
        if indicator == 0. {
            controls.entry(&fit.strata[index]).or_default().push(index);
        }
    }
    let mut correction = std::collections::BTreeMap::<(&str, u32), f64>::new();
    let mut adjusted = vec![0.; n];
    for &(index, weight) in weights {
        adjusted[index] = weight;
        *correction
            .entry((&fit.strata[index], fit.folds[index]))
            .or_default() -= weight;
    }
    for ((stratum, fold), total) in correction {
        let eligible = || {
            controls
                .get(stratum)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&index| fit.folds[index] != fold)
        };
        let count = eligible().count();
        if count == 0 {
            return Err("each matched row requires training controls in its stratum");
        }
        for index in eligible() {
            adjusted[index] += total / count as f64;
        }
    }
    Ok(adjusted
        .into_iter()
        .enumerate()
        .filter(|&(_, weight)| weight != 0.)
        .collect())
}

fn weighted_outcome_contrast(
    values: &[f64],
    weights: &[(usize, f64)],
) -> Result<f64, &'static str> {
    let scale = weights
        .iter()
        .map(|&(index, _)| values[index].abs())
        .fold(0., f64::max);
    if scale == 0. {
        return Ok(0.);
    }
    // A contrast has zero total weight. Center first so a common shift
    // cancels even when the stored weights sum to a rounding residual.
    let anchor = values[weights[0].0];
    finite(
        weights
            .iter()
            .map(|&(index, weight)| {
                let offset = values[index] - anchor;
                let scaled_offset = if offset.is_finite() {
                    offset / scale
                } else {
                    values[index] / scale - anchor / scale
                };
                weight * scaled_offset
            })
            .sum::<f64>()
            * scale,
    )
}

const INCOMPLETE_MATCH: &str = "every treated observation requires a control within the caliper";
const EMPTY_MATCH: &str = "matching requires at least one matched treated observation";

fn matched_treatment_contrast(payload: &Value) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("expected a matched contrast object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method"
                | "outcomes"
                | "treatment"
                | "scores"
                | "fit_scores"
                | "score_draws"
                | "score_refit_omit_indices"
                | "score_bootstrap"
                | "caliper"
                | "allow_partial"
                | "outcome_variances"
                | "balance_covariates"
                | "control_predictions"
                | "independent_control_prediction_covariance_factors"
                | "fit_controls"
                | "outcome_covariance_factors"
        )
    }) {
        return Err("matched contrast received an unsupported field");
    }
    let allow_partial = fields
        .get("allow_partial")
        .map_or(Some(false), Value::as_bool)
        .ok_or("allow_partial must be boolean")?;
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    let score_fit_payload = if let Some(fit) = fields.get("fit_scores") {
        if fields.contains_key("scores") {
            return Err("choose either fit_scores or scores");
        }
        let fit = fit.as_object().ok_or("fit_scores must be an object")?;
        if fit
            .keys()
            .any(|key| !matches!(key.as_str(), "covariates" | "folds" | "ridge"))
        {
            return Err("fit_scores received an unsupported field");
        }
        Some(json!({
            "treatment": treatment, "covariates": fit.get("covariates"),
            "folds": fit.get("folds"), "ridge": fit.get("ridge"),
        }))
    } else {
        None
    };
    let score_fit = score_fit_payload
        .as_ref()
        .map(cross_fit_logistic_propensities)
        .transpose()?;
    let scores = match &score_fit {
        Some(fit) => series(fit, "propensities")?,
        None => series(payload, "scores")?,
    };
    let outcome_variances = fields
        .contains_key("outcome_variances")
        .then(|| series(payload, "outcome_variances"))
        .transpose()?;
    if outcome_variances.as_ref().is_some_and(|values| {
        values.len() != outcomes.len() || values.iter().any(|&variance| variance < 0.)
    }) {
        return Err("outcome variances must be nonnegative and align with outcomes");
    }
    let covariance_factors: Option<Vec<Vec<f64>>> = fields
        .get("outcome_covariance_factors")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| "outcome covariance factors must be numeric arrays")
        })
        .transpose()?;
    if covariance_factors.as_ref().is_some_and(|factors| {
        outcome_variances.is_none()
            || factors.is_empty()
            || factors.iter().any(|factor| factor.len() != outcomes.len())
    }) {
        return Err(
            "outcome covariance factors require outcome variances and aligned factor arrays",
        );
    }
    let control_predictions = fields
        .contains_key("control_predictions")
        .then(|| series(payload, "control_predictions"))
        .transpose()?;
    if control_predictions
        .as_ref()
        .is_some_and(|values| values.len() != outcomes.len())
    {
        return Err("control predictions must align with outcomes");
    }
    let prediction_factors: Option<Vec<Vec<f64>>> = fields
        .get("independent_control_prediction_covariance_factors")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| "control prediction covariance factors must be numeric arrays")
        })
        .transpose()?;
    if prediction_factors.as_ref().is_some_and(|factors| {
        control_predictions.is_none()
            || outcome_variances.is_none()
            || factors.is_empty()
            || factors.iter().any(|factor| factor.len() != outcomes.len())
    }) {
        return Err(
            "control prediction covariance factors require fixed predictions, outcome variances and aligned factor arrays",
        );
    }
    let fit_controls: Option<MatchingControlFit> = fields
        .get("fit_controls")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| "fit_controls requires strata and folds")
        })
        .transpose()?;
    if fit_controls.is_some() && control_predictions.is_some() {
        return Err("choose either fit_controls or control_predictions");
    }
    let balance_covariates: Option<std::collections::BTreeMap<String, Vec<f64>>> = fields
        .get("balance_covariates")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| "balance_covariates must contain named numeric arrays")
        })
        .transpose()?;
    if balance_covariates.as_ref().is_some_and(|columns| {
        columns.is_empty()
            || columns
                .iter()
                .any(|(name, values)| name.trim().is_empty() || values.len() != outcomes.len())
    }) {
        return Err("balance covariates must have nonblank names and align with outcomes");
    }
    let caliper = payload["caliper"]
        .as_f64()
        .filter(|&v| v >= 0.)
        .ok_or("matching requires a nonnegative caliper")?;
    if outcomes.len() != treatment.len()
        || outcomes.len() != scores.len()
        || treatment.iter().any(|&v| v != 0. && v != 1.)
    {
        return Err("outcomes and scores must align with binary treatment");
    }
    let mut controls: Vec<_> = (0..scores.len()).filter(|&i| treatment[i] == 0.).collect();
    controls.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut tied_controls = std::collections::BTreeSet::new();
    controls.dedup_by(|a, b| {
        let tied = scores[*a] == scores[*b];
        if tied {
            tied_controls.insert(*b);
        }
        tied
    });
    let mut differences = Vec::new();
    #[derive(Serialize)]
    struct MatchPair {
        treated_index: usize,
        control_index: usize,
        distance: f64,
    }
    let mut pairs = Vec::new();
    let mut unmatched_treated_indices = Vec::new();
    let mut control_uses = std::collections::BTreeMap::<usize, usize>::new();
    let mut max_score_distance = 0_f64;
    let mut minimum_runner_up_gap = f64::INFINITY;
    for i in (0..scores.len()).filter(|&i| treatment[i] == 1.) {
        let right = controls.partition_point(|&j| scores[j] < scores[i]);
        let nearest = right
            .checked_sub(1)
            .into_iter()
            .chain([right])
            .filter_map(|position| controls.get(position).copied())
            .map(|j| (j, (scores[i] - scores[j]).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .filter(|&(_, distance)| distance <= caliper);
        let Some((control, distance)) = nearest else {
            if !allow_partial {
                return Err(INCOMPLETE_MATCH);
            }
            unmatched_treated_indices.push(i);
            continue;
        };
        max_score_distance = max_score_distance.max(distance);
        let gap = if tied_controls.contains(&control) {
            0.
        } else {
            controls[right.saturating_sub(2)..(right + 2).min(controls.len())]
                .iter()
                .filter(|&&j| j != control)
                .map(|&j| (scores[i] - scores[j]).abs() - distance)
                .fold(f64::INFINITY, f64::min)
        };
        minimum_runner_up_gap = minimum_runner_up_gap.min(gap);
        differences.push(finite(outcomes[i] - outcomes[control])?);
        *control_uses.entry(control).or_default() += 1;
        pairs.push(MatchPair {
            treated_index: i,
            control_index: control,
            distance,
        });
    }
    let estimate = mean(&differences).ok_or(EMPTY_MATCH)?;
    let mut adjusted_estimate = control_predictions
        .as_ref()
        .map(|predictions| {
            let adjusted: Vec<_> = pairs
                .iter()
                .zip(&differences)
                .map(|(pair, &difference)| {
                    let adjustment =
                        finite(predictions[pair.treated_index] - predictions[pair.control_index])?;
                    finite(difference - adjustment)
                })
                .collect::<Result<_, _>>()?;
            Ok::<_, &'static str>(mean(&adjusted).unwrap())
        })
        .transpose()?;
    let estimand = if unmatched_treated_indices.is_empty() {
        "att"
    } else {
        "matched_subset_att"
    };
    let pair_count = pairs.len() as f64;
    let outcome_weights: Vec<_> = pairs
        .iter()
        .map(|pair| (pair.treated_index, 1. / pair_count))
        .chain(
            control_uses
                .iter()
                .map(|(&index, &count)| (index, -(count as f64) / pair_count)),
        )
        .collect();
    let adjusted_outcome_weights = fit_controls
        .as_ref()
        .map(|fit| fitted_matching_weights(fit, &treatment, &outcome_weights))
        .transpose()?;
    if let Some(weights) = &adjusted_outcome_weights {
        adjusted_estimate = Some(weighted_outcome_contrast(&outcomes, weights)?);
    }
    let adjusted_standard_error = adjusted_outcome_weights
        .as_ref()
        .zip(outcome_variances.as_ref())
        .map(|(weights, variances)| {
            let mut error = weights.iter().fold(0_f64, |error, &(index, weight)| {
                error.hypot(weight * variances[index].sqrt())
            });
            for factor in covariance_factors.iter().flatten() {
                error = error.hypot(weighted_outcome_contrast(factor, weights)?);
            }
            finite(error)
        })
        .transpose()?;
    let squared_control_weights = control_uses
        .values()
        .map(|&count| (count as f64 / pairs.len() as f64).powi(2))
        .sum::<f64>();
    let conditional_standard_error = outcome_variances
        .as_ref()
        .map(|variances| {
            let mut error = outcome_weights
                .iter()
                .fold(0_f64, |error, &(index, weight)| {
                    error.hypot(weight * variances[index].sqrt())
                });
            if let Some(factors) = &covariance_factors {
                for factor in factors {
                    let treated: Vec<_> = pairs
                        .iter()
                        .map(|pair| factor[pair.treated_index])
                        .collect();
                    let control: Vec<_> = pairs
                        .iter()
                        .map(|pair| factor[pair.control_index])
                        .collect();
                    let scale = treated
                        .iter()
                        .chain(&control)
                        .map(|value| value.abs())
                        .fold(0., f64::max);
                    let loading = scaled_balance_difference(&treated, &control, scale) * scale;
                    error = error.hypot(loading);
                }
            }
            finite(error)
        })
        .transpose()?;
    // Independent prediction errors contribute a separate covariance term.
    let adjusted_standard_error = if control_predictions.is_some() {
        conditional_standard_error
            .map(|mut error| {
                for factor in prediction_factors.iter().flatten() {
                    error = error.hypot(weighted_outcome_contrast(factor, &outcome_weights)?);
                }
                finite(error)
            })
            .transpose()?
    } else {
        adjusted_standard_error
    };
    let uncertainty_model = outcome_variances
        .as_ref()
        .map(|_| match &covariance_factors {
            Some(factors) => json!({"model":"diagonal_plus_factors","factor_count":factors.len()}),
            None => json!({"model":"independent_outcomes","factor_count":0}),
        });
    let balance = balance_covariates
        .map(|columns| {
            columns
                .into_iter()
                .map(|(name, values)| {
                    let treated: Vec<_> = pairs
                        .iter()
                        .map(|pair| values[pair.treated_index])
                        .collect();
                    let control: Vec<_> = pairs
                        .iter()
                        .map(|pair| values[pair.control_index])
                        .collect();
                    let mut before_treated = Vec::new();
                    let mut before_control = Vec::new();
                    for (&value, &indicator) in values.iter().zip(&treatment) {
                        if indicator == 1. {
                            before_treated.push(value);
                        } else {
                            before_control.push(value);
                        }
                    }
                    let magnitude = values.iter().map(|value| value.abs()).fold(0., f64::max);
                    let exponent = magnitude.to_bits() & f64::INFINITY.to_bits();
                    let scale = f64::from_bits(exponent).max(f64::MIN_POSITIVE);
                    let deviation = pooled_scaled_deviation(&before_treated, &before_control, scale);
                    let scaled_difference = scaled_balance_difference(&treated, &control, scale);
                    let scaled_before = scaled_balance_difference(&before_treated, &before_control, scale);
                    let standardize = |difference| deviation.and_then(|sd| finite(difference / sd).ok());
                    let before_treated = mean(&before_treated).unwrap();
                    let before_control = mean(&before_control).unwrap();
                    let treated_mean = mean(&treated).unwrap();
                    let control_mean = mean(&control).unwrap();
                    let difference = finite(scaled_difference * scale)?;
                    let before_difference = finite(scaled_before * scale)?;
                    Ok((
                        name,
                        json!({"treated_mean":treated_mean,"control_mean":control_mean,
                            "mean_difference":difference,
                            "standardized_mean_difference":standardize(scaled_difference),
                            "before_matching":{"treated_mean":before_treated,"control_mean":before_control,
                                "mean_difference":before_difference,
                                "standardized_mean_difference":standardize(scaled_before)}}),
                    ))
                })
                .collect::<Result<serde_json::Map<String, Value>, &'static str>>()
        })
        .transpose()?;
    let score_sensitivity = if let Some(options) = fields.get("score_bootstrap") {
        if fields.contains_key("score_draws") || fields.contains_key("score_refit_omit_indices") {
            return Err("score_bootstrap cannot be combined with other score draws");
        }
        let fit = score_fit_payload
            .as_ref()
            .ok_or("score_bootstrap requires fit_scores")?;
        Some(matching_score_bootstrap(fields, fit, options, &treatment)?)
    } else if let Some(indices) = fields.get("score_refit_omit_indices") {
        if fields.contains_key("score_draws") {
            return Err("choose either score_draws or score_refit_omit_indices");
        }
        let fit = score_fit_payload
            .as_ref()
            .ok_or("score refits require fit_scores")?;
        let indices: Vec<usize> = serde_json::from_value(indices.clone())
            .map_err(|_| "score refits require integer omission indices")?;
        if !(2..=128).contains(&indices.len())
            || indices.iter().any(|&index| index >= outcomes.len())
            || indices
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != indices.len()
        {
            return Err("score refits require two to 128 distinct in-range omission indices");
        }
        let draws = indices
            .iter()
            .map(|&index| {
                Ok(
                        cross_fit_logistic_scores(
                            fit,
                            (0..outcomes.len()).filter(|&i| i != index),
                        )?["propensities"]
                            .take(),
                    )
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        let mut sensitivity = matching_score_sensitivity(fields, &Value::Array(draws))?;
        sensitivity["method"] = json!("rematch_leave_one_out_score_refits");
        sensitivity["mixture_standard_deviation"] = Value::Null;
        sensitivity["adjusted_mixture_standard_deviation"] = Value::Null;
        sensitivity["omitted_training_indices"] = json!(indices);
        Some(sensitivity)
    } else {
        fields
            .get("score_draws")
            .map(|value| matching_score_sensitivity(fields, value))
            .transpose()?
    };
    Ok(
        json!({"method":"nearest_score_with_replacement","estimand":estimand,"estimate":estimate,"adjusted_estimate":adjusted_estimate,"pairs":pairs,
            "unmatched_treated_indices":unmatched_treated_indices,
            "conditional_standard_error":conditional_standard_error,"uncertainty_model":uncertainty_model,
            "covariate_balance":balance,"score_fit":score_fit,"score_sensitivity":score_sensitivity,
            "score_uncertainty_scope":score_fit.as_ref().map(|_| "conditional_on_fitted_scores_and_pairs"),
            "outcome_weights":outcome_weights,"adjusted_outcome_weights":adjusted_outcome_weights,
            "adjusted_standard_error":adjusted_standard_error,
            "match_diagnostics":{"caliper":caliper,"max_score_distance":max_score_distance,
                "minimum_matched_caliper_slack":caliper - max_score_distance,
                "minimum_runner_up_distance_gap":finite(minimum_runner_up_gap).ok(),
                "matched_treated_fraction":pairs.len() as f64 / (pairs.len() + unmatched_treated_indices.len()) as f64},
            "control_diagnostics":{"unique_controls":control_uses.len(),
            "effective_sample_size":1. / squared_control_weights}}),
    )
}

fn matching_score_bootstrap(
    fields: &serde_json::Map<String, Value>,
    fit: &Value,
    options: &Value,
    treatment: &[f64],
) -> Result<Value, &'static str> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Options {
        draws: usize,
        seed: u64,
    }
    let options: Options = serde_json::from_value(options.clone())
        .map_err(|_| "score_bootstrap requires draws and a nonnegative integer seed")?;
    if !(2..=128).contains(&options.draws) {
        return Err("score_bootstrap requires two to 128 draws");
    }
    let mut groups = std::collections::BTreeMap::<_, Vec<_>>::new();
    for (index, (fold, &arm)) in fit["folds"]
        .as_array()
        .unwrap()
        .iter()
        .zip(treatment)
        .enumerate()
    {
        groups
            .entry((fold.as_u64().unwrap(), arm == 1.))
            .or_default()
            .push(index);
    }
    let mut rng = DeterministicStream::from_run_seed(options.seed);
    let mut draws = Vec::with_capacity(options.draws);
    for _ in 0..options.draws {
        let mut training = Vec::with_capacity(treatment.len());
        for group in groups.values() {
            let index = Uniform::new(0, group.len()).unwrap();
            training.extend((0..group.len()).map(|_| group[rng.sample(index)]));
        }
        draws.push(cross_fit_logistic_scores(fit, training)?["propensities"].take());
    }
    let mut sensitivity = matching_score_sensitivity(fields, &Value::Array(draws))?;
    sensitivity["method"] = json!("rematch_fold_and_treatment_stratified_score_bootstrap");
    sensitivity["seed"] = json!(options.seed);
    Ok(sensitivity)
}

fn matching_score_sensitivity(
    fields: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<Value, &'static str> {
    let draws = value
        .as_array()
        .filter(|draws| (2..=128).contains(&draws.len()))
        .ok_or("score_draws requires two to 128 aligned score arrays")?;
    let mut draw_fields = fields.clone();
    draw_fields.remove("score_draws");
    draw_fields.remove("score_bootstrap");
    draw_fields.remove("score_refit_omit_indices");
    draw_fields.remove("fit_scores");
    draw_fields.remove("balance_covariates");
    let mut draw_payload = Value::Object(draw_fields);
    let mut results = Vec::with_capacity(draws.len());
    for scores in draws {
        draw_payload["scores"] = scores.clone();
        let result = match matched_treatment_contrast(&draw_payload) {
            Ok(result) => result,
            Err(reason @ (INCOMPLETE_MATCH | EMPTY_MATCH)) => {
                results.push(json!({"status": "insufficient_matches", "reason": reason}));
                continue;
            }
            Err(error) => return Err(error),
        };
        results.push(json!({
            "status": "matched",
            "estimate": result["estimate"],
            "adjusted_estimate": result["adjusted_estimate"],
            "estimand": result["estimand"],
            "unmatched_treated_indices": result["unmatched_treated_indices"],
            "pairs": result["pairs"],
            "conditional_standard_error": result["conditional_standard_error"],
            "adjusted_standard_error": result["adjusted_standard_error"],
        }));
    }
    let same_treated_subset = results.iter().all(|draw| {
        draw["status"] == "matched"
            && draw["unmatched_treated_indices"] == results[0]["unmatched_treated_indices"]
    });
    let spread = |estimate: &str, error: &str| -> Option<f64> {
        if !same_treated_subset {
            return None;
        }
        let estimates: Vec<_> = results
            .iter()
            .map(|draw| draw[estimate].as_f64())
            .collect::<Option<_>>()?;
        let center = mean(&estimates)?;
        let scale = (results.len() as f64).sqrt();
        let mut deviation = 0_f64;
        for (draw, estimate) in results.iter().zip(estimates) {
            deviation = deviation
                .hypot(estimate / scale - center / scale)
                .hypot(draw[error].as_f64()? / scale);
        }
        finite(deviation).ok()
    };
    Ok(json!({
        "method": "rematch_caller_supplied_score_draws",
        "same_treated_subset": same_treated_subset,
        "mixture_standard_deviation": spread("estimate", "conditional_standard_error"),
        "adjusted_mixture_standard_deviation": spread("adjusted_estimate", "adjusted_standard_error"),
        "draws": results,
    }))
}

fn score_propensities(payload: &Value) -> Result<Value, &'static str> {
    let treatment = series(payload, "treatment")?;
    let probabilities = series(payload, "propensities")?;
    if treatment.is_empty() || treatment.len() != probabilities.len() {
        return Err("propensity scoring requires nonempty aligned treatment and propensities");
    }
    let mut brier = 0.;
    let mut log_loss = 0.;
    for (&target, &probability) in treatment.iter().zip(&probabilities) {
        if !(0. < probability && probability < 1.) {
            return Err("propensities must lie strictly between zero and one");
        }
        log_loss -= match target {
            0. => (-probability).ln_1p(),
            1. => probability.ln(),
            _ => return Err("treatment indicators must be binary"),
        };
        brier += (probability - target).powi(2);
    }
    let n = treatment.len() as f64;
    Ok(
        json!({"observations":treatment.len(),"brier_score":brier/n,"log_loss":log_loss/n,
        "observed_treated_fraction":mean(&treatment),"mean_propensity":mean(&probabilities)}),
    )
}

fn logistic_probability(logit: f64) -> f64 {
    if logit >= 0. {
        1. / (1. + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1. + exp)
    }
}

fn cross_fit_logistic_propensities(payload: &Value) -> Result<Value, &'static str> {
    cross_fit_logistic_scores(
        payload,
        0..payload["treatment"].as_array().map_or(0, Vec::len),
    )
}

fn cross_fit_logistic_scores(
    payload: &Value,
    training_indices: impl IntoIterator<Item = usize>,
) -> Result<Value, &'static str> {
    let treatment = series(payload, "treatment")?;
    let rows = payload["covariates"]
        .as_array()
        .ok_or("expected covariate rows")?;
    let folds: Vec<u32> = serde_json::from_value(payload["folds"].clone())
        .map_err(|_| "folds must be unsigned integer labels")?;
    if treatment.len() != rows.len() || treatment.len() != folds.len() {
        return Err("treatment, covariates and folds must align");
    }
    let labels: std::collections::BTreeSet<_> = folds.iter().copied().collect();
    if !(2..=32).contains(&labels.len()) {
        return Err("cross fitting requires two to 32 folds");
    }
    let training_indices: Vec<_> = training_indices.into_iter().collect();
    let mut propensities = vec![0.; rows.len()];
    for label in &labels {
        let held_out: Vec<_> = (0..rows.len())
            .filter(|&index| folds[index] == *label)
            .collect();
        let training: Vec<_> = training_indices
            .iter()
            .copied()
            .filter(|&index| folds[index] != *label)
            .collect();
        let fit = fit_logistic_propensities(&json!({
            "treatment":training.iter().map(|&i| treatment[i]).collect::<Vec<_>>(),
            "covariates":training.iter().map(|&i| &rows[i]).collect::<Vec<_>>(),
            "ridge":payload["ridge"],
        }))?;
        let predicted = predict_logistic_propensities(&json!({
            "fit":fit,"covariates":held_out.iter().map(|&i| &rows[i]).collect::<Vec<_>>()
        }))?;
        for (index, probability) in held_out
            .into_iter()
            .zip(series(&predicted, "propensities")?)
        {
            propensities[index] = probability;
        }
    }
    Ok(
        json!({"method":"cross_fitted_ridge_logistic","propensities":propensities,
        "folds":folds,"fold_count":labels.len()}),
    )
}

fn predict_logistic_propensities(payload: &Value) -> Result<Value, &'static str> {
    let fit = &payload["fit"];
    if fit["method"] != "ridge_logistic" {
        return Err("prediction requires a ridge logistic fit");
    }
    let coefficients = series(fit, "scaled_coefficients")?;
    let scales = series(fit, "covariate_scales")?;
    let rows: Vec<Vec<f64>> = serde_json::from_value(payload["covariates"].clone())
        .map_err(|_| "covariates must be a numeric row matrix")?;
    if scales.is_empty()
        || scales.iter().any(|&s| s <= 0.)
        || coefficients.len() != scales.len() + 1
        || rows.is_empty()
        || rows.iter().any(|row| row.len() != scales.len())
    {
        return Err("prediction requires valid fit scales and aligned covariates");
    }
    let propensities = rows
        .iter()
        .map(|row| {
            let logit = row
                .iter()
                .zip(&scales)
                .zip(&coefficients[1..])
                .fold(coefficients[0], |value, ((x, scale), slope)| {
                    value + (x / scale) * slope
                });
            let probability = logistic_probability(finite(logit)?);
            if !(0. < probability && probability < 1.) {
                return Err("predicted propensities must lie strictly between zero and one");
            }
            Ok(probability)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"method":"ridge_logistic","propensities":propensities}))
}

fn fit_logistic_propensities(payload: &Value) -> Result<Value, &'static str> {
    let treatment = series(payload, "treatment")?;
    let rows: Vec<Vec<f64>> = serde_json::from_value(payload["covariates"].clone())
        .map_err(|_| "covariates must be a numeric row matrix")?;
    let width = rows.first().map_or(0, Vec::len);
    let penalty = payload["ridge"]
        .as_f64()
        .filter(|&v| v > 0.)
        .ok_or("logistic propensity fit requires positive ridge")?;
    if width == 0
        || rows.len() != treatment.len()
        || rows.iter().any(|row| row.len() != width)
        || treatment.iter().any(|&v| v != 0. && v != 1.)
        || !treatment.contains(&0.)
        || !treatment.contains(&1.)
    {
        return Err("logistic propensity fit requires aligned covariates and both binary arms");
    }
    let mut scales = vec![0_f64; width];
    for row in &rows {
        for (scale, value) in scales.iter_mut().zip(row) {
            *scale = scale.max(value.abs());
        }
    }
    for scale in &mut scales {
        if *scale == 0. {
            *scale = 1.;
        }
    }
    let design: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            std::iter::once(1.)
                .chain(row.iter().zip(&scales).map(|(x, s)| x / s))
                .collect()
        })
        .collect();
    let mut coefficients = vec![0.; width + 1];
    let mut gradient = vec![0.; width + 1];
    let mut propensities = vec![0.; rows.len()];
    let n = rows.len() as f64;
    let step = 1. / (0.25 * (width + 1) as f64 + penalty);
    for iteration in 0..10_000 {
        gradient.fill(0.);
        for ((row, &target), probability) in design.iter().zip(&treatment).zip(&mut propensities) {
            let logit: f64 = row.iter().zip(&coefficients).map(|(x, b)| x * b).sum();
            *probability = logistic_probability(logit);
            let residual = (*probability - target) / n;
            for (component, x) in gradient.iter_mut().zip(row) {
                *component += residual * x;
            }
        }
        for (component, coefficient) in gradient[1..].iter_mut().zip(&coefficients[1..]) {
            *component += penalty * coefficient;
        }
        if gradient.iter().all(|g| g.abs() <= 1e-8) {
            if propensities.iter().any(|&p| !(0. < p && p < 1.)) {
                return Err("fitted propensities must lie strictly between zero and one");
            }
            return Ok(
                json!({"method":"ridge_logistic","propensities":propensities,
                "scaled_coefficients":coefficients,"covariate_scales":scales,"iterations":iteration}),
            );
        }
        for (coefficient, component) in coefficients.iter_mut().zip(&gradient) {
            *coefficient -= step * component;
        }
    }
    Err("logistic propensity fit did not converge")
}

fn estimate_stratum_propensities(payload: &Value) -> Result<Value, &'static str> {
    let treatment = series(payload, "treatment")?;
    let strata: Vec<String> = serde_json::from_value(payload["strata"].clone())
        .map_err(|_| "strata must contain string labels")?;
    if treatment.is_empty() || treatment.len() != strata.len() {
        return Err("treatment and strata must be nonempty and aligned");
    }
    let mut counts = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    for (label, flag) in strata.iter().zip(treatment) {
        if label.trim().is_empty() {
            return Err("stratum labels must be nonblank");
        }
        let (treated, control) = counts.entry(label).or_default();
        match flag {
            0. => *control += 1,
            1. => *treated += 1,
            _ => return Err("treatment indicators must be binary"),
        }
    }
    if counts
        .values()
        .any(|&(treated, control)| treated == 0 || control == 0)
    {
        return Err("propensity estimation requires both arms in every stratum");
    }
    let by_stratum: serde_json::Map<String, Value> = counts
        .into_iter()
        .map(|(label, (treated, control))| {
            (
                label.to_owned(),
                json!({
                    "propensity":treated as f64 / (treated + control) as f64,
                    "n_treated":treated,"n_control":control
                }),
            )
        })
        .collect();
    let propensities: Vec<_> = strata
        .iter()
        .map(|label| &by_stratum[label]["propensity"])
        .collect();
    Ok(
        json!({"method":"empirical_stratum_frequency","propensities":propensities,"by_stratum":by_stratum}),
    )
}

fn cross_fitted_stratum_contrast(payload: &Value, estimand: &str) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("expected a cross-fitted contrast object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method"
                | "outcomes"
                | "treatment"
                | "strata"
                | "folds"
                | "clusters"
                | "minimum_training_arm_count"
        )
    }) {
        return Err("cross-fitted contrast received an unsupported field");
    }
    let required_arm_count = payload
        .get("minimum_training_arm_count")
        .map(|value| {
            value
                .as_u64()
                .filter(|&count| count > 0)
                .ok_or("minimum_training_arm_count must be a positive integer")
        })
        .transpose()?
        .unwrap_or(1);
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    let strata: Vec<String> = serde_json::from_value(payload["strata"].clone())
        .map_err(|_| "strata must contain string labels")?;
    let folds: Vec<u32> = serde_json::from_value(payload["folds"].clone())
        .map_err(|_| "folds must be unsigned integer labels")?;
    let n = outcomes.len();
    if [treatment.len(), strata.len(), folds.len()]
        .iter()
        .any(|&len| len != n)
        || strata.iter().any(|label| label.trim().is_empty())
        || treatment.iter().any(|&v| v != 0. && v != 1.)
    {
        return Err(
            "cross-fitted contrast requires aligned inputs, nonblank strata and binary treatment",
        );
    }
    let labels: std::collections::BTreeSet<_> = folds.iter().copied().collect();
    if !(2..=32).contains(&labels.len()) {
        return Err("cross fitting requires two to 32 folds");
    }
    let clusters = if let Some(clusters) = payload.get("clusters") {
        let clusters: Vec<String> = serde_json::from_value(clusters.clone())
            .map_err(|_| "clusters must contain string labels")?;
        if clusters.len() != n || clusters.iter().any(|label| label.trim().is_empty()) {
            return Err("cluster labels must be nonblank and align with outcomes");
        }
        let mut cluster_folds = std::collections::BTreeMap::new();
        for (cluster, &fold) in clusters.iter().zip(&folds) {
            if *cluster_folds.entry(cluster).or_insert(fold) != fold {
                return Err("each cluster must belong to a single cross-fitting fold");
            }
        }
        Some(clusters)
    } else {
        None
    };
    let mut propensities = vec![0.; n];
    let mut treated_predictions = vec![0.; n];
    let mut control_predictions = vec![0.; n];
    let mut min_training_arm_count = usize::MAX;
    let mut min_training_arm_clusters = None;
    let mut min_propensity: f64 = 1.;
    let mut max_propensity: f64 = 0.;
    for label in &labels {
        let mut training = std::collections::BTreeMap::<&str, [Vec<f64>; 2]>::new();
        let mut training_clusters =
            std::collections::BTreeMap::<&str, [std::collections::BTreeSet<&str>; 2]>::new();
        for i in (0..n).filter(|&i| folds[i] != *label) {
            if let Some(clusters) = &clusters {
                training_clusters.entry(&strata[i]).or_default()[treatment[i] as usize]
                    .insert(&clusters[i]);
            }
            training.entry(&strata[i]).or_default()[treatment[i] as usize].push(outcomes[i]);
        }
        let fitted: std::collections::BTreeMap<_, _> = training
            .iter()
            .filter_map(|(&stratum, [control, treated])| {
                Some((
                    stratum,
                    (
                        mean(control)?,
                        mean(treated)?,
                        treated.len() as f64 / (control.len() + treated.len()) as f64,
                        control.len().min(treated.len()),
                    ),
                ))
            })
            .collect();
        for i in (0..n).filter(|&i| folds[i] == *label) {
            let &(control, treated, propensity, arm_count) = fitted
                .get(strata[i].as_str())
                .ok_or("each held-out stratum requires both arms in its training complement")?;
            control_predictions[i] = control;
            treated_predictions[i] = treated;
            propensities[i] = propensity;
            min_training_arm_count = min_training_arm_count.min(arm_count);
            if let Some([control, treated]) = training_clusters.get(strata[i].as_str()) {
                let count = control.len().min(treated.len());
                min_training_arm_clusters = Some(
                    min_training_arm_clusters.map_or(count, |previous: usize| previous.min(count)),
                );
            }
            min_propensity = min_propensity.min(propensity);
            max_propensity = max_propensity.max(propensity);
        }
    }
    if (min_training_arm_count as u64) < required_arm_count {
        return Err("a held-out stratum falls below minimum_training_arm_count");
    }
    let mut request = json!({
        "outcomes":outcomes,"treatment":treatment,"propensities":propensities,
        "estimand":estimand,"control_predictions":control_predictions
    });
    if estimand == "ate" {
        request["treated_predictions"] = json!(treated_predictions);
    }
    if let Some(clusters) = payload.get("clusters") {
        request["clusters"] = clusters.clone();
    }
    let contrast = augmented_treatment_contrast(&request)?;
    Ok(
        json!({"contrast":contrast,"folds":folds,"propensities":propensities,
        "treated_predictions":treated_predictions,"control_predictions":control_predictions,
        "overlap":{"min_training_arm_count":min_training_arm_count,
            "min_training_arm_clusters":min_training_arm_clusters,
            "min_propensity":min_propensity,"max_propensity":max_propensity}}),
    )
}

fn augmented_treatment_contrast(payload: &Value) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("expected an augmented contrast object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method"
                | "outcomes"
                | "treatment"
                | "propensities"
                | "treated_predictions"
                | "control_predictions"
                | "estimand"
                | "clusters"
        )
    }) {
        return Err("augmented contrast received an unsupported field");
    }
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    let propensities = series(payload, "propensities")?;
    let att = match fields.get("estimand").map(Value::as_str) {
        None | Some(Some("ate")) => false,
        Some(Some("att")) => true,
        _ => return Err("augmented contrast estimand must be ate or att"),
    };
    if att && fields.contains_key("treated_predictions") {
        return Err("augmented ATT requires only control outcome predictions");
    }
    let treated = (!att)
        .then(|| series(payload, "treated_predictions"))
        .transpose()?;
    let control = series(payload, "control_predictions")?;
    let n = outcomes.len();
    if [treatment.len(), propensities.len(), control.len()]
        .iter()
        .any(|&len| len != n)
        || treated.as_ref().is_some_and(|values| values.len() != n)
        || !treatment.contains(&0.)
        || !treatment.contains(&1.)
    {
        return Err("augmented contrast requires aligned inputs and both treatment arms");
    }
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let probability = propensities[i];
        if !(0. < probability && probability < 1.) {
            return Err("propensities must lie strictly between zero and one");
        }
        let score = match (&treated, treatment[i]) {
            (Some(predictions), 0.) => {
                predictions[i] - control[i] + (control[i] - outcomes[i]) / (1. - probability)
            }
            (Some(predictions), 1.) => {
                predictions[i] - control[i] + (outcomes[i] - predictions[i]) / probability
            }
            (None, 0.) => (control[i] - outcomes[i]) * (probability / (1. - probability)),
            (None, 1.) => outcomes[i] - control[i],
            _ => return Err("treatment indicators must be binary"),
        };
        scores.push(finite(score)?);
    }
    let denominator = if att { mean(&treatment).unwrap() } else { 1. };
    let estimate = finite(mean(&scores).unwrap() / denominator)?;
    let centered_scores = scores.iter().zip(&treatment).map(|(&score, &flag)| {
        let center = if att { flag * estimate } else { estimate };
        let scale = 1. / (n as f64 * denominator);
        scale * score - scale * center
    });
    let (standard_error, cluster_count) = if let Some(clusters) = fields.get("clusters") {
        let labels: Vec<String> = serde_json::from_value(clusters.clone())
            .map_err(|_| "clusters must contain string labels")?;
        if labels.len() != n || labels.iter().any(|label| label.trim().is_empty()) {
            return Err("cluster labels must be nonblank and align with outcomes");
        }
        let mut sums = std::collections::BTreeMap::<String, f64>::new();
        for (label, score) in labels.into_iter().zip(centered_scores) {
            *sums.entry(label).or_default() += score;
        }
        let count = sums.len();
        if count < 2 {
            return Err("cluster inference requires at least two clusters");
        }
        let error = sums.into_values().fold(0_f64, f64::hypot);
        (
            error * (count as f64 / (count - 1) as f64).sqrt(),
            Some(count),
        )
    } else {
        let error = centered_scores.fold(0_f64, f64::hypot);
        (error * (n as f64 / (n - 1) as f64).sqrt(), None)
    };
    Ok(
        json!({"method":"augmented_inverse_probability_weighting","estimand":if att { "att" } else { "ate" },
        "estimate":estimate,"influence_standard_error":finite(standard_error)?,"observations":n,
        "cluster_count":cluster_count,"cluster_degrees_of_freedom":cluster_count.map(|count| count - 1)}),
    )
}

fn propensity_weighted_contrast(payload: &Value) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("expected a weighted contrast object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method" | "outcomes" | "treatment" | "propensities" | "estimand"
        )
    }) {
        return Err(
            "propensity weighted contrast accepts only outcomes, treatment, propensities and estimand",
        );
    }
    let outcomes = series(payload, "outcomes")?;
    let treatment = series(payload, "treatment")?;
    let propensities = series(payload, "propensities")?;
    let estimand = match fields.get("estimand").map(Value::as_str) {
        None | Some(Some("ate")) => "ate",
        Some(Some("att")) => "att",
        _ => return Err("weighted contrast estimand must be ate or att"),
    };
    if outcomes.len() != treatment.len() || outcomes.len() != propensities.len() {
        return Err("outcomes, treatment and propensities must align");
    }
    let mut arms = [Vec::new(), Vec::new()];
    for ((outcome, flag), propensity) in outcomes.into_iter().zip(treatment).zip(propensities) {
        if propensity <= 0. || propensity >= 1. {
            return Err("propensities must lie strictly between zero and one");
        }
        let (arm, log_weight) = match (estimand, flag) {
            ("att", 0.) => (0, propensity.ln() - (-propensity).ln_1p()),
            ("att", 1.) => (1, 0.),
            (_, 0.) => (0, -(-propensity).ln_1p()),
            (_, 1.) => (1, -propensity.ln()),
            _ => return Err("treatment indicators must be binary"),
        };
        arms[arm].push((outcome, log_weight));
    }
    let summarize_arm = |arm: &[(f64, f64)]| -> Result<(f64, Option<f64>, Value), &'static str> {
        let maximum = arm
            .iter()
            .map(|&(_, p)| p)
            .reduce(f64::max)
            .ok_or("propensity weighted contrast requires both treatment arms")?;
        let sum_weights = arm.iter().map(|&(_, w)| (w - maximum).exp()).sum::<f64>();
        let mut weighted_mean = 0.;
        let mut squared_weights = 0.;
        for &(outcome, log_weight) in arm {
            let weight = (log_weight - maximum).exp() / sum_weights;
            weighted_mean += weight * outcome;
            squared_weights += weight * weight;
        }
        let standard_error = (arm.len() > 1).then(|| {
            let correction = (arm.len() as f64 / (arm.len() - 1) as f64).sqrt();
            correction
                * arm.iter().fold(0_f64, |error, &(outcome, log_weight)| {
                    let weight = (log_weight - maximum).exp() / sum_weights;
                    error.hypot(weight * outcome - weight * weighted_mean)
                })
        });
        Ok((
            finite(weighted_mean)?,
            standard_error,
            json!({
                "observations":arm.len(),
                "effective_sample_size":1. / squared_weights,
                "maximum_normalized_weight":1. / sum_weights
            }),
        ))
    };
    let (control, control_error, control_diagnostics) = summarize_arm(&arms[0])?;
    let (treated, treated_error, treated_diagnostics) = summarize_arm(&arms[1])?;
    let standard_error = treated_error
        .zip(control_error)
        .map(|(a, b)| finite(a.hypot(b)))
        .transpose()?;
    Ok(
        json!({"method":"normalized_inverse_probability_weighting","estimand":estimand,
        "estimate":finite(treated-control)?,"treated_mean":treated,"control_mean":control,
        "fixed_propensity_standard_error":standard_error,
        "weight_diagnostics":{"treated":treated_diagnostics,"control":control_diagnostics}}),
    )
}

fn treatment_contrast(payload: &Value) -> Result<Value, &'static str> {
    let fields = payload
        .as_object()
        .ok_or("treatment contrast must be an object")?;
    if fields.keys().any(|key| {
        !matches!(
            key.as_str(),
            "method" | "outcomes" | "treatment" | "bootstrap" | "effect_modifiers"
        )
    }) {
        return Err("treatment contrast supports only unadjusted differences in means");
    }
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
    let scheme = options
        .get("scheme")
        .map_or(Some("stratified"), Value::as_str);
    let mut rng = DeterministicStream::from_run_seed(seed);
    let mut estimates = Vec::with_capacity(draws);
    let method = match scheme {
        Some("stratified") => {
            let treated_index = Uniform::new(0, treated.len()).map_err(|_| "empty treated arm")?;
            let control_index = Uniform::new(0, control.len()).map_err(|_| "empty control arm")?;
            let mut sample_mean = |values: &[f64], index| {
                let sample: Vec<_> = (0..values.len())
                    .map(|_| values[rng.sample(index)])
                    .collect();
                mean(&sample).expect("bootstrap arm is nonempty")
            };
            for _ in 0..draws {
                estimates.push(finite(
                    sample_mean(treated, treated_index) - sample_mean(control, control_index),
                )?);
            }
            "stratified_percentile"
        }
        Some("pooled") => {
            let size = treated.len() + control.len();
            let indices = Uniform::new(0, size).map_err(|_| "empty pooled sample")?;
            let mut treated_sample = Vec::new();
            let mut control_sample = Vec::new();
            for _ in 0..draws {
                treated_sample.clear();
                control_sample.clear();
                for _ in 0..size {
                    let index = rng.sample(indices);
                    if index < treated.len() {
                        treated_sample.push(treated[index]);
                    } else {
                        control_sample.push(control[index - treated.len()]);
                    }
                }
                if let (Some(a), Some(b)) = (mean(&treated_sample), mean(&control_sample)) {
                    estimates.push(finite(a - b)?);
                }
            }
            "pooled_percentile"
        }
        _ => return Err("bootstrap scheme must be stratified or pooled"),
    };
    estimates.sort_by(f64::total_cmp);
    let percentile = |probability: f64| {
        if estimates.len() < 2 {
            return Ok(None);
        }
        let index = probability * (estimates.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let fraction = index.fract();
        finite((1. - fraction) * estimates[lower] + fraction * estimates[upper]).map(Some)
    };
    let tail = (1. - confidence) / 2.;
    Ok(json!({"method":method,"seed":seed,"n_bootstrap":draws,
        "valid_draws":estimates.len(),"empty_arm_draws":draws - estimates.len(),
        "confidence":confidence,"ci_lower":percentile(tail)?,
        "ci_upper":percentile(1. - tail)?}))
}

#[derive(Deserialize, Serialize)]
struct InterventionContract {
    name: String,
    timing: String,
    comparator: String,
    start_time: Option<i64>,
    end_time: Option<i64>,
    rollout_schedule: Option<std::collections::BTreeMap<String, f64>>,
    spillover_regions: Option<Vec<String>>,
    spillover_strength: Option<f64>,
}

fn intervention_contract(payload: &Value) -> Result<Value, &'static str> {
    let contract: InterventionContract = serde_json::from_value(payload.clone())
        .map_err(|_| "invalid intervention contract fields")?;
    if contract.name.trim().is_empty() || contract.comparator.trim().is_empty() {
        return Err("intervention requires a name and comparator");
    }
    if !matches!(
        contract.timing.as_str(),
        "post" | "pre" | "staggered" | "event-study"
    ) {
        return Err("unsupported intervention timing");
    }
    if contract
        .start_time
        .zip(contract.end_time)
        .is_some_and(|(start, end)| start > end)
    {
        return Err("intervention times must be ordered");
    }
    if contract
        .spillover_strength
        .is_some_and(|value| !(0. ..=1.).contains(&value))
    {
        return Err("spillover strength must be between zero and one");
    }
    if let Some(window) = payload.get("outcome_window") {
        let [start, end]: [i64; 2] = serde_json::from_value(window.clone())
            .map_err(|_| "outcome_window must contain two integers")?;
        if start > end || contract.start_time.is_some_and(|time| end < time) {
            return Err("outcome window is reversed or ends before intervention");
        }
    }
    Ok(json!(contract))
}

#[derive(Deserialize, Serialize)]
struct CausalModelContract {
    name: String,
    treatment_variable: String,
    outcome_variable: String,
    confounders: Vec<String>,
    effect_modifiers: Option<Vec<String>>,
    #[serde(default)]
    identifying_assumptions: std::collections::BTreeMap<String, String>,
}

fn causal_model_contract(payload: &Value) -> Result<Value, &'static str> {
    let contract: CausalModelContract = serde_json::from_value(payload.clone())
        .map_err(|_| "invalid causal model contract fields")?;
    if [
        &contract.name,
        &contract.treatment_variable,
        &contract.outcome_variable,
    ]
    .iter()
    .any(|value| value.trim().is_empty())
    {
        return Err("causal model requires a name and treatment and outcome variables");
    }
    if contract.confounders.is_empty() || contract.confounders.iter().any(|v| v.trim().is_empty()) {
        return Err("causal model requires named confounders");
    }
    for variable in &contract.confounders {
        let name = variable.to_ascii_lowercase();
        if ["post_treatment", "affected_", "_outcome", "post_"]
            .iter()
            .any(|indicator| name.contains(indicator))
        {
            return Err("confounder name matches a post-treatment indicator");
        }
    }
    let risk = payload
        .get("unobserved_confounding_risk")
        .map_or(Some("unknown"), Value::as_str)
        .filter(|value| matches!(*value, "low" | "medium" | "high" | "unknown"))
        .ok_or("unobserved_confounding_risk must be low, medium, high or unknown")?;
    let sensitivity = payload
        .get("has_sensitivity_analysis")
        .map_or(Some(false), Value::as_bool)
        .ok_or("has_sensitivity_analysis must be boolean")?;
    if risk == "high" && !sensitivity {
        return Err("high risk of unobserved confounding requires sensitivity analysis");
    }
    Ok(json!(contract))
}

fn policy_evidence(payload: &Value) -> Result<Value, &'static str> {
    let intervention = intervention_contract(&payload["intervention"])?;
    let model = causal_model_contract(&payload["causal_model"])?;
    let confounders = model["confounders"]
        .as_array()
        .expect("validated confounder list");
    Ok(json!({
        "metadata": {
            "intervention_name":intervention["name"],"model_name":model["name"],
            "treatment_variable":model["treatment_variable"],"outcome_variable":model["outcome_variable"],
            "n_confounders":confounders.len()
        },
        "evidence": {
            "assumptions":model["identifying_assumptions"],
            "estimand":{"treatment":model["treatment_variable"],"outcome":model["outcome_variable"],
                "confounders":confounders},
            "limitations":{"unobserved_confounding":"Not addressed without sensitivity analysis",
                "overlap_assumption":"Assumes positive probability of treatment assignment"}
        }
    }))
}

fn policy_specification(payload: &Value) -> Result<Value, &'static str> {
    let intervention = intervention_contract(&payload["intervention"])?;
    let causal_model = causal_model_contract(&payload["causal_model"])?;
    for (key, normalized) in [
        ("intervention", &intervention),
        ("causal_model", &causal_model),
    ] {
        let supplied = payload[key].as_object().expect("validated policy contract");
        if supplied.keys().any(|field| normalized.get(field).is_none()) {
            return Err("policy specification contains unknown contract fields");
        }
    }
    let declared = payload
        .get("n_obs")
        .filter(|value| !value.is_null())
        .map(|value| value.as_u64().ok_or("n_obs must be a nonnegative integer"))
        .transpose()?;
    let n_obs = if let Some(data) = payload.get("data").filter(|value| !value.is_null()) {
        let columns = data
            .as_object()
            .filter(|columns| !columns.is_empty())
            .ok_or("policy data must contain named columns")?;
        let rows = columns
            .values()
            .next()
            .and_then(Value::as_array)
            .ok_or("policy data columns must be arrays")?
            .len();
        if columns
            .values()
            .any(|column| column.as_array().is_none_or(|values| values.len() != rows))
        {
            return Err("policy data columns must be aligned arrays");
        }
        if declared.is_some_and(|count| count != rows as u64) {
            return Err("n_obs does not match policy data");
        }
        Some(rows as u64)
    } else {
        declared
    };
    Ok(json!({"intervention":intervention,"causal_model":causal_model,"n_obs":n_obs}))
}

#[cfg(test)]
mod tests {
    use super::{
        matched_treatment_contrast, pooled_scaled_deviation, scaled_balance_difference,
        weighted_outcome_contrast,
    };

    #[test]
    fn pooled_deviation_resolves_adjacent_covariate_values() {
        let lower = 1.5_f64;
        let upper = lower.next_up();
        let values = [lower, upper];
        let deviation = pooled_scaled_deviation(&values, &values, upper).unwrap();
        let expected = (upper - lower) / upper / 2_f64.sqrt();
        assert!((deviation / expected - 1.).abs() <= 2. * f64::EPSILON);
    }
    #[test]
    fn standardized_balance_resolves_difference_between_rounded_means() {
        let lower = 1.5_f64;
        let upper = lower.next_up();
        let treated = [lower, lower, upper];
        let control = [lower, upper];
        let deviation = pooled_scaled_deviation(&treated, &control, upper);
        let difference = scaled_balance_difference(&treated, &control, upper);
        let raw_expected = -(upper - lower) / 6.;
        assert!((difference * upper / raw_expected - 1.).abs() <= 2. * f64::EPSILON);
        let actual = difference / deviation.unwrap();
        let expected = (-1. / 6.) / (7_f64 / 18.).sqrt();
        assert!((actual - expected).abs() <= 2. * f64::EPSILON);
    }
    #[test]
    fn fitted_outcome_contrast_cancels_a_common_loading() {
        let weights = [(0, -1.), (1, 1. / 3.), (2, 1. / 3.), (3, 1. / 3.)];
        assert_eq!(weighted_outcome_contrast(&[f64::MAX; 4], &weights), Ok(0.));
    }
    #[test]
    fn matching_uses_held_out_logistic_scores() {
        let mut payload = serde_json::json!({
            "outcomes": [0, 4, 2, 6], "treatment": [0, 1, 0, 1], "caliper": 0,
            "fit_scores": {"covariates": [[0], [0], [0], [0]], "folds": [1, 1, 2, 2], "ridge": 1}
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        assert_eq!(
            result["score_fit"]["propensities"],
            serde_json::json!([0.5, 0.5, 0.5, 0.5])
        );
        assert_eq!(result["estimate"].as_f64(), Some(5.));
        assert_eq!(
            result["match_diagnostics"]["minimum_matched_caliper_slack"].as_f64(),
            Some(0.)
        );
        assert_eq!(
            result["score_uncertainty_scope"],
            "conditional_on_fitted_scores_and_pairs"
        );
        payload["caliper"] = serde_json::json!(0.25);
        let result = matched_treatment_contrast(&payload).unwrap();
        assert_eq!(
            result["match_diagnostics"]["minimum_matched_caliper_slack"].as_f64(),
            Some(0.25)
        );

        payload["scores"] = serde_json::json!([0, 0, 0, 0]);
        assert_eq!(
            matched_treatment_contrast(&payload),
            Err("choose either fit_scores or scores")
        );
    }
    #[test]
    fn matching_reports_runner_up_distance_gap() {
        let mut payload = serde_json::json!({
            "outcomes": [0, 4, 2], "treatment": [0, 1, 0], "caliper": 1,
            "scores": [0, 1, 3]
        });
        let gap = |payload: &serde_json::Value| {
            matched_treatment_contrast(payload).unwrap()["match_diagnostics"]
                ["minimum_runner_up_distance_gap"].as_f64()
        };
        assert_eq!(gap(&payload), Some(1.));
        payload["scores"] = serde_json::json!([0, 1, 0]);
        assert_eq!(gap(&payload), Some(0.));
        payload["scores"] = serde_json::json!([0, 1, 2]);
        assert_eq!(gap(&payload), Some(0.));
    }
    #[test]
    fn matching_rematches_score_draws_in_order() {
        let payload = serde_json::json!({
            "outcomes": [0, 4, 2], "treatment": [0, 1, 0],
            "caliper": 0, "scores": [0, 0, 1],
            "score_draws": [[0, 0, 1], [1, 0, 0]],
            "outcome_variances": [0, 1, 6], "control_predictions": [0, 0, 2]
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        assert_eq!(result["estimate"].as_f64(), Some(4.));
        let draws = &result["score_sensitivity"]["draws"];
        assert_eq!(draws[0]["estimate"].as_f64(), Some(4.));
        assert_eq!(draws[1]["estimate"].as_f64(), Some(2.));
        assert_eq!(draws[0]["conditional_standard_error"].as_f64(), Some(1.));
        let second_error = draws[1]["conditional_standard_error"].as_f64().unwrap();
        assert!((second_error - 7_f64.sqrt()).abs() <= 2. * f64::EPSILON);
        let spread = result["score_sensitivity"]["mixture_standard_deviation"]
            .as_f64()
            .unwrap();
        assert!((spread - 5_f64.sqrt()).abs() < 1e-14);
        let adjusted_spread = result["score_sensitivity"]["adjusted_mixture_standard_deviation"]
            .as_f64()
            .unwrap();
        assert!((adjusted_spread - 2.).abs() < 1e-14);
        assert_eq!(draws[1]["pairs"][0]["control_index"].as_u64(), Some(2));
    }
    #[test]
    fn matching_does_not_pool_draws_with_different_treated_subsets() {
        let payload = serde_json::json!({
            "outcomes": [0, 4, 6], "treatment": [0, 1, 1],
            "caliper": 0, "scores": [0, 0, 0], "allow_partial": true,
            "score_draws": [[0, 0, 1], [0, 1, 0]], "outcome_variances": [1, 1, 1]
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        let sensitivity = &result["score_sensitivity"];
        assert_eq!(sensitivity["same_treated_subset"], false);
        assert!(sensitivity["mixture_standard_deviation"].is_null());
        let draws = &sensitivity["draws"];
        assert_eq!(
            draws[0]["unmatched_treated_indices"],
            serde_json::json!([2])
        );
        assert_eq!(
            draws[1]["unmatched_treated_indices"],
            serde_json::json!([1])
        );
        assert_eq!(draws[0]["estimate"].as_f64(), Some(4.));
        assert_eq!(draws[1]["estimate"].as_f64(), Some(6.));
    }
    #[test]
    fn matching_refits_scores_with_shared_training_omissions() {
        let payload = serde_json::json!({
            "outcomes": [0, 4, 2, 6, 8, 12, 10, 14],
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1], "caliper": 1,
            "fit_scores": {"covariates": vec![vec![0]; 8],
                "folds": [1, 1, 1, 1, 2, 2, 2, 2], "ridge": 1},
            "score_refit_omit_indices": [0, 4]
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        let sensitivity = &result["score_sensitivity"];
        assert_eq!(sensitivity["method"], "rematch_leave_one_out_score_refits");
        assert_eq!(
            sensitivity["omitted_training_indices"],
            serde_json::json!([0, 4])
        );
        assert_eq!(sensitivity["draws"][0]["pairs"][0]["control_index"], 0);
        assert_eq!(sensitivity["draws"][0]["pairs"][2]["control_index"], 4);
        assert_eq!(sensitivity["draws"][1]["estimate"].as_f64(), Some(5.));
    }
    #[test]
    fn matching_retains_draws_without_eligible_pairs() {
        let mut payload = serde_json::json!({
            "outcomes": [0, 4], "treatment": [0, 1], "caliper": 0,
            "scores": [0, 0], "score_draws": [[0, 0], [0, 1]],
            "outcome_variances": [1, 1], "allow_partial": true
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        let sensitivity = &result["score_sensitivity"];
        assert_eq!(sensitivity["draws"][0]["estimate"].as_f64(), Some(4.));
        assert_eq!(sensitivity["draws"][1]["status"], "insufficient_matches");
        assert_eq!(sensitivity["same_treated_subset"], false);
        assert!(sensitivity["mixture_standard_deviation"].is_null());
        payload["score_draws"][1] = serde_json::json!([0]);
        assert!(matched_treatment_contrast(&payload).is_err());
    }
    #[test]
    fn matching_bootstraps_training_scores_reproducibly() {
        let payload = serde_json::json!({
            "outcomes": [0, 4, 2, 6, 8, 12, 10, 14],
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1], "caliper": 1,
            "fit_scores": {"covariates": [[0], [3], [2], [5], [1], [4], [3], [6]],
                "folds": [1, 1, 1, 1, 2, 2, 2, 2], "ridge": 1},
            "score_bootstrap": {"draws": 8, "seed": 17},
            "outcome_variances": vec![1; 8]
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        assert_eq!(matched_treatment_contrast(&payload).unwrap(), result);
        let sensitivity = &result["score_sensitivity"];
        assert_eq!(sensitivity["seed"], 17);
        let draws = sensitivity["draws"].as_array().unwrap();
        assert_eq!(draws.len(), 8);
        assert!(draws.iter().any(|draw| draw["pairs"] != draws[0]["pairs"]));
        assert!(sensitivity["mixture_standard_deviation"].is_number());
    }
    #[test]
    fn matching_propagates_independent_prediction_covariance() {
        let payload = serde_json::json!({
            "outcomes": [0, 4, 2], "treatment": [0, 1, 0],
            "caliper": 0, "scores": [0, 0, 1], "score_draws": [[0, 0, 1], [1, 0, 0]],
            "outcome_variances": [0, 1, 6], "control_predictions": [0, 0, 2],
            "independent_control_prediction_covariance_factors": [[9, 9, 9], [0, 0, 2]]
        });
        let result = matched_treatment_contrast(&payload).unwrap();
        let sensitivity = &result["score_sensitivity"];
        let adjusted = sensitivity["adjusted_mixture_standard_deviation"]
            .as_f64()
            .unwrap();
        let raw = sensitivity["mixture_standard_deviation"].as_f64().unwrap();
        assert!((adjusted - 6_f64.sqrt()).abs() < 1e-14);
        assert!((raw - 5_f64.sqrt()).abs() < 1e-14);
    }
}
