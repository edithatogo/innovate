//! Explicit historical Logistic covariate and event-curve contract.
use crate::validation::{numeric_array, numeric_array_from_aliases, object_section};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde_json::{Map, Value, json};
pub(crate) fn selected(r: &KernelRequest) -> bool {
    if r.model_key.as_deref() != Some("logistic") || r.payload.get("simulation_options").is_some() {
        return false;
    }
    let kw = r
        .payload
        .get("constructor_kwargs")
        .or_else(|| r.payload.get("model_kwargs"))
        .or_else(|| {
            r.payload
                .get("state")
                .and_then(|s| s.get("constructor_kwargs"))
        });
    kw.is_some_and(|k| {
        k.get("covariates")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
            || k.get("t_event").is_some_and(|v| !v.is_null())
            || k.get("event_candidates").is_some()
    })
}
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let error = |s: &str| KernelBindingError::invalid_request(op, s);
    let payload = r
        .payload
        .as_object()
        .ok_or_else(|| error("payload must be an object"))?;
    let input = object_section(payload, "inputs", op)?;
    let time = numeric_array_from_aliases(input, &["time", "t"], op)?;
    if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[0] >= w[1]) {
        return Err(error("time must be finite and strictly increasing"));
    }
    if payload
        .get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != "logistic")
    {
        return Err(error("state model_key mismatch"));
    }
    let kw = payload
        .get("constructor_kwargs")
        .or_else(|| payload.get("model_kwargs"))
        .or_else(|| {
            payload
                .get("state")
                .and_then(|s| s.get("constructor_kwargs"))
        })
        .and_then(Value::as_object)
        .ok_or_else(|| error("constructor options must be an object"))?;
    if kw.contains_key("event_candidates") {
        return profile_event(r, kw, &time, execute);
    }
    if kw
        .keys()
        .any(|k| !["covariates", "t_event", "event_covariates"].contains(&k.as_str()))
    {
        return Err(error("unknown Logistic option"));
    }
    for v in [
        payload.get("predict_kwargs"),
        payload.get("fitter_options"),
        payload.get("state").and_then(|s| s.get("predict_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        if !v.as_object().is_some_and(Map::is_empty) {
            return Err(error("custom prediction/fitter options unavailable"));
        }
    }
    let cov: Vec<String> = kw
        .get("covariates")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|_| error("covariates must be names"))?
        .unwrap_or_default();
    if cov.iter().any(String::is_empty)
        || cov.iter().collect::<std::collections::HashSet<_>>().len() != cov.len()
    {
        return Err(error("covariate names must be nonempty and unique"));
    }
    let event = kw
        .get("t_event")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| error("t_event must be finite"))
        })
        .transpose()?;
    let mode = kw
        .get("event_covariates")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| error("event_covariates must be a string"))
        })
        .transpose()?
        .unwrap_or("historical_ignored");
    if !["historical_ignored", "additive"].contains(&mode) {
        return Err(error("unknown event_covariates mode"));
    }
    let mut names: Vec<String> = ["L", "k", "x0"].into_iter().map(String::from).collect();
    if event.is_some() {
        names.extend(
            ["L_post", "k_post", "x0_post"]
                .into_iter()
                .map(String::from),
        );
    }
    for c in &cov {
        for k in ["L", "k", "x0"] {
            names.push(format!("beta_{k}_{c}"));
        }
    }
    let empty = Map::new();
    let supplied = input
        .get("covariates")
        .map(|v| {
            v.as_object()
                .ok_or_else(|| error("input covariates must be an object"))
        })
        .transpose()?
        .unwrap_or(&empty);
    if supplied.len() != cov.len() || cov.iter().any(|c| !supplied.contains_key(c)) {
        return Err(error("input covariates must match declared names"));
    }
    let series = cov
        .iter()
        .map(|c| numeric_array(&supplied[c], c, op))
        .collect::<Result<Vec<_>, _>>()?;
    if series
        .iter()
        .any(|v| v.len() != time.len() || v.iter().any(|x| !x.is_finite()))
    {
        return Err(error(
            "covariate arrays must be finite and aligned with time",
        ));
    }
    let control_slopes = input
        .get("covariate_rates")
        .map(|value| {
            let rates = value
                .as_object()
                .filter(|rates| {
                    rates.len() == cov.len() && cov.iter().all(|name| rates.contains_key(name))
                })
                .ok_or_else(|| error("covariate_rates must match declared names"))?;
            cov.iter()
                .map(|name| {
                    let values = numeric_array(&rates[name], name, op)?;
                    if values.len() != time.len() || values.iter().any(|value| !value.is_finite()) {
                        return Err(error(
                            "covariate_rates must be finite and aligned with time",
                        ));
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let parse = |v: &Value| -> Result<Vec<f64>, KernelBindingError> {
        let p = v
            .as_object()
            .ok_or_else(|| error("parameters must be an object"))?;
        if p.len() != names.len() {
            return Err(error("parameter names must match the complete model"));
        }
        names
            .iter()
            .map(|n| {
                p.get(n)
                    .and_then(Value::as_f64)
                    .filter(|v| {
                        v.is_finite()
                            && (!["L", "k", "L_post", "k_post"].contains(&n.as_str()) || *v > 0.)
                    })
                    .ok_or_else(|| error("missing or invalid parameter"))
            })
            .collect()
    };
    let point = |p: &[f64], i: usize, t: f64| -> Result<(f64, f64), String> {
        let post = event.is_some_and(|e| t >= e);
        let offset = if post { 3 } else { 0 };
        let mut a = [p[offset], p[offset + 1], p[offset + 2]];
        let mut derivative = [0.; 3];
        let apply_controls = event.is_none() || mode == "additive";
        if apply_controls {
            let start = if event.is_some() { 6 } else { 3 };
            for (j, x) in series.iter().enumerate() {
                for k in 0..3 {
                    let beta = p[start + 3 * j + k];
                    a[k] += beta * x[i];
                    if let Some(slopes) = &control_slopes {
                        derivative[k] += beta * slopes[j][i];
                    }
                }
            }
        }
        if a.iter().any(|x| !x.is_finite()) || a[0] <= 0. || a[1] <= 0. {
            return Err("effective L/k must stay positive and finite".into());
        }
        let z = a[1] * (t - a[2]);
        let tail = (-z.abs()).exp();
        let denominator = 1. + tail;
        let fraction = if z >= 0. {
            1. / denominator
        } else {
            tail / denominator
        };
        let slope = (a[0] * tail / denominator) * (a[1] / denominator);
        let slope = if apply_controls && control_slopes.is_some() {
            let dz = derivative[1] * (t - a[2]) + a[1] * (1. - derivative[2]);
            derivative[0] * fraction + (a[0] * tail / denominator) * (dz / denominator)
        } else {
            slope
        };
        Ok((a[0] * fraction, slope))
    };
    let weights = if op == KernelOperation::FitModel {
        let active = if event.is_some() && mode == "historical_ignored" {
            6
        } else {
            names.len()
        };
        crate::fitting::weights(r, time.len(), active.max(4))?
    } else {
        None
    };
    let max_weight = weights
        .as_ref()
        .map_or(1., |w| w.iter().copied().fold(0., f64::max));
    let factors: Vec<_> = weights.as_ref().map_or_else(
        || vec![1.; time.len()],
        |w| w.iter().map(|v| v.sqrt() / max_weight.sqrt()).collect(),
    );
    let mut quality = None;
    let params = if op == KernelOperation::FitModel {
        let observed = numeric_array_from_aliases(input, &["observed", "y"], op)?;
        if observed.len() != time.len()
            || time.len() < 4
            || observed.iter().any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(error(
                "fitting needs four aligned finite nonnegative observations",
            ));
        }
        let (initial, limit) = crate::fitting::scalar_options(r)?;
        let ys = observed
            .iter()
            .zip(&factors)
            .filter(|(_, w)| **w > 0.)
            .map(|(y, _)| *y)
            .fold(0., f64::max)
            .max(1e-12);
        let ts = time.iter().copied().map(f64::abs).fold(1_f64, f64::max);
        let scale: Vec<_> = names
            .iter()
            .map(|n| {
                if n == "L" || n == "L_post" || n.starts_with("beta_L_") {
                    ys
                } else if n == "k" || n == "k_post" || n.starts_with("beta_k_") {
                    1. / ts
                } else {
                    ts
                }
            })
            .collect();
        let positive = |i: usize| ["L", "k", "L_post", "k_post"].contains(&names[i].as_str());
        let transform = |parameters: &[f64]| {
            parameters
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if positive(i) {
                        (v / scale[i]).ln()
                    } else {
                        v / scale[i]
                    }
                })
                .collect::<Vec<_>>()
        };
        let starts = if let Some(initial) = initial {
            vec![transform(&parse(&initial)?)]
        } else {
            let segment = |range: std::ops::Range<usize>, rate: f64| {
                let mut eligible = range.filter(|&i| factors[i] > 0.);
                let capacity = eligible
                    .clone()
                    .map(|i| observed[i])
                    .fold(0., f64::max)
                    .max(ys * 0.1)
                    * 1.1;
                let midpoint = eligible
                    .clone()
                    .min_by(|&a, &b| {
                        (observed[a] - capacity * 0.5)
                            .abs()
                            .total_cmp(&(observed[b] - capacity * 0.5).abs())
                    })
                    .map_or(time[0], |i| time[i]);
                let span = eligible
                    .clone()
                    .next()
                    .zip(eligible.next_back())
                    .map_or(ts, |(first, last)| (time[last] - time[first]).max(1.));
                [capacity, rate / span, midpoint]
            };
            let split = event.map_or(time.len(), |e| time.partition_point(|t| *t < e));
            [2., 5., 10.]
                .into_iter()
                .map(|rate| {
                    let mut p = segment(0..split, rate).to_vec();
                    if event.is_some() {
                        p.extend(segment(split..time.len(), rate));
                    }
                    p.resize(names.len(), 0.);
                    transform(&p)
                })
                .collect()
        };
        let untransform = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, v)| scale[i] * if positive(i) { v.exp() } else { *v })
                .collect::<Vec<_>>()
        };
        let fit = crate::numerics::least_squares(
            &observed
                .iter()
                .zip(&factors)
                .map(|(y, w)| (y * w) / ys)
                .collect::<Vec<_>>(),
            &starts,
            |v| {
                let p = untransform(v);
                time.iter()
                    .enumerate()
                    .map(|(i, &t)| point(&p, i, t).map(|(y, _)| (y * factors[i]) / ys))
                    .collect()
            },
            limit,
        )
        .map_err(|e| error(&e))?;
        let p = untransform(&fit.parameters);
        quality = Some((fit.report, fit.identifiability));
        p
    } else {
        parse(
            payload
                .get("state")
                .and_then(|s| s.get("parameters"))
                .or_else(|| payload.get("parameters"))
                .ok_or_else(|| error("parameters required"))?,
        )?
    };
    let (values, rates): (Vec<_>, Vec<_>) = time
        .iter()
        .enumerate()
        .map(|(i, &t)| point(&params, i, t).map(|(y, rate)| (y, [rate])))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| error(&e))?
        .into_iter()
        .unzip();
    let parameters: Map<_, _> = names
        .iter()
        .cloned()
        .zip(params.iter().map(|v| json!(v)))
        .collect();
    let predictions = json!({"values":values,"rates":rates,"shape":[time.len()],"dtype":"float64","metadata":{"event_boundary":"post_at_t_event","event_covariates":mode,"covariate_semantics":"pointwise additive parameters","rate_semantics":if control_slopes.is_some() && (event.is_none() || mode == "additive") {"time derivative with supplied covariate rates; post-event branch at the boundary, excluding jump impulses"} else {"time derivative holding covariates fixed; post-event branch at the boundary, excluding jump impulses"},"nonfinite_rate":"null"}});
    let state = json!({"model_key":"logistic","model_name":"LogisticModel","parameters":parameters,"constructor_kwargs":kw,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut result = json!({"model_key":"logistic","model_name":"LogisticModel","family":"diffusion","parameters":parameters,"state":state,"predictions":predictions});
        if op != KernelOperation::SummarizeModel
            || input.contains_key("observed")
            || input.contains_key("y")
        {
            let y = numeric_array_from_aliases(input, &["observed", "y"], op)?;
            if y.len() != time.len() || y.iter().any(|v| !v.is_finite() || *v < 0.) {
                return Err(error("invalid observations"));
            }
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &values,
                "LogisticModel",
                names.len() + 1,
            );
            if weights.is_some() {
                let weighted_rss: f64 = y
                    .iter()
                    .zip(&values)
                    .zip(&factors)
                    .map(|((y, prediction), factor)| (y * factor - prediction * factor).powi(2))
                    .sum();
                d["metrics"]["AIC"] = Value::Null;
                d["metrics"]["BIC"] = Value::Null;
                d["information_criteria"] =
                    json!({"available":false,"reason":"weighted likelihood not qualified"});
                d["weighting"] = json!({
                    "maximum_weight": max_weight,
                    "positive_observations": factors.iter().filter(|w| **w > 0.).count(),
                    "weighted_rss_normalized_by_max": weighted_rss,
                    "rank_semantics": "weighted local Jacobian; zero-weight residuals excluded",
                    "descriptive_metrics": "unweighted"
                });
            }
            if let Some((optimizer, rank)) = quality {
                if optimizer["converged"] != true {
                    d["warnings"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!("optimizer did not converge"));
                }
                if rank["status"] != "local_full_rank" {
                    d["warnings"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!("parameters are locally rank deficient"));
                }
                d["optimizer"] = optimizer;
                d["identifiability"] = rank;
            }
            if event.is_some() && !cov.is_empty() && mode == "historical_ignored" {
                d["warnings"].as_array_mut().unwrap().push(json!(
                    "Historical event path ignores covariates; coefficients are unidentifiable"
                ));
            }
            result["diagnostics"] = d;
        }
        result
    };
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        result: Some(result),
        error: None,
        metadata: json!({"runtime":"rust_native","scope":"historical_logistic"}),
    })
}

/// Profile a discrete split by refitting the historical curve for each candidate.
pub(crate) fn profile_event(
    request: &KernelRequest,
    options: &Map<String, Value>,
    time: &[f64],
    fit: fn(&KernelRequest) -> Result<KernelResponse, KernelBindingError>,
) -> Result<KernelResponse, KernelBindingError> {
    let error = |message: &str| KernelBindingError::invalid_request(request.operation, message);
    if request.operation != KernelOperation::FitModel
        || options.get("t_event").is_some_and(|event| !event.is_null())
        || request.payload.get("state").is_some()
    {
        return Err(error(
            "profiling requires a fresh fit without fixed t_event",
        ));
    }
    let launch = options.contains_key("launch_candidates");
    let event = options.contains_key("event_candidates");
    let joint = launch && event;
    if launch && options.get("variant").and_then(Value::as_str) != Some("successive_generations") {
        return Err(error("launch profiling requires successive_generations"));
    }
    let profile_key = match (launch, event) {
        (true, true) => "joint_time_profile",
        (true, false) => "launch_profile",
        _ => "event_profile",
    };
    let parse_times = |key: &str| {
        let candidates = options[key]
            .as_array()
            .filter(|values| (2..=16).contains(&values.len()))
            .ok_or_else(|| error("profiling requires 2 to 16 candidate times"))?;
        let times = candidates
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| error("candidate times must be finite numbers"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if times.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(error("candidate times must be strictly increasing"));
        }
        Ok(times.into_iter().map(Some).collect::<Vec<_>>())
    };
    let launches = if launch {
        parse_times("launch_candidates")?
    } else {
        vec![None]
    };
    let events = if event {
        parse_times("event_candidates")?
    } else {
        vec![None]
    };
    let trials = launches.len() * events.len();
    if trials > 16 {
        return Err(error("joint profiling allows at most 16 candidate pairs"));
    }
    let weights = crate::fitting::weights(request, time.len(), 6)?;
    let mut trial = request.clone();
    let mut fixed = options.clone();
    fixed.remove("event_candidates");
    fixed.remove("launch_candidates");
    let launch_index = if launch {
        let index = fixed
            .remove("launch_index")
            .and_then(|index| index.as_u64())
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index > 0)
            .ok_or_else(|| error("launch_index must select a later generation"))?;
        if fixed
            .get("launch_times")
            .and_then(|times| times.get(index))
            .is_none()
        {
            return Err(error("launch_index must select an existing launch time"));
        }
        Some(index)
    } else {
        None
    };
    let event_index = if event
        && matches!(
            request.model_key.as_deref(),
            Some(
                "norton_bass"
                    | "lotka_volterra"
                    | "complementary_goods"
                    | "lock_in"
                    | "multi_product"
            )
        ) {
        let index = fixed
            .remove("event_index")
            .and_then(|index| index.as_u64())
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| error("Parameter event profiling requires a zero-based event_index"))?;
        if !fixed
            .get("parameter_events")
            .and_then(|events| events.get(index))
            .is_some_and(Value::is_object)
        {
            return Err(error("event_index must select an existing parameter event"));
        }
        Some(index)
    } else {
        None
    };
    let mut profile = Vec::with_capacity(trials);
    let mut best = None;
    let mut best_score = f64::INFINITY;
    let pairs = launches
        .into_iter()
        .flat_map(|launch| events.iter().map(move |event| (launch, *event)));
    for (launch_time, event_time) in pairs {
        let mut row = Map::new();
        if let Some(t) = launch_time {
            row.insert("launch_time".into(), json!(t));
        }
        if let Some(t) = event_time {
            row.insert("t_event".into(), json!(t));
        }
        let mut insufficient_weight = false;
        for time_candidate in [launch_time, event_time].into_iter().flatten() {
            let split = time.partition_point(|time| *time < time_candidate);
            if split < 3 || time.len() - split < 3 {
                return Err(error(
                    "each candidate requires at least three observations per branch",
                ));
            }
            if let Some(weights) = &weights {
                let positive =
                    |weights: &[f64]| weights.iter().filter(|weight| **weight > 0.).count();
                insufficient_weight |=
                    positive(&weights[..split]) < 3 || positive(&weights[split..]) < 3;
            }
        }
        if insufficient_weight {
            row.insert(
                "error".into(),
                json!("fewer than three positive-weight observations on a branch"),
            );
            profile.push(row);
            continue;
        }
        if let Some((index, t)) = launch_index.zip(launch_time) {
            fixed
                .get_mut("launch_times")
                .expect("validated launch list")[index] = json!(t);
        }
        if let Some(t) = event_time {
            if let Some(index) = event_index {
                fixed
                    .get_mut("parameter_events")
                    .expect("validated event list")[index]["time"] = json!(t);
            } else {
                fixed.insert("t_event".into(), json!(t));
            }
        }
        trial.payload["constructor_kwargs"] = json!(fixed);
        match fit(&trial) {
            Ok(response) => {
                let diagnostics = &response.result.as_ref().expect("fit result")["diagnostics"];
                let weighted = diagnostics.get("weighting").is_some();
                let score = if weighted {
                    diagnostics["weighting"]["weighted_rss_normalized_by_max"].as_f64()
                } else {
                    diagnostics["metrics"]["RSS"].as_f64()
                };
                let converged = diagnostics["optimizer"]["converged"] == true;
                row.insert("objective".into(), json!(score));
                row.insert("converged".into(), json!(converged));
                if let Some(score) = score.filter(|score| converged && *score < best_score) {
                    best_score = score;
                    best = Some(response);
                }
            }
            Err(failure) => {
                row.insert("error".into(), json!(failure.to_string()));
            }
        }
        profile.push(row);
    }
    let mut response = best.ok_or_else(|| error("no candidate produced a converged finite fit"))?;
    response.result.as_mut().expect("fit result")[profile_key] = json!({
        "method":"discrete_profile_least_squares","candidates":profile,
        "tie_break":if joint {"earliest_launch_then_event"} else {"earliest_candidate"}
    });
    if let Some(index) = event_index {
        response.result.as_mut().expect("fit result")[profile_key]["event_index"] = json!(index);
    }
    if let Some(index) = launch_index {
        response.result.as_mut().expect("fit result")[profile_key]["launch_index"] = json!(index);
    }
    Ok(response)
}
