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
    let predict = |p: &[f64]| -> Result<Vec<f64>, String> {
        time.iter()
            .enumerate()
            .map(|(i, t)| {
                let post = event.is_some_and(|e| *t >= e);
                let offset = if post { 3 } else { 0 };
                let mut a = [p[offset], p[offset + 1], p[offset + 2]];
                if event.is_none() || mode == "additive" {
                    let start = if event.is_some() { 6 } else { 3 };
                    for (j, x) in series.iter().enumerate() {
                        for k in 0..3 {
                            a[k] += p[start + 3 * j + k] * x[i];
                        }
                    }
                }
                if a.iter().any(|x| !x.is_finite()) || a[0] <= 0. || a[1] <= 0. {
                    return Err("effective L/k must stay positive and finite".into());
                }
                let z = a[1] * (t - a[2]);
                let f = if z >= 0. {
                    1. / (1. + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1. + e)
                };
                Ok(a[0] * f)
            })
            .collect()
    };
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
        let (initial, limit) = crate::fitting::options(r)?;
        let initial = initial.ok_or_else(|| {
            error("advanced Logistic fitting requires explicit initial_parameters")
        })?;
        let initial = parse(&initial)?;
        let ys = observed.iter().copied().fold(0_f64, f64::max).max(1e-12);
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
        let start: Vec<_> = initial
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if positive(i) {
                    (v / scale[i]).ln()
                } else {
                    v / scale[i]
                }
            })
            .collect();
        let untransform = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, v)| scale[i] * if positive(i) { v.exp() } else { *v })
                .collect::<Vec<_>>()
        };
        let fit = crate::numerics::least_squares(
            &observed.iter().map(|y| y / ys).collect::<Vec<_>>(),
            &[start],
            |v| predict(&untransform(v)).map(|p| p.iter().map(|x| x / ys).collect()),
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
    let values = predict(&params).map_err(|e| error(&e))?;
    let parameters: Map<_, _> = names
        .iter()
        .cloned()
        .zip(params.iter().map(|v| json!(v)))
        .collect();
    let predictions = json!({"values":values,"shape":[time.len()],"dtype":"float64","metadata":{"event_boundary":"post_at_t_event","event_covariates":mode,"covariate_semantics":"pointwise additive parameters"}});
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
