//! Grouped scalar point fits with explicit historical or centered parameters.
use crate::validation::{numeric_array, numeric_array_from_aliases};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBinding, KernelBindingError, KernelOperation, KernelRequest,
    KernelResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default = "base")]
    base_model: String,
    groups: Vec<String>,
    #[serde(default = "parameterization")]
    parameterization: String,
    #[serde(default = "output")]
    output: String,
}
fn base() -> String {
    "bass".into()
}
fn parameterization() -> String {
    "centered".into()
}
fn output() -> String {
    "aggregate".into()
}
fn observations(
    v: &Value,
    time: usize,
    groups: &[String],
    op: KernelOperation,
) -> Result<(bool, Vec<Vec<f64>>), KernelBindingError> {
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    let (shared, rows) = if let Some(map) = v.as_object() {
        if map.len() != groups.len() || groups.iter().any(|g| !map.contains_key(g)) {
            return Err(invalid("observed group mapping must match groups exactly"));
        }
        let cols: Vec<_> = groups
            .iter()
            .map(|g| numeric_array(&map[g], g, op))
            .collect::<Result<_, _>>()?;
        if cols.iter().any(|v| v.len() != time) {
            return Err(invalid("group series must match time"));
        }
        (
            false,
            (0..time)
                .map(|i| cols.iter().map(|c| c[i]).collect())
                .collect::<Vec<Vec<f64>>>(),
        )
    } else if v
        .as_array()
        .and_then(|a| a.first())
        .is_some_and(Value::is_array)
    {
        (
            false,
            serde_json::from_value::<Vec<Vec<f64>>>(v.clone())
                .map_err(|_| invalid("observations must be numeric time-by-group rows"))?,
        )
    } else {
        let y = numeric_array(v, "observed", op)?;
        (true, y.into_iter().map(|v| vec![v]).collect())
    };
    if rows.len() != time
        || rows.iter().any(|row| {
            row.len() != if shared { 1 } else { groups.len() }
                || row.iter().any(|v| !v.is_finite() || *v < 0.)
        })
    {
        return Err(invalid(
            "observations must be finite nonnegative and aligned with time/groups",
        ));
    }
    Ok((shared, rows))
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
            "group_fit_options",
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid("unknown hierarchical payload option"));
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
        .is_some_and(|m| m != "hierarchical")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let o: Options = serde_json::from_value(
        p.get("constructor_kwargs")
            .or_else(|| p.get("model_kwargs"))
            .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let names: &[&str] = match o.base_model.as_str() {
        "bass" => &["p", "q", "m"],
        "logistic" => &["L", "k", "x0"],
        "gompertz" => &["a", "b", "c"],
        "fisher_pry" => &["alpha", "t0"],
        _ => return Err(invalid("unsupported hierarchical base curve")),
    };
    let mut unique = std::collections::HashSet::new();
    if o.groups.is_empty()
        || o.groups.len() > 128
        || o.groups
            .iter()
            .any(|g| g.is_empty() || g == "global" || !unique.insert(g))
        || !["historical_additive", "centered"].contains(&o.parameterization.as_str())
        || !["aggregate", "groups"].contains(&o.output.as_str())
    {
        return Err(invalid(
            "groups must be 1..=128 unique nonempty labels excluding global; invalid parameterization/output",
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
        return Err(invalid("hierarchical base covariates not ported"));
    }
    let time = if op == KernelOperation::SummarizeModel
        && !input.contains_key("time")
        && !input.contains_key("t")
    {
        vec![]
    } else {
        numeric_array_from_aliases(input, &["time", "t"], op)?
    };
    if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("time must be finite and strictly increasing"));
    }
    let observed = if op == KernelOperation::FitModel
        || op == KernelOperation::DiagnoseModel
        || input.contains_key("observed")
        || input.contains_key("y")
    {
        Some(observations(
            input
                .get("observed")
                .or_else(|| input.get("y"))
                .ok_or_else(|| invalid("observed required"))?,
            time.len(),
            &o.groups,
            op,
        )?)
    } else {
        None
    };
    let mut parameters = Map::new();
    let mut group_diagnostics = vec![];
    if op == KernelOperation::FitModel {
        let (shared, rows) = observed.as_ref().unwrap();
        let common = p
            .get("base_fit_options")
            .or_else(|| p.get("fit_options"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        if !common.is_object()
            || (p.contains_key("base_fit_options")
                && p.get("fit_options")
                    .is_some_and(|v| !v.as_object().is_some_and(Map::is_empty)))
        {
            return Err(invalid(
                "provide common fit_options or base_fit_options object, not both",
            ));
        }
        let individual = p
            .get("group_fit_options")
            .map(|v| {
                v.as_object()
                    .ok_or_else(|| invalid("group_fit_options must be a group mapping"))
            })
            .transpose()?;
        if individual.is_some_and(|v| {
            *shared
                || !common.as_object().unwrap().is_empty()
                || v.len() != o.groups.len()
                || o.groups.iter().any(|g| !v.contains_key(g))
        }) {
            return Err(invalid(
                "group_fit_options requires grouped data, exact group keys and no common fit options",
            ));
        }
        let count = if *shared { 1 } else { o.groups.len() };
        let mut fits = vec![];
        for i in 0..count {
            let y: Vec<_> = rows.iter().map(|r| r[i]).collect();
            let options = individual
                .map(|m| m[&o.groups[i]].clone())
                .unwrap_or_else(|| common.clone());
            let f = KernelBinding::new()
                .fit_model(
                    &o.base_model,
                    json!({"inputs":{"time":time,"observed":y},"fit_options":options}),
                )?
                .result
                .ok_or_else(|| invalid("missing group fit"))?;
            group_diagnostics.push(f["diagnostics"].clone());
            fits.push(f["parameters"].clone());
        }
        for name in names {
            let global = fits
                .iter()
                .map(|f| f[name].as_f64().unwrap() / count as f64)
                .sum::<f64>();
            if !global.is_finite() {
                return Err(invalid("global parameter mean overflow"));
            }
            parameters.insert(format!("global_{name}"), json!(global));
            for (i, g) in o.groups.iter().enumerate() {
                let adjustment = if *shared {
                    0.
                } else {
                    fits[i][name].as_f64().unwrap()
                        - if o.parameterization == "centered" {
                            global
                        } else {
                            0.
                        }
                };
                if !adjustment.is_finite() {
                    return Err(invalid("group parameter adjustment overflow"));
                }
                parameters.insert(format!("{g}_{name}"), json!(adjustment));
            }
        }
    } else {
        parameters = p
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| p.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?
            .clone();
        if parameters.len() != (o.groups.len() + 1) * names.len() {
            return Err(invalid("hierarchical parameter count mismatch"));
        }
    }
    let eval = if time.is_empty() {
        vec![0.]
    } else {
        time.clone()
    };
    let mut columns = vec![];
    let mut effective = Map::new();
    for g in &o.groups {
        let mut params = Map::new();
        for name in names {
            let global = parameters
                .get(&format!("global_{name}"))
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("missing global parameter"))?;
            let adjustment = parameters
                .get(&format!("{g}_{name}"))
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("missing group parameter"))?;
            let value = global + adjustment;
            if !global.is_finite() || !adjustment.is_finite() || !value.is_finite() {
                return Err(invalid("effective group parameter overflow"));
            }
            params.insert(name.to_string(), json!(value));
        }
        let a = KernelBinding::new()
            .predict_model(
                &o.base_model,
                json!({"inputs":{"time":eval},"parameters":params}),
            )?
            .result
            .ok_or_else(|| invalid("missing group predictions"))?;
        columns.push(numeric_array(&a["values"], "group predictions", op)?);
        effective.insert(g.clone(), json!(params));
    }
    let rows: Vec<Vec<_>> = (0..time.len())
        .map(|i| columns.iter().map(|c| c[i]).collect())
        .collect();
    let aggregate: Vec<f64> = rows.iter().map(|r| r.iter().sum()).collect();
    if aggregate.iter().any(|v| !v.is_finite()) {
        return Err(invalid("aggregate prediction overflow"));
    }
    let increments: Vec<_> = aggregate
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if aggregate.len() < 2 {
                0.
            } else if i == 0 {
                aggregate[1] - v
            } else {
                v - aggregate[i - 1]
            }
        })
        .collect();
    let mut predictions = json!({"aggregate_values":aggregate,"group_rows":rows,"increments":increments,"dtype":"float64","metadata":{"groups":o.groups,"parameterization":o.parameterization,"output":o.output,"hierarchical_semantics":"independent grouped point fits, not partial pooling or Bayesian inference","rate_semantics":"historical sample differences with first difference prepended; singleton defined as zero; not derivative","simulation_semantics":"deterministic point path; stochastic draws unavailable"}});
    if o.output == "aggregate" {
        predictions["values"] = json!(aggregate);
        predictions["shape"] = json!([time.len()]);
    } else {
        predictions["rows"] = json!(rows);
        predictions["shape"] = json!([time.len(), o.groups.len()]);
    }
    let state = json!({"model_key":"hierarchical","parameters":parameters,"constructor_kwargs":o,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut out = json!({"model_key":"hierarchical","model_name":"HierarchicalModel","parameters":parameters,"state":state,"predictions":predictions,"group_diagnostics":group_diagnostics,"details":{"groups":o.groups,"base_model":o.base_model,"parameterization":o.parameterization,"effective_group_parameters":effective,"fit_input_kind":observed.as_ref().map(|(shared,_)|if *shared{"shared_series"}else{"grouped_series"})}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some((shared, y)) = observed {
            let predicted = if shared {
                aggregate.iter().map(|v| vec![*v]).collect()
            } else {
                rows
            };
            let rss = y
                .iter()
                .flatten()
                .zip(predicted.iter().flatten())
                .map(|(y, v)| (y - v).powi(2))
                .sum::<f64>();
            if !rss.is_finite() {
                return Err(invalid("hierarchical residual sum of squares overflow"));
            }
            let mut d = crate::substitution::diagnostics(
                &time,
                &y,
                &predicted,
                "HierarchicalModel",
                parameters.len(),
            );
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"global/group decomposition is constrained or nonidentifiable; effective degrees of freedom unqualified"});
            d["warnings"].as_array_mut().unwrap().push(json!("Groups are fitted independently; historical_additive adds pooled means to absolute group estimates. Shared data is per-group, while aggregate predictions sum all groups. This is not partial pooling or calibrated uncertainty."));
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
        metadata: json!({"runtime":"rust_native","family":"hierarchical"}),
    })
}
