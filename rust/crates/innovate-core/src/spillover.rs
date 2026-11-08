//! Historical deterministic network wrapper; distinct from Kairos event simulation.
use crate::validation::{numeric_array, numeric_array_from_aliases, object_section};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBinding, KernelBindingError, KernelOperation, KernelRequest,
    KernelResponse,
};
use serde_json::{Map, Value, json};
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    let payload = r
        .payload
        .as_object()
        .ok_or_else(|| invalid("payload must be an object"))?;
    if payload.keys().any(|k| {
        ![
            "inputs",
            "constructor_kwargs",
            "model_kwargs",
            "state",
            "parameters",
            "fit_options",
            "node_fit_options",
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid(
            "unknown network payload option; stochastic draws/interventions are not implemented",
        ));
    }
    let empty_input = Map::new();
    let input = if op == KernelOperation::SummarizeModel && !payload.contains_key("inputs") {
        &empty_input
    } else {
        object_section(payload, "inputs", op)?
    };
    let time = if op == KernelOperation::SummarizeModel
        && !input.contains_key("time")
        && !input.contains_key("t")
    {
        vec![]
    } else {
        numeric_array_from_aliases(input, &["time", "t"], op)?
    };
    if time.iter().any(|x| !x.is_finite()) || time.windows(2).any(|p| p[0] >= p[1]) {
        return Err(invalid("time must be finite and strictly increasing"));
    }
    let empty = json!({});
    let kw = payload
        .get("constructor_kwargs")
        .or_else(|| payload.get("model_kwargs"))
        .or_else(|| {
            payload
                .get("state")
                .and_then(|s| s.get("constructor_kwargs"))
        })
        .unwrap_or(&empty)
        .as_object()
        .ok_or_else(|| invalid("constructor options must be an object"))?;
    if kw.keys().any(|k| {
        ![
            "adjacency",
            "node_labels",
            "base_model",
            "spillover_strength",
            "orientation",
            "coordinates",
        ]
        .contains(&k.as_str())
    }) || input.contains_key("covariates")
        || payload.contains_key("simulation_options")
    {
        return Err(invalid(
            "unsupported network option/covariates/stochastic simulation",
        ));
    }
    for value in [
        payload.get("predict_kwargs"),
        payload.get("fitter_options"),
        payload.get("state").and_then(|s| s.get("predict_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        if !value.as_object().is_some_and(Map::is_empty) {
            return Err(invalid("custom prediction/fitter options unavailable"));
        }
    }
    if payload
        .get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != "network_diffusion")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let adjacency: Vec<Vec<f64>> = serde_json::from_value(
        kw.get("adjacency")
            .cloned()
            .ok_or_else(|| invalid("adjacency required"))?,
    )
    .map_err(|_| invalid("adjacency must be numeric matrix"))?;
    let n = adjacency.len();
    if n == 0
        || n > 128
        || adjacency
            .iter()
            .any(|row| row.len() != n || row.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "adjacency must be finite square matrix with 1..128 nodes",
        ));
    }
    let adjacency: Vec<Vec<f64>> = adjacency
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, v)| if i == j { 0. } else { *v })
                .collect()
        })
        .collect();
    let coordinates: Option<Vec<Vec<f64>>> = kw
        .get("coordinates")
        .filter(|v| !v.is_null())
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|_| invalid("coordinates must be a numeric matrix"))?;
    if coordinates.as_ref().is_some_and(|c| {
        c.len() != n
            || c[0].is_empty()
            || c.iter()
                .any(|r| r.len() != c[0].len() || r.iter().any(|x| !x.is_finite()))
    }) {
        return Err(invalid(
            "coordinates must be finite, rectangular and match nodes",
        ));
    }
    let labels: Vec<String> = kw
        .get("node_labels")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|_| invalid("node_labels must be strings"))?
        .unwrap_or_else(|| (0..n).map(|i| i.to_string()).collect());
    if labels.len() != n
        || labels.iter().any(String::is_empty)
        || labels
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != n
    {
        return Err(invalid("node labels must be unique and match dimensions"));
    }
    let base = kw
        .get("base_model")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("base_model must be a string"))
        })
        .transpose()?
        .unwrap_or("bass");
    let names: &[&str] = match base {
        "bass" => &["p", "q", "m"],
        "logistic" => &["L", "k", "x0"],
        "gompertz" => &["a", "b", "c"],
        "fisher_pry" => &["alpha", "t0"],
        _ => {
            return Err(invalid(
                "base model currently requires a native scalar curve",
            ));
        }
    };
    let orientation = kw
        .get("orientation")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("orientation must be a string"))
        })
        .transpose()?
        .unwrap_or("time_node");
    if !["time_node", "node_time", "historical_auto"].contains(&orientation) {
        return Err(invalid("unknown observation orientation"));
    }
    let observations = || -> Result<Vec<Vec<f64>>, KernelBindingError> {
        let v = input
            .get("observed")
            .or_else(|| input.get("y"))
            .ok_or_else(|| invalid("observed required"))?;
        let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone())
            .map_err(|_| invalid("observations must be a numeric matrix"))?;
        let transpose = orientation == "node_time"
            || (orientation == "historical_auto"
                && rows.len() == n
                && rows.iter().all(|r| r.len() == time.len()));
        let rows = if transpose {
            if rows.len() != n || rows.iter().any(|r| r.len() != time.len()) {
                return Err(invalid("node_time observation shape mismatch"));
            }
            (0..time.len())
                .map(|i| rows.iter().map(|r| r[i]).collect())
                .collect()
        } else {
            rows
        };
        if rows.len() != time.len()
            || rows
                .iter()
                .any(|r| r.len() != n || r.iter().any(|x| !x.is_finite() || *x < 0.))
        {
            return Err(invalid(
                "time_node observations must be aligned finite nonnegative values",
            ));
        }
        Ok(rows)
    };
    let mut parameters = Map::new();
    let mut node_diagnostics = Vec::new();
    let mut strength = kw
        .get("spillover_strength")
        .map(|v| {
            v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid("spillover strength must be finite"))
        })
        .transpose()?
        .unwrap_or(0.15);
    if op == KernelOperation::FitModel {
        if payload
            .get("fit_options")
            .is_some_and(|v| !v.as_object().is_some_and(Map::is_empty))
        {
            return Err(invalid(
                "use node_fit_options for independent historical node fitting",
            ));
        }
        let rows = observations()?;
        let options = payload
            .get("node_fit_options")
            .map(|v| {
                v.as_array()
                    .filter(|v| v.len() == n)
                    .ok_or_else(|| invalid("node_fit_options must match nodes"))
            })
            .transpose()?;
        for j in 0..n {
            let y: Vec<_> = rows.iter().map(|r| r[j]).collect();
            let fit=KernelBinding::new().fit_model(base,json!({"inputs":{"time":time,"observed":y},"fit_options":options.map(|o|o[j].clone()).unwrap_or_else(||json!({}))}))?.result.ok_or_else(||invalid("missing node fit"))?;
            for name in names {
                parameters.insert(format!("node_{j}_{name}"), fit["parameters"][name].clone());
            }
            node_diagnostics.push(fit["diagnostics"].clone());
        }
        parameters.insert("spillover_strength".into(), json!(strength));
    } else {
        parameters = payload
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| payload.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?
            .clone();
        if parameters.len() != n * names.len() + 1 {
            return Err(invalid(
                "parameter count does not match nodes and base curve",
            ));
        }
        strength = parameters
            .get("spillover_strength")
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite())
            .ok_or_else(|| invalid("spillover_strength required and finite"))?;
    }
    let evaluation_time = if time.is_empty() {
        vec![0.]
    } else {
        time.clone()
    };
    let mut base_predictions = Vec::new();
    for j in 0..n {
        let mut p = Map::new();
        for name in names {
            p.insert(
                (*name).to_string(),
                parameters
                    .get(&format!("node_{j}_{name}"))
                    .cloned()
                    .ok_or_else(|| invalid("missing node parameter"))?,
            );
        }
        let result = KernelBinding::new()
            .predict_model(
                base,
                json!({"inputs":{"time":evaluation_time},"parameters":p}),
            )?
            .result
            .ok_or_else(|| invalid("missing node prediction"))?;
        base_predictions.push(numeric_array(&result["values"], "node predictions", op)?);
    }
    let mut rows = vec![vec![0.; n]; time.len()];
    for i in 0..n {
        let sum: f64 = adjacency[i].iter().sum();
        if !sum.is_finite() {
            return Err(invalid("adjacency row sum overflow"));
        }
        let mut maximum = 0_f64;
        for k in 0..time.len() {
            let peer = if sum == 0. {
                0.
            } else {
                adjacency[i]
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w / sum * base_predictions[j][k])
                    .sum()
            };
            let value = base_predictions[i][k] + strength * peer;
            if !value.is_finite() {
                return Err(invalid("spillover prediction overflow"));
            }
            maximum = maximum.max(value);
            rows[k][i] = maximum;
        }
    }
    let increments: Vec<Vec<f64>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, v)| if i == 0 { 0. } else { v - rows[i - 1][j] })
                .collect()
        })
        .collect();
    let predictions = json!({"rows":rows,"columns":labels,"increments":increments,"metadata":{"shape":[time.len(),n],"rate_semantics":"historical sample increments, not time derivatives","orientation":"time_node","batch_semantics":"cumulative maximum within requested batch"}});
    let kwargs = json!({"adjacency":adjacency,"node_labels":labels,"base_model":base,"spillover_strength":strength,"orientation":orientation,"coordinates":coordinates});
    let state = json!({"model_key":"network_diffusion","parameters":parameters,"constructor_kwargs":kwargs,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let edge_count = adjacency.iter().flatten().filter(|v| **v != 0.).count();
        let possible = n * (n - 1);
        let average_degree = adjacency
            .iter()
            .map(|row| row.iter().sum::<f64>() / n as f64)
            .sum::<f64>();
        if !average_degree.is_finite() {
            return Err(invalid("graph statistic exceeded finite domain"));
        }
        let mut out = json!({"model_key":"network_diffusion","model_name":"NetworkDiffusionModel","parameters":parameters,"state":state,"predictions":predictions,"node_diagnostics":node_diagnostics,"details":{"node_count":n,"node_labels":labels,"edge_count":edge_count,"edge_density":if possible==0 {0.}else{edge_count as f64/possible as f64},"average_degree":average_degree,"coordinates":coordinates,"spillover_strength":strength,"base_model":base}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if op != KernelOperation::SummarizeModel
            || input.contains_key("observed")
            || input.contains_key("y")
        {
            let observed = observations()?;
            let mut d = crate::substitution::diagnostics(
                &time,
                &observed,
                &rows,
                "NetworkDiffusionModel",
                n * names.len(),
            );
            d["fit_semantics"] = json!(
                "independent node fits ignore spillover; strength held fixed, preserving historical algorithm"
            );
            d["warnings"].as_array_mut().unwrap().push(json!("Adjusted network training predictions need not fit observations because node fits ignore spillover"));
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
        metadata: json!({"runtime":"rust_native","family":"network_diffusion"}),
    })
}
