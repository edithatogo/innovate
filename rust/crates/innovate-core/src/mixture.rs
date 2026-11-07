//! Scalar-curve mixtures: Gaussian residual EM and explicit historical heuristic.
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
    components: Vec<String>,
    #[serde(default = "mode")]
    mode: String,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    #[serde(default = "iterations")]
    max_iter: usize,
    #[serde(default = "tolerance")]
    tol: f64,
    #[serde(default = "floor")]
    noise_floor: f64,
    #[serde(default)]
    noise_scale: Option<f64>,
}
fn mode() -> String {
    "gaussian_residual".into()
}
fn iterations() -> usize {
    100
}
fn tolerance() -> f64 {
    1e-6
}
fn floor() -> f64 {
    1e-6
}
fn names(model: &str) -> Option<&'static [&'static str]> {
    match model {
        "bass" => Some(&["p", "q", "m"]),
        "logistic" => Some(&["L", "k", "x0"]),
        "gompertz" => Some(&["a", "b", "c"]),
        "fisher_pry" => Some(&["alpha", "t0"]),
        _ => None,
    }
}
fn normalize(w: &mut [f64]) -> Result<(), String> {
    let sum = w.iter().sum::<f64>();
    if w.iter().any(|w| !w.is_finite() || *w < 0.)
        || !sum.is_finite()
        || sum <= 0.
        || (sum - 1.).abs() > 1e-10
    {
        return Err("mixture weights must be finite nonnegative and sum to one".into());
    }
    for v in w {
        *v /= sum;
    }
    Ok(())
}
fn curves(
    o: &Options,
    params: &[Map<String, Value>],
    time: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    o.components
        .iter()
        .zip(params)
        .map(|(model, p)| {
            let result = KernelBinding::new()
                .predict_model(model, json!({"inputs":{"time":time},"parameters":p}))
                .map_err(|e| e.to_string())?
                .result
                .ok_or("missing component prediction")?;
            numeric_array(
                &result["values"],
                "component values",
                KernelOperation::PredictModel,
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}
fn posterior(
    o: &Options,
    columns: &[Vec<f64>],
    weights: &[f64],
    noise: f64,
    y: &[f64],
) -> Result<(Vec<Vec<f64>>, f64), String> {
    let mut rows = vec![];
    let mut objective = 0.;
    for (i, y) in y.iter().enumerate() {
        let scores: Vec<_> = columns
            .iter()
            .zip(weights)
            .map(|(c, w)| {
                w.ln()
                    + if o.mode == "historical_adoption" {
                        (c[i] + 1e-9).ln()
                    } else {
                        -0.5 * ((y - c[i]) / noise).powi(2)
                            - noise.ln()
                            - 0.5 * (2. * std::f64::consts::PI).ln()
                    }
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return Err("component log scores exceed representable domain".into());
        }
        let relative: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
        let total = relative.iter().sum::<f64>();
        objective += max + total.ln();
        rows.push(relative.iter().map(|v| v / total).collect());
    }
    if !objective.is_finite() {
        return Err("mixture objective overflow".into());
    }
    Ok((rows, objective))
}
struct ModelParameters {
    components: Vec<Map<String, Value>>,
    weights: Vec<f64>,
    noise: f64,
}
fn parse(o: &Options, p: &Map<String, Value>) -> Result<ModelParameters, String> {
    let gaussian = o.mode == "gaussian_residual";
    let expected = o
        .components
        .iter()
        .map(|c| names(c).unwrap().len())
        .sum::<usize>()
        + o.components.len()
        + usize::from(gaussian);
    if p.len() != expected {
        return Err("mixture parameter names/count mismatch".into());
    }
    let mut components = vec![];
    let mut weights = vec![];
    for (i, c) in o.components.iter().enumerate() {
        let mut params = Map::new();
        for name in names(c).unwrap() {
            let v = p
                .get(&format!("model_{i}_{name}"))
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .ok_or("missing/nonfinite component parameter")?;
            params.insert(name.to_string(), json!(v));
        }
        components.push(params);
        weights.push(
            p.get(&format!("weight_{i}"))
                .and_then(Value::as_f64)
                .ok_or("missing component weight")?,
        );
    }
    normalize(&mut weights)?;
    let noise = if gaussian {
        p.get("noise_scale")
            .and_then(Value::as_f64)
            .ok_or("noise_scale required")?
    } else {
        1.
    };
    if !noise.is_finite() || noise < o.noise_floor || noise <= 0. {
        return Err("noise_scale must be finite, positive and at least noise_floor".into());
    }
    Ok(ModelParameters {
        components,
        weights,
        noise,
    })
}
fn flatten(
    params: &[Map<String, Value>],
    weights: &[f64],
    noise: f64,
    gaussian: bool,
) -> Map<String, Value> {
    let mut out = Map::new();
    for (i, p) in params.iter().enumerate() {
        for (k, v) in p {
            out.insert(format!("model_{i}_{k}"), v.clone());
        }
        out.insert(format!("weight_{i}"), json!(weights[i]));
    }
    if gaussian {
        out.insert("noise_scale".into(), json!(noise));
    }
    out
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
            "component_fit_options",
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid("unknown mixture payload option"));
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
        .is_some_and(|m| m != "mixture")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let mut o: Options = serde_json::from_value(
        p.get("constructor_kwargs")
            .or_else(|| p.get("model_kwargs"))
            .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    if o.components.is_empty()
        || o.components.len() > 16
        || o.components.iter().any(|c| names(c).is_none())
        || !["gaussian_residual", "historical_adoption"].contains(&o.mode.as_str())
        || o.max_iter == 0
        || o.max_iter > 200
        || !o.tol.is_finite()
        || o.tol <= 0.
        || !o.noise_floor.is_finite()
        || o.noise_floor <= 0.
        || o.noise_scale
            .is_some_and(|v| !v.is_finite() || v < o.noise_floor)
    {
        return Err(invalid(
            "invalid mixture components/mode/iteration/tolerance/noise bounds",
        ));
    }
    let gaussian = o.mode == "gaussian_residual";
    let count = o.components.len();
    if let Some(w) = &mut o.weights {
        if w.len() != count {
            return Err(invalid("weights must align with components"));
        }
        normalize(w).map_err(|e| invalid(&e))?;
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
        return Err(invalid("mixture base covariates not ported"));
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
        let y = numeric_array_from_aliases(input, &["observed", "y"], op)?;
        if y.len() != time.len() || y.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid(
                "observations must be finite nonnegative and aligned",
            ));
        }
        Some(y)
    } else {
        None
    };
    let mut initialization = vec![];
    let mut updates = vec![];
    let mut history = vec![];
    let mut converged = false;
    let mut termination = "not_evaluated_in_this_call";
    let mut iterations = 0;
    let (mut components, mut weights, mut noise);
    if op == KernelOperation::FitModel {
        if p.contains_key("parameters") || p.contains_key("state") {
            return Err(invalid(
                "fit initialization uses fit_options.initial_parameters",
            ));
        }
        let fit_options = p
            .get("fit_options")
            .map(|v| {
                v.as_object()
                    .ok_or_else(|| invalid("fit_options must be an object"))
            })
            .transpose()?
            .unwrap_or(&empty);
        if fit_options
            .keys()
            .any(|k| !["initial_parameters", "component_max_iterations"].contains(&k.as_str()))
        {
            return Err(invalid("unsupported mixture fit option"));
        }
        let limit = fit_options
            .get("component_max_iterations")
            .map(|v| {
                v.as_u64()
                    .filter(|v| *v > 0 && *v <= 2000)
                    .ok_or_else(|| invalid("component_max_iterations must be in 1..=2000"))
            })
            .transpose()?
            .unwrap_or(300);
        let individual = p
            .get("component_fit_options")
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| invalid("component_fit_options must be an array"))
            })
            .transpose()?;
        if individual.is_some_and(|v| {
            v.len() != count
                || v.iter().any(|o| {
                    !o.as_object().is_some_and(|o| {
                        o.keys()
                            .all(|k| ["initial_parameters", "max_iterations"].contains(&k.as_str()))
                    })
                })
        }) {
            return Err(invalid(
                "component_fit_options must align and contain only initial_parameters/max_iterations",
            ));
        }
        let y = observed.as_ref().unwrap();
        if let Some(initial) = fit_options.get("initial_parameters") {
            if individual.is_some() {
                return Err(invalid(
                    "provide mixture initial_parameters or component initial fits, not both",
                ));
            }
            let parsed = parse(
                &o,
                initial
                    .as_object()
                    .ok_or_else(|| invalid("initial_parameters must be an object"))?,
            )
            .map_err(|e| invalid(&e))?;
            components = parsed.components;
            weights = parsed.weights;
            noise = parsed.noise;
        } else {
            components = vec![];
            weights = o
                .weights
                .clone()
                .unwrap_or_else(|| vec![1. / count as f64; count]);
            for (i, model) in o.components.iter().enumerate() {
                let mut options = individual
                    .map(|a| a[i].clone())
                    .unwrap_or_else(|| json!({}));
                if options.get("max_iterations").is_none() {
                    options["max_iterations"] = json!(limit);
                }
                let f = KernelBinding::new()
                    .fit_model(
                        model,
                        json!({"inputs":{"time":time,"observed":y},"fit_options":options}),
                    )?
                    .result
                    .ok_or_else(|| invalid("missing component initialization"))?;
                initialization.push(f["diagnostics"].clone());
                let mut params = f["parameters"].as_object().unwrap().clone();
                if gaussian && count > 1 {
                    let spread = (i as f64 / (count - 1) as f64 - 0.5) * 0.4;
                    for (name, value) in &mut params {
                        let value_number = value.as_f64().unwrap();
                        *value = json!(if ["x0", "t0"].contains(&name.as_str()) {
                            value_number + spread * (time.last().unwrap() - time[0])
                        } else {
                            value_number * (1. + spread)
                        });
                    }
                }
                components.push(params);
            }
            let columns = curves(&o, &components, &time).map_err(|e| invalid(&e))?;
            let variance = y
                .iter()
                .enumerate()
                .map(|(i, y)| {
                    let mean = columns
                        .iter()
                        .zip(&weights)
                        .map(|(c, w)| c[i] * w)
                        .sum::<f64>();
                    (y - mean).powi(2) / time.len() as f64
                })
                .sum::<f64>();
            noise = if gaussian {
                o.noise_scale.unwrap_or(variance.sqrt().max(o.noise_floor))
            } else {
                1.
            };
            if !noise.is_finite() {
                return Err(invalid("initial residual variance overflow"));
            }
        }
        let mut columns = curves(&o, &components, &time).map_err(|e| invalid(&e))?;
        let (mut responsibilities, mut objective) =
            posterior(&o, &columns, &weights, noise, y).map_err(|e| invalid(&e))?;
        history.push(objective);
        termination = "iteration_limit";
        let mut previous_pre_score = None;
        for iteration in 0..o.max_iter {
            iterations = iteration + 1;
            let mut candidate_weights: Vec<_> = (0..count)
                .map(|i| {
                    responsibilities
                        .iter()
                        .map(|r| r[i] / y.len() as f64)
                        .sum::<f64>()
                })
                .collect();
            normalize(&mut candidate_weights).map_err(|e| invalid(&e))?;
            let mut candidate = components.clone();
            let mut failed = false;
            let mut component_reports = vec![];
            for (i, model) in o.components.iter().enumerate() {
                if candidate_weights[i] == 0. {
                    component_reports.push(json!({"component":i,"status":"zero_weight_retained"}));
                    continue;
                }
                let w: Vec<_> = responsibilities
                    .iter()
                    .map(|r| r[i] + if gaussian { 0. } else { 1e-9 })
                    .collect();
                let component_limit = individual
                    .and_then(|a| a[i].get("max_iterations"))
                    .cloned()
                    .unwrap_or_else(|| json!(limit));
                match KernelBinding::new().fit_model(model,json!({"inputs":{"time":time,"observed":y},"fit_options":{"initial_parameters":components[i],"weights":w,"max_iterations":component_limit}})){
                    Ok(f)=>{let result=f.result.ok_or_else(||invalid("missing component update"))?;
                        let success=result["diagnostics"]["optimizer"]["converged"]==true;
                        component_reports.push(json!({"component":i,"status":if success{"updated"}else{"optimizer_not_converged"},"diagnostics":result["diagnostics"]}));
                        if success{candidate[i]=result["parameters"].as_object().unwrap().clone();}else{failed=true;}
                    },Err(e)=>{failed=true;component_reports.push(json!({"component":i,"status":"update_failed","code":e.code,"message":e.message}));}
                }
            }
            updates.push(
                json!({"iteration":iterations,"accepted":false,"components":component_reports}),
            );
            if failed {
                termination = "component_update_failed_rolled_back";
                break;
            }
            let candidate_columns = curves(&o, &candidate, &time).map_err(|e| invalid(&e))?;
            let mut candidate_noise = noise;
            if gaussian {
                let variance = responsibilities
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        r.iter()
                            .enumerate()
                            .map(|(k, g)| {
                                g * (y[i] - candidate_columns[k][i]).powi(2) / y.len() as f64
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                candidate_noise = variance.sqrt().max(o.noise_floor);
                if !variance.is_finite() {
                    return Err(invalid("mixture residual variance overflow"));
                }
            }
            let (candidate_responsibilities, candidate_objective) = posterior(
                &o,
                &candidate_columns,
                &candidate_weights,
                candidate_noise,
                y,
            )
            .map_err(|e| invalid(&e))?;
            if gaussian && candidate_objective < objective - 1e-10 * (1. + objective.abs()) {
                termination = "likelihood_decreased_rolled_back";
                break;
            }
            let stop = if gaussian {
                (candidate_objective - objective).abs() < o.tol
            } else {
                previous_pre_score.is_some_and(|previous: f64| (objective - previous).abs() < o.tol)
            };
            updates.last_mut().unwrap()["accepted"] = json!(true);
            previous_pre_score = Some(objective);
            components = candidate;
            weights = candidate_weights;
            noise = candidate_noise;
            columns = candidate_columns;
            responsibilities = candidate_responsibilities;
            objective = candidate_objective;
            history.push(objective);
            if stop {
                converged = true;
                termination = "objective_tolerance";
                break;
            }
        }
        let _ = columns;
    } else {
        let params = p
            .get("state")
            .and_then(|s| s.get("parameters"))
            .or_else(|| p.get("parameters"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("parameters required"))?;
        let parsed = parse(&o, params).map_err(|e| invalid(&e))?;
        components = parsed.components;
        weights = parsed.weights;
        noise = parsed.noise;
    }
    let evaluation = if time.is_empty() {
        vec![0.]
    } else {
        time.clone()
    };
    let columns = curves(&o, &components, &evaluation).map_err(|e| invalid(&e))?;
    let values: Vec<_> = (0..time.len())
        .map(|i| {
            columns
                .iter()
                .zip(&weights)
                .map(|(c, w)| c[i] * w)
                .sum::<f64>()
        })
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("mixture prediction overflow"));
    }
    let rows: Vec<Vec<_>> = (0..time.len())
        .map(|i| columns.iter().map(|c| c[i]).collect())
        .collect();
    let (responsibilities, objective) = if let Some(y) = &observed {
        let (r, l) = posterior(&o, &columns, &weights, noise, y).map_err(|e| invalid(&e))?;
        (json!(r), json!(l))
    } else {
        (Value::Null, Value::Null)
    };
    o.weights = Some(weights.clone());
    o.noise_scale = if gaussian { Some(noise) } else { None };
    let parameters = flatten(&components, &weights, noise, gaussian);
    let objective_name = if gaussian {
        "conditional_gaussian_mixture_log_likelihood"
    } else {
        "adoption_log_score_not_observation_likelihood"
    };
    let optimizer = json!({"converged":if op==KernelOperation::FitModel{json!(converged)}else{Value::Null},"termination_reason":termination,"iterations":iterations,"max_iterations":o.max_iter,"objective_name":objective_name,"objective_history":history,"noise_floor":o.noise_floor,"noise_floor_active":gaussian && noise==o.noise_floor,"global_optimality":"not_established"});
    let predictions = json!({"values":values,"component_rows":rows,"shape":[time.len()],"dtype":"float64","metadata":{"mode":o.mode,"weights":weights,"simulation_semantics":"deterministic weighted point path; stochastic draws unavailable","uncertainty":"not calibrated","class_identifiability":"not established"}});
    let state = json!({"model_key":"mixture","parameters":parameters,"constructor_kwargs":o,"predict_kwargs":{}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut out = json!({"model_key":"mixture","model_name":"MixtureModel","parameters":parameters,"state":state,"predictions":predictions,"responsibilities":responsibilities,"objective":objective,"optimizer":optimizer,"component_initialization":initialization,"component_updates":updates,"details":{"components":o.components,"weights":weights,"mode":o.mode,"noise_scale":o.noise_scale}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some(y) = observed {
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &values,
                "MixtureModel",
                parameters.len(),
            );
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"mixture identifiability and effective degrees of freedom are not qualified"});
            d["warnings"].as_array_mut().unwrap().push(json!("Residual metrics describe the weighted mean curve, not the latent-class observation likelihood. EM local convergence does not establish class identifiability, calibrated uncertainty or global optimality."));
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
        metadata: json!({"runtime":"rust_native","family":"mixture"}),
    })
}
