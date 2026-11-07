//! Shared scalar fit refinement and numerical-quality reporting.
use crate::numerics::least_squares;
use crate::validation::{numeric_array_from_aliases, object_section};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBinding, KernelBindingError, KernelOperation, KernelRequest,
    KernelResponse,
};
use serde_json::{Value, json};

fn error(message: impl Into<String>) -> KernelBindingError {
    KernelBindingError::invalid_request(KernelOperation::FitModel, message)
}

pub(crate) fn weights(
    request: &KernelRequest,
    n: usize,
    minimum: usize,
) -> Result<Option<Vec<f64>>, KernelBindingError> {
    let Some(value) = request
        .payload
        .get("fit_options")
        .and_then(|v| v.get("weights"))
    else {
        return Ok(None);
    };
    let values: Vec<f64> = serde_json::from_value(value.clone())
        .map_err(|_| error("weights must be a numeric array"))?;
    if values.len() != n
        || values.iter().any(|v| !v.is_finite() || *v < 0.)
        || values.iter().filter(|v| **v > 0.).count() < minimum
    {
        return Err(error(
            "weights must be finite nonnegative, aligned, and retain at least as many positive observations as curve parameters",
        ));
    }
    Ok(Some(values))
}

fn scalar_minimum(request: &KernelRequest) -> usize {
    if request.model_key.as_deref() == Some("fisher_pry") {
        2
    } else {
        3
    }
}

pub(crate) fn options(
    request: &KernelRequest,
) -> Result<(Option<Value>, usize), KernelBindingError> {
    parse_options(request, false)
}

pub(crate) fn scalar_options(
    request: &KernelRequest,
) -> Result<(Option<Value>, usize), KernelBindingError> {
    parse_options(request, true)
}

fn parse_options(
    request: &KernelRequest,
    allow_weights: bool,
) -> Result<(Option<Value>, usize), KernelBindingError> {
    let Some(options) = request.payload.get("fit_options") else {
        return Ok((None, 300));
    };
    let options = options
        .as_object()
        .ok_or_else(|| error("fit_options must be an object"))?;
    if options.keys().any(|k| {
        !["initial_parameters", "max_iterations"].contains(&k.as_str())
            && !(allow_weights && k == "weights")
    }) {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "unsupported native fit option",
        ));
    }
    if options.contains_key("weights") {
        let input = object_section(
            request.payload.as_object().unwrap(),
            "inputs",
            KernelOperation::FitModel,
        )?;
        let time = numeric_array_from_aliases(input, &["time", "t"], KernelOperation::FitModel)?;
        weights(request, time.len(), scalar_minimum(request))?;
    }
    let limit = options
        .get("max_iterations")
        .map(|v| {
            v.as_u64()
                .filter(|i| *i > 0 && *i <= 2000)
                .ok_or_else(|| error("max_iterations must be in 1..=2000"))
        })
        .transpose()?
        .unwrap_or(300) as usize;
    Ok((options.get("initial_parameters").cloned(), limit))
}

pub(crate) fn seed_response(
    request: &KernelRequest,
    parameters: Value,
) -> Result<KernelResponse, KernelBindingError> {
    let model = request.model_key.as_deref().unwrap_or("");
    let (name, family) = match model {
        "bass" => ("BassModel", "diffusion"),
        "logistic" => ("LogisticModel", "diffusion"),
        "gompertz" => ("GompertzModel", "diffusion"),
        "fisher_pry" => ("FisherPryModel", "substitution"),
        _ => return Err(error("model has no scalar native fitter")),
    };
    let kwargs = request
        .payload
        .get("constructor_kwargs")
        .or_else(|| request.payload.get("model_kwargs"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let state = json!({"model_key":model,"model_name":name,"parameters":parameters,"constructor_kwargs":kwargs,"predict_kwargs":{}});
    // Validate the exact initial state through the same execution path consumers use.
    KernelBinding::new().predict_model(
        model,
        json!({"inputs":request.payload["inputs"],"state":state}),
    )?;
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: KernelOperation::FitModel,
        model_key: Some(model.into()),
        error: None,
        result: Some(
            json!({"model_key":model,"model_name":name,"family":family,"parameters":parameters,"state":state}),
        ),
        metadata: json!({"model_key":model,"model_name":name,"family":family,"runtime":"rust_native"}),
    })
}

pub(crate) fn refine(
    request: &KernelRequest,
    mut response: KernelResponse,
    max_iterations: usize,
) -> Result<KernelResponse, KernelBindingError> {
    let model = request.model_key.as_deref().unwrap_or("");
    let op = KernelOperation::FitModel;
    let payload = request
        .payload
        .as_object()
        .ok_or_else(|| error("payload must be an object"))?;
    let input = object_section(payload, "inputs", op)?;
    let time = numeric_array_from_aliases(input, &["time", "t"], op)?;
    let observed =
        numeric_array_from_aliases(input, &["observed", "y", "values", "adoption", "share"], op)?;
    let minimum = if model == "fisher_pry" { 2 } else { 4 };
    if observed.len() != time.len()
        || time.len() < minimum
        || observed.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(error(format!(
            "fitting needs at least {minimum} aligned, finite, nonnegative observations"
        )));
    }
    let weights = weights(request, time.len(), scalar_minimum(request))?;
    let maximum_weight = weights
        .as_ref()
        .map(|w| w.iter().copied().fold(0., f64::max))
        .unwrap_or(1.);
    let factors: Vec<f64> = weights
        .as_ref()
        .map(|w| w.iter().map(|w| w.sqrt() / maximum_weight.sqrt()).collect())
        .unwrap_or_else(|| vec![1.; time.len()]);
    let positive_count = factors.iter().filter(|w| **w > 0.).count();
    let y_scale = observed
        .iter()
        .zip(&factors)
        .filter(|(_, w)| **w > 0.)
        .map(|(v, _)| *v)
        .fold(0., f64::max)
        .max(1e-12);
    let t_scale = time
        .iter()
        .zip(&factors)
        .filter(|(_, w)| **w > 0.)
        .map(|(v, _)| v.abs())
        .fold(1., f64::max);
    let names: &[&str] = match model {
        "bass" => &["p", "q", "m"],
        "logistic" => &["L", "k", "x0"],
        "gompertz" => &["a", "b", "c"],
        "fisher_pry" => &["alpha", "t0"],
        _ => return Err(error("unknown scalar model")),
    };
    let result = response
        .result
        .as_mut()
        .ok_or_else(|| error("missing fit seed"))?;
    let initial = result["parameters"]
        .as_object()
        .ok_or_else(|| error("initial_parameters must be an object"))?;
    if initial.len() != names.len() {
        return Err(error(
            "initial_parameters must match the model's parameter names",
        ));
    }
    let scale = |name: &str| match name {
        "L" | "m" | "a" => y_scale,
        "x0" | "t0" => t_scale,
        "b" => 1.0,
        _ => 1.0 / t_scale,
    };
    let mut initial_theta = Vec::new();
    for name in names {
        let value = initial
            .get(*name)
            .and_then(Value::as_f64)
            .ok_or_else(|| error(format!("missing initial parameter {name}")))?;
        let scaled = value / scale(name);
        initial_theta.push(if ["x0", "t0"].contains(name) {
            scaled
        } else {
            scaled.max(1e-12).ln()
        });
    }
    let untransform = |theta: &[f64]| {
        let mut parameters = serde_json::Map::new();
        for (name, value) in names.iter().zip(theta) {
            parameters.insert(
                (*name).to_string(),
                json!(
                    scale(name)
                        * if ["x0", "t0"].contains(name) {
                            *value
                        } else {
                            value.exp()
                        }
                ),
            );
        }
        Value::Object(parameters)
    };
    let base_state = result["state"].clone();
    let evaluate_raw = |theta: &[f64]| -> Result<Vec<f64>, String> {
        let mut state = base_state.clone();
        state["parameters"] = untransform(theta);
        let prediction = KernelBinding::new()
            .predict_model(model, json!({"inputs":{"time":time},"state":state}))
            .map_err(|e| e.to_string())?
            .result
            .ok_or("missing native predictions")?;
        prediction["values"]
            .as_array()
            .ok_or("native scalar shape changed")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| "native prediction is nonfinite".into())
            })
            .collect()
    };
    let evaluate = |theta: &[f64]| -> Result<Vec<f64>, String> {
        Ok(evaluate_raw(theta)?
            .iter()
            .zip(&factors)
            .map(|(x, w)| (x * w) / y_scale)
            .collect())
    };
    let target: Vec<_> = observed
        .iter()
        .zip(&factors)
        .map(|(x, w)| (x * w) / y_scale)
        .collect();
    let starts = vec![
        initial_theta.clone(),
        initial_theta.iter().map(|x| x + 0.1).collect(),
        initial_theta.iter().map(|x| x - 0.1).collect(),
    ];
    let fit = least_squares(&target, &starts, evaluate, max_iterations).map_err(error)?;
    let predictions = evaluate_raw(&fit.parameters).map_err(error)?;
    let mut diagnostics = crate::kernel::fit_diagnostics_count(
        &time,
        &observed,
        &predictions,
        result["model_name"].as_str().unwrap(),
        names.len() + 1,
    );
    if weights.is_some() {
        let weighted_rss: f64 = observed
            .iter()
            .zip(&predictions)
            .zip(&factors)
            .map(|((y, p), w)| ((y - p) * w).powi(2))
            .sum();
        diagnostics["weighting"] = json!({"objective":"sum(w_i * residual_i^2), normalized by maximum weight for optimization","maximum_weight":maximum_weight,"positive_observations":positive_count,"weighted_rss_normalized_by_max":weighted_rss,"weighted_rss_representable":weighted_rss.is_finite(),"rank_semantics":"weighted local Jacobian; zero-weight observations excluded"});
        diagnostics["metrics"]["AIC"] = Value::Null;
        diagnostics["metrics"]["BIC"] = Value::Null;
        diagnostics["information_criteria"] = json!({"available":false,"reason":"weight precision/variance likelihood convention not qualified"});
        diagnostics["warnings"].as_array_mut().unwrap().push(json!("descriptive residual metrics are unweighted; optimizer and local rank use supplied weights; calibrated uncertainty unavailable"));
    }
    if positive_count <= names.len() {
        diagnostics["metrics"]["AIC"] = Value::Null;
        diagnostics["metrics"]["BIC"] = Value::Null;
        diagnostics["information_criteria"] =
            json!({"available":false,"reason":"no residual degrees of freedom"});
        diagnostics["warnings"].as_array_mut().unwrap().push(json!("point fit has no residual degrees of freedom; residual variance and forecast uncertainty are unqualified"));
    }
    if fit.report["converged"] != true {
        diagnostics["warnings"]
            .as_array_mut()
            .unwrap()
            .push(json!("optimizer did not converge"));
    }
    if fit.identifiability["status"] != "local_full_rank" {
        diagnostics["warnings"]
            .as_array_mut()
            .unwrap()
            .push(json!("parameters are locally rank deficient"));
    }
    diagnostics["optimizer"] = fit.report;
    diagnostics["optimizer"]["initialization"] = json!(if request.payload["fit_options"]
        .get("initial_parameters")
        .is_some()
    {
        "user_parameters"
    } else {
        "model_specific_native_estimate"
    });
    diagnostics["identifiability"] = fit.identifiability;
    result["parameters"] = untransform(&fit.parameters);
    result["state"]["parameters"] = result["parameters"].clone();
    result["predictions"] = json!({"values":predictions,"shape":[time.len()],"dtype":"float64","metadata":{"shape":[time.len()]}});
    result["diagnostics"] = diagnostics;
    Ok(response)
}
