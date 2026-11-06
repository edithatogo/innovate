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

pub(crate) fn options(
    request: &KernelRequest,
) -> Result<(Option<Value>, usize), KernelBindingError> {
    let Some(options) = request.payload.get("fit_options") else {
        return Ok((None, 300));
    };
    let options = options
        .as_object()
        .ok_or_else(|| error("fit_options must be an object"))?;
    if options
        .keys()
        .any(|k| !["initial_parameters", "max_iterations"].contains(&k.as_str()))
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "unsupported native fit option",
        ));
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
    if observed.len() != time.len()
        || time.len() < 4
        || observed.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(error(
            "fitting needs at least four aligned, finite, nonnegative observations",
        ));
    }
    let y_scale = observed.iter().copied().fold(0.0, f64::max).max(1e-12);
    let t_scale = time.iter().copied().map(f64::abs).fold(1.0, f64::max);
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
    let evaluate = |theta: &[f64]| -> Result<Vec<f64>, String> {
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
                    .map(|x| x / y_scale)
                    .ok_or_else(|| "native prediction is nonfinite".into())
            })
            .collect()
    };
    let target: Vec<_> = observed.iter().map(|x| x / y_scale).collect();
    let starts = vec![
        initial_theta.clone(),
        initial_theta.iter().map(|x| x + 0.1).collect(),
        initial_theta.iter().map(|x| x - 0.1).collect(),
    ];
    let fit = least_squares(&target, &starts, evaluate, max_iterations).map_err(error)?;
    let predictions: Vec<_> = evaluate(&fit.parameters)
        .map_err(error)?
        .iter()
        .map(|x| x * y_scale)
        .collect();
    let mut diagnostics = crate::kernel::fit_diagnostics_count(
        &time,
        &observed,
        &predictions,
        result["model_name"].as_str().unwrap(),
        names.len() + 1,
    );
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
