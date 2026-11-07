use crate::contracts::*;
use serde_json::{Map, Value};
pub(crate) fn object_section<'a>(
    payload: &'a Map<String, Value>,
    key: &str,
    operation: KernelOperation,
) -> Result<&'a Map<String, Value>, KernelBindingError> {
    payload.get(key).and_then(Value::as_object).ok_or_else(|| {
        KernelBindingError::invalid_request(operation, format!("{key} section must be an object"))
    })
}

pub(crate) fn optional_object_section<'a>(
    payload: &'a Map<String, Value>,
    key: &str,
) -> Option<&'a Map<String, Value>> {
    payload.get(key).and_then(Value::as_object)
}

pub(crate) fn ensure_no_covariates_or_event_splits(
    operation: KernelOperation,
    constructor_kwargs: Option<&Map<String, Value>>,
    inputs: &Map<String, Value>,
    message: &'static str,
) -> Result<(), KernelBindingError> {
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    if has_covariates || has_event || input_covariates {
        return Err(KernelBindingError::unsupported_native_operation(
            operation, message,
        ));
    }

    Ok(())
}

pub(crate) fn numeric_array_from_aliases(
    values: &Map<String, Value>,
    aliases: &[&str],
    operation: KernelOperation,
) -> Result<Vec<f64>, KernelBindingError> {
    for alias in aliases {
        if let Some(value) = values.get(*alias) {
            return numeric_array(value, alias, operation);
        }
    }
    Err(KernelBindingError::invalid_request(
        operation,
        "kernel requests require time points in the inputs section",
    ))
}

pub(crate) fn optional_numeric_array_from_aliases(
    values: &Map<String, Value>,
    aliases: &[&str],
    operation: KernelOperation,
) -> Result<Option<Vec<f64>>, KernelBindingError> {
    for alias in aliases {
        if let Some(value) = values.get(*alias) {
            return numeric_array(value, alias, operation).map(Some);
        }
    }
    Ok(None)
}

pub(crate) fn numeric_array(
    value: &Value,
    name: &str,
    operation: KernelOperation,
) -> Result<Vec<f64>, KernelBindingError> {
    let array = value.as_array().ok_or_else(|| {
        KernelBindingError::invalid_request(operation, format!("{name} must be an array"))
    })?;
    if array.is_empty() {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("{name} must not be empty"),
        ));
    }
    array
        .iter()
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                KernelBindingError::invalid_request(
                    operation,
                    format!("{name} must contain only numbers"),
                )
            })
        })
        .collect()
}

pub(crate) fn required_f64(
    values: &Map<String, Value>,
    key: &str,
    operation: KernelOperation,
) -> Result<f64, KernelBindingError> {
    values.get(key).and_then(Value::as_f64).ok_or_else(|| {
        KernelBindingError::invalid_request(operation, format!("missing numeric parameter '{key}'"))
    })
}

pub(crate) fn _fit_constructor_kwargs(payload: &Map<String, Value>) -> Option<&Map<String, Value>> {
    payload
        .get("model_kwargs")
        .and_then(Value::as_object)
        .or_else(|| payload.get("constructor_kwargs").and_then(Value::as_object))
}

/// Validate the scientific input domain consistently for every language binding.
pub(crate) fn validate_request(request: &KernelRequest) -> Result<(), KernelBindingError> {
    let operation = request.operation;
    let fail = |message| KernelBindingError::invalid_request(operation, message);
    if operation == KernelOperation::DiscoverModels {
        return Ok(());
    }
    let model = request.model_key.as_deref().unwrap_or("");
    if ![
        "bass",
        "logistic",
        "gompertz",
        "fisher_pry",
        "norton_bass",
        "lotka_volterra",
        "complementary_goods",
        "multi_product",
    ]
    .contains(&model)
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            format!("native model '{model}' is unavailable pending migration"),
        ));
    }
    let payload = request
        .payload
        .as_object()
        .ok_or_else(|| fail("payload must be an object"))?;
    let inputs = object_section(payload, "inputs", operation)?;
    if [
        "norton_bass",
        "lotka_volterra",
        "multi_product",
        "complementary_goods",
    ]
    .contains(&model)
    {
        return Ok(());
    } // Its module validates scalar/matrix observations and generation-specific parameters.
    if payload
        .get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != model)
    {
        return Err(fail("request and state model_key differ"));
    }
    for value in [
        payload.get("constructor_kwargs"),
        payload.get("model_kwargs"),
        payload
            .get("state")
            .and_then(|s| s.get("constructor_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        let options = value
            .as_object()
            .ok_or_else(|| fail("constructor options must be an object"))?;
        if options.iter().any(|(k, v)| {
            !((k == "covariates" && v.as_array().is_some_and(Vec::is_empty))
                || (k == "t_event" && v.is_null()))
        }) {
            return Err(KernelBindingError::unsupported_native_operation(
                operation,
                "scalar model constructor options remain unavailable",
            ));
        }
    }
    for value in [
        payload.get("fitter_options"),
        payload.get("predict_kwargs"),
        payload.get("state").and_then(|s| s.get("predict_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        if !value.as_object().is_some_and(Map::is_empty) {
            return Err(KernelBindingError::unsupported_native_operation(
                operation,
                "custom prediction/fitter options remain unavailable",
            ));
        }
    }
    if inputs.contains_key("covariates") {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "scalar covariates remain unavailable",
        ));
    }
    if let Some(value) = inputs.get("time").or_else(|| inputs.get("t")) {
        let time = numeric_array(value, "time", operation)?;
        if time.iter().any(|v| !v.is_finite()) || time.windows(2).any(|p| p[1] <= p[0]) {
            return Err(fail("time must be finite and strictly increasing"));
        }
        if let Some(y) = inputs.get("observed").or_else(|| inputs.get("y")) {
            let y = numeric_array(y, "observed", operation)?;
            if y.len() != time.len() || y.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(fail(
                    "observations must be finite, nonnegative and aligned with time",
                ));
            }
            if operation == KernelOperation::FitModel && time.len() < 4 {
                return Err(fail("fitting requires at least four observations"));
            }
        }
    }
    if let Some(parameters) = payload
        .get("parameters")
        .or_else(|| payload.get("state").and_then(|s| s.get("parameters")))
    {
        let parameters = parameters
            .as_object()
            .ok_or_else(|| fail("parameters must be an object"))?;
        let positive: &[&str] = match model {
            "logistic" => &["L", "k"],
            "gompertz" => &["a", "b", "c"],
            "fisher_pry" => &["alpha"],
            "bass" => &["p", "m"],
            "norton_bass" => &["p1", "m1"],
            _ => &[],
        };
        for (key, value) in parameters {
            let value = value
                .as_f64()
                .ok_or_else(|| fail("parameters must be numeric"))?;
            if !value.is_finite()
                || (positive.contains(&key.as_str())
                    && value <= 0.0
                    && !(key == "p"
                        && value == 0.0
                        && operation == KernelOperation::SimulateModel
                        && payload.contains_key("simulation_options")))
                || (["q", "q1"].contains(&key.as_str()) && value < 0.0)
            {
                return Err(fail("parameters are outside the model's finite domain"));
            }
        }
    }
    Ok(())
}
