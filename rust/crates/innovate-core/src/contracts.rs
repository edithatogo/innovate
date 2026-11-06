use crate::KERNEL_SCHEMA_VERSION;
use crate::validation::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelOperation {
    DiscoverModels,
    FitModel,
    PredictModel,
    SimulateModel,
    SummarizeModel,
    DiagnoseModel,
}

impl KernelOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelOperation::DiscoverModels => "discover_models",
            KernelOperation::FitModel => "fit_model",
            KernelOperation::PredictModel => "predict_model",
            KernelOperation::SimulateModel => "simulate_model",
            KernelOperation::SummarizeModel => "summarize_model",
            KernelOperation::DiagnoseModel => "diagnose_model",
        }
    }
}

impl fmt::Display for KernelOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelBindingError {
    pub code: String,
    pub message: String,
    pub operation: Option<KernelOperation>,
}

impl KernelBindingError {
    pub fn unimplemented(message: &'static str) -> Self {
        Self {
            code: "unimplemented".to_string(),
            message: message.to_string(),
            operation: None,
        }
    }

    pub(crate) fn invalid_response(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_response".to_string(),
            message: message.into(),
            operation: None,
        }
    }

    pub(crate) fn invalid_request(operation: KernelOperation, message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request".to_string(),
            message: message.into(),
            operation: Some(operation),
        }
    }

    pub(crate) fn unsupported_native_operation(
        operation: KernelOperation,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: "unsupported_native_operation".to_string(),
            message: message.into(),
            operation: Some(operation),
        }
    }
}

impl fmt::Display for KernelBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(operation) = self.operation {
            write!(f, "{} for {}: {}", self.code, operation, self.message)
        } else {
            write!(f, "{}: {}", self.code, self.message)
        }
    }
}

impl std::error::Error for KernelBindingError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelError {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<KernelOperation>,
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelRequest {
    pub schema_version: String,
    pub operation: KernelOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub metadata: Value,
}

impl KernelRequest {
    pub fn new(
        operation: KernelOperation,
        model_key: Option<String>,
        payload: Value,
        metadata: Value,
    ) -> Self {
        // Execution validates requests uniformly and returns structured errors.

        Self {
            schema_version: KERNEL_SCHEMA_VERSION.to_string(),
            operation,
            model_key,
            payload,
            metadata,
        }
    }

    pub fn discover_models() -> Self {
        Self::new(
            KernelOperation::DiscoverModels,
            None,
            Value::Object(Map::new()),
            Value::Object(Map::new()),
        )
    }

    pub fn fit_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::FitModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    pub fn predict_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::PredictModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    pub fn simulate_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::SimulateModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    pub fn summarize_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::SummarizeModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    pub fn diagnose_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::DiagnoseModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelResponse {
    pub schema_version: String,
    pub operation: KernelOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<KernelError>,
    #[serde(default)]
    pub metadata: Value,
}

impl KernelResponse {
    pub fn result_object(&self) -> Option<&Map<String, Value>> {
        self.result.as_ref()?.as_object()
    }

    pub fn diagnostics(&self) -> Option<&Value> {
        self.result_object()?.get("diagnostics")
    }

    pub fn diagnostics_summary(&self) -> Option<KernelDiagnosticsSummary> {
        KernelDiagnosticsSummary::from_response(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDiscoveryRecord {
    pub key: String,
    pub family: String,
    pub import_path: String,
    pub stability: String,
    pub supports_covariates: bool,
    pub supports_multivariate_output: bool,
    pub supported_backends: Vec<String>,
    #[serde(default)]
    pub optional_dependencies: Vec<String>,
    #[serde(default)]
    pub supports_simulation: bool,
    #[serde(default)]
    pub supports_summarize: bool,
    pub supported_operations: Vec<String>,
    pub simulation_semantics: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDiscoveryResponse {
    pub schema_version: String,
    pub models: Vec<KernelDiscoveryRecord>,
    #[serde(default)]
    pub metadata: Value,
}

impl KernelDiscoveryResponse {
    pub fn from_response(response: KernelResponse) -> Result<Self, KernelBindingError> {
        let result = response.result.ok_or_else(|| {
            KernelBindingError::invalid_response("discover_models did not return a result")
        })?;
        serde_json::from_value(result).map_err(|err| {
            KernelBindingError::invalid_response(format!(
                "failed to decode discovery response: {err}"
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDiagnosticsSummary {
    pub support_level: String,
    pub provenance: String,
    pub comparison_family: String,
    pub warning_count: usize,
    pub metric_count: usize,
    pub model_name: Option<String>,
}

impl KernelDiagnosticsSummary {
    fn from_response(response: &KernelResponse) -> Option<Self> {
        let diagnostics = response.diagnostics()?;
        let diagnostics = diagnostics.as_object()?;
        let warnings = diagnostics
            .get("warnings")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let metrics = diagnostics
            .get("metrics")
            .and_then(Value::as_object)
            .map_or(0, Map::len);

        Some(Self {
            support_level: diagnostics.get("support_level")?.as_str()?.to_string(),
            provenance: diagnostics.get("provenance")?.as_str()?.to_string(),
            comparison_family: diagnostics.get("comparison_family")?.as_str()?.to_string(),
            warning_count: warnings,
            metric_count: metrics,
            model_name: diagnostics
                .get("model_name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StableModelState {
    pub model_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default)]
    pub constructor_kwargs: Value,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default)]
    pub predict_kwargs: Value,
}

impl StableModelState {
    fn from_object(
        model_key: &str,
        operation: KernelOperation,
        state: &Map<String, Value>,
    ) -> Result<Self, KernelBindingError> {
        let state_model_key = state
            .get("model_key")
            .and_then(Value::as_str)
            .unwrap_or(model_key);
        if state_model_key != model_key {
            return Err(KernelBindingError::invalid_request(
                operation,
                format!(
                    "kernel request model_key '{model_key}' does not match state model_key '{state_model_key}'"
                ),
            ));
        }

        Ok(Self {
            model_key: state_model_key.to_string(),
            model_name: state
                .get("model_name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            constructor_kwargs: state
                .get("constructor_kwargs")
                .cloned()
                .unwrap_or_else(|| json!({})),
            parameters: state
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({})),
            predict_kwargs: state
                .get("predict_kwargs")
                .cloned()
                .unwrap_or_else(|| json!({})),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimplePositiveObservationsFitPayload {
    pub payload_shape: String,
    pub model_key: String,
    pub time: Vec<f64>,
    pub observed: Vec<f64>,
    #[serde(default)]
    pub constructor_kwargs: Value,
}

impl SimplePositiveObservationsFitPayload {
    pub fn from_value(
        model_key: &str,
        operation: KernelOperation,
        payload: &Value,
    ) -> Result<Self, KernelBindingError> {
        let payload = payload.as_object().ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                format!("{operation} payload must be an object"),
            )
        })?;
        let inputs = object_section(payload, "inputs", operation)?;
        let time = numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
        let observed = numeric_array_from_aliases(
            inputs,
            &["observed", "y", "values", "adoption", "share"],
            operation,
        )?;
        if time.len() != observed.len() {
            return Err(KernelBindingError::invalid_request(
                operation,
                "time and observed arrays must have the same length",
            ));
        }
        if time.iter().any(|value| !value.is_finite())
            || observed
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(KernelBindingError::invalid_request(
                operation,
                "time and observed arrays must contain finite numeric values and observed values must be non-negative",
            ));
        }
        let constructor_kwargs = _fit_constructor_kwargs(payload);
        ensure_no_covariates_or_event_splits(
            operation,
            constructor_kwargs,
            inputs,
            "native fitting currently supports simple fitted states without covariates, events, or custom fitter options",
        )?;
        let fit_options = payload.get("fit_options").and_then(Value::as_object);
        let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
        if fit_options.is_some_and(|values| !values.is_empty())
            || fitter_options.is_some_and(|values| !values.is_empty())
        {
            return Err(KernelBindingError::unsupported_native_operation(
                operation,
                "native fitting currently supports simple fitted states without covariates, events, or custom fitter options",
            ));
        }

        Ok(Self {
            payload_shape: "simple_positive_observations_fit".to_string(),
            model_key: model_key.to_string(),
            time,
            observed,
            constructor_kwargs: constructor_kwargs
                .map(|values| Value::Object(values.clone()))
                .unwrap_or_else(|| json!({})),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedFittedStatePayload {
    pub payload_shape: String,
    pub model_key: String,
    pub time: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<Vec<f64>>,
    pub state: StableModelState,
}

impl BoundedFittedStatePayload {
    pub fn from_value(
        model_key: &str,
        operation: KernelOperation,
        payload: &Value,
    ) -> Result<Self, KernelBindingError> {
        let payload = payload.as_object().ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                format!("{operation} payload must be an object"),
            )
        })?;
        let inputs = object_section(payload, "inputs", operation)?;
        let time = numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
        if time.iter().any(|value| !value.is_finite()) {
            return Err(KernelBindingError::invalid_request(
                operation,
                "time values must be finite for native fitted-state execution",
            ));
        }
        let observed = optional_numeric_array_from_aliases(
            inputs,
            &["observed", "y", "values", "adoption", "share"],
            operation,
        )?;
        if observed
            .as_ref()
            .is_some_and(|values| values.len() != time.len())
        {
            return Err(KernelBindingError::invalid_request(
                operation,
                "time and observed arrays must have the same length",
            ));
        }
        if observed
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(KernelBindingError::invalid_request(
                operation,
                "observed arrays must contain finite numeric values",
            ));
        }

        let state = object_section(payload, "state", operation)?;
        let decoded_state = StableModelState::from_object(model_key, operation, state)?;
        let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
        ensure_no_covariates_or_event_splits(
            operation,
            constructor_kwargs,
            inputs,
            "native fitted-state execution currently supports fitted states without covariates or event splits",
        )?;

        Ok(Self {
            payload_shape: "bounded_fitted_state".to_string(),
            model_key: model_key.to_string(),
            time,
            observed,
            state: decoded_state,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicSimulationPayload {
    pub payload_shape: String,
    pub fitted_state: BoundedFittedStatePayload,
}

impl DeterministicSimulationPayload {
    pub fn from_value(model_key: &str, payload: &Value) -> Result<Self, KernelBindingError> {
        Ok(Self {
            payload_shape: "deterministic_simulation".to_string(),
            fitted_state: BoundedFittedStatePayload::from_value(
                model_key,
                KernelOperation::SimulateModel,
                payload,
            )?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicDiagnosticsPayload {
    pub payload_shape: String,
    pub fitted_state: BoundedFittedStatePayload,
}

impl DeterministicDiagnosticsPayload {
    pub fn from_value(model_key: &str, payload: &Value) -> Result<Self, KernelBindingError> {
        let fitted_state = BoundedFittedStatePayload::from_value(
            model_key,
            KernelOperation::DiagnoseModel,
            payload,
        )?;
        if fitted_state.observed.is_none() {
            return Err(KernelBindingError::invalid_request(
                KernelOperation::DiagnoseModel,
                "diagnose_model requires time and observed arrays in the inputs section",
            ));
        }
        Ok(Self {
            payload_shape: "deterministic_diagnostics".to_string(),
            fitted_state,
        })
    }
}
