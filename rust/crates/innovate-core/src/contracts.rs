use crate::KERNEL_SCHEMA_VERSION;
use crate::validation::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Typed operations supported by the core request envelope.
///
/// Additional JSON-only operations are exposed by `execute_json`; this enum is
/// the bounded typed interface, not a claim that every model supports every operation.
pub enum KernelOperation {
    /// Read model identities and declared native capabilities.
    DiscoverModels,
    /// Estimate model parameters from the supplied observations.
    FitModel,
    /// Evaluate a model on supplied times using parameters or fitted state.
    PredictModel,
    /// Run the selected model's supported simulation semantics.
    SimulateModel,
    /// Describe fitted state using the model's supported summary fields.
    SummarizeModel,
    /// Compare fitted predictions with observations and report diagnostics.
    DiagnoseModel,
}

impl KernelOperation {
    /// Return the stable snake-case operation name used in JSON requests.
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
/// A typed native request, capability or response-decoding failure.
///
/// Inspect `code` to classify the error. `message` provides context and is not
/// a stable machine-readable interface; an unsupported operation has no fallback.
pub struct KernelBindingError {
    /// Machine-readable failure category, such as `invalid_request`.
    pub code: String,
    /// Human-readable explanation of the failed request or response.
    pub message: String,
    /// Operation associated with the failure, when known.
    pub operation: Option<KernelOperation>,
}

impl KernelBindingError {
    /// Construct an `unimplemented` error with the supplied static explanation.
    ///
    /// This constructor leaves `operation` unset and performs no execution.
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
/// Serializable error carried inside a native response envelope.
pub struct KernelError {
    /// Machine-readable error category supplied by the native operation.
    pub code: String,
    /// Human-readable context; consumers should branch on `code` instead.
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Failing operation, omitted when no typed operation could be identified.
    pub operation: Option<KernelOperation>,
    #[serde(default)]
    /// Operation-specific structured details; absent input deserializes as null.
    pub details: Value,
    #[serde(default)]
    /// Whether the producer marks the failure as potentially retryable.
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Versioned input envelope for a typed native operation.
///
/// Construction and deserialization do not establish scientific validity.
/// [`crate::KernelBinding::invoke`] validates the schema, model and payload.
pub struct KernelRequest {
    /// Wire schema version; current constructors use [`KERNEL_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Requested typed operation.
    pub operation: KernelOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Model identity; required for operations other than discovery.
    pub model_key: Option<String>,
    #[serde(default)]
    /// Operation-specific inputs, state and options as structured JSON.
    pub payload: Value,
    #[serde(default)]
    /// Caller-supplied contextual metadata; absence deserializes as null.
    pub metadata: Value,
}

impl KernelRequest {
    /// Wrap an operation, optional model, payload and metadata at the current schema.
    ///
    /// This method preserves the supplied values without validating them. Validation
    /// occurs during native invocation, where failures become structured errors.
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

    /// Construct a discovery request with empty payload and metadata objects.
    pub fn discover_models() -> Self {
        Self::new(
            KernelOperation::DiscoverModels,
            None,
            Value::Object(Map::new()),
            Value::Object(Map::new()),
        )
    }

    /// Wrap fitting inputs/options for `model_key`, with empty request metadata.
    ///
    /// The payload is validated by execution rather than by this constructor.
    pub fn fit_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::FitModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    /// Wrap prediction times and parameters/state for `model_key` without execution.
    pub fn predict_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::PredictModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    /// Wrap a model-specific simulation payload without asserting capability support.
    pub fn simulate_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::SimulateModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    /// Wrap a fitted-state summary request, preserving the supplied payload.
    pub fn summarize_model(model_key: impl Into<String>, payload: Value) -> Self {
        Self::new(
            KernelOperation::SummarizeModel,
            Some(model_key.into()),
            payload,
            Value::Object(Map::new()),
        )
    }

    /// Wrap a diagnostic request containing observations and parameters or fitted state.
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
/// Result envelope returned by a typed native operation.
///
/// Inspect `error` before using `result`. The data type permits either optional
/// field to be absent; deserializing an envelope alone does not prove success.
pub struct KernelResponse {
    /// Schema version used by the response producer.
    pub schema_version: String,
    /// Typed operation associated with this response.
    pub operation: KernelOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Model identity, when the operation addresses a model.
    pub model_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Operation-specific result, absent when no result was returned.
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Structured operation failure, when provided by the producer.
    pub error: Option<KernelError>,
    #[serde(default)]
    /// Response provenance and runtime metadata supplied by the producer.
    pub metadata: Value,
}

impl KernelResponse {
    /// Borrow the result object, returning `None` for an absent or non-object result.
    pub fn result_object(&self) -> Option<&Map<String, Value>> {
        self.result.as_ref()?.as_object()
    }

    /// Borrow the `diagnostics` result member, without validating its structure.
    pub fn diagnostics(&self) -> Option<&Value> {
        self.result_object()?.get("diagnostics")
    }

    /// Extract diagnostic labels/counts when the required label fields are strings.
    ///
    /// Missing warning arrays or metric objects contribute zero counts. This
    /// convenience projection does not establish diagnostic or scientific validity.
    pub fn diagnostics_summary(&self) -> Option<KernelDiagnosticsSummary> {
        KernelDiagnosticsSummary::from_response(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Declared capabilities for one model in the embedded discovery manifest.
///
/// These fields describe available routes and constraints. They are not evidence
/// that every parameterization has been independently validated.
pub struct KernelDiscoveryRecord {
    /// Stable model identity used in native requests.
    pub key: String,
    /// Model-family label used to organize discovery results.
    pub family: String,
    /// Manifest-provided import/reference path; not an instruction to import Python.
    pub import_path: String,
    /// Declared maturity or stability tier for this model surface.
    pub stability: String,
    /// Whether the manifest declares covariate support.
    pub supports_covariates: bool,
    #[serde(default)]
    /// Whether the manifest declares parameter-event support.
    pub supports_parameter_events: bool,
    /// Whether model output may have more than one value per time point.
    pub supports_multivariate_output: bool,
    #[serde(default)]
    /// Whether the manifest declares a native rate output.
    pub supports_rates: bool,
    #[serde(default)]
    /// Named model variants declared by the manifest.
    pub variants: Vec<String>,
    /// Backend identities declared for this model.
    pub supported_backends: Vec<String>,
    #[serde(default)]
    /// Optional dependency names reported by the manifest.
    pub optional_dependencies: Vec<String>,
    #[serde(default)]
    /// Whether a simulation operation is declared for this model.
    pub supports_simulation: bool,
    #[serde(default)]
    /// Whether a fitted-state summary operation is declared.
    pub supports_summarize: bool,
    /// Supported operation names reported by discovery.
    pub supported_operations: Vec<String>,
    /// Text describing the model's simulation interpretation or limitations.
    pub simulation_semantics: String,
    #[serde(default)]
    /// Supported simulation engine names, where applicable.
    pub simulation_engines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Decoded discovery result containing model records and manifest metadata.
pub struct KernelDiscoveryResponse {
    /// Schema version attached to the discovery result.
    pub schema_version: String,
    /// Models and their declared native capability boundaries.
    pub models: Vec<KernelDiscoveryRecord>,
    #[serde(default)]
    /// Additional manifest metadata; missing values deserialize as null.
    pub metadata: Value,
}

impl KernelDiscoveryResponse {
    /// Decode a generic response's result as a discovery payload.
    ///
    /// # Errors
    /// Returns `invalid_response` if the result is absent or does not deserialize
    /// to this structure. The outer operation and error fields are not inspected.
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
/// Lightweight projection of diagnostic metadata and collection sizes.
pub struct KernelDiagnosticsSummary {
    /// Producer-declared support level for the diagnostic output.
    pub support_level: String,
    /// Producer-declared provenance label.
    pub provenance: String,
    /// Label identifying which model results may be compared.
    pub comparison_family: String,
    /// Number of entries in the warnings array, or zero when absent/malformed.
    pub warning_count: usize,
    /// Number of named metrics, or zero when the metrics object is absent/malformed.
    pub metric_count: usize,
    /// Optional human-readable model name from the diagnostic object.
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
/// Portable fitted-state fields consumed by native prediction and diagnostics.
///
/// The structure retains JSON parameter/options values. Model-specific numerical
/// validity is checked by execution, not by deserializing this structure.
pub struct StableModelState {
    /// Model identity that must agree with the enclosing request.
    pub model_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional display name retained with the fitted state.
    pub model_name: Option<String>,
    #[serde(default)]
    /// Model construction options such as declared covariate names.
    pub constructor_kwargs: Value,
    #[serde(default)]
    /// Fitted or explicitly supplied model parameters in model-specific units.
    pub parameters: Value,
    #[serde(default)]
    /// Prediction options retained with the state.
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
/// Validated scalar observations for the bounded simple fitting route.
///
/// Observations may be zero despite the historical type name. This route excludes
/// covariates, event splits and nonempty custom fitter options.
pub struct SimplePositiveObservationsFitPayload {
    /// Shape discriminator set to `simple_positive_observations_fit` by the parser.
    pub payload_shape: String,
    /// Requested model identity retained by the parser.
    pub model_key: String,
    /// Finite input times in the caller's consistent time unit.
    pub time: Vec<f64>,
    /// Aligned finite, nonnegative observations in the model's outcome unit.
    pub observed: Vec<f64>,
    #[serde(default)]
    /// Permitted construction options retained as an object.
    pub constructor_kwargs: Value,
}

impl SimplePositiveObservationsFitPayload {
    /// Parse the bounded scalar fitting payload for `model_key` and `operation`.
    ///
    /// Accepts `time`/`t` and supported observation aliases in `inputs`; time and
    /// observations must be aligned and finite, with nonnegative observations.
    ///
    /// # Errors
    /// Returns `invalid_request` for malformed sections or invalid values, and
    /// `unsupported_native_operation` for unsupported controls or fitter options.
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
/// Validated times, optional observations and state for the bounded native route.
pub struct BoundedFittedStatePayload {
    /// Shape discriminator set to `bounded_fitted_state` by the parser.
    pub payload_shape: String,
    /// Model identity required to match the decoded state's model identity.
    pub model_key: String,
    /// Finite evaluation times in the model's consistent time unit.
    pub time: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional finite observations aligned with the time vector.
    pub observed: Option<Vec<f64>>,
    /// Decoded model parameters and construction/prediction options.
    pub state: StableModelState,
}

impl BoundedFittedStatePayload {
    /// Decode `inputs` and `state` for a bounded native operation.
    ///
    /// This validates alignment, finiteness and model identity. It does not infer
    /// units or require a sorted time grid; individual model routes add constraints.
    ///
    /// # Errors
    /// Returns a structured request error for malformed data/model mismatch or an
    /// unsupported-operation error for covariates and event splits on this route.
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
/// Bounded fitted-state payload tagged for deterministic simulation.
///
/// This wrapper represents the deterministic route, not a stochastic population
/// simulation or proof that a requested runtime engine is available.
pub struct DeterministicSimulationPayload {
    /// Shape discriminator set to `deterministic_simulation` by the parser.
    pub payload_shape: String,
    /// Validated times, optional observations and model state.
    pub fitted_state: BoundedFittedStatePayload,
}

impl DeterministicSimulationPayload {
    /// Decode a bounded state using the `simulate_model` operation context.
    ///
    /// # Errors
    /// Propagates the validation and unsupported-capability errors from
    /// [`BoundedFittedStatePayload::from_value`].
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
/// Bounded diagnostic input with observed values available for comparison.
pub struct DeterministicDiagnosticsPayload {
    /// Shape discriminator set to `deterministic_diagnostics` by the parser.
    pub payload_shape: String,
    /// Validated state whose `observed` field is present after parsing.
    pub fitted_state: BoundedFittedStatePayload,
}

impl DeterministicDiagnosticsPayload {
    /// Decode bounded fitted state and require observations for diagnostics.
    ///
    /// # Errors
    /// Propagates fitted-state validation errors and returns `invalid_request`
    /// when the input does not contain observations.
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
