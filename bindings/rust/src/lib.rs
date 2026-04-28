use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const KERNEL_SCHEMA_VERSION: &str = "1.0";
const DISCOVERY_MANIFEST_JSON: &str = include_str!("../inst/discovery_manifest.json");

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

    fn bridge_command_failed(message: impl Into<String>) -> Self {
        Self {
            code: "bridge_command_failed".to_string(),
            message: message.into(),
            operation: None,
        }
    }

    fn invalid_request(operation: KernelOperation, message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request".to_string(),
            message: message.into(),
            operation: Some(operation),
        }
    }

    fn unsupported_native_operation(
        operation: KernelOperation,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: "unsupported_native_operation".to_string(),
            message: message.into(),
            operation: Some(operation),
        }
    }

    fn from_kernel_error(error: &KernelError) -> Self {
        Self {
            code: error.code.clone(),
            message: error.message.clone(),
            operation: error.operation,
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
        if operation != KernelOperation::DiscoverModels
            && model_key.as_deref().unwrap_or("").is_empty()
        {
            panic!("kernel requests for {operation} require a model_key");
        }

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
            KernelBindingError::bridge_command_failed("discover_models did not return a result")
        })?;
        serde_json::from_value(result).map_err(|err| {
            KernelBindingError::bridge_command_failed(format!(
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

#[derive(Debug, Default, Clone, Copy)]
pub struct KernelBinding;

impl KernelBinding {
    pub fn new() -> Self {
        Self
    }

    pub fn schema_version(&self) -> &'static str {
        KERNEL_SCHEMA_VERSION
    }

    pub fn bridge_script_path(&self) -> PathBuf {
        PathBuf::from("inst")
            .join("python")
            .join("kernel_bridge.py")
    }

    pub fn bridge_script_exists(&self) -> bool {
        self.bridge_script_absolute_path().exists()
    }

    pub fn available_operations(&self) -> [KernelOperation; 6] {
        [
            KernelOperation::DiscoverModels,
            KernelOperation::FitModel,
            KernelOperation::PredictModel,
            KernelOperation::SimulateModel,
            KernelOperation::SummarizeModel,
            KernelOperation::DiagnoseModel,
        ]
    }

    pub fn discover_models_request(&self) -> KernelRequest {
        KernelRequest::discover_models()
    }

    pub fn fit_model_request(&self, model_key: impl Into<String>, payload: Value) -> KernelRequest {
        KernelRequest::fit_model(model_key, payload)
    }

    pub fn predict_model_request(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> KernelRequest {
        KernelRequest::predict_model(model_key, payload)
    }

    pub fn simulate_model_request(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> KernelRequest {
        KernelRequest::simulate_model(model_key, payload)
    }

    pub fn summarize_model_request(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> KernelRequest {
        KernelRequest::summarize_model(model_key, payload)
    }

    pub fn diagnose_model_request(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> KernelRequest {
        KernelRequest::diagnose_model(model_key, payload)
    }

    pub fn discover_models_native(&self) -> KernelDiscoveryResponse {
        serde_json::from_str(DISCOVERY_MANIFEST_JSON)
            .expect("embedded Rust discovery manifest must decode")
    }

    pub fn discover_models_via_bridge(
        &self,
    ) -> Result<KernelDiscoveryResponse, KernelBindingError> {
        let response = self.invoke(&self.discover_models_request())?;
        KernelDiscoveryResponse::from_response(response)
    }

    pub fn discover_models(&self) -> Result<KernelDiscoveryResponse, KernelBindingError> {
        Ok(self.discover_models_native())
    }

    pub fn fit_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.fit_model_request(model_key, payload))
    }

    pub fn predict_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.predict_model_request(model_key, payload);
        self.predict_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                self.invoke(&request)
            } else {
                Err(err)
            }
        })
    }

    pub fn predict_model_via_bridge(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.predict_model_request(model_key, payload))
    }

    pub fn predict_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        logistic_native_response(KernelOperation::PredictModel, request)
    }

    pub fn simulate_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.simulate_model_request(model_key, payload);
        self.simulate_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                self.invoke(&request)
            } else {
                Err(err)
            }
        })
    }

    pub fn simulate_model_via_bridge(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.simulate_model_request(model_key, payload))
    }

    pub fn simulate_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        logistic_native_response(KernelOperation::SimulateModel, request)
    }

    pub fn summarize_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.summarize_model_request(model_key, payload))
    }

    pub fn diagnose_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.diagnose_model_request(model_key, payload))
    }

    pub fn invoke(&self, request: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
        let request_path = unique_temp_path("innovate-rust-kernel-request", "json");
        let response_path = unique_temp_path("innovate-rust-kernel-response", "json");
        let bridge_path = self.bridge_script_absolute_path();
        let command = self.python_command_segments();

        if command.is_empty() {
            return Err(KernelBindingError::bridge_command_failed(
                "INNOVATE_PYTHON_COMMAND must not be empty",
            ));
        }

        let request_json = serde_json::to_string_pretty(request).map_err(|err| {
            KernelBindingError::bridge_command_failed(format!(
                "failed to serialize kernel request: {err}"
            ))
        })?;
        fs::write(&request_path, request_json).map_err(|err| {
            KernelBindingError::bridge_command_failed(format!(
                "failed to write request file '{}': {err}",
                request_path.display()
            ))
        })?;

        let status = Command::new(&command[0])
            .args(&command[1..])
            .arg(&bridge_path)
            .arg(&request_path)
            .arg(&response_path)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("PYTHONPATH", self.kernel_pythonpath())
            .status()
            .map_err(|err| {
                KernelBindingError::bridge_command_failed(format!(
                    "failed to launch kernel bridge: {err}"
                ))
            })?;

        if !status.success() {
            let _ = fs::remove_file(&request_path);
            let _ = fs::remove_file(&response_path);
            return Err(KernelBindingError::bridge_command_failed(format!(
                "kernel bridge exited with status {status}"
            )));
        }

        let response_json = fs::read_to_string(&response_path).map_err(|err| {
            KernelBindingError::bridge_command_failed(format!(
                "failed to read response file '{}': {err}",
                response_path.display()
            ))
        })?;

        let response: KernelResponse = serde_json::from_str(&response_json).map_err(|err| {
            KernelBindingError::bridge_command_failed(format!(
                "failed to decode kernel response: {err}"
            ))
        })?;

        let _ = fs::remove_file(&request_path);
        let _ = fs::remove_file(&response_path);

        if let Some(error) = &response.error {
            return Err(KernelBindingError::from_kernel_error(error));
        }

        Ok(response)
    }

    fn bridge_script_absolute_path(&self) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("inst")
            .join("python")
            .join("kernel_bridge.py")
    }

    fn kernel_pythonpath(&self) -> String {
        let mut paths = vec![Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("src")];
        if let Ok(existing) = env::var("PYTHONPATH") {
            if !existing.is_empty() {
                paths.push(PathBuf::from(existing));
            }
        }
        let joined = env::join_paths(paths).unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("src")
                .into_os_string()
        });
        joined.to_string_lossy().into_owned()
    }

    fn python_command_segments(&self) -> Vec<String> {
        let command =
            env::var("INNOVATE_PYTHON_COMMAND").unwrap_or_else(|_| "uv run python".to_string());
        command
            .split_whitespace()
            .map(ToString::to_string)
            .collect()
    }
}

fn object_section<'a>(
    payload: &'a Map<String, Value>,
    key: &str,
    operation: KernelOperation,
) -> Result<&'a Map<String, Value>, KernelBindingError> {
    payload.get(key).and_then(Value::as_object).ok_or_else(|| {
        KernelBindingError::invalid_request(operation, format!("{key} section must be an object"))
    })
}

fn optional_object_section<'a>(
    payload: &'a Map<String, Value>,
    key: &str,
) -> Option<&'a Map<String, Value>> {
    payload.get(key).and_then(Value::as_object)
}

fn numeric_array_from_aliases(
    values: &Map<String, Value>,
    aliases: &[&str],
    operation: KernelOperation,
) -> Result<Vec<f64>, KernelBindingError> {
    for alias in aliases {
        if let Some(value) = values.get(*alias) {
            return numeric_array(value, *alias, operation);
        }
    }
    Err(KernelBindingError::invalid_request(
        operation,
        "kernel requests require time points in the inputs section",
    ))
}

fn numeric_array(
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

fn required_f64(
    values: &Map<String, Value>,
    key: &str,
    operation: KernelOperation,
) -> Result<f64, KernelBindingError> {
    values.get(key).and_then(Value::as_f64).ok_or_else(|| {
        KernelBindingError::invalid_request(operation, format!("missing numeric parameter '{key}'"))
    })
}

fn logistic_native_response(
    operation: KernelOperation,
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != operation {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            format!("native {operation} requires a {operation} request"),
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "logistic" {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            format!("native {operation} is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            operation,
            format!("{operation} payload must be an object"),
        )
    })?;
    let inputs = object_section(payload, "inputs", operation)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let state = optional_object_section(payload, "state");

    let state_model_key = state
        .and_then(|state| state.get("model_key"))
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

    let constructor_kwargs = state
        .and_then(|state| state.get("constructor_kwargs"))
        .and_then(Value::as_object);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    if has_covariates || has_event || input_covariates {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native logistic execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = payload
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| state.and_then(|state| state.get("parameters")).and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let l = required_f64(parameters, "L", operation)?;
    let k = required_f64(parameters, "k", operation)?;
    let x0 = required_f64(parameters, "x0", operation)?;
    let predictions: Vec<f64> = time
        .iter()
        .map(|t| l / (1.0 + (-k * (t - x0)).exp()))
        .collect();

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(json!({
            "shape": [predictions.len()],
            "dtype": "float64",
            "values": predictions,
            "metadata": {
                "shape": [time.len()]
            }
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "LogisticModel",
            "runtime": "rust_native"
        }),
    })
}

fn unique_temp_path(prefix: &str, extension: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let filename = format!("{prefix}-{nanos}-{}.{}", std::process::id(), extension);
    env::temp_dir().join(filename)
}

pub use serde_json::json;
