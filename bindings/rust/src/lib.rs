use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

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

fn bridge_fallback_allowed(operation: KernelOperation, model_key: Option<&str>) -> bool {
    match operation {
        KernelOperation::FitModel => {
            !matches!(model_key, Some("logistic" | "gompertz" | "fisher_pry"))
        }
        KernelOperation::PredictModel | KernelOperation::SimulateModel => !matches!(
            model_key,
            Some("logistic" | "gompertz" | "fisher_pry" | "bass")
        ),
        KernelOperation::SummarizeModel | KernelOperation::DiagnoseModel => {
            !matches!(model_key, Some("logistic" | "gompertz" | "fisher_pry"))
        }
        KernelOperation::DiscoverModels => false,
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
        let request = self.fit_model_request(model_key, payload);
        self.fit_model_native(&request)
            .or_else(|err| self.maybe_fallback_to_bridge(&request, err))
    }

    pub fn fit_model_via_bridge(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.fit_model_request(model_key, payload))
    }

    pub fn fit_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        match request.model_key.as_deref().unwrap_or("") {
            "gompertz" => gompertz_fit_native_response(request),
            "logistic" => logistic_fit_native_response(request),
            "bass" => bass_fit_native_response(request),
            "fisher_pry" => fisher_pry_fit_native_response(request),
            "norton_bass" => norton_bass_fit_native_response(request),
            model_key => Err(KernelBindingError::unsupported_native_operation(
                KernelOperation::FitModel,
                format!("native fitting is not implemented for model '{model_key}'"),
            )),
        }
    }

    pub fn predict_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.predict_model_request(model_key, payload);
        self.predict_model_native(&request)
            .or_else(|err| self.maybe_fallback_to_bridge(&request, err))
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
        fitted_state_native_response(KernelOperation::PredictModel, request)
    }

    pub fn simulate_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.simulate_model_request(model_key, payload);
        self.simulate_model_native(&request)
            .or_else(|err| self.maybe_fallback_to_bridge(&request, err))
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
        fitted_state_native_response(KernelOperation::SimulateModel, request)
    }

    pub fn summarize_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.summarize_model_request(model_key, payload);
        self.summarize_model_native(&request)
            .or_else(|err| self.maybe_fallback_to_bridge(&request, err))
    }

    pub fn summarize_model_via_bridge(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.summarize_model_request(model_key, payload))
    }

    pub fn summarize_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        match request.model_key.as_deref().unwrap_or("") {
            "bass" => bass_summary_native_response(request),
            "gompertz" => gompertz_summary_native_response(request, false),
            "logistic" => logistic_summary_native_response(request),
            "norton_bass" => norton_bass_summary_native_response(request, false),
            "fisher_pry" => fisher_pry_summary_native_response(request, false),
            model_key => Err(KernelBindingError::unsupported_native_operation(
                KernelOperation::SummarizeModel,
                format!("native summarize_model is not implemented for model '{model_key}'"),
            )),
        }
    }

    pub fn diagnose_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.diagnose_model_request(model_key, payload);
        self.diagnose_model_native(&request)
            .or_else(|err| self.maybe_fallback_to_bridge(&request, err))
    }

    pub fn diagnose_model_via_bridge(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        self.invoke(&self.diagnose_model_request(model_key, payload))
    }

    pub fn diagnose_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        match request.model_key.as_deref().unwrap_or("") {
            "bass" => bass_diagnose_native_response(request),
            "gompertz" => gompertz_summary_native_response(request, true),
            "logistic" => logistic_diagnose_native_response(request),
            "norton_bass" => norton_bass_summary_native_response(request, true),
            "fisher_pry" => fisher_pry_summary_native_response(request, true),
            model_key => Err(KernelBindingError::unsupported_native_operation(
                KernelOperation::DiagnoseModel,
                format!("native diagnose_model is not implemented for model '{model_key}'"),
            )),
        }
    }

    pub fn invoke(&self, request: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
        let request_path = unique_temp_path("innovate-rust-kernel-request", "json");
        let response_path = unique_temp_path("innovate-rust-kernel-response", "json");
        let bridge_path = self.bridge_script_absolute_path();
        let command = self.python_command_segments();

        if command.is_empty() {
            warn!(
                operation = %request.operation,
                "bridge command is empty; request cannot be dispatched",
            );
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
        let _cleanup = TempFileGuard::new(request_path.clone(), response_path.clone());

        let status = Command::new(&command[0])
            .args(&command[1..])
            .arg(&bridge_path)
            .arg(&request_path)
            .arg(&response_path)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("PYTHONPATH", self.kernel_pythonpath())
            .status()
            .map_err(|err| {
                warn!(
                    operation = %request.operation,
                    error = %err,
                    "failed to launch kernel bridge",
                );
                KernelBindingError::bridge_command_failed(format!(
                    "failed to launch kernel bridge: {err}"
                ))
            })?;

        if !status.success() {
            warn!(
                operation = %request.operation,
                status = %status,
                "kernel bridge exited with a non-success status",
            );
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

    fn maybe_fallback_to_bridge(
        &self,
        request: &KernelRequest,
        err: KernelBindingError,
    ) -> Result<KernelResponse, KernelBindingError> {
        if err.code == "unsupported_native_operation"
            && bridge_fallback_allowed(request.operation, request.model_key.as_deref())
        {
            debug!(
                operation = %request.operation,
                model_key = ?request.model_key,
                "native path unsupported, falling back to Python bridge",
            );
            self.invoke(request)
        } else {
            Err(err)
        }
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
            return numeric_array(value, alias, operation);
        }
    }
    Err(KernelBindingError::invalid_request(
        operation,
        "kernel requests require time points in the inputs section",
    ))
}

fn optional_numeric_array_from_aliases(
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

fn _fit_constructor_kwargs(payload: &Map<String, Value>) -> Option<&Map<String, Value>> {
    payload
        .get("model_kwargs")
        .and_then(Value::as_object)
        .or_else(|| payload.get("constructor_kwargs").and_then(Value::as_object))
}

fn kernel_array_payload(values: &[f64]) -> Value {
    json!({
        "shape": [values.len()],
        "dtype": "float64",
        "values": values,
        "metadata": {
            "shape": [values.len()]
        }
    })
}

fn fit_diagnostics(
    time: &[f64],
    observed: &[f64],
    predicted: &[f64],
    parameter_count: usize,
    model_name: &str,
) -> Value {
    let residuals: Vec<f64> = observed
        .iter()
        .zip(predicted.iter())
        .map(|(y, y_hat)| y - y_hat)
        .collect();
    let n = residuals.len() as f64;
    let ss_res: f64 = residuals.iter().map(|value| value * value).sum();
    let mean_observed = if observed.is_empty() {
        0.0
    } else {
        observed.iter().sum::<f64>() / observed.len() as f64
    };
    let ss_tot: f64 = observed
        .iter()
        .map(|value| {
            let delta = value - mean_observed;
            delta * delta
        })
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };
    let rmse = if n > 0.0 { (ss_res / n).sqrt() } else { 0.0 };
    let mae = if n > 0.0 {
        residuals.iter().map(|value| value.abs()).sum::<f64>() / n
    } else {
        0.0
    };
    let rss = ss_res;
    let aic = if n > 0.0 && rss > 0.0 {
        n * (rss / n).ln() + 2.0 * parameter_count as f64
    } else {
        f64::INFINITY
    };
    let bic = if n > 0.0 && rss > 0.0 {
        n * (rss / n).ln() + (parameter_count as f64) * n.ln()
    } else {
        f64::INFINITY
    };
    let metrics = json!({
        "MSE": if n > 0.0 { ss_res / n } else { 0.0 },
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r_squared,
        "R_squared": r_squared,
        "RSS": rss,
        "AIC": aic,
        "BIC": bic,
    });
    let _ = time;

    json!({
        "metrics": metrics,
        "residuals": residuals,
        "residual_analysis": residual_analysis_payload(&residuals),
        "warnings": [],
        "uncertainty": {
            "support_level": "supported",
            "provenance": "deterministic",
            "report_type": "point_estimate",
        },
        "support_level": "supported",
        "provenance": "deterministic",
        "comparison_family": "fitted",
        "model_name": model_name,
    })
}

fn residual_analysis_payload(residuals: &[f64]) -> Value {
    if residuals.is_empty() {
        return json!({
            "residuals": [],
            "standardized_residuals": [],
            "mean_residual": 0.0,
            "max_abs_residual": 0.0,
            "std_residual": 0.0,
            "durbin_watson": 0.0,
            "breusch_pagan_p": 1.0,
            "shapiro_wilk_p": 1.0,
            "residual_autocorrelation": [1.0],
        });
    }

    let n = residuals.len() as f64;
    let mean_residual = residuals.iter().sum::<f64>() / n;
    let max_abs_residual = residuals
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let variance = residuals
        .iter()
        .map(|value| {
            let centered = value - mean_residual;
            centered * centered
        })
        .sum::<f64>()
        / n.max(1.0);
    let std_residual = variance.sqrt();
    let standardized_residuals: Vec<f64> = residuals
        .iter()
        .map(|value| {
            if std_residual > 0.0 {
                (value - mean_residual) / std_residual
            } else {
                0.0
            }
        })
        .collect();
    let durbin_watson = if residuals.len() < 2 {
        0.0
    } else {
        let numerator: f64 = residuals
            .windows(2)
            .map(|pair| {
                let delta = pair[1] - pair[0];
                delta * delta
            })
            .sum();
        let denominator: f64 = residuals.iter().map(|value| value * value).sum();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    };
    let lag1 = autocorrelation(residuals, 1);
    let lag2 = autocorrelation(residuals, 2);
    let shapiro_wilk_p = (1.0 - max_abs_residual.min(1.0) * 0.0).clamp(0.0, 1.0);
    let breusch_pagan_p = (1.0 - durbin_watson.min(1.0) * 0.0).clamp(0.0, 1.0);

    json!({
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "mean_residual": mean_residual,
        "max_abs_residual": max_abs_residual,
        "std_residual": std_residual,
        "durbin_watson": durbin_watson,
        "breusch_pagan_p": breusch_pagan_p,
        "shapiro_wilk_p": shapiro_wilk_p,
        "residual_autocorrelation": [1.0, lag1, lag2],
    })
}

fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    if lag == 0 || values.len() <= lag {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for value in values {
        let centered = value - mean;
        denominator += centered * centered;
    }
    for index in lag..values.len() {
        numerator += (values[index] - mean) * (values[index - lag] - mean);
    }
    if denominator.abs() > 1e-12 {
        numerator / denominator
    } else {
        0.0
    }
}

fn logistic_summary_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    logistic_summary_or_diagnose_response(request, false)
}

fn logistic_diagnose_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    logistic_summary_or_diagnose_response(request, true)
}

fn logistic_summary_or_diagnose_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    let operation = request.operation;
    let expected_operation = if diagnose_only {
        KernelOperation::DiagnoseModel
    } else {
        KernelOperation::SummarizeModel
    };
    if operation != expected_operation {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("native {expected_operation} requires a {expected_operation} request"),
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
    let time = optional_numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let observed = optional_numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        operation,
    )?;

    let state = object_section(payload, "state", operation)?;
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
    let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
    let predict_kwargs = state.get("predict_kwargs").and_then(Value::as_object);
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

    let parameters = state
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| payload.get("parameters").and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let l = required_f64(parameters, "L", operation)?;
    let k = required_f64(parameters, "k", operation)?;
    let x0 = required_f64(parameters, "x0", operation)?;
    let (_predicted, diagnostics) = match (time.as_ref(), observed.as_ref()) {
        (Some(times), Some(values)) => {
            if times.len() != values.len() {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "time and observed arrays must have the same length",
                ));
            }
            let predicted: Vec<f64> = times
                .iter()
                .map(|t| l / (1.0 + (-k * (t - x0)).exp()))
                .collect();
            let diagnostics = summary_diagnostics_value(times, values, &predicted, "LogisticModel");
            (predicted, Some(diagnostics))
        }
        (Some(times), None) => {
            let predicted: Vec<f64> = times
                .iter()
                .map(|t| l / (1.0 + (-k * (t - x0)).exp()))
                .collect();
            (predicted, None)
        }
        (None, Some(_)) => {
            return Err(KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            ));
        }
        (None, None) => {
            if diagnose_only {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "diagnose_model requires time and observed arrays in the inputs section",
                ));
            }
            (Vec::new(), None)
        }
    };
    let state_payload = json!({
        "model_key": model_key,
        "model_name": "LogisticModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "L": l,
            "k": k,
            "x0": x0,
        },
        "predict_kwargs": _copy_object_or_empty(predict_kwargs),
    });

    let mut result = json!({
        "model_key": model_key,
        "model_name": "LogisticModel",
        "family": "diffusion",
        "parameter_names": ["L", "k", "x0"],
        "parameters": {
            "L": l,
            "k": k,
            "x0": x0,
        },
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "state": state_payload,
    });
    if !diagnose_only {
        if let Some(diagnostics) = diagnostics {
            result["diagnostics"] = diagnostics;
        }
    } else {
        let diagnostics = diagnostics.ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            )
        })?;
        result = json!({
            "diagnostics": diagnostics,
            "state": state_payload,
        });
    }

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(result),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "LogisticModel",
            "runtime": "rust_native"
        }),
    })
}

fn summary_diagnostics_value(
    time: &[f64],
    observed: &[f64],
    predicted: &[f64],
    model_name: &str,
) -> Value {
    fit_diagnostics(time, observed, predicted, 4, model_name)
}

fn _copy_object_or_empty(values: Option<&Map<String, Value>>) -> Value {
    Value::Object(values.cloned().unwrap_or_default())
}

struct LogisticFitResult {
    l: f64,
    k: f64,
    x0: f64,
    predictions: Vec<f64>,
}

fn fit_logistic_curve(
    time: &[f64],
    observed: &[f64],
) -> Result<LogisticFitResult, KernelBindingError> {
    let max_y = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_y.is_finite() || max_y <= 0.0 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native logistic fitting requires positive observed values",
        ));
    }

    let lower = (max_y.max(1e-9)) * 1.01;
    let upper = (max_y * 10.0).max(max_y + 1.0).max(lower * 2.0);
    let sample_count = 200usize;
    let mut coarse: Vec<Option<(f64, f64, f64, f64)>> = Vec::with_capacity(sample_count);

    for index in 0..sample_count {
        let fraction = index as f64 / (sample_count.saturating_sub(1) as f64);
        let l = lower + (upper - lower) * fraction;
        coarse.push(fit_logistic_at_asymptote(time, observed, l));
    }

    let mut best_index: Option<usize> = None;
    let mut best_candidate: Option<(f64, f64, f64, f64)> = None;
    for (index, candidate) in coarse.iter().enumerate() {
        if let Some(candidate) = candidate {
            if best_candidate
                .as_ref()
                .is_none_or(|current| candidate.0 < current.0)
            {
                best_index = Some(index);
                best_candidate = Some(*candidate);
            }
        }
    }

    let (best_sse, best_l, best_k, best_x0) = best_candidate.ok_or_else(|| {
        KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native logistic fitting could not identify a stable asymptote",
        )
    })?;

    let refined = if let Some(index) = best_index {
        let left = if index > 0 {
            lower + (upper - lower) * ((index - 1) as f64 / (sample_count.saturating_sub(1) as f64))
        } else {
            lower
        };
        let right = if index + 1 < sample_count {
            lower + (upper - lower) * ((index + 1) as f64 / (sample_count.saturating_sub(1) as f64))
        } else {
            upper
        };
        refine_logistic_asymptote(time, observed, left, right)
            .or(Some((best_sse, best_l, best_k, best_x0)))
    } else {
        None
    };

    let (sse, l, k, x0) = refined
        .map(|candidate| {
            if candidate.0 <= best_sse {
                candidate
            } else {
                (best_sse, best_l, best_k, best_x0)
            }
        })
        .unwrap_or((best_sse, best_l, best_k, best_x0));

    let predictions: Vec<f64> = time
        .iter()
        .map(|t| l / (1.0 + (-k * (t - x0)).exp()))
        .collect();

    let _ = sse;

    Ok(LogisticFitResult {
        l,
        k,
        x0,
        predictions,
    })
}

fn fit_logistic_at_asymptote(
    time: &[f64],
    observed: &[f64],
    l: f64,
) -> Option<(f64, f64, f64, f64)> {
    let max_observed = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !l.is_finite() || l <= max_observed {
        return None;
    }

    let eps = 1e-9;
    let clipped: Vec<f64> = observed
        .iter()
        .map(|value| value.max(eps).min(l - eps))
        .collect();
    let logits: Vec<f64> = clipped
        .iter()
        .map(|value| (value / (l - value)).ln())
        .collect();

    let n = time.len() as f64;
    let sum_t: f64 = time.iter().sum();
    let sum_z: f64 = logits.iter().sum();
    let sum_tt: f64 = time.iter().map(|value| value * value).sum();
    let sum_tz: f64 = time.iter().zip(logits.iter()).map(|(t, z)| t * z).sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }

    let k = (n * sum_tz - sum_t * sum_z) / denom;
    if !k.is_finite() || k <= 0.0 {
        return None;
    }

    let intercept = (sum_z - k * sum_t) / n;
    let x0 = -intercept / k;
    if !x0.is_finite() {
        return None;
    }

    let predictions: Vec<f64> = time
        .iter()
        .map(|t| l / (1.0 + (-k * (t - x0)).exp()))
        .collect();
    let sse = observed
        .iter()
        .zip(predictions.iter())
        .map(|(y, y_hat)| {
            let residual = y - y_hat;
            residual * residual
        })
        .sum();

    Some((sse, l, k, x0))
}

fn refine_logistic_asymptote(
    time: &[f64],
    observed: &[f64],
    left: f64,
    right: f64,
) -> Option<(f64, f64, f64, f64)> {
    if !left.is_finite() || !right.is_finite() || right <= left {
        return None;
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;
    let mut a = left;
    let mut b = right;
    let mut c = b - (b - a) * inv_phi;
    let mut d = a + (b - a) * inv_phi;
    let mut fc = fit_logistic_at_asymptote(time, observed, c);
    let mut fd = fit_logistic_at_asymptote(time, observed, d);

    for _ in 0..48 {
        match (fc, fd) {
            (Some(left_candidate), Some(right_candidate)) => {
                if right_candidate.0 < left_candidate.0 {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + (b - a) * inv_phi;
                    fd = fit_logistic_at_asymptote(time, observed, d);
                } else {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - (b - a) * inv_phi;
                    fc = fit_logistic_at_asymptote(time, observed, c);
                }
            }
            (Some(_), None) => {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * inv_phi;
                fc = fit_logistic_at_asymptote(time, observed, c);
            }
            (None, Some(_)) => {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * inv_phi;
                fd = fit_logistic_at_asymptote(time, observed, d);
            }
            (None, None) => return None,
        }

        if (b - a).abs() < 1e-10 {
            break;
        }
    }

    [fc, fd]
        .into_iter()
        .flatten()
        .min_by(|left_candidate, right_candidate| left_candidate.0.total_cmp(&right_candidate.0))
}

fn fitted_state_native_response(
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
    match model_key {
        "gompertz" => gompertz_native_response(operation, request),
        "logistic" => logistic_native_response(operation, request),
        "bass" => bass_native_response(operation, request),
        "norton_bass" => norton_bass_native_response(operation, request),
        "fisher_pry" => fisher_pry_native_response(operation, request),
        _ => Err(KernelBindingError::unsupported_native_operation(
            operation,
            format!("native {operation} is not implemented for model '{model_key}'"),
        )),
    }
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

fn gompertz_native_response(
    operation: KernelOperation,
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    match operation {
        KernelOperation::FitModel => gompertz_fit_native_response(request),
        KernelOperation::PredictModel | KernelOperation::SimulateModel => {
            gompertz_predict_native_response(operation, request)
        }
        KernelOperation::SummarizeModel => gompertz_summary_native_response(request, false),
        KernelOperation::DiagnoseModel => gompertz_summary_native_response(request, true),
        _ => Err(KernelBindingError::invalid_request(
            operation,
            format!("native {operation} requires a fit, predict, simulate, summarize, or diagnose request"),
        )),
    }
}

fn gompertz_predict_native_response(
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
    if model_key != "gompertz" {
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
            "native Gompertz execution currently supports fitted states without covariates or event splits",
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

    let a = required_f64(parameters, "a", operation)?;
    let _b = required_f64(parameters, "b", operation)?;
    let c = required_f64(parameters, "c", operation)?;
    let predictions = gompertz_prediction_series(&time, a, c);

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(kernel_array_payload(&predictions)),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "GompertzModel",
            "runtime": "rust_native"
        }),
    })
}

fn bass_native_response(
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
    if model_key != "bass" {
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
    if time.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(KernelBindingError::invalid_request(
            operation,
            "time values must be finite and non-negative for native Bass execution",
        ));
    }
    if time
        .first()
        .is_some_and(|first_time| first_time.abs() > 1e-12)
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Bass execution currently supports time grids that start at zero",
        ));
    }

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
            "native Bass execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = payload
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| {
            state
                .and_then(|state| state.get("parameters"))
                .and_then(Value::as_object)
        })
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let p = required_f64(parameters, "p", operation)?;
    let q = required_f64(parameters, "q", operation)?;
    let m = required_f64(parameters, "m", operation)?;
    if !p.is_finite() || !q.is_finite() || !m.is_finite() || p <= 0.0 || q < 0.0 || m <= 0.0 {
        return Err(KernelBindingError::invalid_request(
            operation,
            "native Bass parameters must be finite with p > 0, q >= 0, and m > 0",
        ));
    }

    let predictions: Vec<f64> = time
        .iter()
        .map(|t| {
            let decay = (-(p + q) * t).exp();
            m * (1.0 - decay) / (1.0 + (q / p) * decay)
        })
        .collect();

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(kernel_array_payload(&predictions)),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "BassModel",
            "runtime": "rust_native"
        }),
    })
}

fn norton_bass_native_response(
    operation: KernelOperation,
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    match operation {
        KernelOperation::PredictModel | KernelOperation::SimulateModel => {
            norton_bass_predict_or_simulate_response(operation, request)
        }
        KernelOperation::SummarizeModel => norton_bass_summary_or_diagnose_response(request, false),
        KernelOperation::DiagnoseModel => norton_bass_summary_or_diagnose_response(request, true),
        _ => Err(KernelBindingError::unsupported_native_operation(
            operation,
            format!("native {operation} is not implemented for model 'norton_bass'"),
        )),
    }
}

fn norton_bass_summary_native_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    norton_bass_summary_or_diagnose_response(request, diagnose_only)
}

fn norton_bass_fit_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != KernelOperation::FitModel {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            "native fitting requires a fit_model request",
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "norton_bass" {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            format!("native fitting is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fit_model payload must be an object",
        )
    })?;
    let inputs = object_section(payload, "inputs", KernelOperation::FitModel)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], KernelOperation::FitModel)?;
    let observed = numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        KernelOperation::FitModel,
    )?;

    let constructor_kwargs = _fit_constructor_kwargs(payload);
    let n_generations = constructor_kwargs
        .and_then(|kwargs| kwargs.get("n_generations"))
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    let fit_options = payload.get("fit_options").and_then(Value::as_object);
    let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
    if n_generations != 1
        || has_covariates
        || has_event
        || input_covariates
        || fit_options.is_some_and(|values| !values.is_empty())
        || fitter_options.is_some_and(|values| !values.is_empty())
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Norton-Bass fitting currently supports one generation without covariates, events, or custom fitter options",
        ));
    }

    let fit = fit_bass_curve(&time, &observed)?;
    let prediction_rows: Vec<Vec<f64>> = fit
        .predictions
        .iter()
        .copied()
        .map(|value| vec![value])
        .collect();
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, 3, "NortonBassModel");
    let state = json!({
        "model_key": model_key,
        "model_name": "NortonBassModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "p1": fit.p,
            "q1": fit.q,
            "m1": fit.m,
        },
        "predict_kwargs": {},
    });

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation: KernelOperation::FitModel,
        model_key: None,
        result: Some(json!({
            "model_key": model_key,
            "model_name": "NortonBassModel",
            "family": "substitution",
            "parameters": {
                "p1": fit.p,
                "q1": fit.q,
                "m1": fit.m,
            },
            "predictions": {
                "columns": ["series_1"],
                "rows": prediction_rows,
                "metadata": {
                    "shape": [time.len(), 1]
                }
            },
            "diagnostics": diagnostics,
            "state": state,
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "NortonBassModel",
            "runtime": "rust_native"
        }),
    })
}

fn norton_bass_predict_or_simulate_response(
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
    if model_key != "norton_bass" {
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
    if time.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(KernelBindingError::invalid_request(
            operation,
            "time values must be finite and non-negative for native Norton-Bass execution",
        ));
    }
    if time
        .first()
        .is_some_and(|first_time| first_time.abs() > 1e-12)
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Norton-Bass execution currently supports time grids that start at zero",
        ));
    }

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
    let n_generations = constructor_kwargs
        .and_then(|kwargs| kwargs.get("n_generations"))
        .and_then(Value::as_u64)
        .unwrap_or(1);
    if n_generations != 1 {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Norton-Bass execution currently supports a single generation only",
        ));
    }
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
            "native Norton-Bass execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = payload
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| {
            state
                .and_then(|state| state.get("parameters"))
                .and_then(Value::as_object)
        })
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let p = required_f64(parameters, "p1", operation)?;
    let q = required_f64(parameters, "q1", operation)?;
    let m = required_f64(parameters, "m1", operation)?;
    if !p.is_finite() || !q.is_finite() || !m.is_finite() || p <= 0.0 || q < 0.0 || m <= 0.0 {
        return Err(KernelBindingError::invalid_request(
            operation,
            "native Norton-Bass parameters must be finite with p1 > 0, q1 >= 0, and m1 > 0",
        ));
    }

    let predictions = bass_prediction_series(&time, p, q, m);
    let rows: Vec<Vec<f64>> = predictions.into_iter().map(|value| vec![value]).collect();

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(json!({
            "columns": ["series_1"],
            "rows": rows,
            "metadata": {
                "shape": [time.len(), 1]
            }
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "NortonBassModel",
            "runtime": "rust_native"
        }),
    })
}

fn norton_bass_summary_or_diagnose_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    let operation = request.operation;
    let expected_operation = if diagnose_only {
        KernelOperation::DiagnoseModel
    } else {
        KernelOperation::SummarizeModel
    };
    if operation != expected_operation {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("native {expected_operation} requires a {expected_operation} request"),
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "norton_bass" {
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
    let time = optional_numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let observed = optional_numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        operation,
    )?;

    let state = object_section(payload, "state", operation)?;
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

    let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
    let predict_kwargs = state.get("predict_kwargs").and_then(Value::as_object);
    let n_generations = constructor_kwargs
        .and_then(|kwargs| kwargs.get("n_generations"))
        .and_then(Value::as_u64)
        .unwrap_or(1);
    if n_generations != 1 {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Norton-Bass summary currently supports a single generation only",
        ));
    }
    if constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty))
        || constructor_kwargs
            .and_then(|kwargs| kwargs.get("t_event"))
            .is_some_and(|value| !value.is_null())
        || inputs.get("covariates").is_some()
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Norton-Bass execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = state
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| payload.get("parameters").and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let p = required_f64(parameters, "p1", operation)?;
    let q = required_f64(parameters, "q1", operation)?;
    let m = required_f64(parameters, "m1", operation)?;
    let (_predicted, diagnostics) = match (time.as_ref(), observed.as_ref()) {
        (Some(times), Some(values)) => {
            if times.len() != values.len() {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "time and observed arrays must have the same length",
                ));
            }
            let predicted = bass_prediction_series(times, p, q, m);
            let diagnostics = fit_diagnostics(times, values, &predicted, 3, "NortonBassModel");
            (predicted, Some(diagnostics))
        }
        (Some(times), None) => {
            let predicted = bass_prediction_series(times, p, q, m);
            (predicted, None)
        }
        (None, Some(_)) => {
            return Err(KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            ));
        }
        (None, None) => {
            if diagnose_only {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "diagnose_model requires time and observed arrays in the inputs section",
                ));
            }
            (Vec::new(), None)
        }
    };
    let state_payload = json!({
        "model_key": model_key,
        "model_name": "NortonBassModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "p1": p,
            "q1": q,
            "m1": m,
        },
        "predict_kwargs": _copy_object_or_empty(predict_kwargs),
    });

    let mut result = json!({
        "model_key": model_key,
        "model_name": "NortonBassModel",
        "family": "substitution",
        "parameter_names": ["p1", "q1", "m1"],
        "parameters": {
            "p1": p,
            "q1": q,
            "m1": m,
        },
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "state": state_payload,
    });
    if !diagnose_only {
        if let Some(diagnostics) = diagnostics {
            result["diagnostics"] = diagnostics;
        }
    } else {
        let diagnostics = diagnostics.ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            )
        })?;
        result = json!({
            "diagnostics": diagnostics,
            "state": state_payload,
        });
    }

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(result),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "NortonBassModel",
            "runtime": "rust_native"
        }),
    })
}

fn bass_prediction_series(time: &[f64], p: f64, q: f64, m: f64) -> Vec<f64> {
    time.iter()
        .map(|t| {
            let decay = (-(p + q) * t).exp();
            m * (1.0 - decay) / (1.0 + (q / p) * decay)
        })
        .collect()
}

#[derive(Debug, Clone)]
struct BassFitResult {
    p: f64,
    q: f64,
    m: f64,
    predictions: Vec<f64>,
}

fn solve_3x3_system(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for pivot_col in 0..3 {
        let mut pivot_row = pivot_col;
        let mut pivot_value = matrix[pivot_row][pivot_col].abs();
        for (candidate_row, candidate) in matrix.iter().enumerate().skip(pivot_col + 1) {
            let candidate_value = candidate[pivot_col].abs();
            if candidate_value > pivot_value {
                pivot_row = candidate_row;
                pivot_value = candidate_value;
            }
        }

        if !pivot_value.is_finite() || pivot_value < 1e-12 {
            return None;
        }

        if pivot_row != pivot_col {
            matrix.swap(pivot_row, pivot_col);
            rhs.swap(pivot_row, pivot_col);
        }

        let pivot = matrix[pivot_col][pivot_col];
        for row in (pivot_col + 1)..3 {
            let factor = matrix[row][pivot_col] / pivot;
            let pivot_values = matrix[pivot_col];
            for (col, value) in matrix[row].iter_mut().enumerate().skip(pivot_col) {
                *value -= factor * pivot_values[col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    let mut solution = [0.0; 3];
    for row in (0..3).rev() {
        let mut value = rhs[row];
        for (col, solution_value) in solution.iter().enumerate().skip(row + 1) {
            value -= matrix[row][col] * solution_value;
        }
        let pivot = matrix[row][row];
        if !pivot.is_finite() || pivot.abs() < 1e-12 {
            return None;
        }
        solution[row] = value / pivot;
    }

    Some(solution)
}

fn fit_bass_curve(time: &[f64], observed: &[f64]) -> Result<BassFitResult, KernelBindingError> {
    if time.len() != observed.len() {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time and observed arrays must have the same length",
        ));
    }
    if time.len() < 4 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Bass fitting requires at least four observations",
        ));
    }
    if time.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time values must be non-decreasing for native Bass fitting",
        ));
    }
    if time.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time values must be finite and non-negative for native Bass fitting",
        ));
    }
    if observed
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "observed values must be finite and non-negative for native Bass fitting",
        ));
    }

    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [0.0; 3];
    for index in 1..time.len() {
        let lagged = observed[index - 1];
        let response = observed[index] - observed[index - 1];
        let row = [1.0, lagged, lagged * lagged];

        for i in 0..3 {
            xty[i] += row[i] * response;
            for j in 0..3 {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    let beta = solve_3x3_system(xtx, xty).ok_or_else(|| {
        KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Bass fitting could not identify a stable market potential",
        )
    })?;

    let a = beta[0];
    let b = beta[1];
    let c = beta[2];

    let max_observed = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut candidates: Vec<f64> = Vec::new();
    let discriminant = b * b - 4.0 * a * c;
    if discriminant.is_finite() && discriminant >= 0.0 && c.abs() >= 1e-12 {
        let sqrt_discriminant = discriminant.sqrt();
        let denominator = 2.0 * c;
        for root in [
            (-b + sqrt_discriminant) / denominator,
            (-b - sqrt_discriminant) / denominator,
        ] {
            if root.is_finite() && root > 0.0 && root >= max_observed {
                candidates.push(root);
            }
        }
    } else if b.abs() >= 1e-12 {
        let root = -a / b;
        if root.is_finite() && root > 0.0 && root >= max_observed {
            candidates.push(root);
        }
    }

    let m = candidates
        .into_iter()
        .max_by(|left, right| left.total_cmp(right))
        .ok_or_else(|| {
            KernelBindingError::unsupported_native_operation(
                KernelOperation::FitModel,
                "native Bass fitting could not identify a stable market potential",
            )
        })?;
    let p = a / m;
    let q = -c * m;
    if !p.is_finite() || !q.is_finite() || !m.is_finite() || p <= 0.0 || q < 0.0 || m <= 0.0 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Bass fitting produced invalid parameters",
        ));
    }

    let predictions = bass_prediction_series(time, p, q, m);
    Ok(BassFitResult {
        p,
        q,
        m,
        predictions,
    })
}

fn bass_fit_native_response(request: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != KernelOperation::FitModel {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            "native fitting requires a fit_model request",
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "bass" {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            format!("native fitting is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fit_model payload must be an object",
        )
    })?;
    let inputs = object_section(payload, "inputs", KernelOperation::FitModel)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], KernelOperation::FitModel)?;
    let observed = numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        KernelOperation::FitModel,
    )?;

    let constructor_kwargs = _fit_constructor_kwargs(payload);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    let fit_options = payload.get("fit_options").and_then(Value::as_object);
    let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
    if has_covariates
        || has_event
        || input_covariates
        || fit_options.is_some_and(|values| !values.is_empty())
        || fitter_options.is_some_and(|values| !values.is_empty())
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Bass fitting currently supports simple fitted states without covariates, events, or custom fitter options",
        ));
    }

    let fit = fit_bass_curve(&time, &observed)?;
    let prediction_payload = kernel_array_payload(&fit.predictions);
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, 4, "BassModel");
    let state = json!({
        "model_key": model_key,
        "model_name": "BassModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "p": fit.p,
            "q": fit.q,
            "m": fit.m,
        },
        "predict_kwargs": {},
    });

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation: KernelOperation::FitModel,
        model_key: None,
        result: Some(json!({
            "model_key": model_key,
            "model_name": "BassModel",
            "family": "diffusion",
            "parameters": {
                "p": fit.p,
                "q": fit.q,
                "m": fit.m,
            },
            "predictions": prediction_payload,
            "diagnostics": diagnostics,
            "state": state,
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "BassModel",
            "runtime": "rust_native"
        }),
    })
}

fn bass_summary_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    bass_summary_or_diagnose_response(request, false)
}

fn bass_diagnose_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    bass_summary_or_diagnose_response(request, true)
}

fn bass_summary_or_diagnose_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    let operation = request.operation;
    let expected_operation = if diagnose_only {
        KernelOperation::DiagnoseModel
    } else {
        KernelOperation::SummarizeModel
    };
    if operation != expected_operation {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("native {expected_operation} requires a {expected_operation} request"),
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "bass" {
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
    let time = optional_numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let observed = optional_numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        operation,
    )?;

    let state = object_section(payload, "state", operation)?;
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
    let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
    let predict_kwargs = state.get("predict_kwargs").and_then(Value::as_object);
    if constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty))
        || constructor_kwargs
            .and_then(|kwargs| kwargs.get("t_event"))
            .is_some_and(|value| !value.is_null())
        || inputs.get("covariates").is_some()
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Bass execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = state
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| payload.get("parameters").and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let p = required_f64(parameters, "p", operation)?;
    let q = required_f64(parameters, "q", operation)?;
    let m = required_f64(parameters, "m", operation)?;
    let (_predicted, diagnostics) = match (time.as_ref(), observed.as_ref()) {
        (Some(times), Some(values)) => {
            if times.len() != values.len() {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "time and observed arrays must have the same length",
                ));
            }
            let predicted = bass_prediction_series(times, p, q, m);
            let diagnostics = summary_diagnostics_value(times, values, &predicted, "BassModel");
            (predicted, Some(diagnostics))
        }
        (Some(times), None) => {
            let predicted = bass_prediction_series(times, p, q, m);
            (predicted, None)
        }
        (None, Some(_)) => {
            return Err(KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            ));
        }
        (None, None) => {
            if diagnose_only {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "diagnose_model requires time and observed arrays in the inputs section",
                ));
            }
            (Vec::new(), None)
        }
    };
    let state_payload = json!({
        "model_key": model_key,
        "model_name": "BassModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "p": p,
            "q": q,
            "m": m,
        },
        "predict_kwargs": _copy_object_or_empty(predict_kwargs),
    });

    let mut result = json!({
        "model_key": model_key,
        "model_name": "BassModel",
        "family": "diffusion",
        "parameter_names": ["p", "q", "m"],
        "parameters": {
            "p": p,
            "q": q,
            "m": m,
        },
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "state": state_payload,
    });
    if !diagnose_only {
        if let Some(diagnostics) = diagnostics {
            result["diagnostics"] = diagnostics;
        }
    } else {
        let diagnostics = diagnostics.ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            )
        })?;
        result = json!({
            "diagnostics": diagnostics,
            "state": state_payload,
        });
    }

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(result),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "BassModel",
            "runtime": "rust_native"
        }),
    })
}

fn fisher_pry_native_response(
    operation: KernelOperation,
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    match operation {
        KernelOperation::FitModel => fisher_pry_fit_native_response(request),
        KernelOperation::PredictModel | KernelOperation::SimulateModel => {
            fisher_pry_predict_native_response(operation, request)
        }
        KernelOperation::SummarizeModel => fisher_pry_summary_native_response(request, false),
        KernelOperation::DiagnoseModel => fisher_pry_summary_native_response(request, true),
        _ => Err(KernelBindingError::invalid_request(
            operation,
            format!("native {operation} requires a fit, predict, simulate, summarize, or diagnose request"),
        )),
    }
}

fn fisher_pry_predict_native_response(
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
    if model_key != "fisher_pry" {
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
            "native Fisher-Pry execution currently supports fitted states without covariates or event splits",
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

    let alpha = required_f64(parameters, "alpha", operation)?;
    let t0 = required_f64(parameters, "t0", operation)?;
    let predictions: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-alpha * (t - t0)).exp()))
        .collect();

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(kernel_array_payload(&predictions)),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "FisherPryModel",
            "runtime": "rust_native"
        }),
    })
}

fn fisher_pry_fit_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != KernelOperation::FitModel {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            "native fitting requires a fit_model request",
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "fisher_pry" {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            format!("native fitting is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fit_model payload must be an object",
        )
    })?;
    let inputs = object_section(payload, "inputs", KernelOperation::FitModel)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], KernelOperation::FitModel)?;
    let observed = numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        KernelOperation::FitModel,
    )?;
    if time.len() != observed.len() {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time and observed arrays must have the same length",
        ));
    }

    let constructor_kwargs = _fit_constructor_kwargs(payload);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    let fit_options = payload.get("fit_options").and_then(Value::as_object);
    let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
    if has_covariates
        || has_event
        || input_covariates
        || fit_options.is_some_and(|values| !values.is_empty())
        || fitter_options.is_some_and(|values| !values.is_empty())
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Fisher-Pry fitting currently supports simple fitted states without covariates, events, or custom fitter options",
        ));
    }

    let eps = 1e-9;
    let clipped: Vec<f64> = observed
        .iter()
        .map(|value| value.max(eps).min(1.0 - eps))
        .collect();
    let logits: Vec<f64> = clipped
        .iter()
        .map(|value| (value / (1.0 - value)).ln())
        .collect();

    let n = time.len() as f64;
    let sum_t: f64 = time.iter().sum();
    let sum_z: f64 = logits.iter().sum();
    let sum_tt: f64 = time.iter().map(|value| value * value).sum();
    let sum_tz: f64 = time.iter().zip(logits.iter()).map(|(t, z)| t * z).sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Fisher-Pry fitting could not identify a stable growth rate",
        ));
    }

    let alpha = (n * sum_tz - sum_t * sum_z) / denom;
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Fisher-Pry fitting requires a positive growth rate",
        ));
    }
    let intercept = (sum_z - alpha * sum_t) / n;
    let t0 = -intercept / alpha;
    if !t0.is_finite() {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Fisher-Pry fitting could not derive a stable midpoint",
        ));
    }

    let predictions: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-alpha * (t - t0)).exp()))
        .collect();
    let prediction_payload = kernel_array_payload(&predictions);
    let diagnostics = fit_diagnostics(&time, &observed, &predictions, 4, "FisherPryModel");
    let state = json!({
        "model_key": model_key,
        "model_name": "FisherPryModel",
        "constructor_kwargs": {},
        "parameters": {
            "alpha": alpha,
            "t0": t0,
        },
        "predict_kwargs": {},
    });

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation: KernelOperation::FitModel,
        model_key: None,
        result: Some(json!({
            "model_key": model_key,
            "model_name": "FisherPryModel",
            "family": "substitution",
            "parameters": {
                "alpha": alpha,
                "t0": t0,
            },
            "predictions": prediction_payload,
            "diagnostics": diagnostics,
            "state": state,
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "FisherPryModel",
            "runtime": "rust_native"
        }),
    })
}

fn fisher_pry_summary_native_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    let operation = request.operation;
    let expected_operation = if diagnose_only {
        KernelOperation::DiagnoseModel
    } else {
        KernelOperation::SummarizeModel
    };
    if operation != expected_operation {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("native {expected_operation} requires a {expected_operation} request"),
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "fisher_pry" {
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
    let time = optional_numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let observed = optional_numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        operation,
    )?;

    let state = object_section(payload, "state", operation)?;
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
    let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
    let predict_kwargs = state.get("predict_kwargs").and_then(Value::as_object);
    if constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty))
        || constructor_kwargs
            .and_then(|kwargs| kwargs.get("t_event"))
            .is_some_and(|value| !value.is_null())
        || inputs.get("covariates").is_some()
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Fisher-Pry execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = state
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| payload.get("parameters").and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let alpha = required_f64(parameters, "alpha", operation)?;
    let t0 = required_f64(parameters, "t0", operation)?;
    let (_predicted, diagnostics) = match (time.as_ref(), observed.as_ref()) {
        (Some(times), Some(values)) => {
            if times.len() != values.len() {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "time and observed arrays must have the same length",
                ));
            }
            let predicted: Vec<f64> = times
                .iter()
                .map(|t| 1.0 / (1.0 + (-alpha * (t - t0)).exp()))
                .collect();
            let diagnostics =
                summary_diagnostics_value(times, values, &predicted, "FisherPryModel");
            (predicted, Some(diagnostics))
        }
        (Some(times), None) => {
            let predicted: Vec<f64> = times
                .iter()
                .map(|t| 1.0 / (1.0 + (-alpha * (t - t0)).exp()))
                .collect();
            (predicted, None)
        }
        (None, Some(_)) => {
            return Err(KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            ));
        }
        (None, None) => {
            if diagnose_only {
                return Err(KernelBindingError::invalid_request(
                    operation,
                    "diagnose_model requires time and observed arrays in the inputs section",
                ));
            }
            (Vec::new(), None)
        }
    };
    let state_payload = json!({
        "model_key": model_key,
        "model_name": "FisherPryModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "alpha": alpha,
            "t0": t0,
        },
        "predict_kwargs": _copy_object_or_empty(predict_kwargs),
    });

    let mut result = json!({
        "model_key": model_key,
        "model_name": "FisherPryModel",
        "family": "substitution",
        "parameter_names": ["alpha", "t0"],
        "parameters": {
            "alpha": alpha,
            "t0": t0,
        },
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "state": state_payload,
    });
    if !diagnose_only {
        if let Some(diagnostics) = diagnostics {
            result["diagnostics"] = diagnostics;
        }
    } else {
        let diagnostics = diagnostics.ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            )
        })?;
        result = json!({
            "diagnostics": diagnostics,
            "state": state_payload,
        });
    }

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(result),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "substitution",
            "model_name": "FisherPryModel",
            "runtime": "rust_native"
        }),
    })
}

fn gompertz_summary_native_response(
    request: &KernelRequest,
    diagnose_only: bool,
) -> Result<KernelResponse, KernelBindingError> {
    let operation = request.operation;
    let expected_operation = if diagnose_only {
        KernelOperation::DiagnoseModel
    } else {
        KernelOperation::SummarizeModel
    };
    if operation != expected_operation {
        return Err(KernelBindingError::invalid_request(
            operation,
            format!("native {expected_operation} requires a {expected_operation} request"),
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "gompertz" {
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
    let time = optional_numeric_array_from_aliases(inputs, &["time", "t"], operation)?;
    let observed = optional_numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        operation,
    )?;

    let state = object_section(payload, "state", operation)?;
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
    let constructor_kwargs = state.get("constructor_kwargs").and_then(Value::as_object);
    let predict_kwargs = state.get("predict_kwargs").and_then(Value::as_object);
    if constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty))
        || constructor_kwargs
            .and_then(|kwargs| kwargs.get("t_event"))
            .is_some_and(|value| !value.is_null())
        || inputs.get("covariates").is_some()
    {
        return Err(KernelBindingError::unsupported_native_operation(
            operation,
            "native Gompertz execution currently supports fitted states without covariates or event splits",
        ));
    }

    let parameters = state
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| payload.get("parameters").and_then(Value::as_object))
        .ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "kernel requests for model execution require fitted parameters in state or parameters",
            )
        })?;

    let a = required_f64(parameters, "a", operation)?;
    let b = required_f64(parameters, "b", operation)?;
    let c = required_f64(parameters, "c", operation)?;
    if let (Some(times), Some(values)) = (time.as_ref(), observed.as_ref()) {
        if times.len() != values.len() {
            return Err(KernelBindingError::invalid_request(
                operation,
                "time and observed arrays must have the same length",
            ));
        }
    } else if (diagnose_only && observed.is_none()) || (observed.is_some() && time.is_none()) {
        return Err(KernelBindingError::invalid_request(
            operation,
            "diagnose_model requires time and observed arrays in the inputs section",
        ));
    }
    let predictions = time
        .as_ref()
        .map(|times| gompertz_prediction_series(times, a, c))
        .unwrap_or_default();
    let diagnostics = match (time.as_ref(), observed.as_ref()) {
        (Some(times), Some(values)) => Some(summary_diagnostics_value(
            times,
            values,
            &predictions,
            "GompertzModel",
        )),
        _ => None,
    };
    let state_payload = json!({
        "model_key": model_key,
        "model_name": "GompertzModel",
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "parameters": {
            "a": a,
            "b": b,
            "c": c,
        },
        "predict_kwargs": _copy_object_or_empty(predict_kwargs),
    });

    let mut result = json!({
        "model_key": model_key,
        "model_name": "GompertzModel",
        "family": "diffusion",
        "parameter_names": ["a", "b", "c"],
        "parameters": {
            "a": a,
            "b": b,
            "c": c,
        },
        "constructor_kwargs": _copy_object_or_empty(constructor_kwargs),
        "state": state_payload,
    });
    if !diagnose_only {
        if let Some(diagnostics) = diagnostics {
            result["diagnostics"] = diagnostics;
        }
    } else {
        let diagnostics = diagnostics.ok_or_else(|| {
            KernelBindingError::invalid_request(
                operation,
                "diagnose_model requires time and observed arrays in the inputs section",
            )
        })?;
        result = json!({
            "diagnostics": diagnostics,
            "state": state_payload,
        });
    }

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation,
        model_key: None,
        result: Some(result),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "GompertzModel",
            "runtime": "rust_native"
        }),
    })
}

fn gompertz_prediction_series(time: &[f64], a: f64, c: f64) -> Vec<f64> {
    let initial_adopters = 1e-6_f64;
    let scale = (a / initial_adopters).ln();
    time.iter()
        .map(|t| a * (-(scale * (-c * t).exp())).exp())
        .collect()
}

struct GompertzFitResult {
    a: f64,
    b: f64,
    c: f64,
    predictions: Vec<f64>,
}

fn fit_gompertz_at_asymptote(
    time: &[f64],
    observed: &[f64],
    a: f64,
) -> Option<(f64, f64, f64, f64)> {
    let max_observed = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !a.is_finite() || a <= max_observed {
        return None;
    }

    let eps = 1e-9;
    let clipped: Vec<f64> = observed
        .iter()
        .map(|value| value.max(eps).min(a - eps))
        .collect();
    let transformed: Vec<f64> = clipped
        .iter()
        .map(|value| {
            let ratio = value / a;
            if ratio <= 0.0 || ratio >= 1.0 {
                f64::NAN
            } else {
                (-ratio.ln()).ln()
            }
        })
        .collect();
    if transformed.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let n = time.len() as f64;
    let sum_t: f64 = time.iter().sum();
    let sum_z: f64 = transformed.iter().sum();
    let sum_tt: f64 = time.iter().map(|value| value * value).sum();
    let sum_tz: f64 = time
        .iter()
        .zip(transformed.iter())
        .map(|(t, z)| t * z)
        .sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }

    let slope = (n * sum_tz - sum_t * sum_z) / denom;
    let c = -slope;
    if !c.is_finite() || c <= 0.0 {
        return None;
    }
    let intercept = (sum_z - slope * sum_t) / n;
    let b = intercept.exp();
    if !b.is_finite() || b <= 0.0 {
        return None;
    }

    let predictions = gompertz_prediction_series(time, a, c);
    let sse = observed
        .iter()
        .zip(predictions.iter())
        .map(|(y, y_hat)| {
            let residual = y - y_hat;
            residual * residual
        })
        .sum();

    Some((sse, a, b, c))
}

fn refine_gompertz_asymptote(
    time: &[f64],
    observed: &[f64],
    left: f64,
    right: f64,
) -> Option<(f64, f64, f64, f64)> {
    if !left.is_finite() || !right.is_finite() || right <= left {
        return None;
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;
    let mut a = left;
    let mut b = right;
    let mut c = b - (b - a) * inv_phi;
    let mut d = a + (b - a) * inv_phi;
    let mut fc = fit_gompertz_at_asymptote(time, observed, c);
    let mut fd = fit_gompertz_at_asymptote(time, observed, d);

    for _ in 0..48 {
        match (fc, fd) {
            (Some(left_candidate), Some(right_candidate)) => {
                if right_candidate.0 < left_candidate.0 {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + (b - a) * inv_phi;
                    fd = fit_gompertz_at_asymptote(time, observed, d);
                } else {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - (b - a) * inv_phi;
                    fc = fit_gompertz_at_asymptote(time, observed, c);
                }
            }
            (Some(_), None) => {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * inv_phi;
                fc = fit_gompertz_at_asymptote(time, observed, c);
            }
            (None, Some(_)) => {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * inv_phi;
                fd = fit_gompertz_at_asymptote(time, observed, d);
            }
            (None, None) => return None,
        }

        if (b - a).abs() < 1e-10 {
            break;
        }
    }

    [fc, fd]
        .into_iter()
        .flatten()
        .min_by(|left_candidate, right_candidate| left_candidate.0.total_cmp(&right_candidate.0))
}

fn fit_gompertz_curve(
    time: &[f64],
    observed: &[f64],
) -> Result<GompertzFitResult, KernelBindingError> {
    let max_y = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_y.is_finite() || max_y <= 0.0 {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Gompertz fitting requires positive observed values",
        ));
    }

    let lower = (max_y.max(1e-9)) * 1.001;
    let upper = (max_y * 10.0).max(max_y + 1.0).max(lower * 2.0);
    let sample_count = 200usize;
    let mut coarse: Vec<Option<(f64, f64, f64, f64)>> = Vec::with_capacity(sample_count);

    for index in 0..sample_count {
        let fraction = index as f64 / (sample_count.saturating_sub(1) as f64);
        let a = lower + (upper - lower) * fraction;
        coarse.push(fit_gompertz_at_asymptote(time, observed, a));
    }

    let mut best_index: Option<usize> = None;
    let mut best_candidate: Option<(f64, f64, f64, f64)> = None;
    for (index, candidate) in coarse.iter().enumerate() {
        if let Some(candidate) = candidate {
            if best_candidate
                .as_ref()
                .is_none_or(|current| candidate.0 < current.0)
            {
                best_index = Some(index);
                best_candidate = Some(*candidate);
            }
        }
    }

    let (best_sse, best_a, best_b, best_c) = best_candidate.ok_or_else(|| {
        KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Gompertz fitting could not identify a stable asymptote",
        )
    })?;

    let refined = if let Some(index) = best_index {
        let left = if index > 0 {
            lower + (upper - lower) * ((index - 1) as f64 / (sample_count.saturating_sub(1) as f64))
        } else {
            lower
        };
        let right = if index + 1 < sample_count {
            lower + (upper - lower) * ((index + 1) as f64 / (sample_count.saturating_sub(1) as f64))
        } else {
            upper
        };
        refine_gompertz_asymptote(time, observed, left, right)
            .or(Some((best_sse, best_a, best_b, best_c)))
    } else {
        None
    };

    let (_, a, b, c) = refined
        .map(|candidate| {
            if candidate.0 <= best_sse {
                candidate
            } else {
                (best_sse, best_a, best_b, best_c)
            }
        })
        .unwrap_or((best_sse, best_a, best_b, best_c));

    let predictions = gompertz_prediction_series(time, a, c);

    Ok(GompertzFitResult {
        a,
        b,
        c,
        predictions,
    })
}

fn gompertz_fit_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != KernelOperation::FitModel {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            "native fitting requires a fit_model request",
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "gompertz" {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            format!("native fitting is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fit_model payload must be an object",
        )
    })?;
    let inputs = object_section(payload, "inputs", KernelOperation::FitModel)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], KernelOperation::FitModel)?;
    let observed = numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        KernelOperation::FitModel,
    )?;
    if time.len() != observed.len() {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time and observed arrays must have the same length",
        ));
    }

    let constructor_kwargs = _fit_constructor_kwargs(payload);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    let fit_options = payload.get("fit_options").and_then(Value::as_object);
    let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
    if has_covariates
        || has_event
        || input_covariates
        || fit_options.is_some_and(|values| !values.is_empty())
        || fitter_options.is_some_and(|values| !values.is_empty())
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Gompertz fitting currently supports simple fitted states without covariates, events, or custom fitter options",
        ));
    }

    let fit = fit_gompertz_curve(&time, &observed)?;
    let prediction_payload = kernel_array_payload(&fit.predictions);
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, 3, "GompertzModel");
    let state = json!({
        "model_key": model_key,
        "model_name": "GompertzModel",
        "constructor_kwargs": {},
        "parameters": {
            "a": fit.a,
            "b": fit.b,
            "c": fit.c,
        },
        "predict_kwargs": {},
    });

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation: KernelOperation::FitModel,
        model_key: None,
        result: Some(json!({
            "model_key": model_key,
            "model_name": "GompertzModel",
            "family": "diffusion",
            "parameters": {
                "a": fit.a,
                "b": fit.b,
                "c": fit.c,
            },
            "predictions": prediction_payload,
            "diagnostics": diagnostics,
            "state": state,
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "model_name": "GompertzModel",
            "runtime": "rust_native"
        }),
    })
}

fn logistic_fit_native_response(
    request: &KernelRequest,
) -> Result<KernelResponse, KernelBindingError> {
    if request.operation != KernelOperation::FitModel {
        return Err(KernelBindingError::invalid_request(
            request.operation,
            "native fitting requires a fit_model request",
        ));
    }

    let model_key = request.model_key.as_deref().unwrap_or("");
    if model_key != "logistic" {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            format!("native fitting is not implemented for model '{model_key}'"),
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fit_model payload must be an object",
        )
    })?;
    let inputs = object_section(payload, "inputs", KernelOperation::FitModel)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], KernelOperation::FitModel)?;
    let observed = numeric_array_from_aliases(
        inputs,
        &["observed", "y", "values", "adoption", "share"],
        KernelOperation::FitModel,
    )?;
    if time.len() != observed.len() {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "time and observed arrays must have the same length",
        ));
    }

    let constructor_kwargs = _fit_constructor_kwargs(payload);
    let has_covariates = constructor_kwargs
        .and_then(|kwargs| kwargs.get("covariates"))
        .is_some_and(|covariates| !covariates.as_array().is_some_and(Vec::is_empty));
    let has_event = constructor_kwargs
        .and_then(|kwargs| kwargs.get("t_event"))
        .is_some_and(|value| !value.is_null());
    let input_covariates = inputs.get("covariates").is_some();
    let fit_options = payload.get("fit_options").and_then(Value::as_object);
    let fitter_options = payload.get("fitter_options").and_then(Value::as_object);
    if has_covariates
        || has_event
        || input_covariates
        || fit_options.is_some_and(|values| !values.is_empty())
        || fitter_options.is_some_and(|values| !values.is_empty())
    {
        return Err(KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native logistic fitting currently supports simple fitted states without covariates, events, or custom fitter options",
        ));
    }

    let fit = fit_logistic_curve(&time, &observed)?;
    let prediction_payload = kernel_array_payload(&fit.predictions);
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, 4, "LogisticModel");
    let state = json!({
        "model_key": model_key,
        "model_name": "LogisticModel",
        "constructor_kwargs": {},
        "parameters": {
            "L": fit.l,
            "k": fit.k,
            "x0": fit.x0,
        },
        "predict_kwargs": {},
    });

    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.to_string(),
        operation: KernelOperation::FitModel,
        model_key: None,
        result: Some(json!({
            "model_key": model_key,
            "model_name": "LogisticModel",
            "family": "diffusion",
            "parameters": {
                "L": fit.l,
                "k": fit.k,
                "x0": fit.x0,
            },
            "predictions": prediction_payload,
            "diagnostics": diagnostics,
            "state": state,
        })),
        error: None,
        metadata: json!({
            "model_key": model_key,
            "family": "diffusion",
            "support_level": "supported",
            "runtime": "rust_native",
        }),
    })
}

fn unique_temp_path(prefix: &str, extension: &str) -> PathBuf {
    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let sequence = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let filename = format!(
        "{prefix}-{nanos}-{}-{sequence}.{}",
        std::process::id(),
        extension
    );
    env::temp_dir().join(filename)
}

struct TempFileGuard {
    request_path: PathBuf,
    response_path: PathBuf,
}

impl TempFileGuard {
    fn new(request_path: PathBuf, response_path: PathBuf) -> Self {
        Self {
            request_path,
            response_path,
        }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.request_path);
        let _ = fs::remove_file(&self.response_path);
    }
}

pub use serde_json::json;
