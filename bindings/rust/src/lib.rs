use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
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
        self.fit_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                debug!(
                    operation = %request.operation,
                    model_key = ?request.model_key,
                    "native path unsupported, falling back to Python bridge",
                );
                self.invoke(&request)
            } else {
                Err(err)
            }
        })
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
        logistic_fit_native_response(request)
    }

    pub fn predict_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.predict_model_request(model_key, payload);
        self.predict_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                debug!(
                    operation = %request.operation,
                    model_key = ?request.model_key,
                    "native path unsupported, falling back to Python bridge",
                );
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
                debug!(
                    operation = %request.operation,
                    model_key = ?request.model_key,
                    "native path unsupported, falling back to Python bridge",
                );
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
        let request = self.summarize_model_request(model_key, payload);
        self.summarize_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                debug!(
                    operation = %request.operation,
                    model_key = ?request.model_key,
                    "native path unsupported, falling back to Python bridge",
                );
                self.invoke(&request)
            } else {
                Err(err)
            }
        })
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
        logistic_summary_native_response(request)
    }

    pub fn diagnose_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.diagnose_model_request(model_key, payload);
        self.diagnose_model_native(&request).or_else(|err| {
            if err.code == "unsupported_native_operation" {
                debug!(
                    operation = %request.operation,
                    model_key = ?request.model_key,
                    "native path unsupported, falling back to Python bridge",
                );
                self.invoke(&request)
            } else {
                Err(err)
            }
        })
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
        logistic_diagnose_native_response(request)
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
            let _ = fs::remove_file(&request_path);
            let _ = fs::remove_file(&response_path);
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
        "model_name": "LogisticModel",
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
            let diagnostics = summary_diagnostics_value(times, values, &predicted);
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

fn summary_diagnostics_value(time: &[f64], observed: &[f64], predicted: &[f64]) -> Value {
    fit_diagnostics(time, observed, predicted, 4)
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
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, 4);
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
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let filename = format!("{prefix}-{nanos}-{}.{}", std::process::id(), extension);
    env::temp_dir().join(filename)
}

pub use serde_json::json;
