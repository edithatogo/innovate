use crate::KERNEL_SCHEMA_VERSION;
use crate::contracts::*;
use crate::validation::*;
use serde_json::{Map, Value, json};
const DISCOVERY_MANIFEST_JSON: &str = include_str!("discovery.json");

#[derive(Debug, Default, Clone, Copy)]
pub struct KernelBinding;

impl KernelBinding {
    pub fn new() -> Self {
        Self
    }

    pub fn schema_version(&self) -> &'static str {
        KERNEL_SCHEMA_VERSION
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

    pub fn discover_models(&self) -> Result<KernelDiscoveryResponse, KernelBindingError> {
        Ok(self.discover_models_native())
    }

    pub fn fit_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.fit_model_request(model_key, payload);
        self.invoke(&request)
    }

    /// Fits independent datasets in input order, retaining each fit's state or error.
    pub fn fit_batch(
        &self,
        model_key: &str,
        datasets: impl IntoIterator<Item = Value>,
    ) -> Vec<Result<KernelResponse, KernelBindingError>> {
        datasets
            .into_iter()
            .map(|payload| self.fit_model(model_key, payload))
            .collect()
    }

    /// Predicts independently for each payload and its fitted state.
    pub fn predict_batch(
        &self,
        model_key: &str,
        datasets: impl IntoIterator<Item = Value>,
    ) -> Vec<Result<KernelResponse, KernelBindingError>> {
        datasets
            .into_iter()
            .map(|payload| self.predict_model(model_key, payload))
            .collect()
    }

    fn fit_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        if request.model_key.as_deref() == Some("norton_bass") {
            return crate::substitution::execute(request);
        }
        let (initial, max_iterations) = crate::fitting::scalar_options(request)?;
        let mut seed_request = request.clone();
        if let Some(payload) = seed_request.payload.as_object_mut() {
            payload.remove("fit_options");
        }
        let seed = if let Some(initial) = initial {
            crate::fitting::seed_response(&seed_request, initial)?
        } else {
            match request.model_key.as_deref().unwrap_or("") {
                "gompertz" => gompertz_fit_native_response(&seed_request),
                "logistic" => logistic_fit_native_response(&seed_request),
                "bass" => bass_fit_native_response(&seed_request),
                "fisher_pry" => fisher_pry_fit_native_response(&seed_request),
                model_key => Err(KernelBindingError::unsupported_native_operation(
                    KernelOperation::FitModel,
                    format!("native fitting is not implemented for model '{model_key}'"),
                )),
            }?
        };
        crate::fitting::refine(request, seed, max_iterations)
    }

    pub fn predict_model(
        &self,
        model_key: impl Into<String>,
        payload: Value,
    ) -> Result<KernelResponse, KernelBindingError> {
        let request = self.predict_model_request(model_key, payload);
        self.invoke(&request)
    }

    fn predict_model_native(
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
        self.invoke(&request)
    }

    fn simulate_model_native(
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
        self.invoke(&request)
    }

    fn summarize_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        match request.model_key.as_deref().unwrap_or("") {
            "bass" => bass_summary_native_response(request),
            "gompertz" => gompertz_summary_native_response(request, false),
            "logistic" => logistic_summary_native_response(request),
            "norton_bass" => crate::substitution::execute(request),
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
        self.invoke(&request)
    }

    fn diagnose_model_native(
        &self,
        request: &KernelRequest,
    ) -> Result<KernelResponse, KernelBindingError> {
        match request.model_key.as_deref().unwrap_or("") {
            "bass" => bass_diagnose_native_response(request),
            "gompertz" => gompertz_summary_native_response(request, true),
            "logistic" => logistic_diagnose_native_response(request),
            "norton_bass" => crate::substitution::execute(request),
            "fisher_pry" => fisher_pry_summary_native_response(request, true),
            model_key => Err(KernelBindingError::unsupported_native_operation(
                KernelOperation::DiagnoseModel,
                format!("native diagnose_model is not implemented for model '{model_key}'"),
            )),
        }
    }
}

impl KernelBinding {
    /// Dispatch only within Rust. No process, interpreter or compatibility fallback.
    pub fn invoke(&self, request: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
        if request.schema_version != KERNEL_SCHEMA_VERSION {
            return Err(KernelBindingError::invalid_request(
                request.operation,
                "unsupported schema version",
            ));
        }
        if request.operation != KernelOperation::DiscoverModels
            && request.model_key.as_deref().unwrap_or("").is_empty()
        {
            return Err(KernelBindingError::invalid_request(
                request.operation,
                "model_key is required",
            ));
        }
        // All models accept either explicit parameters or a fitted state. Resolve
        // the transport once in Rust so every binding has the same semantics.
        let mut normalized;
        let request = if request.operation != KernelOperation::FitModel
            && request.operation != KernelOperation::DiscoverModels
            && request.payload.get("parameters").is_some()
        {
            if request.payload.get("state").is_some() {
                return Err(KernelBindingError::invalid_request(
                    request.operation,
                    "provide parameters or state, not both",
                ));
            }
            normalized = request.clone();
            let payload = normalized.payload.as_object_mut().ok_or_else(|| {
                KernelBindingError::invalid_request(request.operation, "payload must be an object")
            })?;
            let parameters = payload.remove("parameters").expect("checked above");
            payload.insert("state".into(), json!({
                "model_key": request.model_key,
                "parameters": parameters,
                "constructor_kwargs": payload.get("constructor_kwargs").or_else(|| payload.get("model_kwargs")).cloned().unwrap_or_else(|| json!({})),
                "predict_kwargs": payload.get("predict_kwargs").cloned().unwrap_or_else(|| json!({}))
            }));
            &normalized
        } else {
            request
        };
        if crate::mechanistic::selected(request) {
            return crate::mechanistic::execute(request);
        }
        if crate::historical::selected(request) {
            return crate::historical::execute(request);
        }
        if request.model_key.as_deref() == Some("network_diffusion") {
            return crate::spillover::execute(request);
        }
        if request.model_key.as_deref() == Some("policy_hazard") {
            return crate::policy::execute(request);
        }
        if request.model_key.as_deref() == Some("latent_process") {
            return crate::latent::execute(request);
        }
        if request.model_key.as_deref() == Some("regime_switching") {
            return crate::regime::execute(request);
        }
        if request.model_key.as_deref() == Some("hierarchical") {
            return crate::hierarchical::execute(request);
        }
        if request.model_key.as_deref() == Some("mixture") {
            return crate::mixture::execute(request);
        }
        if request.model_key.as_deref() == Some("composite") {
            return crate::composite::execute(request);
        }
        validate_request(request)?;
        if request.payload.get("simulation_options").is_some() {
            return crate::simulation::execute(request);
        }
        if matches!(
            request.model_key.as_deref(),
            Some("lotka_volterra" | "multi_product" | "complementary_goods" | "lock_in")
        ) {
            return crate::competition::execute(request);
        }
        match request.operation {
            KernelOperation::DiscoverModels => Ok(KernelResponse {
                schema_version: KERNEL_SCHEMA_VERSION.into(),
                operation: request.operation,
                model_key: None,
                result: Some(
                    serde_json::to_value(self.discover_models_native()).expect("embedded manifest"),
                ),
                error: None,
                metadata: json!({"runtime":"rust_native"}),
            }),
            KernelOperation::FitModel => self.fit_model_native(request),
            KernelOperation::PredictModel => self.predict_model_native(request),
            KernelOperation::SimulateModel => self.simulate_model_native(request),
            KernelOperation::SummarizeModel => self.summarize_model_native(request),
            KernelOperation::DiagnoseModel => self.diagnose_model_native(request),
        }
    }
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

fn fit_diagnostics(time: &[f64], observed: &[f64], predicted: &[f64], model_name: &str) -> Value {
    fit_diagnostics_count(
        time,
        observed,
        predicted,
        model_name,
        if model_name == "FisherPryModel" { 3 } else { 4 },
    )
}

pub(crate) fn fit_diagnostics_count(
    time: &[f64],
    observed: &[f64],
    predicted: &[f64],
    model_name: &str,
    parameter_count: usize,
) -> Value {
    // Gaussian residual variance is estimated in addition to structural parameters.
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
        observed
            .iter()
            .map(|v| v / observed.len() as f64)
            .sum::<f64>()
    };
    let ss_tot: f64 = observed
        .iter()
        .map(|value| {
            let delta = value - mean_observed;
            delta * delta
        })
        .sum();
    let r_squared = if ss_tot > 0.0 && ss_tot.is_finite() {
        1.0 - (ss_res / ss_tot)
    } else {
        f64::NAN
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
        "information_criteria": {
            "assumption": "iid_gaussian_residuals",
            "parameter_count_including_variance": parameter_count,
            "convention": "n_log_rss_over_n_plus_penalty; common_gaussian_constant_omitted",
            "undefined_when_zero_rss": true
        },
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
            "breusch_pagan_p": null,
            "shapiro_wilk_p": null,
            "significance_tests": "not_implemented",
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

    json!({
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "mean_residual": mean_residual,
        "max_abs_residual": max_abs_residual,
        "std_residual": std_residual,
        "durbin_watson": durbin_watson,
        "breusch_pagan_p": null,
        "shapiro_wilk_p": null,
        "significance_tests": "not_implemented",
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
    fit_diagnostics(time, observed, predicted, model_name)
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

    let lower = max_y + max_y.abs().max(1.0) * 1e-12;
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
        if let Some(candidate) = candidate
            && best_candidate
                .as_ref()
                .is_none_or(|current| candidate.0 < current.0)
        {
            best_index = Some(index);
            best_candidate = Some(*candidate);
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
        "norton_bass" => crate::substitution::execute(request),
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
            format!(
                "native {operation} requires a fit, predict, simulate, summarize, or diagnose request"
            ),
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
    let b = required_f64(parameters, "b", operation)?;
    let c = required_f64(parameters, "c", operation)?;
    let predictions = gompertz_prediction_series(&time, a, b, c);

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

    let stable_payload = if operation == KernelOperation::SimulateModel {
        DeterministicSimulationPayload::from_value(model_key, &request.payload)?.fitted_state
    } else {
        BoundedFittedStatePayload::from_value(model_key, operation, &request.payload)?
    };
    if stable_payload
        .time
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(KernelBindingError::invalid_request(
            operation,
            "time values must be finite and non-negative for native Bass execution",
        ));
    }

    let payload = request.payload.as_object().ok_or_else(|| {
        KernelBindingError::invalid_request(
            operation,
            format!("{operation} payload must be an object"),
        )
    })?;
    let parameters = payload
        .get("parameters")
        .and_then(Value::as_object)
        .or_else(|| stable_payload.state.parameters.as_object())
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

    let predictions: Vec<f64> = stable_payload
        .time
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

    // Fit the continuous-time curve directly. The retired difference regression
    // ignored time spacing and could reject valid data near saturation.
    let y_scale = observed.iter().copied().fold(0.0_f64, f64::max);
    let t_scale = time.iter().copied().fold(0.0_f64, f64::max);
    if y_scale <= 0.0 || t_scale <= 0.0 {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "Bass fitting needs positive adoption and a positive time horizon",
        ));
    }
    let times: Vec<f64> = time.iter().map(|t| t / t_scale).collect();
    let observations: Vec<f64> = observed.iter().map(|y| y / y_scale).collect();
    let evaluate = |theta: [f64; 3]| {
        bass_prediction_series(&times, theta[0].exp(), theta[1].exp(), theta[2].exp())
    };
    let loss = |values: &[f64]| {
        values
            .iter()
            .zip(&observations)
            .map(|(prediction, observation)| (prediction - observation).powi(2))
            .sum::<f64>()
    };
    let mut best: Option<(f64, [f64; 3])> = None;
    // Deterministic starts cover innovation-led and imitation-led trajectories.
    for initial_p in [0.01_f64, 0.1, 1.0] {
        for initial_q in [0.1_f64, 1.0, 10.0] {
            let mut theta = [initial_p.ln(), initial_q.ln(), 1.1_f64.ln()];
            let mut prediction = evaluate(theta);
            let mut error = loss(&prediction);
            let mut damping = 1e-3_f64;
            for _ in 0..200 {
                let mut jacobian = vec![[0.0; 3]; times.len()];
                for parameter in 0..3 {
                    let step = 1e-5 * theta[parameter].abs().max(1.0);
                    let mut plus = theta;
                    let mut minus = theta;
                    plus[parameter] += step;
                    minus[parameter] -= step;
                    let upper = evaluate(plus);
                    let lower = evaluate(minus);
                    for (i, row) in jacobian.iter_mut().enumerate() {
                        row[parameter] = (upper[i] - lower[i]) / (2.0 * step);
                    }
                }
                let mut normal = [[0.0; 3]; 3];
                let mut rhs = [0.0; 3];
                for (i, row) in jacobian.iter().enumerate() {
                    let residual = prediction[i] - observations[i];
                    for a in 0..3 {
                        rhs[a] -= row[a] * residual;
                        for b in 0..3 {
                            normal[a][b] += row[a] * row[b];
                        }
                    }
                }
                for (i, row) in normal.iter_mut().enumerate() {
                    row[i] += damping * row[i].max(1e-8);
                }
                let Some(delta) = solve_3x3_system(normal, rhs) else {
                    damping *= 10.0;
                    if damping > 1e12 {
                        break;
                    }
                    continue;
                };
                let candidate = std::array::from_fn(|i| {
                    (theta[i] + delta[i].clamp(-5.0, 5.0)).clamp(-30.0, 30.0)
                });
                let candidate_prediction = evaluate(candidate);
                let candidate_error = loss(&candidate_prediction);
                if candidate_error.is_finite() && candidate_error < error {
                    let improvement = error - candidate_error;
                    theta = candidate;
                    prediction = candidate_prediction;
                    error = candidate_error;
                    damping = (damping * 0.25).max(1e-12);
                    if error < 1e-20 || improvement < 1e-14 * (1.0 + error) {
                        break;
                    }
                } else {
                    damping *= 10.0;
                    if damping > 1e12 {
                        break;
                    }
                }
            }
            if error.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|(best_error, _)| error < *best_error)
            {
                best = Some((error, theta));
            }
        }
    }
    let (_, theta) = best.ok_or_else(|| {
        KernelBindingError::unsupported_native_operation(
            KernelOperation::FitModel,
            "native Bass optimizer could not obtain a finite solution",
        )
    })?;
    let p = theta[0].exp() / t_scale;
    let q = theta[1].exp() / t_scale;
    let m = theta[2].exp() * y_scale;
    let predictions = bass_prediction_series(time, p, q, m);
    if !p.is_finite()
        || !q.is_finite()
        || !m.is_finite()
        || predictions.iter().any(|x| !x.is_finite())
    {
        return Err(KernelBindingError::invalid_request(
            KernelOperation::FitModel,
            "fitted parameters exceed the finite numerical domain",
        ));
    }
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

    let stable_payload = SimplePositiveObservationsFitPayload::from_value(
        model_key,
        KernelOperation::FitModel,
        &request.payload,
    )?;
    let constructor_kwargs = stable_payload
        .constructor_kwargs
        .as_object()
        .filter(|values| !values.is_empty());

    let fit = fit_bass_curve(&stable_payload.time, &stable_payload.observed)?;
    let prediction_payload = kernel_array_payload(&fit.predictions);
    let diagnostics = fit_diagnostics(
        &stable_payload.time,
        &stable_payload.observed,
        &fit.predictions,
        "BassModel",
    );
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
    if diagnose_only {
        let _diagnostics_payload =
            DeterministicDiagnosticsPayload::from_value(model_key, &request.payload)?;
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
            format!(
                "native {operation} requires a fit, predict, simulate, summarize, or diagnose request"
            ),
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
    let diagnostics = fit_diagnostics(&time, &observed, &predictions, "FisherPryModel");
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
        .map(|times| gompertz_prediction_series(times, a, b, c))
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

fn gompertz_prediction_series(time: &[f64], a: f64, b: f64, c: f64) -> Vec<f64> {
    time.iter()
        .map(|t| a * (-(b * (-c * t).exp())).exp())
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

    let predictions = gompertz_prediction_series(time, a, b, c);
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

    let lower = max_y + max_y.abs().max(1.0) * 1e-12;
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
        if let Some(candidate) = candidate
            && best_candidate
                .as_ref()
                .is_none_or(|current| candidate.0 < current.0)
        {
            best_index = Some(index);
            best_candidate = Some(*candidate);
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

    let predictions = gompertz_prediction_series(time, a, b, c);

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
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, "GompertzModel");
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
    let diagnostics = fit_diagnostics(&time, &observed, &fit.predictions, "LogisticModel");
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
