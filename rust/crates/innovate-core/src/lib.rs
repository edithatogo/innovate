//! Rust-owned diffusion fitting, prediction, diagnostics and execution contracts.
#![forbid(unsafe_code)]
mod adopters;
mod bayesian;
mod benchmark;
mod bootstrap;
mod calendar;
mod calibration;
mod causal;
mod competition;
mod composite;
mod contracts;
mod dataset;
mod ensemble;
mod events;
mod explainability;
mod extensions;
mod oecd;
mod residuals;
mod streaming;
pub use extensions::invoke_registered_extension;
mod fitting;
mod forcing;
mod hierarchical;
mod historical;
mod kernel;
mod latent;
mod mechanistic;
mod mixture;
mod model_cards;
mod network;
mod numerics;
mod outcome_contract;
pub use outcome_contract::validate_paired_outcomes;
mod policy;

mod provenance;
mod regime;
mod remote;
mod report;
mod sarima;
mod scenario;
mod seasonal;
mod sensitivity;
mod simulation;
mod spillover;
mod substitution;
mod validation;
pub use adopters::classify_adopters;
pub use contracts::*;
pub use kernel::KernelBinding;
pub use serde_json::json;
pub const KERNEL_SCHEMA_VERSION: &str = "2.0";

/// JSON transport shared by Python, Julia and R. Errors are data, not panics.
pub fn execute_json(input: &str) -> String {
    let result = serde_json::from_str::<KernelRequest>(input);
    if result.is_err()
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(input)
    {
        // Keep discovery and dispatch derived from the same JSON routes.
        macro_rules! dispatch {
            ($($($operation:literal)|+ => $response:expr),+ $(,)?) => {
                match value["operation"].as_str() {
                    $(Some($($operation)|+) => return $response,)+
                    Some("list_kernel_operations") if value["schema_version"] == KERNEL_SCHEMA_VERSION => {
                        use KernelOperation::*;
                        let typed = [DiscoverModels, FitModel, PredictModel, SimulateModel, SummarizeModel, DiagnoseModel];
                        let operations: Vec<_> = typed.into_iter().map(KernelOperation::as_str)
                            .chain([$($($operation,)+)+ "list_kernel_operations"]).collect();
                        return json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"list_kernel_operations",
                            "model_key":null,"result":{"operations":operations},"error":null,
                            "metadata":{"runtime":"rust_native"}}).to_string();
                    }
                    _ => {}
                }
            };
        }
        dispatch! {
            "fit_batch" | "predict_batch" => execute_batch_json(value).to_string(),
            "fetch_oecd_dataset" => oecd::fetch(&value).to_string(),
            "preprocess_series"
                | "calibrate_prediction_intervals"
                | "aggregate_dates"
                | "summarize_causal"
                | "compose_regime_ensemble"
                | "summarize_competition_equilibrium"
                | "validate_paired_outcomes"
                | "threshold_sensitivity"
                | "parameter_sensitivity"
                | "assumption_sensitivity"
                | "intervention_timing"
                | "summarize_explainability"
                | "combine_report_blocks"
                | "export_decision_report"
                | "export_decision_report_markdown"
                | "discover_model_cards"
                | "discover_execution_modes"
                | "list_advanced_capabilities"
                | "update_streaming_forecast"
                | "validate_provenance"
                | "execute_remote_in_process"
                | "describe_remote_execution_contract"
                | "bootstrap_fit"
                | "bayesian_parameter_grid"
                | "bayesian_metropolis"
                | "bayesian_metropolis_chains"
                | "bayesian_split_rhat"
                | "bayesian_effective_sample_size"
                | "describe_stability_tier"
                | "validate_extension_manifests"
                | "register_extension_manifest"
                | "unregister_extension_manifest"
                | "list_extension_manifests"
                | "resolve_extension_manifest"
                | "summarize_extension_diagnostics"
                | "diagnostics_artifact_tables"
                | "residual_diagnostic_series"
                | "render_residual_diagnostics_svg"
                | "resolve_extension_model_aliases"
                | "dataset_checksum"
                | "build_oecd_dataset_request"
                | "validate_dataset" => execute_preprocessing_json(&value).to_string(),
            "ingest_dataset" => dataset::ingest_request(&value).to_string(),
            "execute_scenario_grid" => scenario::execute_grid(&value).to_string(),
            "discover_benchmark_cases" | "run_benchmark_case" | "run_benchmark_suite" | "validate_benchmark_corpus" => benchmark::execute(&value).to_string(),
            "discover_scenarios" => scenario::discover(&value).to_string(),
            "import_extension_artifact" => extensions::import_serialized_artifact(&value).to_string(),
            "import_scenario_artifact" => scenario::import_artifact(&value).to_string(),
            "categorize_adopters" => execute_adopters_json(&value).to_string(),
        }
    }
    let response = match result {
        Ok(request) => match KernelBinding::new().invoke(&request) {
            Ok(response) => serde_json::to_value(response).expect("serializable response"),
            Err(error) => {
                json!({"schema_version":KERNEL_SCHEMA_VERSION, "operation":request.operation,
                "model_key":request.model_key, "result":null, "error":error, "metadata":{"runtime":"rust_native"}})
            }
        },
        Err(error) => json!({"schema_version":KERNEL_SCHEMA_VERSION, "result":null,
            "error":{"code":"invalid_request","message":error.to_string()}, "metadata":{"runtime":"rust_native"}}),
    };
    response.to_string()
}

fn execute_batch_json(value: serde_json::Value) -> serde_json::Value {
    let operation = &value["operation"];
    let model = &value["model_key"];
    let invalid = || {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":operation,
        "model_key":model,"result":null,"error":{"code":"invalid_request",
        "message":"batch requires schema 2.0, a model key, an array of datasets and optional workers in 1..=32"},
        "metadata":{"runtime":"rust_native"}})
    };
    let Some(model_key) = model.as_str().filter(|key| !key.is_empty()) else {
        return invalid();
    };
    let Some(datasets) = value["payload"]["datasets"].as_array() else {
        return invalid();
    };
    if value["schema_version"] != KERNEL_SCHEMA_VERSION {
        return invalid();
    }
    let Some(workers) = batch_workers(&value["payload"]) else {
        return invalid();
    };
    let binding = KernelBinding::new();
    let run = |batch: &[serde_json::Value]| {
        if operation == "fit_batch" {
            binding.fit_batch(model_key, batch.iter().cloned())
        } else {
            binding.predict_batch(model_key, batch.iter().cloned())
        }
    };
    let results = run_ordered_chunks(datasets, workers, run, |_| {
        Err(KernelBindingError {
            code: "batch_worker_panicked".into(),
            message: "Native batch worker failed".into(),
            operation: Some(if operation == "fit_batch" {
                KernelOperation::FitModel
            } else {
                KernelOperation::PredictModel
            }),
        })
    });
    let rows: Vec<_> = results
        .into_iter()
        .map(|result| match result {
            Ok(response) => json!({"result":response.result,"error":null}),
            Err(error) => json!({"result":null,"error":error}),
        })
        .collect();
    json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":operation,"model_key":model,
        "result":{"datasets":rows},"error":null,"metadata":{"runtime":"rust_native"}})
}

fn execute_adopters_json(value: &serde_json::Value) -> serde_json::Value {
    let result = (|| {
        if value["schema_version"] != KERNEL_SCHEMA_VERSION {
            return Err("categorize_adopters requires schema 2.0");
        }
        let parse = |key: &str| {
            serde_json::from_value::<Vec<f64>>(value["payload"][key].clone())
                .map_err(|_| "categorize_adopters requires numeric time and adoption_rate arrays")
        };
        let time = parse("time")?;
        let rates = parse("adoption_rate")?;
        let labels = classify_adopters(&time, &rates)?;
        Ok(json!({"time":time,"adoption_rate":rates,"category":labels}))
    })();
    let (result, error) = match result {
        Ok(result) => (result, serde_json::Value::Null),
        Err(message) => (
            serde_json::Value::Null,
            json!({"code":"invalid_request","message":message}),
        ),
    };
    json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"categorize_adopters",
        "model_key":null,"result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}

fn execute_preprocessing_json(value: &serde_json::Value) -> serde_json::Value {
    let result = (|| {
        if value["schema_version"] != KERNEL_SCHEMA_VERSION {
            return Err("analysis request requires schema 2.0");
        }
        let payload = &value["payload"];
        if value["operation"] == "update_streaming_forecast" {
            return streaming::update(payload);
        }
        if value["operation"] == "list_advanced_capabilities" {
            let capabilities = [
                (
                    "policy_scenario",
                    "policy",
                    "stable",
                    "summarize_causal",
                    Some("policy_scenario"),
                    false,
                ),
                (
                    "regime_ensemble",
                    "ensemble",
                    "provisional",
                    "compose_regime_ensemble",
                    None,
                    false,
                ),
                (
                    "streaming_update",
                    "streaming",
                    "provisional",
                    "update_streaming_forecast",
                    None,
                    false,
                ),
                (
                    "uncertainty_calibration",
                    "uncertainty",
                    "stable",
                    "calibrate_prediction_intervals",
                    None,
                    true,
                ),
            ]
            .map(
                |(key, family, stability, operation, method, supports_uncertainty)| {
                    json!({
                        "key":key,"family":family,"stability":stability,
                        "operation":operation,"method":method,"result_schema":KERNEL_SCHEMA_VERSION,
                        "optional_dependencies":[],"supported_backends":["rust_native"],
                        "supports_incremental":key == "streaming_update","supports_uncertainty":supports_uncertainty
                    })
                },
            );
            return Ok(json!({"capabilities":capabilities,"pending":[]}));
        }
        if value["operation"] == "discover_execution_modes" {
            return Ok(json!({
                "local_cpu":{"status":"available","max_workers":MAX_BATCH_WORKERS,
                    "operations":["fit_batch","predict_batch","execute_scenario_grid","run_benchmark_suite"],
                    "preserves_input_order":true},
                "benchmark_case_selection":{"operation":"run_benchmark_suite",
                    "catalog_operation":"discover_benchmark_cases",
                    "explicit_case_ids":true,"preserves_case_order":true},
                "accelerator":{"status":"unqualified"},
                "distributed":{"status":"unqualified"},
                "scheduler":{"status":"unqualified"}
            }));
        }
        if value["operation"] == "discover_model_cards" {
            return model_cards::discover(payload);
        }
        if value["operation"] == "export_decision_report_markdown" {
            return report::export_markdown(payload);
        }
        if value["operation"] == "export_decision_report" {
            return report::export(payload);
        }
        if value["operation"] == "combine_report_blocks" {
            return explainability::combine_blocks(payload);
        }
        if value["operation"] == "summarize_explainability" {
            return explainability::summarize(payload);
        }
        if value["operation"] == "threshold_sensitivity" {
            return sensitivity::thresholds(payload);
        }
        if value["operation"] == "assumption_sensitivity" {
            return sensitivity::assumptions(payload);
        }
        if value["operation"] == "intervention_timing" {
            return sensitivity::timing(payload);
        }
        if value["operation"] == "parameter_sensitivity" {
            return sensitivity::perturb_parameters(payload);
        }
        if value["operation"] == "aggregate_dates" {
            return calendar::aggregate(payload);
        }
        if value["operation"] == "validate_dataset" {
            return dataset::validate(payload);
        }
        if value["operation"] == "dataset_checksum" {
            return provenance::checksum(&payload["dataset"], payload.get("encoding"));
        }
        if value["operation"] == "register_extension_manifest" {
            return extensions::register(payload);
        }
        if value["operation"] == "unregister_extension_manifest" {
            return extensions::unregister(payload);
        }
        if value["operation"] == "resolve_extension_model_aliases" {
            return extensions::resolve_model_aliases(&payload["aliases"]);
        }
        if value["operation"] == "render_residual_diagnostics_svg" {
            return residuals::render_svg(payload);
        }
        if value["operation"] == "build_oecd_dataset_request" {
            return oecd::request(payload);
        }
        if value["operation"] == "residual_diagnostic_series" {
            return residuals::residual_series(payload);
        }
        if value["operation"] == "diagnostics_artifact_tables" {
            return diagnostics_artifact_tables(payload);
        }
        if value["operation"] == "summarize_extension_diagnostics" {
            return extensions::summarize_diagnostics(&payload["checks"]);
        }
        if value["operation"] == "resolve_extension_manifest" {
            return extensions::resolve(payload);
        }
        if value["operation"] == "list_extension_manifests" {
            return extensions::registered(payload);
        }
        if value["operation"] == "describe_stability_tier" {
            return extensions::describe_stability(payload);
        }
        if value["operation"] == "validate_extension_manifests" {
            return extensions::validate(payload);
        }
        if value["operation"] == "bayesian_effective_sample_size" {
            return bayesian::effective_sample_size(payload);
        }
        if value["operation"] == "bayesian_split_rhat" {
            return bayesian::split_rhat(payload);
        }
        if value["operation"] == "bayesian_metropolis_chains" {
            return bayesian::sample_chains(payload);
        }
        if value["operation"] == "bayesian_metropolis" {
            return bayesian::sample(payload);
        }
        if value["operation"] == "bayesian_parameter_grid" {
            return bayesian::grid(payload);
        }
        if value["operation"] == "calibrate_prediction_intervals" {
            return calibration::intervals(payload);
        }
        if value["operation"] == "bootstrap_fit" {
            return bootstrap::fit(payload);
        }
        if value["operation"] == "describe_remote_execution_contract" {
            return Ok(remote::describe());
        }
        if value["operation"] == "execute_remote_in_process" {
            return remote::execute(payload);
        }
        if value["operation"] == "validate_provenance" {
            return provenance::validate(payload);
        }
        if value["operation"] == "validate_paired_outcomes" {
            validate_paired_outcomes(payload)?;
            return Ok(payload.clone());
        }

        if value["operation"] == "summarize_competition_equilibrium" {
            return competition::summarize_equilibrium(&value["model_key"], payload);
        }
        if value["operation"] == "compose_regime_ensemble" {
            return ensemble::compose(payload);
        }
        if value["operation"] == "summarize_causal" {
            return causal::summarize(payload);
        }
        let values = serde_json::from_value::<Vec<f64>>(payload["values"].clone())
            .map_err(|_| "preprocess_series requires a numeric values array")?;
        if payload["method"] == "loess_trend" {
            let span = payload["window"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("LOESS requires an integer window")?;
            return Ok(json!({"values":seasonal::loess_trend(&values, span)?}));
        }
        if matches!(
            payload["method"].as_str(),
            Some("sarima_filter" | "sarima_fit")
        ) {
            let order = serde_json::from_value(payload["order"].clone())
                .map_err(|_| "SARIMA order must contain three nonnegative integers")?;
            let seasonal_order = serde_json::from_value(payload["seasonal_order"].clone())
                .map_err(|_| "SARIMA seasonal_order must contain four nonnegative integers")?;
            let coefficients: Option<Vec<f64>> = if payload["method"] == "sarima_fit" {
                None
            } else {
                Some(
                    serde_json::from_value(payload["coefficients"].clone())
                        .map_err(|_| "SARIMA coefficients must be a numeric array")?,
                )
            };
            let historical = payload.get("historical").map_or(Ok(false), |v| {
                v.as_bool().ok_or("historical must be boolean")
            })?;
            return sarima::filter(
                &values,
                order,
                seasonal_order,
                coefficients.as_deref(),
                historical,
            );
        }
        if payload["method"] == "seasonal_decomposition" {
            let integer = |key: &str| {
                payload[key]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("seasonal decomposition requires integer period and window")
            };
            let robust = payload
                .get("robust")
                .map_or(Ok(false), |v| v.as_bool().ok_or("robust must be boolean"))?;
            let historical = payload.get("historical").map_or(Ok(false), |v| {
                v.as_bool().ok_or("historical must be boolean")
            })?;
            let iterations = |key: &str, default, minimum| {
                payload.get(key).map_or(Ok(default), |value| {
                    value.as_u64()
                        .filter(|n| (minimum..=100).contains(n))
                        .map(|n| n as usize)
                        .ok_or("STL iteration counts must be integers within bounds (inner 1–100, outer 0–100)")
                })
            };
            let inner = iterations("inner_iterations", 5, 1)?;
            let outer = iterations("outer_iterations", if robust { 5 } else { 0 }, 0)?;
            if !robust && outer > 0 {
                return Err("positive outer_iterations require robust mode");
            }
            let optional_span = |key: &str| {
                payload.get(key)
                    .filter(|value| !value.is_null()).map(|v| {
                    v.as_u64()
                        .filter(|n| *n >= 3 && *n % 2 == 1 && *n <= i32::MAX as u64)
                        .map(|n| n as usize)
                        .ok_or("seasonal_span and low_pass_span must be odd integers of at least three")
                }).transpose()
            };
            let smoother = |degree, jump| {
                let degree = if payload[degree].is_null() {
                    1
                } else {
                    integer(degree)?
                };
                let jump = if payload[jump].is_null() {
                    1
                } else {
                    integer(jump)?
                };
                if degree > 1 || jump == 0 {
                    return Err("STL degree must be zero or one and jump must be positive");
                }
                Ok((degree, jump))
            };
            let period = integer("period")?;
            let seasonal_span = optional_span("seasonal_span")?;
            let trend_span = if historical && payload["window"].is_null() {
                if period < 2 || period > values.len() / 3 {
                    return Err("seasonal period must leave at least three complete cycles");
                }
                // Historical STL chooses the smallest odd span at or above this ratio.
                let seasonal = seasonal_span.unwrap_or(7) as f64;
                ((1.5 * period as f64 / (1. - 1.5 / seasonal)).ceil() as usize) | 1
            } else {
                integer("window")?
            };
            return seasonal::decompose(
                &values,
                period,
                (trend_span, seasonal_span, optional_span("low_pass_span")?),
                robust,
                historical,
                (inner, outer),
                [
                    smoother("seasonal_degree", "seasonal_jump")?,
                    smoother("trend_degree", "trend_jump")?,
                    smoother("low_pass_degree", "low_pass_jump")?,
                ],
            );
        }
        let window = match payload["method"].as_str() {
            Some("cumulative_sum") => None,
            Some("rolling_average") => Some(
                payload["window"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .ok_or("window must be a positive integer")?,
            ),
            _ => return Err("unknown preprocessing method"),
        };
        let mut sum: f64 = 0.;
        let mut correction = 0.;
        let mut add = |term: f64| {
            let next = sum + term;
            correction += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
            sum + correction
        };
        let rolling_scale = if window.is_some() && values.iter().any(|v| v.abs() > f64::MAX / 2.) {
            0.5
        } else {
            1.
        };
        // Accumulate tiny window terms before rounding them to subnormal units.
        let mut tiny_sum = 0.;
        let mut add_sample = |value: f64, window: Option<usize>| {
            let Some(window) = window else {
                return add(value);
            };
            const SCALE: f64 = (1_u64 << 52) as f64;
            let term = value / window as f64;
            let current = if term.abs() < f64::MIN_POSITIVE {
                tiny_sum += (value * SCALE) / window as f64;
                add(0.)
            } else {
                add(term * rolling_scale)
            };
            // A mean of finite inputs cannot exceed the binary64 range.
            (current / rolling_scale).clamp(-f64::MAX, f64::MAX) + tiny_sum / SCALE
        };
        let mut output = Vec::with_capacity(values.len());
        for (index, value) in values.iter().copied().enumerate() {
            let current = if let Some(window) = window {
                if index >= window {
                    add_sample(-values[index - window], Some(window));
                }
                let current = add_sample(value, Some(window));
                output.push((index >= window - 1).then_some(current));
                current
            } else {
                let current = add_sample(value, None);
                output.push(Some(current));
                current
            };
            if !current.is_finite() {
                return Err("preprocessing result exceeded the finite numerical domain");
            }
        }
        Ok(json!({"values":output}))
    })();
    let (result, error) = match result {
        Ok(result) => (result, serde_json::Value::Null),
        Err(message) => (
            serde_json::Value::Null,
            json!({"code":"invalid_request","message":message}),
        ),
    };
    json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":value["operation"],
        "model_key":null,"result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}

const MAX_BATCH_WORKERS: u64 = 32;

fn batch_workers(payload: &serde_json::Value) -> Option<usize> {
    match payload.get("workers") {
        None => Some(1),
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=MAX_BATCH_WORKERS).contains(n))
            .map(|n| n as usize),
    }
}

fn run_ordered_chunks<T: Sync, R: Send>(
    inputs: &[T],
    workers: usize,
    run: impl Fn(&[T]) -> Vec<R> + Sync,
    worker_failed: impl Fn(&T) -> R,
) -> Vec<R> {
    if workers == 1 || inputs.len() <= 1 {
        return run(inputs);
    }
    std::thread::scope(|scope| {
        let run = &run;
        let jobs: Vec<_> = inputs
            .chunks(inputs.len().div_ceil(workers))
            .map(|chunk| {
                let worker = std::thread::Builder::new().spawn_scoped(scope, move || run(chunk));
                (chunk, worker)
            })
            .collect();
        jobs.into_iter()
            .flat_map(|(chunk, worker)| match worker {
                Ok(worker) => worker
                    .join()
                    .unwrap_or_else(|_| chunk.iter().map(&worker_failed).collect()),
                // This chunk never started and can safely execute inline.
                Err(_) => run(chunk),
            })
            .collect()
    })
}

fn diagnostics_artifact_tables(
    payload: &serde_json::Value,
) -> Result<serde_json::Value, &'static str> {
    let artifacts = payload["artifacts"]
        .as_object()
        .ok_or("artifacts must be an object")?;
    let mut tables = serde_json::Map::new();
    for (name, artifact) in artifacts {
        let (Some(columns), Some(rows)) =
            (artifact["columns"].as_array(), artifact["rows"].as_array())
        else {
            continue;
        };
        if columns.is_empty() {
            continue;
        }
        let columns = columns
            .iter()
            .map(|column| column.as_str().ok_or("column names must be strings"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows: Vec<Vec<_>> = rows
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|&column| row.get(column).cloned().unwrap_or_default())
                    .collect()
            })
            .collect();
        tables.insert(name.clone(), json!({"columns":columns,"rows":rows,"metadata":{
            "diagnostics_artifact":name,"diagnostics_artifact_kind":artifact["kind"].as_str().unwrap_or_default(),
            "diagnostics_artifact_schema_version":payload["schema_version"],
            "model_name":payload["model_name"],"support_level":payload["support_level"]
        }}));
    }
    Ok(json!({"tables":tables}))
}
