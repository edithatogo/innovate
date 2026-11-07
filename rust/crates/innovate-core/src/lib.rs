//! Rust-owned diffusion fitting, prediction, diagnostics and execution contracts.
#![forbid(unsafe_code)]
mod adopters;
mod benchmark;
mod calendar;
mod causal;
mod competition;
mod composite;
mod contracts;
mod dataset;
mod events;
mod explainability;
mod extensions;
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
mod policy;
mod provenance;
mod regime;
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
        match value["operation"].as_str() {
            Some("fit_batch" | "predict_batch") => return execute_batch_json(value).to_string(),
            Some(
                "preprocess_series"
                | "aggregate_dates"
                | "summarize_causal"
                | "threshold_sensitivity"
                | "parameter_sensitivity"
                | "assumption_sensitivity"
                | "intervention_timing"
                | "summarize_explainability"
                | "combine_report_blocks"
                | "export_decision_report"
                | "export_decision_report_markdown"
                | "discover_model_cards"
                | "validate_provenance"
                | "validate_extension_manifests"
                | "register_extension_manifest"
                | "list_extension_manifests"
                | "dataset_checksum"
                | "validate_dataset",
            ) => {
                return execute_preprocessing_json(&value).to_string();
            }
            Some("ingest_dataset") => return dataset::ingest_request(&value).to_string(),
            Some("execute_scenario_grid") => return scenario::execute_grid(&value).to_string(),
            Some(
                "discover_benchmark_cases"
                | "run_benchmark_case"
                | "run_benchmark_suite"
                | "validate_benchmark_corpus",
            ) => {
                return benchmark::execute(&value).to_string();
            }
            Some("discover_scenarios") => return scenario::discover(&value).to_string(),
            Some("import_scenario_artifact") => {
                return scenario::import_artifact(&value).to_string();
            }
            Some("categorize_adopters") => return execute_adopters_json(&value).to_string(),
            _ => {}
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
            return provenance::checksum(payload);
        }
        if value["operation"] == "register_extension_manifest" {
            return extensions::register(payload);
        }
        if value["operation"] == "list_extension_manifests" {
            return extensions::registered();
        }
        if value["operation"] == "validate_extension_manifests" {
            return extensions::validate(payload);
        }
        if value["operation"] == "validate_provenance" {
            return provenance::validate(payload);
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
        let mut output = Vec::with_capacity(values.len());
        for (index, value) in values.iter().copied().enumerate() {
            let current = if let Some(window) = window {
                if index >= window {
                    add(-values[index - window] / window as f64);
                }
                let current = add(value / window as f64);
                output.push((index >= window - 1).then_some(current));
                current
            } else {
                let current = add(value);
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

fn batch_workers(payload: &serde_json::Value) -> Option<usize> {
    match payload.get("workers") {
        None => Some(1),
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=32).contains(n))
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
