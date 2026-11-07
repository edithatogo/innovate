//! Rust-owned diffusion fitting, prediction, diagnostics and execution contracts.
#![forbid(unsafe_code)]
mod adopters;
mod competition;
mod composite;
mod contracts;
mod events;
mod fitting;
mod forcing;
mod hierarchical;
mod historical;
mod kernel;
mod latent;
mod mechanistic;
mod mixture;
mod network;
mod numerics;
mod policy;
mod regime;
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
            Some("preprocess_series") => return execute_preprocessing_json(&value).to_string(),
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
    let workers = match value["payload"].get("workers") {
        None => 1,
        Some(workers) => match workers.as_u64().filter(|n| (1..=32).contains(n)) {
            Some(n) => n as usize,
            None => return invalid(),
        },
    };
    let binding = KernelBinding::new();
    let run = |batch: &[serde_json::Value]| {
        if operation == "fit_batch" {
            binding.fit_batch(model_key, batch.iter().cloned())
        } else {
            binding.predict_batch(model_key, batch.iter().cloned())
        }
    };
    let results = if workers == 1 || datasets.len() <= 1 {
        run(datasets)
    } else {
        std::thread::scope(|scope| {
            let chunks = datasets.chunks(datasets.len().div_ceil(workers));
            let jobs: Vec<_> = chunks
                .map(|chunk| {
                    let worker =
                        std::thread::Builder::new().spawn_scoped(scope, move || run(chunk));
                    (chunk, worker)
                })
                .collect();
            jobs.into_iter()
                .flat_map(|(chunk, worker)| match worker {
                    Ok(worker) => worker.join().unwrap_or_else(|_| {
                        let operation = if operation == "fit_batch" {
                            KernelOperation::FitModel
                        } else {
                            KernelOperation::PredictModel
                        };
                        (0..chunk.len())
                            .map(|_| {
                                Err(KernelBindingError {
                                    code: "batch_worker_panicked".into(),
                                    message: "Native batch worker failed".into(),
                                    operation: Some(operation),
                                })
                            })
                            .collect()
                    }),
                    // Thread creation can fail under process resource limits.
                    // This chunk has not started, so it is safe to run inline.
                    Err(_) => run(chunk),
                })
                .collect()
        })
    };
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
            return Err("preprocess_series requires schema 2.0");
        }
        let payload = &value["payload"];
        let values = serde_json::from_value::<Vec<f64>>(payload["values"].clone())
            .map_err(|_| "preprocess_series requires a numeric values array")?;
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
    json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"preprocess_series",
        "model_key":null,"result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}
