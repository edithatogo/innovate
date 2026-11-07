//! Rust-owned diffusion fitting, prediction, diagnostics and execution contracts.
#![forbid(unsafe_code)]
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
pub use contracts::*;
pub use kernel::KernelBinding;
pub use serde_json::json;
pub const KERNEL_SCHEMA_VERSION: &str = "2.0";

/// JSON transport shared by Python, Julia and R. Errors are data, not panics.
pub fn execute_json(input: &str) -> String {
    let result = serde_json::from_str::<KernelRequest>(input);
    if result.is_err()
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(input)
        && matches!(
            value["operation"].as_str(),
            Some("fit_batch" | "predict_batch")
        )
    {
        return execute_batch_json(value).to_string();
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
        "message":"batch requires schema 2.0, a model key and an array of datasets"},
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
    let binding = KernelBinding::new();
    let results = if operation == "fit_batch" {
        binding.fit_batch(model_key, datasets.iter().cloned())
    } else {
        binding.predict_batch(model_key, datasets.iter().cloned())
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
