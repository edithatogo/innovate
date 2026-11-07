//! Rust-owned diffusion fitting, prediction, diagnostics and execution contracts.
#![forbid(unsafe_code)]
mod competition;
mod contracts;
mod fitting;
mod forcing;
mod hierarchical;
mod historical;
mod kernel;
mod latent;
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
