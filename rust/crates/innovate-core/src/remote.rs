//! In-process adapter for the remote execution envelope.
use crate::{KERNEL_SCHEMA_VERSION, KernelRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Instant;

const REQUIRED_SCOPE: &str = "kernel:execute";
const MAX_KERNEL_REQUEST_BYTES: usize = 1_000_000;
const MAX_CONTEXT_IDENTIFIER_BYTES: usize = 256;
const DATA_RETENTION: &str = "ephemeral";
const ENVELOPE_FIELDS: &[&str] = &["schema_version", "context", "kernel_request"];
const KERNEL_REQUEST_FIELDS: &[&str] = &[
    "schema_version",
    "operation",
    "model_key",
    "payload",
    "metadata",
];
const ELIGIBLE_OPERATIONS: &[&str] = &[
    "discover_models",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
];

pub(crate) fn describe() -> Value {
    json!({"schema_version":KERNEL_SCHEMA_VERSION,"execution_location":"in_process",
        "runtime":"rust_native","backend":"cpu","required_auth_scope":REQUIRED_SCOPE,
        "identity_verification":"caller_supplied_context","data_retention":DATA_RETENTION,
        "max_kernel_request_bytes":MAX_KERNEL_REQUEST_BYTES,
        "max_context_identifier_bytes":MAX_CONTEXT_IDENTIFIER_BYTES,
        "envelope_fields":ENVELOPE_FIELDS,"kernel_request_fields":KERNEL_REQUEST_FIELDS,
        "eligible_operations":ELIGIBLE_OPERATIONS,"local_only_by_default":["fit_model"],
        "observability_fields":["request_id","tenant_id","principal","trace_id","operation","duration_ms","status"]})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Context {
    request_id: String,
    tenant_id: String,
    principal: String,
    auth_scope: String,
    #[serde(default)]
    trace_id: String,
    data_retention: Option<String>,
}

pub(crate) fn execute(payload: &Value) -> Result<Value, &'static str> {
    let started = Instant::now();
    if payload["schema_version"] != KERNEL_SCHEMA_VERSION {
        return Err("remote envelope requires schema 2.0");
    }
    if payload.as_object().is_none_or(|fields| {
        fields
            .keys()
            .any(|key| !ENVELOPE_FIELDS.contains(&key.as_str()))
    }) {
        return Err("remote envelope contains unsupported fields");
    }
    let context: Context = serde_json::from_value(payload["context"].clone())
        .map_err(|_| "invalid remote execution context")?;
    if [&context.request_id, &context.tenant_id, &context.principal]
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err("remote execution requires nonblank request, tenant and principal identifiers");
    }
    if [
        &context.request_id,
        &context.tenant_id,
        &context.principal,
        &context.trace_id,
    ]
    .iter()
    .any(|value| value.len() > MAX_CONTEXT_IDENTIFIER_BYTES || value.chars().any(char::is_control))
    {
        return Err(
            "remote execution identifiers must fit in 256 bytes and exclude control characters",
        );
    }
    let kernel_request = &payload["kernel_request"];
    let fields = kernel_request.as_object().ok_or("invalid kernel request")?;
    if fields
        .keys()
        .any(|key| !KERNEL_REQUEST_FIELDS.contains(&key.as_str()))
    {
        return Err("remote kernel request contains unsupported fields");
    }
    let raw = bounded_kernel_json(kernel_request)?;
    let request: KernelRequest =
        serde_json::from_slice(&raw).map_err(|_| "invalid kernel request")?;
    if request.schema_version != KERNEL_SCHEMA_VERSION {
        return Err("remote kernel request requires schema 2.0");
    }
    let operation = &kernel_request["operation"];
    let denial = if context.auth_scope != REQUIRED_SCOPE {
        Some("remote execution auth_scope does not satisfy the local policy")
    } else if context
        .data_retention
        .as_deref()
        .is_some_and(|retention| retention != DATA_RETENTION)
    {
        Some("local policy requires ephemeral data retention")
    } else if !operation
        .as_str()
        .is_some_and(|name| ELIGIBLE_OPERATIONS.contains(&name))
    {
        Some("operation is ineligible for remote execution")
    } else {
        None
    };
    let kernel_response: Value = if let Some(message) = denial {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":operation,
            "model_key":request.model_key,"result":null,
            "error":{"code":"unsupported_operation","message":message}})
    } else {
        serde_json::from_str(&crate::execute_json(
            std::str::from_utf8(&raw).map_err(|_| "invalid kernel request")?,
        ))
        .map_err(|_| "invalid native response")?
    };
    let status = if kernel_response["error"].is_null() {
        "ok"
    } else {
        "error"
    };
    Ok(
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"status":status,"kernel_response":kernel_response,
        "provenance":{"execution_location":"in_process","runtime":"rust_native","backend":"cpu",
            "data_retention":DATA_RETENTION,"identity_verification":"caller_supplied_context"},
        "observability":{"request_id":context.request_id,"tenant_id":context.tenant_id,
            "principal":context.principal,"trace_id":context.trace_id,"operation":operation,
            "duration_ms":started.elapsed().as_secs_f64()*1000.,"status":status}}),
    )
}

fn bounded_kernel_json(request: &Value) -> Result<Vec<u8>, &'static str> {
    struct Buffer(Vec<u8>);

    impl std::io::Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > MAX_KERNEL_REQUEST_BYTES - self.0.len() {
                return Err(std::io::ErrorKind::FileTooLarge.into());
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut buffer = Buffer(Vec::new());
    serde_json::to_writer(&mut buffer, request)
        .map_err(|_| "remote kernel request exceeds 1000000 bytes")?;
    Ok(buffer.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_encoded_json_including_escape_bytes() {
        let exact = json!("\n".repeat((MAX_KERNEL_REQUEST_BYTES - 2) / 2));
        assert_eq!(
            bounded_kernel_json(&exact).unwrap().len(),
            MAX_KERNEL_REQUEST_BYTES
        );
        let oversized = json!("\n".repeat(MAX_KERNEL_REQUEST_BYTES / 2));
        assert!(bounded_kernel_json(&oversized).is_err());
    }
}
