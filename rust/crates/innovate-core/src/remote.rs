//! In-process adapter for the remote execution envelope.
use crate::{KERNEL_SCHEMA_VERSION, KernelRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Instant;

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
    let context: Context = serde_json::from_value(payload["context"].clone())
        .map_err(|_| "invalid remote execution context")?;
    if [&context.request_id, &context.tenant_id, &context.principal]
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err("remote execution requires nonblank request, tenant and principal identifiers");
    }
    let raw =
        serde_json::to_string(&payload["kernel_request"]).map_err(|_| "invalid kernel request")?;
    if raw.len() > 1_000_000 {
        return Err("remote kernel request exceeds 1000000 bytes");
    }
    let request: KernelRequest =
        serde_json::from_str(&raw).map_err(|_| "invalid kernel request")?;
    if request.schema_version != KERNEL_SCHEMA_VERSION {
        return Err("remote kernel request requires schema 2.0");
    }
    let operation = &payload["kernel_request"]["operation"];
    let denial = if context.auth_scope != "kernel:execute" {
        Some("remote execution auth_scope does not satisfy the local policy")
    } else if context
        .data_retention
        .as_deref()
        .is_some_and(|retention| retention != "ephemeral")
    {
        Some("local policy requires ephemeral data retention")
    } else if !matches!(
        operation.as_str(),
        Some(
            "discover_models"
                | "predict_model"
                | "simulate_model"
                | "summarize_model"
                | "diagnose_model"
        )
    ) {
        Some("operation is ineligible for remote execution")
    } else {
        None
    };
    let kernel_response: Value = if let Some(message) = denial {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":operation,
            "model_key":request.model_key,"result":null,
            "error":{"code":"unsupported_operation","message":message}})
    } else {
        serde_json::from_str(&crate::execute_json(&raw)).map_err(|_| "invalid native response")?
    };
    let status = if kernel_response["error"].is_null() {
        "ok"
    } else {
        "error"
    };
    Ok(
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"status":status,"kernel_response":kernel_response,
        "provenance":{"execution_location":"in_process","runtime":"rust_native","backend":"cpu",
            "data_retention":"ephemeral","identity_verification":"caller_supplied_context"},
        "observability":{"request_id":context.request_id,"tenant_id":context.tenant_id,
            "principal":context.principal,"trace_id":context.trace_id,"operation":operation,
            "duration_ms":started.elapsed().as_secs_f64()*1000.,"status":status}}),
    )
}
