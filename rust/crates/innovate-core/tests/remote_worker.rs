use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_innovate-remote-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn worker_returns_native_results_and_policy_denials() {
    let mut envelope = json!({
        "schema_version":"2.0",
        "context":{"request_id":"r","tenant_id":"t","principal":"p","auth_scope":"kernel:execute","trace_id":"trace-1"},
        "kernel_request":{"schema_version":"2.0","operation":"predict_model","model_key":"logistic",
            "payload":{"inputs":{"time":[1]},"parameters":{"L":100,"k":1,"x0":1}}}
    });
    let output = run(envelope.to_string().as_bytes());
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    let observability = &response["result"]["observability"];
    assert_eq!(observability["request_id"], "r");
    assert_eq!(observability["tenant_id"], "t");
    assert_eq!(observability["principal"], "p");
    assert_eq!(observability["trace_id"], "trace-1");
    assert_eq!(observability["status"], "ok");
    assert_eq!(
        response["result"]["kernel_response"]["result"]["values"],
        json!([50.])
    );

    envelope["context"]["auth_scope"] = json!("read");
    let output = run(envelope.to_string().as_bytes());
    assert!(output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["result"]["status"], "error");
    assert_eq!(response["result"]["observability"]["trace_id"], "trace-1");
    assert_eq!(response["result"]["observability"]["status"], "error");
    assert_eq!(
        response["result"]["kernel_response"]["error"]["code"],
        "unsupported_operation"
    );
}

#[test]
fn worker_rejects_invalid_framing_without_writing_a_response() {
    for (input, diagnostic) in [
        (b"{".to_vec(), "invalid remote envelope JSON"),
        (
            vec![b' '; 2_000_001],
            "remote envelope exceeds 2000000 bytes",
        ),
    ] {
        let output = run(&input);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(String::from_utf8(output.stderr).unwrap().trim(), diagnostic);
    }
}

#[test]
fn worker_reports_a_closed_response_pipe() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_innovate-remote-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take().unwrap());
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(output.stderr).unwrap().trim(),
        "could not write remote response"
    );
}

#[test]
fn worker_describes_its_transport_without_reading_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_innovate-remote-worker"))
        .arg("--describe")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    let contract = &response["result"];
    assert_eq!(contract["worker_transport"], "stdin_stdout");
    assert_eq!(contract["max_raw_envelope_bytes"], 2_000_000);
    assert_eq!(contract["max_kernel_request_bytes"], 1_000_000);
    assert_eq!(contract["identity_verification"], "caller_supplied_context");
}

#[test]
fn worker_rejects_invalid_context_identifiers() {
    for (field, value) in [
        ("request_id", " ".to_owned()),
        ("tenant_id", "é".repeat(129)),
        ("principal", "spoof\nrecord".to_owned()),
        ("trace_id", "trace\tspoof".to_owned()),
    ] {
        let mut envelope = json!({
            "schema_version":"2.0",
            "context":{"request_id":"r","tenant_id":"t","principal":"p","auth_scope":"kernel:execute"},
            "kernel_request":{"schema_version":"2.0","operation":"discover_models"}
        });
        envelope["context"][field] = json!(value);
        let output = run(envelope.to_string().as_bytes());
        assert!(output.status.success());
        let response: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(response["result"].is_null(), "{field}");
        assert_eq!(response["error"]["code"], "invalid_request", "{field}");
    }
}
