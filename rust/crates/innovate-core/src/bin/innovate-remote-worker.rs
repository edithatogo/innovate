//! Single-request subprocess adapter for the native remote envelope.
use serde_json::{Value, json};
use std::io::{self, Read, Write};
use std::process::ExitCode;

const MAX_ENVELOPE_BYTES: u64 = 2_000_000;

fn main() -> ExitCode {
    match response().and_then(|response| {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{response}")
            .and_then(|()| stdout.flush())
            .map_err(|_| "could not write remote response")
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn response() -> Result<String, &'static str> {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next()) {
        (None, None) => dispatch(),
        (Some(flag), None) if flag == "--describe" => {
            let request = json!({
                "schema_version": innovate_core::KERNEL_SCHEMA_VERSION,
                "operation": "describe_remote_execution_contract",
            });
            let mut response: Value =
                serde_json::from_str(&innovate_core::execute_json(&request.to_string()))
                    .map_err(|_| "invalid native contract response")?;
            response["result"]["worker_transport"] = json!("stdin_stdout");
            response["result"]["max_raw_envelope_bytes"] = json!(MAX_ENVELOPE_BYTES);
            Ok(response.to_string())
        }
        _ => Err("usage: innovate-remote-worker [--describe]"),
    }
}

fn dispatch() -> Result<String, &'static str> {
    let mut input = Vec::new();
    io::stdin()
        .take(MAX_ENVELOPE_BYTES + 1)
        .read_to_end(&mut input)
        .map_err(|_| "could not read remote envelope")?;
    if input.len() as u64 > MAX_ENVELOPE_BYTES {
        return Err("remote envelope exceeds 2000000 bytes");
    }
    let envelope: Value =
        serde_json::from_slice(&input).map_err(|_| "invalid remote envelope JSON")?;
    let request = json!({
        "schema_version": innovate_core::KERNEL_SCHEMA_VERSION,
        "operation": "execute_remote_in_process",
        "payload": envelope,
    });
    Ok(innovate_core::execute_json(&request.to_string()))
}
