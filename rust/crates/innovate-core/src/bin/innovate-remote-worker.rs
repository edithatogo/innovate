//! Single-request subprocess adapter for the native remote envelope.
use serde_json::{Value, json};
use std::io::{self, Read};
use std::process::ExitCode;

const MAX_ENVELOPE_BYTES: u64 = 2_000_000;

fn main() -> ExitCode {
    match dispatch() {
        Ok(response) => {
            println!("{response}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
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
        "schema_version": "2.0",
        "operation": "execute_remote_in_process",
        "payload": envelope,
    });
    Ok(innovate_core::execute_json(&request.to_string()))
}
