//! Python-free native CLI for `fit` and `predict` requests (R07).

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn parse_args() -> Result<(String, PathBuf, PathBuf), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 || !matches!(args[0].as_str(), "fit" | "predict") {
        return Err("usage: innovate-native <fit|predict> --request FILE --output FILE".into());
    }
    let subcommand = args[0].clone();
    let mut request_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--request" if i + 1 < args.len() => {
                request_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => return Err(format!("unknown or incomplete argument: {other}")),
        }
    }
    Ok((
        subcommand,
        request_path.ok_or_else(|| "missing --request FILE".to_string())?,
        output_path.ok_or_else(|| "missing --output FILE".to_string())?,
    ))
}

fn main() -> ExitCode {
    let (subcommand, request_path, output_path) = match parse_args() {
        Ok(parsed) => parsed,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let raw = match fs::read_to_string(&request_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("failed to read {}: {err}", request_path.display());
            return ExitCode::from(1);
        }
    };
    let mut value: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("malformed JSON request: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(obj) = value.as_object_mut() else {
        eprintln!("request JSON must be an object");
        return ExitCode::from(1);
    };
    let expected_op = if subcommand == "fit" {
        "fit_model"
    } else {
        "predict_model"
    };
    if let Some(op) = obj.get("operation").and_then(|v| v.as_str())
        && op != expected_op
    {
        eprintln!("subcommand '{subcommand}' does not allow operation; expected '{expected_op}'");
        return ExitCode::from(1);
    }
    obj.entry("schema_version")
        .or_insert_with(|| serde_json::json!(innovate_core::KERNEL_SCHEMA_VERSION));
    obj.insert("operation".into(), serde_json::json!(expected_op));
    let response_str = innovate_core::execute_json(&value.to_string());
    if let Err(err) = fs::write(&output_path, &response_str) {
        eprintln!("failed to write {}: {err}", output_path.display());
        return ExitCode::from(1);
    }
    let parsed: serde_json::Value = match serde_json::from_str(&response_str) {
        Ok(v) => v,
        Err(_) => return ExitCode::from(1),
    };
    if !parsed["error"].is_null() {
        eprintln!("{}", parsed["error"]);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
