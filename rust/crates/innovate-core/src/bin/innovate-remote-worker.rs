//! Single-request subprocess adapter for the native remote envelope.
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
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
    let UniqueJson(envelope) =
        serde_json::from_slice(&input).map_err(|_| "invalid remote envelope JSON")?;
    let request = json!({
        "schema_version": innovate_core::KERNEL_SCHEMA_VERSION,
        "operation": "execute_remote_in_process",
        "payload": envelope,
    });
    Ok(innovate_core::execute_json(&request.to_string()))
}

// Preserve JSON values while rejecting ambiguous object keys before dispatch.
struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueJsonVisitor).map(Self)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JSON with unique object keys")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Value, M::Error> {
        let mut object = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(M::Error::custom("duplicate JSON object key"));
            }
            object.insert(key, map.next_value::<UniqueJson>()?.0);
        }
        Ok(Value::Object(object))
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Value, S::Error> {
        let mut array = Vec::new();
        while let Some(UniqueJson(value)) = seq.next_element()? {
            array.push(value);
        }
        Ok(Value::Array(array))
    }

    fn visit_unit<E: Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::from(value))
    }
}
