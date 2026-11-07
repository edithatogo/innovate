use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize, Serialize)]
struct DatasetProvenance {
    source: String,
    license: String,
    extraction_time: String,
    #[serde(default)]
    transform_steps: Vec<String>,
    #[serde(default = "schema_version")]
    schema_version: String,
    #[serde(default)]
    checksum: String,
    #[serde(default)]
    citation: String,
    #[serde(default)]
    extra: BTreeMap<String, Value>,
}

fn schema_version() -> String {
    "1.0".into()
}

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    if !payload.is_object() {
        return Err("dataset provenance must be an object");
    }
    let mut provenance: DatasetProvenance =
        serde_json::from_value(payload.clone()).map_err(|_| "invalid dataset provenance fields")?;
    for value in [
        &mut provenance.source,
        &mut provenance.license,
        &mut provenance.extraction_time,
        &mut provenance.schema_version,
    ]
    .into_iter()
    .chain(&mut provenance.transform_steps)
    {
        *value = value.trim().to_owned();
        if value.is_empty() {
            return Err(
                "provenance source, license, extraction time, schema and transform steps must be nonempty",
            );
        }
    }
    if matches!(
        provenance.license.to_ascii_lowercase().as_str(),
        "unknown" | "n/a" | "na" | "none" | "unspecified"
    ) {
        return Err("dataset provenance requires an explicit license");
    }
    if provenance.schema_version != "1.0" {
        return Err("unsupported dataset provenance schema version");
    }
    Ok(json!(provenance))
}

pub(crate) fn checksum(payload: &Value) -> Result<Value, &'static str> {
    use sha2::{Digest, Sha256};

    let mut content = payload["dataset"].clone();
    if !content.is_object() {
        return Err("dataset checksum requires a dataset object");
    }
    let expected = if let Some(provenance) = content.get_mut("provenance") {
        validate(provenance)?;
        let record = provenance
            .as_object_mut()
            .expect("validated provenance object");
        record.remove("extraction_time");
        record.remove("checksum")
    } else {
        None
    };
    let expected = expected
        .as_ref()
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty());
    if expected.is_some_and(|digest| {
        digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        return Err("dataset provenance checksum must be a 64-digit SHA-256 hex string");
    }
    content.sort_all_objects();
    let encoding = payload
        .get("encoding")
        .map_or(Some("rust_sorted_json_v1"), Value::as_str);
    let (bytes, encoding) = match encoding {
        Some("rust_sorted_json_v1") => (
            serde_json::to_vec(&content).map_err(|_| "dataset checksum serialization failed")?,
            "rust_sorted_json_v1",
        ),
        Some("python_sorted_json_v1") => {
            let mut output = String::new();
            python_json(&content, &mut output);
            (output.into_bytes(), "python_sorted_json_v1")
        }
        _ => return Err("unsupported dataset checksum encoding"),
    };
    let actual = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if expected.is_some_and(|digest| !actual.eq_ignore_ascii_case(digest)) {
        return Err("dataset content checksum does not match provenance");
    }
    Ok(json!({"checksum":actual,"algorithm":"sha256","encoding":encoding}))
}

fn python_json(value: &Value, output: &mut String) {
    use std::fmt::Write as _;

    match value {
        Value::String(text) => {
            for unit in json!(text).to_string().encode_utf16() {
                if unit < 127 {
                    output.push(char::from_u32(u32::from(unit)).expect("ASCII code unit"));
                } else {
                    write!(output, "\\u{unit:04x}").expect("writing to String");
                }
            }
        }
        Value::Number(number) if number.is_f64() => {
            let number = number.as_f64().expect("floating-point number");
            let repr = format!("{number:?}");
            if let Some((mantissa, _)) = repr.split_once('e') {
                let precision = mantissa
                    .split_once('.')
                    .map_or(0, |(_, fraction)| fraction.len());
                let rounded = format!("{number:.precision$e}");
                let (mantissa, exponent) = rounded.split_once('e').expect("scientific notation");
                let exponent: i32 = exponent.parse().expect("formatted exponent");
                write!(output, "{mantissa}e{exponent:+03}").expect("writing to String");
            } else if let Some((_, fraction)) = repr.split_once('.') {
                // Fixed precision rounds exact decimal ties to even, as CPython does.
                let rounded = format!("{number:.precision$}", precision = fraction.len());
                output.push_str(&rounded);
            } else {
                output.push_str(&repr);
            }
        }
        Value::Array(values) => {
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                python_json(item, output);
            }
            output.push(']');
        }
        Value::Object(fields) => {
            output.push('{');
            for (index, (key, item)) in fields.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                python_json(&json!(key), output);
                output.push(':');
                python_json(item, output);
            }
            output.push('}');
        }
        _ => output.push_str(&value.to_string()),
    }
}
