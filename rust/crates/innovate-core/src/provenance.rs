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
    content.sort_all_objects();
    let bytes =
        serde_json::to_vec(&content).map_err(|_| "dataset checksum serialization failed")?;
    let actual = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if let Some(expected) = expected
        .as_ref()
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        && !actual.eq_ignore_ascii_case(expected)
    {
        return Err("dataset content checksum does not match provenance");
    }
    Ok(json!({"checksum":actual,"algorithm":"sha256","encoding":"rust_sorted_json_v1"}))
}
