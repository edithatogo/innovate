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
