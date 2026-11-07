//! Validate declarative extension manifests at the shared native boundary.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    name: String,
    version: String,
    entrypoint: String,
    stability: String,
    extension_points: Vec<String>,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    requires: Vec<String>,
}

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    let mut manifests: Vec<Manifest> = serde_json::from_value(payload["manifests"].clone())
        .map_err(|_| "manifests must be an array of extension manifests")?;
    let mut names = HashSet::new();
    for manifest in &mut manifests {
        if manifest.name.trim().is_empty() || manifest.version.trim().is_empty() {
            return Err("extension name and version must be nonblank");
        }
        if !names.insert(manifest.name.as_str()) {
            return Err("extension names must be unique");
        }
        if !manifest
            .entrypoint
            .split_once(':')
            .is_some_and(|(module, callable)| {
                !module.trim().is_empty() && !callable.trim().is_empty() && !callable.contains(':')
            })
        {
            return Err("extension entrypoint must have module:callable form");
        }
        manifest.stability = match manifest.stability.trim().to_ascii_lowercase().as_str() {
            "stable" => "stable",
            "beta" | "experimental" | "provisional" => "provisional",
            "internal" | "internal-only" => "internal",
            _ => return Err("unknown extension stability tier"),
        }
        .into();
        if manifest.extension_points.is_empty()
            || manifest.extension_points.iter().any(|point| {
                !matches!(
                    point.as_str(),
                    "dataset_provider" | "diagnostics" | "model_registry" | "serialization_adapter"
                )
            })
        {
            return Err("extension points must be nonempty and recognized");
        }
    }
    Ok(json!({"manifests":manifests}))
}
