//! Validate declarative extension manifests at the shared native boundary.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::sync::{LazyLock, Mutex};

static REGISTRY: LazyLock<Mutex<BTreeMap<String, Manifest>>> = LazyLock::new(Mutex::default);

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
        normalize(manifest)?;
        if !names.insert(manifest.name.as_str()) {
            return Err("extension names must be unique");
        }
    }
    Ok(json!({"manifests":manifests}))
}

fn normalize(manifest: &mut Manifest) -> Result<(), &'static str> {
    if manifest.name.trim().is_empty() || manifest.version.trim().is_empty() {
        return Err("extension name and version must be nonblank");
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
    Ok(())
}

pub(crate) fn register(payload: &Value) -> Result<Value, &'static str> {
    let mut manifest: Manifest = serde_json::from_value(payload["manifest"].clone())
        .map_err(|_| "manifest must be an extension manifest")?;
    normalize(&mut manifest)?;
    let mut registry = REGISTRY
        .lock()
        .map_err(|_| "extension registry unavailable")?;
    if registry.contains_key(&manifest.name) {
        return Err("extension name is already registered");
    }
    let result = json!({"manifest":manifest});
    registry.insert(manifest.name.clone(), manifest);
    Ok(result)
}

pub(crate) fn registered() -> Result<Value, &'static str> {
    let registry = REGISTRY
        .lock()
        .map_err(|_| "extension registry unavailable")?;
    Ok(json!({"manifests":registry.values().collect::<Vec<_>>()}))
}
