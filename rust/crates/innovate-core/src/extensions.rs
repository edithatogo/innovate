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
        || manifest
            .extension_points
            .iter()
            .any(|point| !recognized_point(point))
    {
        return Err("extension points must be nonempty and recognized");
    }
    Ok(())
}

fn recognized_point(point: &str) -> bool {
    matches!(
        point,
        "dataset_provider" | "diagnostics" | "model_registry" | "serialization_adapter"
    )
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

pub(crate) fn registered(payload: &Value) -> Result<Value, &'static str> {
    let registry = REGISTRY
        .lock()
        .map_err(|_| "extension registry unavailable")?;
    let point = payload
        .get("extension_point")
        .map(|point| {
            point
                .as_str()
                .filter(|point| recognized_point(point))
                .ok_or("unknown extension point")
        })
        .transpose()?;
    let manifests: Vec<_> = registry
        .values()
        .filter(|manifest| {
            point.is_none_or(|point| manifest.extension_points.iter().any(|value| value == point))
        })
        .collect();
    Ok(json!({"manifests":manifests}))
}

pub(crate) fn unregister(payload: &Value) -> Result<Value, &'static str> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Removal {
        name: String,
        expected_version: String,
    }

    let removal = Removal::deserialize(payload)
        .map_err(|_| "extension removal requires name and expected_version")?;
    if removal.name.trim().is_empty() || removal.expected_version.trim().is_empty() {
        return Err("extension removal requires nonblank name and expected_version");
    }
    let mut registry = REGISTRY
        .lock()
        .map_err(|_| "extension registry unavailable")?;
    match registry.entry(removal.name) {
        std::collections::btree_map::Entry::Occupied(entry) => {
            if entry.get().version != removal.expected_version {
                return Err("registered extension version does not match expected_version");
            }
            Ok(json!({"removed":true,"manifest":entry.remove()}))
        }
        std::collections::btree_map::Entry::Vacant(_) => {
            Ok(json!({"removed":false,"manifest":null}))
        }
    }
}

pub(crate) fn resolve(payload: &Value) -> Result<Value, &'static str> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Resolution {
        name: String,
        expected_version: String,
        extension_point: String,
    }

    let request = Resolution::deserialize(payload)
        .map_err(|_| "extension resolution requires name, expected_version and extension_point")?;
    if !recognized_point(&request.extension_point) {
        return Err("unknown extension point");
    }
    let registry = REGISTRY
        .lock()
        .map_err(|_| "extension registry unavailable")?;
    let manifest = registry
        .get(&request.name)
        .ok_or("extension is not registered")?;
    if manifest.version != request.expected_version {
        return Err("registered extension version does not match expected_version");
    }
    if !manifest.extension_points.contains(&request.extension_point) {
        return Err("extension does not declare the requested extension point");
    }
    Ok(json!({"manifest":manifest,"extension_point":request.extension_point}))
}

/// Resolve a registered extension and pass its manifest snapshot to a trusted host callback.
///
/// The registry lock is released before the callback runs. Registration changes
/// after resolution do not invalidate the snapshot. The host selects the callable
/// and owns its input and output contract.
pub fn invoke_registered_extension(
    selection: &Value,
    input: &Value,
    callback: impl FnOnce(&Value, &Value) -> Result<Value, &'static str>,
) -> Result<Value, &'static str> {
    let resolved = resolve(selection)?;
    callback(&resolved, input)
}
