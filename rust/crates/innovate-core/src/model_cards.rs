//! Historical benchmark card descriptions paired with live native discovery.
use serde_json::{Value, json};

pub(crate) fn discover(payload: &Value) -> Result<Value, &'static str> {
    let key = match payload.get("key") {
        None => None,
        Some(value) => Some(
            value
                .as_str()
                .filter(|key| !key.trim().is_empty())
                .ok_or("model card key must be a nonblank string")?,
        ),
    };
    let blueprints: Value = serde_json::from_str(include_str!("model_card_blueprints.json"))
        .expect("embedded model card blueprints");
    let discovery = crate::KernelBinding::new().discover_models_native();
    let cards: Vec<_> = discovery
        .models
        .into_iter()
        .filter(|model| key.is_none_or(|key| model.key == key))
        .map(|model| {
            json!({"model_key":model.key,"native_capability":model,
            "historical_benchmark_card":blueprints[&model.key]})
        })
        .collect();
    if key.is_some() && cards.is_empty() {
        return Err("unknown native model card key");
    }
    Ok(json!({"cards":cards,"benchmark_evidence":"historical descriptions and case IDs"}))
}
