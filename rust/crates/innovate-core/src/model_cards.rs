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
            "historical_benchmark_card":blueprints[&model.key],
                "native_model_card":native_description(&model.key)})
        })
        .collect();
    if key.is_some() && cards.is_empty() {
        return Err("unknown native model card key");
    }
    Ok(json!({"cards":cards,"benchmark_evidence":"historical descriptions and case IDs"}))
}

fn native_description(key: &str) -> Option<Value> {
    let (summary, limitation, reference) = match key {
        "network_diffusion" => (
            "Models deterministic adoption spillovers across linked groups.",
            "This model describes group trajectories. Agent-level adoption and resource queues use the separate Kairos network engine.",
            "network_diffusion",
        ),
        "policy_hazard" => (
            "Applies a time-varying policy multiplier to a baseline diffusion path.",
            "Policy effects are scenario inputs. The model does not identify a causal effect.",
            "policy",
        ),
        "latent_process" => (
            "Smooths fitted residuals to represent a latent adoption path.",
            "The native forecast uses a deterministic residual path without stochastic draws.",
            "latent",
        ),
        "regime_switching" => (
            "Fits piecewise diffusion curves with historical-index or fixed-time regimes.",
            "Forecasts are conditional on the fitted or supplied regime structure.",
            "regime",
        ),
        "hierarchical" => (
            "Fits grouped diffusion paths and aggregates their predicted levels.",
            "Grouped point fits do not perform partial pooling or Bayesian inference.",
            "hierarchical",
        ),
        "mixture" => (
            "Combines component trajectories using native mixture weights.",
            "Native simulation uses deterministic weighted paths.",
            "mixture",
        ),
        _ => return None,
    };
    Some(json!({"summary":summary,"limitations":[limitation],
        "native_test_target":reference,"benchmark_case_ids":match key { "network_diffusion" => vec!["network_diffusion_smoke", "network_spillover_reference"], "policy_hazard" => vec!["policy_hazard_reference", "policy_hazard_fit"], "latent_process" => vec!["latent_process_reference", "latent_process_fit"], "regime_switching" => vec!["regime_switching_reference"], _ => vec![] }}))
}
