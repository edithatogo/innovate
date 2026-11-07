use crate::{KERNEL_SCHEMA_VERSION, KernelBinding};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    model_key: String,
    simulation: Value,
}

pub(crate) fn execute_grid(request: &Value) -> Value {
    let envelope = |result: Value, error: Value| {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"execute_scenario_grid",
            "result":result,"error":error,"metadata":{"runtime":"rust_native"}})
    };
    let invalid = |message: &str| {
        envelope(
            Value::Null,
            json!({"code":"invalid_request","message":message}),
        )
    };
    if request["schema_version"] != KERNEL_SCHEMA_VERSION {
        return invalid("execute_scenario_grid requires schema 2.0");
    }
    let scenarios: Vec<Scenario> =
        match serde_json::from_value(request["payload"]["scenarios"].clone()) {
            Ok(scenarios) => scenarios,
            Err(error) => return invalid(&format!("invalid scenario grid: {error}")),
        };
    let mut ids = HashSet::new();
    if scenarios.iter().any(|scenario| {
        scenario.id.trim().is_empty()
            || scenario.model_key.trim().is_empty()
            || !ids.insert(&scenario.id)
    }) {
        return invalid(
            "scenario IDs must be nonblank and unique, and model keys must be nonblank",
        );
    }
    let binding = KernelBinding::new();
    let executions: Vec<_> = scenarios
        .into_iter()
        .map(|scenario| {
            let execution = binding.simulate_model(&scenario.model_key, scenario.simulation);
            let (result, error) = match execution {
                Ok(response) => (response.result, Value::Null),
                Err(error) => (None, json!(error)),
            };
            json!({"id":scenario.id,"model_key":scenario.model_key,"result":result,"error":error})
        })
        .collect();
    envelope(json!({"executions":executions}), Value::Null)
}
