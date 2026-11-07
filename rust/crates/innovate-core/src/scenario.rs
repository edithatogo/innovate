use crate::{KERNEL_SCHEMA_VERSION, KernelBinding};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    model_key: String,
    simulation: Value,
    specification: Option<ScenarioSpecification>,
}

#[derive(Deserialize, Serialize)]
struct ScenarioSpecification {
    name: String,
    description: String,
    time_horizon: f64,
    time_unit: String,
    reference_year: i64,
    market_size: f64,
    initial_adoption: f64,
    #[serde(flatten)]
    kind: ScenarioKind,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "scenario_type", rename_all = "snake_case", deny_unknown_fields)]
enum ScenarioKind {
    Baseline,
    Intervention {
        intervention_type: String,
        intervention_start_time: f64,
        intervention_magnitude: f64,
    },
    Substitution {
        incumbent_name: String,
        entrant_name: String,
        substitution_rate: f64,
    },
    Competition {
        num_competitors: usize,
        competitor_names: Vec<String>,
        market_share_initial: Vec<f64>,
    },
    Network {
        network_type: String,
        num_nodes: usize,
        average_degree: f64,
    },
}

impl ScenarioSpecification {
    fn validate(&self) -> Result<(), &'static str> {
        if self.time_horizon <= 0. || self.market_size <= 0. {
            return Err("time_horizon and market_size must be positive");
        }
        if !(0. ..=1.).contains(&self.initial_adoption) {
            return Err("initial_adoption must be in [0, 1]");
        }
        match &self.kind {
            ScenarioKind::Intervention {
                intervention_start_time,
                ..
            } if *intervention_start_time < 0. => {
                Err("intervention_start_time must be nonnegative")
            }
            ScenarioKind::Substitution {
                substitution_rate, ..
            } if !(0. ..=1.).contains(substitution_rate) => {
                Err("substitution_rate must be in [0, 1]")
            }
            ScenarioKind::Competition {
                num_competitors,
                competitor_names,
                market_share_initial,
            } if competitor_names.len() != *num_competitors
                || market_share_initial.len() != *num_competitors
                || market_share_initial
                    .iter()
                    .any(|share| !(0. ..=1.).contains(share))
                || (market_share_initial.iter().sum::<f64>() - 1.).abs() > 1e-6 =>
            {
                Err(
                    "competition names and shares must match num_competitors and shares must sum to one",
                )
            }
            ScenarioKind::Network {
                num_nodes,
                average_degree,
                ..
            } if *num_nodes == 0 || *average_degree <= 0. => {
                Err("num_nodes and average_degree must be positive")
            }
            _ => Ok(()),
        }
    }
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
    for scenario in &scenarios {
        if let Some(specification) = &scenario.specification
            && let Err(message) = specification.validate()
        {
            return invalid(&format!("scenario '{}': {message}", scenario.id));
        }
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
            let mut row = json!({"id":scenario.id,"model_key":scenario.model_key,"result":result,"error":error});
            if let Some(specification) = scenario.specification {
                row["specification"] = json!(specification);
            }
            row
        })
        .collect();
    envelope(json!({"executions":executions}), Value::Null)
}
