use serde_json::{Value, json};
use std::{env, fs};

#[test]
fn write_historical_scenario_results() {
    let mut scenarios = vec![json!({
        "id": "logistic", "model_key": "logistic",
        "simulation": {"inputs": {"time": [1.0]},
                       "parameters": {"L": 100.0, "k": 1.0, "x0": 1.0}}
    })];
    for engine in ["kairos_des", "kairos_abm"] {
        for seed in [0, 1, 42, 99, 65535] {
            for q in [0.0, 0.3] {
                scenarios.push(json!({
                    "id": format!("{engine}-{seed}-{q}"),
                    "model_key": "bass",
                    "simulation": {
                        "inputs": {"time": [0.0, 1.0, 5.0]},
                        "parameters": {"p": 0.2, "q": q, "m": 100.0},
                        "simulation_options": {"engine": engine, "seed": seed,
                                               "population": 100, "ticks_per_unit": 1000000}
                    }
                }));
            }
        }
    }
    for (id, edges) in [("chain", json!([[0, 1], [1, 2]])), ("edgeless", json!([]))] {
        scenarios.push(json!({
            "id": id, "model_key": "bass",
            "simulation": {
                "inputs": {"time": [0.0, 5.0, 20.0, 100.0]},
                "parameters": {"p": 0.0, "q": 3.0, "m": 3.0},
                "simulation_options": {"engine": "kairos_network", "seed": 42,
                    "population": 3, "edges": edges, "capacity": 1,
                    "service_ticks": 5, "ticks_per_unit": 1, "initial_agents": [0]}
            }
        }));
    }
    scenarios.push(json!({
        "id": "directed-weighted", "model_key": "bass",
        "simulation": {
            "inputs": {"time": [0.0, 5.0, 20.0]},
            "parameters": {"p": 0.0, "q": 1e12, "m": 3.0},
            "simulation_options": {"engine": "kairos_network", "seed": 42,
                "population": 3, "edges": [[0, 1], [2, 1]], "directed": true,
                "edge_weights": [1e308, 1e308], "capacity": 1, "service_ticks": 5,
                "ticks_per_unit": 1, "initial_agents": [0]}
        }
    }));
    let request = json!({"schema_version": "2.0", "operation": "execute_scenario_grid",
                         "payload": {"include_execution_metadata": true, "scenarios": scenarios}});
    let response: Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null(), "{response}");
    let rows = response["result"]["executions"].as_array().unwrap();
    assert_eq!(rows.len(), 24);
    assert!(rows.iter().all(|row| row["error"].is_null()), "{response}");
    assert_eq!(rows[0]["result"]["values"][0], 50.0);
    fs::write(
        env::var("SCENARIO_RECEIPT_PATH").unwrap(),
        serde_json::to_vec_pretty(&response).unwrap(),
    )
    .unwrap();
}
