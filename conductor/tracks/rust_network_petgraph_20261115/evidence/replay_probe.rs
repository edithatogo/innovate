//! Exact full-response comparison of separately compiled baseline/candidate cores.
//!
//! Compile with rustc --extern baseline=<baseline rlib> --extern candidate=<candidate
//! rlib> --extern serde_json=<matching rlib> -L dependency=<Cargo deps directory>.
//! This is synthetic migration parity, not an independent scientific oracle.
use serde_json::{Value, json};

fn main() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../../rust/crates/innovate-core/tests/fixtures/network_petgraph_20261115.json"
    ))
    .unwrap();
    let seeds = [
        0,
        1,
        2,
        3,
        7,
        15,
        31,
        42,
        63,
        127,
        255,
        1024,
        65535,
        u64::from(u32::MAX),
        u64::MAX - 1,
        u64::MAX,
    ];
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let population = case["population"].as_u64().unwrap();
        for seed in seeds {
            let mut payload = json!({
                "inputs": {"time": [0, 0.001, 0.01, 0.1, 1, 5, 20]},
                "parameters": {"p": 0.07, "q": 1.3, "m": population * 100},
                "simulation_options": {
                    "engine": "kairos_network", "population": population,
                    "seed": seed, "edges": case["edges"], "directed": case["directed"],
                    "capacity": 2, "service_ticks": 3, "ticks_per_unit": 1000,
                    "initial_agents": [seed % population]
                }
            });
            if !case["weights"].is_null() {
                payload["simulation_options"]["edge_weights"] = case["weights"].clone();
            }
            let expected = serde_json::to_value(
                baseline::KernelBinding::new()
                    .simulate_model("bass", payload.clone())
                    .unwrap(),
            )
            .unwrap();
            let actual = serde_json::to_value(
                candidate::KernelBinding::new()
                    .simulate_model("bass", payload)
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(actual, expected, "{} seed {seed}", case["name"]);
            println!(
                "{}",
                json!({"case": case["name"], "seed": seed, "response": actual})
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 160);
    eprintln!("{compared} exact full-response baseline/candidate comparisons passed");
}
