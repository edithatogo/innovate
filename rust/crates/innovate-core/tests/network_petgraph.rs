//! Synthetic graph-contract checks for the bounded petgraph migration.
//!
//! Expected event times below are hand-calculated for a single exposed agent at
//! each step: its request takes one tick, and service takes two ticks. The fixed
//! large finite hazard makes the sampled waiting times round up to one tick for
//! these seeds. No empirical or whole-model qualification is asserted.
use innovate_core::{KernelBinding, json};
use serde_json::Value;

fn chain_payload(seed: u64) -> Value {
    json!({
        "inputs": {"time": [0, 1, 3, 4, 6, 20]},
        "parameters": {"p": 0, "q": 1e12, "m": 6},
        "simulation_options": {
            "engine": "kairos_network", "population": 6, "seed": seed,
            "edges": [[0, 1], [1, 2], [3, 4]], "directed": true,
            "capacity": 1, "service_ticks": 2, "ticks_per_unit": 1,
            "initial_agents": [0]
        }
    })
}

fn run(payload: Value) -> Value {
    KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap()
}

#[test]
fn network_petgraph_directed_chain_preserves_analytical_times_and_disconnection() {
    for seed in [0, 42, u64::MAX] {
        let result = run(chain_payload(seed));
        assert_eq!(result["counts"], json!([1, 1, 2, 2, 3, 3]), "seed {seed}");
        assert_eq!(
            result["requests"],
            json!([{"tick": 1, "agent": 1}, {"tick": 4, "agent": 2}]),
            "seed {seed}"
        );
        assert_eq!(
            result["completions"],
            json!([{"tick": 3, "agent": 1}, {"tick": 6, "agent": 2}]),
            "seed {seed}"
        );
        for index in 3..6 {
            assert_eq!(result["agents"][index]["index"], json!(index));
            assert_eq!(result["agents"][index]["state"], "susceptible");
        }
        assert_eq!(result["metadata"]["population"], 6);
        assert_eq!(result["metadata"]["kairos_dispatched_events"], 4);
    }
}

#[test]
fn network_petgraph_edge_reversal_respects_directedness() {
    let mut reversed = chain_payload(42);
    reversed["simulation_options"]["edges"] = json!([[1, 0], [2, 1], [4, 3]]);
    let result = run(reversed.clone());
    assert_eq!(result["counts"], json!([1, 1, 1, 1, 1, 1]));
    assert_eq!(result["requests"], json!([]));
    assert_eq!(result["completions"], json!([]));

    // Endpoint reversal leaves each destination's input-relative incidence order
    // intact for an undirected graph, including asymmetric normalization scales.
    let mut original = chain_payload(42);
    for payload in [&mut original, &mut reversed] {
        payload["simulation_options"]["directed"] = json!(false);
        payload["simulation_options"]["edge_weights"] = json!([2, 8, 16]);
    }
    let expected = run(original);
    assert_eq!(expected["counts"], json!([1, 1, 2, 2, 3, 3]));
    assert_eq!(run(reversed), expected);
}

#[test]
fn network_petgraph_scheduled_graph_replacement_survives_tied_resume_boundaries() {
    let mut payload = chain_payload(42);
    payload["inputs"]["time"] = json!([0, 1, 3, 4, 6, 7, 9, 12]);
    payload["simulation_options"]["edges"] = json!([[0, 1]]);
    payload["simulation_options"]["edge_weights"] = json!([1e308]);
    payload["simulation_options"]["schedule"] = json!([
        {"tick": 3, "edges": [[1, 2], [3, 4]], "edge_weights": [2, 8]},
        {"tick": 6, "edges": [[2, 5]]},
        {"tick": 9, "edges": [], "edge_weights": []}
    ]);
    let expected = run(payload.clone());
    assert_eq!(expected["counts"], json!([1, 1, 2, 2, 3, 3, 4, 4]));
    assert_eq!(
        expected["requests"],
        json!([
            {"tick": 1, "agent": 1},
            {"tick": 4, "agent": 2},
            {"tick": 7, "agent": 5}
        ])
    );
    assert_eq!(
        expected["completions"],
        json!([
            {"tick": 3, "agent": 1},
            {"tick": 6, "agent": 2},
            {"tick": 9, "agent": 5}
        ])
    );
    assert_eq!(
        expected["metadata"]["applied_schedule_ticks"],
        json!([3, 6, 9])
    );
    assert_eq!(expected["metadata"]["weighted"], true);
    for (option, value, applied) in [
        ("pause_at_tick", 2, json!([])),
        ("pause_after_completions", 1, json!([])),
        ("pause_at_tick", 3, json!([3])),
        ("pause_after_completions", 2, json!([3])),
        ("pause_at_tick", 6, json!([3, 6])),
        ("pause_at_tick", 9, json!([3, 6, 9])),
    ] {
        let mut request = payload.clone();
        request["simulation_options"][option] = json!(value);
        let paused = run(request.clone());
        assert_eq!(paused["status"], "paused");
        let checkpoint: Value =
            serde_json::from_str(paused["checkpoint"].as_str().unwrap()).unwrap();
        assert_eq!(checkpoint["applied_ticks"], applied, "{option}={value}");
        let options = request["simulation_options"].as_object_mut().unwrap();
        options.remove(option);
        options.insert("resume_checkpoint".into(), paused["checkpoint"].clone());
        assert_eq!(run(request), expected, "{option}={value}");
    }
}

#[test]
fn network_petgraph_empty_weighted_graph_retains_public_metadata() {
    for weights in [None, Some(json!([]))] {
        let mut payload = chain_payload(42);
        payload["simulation_options"]["edges"] = json!([]);
        if let Some(weights) = &weights {
            payload["simulation_options"]["edge_weights"] = weights.clone();
        }
        let result = run(payload);
        assert_eq!(result["counts"], json!([1, 1, 1, 1, 1, 1]));
        assert_eq!(result["requests"], json!([]));
        assert_eq!(result["metadata"]["weighted"], weights.is_some());
    }
}

#[test]
fn network_petgraph_rejects_invalid_graphs_and_future_replacements() {
    let edge_error = "edges must be unique non-self pairs within population";
    let weight_error = "edge_weights must contain one finite positive weight per edge";
    let scheduled_error = "invalid scheduled tick, graph";
    let cases = [
        ("self_loop", json!({"edges": [[0, 0]]}), edge_error),
        ("out_of_range", json!({"edges": [[0, 6]]}), edge_error),
        (
            "directed_duplicate",
            json!({"edges": [[0, 1], [0, 1]]}),
            edge_error,
        ),
        (
            "undirected_reverse_duplicate",
            json!({"directed": false, "edges": [[0, 1], [1, 0]]}),
            edge_error,
        ),
        ("weight_count", json!({"edge_weights": []}), weight_error),
        (
            "zero_weight",
            json!({"edge_weights": [0, 1, 1]}),
            weight_error,
        ),
        (
            "negative_weight",
            json!({"edge_weights": [-1, 1, 1]}),
            weight_error,
        ),
        (
            "weight_validation_precedes_edges",
            json!({"edges": [[0, 0]], "edge_weights": [0]}),
            weight_error,
        ),
        (
            "future_out_of_range",
            json!({"schedule": [{"tick": 100, "edges": [[0, 6]]}]}),
            edge_error,
        ),
        (
            "future_duplicate",
            json!({"schedule": [{"tick": 100, "edges": [[0, 1], [0, 1]]}]}),
            edge_error,
        ),
        (
            "future_weight_count",
            json!({"schedule": [{"tick": 100, "edges": [[0, 1]], "edge_weights": []}]}),
            weight_error,
        ),
        (
            "weights_without_replacement",
            json!({"schedule": [{"tick": 100, "edge_weights": [1]}]}),
            scheduled_error,
        ),
    ];
    for (name, changes, expected_error) in cases {
        let mut payload = chain_payload(42);
        payload["simulation_options"]
            .as_object_mut()
            .unwrap()
            .extend(changes.as_object().unwrap().clone());
        let error = KernelBinding::new()
            .simulate_model("bass", payload)
            .unwrap_err();
        assert!(
            format!("{error:?}").contains(expected_error),
            "{name}: expected {expected_error}, got {error:?}"
        );
    }

    // The same two antiparallel arcs are valid when directed.
    let mut antiparallel = chain_payload(42);
    antiparallel["simulation_options"]["edges"] = json!([[0, 1], [1, 0]]);
    assert_eq!(run(antiparallel)["counts"], json!([1, 1, 2, 2, 2, 2]));
}
