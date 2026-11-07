use innovate_core::{KernelBinding, json};
fn run(
    edges: serde_json::Value,
    p: f64,
    initial: serde_json::Value,
    capacity: u64,
) -> serde_json::Value {
    KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[0.,5.,20.,100.]},"parameters":{"p":p,"q":3.,"m":3.},"simulation_options":{"engine":"kairos_network","population":3,"seed":42,"edges":edges,"capacity":capacity,"service_ticks":5,"ticks_per_unit":1,"initial_agents":initial}})).unwrap().result.unwrap()
}
#[test]
fn contacts_control_pure_contagion_and_replay() {
    let a = run(json!([[0, 1], [1, 2]]), 0., json!([0]), 1);
    assert_eq!(a, run(json!([[0, 1], [1, 2]]), 0., json!([0]), 1));
    assert_eq!(a["counts"][3], 3);
    assert_eq!(
        run(json!([]), 0., json!([0]), 1)["counts"],
        json!([1, 1, 1, 1])
    );
    let agents = a["agents"].as_array().unwrap();
    assert_eq!(agents.len(), 3);
    assert!(agents.iter().all(|a| a["state"] == "adopted"));
    assert_eq!(a["metadata"]["queue_length"], 0);
    assert_eq!(a["metadata"]["available"], 1);
}
#[test]
fn fifo_service_has_exact_capacity_limited_completion_times() {
    let a = run(json!([]), 1e12, json!([]), 1);
    assert_eq!(
        a["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["agent"].clone())
            .collect::<Vec<_>>(),
        a["completions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["agent"].clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        a["completions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["tick"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![6, 11, 16]
    );
    let b = run(json!([]), 1e12, json!([]), 3);
    assert_eq!(
        b["completions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["tick"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![6, 7, 8]
    );
}
#[test]
fn malformed_networks_resources_and_initial_nodes_are_rejected() {
    let base = json!({"inputs":{"time":[1.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":{"engine":"kairos_network","population":3,"seed":42,"edges":[],"capacity":1,"service_ticks":5}});
    for (key, value) in [
        ("edges", json!([[0, 3]])),
        ("edges", json!([[0, 0]])),
        ("edges", json!([[0, 1], [1, 0]])),
        ("capacity", json!(0)),
        ("service_ticks", json!(0)),
        ("initial_agents", json!([0, 0])),
    ] {
        let mut p = base.clone();
        p["simulation_options"][key] = value;
        assert!(KernelBinding::new().simulate_model("bass", p).is_err());
    }
}
#[test]
fn pending_agents_are_not_counted_as_adopted() {
    let a=KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[5.]},"parameters":{"p":1e12,"q":0.,"m":3.},"simulation_options":{"engine":"kairos_network","population":3,"seed":42,"edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1}})).unwrap().result.unwrap();
    assert_eq!(a["counts"][0], 0);
    assert_eq!(a["metadata"]["queue_length"], 2);
    assert_eq!(a["metadata"]["available"], 0);
    assert!(
        a["agents"]
            .as_array()
            .unwrap()
            .iter()
            .all(|a| a["state"] == "pending")
    );
}

#[test]
fn heterogeneous_service_uses_the_queued_agents_duration() {
    let payload = json!({"inputs":{"time":[0.,100.]},"parameters":{"p":1e12,"q":0.,"m":3.},"simulation_options":{"engine":"kairos_network","population":3,"seed":42,"edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,"innovation_multipliers":[1.,0.,1.],"imitation_multipliers":[1.,1.,1.],"agent_service_ticks":[4,9,7]}});
    let a = KernelBinding::new()
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        a,
        KernelBinding::new()
            .simulate_model("bass", payload)
            .unwrap()
            .result
            .unwrap()
    );
    assert_eq!(a["counts"], json!([0, 2]));
    let requests = a["requests"].as_array().unwrap();
    let completions = a["completions"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    let durations = [4, 9, 7];
    let mut expected = 1;
    for (request, completion) in requests.iter().zip(completions) {
        assert_eq!(request["agent"], completion["agent"]);
        let i = request["agent"].as_u64().unwrap() as usize;
        assert_ne!(i, 1);
        expected += durations[i];
        assert_eq!(completion["tick"].as_u64().unwrap(), expected);
    }
}

#[test]
fn heterogeneous_vectors_reject_invalid_domains() {
    for (key, value) in [
        ("innovation_multipliers", json!([1.])),
        ("innovation_multipliers", json!([1., -1.])),
        ("imitation_multipliers", json!([1., null])),
        ("agent_service_ticks", json!([1, 0])),
        ("agent_service_ticks", json!([1, 1000000000001_u64])),
    ] {
        let mut p = json!({"inputs":{"time":[1.]},"parameters":{"p":1.,"q":0.,"m":2.},"simulation_options":{"engine":"kairos_network","population":2,"seed":42,"edges":[],"capacity":1,"service_ticks":5}});
        p["simulation_options"][key] = value;
        assert!(KernelBinding::new().simulate_model("bass", p).is_err());
    }
}

#[test]
fn imitation_weights_control_seeded_contagion() {
    let mut p = json!({"inputs":{"time":[100.]},"parameters":{"p":0.,"q":1e12,"m":2.},"simulation_options":{"engine":"kairos_network","population":2,"seed":42,"edges":[[0,1]],"initial_agents":[0],"capacity":1,"service_ticks":5,"ticks_per_unit":1,"imitation_multipliers":[1.,0.]}});
    assert_eq!(
        KernelBinding::new()
            .simulate_model("bass", p.clone())
            .unwrap()
            .result
            .unwrap()["counts"],
        json!([1])
    );
    p["simulation_options"]["imitation_multipliers"] = json!([1., 1.]);
    assert_eq!(
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()["counts"],
        json!([2])
    );
}

#[test]
fn directed_weights_control_exposure_without_overflow() {
    let mut payload = json!({"inputs":{"time":[0.,5.,20.]},
        "parameters":{"p":0.,"q":1e12,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
        "edges":[[0,1],[2,1]],"directed":true,"edge_weights":[1e308,1e308],
        "capacity":1,"service_ticks":5,"ticks_per_unit":1,"initial_agents":[0]}});
    let run = |p| {
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()
    };
    let a = run(payload.clone());
    assert_eq!(a["counts"], json!([1, 1, 2]));
    assert_eq!(a["completions"][0]["tick"], 6);
    payload["simulation_options"]["edge_weights"] = json!([1., 1.]);
    assert_eq!(a, run(payload.clone()));
    payload["simulation_options"]["edge_weights"] = json!([1e-100, 1.]);
    assert_eq!(run(payload.clone())["counts"], json!([1, 1, 1]));
    payload["simulation_options"]["edges"] = json!([[1, 0], [2, 1]]);
    payload["simulation_options"]["edge_weights"] = json!([1., 1.]);
    assert_eq!(run(payload.clone())["counts"], json!([1, 1, 1]));
    payload["simulation_options"]["edges"] = json!([[0, 1], [1, 0]]);
    assert_eq!(run(payload.clone())["counts"], json!([1, 1, 2]));
    payload["simulation_options"]["directed"] = json!(false);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload.clone())
            .is_err()
    );
    payload["simulation_options"]["directed"] = json!(true);
    for invalid in [json!([1.]), json!([0., 1.]), json!([-1., 1.])] {
        payload["simulation_options"]["edge_weights"] = invalid;
        assert!(
            KernelBinding::new()
                .simulate_model("bass", payload.clone())
                .is_err()
        );
    }
}

#[test]
fn scheduled_graph_and_service_changes_take_effect_at_boundaries() {
    let mut payload = json!({
        "inputs":{"time":[0.,2.,7.,8.,10.,11.]},
        "parameters":{"p":0.,"q":1e12,"m":3.},
        "simulation_options":{
            "engine":"kairos_network","population":3,"seed":42,"directed":true,
            "edges":[],"initial_agents":[0],"capacity":1,"service_ticks":5,
            "ticks_per_unit":1,"schedule":[
                {"tick":2,"edges":[[0,1]],"edge_weights":[2.]},
                {"tick":8,"edges":[[1,2]],"service_ticks":2}
            ]
        }
    });
    let r = KernelBinding::new()
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(r["counts"], json!([1, 1, 1, 2, 2, 3]));
    assert_eq!(
        r["requests"],
        json!([{"tick":3,"agent":1},{"tick":9,"agent":2}])
    );
    assert_eq!(
        r["completions"],
        json!([{"tick":8,"agent":1},{"tick":11,"agent":2}])
    );
    assert_eq!(r["metadata"]["applied_schedule_ticks"], json!([2, 8]));
    for invalid in [
        json!([{"tick":0,"edges":[]}]),
        json!([{"tick":2,"service_ticks":2},{"tick":2,"edges":[]}]),
        json!([{"tick":20,"edge_weights":[1.]}]),
        json!([{"tick":20,"edges":[[0,3]]}]),
    ] {
        payload["simulation_options"]["schedule"] = invalid;
        assert!(
            KernelBinding::new()
                .simulate_model("bass", payload.clone())
                .is_err()
        );
    }
}

#[test]
fn priority_reorders_waiters_without_preempting_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,3.,6.,11.,16.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1}
    });
    let run = |p| {
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()
    };
    let fifo = run(payload.clone());
    let arrivals = fifo["requests"].as_array().unwrap();
    let late = arrivals[2]["agent"].as_u64().unwrap() as usize;
    let mut priorities = [0; 3];
    priorities[late] = 10;
    payload["simulation_options"]["agent_priorities"] = json!(priorities);
    let result = run(payload.clone());
    assert_eq!(result["requests"], fifo["requests"]);
    assert_eq!(result["completions"][0], fifo["completions"][0]);
    assert_eq!(result["completions"][1], json!({"agent":late,"tick":11}));
    assert_eq!(result["completions"][2]["agent"], arrivals[1]["agent"]);
    assert_eq!(result["metadata"]["available"], 1);
    payload["inputs"]["time"] = json!([3.]);
    let pending = run(payload.clone());
    assert_eq!(pending["metadata"]["available"], 0);
    assert_eq!(pending["metadata"]["queue_length"], 2);
    payload["simulation_options"]["agent_priorities"] = json!([10]);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}

#[test]
fn waiting_deadline_withdraws_only_agents_that_miss_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,6.,7.,11.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,"max_wait_ticks":4}
    });
    let run = |p| {
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()
    };
    let result = run(payload.clone());
    let arrivals = result["requests"].as_array().unwrap();
    assert_eq!(result["counts"], json!([0, 1, 1, 2]));
    assert_eq!(
        result["completions"][1],
        json!({"agent":arrivals[1]["agent"],"tick":11})
    );
    assert_eq!(
        result["abandonments"],
        json!([{"agent":arrivals[2]["agent"],"tick":7}])
    );
    assert_eq!(result["metadata"]["queue_length"], 0);
    assert_eq!(result["metadata"]["available"], 1);
    payload["inputs"]["time"] = json!([7.]);
    let pending = run(payload.clone());
    assert_eq!(pending["metadata"]["available"], 0);
    assert_eq!(pending["metadata"]["queue_length"], 0);
    payload["simulation_options"]["max_wait_ticks"] = json!(0);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}
