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

#[test]
fn scheduled_capacity_pauses_admission_and_reopens_waiting_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,5.,6.,7.,8.,9.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "schedule":[{"tick":2,"capacity":0},{"tick":7,"capacity":2,"service_ticks":2}]}
    });
    let run = |p| {
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()
    };
    let result = run(payload.clone());
    assert_eq!(result["counts"], json!([0, 0, 1, 1, 1, 3]));
    assert_eq!(result["metadata"]["current_capacity"], 2);
    assert_eq!(result["metadata"]["available"], 2);
    payload["inputs"]["time"] = json!([5.]);
    let paused = run(payload.clone());
    assert_eq!(paused["metadata"]["in_service"], 1);
    assert_eq!(paused["metadata"]["current_capacity"], 0);
    assert_eq!(paused["metadata"]["available"], 0);
    assert_eq!(paused["metadata"]["queue_length"], 2);
    payload["simulation_options"]["schedule"][1]["capacity"] = json!(4);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}

#[test]
fn retry_restores_hazard_after_balking_or_expiry() {
    let mut payload = json!({
        "inputs":{"time":[0.,6.,12.]},
        "parameters":{"p":1e12,"q":0.,"m":2.},
        "simulation_options":{"engine":"kairos_network","population":2,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "max_queue_length":0,"retry":{"delay_ticks":4,"max_retries":1}}
    });
    let run = |p| {
        KernelBinding::new()
            .simulate_model("bass", p)
            .unwrap()
            .result
            .unwrap()
    };
    let result = run(payload.clone());
    let waiter = &result["requests"][1]["agent"];
    assert_eq!(result["counts"], json!([0, 1, 2]));
    assert_eq!(result["balks"], json!([{"tick":2,"agent":waiter}]));
    assert_eq!(
        result["retry_eligibility"],
        json!([{"tick":6,"agent":waiter}])
    );
    assert_eq!(result["completions"][1], json!({"tick":12,"agent":waiter}));

    payload["simulation_options"]["max_queue_length"] = json!(1);
    payload["simulation_options"]["max_wait_ticks"] = json!(2);
    payload["simulation_options"]["retry"]["delay_ticks"] = json!(2);
    let expired = run(payload.clone());
    assert_eq!(expired["abandonments"], json!([{"tick":4,"agent":waiter}]));
    assert_eq!(expired["balks"], json!([]));
    assert_eq!(expired["completions"], result["completions"]);

    payload["simulation_options"]["max_queue_length"] = json!(0);
    payload["simulation_options"]["retry"]["delay_ticks"] = json!(1);
    let exhausted = run(payload.clone());
    assert_eq!(exhausted["counts"], json!([0, 1, 1]));
    assert_eq!(exhausted["balks"].as_array().unwrap().len(), 2);
    assert_eq!(exhausted["retry_eligibility"].as_array().unwrap().len(), 1);
    assert_eq!(
        exhausted["agents"][waiter.as_u64().unwrap() as usize]["state"],
        "withdrawn"
    );
    payload["simulation_options"]["retry"]["max_retries"] = json!(17);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}

#[test]
fn scheduled_hazards_enable_and_disable_requests_without_interrupting_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,2.,5.,7.,12.]},
        "parameters":{"p":0.,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[[0,1]],"directed":true,"initial_agents":[0],"capacity":1,
            "service_ticks":4,"ticks_per_unit":1,
            "schedule":[{"tick":2,"q":1e12},{"tick":4,"q":0.},{"tick":7,"p":1e12}]}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([1, 1, 1, 2, 3]));
    assert_eq!(
        result["requests"],
        json!([{"tick":3,"agent":1},{"tick":8,"agent":2}])
    );
    assert_eq!(
        result["completions"],
        json!([{"tick":7,"agent":1},{"tick":12,"agent":2}])
    );
    assert_eq!(result["metadata"]["current_q"], 0.);
    payload["simulation_options"]["schedule"][2]["p"] = json!(-1.);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}

#[test]
fn schedule_retargets_innovation_without_resetting_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,2.,5.,8.]},
        "parameters":{"p":1e12,"q":0.,"m":2.},
        "simulation_options":{"engine":"kairos_network","population":2,"seed":42,
            "edges":[],"capacity":1,"service_ticks":4,"ticks_per_unit":1,
            "innovation_multipliers":[1.,0.],
            "schedule":[{"tick":2,"innovation_multipliers":[0.,1.]}]}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        result["requests"],
        json!([{"tick":1,"agent":0},{"tick":3,"agent":1}])
    );
    assert_eq!(result["completions"][0], json!({"tick":5,"agent":0}));
    assert_eq!(
        result["metadata"]["innovation_multipliers"],
        json!([0., 1.])
    );
    payload["simulation_options"]["schedule"][0]["innovation_multipliers"] = json!([1.]);
    assert!(
        KernelBinding::new()
            .simulate_model("bass", payload)
            .is_err()
    );
}

#[test]
fn continuous_hazard_can_wake_from_zero_and_replays() {
    let payload = json!({
        "inputs":{"time":[0.,2.,3.,4.]},
        "parameters":{"p":1e12,"q":0.,"m":1.},
        "simulation_options":{"engine":"kairos_network","population":1,"seed":42,
            "edges":[],"capacity":1,"service_ticks":1,"ticks_per_unit":1,
            "hazard_forcing":{"time":[0.,2.,3.],"innovation":[0.,0.,1.],"imitation":[1.,1.,1.]}}
    });
    let kernel = KernelBinding::new();
    let result = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["requests"], json!([{"tick":3,"agent":0}]));
    assert_eq!(result["counts"], json!([0, 0, 0, 1]));
    assert_eq!(
        result,
        kernel
            .simulate_model("bass", payload)
            .unwrap()
            .result
            .unwrap()
    );
}

#[test]
fn linear_hazard_clock_matches_analytic_crossing() {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Exp};

    let mut rng = DeterministicStream::from_run_seed(42);
    let budget: f64 = rng.sample(Exp::new(1.).unwrap());
    // With the ramp factor t, cumulative hazard is p*t^2/2.
    let crossing = 0.435_f64;
    let p = 2. * budget / crossing.powi(2);
    let payload = json!({
        "inputs":{"time":[0.,0.43,0.44,0.45]},
        "parameters":{"p":p,"q":0.,"m":1.},
        "simulation_options":{"engine":"kairos_network","population":1,"seed":42,
            "edges":[],"capacity":1,"service_ticks":1,"ticks_per_unit":100,
            "hazard_forcing":{"time":[0.,1.],"innovation":[0.,1.],"imitation":[1.,1.]}}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["requests"], json!([{"tick":44,"agent":0}]));
    assert_eq!(result["counts"], json!([0, 0, 0, 1]));
}

#[test]
fn continuous_forcing_validates_effective_hazard_instead_of_unforced_total() {
    let payload = json!({
        "inputs":{"time":[0.,4.]},
        "parameters":{"p":1e308,"q":0.,"m":2.},
        "simulation_options":{"engine":"kairos_network","population":2,"seed":42,
            "edges":[],"capacity":2,"service_ticks":1,"ticks_per_unit":1,
            "hazard_forcing":{"time":[0.],"innovation":[1e-296],"imitation":[1.]}}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([0, 2]));
}

#[test]
fn continuous_hazard_scales_agent_weights_before_aggregation() {
    let payload = json!({
        "inputs":{"time":[0.,4.]},
        "parameters":{"p":1e-296,"q":0.,"m":2.},
        "simulation_options":{"engine":"kairos_network","population":2,"seed":42,
            "edges":[],"capacity":2,"service_ticks":1,"ticks_per_unit":1,
            "innovation_multipliers":[1e308,1e308],
            "hazard_forcing":{"time":[0.],"innovation":[1.],"imitation":[1.]}}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([0, 2]));
}

#[test]
fn scheduled_queue_limit_preserves_waiters_and_blocks_new_entries() {
    let payload = json!({
        "inputs":{"time":[0.,6.,12.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "schedule":[{"tick":3,"max_queue_length":0}]}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([0, 1, 2]));
    assert_eq!(
        result["balks"],
        json!([{"tick":4,"agent":result["requests"][2]["agent"]}])
    );
    assert_eq!(
        result["completions"][1]["agent"],
        result["requests"][1]["agent"]
    );
    assert_eq!(result["metadata"]["current_max_queue_length"], 0);
}

#[test]
fn scheduled_wait_deadline_applies_only_to_new_waiters() {
    let payload = json!({
        "inputs":{"time":[0.,6.,12.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "max_wait_ticks":10,"schedule":[{"tick":3,"max_wait_ticks":1}]}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([0, 1, 2]));
    assert_eq!(
        result["abandonments"],
        json!([{"tick":5,"agent":result["requests"][2]["agent"]}])
    );
    assert_eq!(
        result["completions"][1]["agent"],
        result["requests"][1]["agent"]
    );
    assert_eq!(result["metadata"]["current_max_wait_ticks"], 1);
}

#[test]
fn scheduled_null_clears_queue_restrictions() {
    let payload = json!({
        "inputs":{"time":[0.,6.,12.,16.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "max_wait_ticks":1,"max_queue_length":0,
            "schedule":[{"tick":2,"max_wait_ticks":null,"max_queue_length":null}]}
    });
    let result = KernelBinding::new()
        .simulate_model("bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["counts"], json!([0, 1, 2, 3]));
    assert_eq!(result["balks"], json!([]));
    assert_eq!(result["abandonments"], json!([]));
    assert_eq!(result["metadata"]["current_max_wait_ticks"], json!(null));
    assert_eq!(result["metadata"]["current_max_queue_length"], json!(null));
}

#[test]
fn reopening_admits_priority_waiter_at_deadline() {
    let run = |priorities| {
        KernelBinding::new()
            .simulate_model("bass", json!({
                "inputs":{"time":[0.,5.,6.,7.,8.,12.]},
                "parameters":{"p":1e12,"q":0.,"m":3.},
                "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
                    "edges":[[0,1],[1,2]],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
                    "agent_priorities":priorities,"max_wait_ticks":4,
                    "schedule":[{"tick":2,"capacity":0},{"tick":7,"capacity":1,"service_ticks":2}]}
            }))
            .unwrap()
            .result
            .unwrap()
    };
    let baseline = run([0; 3]);
    let requests = baseline["requests"].as_array().unwrap();
    let late = requests[2]["agent"].as_u64().unwrap() as usize;
    let mut priorities = [0; 3];
    priorities[late] = 10;
    let result = run(priorities);
    assert_eq!(result["requests"], baseline["requests"]);
    assert_eq!(result["counts"], json!([0, 0, 1, 1, 1, 2]));
    assert_eq!(result["completions"][0], baseline["completions"][0]);
    assert_eq!(result["completions"][1], json!({"agent":late,"tick":9}));
    assert_eq!(
        result["abandonments"],
        json!([{"agent":requests[1]["agent"],"tick":7}])
    );
    assert_eq!(result["metadata"]["queue_length"], 0);
    assert_eq!(result["metadata"]["available"], 1);
}

#[test]
fn scheduled_priorities_reorder_waiters_without_preempting_service() {
    let mut payload = json!({
        "inputs":{"time":[0.,5.,6.,11.,16.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1}
    });
    let kernel = KernelBinding::new();
    let baseline = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    let late = baseline["requests"][2]["agent"].as_u64().unwrap() as usize;
    let mut priorities = [0; 3];
    priorities[late] = 10;
    payload["simulation_options"]["schedule"] = json!([{"tick":4,"agent_priorities":priorities}]);
    let result = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["requests"], baseline["requests"]);
    assert_eq!(result["completions"][0], baseline["completions"][0]);
    assert_eq!(result["completions"][1], json!({"agent":late,"tick":11}));
    assert_eq!(
        result["completions"][2]["agent"],
        baseline["requests"][1]["agent"]
    );
    assert_eq!(result["metadata"]["agent_priorities"], json!(priorities));
    payload["simulation_options"]["schedule"] = json!([
        {"tick":4,"agent_priorities":priorities},
        {"tick":5,"agent_priorities":[0,0,0]}
    ]);
    let tied = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(tied["completions"], baseline["completions"]);
    assert_eq!(tied["metadata"]["applied_schedule_ticks"], json!([4, 5]));
    payload["simulation_options"]["schedule"][0]["agent_priorities"] = json!([10]);
    assert!(kernel.simulate_model("bass", payload).is_err());
}

#[test]
fn scheduled_agent_durations_replace_scalar_override_at_admission() {
    let mut payload = json!({
        "inputs":{"time":[0.,6.,20.]},
        "parameters":{"p":1e12,"q":0.,"m":3.},
        "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "schedule":[{"tick":2,"service_ticks":9},{"tick":4,"agent_service_ticks":[2,3,4]}]}
    });
    let kernel = KernelBinding::new();
    let result = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    let durations = [2, 3, 4];
    let mut start = 1;
    for index in 0..3 {
        let request = &result["requests"][index];
        let agent = request["agent"].as_u64().unwrap() as usize;
        let requested = request["tick"].as_u64().unwrap();
        let duration = if index == 0 { 5 } else { durations[agent] };
        let completion = start + duration;
        assert_eq!(
            result["service_starts"][index],
            json!({"agent":agent,"tick":start,"request_tick":requested,
                "wait_ticks":start-requested,"service_ticks":duration,
                "scheduled_completion_tick":completion})
        );
        assert_eq!(
            result["completions"][index],
            json!({"agent":agent,"tick":completion})
        );
        start = completion;
    }
    assert!(result["metadata"]["service_override_ticks"].is_null());
    payload["simulation_options"]["schedule"][1]["service_ticks"] = json!(4);
    assert!(kernel.simulate_model("bass", payload).is_err());
}

#[test]
fn scheduled_retry_preserves_pending_wake_after_policy_clears() {
    let mut payload = json!({
        "inputs":{"time":[0.,6.,12.]},
        "parameters":{"p":1e12,"q":0.,"m":2.},
        "simulation_options":{"engine":"kairos_network","population":2,"seed":42,
            "edges":[],"capacity":1,"service_ticks":5,"ticks_per_unit":1,
            "max_queue_length":0,"retry":{"delay_ticks":4,"max_retries":1},
            "schedule":[{"tick":3,"retry":null}]}
    });
    let kernel = KernelBinding::new();
    let result = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    let waiter = &result["requests"][1]["agent"];
    assert_eq!(result["balks"], json!([{"tick":2,"agent":waiter}]));
    assert_eq!(
        result["retry_eligibility"],
        json!([{"tick":6,"agent":waiter}])
    );
    assert_eq!(result["completions"][1], json!({"tick":12,"agent":waiter}));
    assert_eq!(result["metadata"]["applied_schedule_ticks"], json!([3]));

    payload["simulation_options"]["retry"] = json!({"delay_ticks":1,"max_retries":2});
    let cleared = kernel
        .simulate_model("bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        cleared["retry_eligibility"],
        json!([{"tick":3,"agent":waiter}])
    );
    assert_eq!(cleared["balks"][1], json!({"tick":4,"agent":waiter}));
    assert_eq!(cleared["counts"], json!([0, 1, 1]));

    payload["simulation_options"]["schedule"][0]["retry"]["delay_ticks"] = json!(0);
    assert!(kernel.simulate_model("bass", payload).is_err());
}

#[test]
fn resource_metrics_clip_service_and_queue_wait_at_horizon() {
    let result = KernelBinding::new()
        .simulate_model(
            "bass",
            json!({
                "inputs":{"time":[0,6]},
                "parameters":{"p":1e12,"q":0,"m":3},
                "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
                    "edges":[],"capacity":1,"service_ticks":8,"ticks_per_unit":1,
                    "max_wait_ticks":2,"schedule":[{"tick":3,"capacity":0}]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        result["resource_metrics"],
        json!({
            "horizon_ticks":6,"occupied_slot_ticks":5,"configured_slot_ticks":3,"over_capacity_slot_ticks":3,"idle_slot_ticks":1,
            "peak_queue_length":1,"peak_in_service":1,
            "admitted_wait_ticks":0,"abandoned_wait_ticks":2,"pending_wait_ticks":1
        })
    );
    assert_eq!(result["completions"], json!([]));
    assert_eq!(result["metadata"]["in_service"], 1);
    assert_eq!(result["metadata"]["queue_length"], 1);
}

#[test]
fn unweighted_imitation_preserves_finite_hazard_at_large_rates() {
    let result = KernelBinding::new()
        .simulate_model(
            "bass",
            json!({
                "inputs":{"time":[0,2]},
                "parameters":{"p":0,"q":1e308,"m":3},
                "simulation_options":{"engine":"kairos_network","population":3,"seed":42,
                    "edges":[[0,2],[1,2]],"initial_agents":[0,1],
                    "capacity":1,"service_ticks":1,"ticks_per_unit":1}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["metadata"]["kairos_dispatched_events"], 2);
    assert_eq!(result["counts"], json!([2, 3]));
    assert_eq!(result["completions"], json!([{"tick":2,"agent":2}]));
}

#[test]
fn heterogeneous_imitation_scales_large_coefficients_by_exposure() {
    for weights in [json!(null), json!([1, 1])] {
        let mut options = json!({
            "engine":"kairos_network","population":3,"seed":42,"directed":true,
            "edges":[[0,2],[1,2]],"initial_agents":[0],"imitation_multipliers":[1,1,2],
            "capacity":1,"service_ticks":1,"ticks_per_unit":1
        });
        if !weights.is_null() {
            options["edge_weights"] = weights;
        }
        let result = KernelBinding::new()
            .simulate_model(
                "bass",
                json!({
                    "inputs":{"time":[0,2]},
                    "parameters":{"p":0,"q":1e308,"m":3},
                    "simulation_options":options
                }),
            )
            .unwrap()
            .result
            .unwrap();
        assert_eq!(result["counts"], json!([1, 2]));
        assert_eq!(result["completions"], json!([{"tick":2,"agent":2}]));
    }
}
