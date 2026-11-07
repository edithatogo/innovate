use innovate_core::{KernelBinding, json};
fn run(seed: u64, time: Vec<f64>, q: f64) -> serde_json::Value {
    KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":time},"parameters":{"p":0.2,"q":q,"m":100.},"simulation_options":{"engine":"kairos_des","seed":seed,"population":100,"ticks_per_unit":1000000}})).unwrap().result.unwrap()
}
#[test]
fn native_kairos_replays_and_does_not_depend_on_sampling_grid() {
    let a = run(42, vec![0., 1., 5., 20.], 0.3);
    assert_eq!(a, run(42, vec![0., 1., 5., 20.], 0.3));
    assert_ne!(a["counts"], run(43, vec![0., 1., 5., 20.], 0.3)["counts"]);
    assert_eq!(a["counts"][2], run(42, vec![5., 20.], 0.3)["counts"][0]);
    assert_eq!(a["metadata"]["engine"], "kairos_des");
    let c = a["counts"].as_array().unwrap();
    assert_eq!(c[0], 0);
    assert!(c.windows(2).all(|p| p[0].as_u64() <= p[1].as_u64()));
}
#[test]
fn innovation_only_ensemble_matches_binomial_mean() {
    let n = 500;
    let sum: f64 = (0..n)
        .map(|seed| run(seed, vec![5.], 0.)["counts"][0].as_f64().unwrap())
        .sum();
    let expected = 100. * (1. - (-1_f64).exp());
    let se = (100. * (1. - (-1_f64).exp()) * (-1_f64).exp() / n as f64).sqrt();
    assert!((sum / n as f64 - expected).abs() < 5. * se);
}
#[test]
fn unsupported_or_malformed_engines_never_silently_use_a_curve() {
    for options in [
        json!({"engine":"other"}),
        json!({"engine":"kairos_des","seed":42,"population":0}),
        json!([]),
    ] {
        assert!(KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[0.,1.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":options})).is_err());
    }
}
#[test]
fn finite_domain_and_option_guards_are_enforced() {
    for options in [
        json!({"engine":"kairos_des","seed":-1,"population":10}),
        json!({"engine":"kairos_des","seed":1,"population":100001}),
        json!({"engine":"kairos_des","seed":1,"population":10,"ticks_per_unit":0}),
        json!({"engine":"kairos_des","seed":1,"population":10,"extra":true}),
    ] {
        assert!(KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[0.,1.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":options})).is_err());
    }
    for time in [vec![-1., 1.], vec![1e100]] {
        assert!(KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":time},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":{"engine":"kairos_des","seed":1,"population":10}})).is_err());
    }
}
#[test]
fn abm_agents_are_conserved_replayable_and_match_homogeneous_des() {
    let payload = json!({"inputs":{"time":[0.,1.,5.,20.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":42,"population":100}});
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
    assert_eq!(a["counts"], run(42, vec![0., 1., 5., 20.], 0.3)["counts"]);
    let agents = a["agents"].as_array().unwrap();
    assert_eq!(agents.len(), 100);
    let ids: std::collections::HashSet<_> = agents
        .iter()
        .map(|a| a["index"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), 100);
    assert_eq!(
        agents.iter().filter(|a| a["adopted"] == true).count() as u64,
        a["counts"][3].as_u64().unwrap()
    );
}
#[test]
fn covariates_and_events_can_pause_and_restart_pressure() {
    let base = json!({"inputs":{"time":[0.,1.,3.,5.,20.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":42,"population":100,"covariate":{"time":[0.,3.],"values":[-1.,0.],"beta_p":0.2,"beta_q":0.3}}});
    let a = KernelBinding::new()
        .simulate_model("bass", base.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(a["counts"][1], 0);
    assert_eq!(a["counts"][2], 0);
    assert!(a["counts"][3].as_u64().unwrap() > 0);
    let mut events = base.clone();
    events["simulation_options"]
        .as_object_mut()
        .unwrap()
        .remove("covariate");
    events["simulation_options"]["events"] =
        json!([{"time":0.,"delta_p":-0.2,"delta_q":-0.3},{"time":3.,"delta_p":0.2,"delta_q":0.3}]);
    assert_eq!(
        a["counts"],
        KernelBinding::new()
            .simulate_model("bass", events)
            .unwrap()
            .result
            .unwrap()["counts"]
    );
    let mut bad = base;
    bad["simulation_options"]["covariate"]["beta_p"] = json!(2.);
    assert!(KernelBinding::new().simulate_model("bass", bad).is_err());
}
#[test]
fn forced_innovation_ensemble_matches_integrated_hazard() {
    let n = 300;
    let sum:f64=(0..n).map(|seed|KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[5.]},"parameters":{"p":0.2,"q":0.,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":seed,"population":100,"events":[{"time":0.,"delta_p":-0.2,"delta_q":0.},{"time":3.,"delta_p":0.2,"delta_q":0.}]}})).unwrap().result.unwrap()["counts"][0].as_f64().unwrap()).sum();
    let probability = 1. - (-0.4_f64).exp();
    let se = (100. * probability * (1. - probability) / n as f64).sqrt();
    assert!((sum / n as f64 - 100. * probability).abs() < 5. * se);
}
#[test]
fn malformed_forcing_and_colliding_boundaries_are_errors() {
    for forcing in [
        json!({"time":[1.],"values":[1.],"beta_p":1.,"beta_q":0.}),
        json!({"time":[0.,1.],"values":[1.],"beta_p":1.,"beta_q":0.}),
        json!({"time":[0.],"values":[1.],"beta_p":1.,"beta_q":0.,"unknown":1}),
    ] {
        assert!(KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[5.]},"parameters":{"p":0.2,"q":0.,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":1,"population":10,"covariate":forcing}})).is_err());
    }
    let p = json!({"inputs":{"time":[5.]},"parameters":{"p":0.2,"q":0.,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":1,"population":10,"ticks_per_unit":1,"events":[{"time":0.1,"delta_p":0.,"delta_q":0.},{"time":0.2,"delta_p":0.,"delta_q":0.}]}});
    assert!(KernelBinding::new().simulate_model("bass", p).is_err());
}
#[test]
fn abm_initial_adoption_is_conserved_at_zero_and_saturation() {
    for initial in [20, 100] {
        let r=KernelBinding::new().simulate_model("bass",json!({"inputs":{"time":[0.,20.]},"parameters":{"p":0.2,"q":0.3,"m":100.},"simulation_options":{"engine":"kairos_abm","seed":42,"population":100,"initial_adopters":initial}})).unwrap().result.unwrap();
        assert_eq!(r["counts"][0], initial);
        assert_eq!(
            r["metadata"]["kairos_dispatched_events"],
            r["counts"][1].as_u64().unwrap() - initial
        );
        assert_eq!(
            r["agents"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|a| a["adopted"] == true)
                .count() as u64,
            r["counts"][1].as_u64().unwrap()
        );
    }
}

#[test]
fn scenario_ensemble_replays_and_recovers_innovation_mean() {
    let replicates = 128;
    let mut scenarios = Vec::new();
    let mut pairs = Vec::new();
    for seed in 0..replicates {
        let baseline = format!("baseline-{seed}");
        let alternative = format!("alternative-{seed}");
        for (id, innovation) in [(&baseline, 0.), (&alternative, 0.2)] {
            scenarios.push(json!({"id":id,"model_key":"bass","simulation":{
                "inputs":{"time":[0.,5.]},"parameters":{"p":innovation,"q":0.,"m":100.},
                "simulation_options":{"engine":"kairos_abm","seed":seed,"population":100}
            }}));
        }
        pairs.push(json!({"baseline":baseline,"alternative":alternative}));
    }
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"workers":4,"scenarios":scenarios,
            "comparisons":[{"pairs":pairs,"interval_mass":0.9,"require_distinct_pair_seeds":true}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    request["payload"]["workers"] = json!(1);
    assert_eq!(run(&request), response);
    let summary = &response["result"]["comparisons"][0]["result"];
    assert_eq!(
        summary["seed_design"],
        "matched_within_pair_distinct_between_pairs"
    );
    for index in [2, 3] {
        request["payload"]["scenarios"][index]["simulation"]["simulation_options"]["seed"] =
            json!(0);
        let comparison = &run(&request)["result"]["comparisons"][0];
        assert_eq!(comparison["error"]["code"], "invalid_comparison");
        let expected = if index == 2 {
            "same effective seed"
        } else {
            "distinct effective seed"
        };
        assert!(
            comparison["error"]["message"]
                .as_str()
                .unwrap()
                .contains(expected)
        );
    }
    let differences = summary["differences"].as_array().unwrap();
    assert_eq!(differences.len(), replicates);
    let mean = differences
        .iter()
        .map(|value| value.as_f64().unwrap())
        .sum::<f64>()
        / replicates as f64;
    // With no imitation, the adoption count is approximately binomial.
    // This tolerance is wider than three standard errors for this ensemble.
    let expected = 100. * (1. - (-1_f64).exp());
    assert!(
        (mean - expected).abs() < 1.5,
        "mean: {mean}, expected: {expected}"
    );
}

#[test]
fn pinned_scheduler_dispatches_ties_in_order_and_skips_cancelled_events() {
    use kairo_ecs_core::Scheduler;
    use kairo_ecs_types::{EventKind, ScheduleRequest, SimTime, StepOutcome};

    let mut scheduler = Scheduler::new();
    let tick = SimTime::from_ticks(12);
    let mut schedule = |kind| {
        scheduler.schedule(ScheduleRequest {
            at: tick,
            priority: 0,
            entity: None,
            kind: EventKind::Custom(kind),
        })
    };
    let first = schedule(1);
    let cancelled = schedule(2);
    let last = schedule(3);
    assert!(scheduler.cancel(cancelled));

    for (id, kind) in [(first, 1), (last, 3)] {
        let StepOutcome::Dispatched(event) = scheduler.step() else {
            panic!("expected Kairos to dispatch event {kind}");
        };
        assert_eq!(event.id, id);
        assert_eq!(event.at, tick);
        assert_eq!(event.kind, EventKind::Custom(kind));
    }
    assert!(matches!(scheduler.step(), StepOutcome::Empty));
}

#[test]
fn pinned_abm_replays_entity_state_and_rng() {
    use kairo_ecs_abm::ABMContext;
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
    use kairo_ecs_types::{EventKind, SimTime, StepOutcome};

    let replay = || {
        let mut context = ABMContext::new(42);
        let entity = context.spawn_agent();
        assert!(context.get::<u64>(entity).is_none());
        assert!(context.attach(entity, 0_u64));
        let mut rng = DeterministicStream::from_entity_purpose(42, entity, "innovate.test");
        let value = rng.sample(Uniform::new(1_u64, 100).unwrap());
        let tick = SimTime::from_ticks(7);
        context.schedule_behaviour(entity, 1, tick);
        let StepOutcome::Dispatched(event) = context.step() else {
            panic!("expected a Kairos behavior dispatch");
        };
        assert_eq!(event.entity, Some(entity));
        assert_eq!(event.kind, EventKind::Custom(1));
        assert_eq!(event.at, tick);
        assert!(context.attach(entity, value));
        assert_eq!(context.get::<u64>(entity), Some(&value));
        (entity, value, json!(rng.checkpoint()))
    };
    assert_eq!(replay(), replay());
}

#[test]
fn emitted_rng_checkpoint_restores_after_all_adoptions() {
    use kairo_ecs_rng::{DeterministicStream, RngCheckpoint, rand_distr::Exp};

    let response = KernelBinding::new()
        .simulate_model(
            "bass",
            json!({"inputs":{"time":[0.,4.]},
                "parameters":{"p":1e12,"q":0.,"m":4.},
                "simulation_options":{"engine":"kairos_des","seed":42,
                    "population":4,"ticks_per_unit":1}}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(response["counts"], json!([0, 4]));
    let checkpoint: RngCheckpoint =
        serde_json::from_value(response["metadata"]["rng_checkpoint"].clone()).unwrap();
    let mut restored = DeterministicStream::restore(&checkpoint).unwrap();
    let mut reference = DeterministicStream::from_run_seed(42);
    for remaining in (1..=4).rev() {
        reference.sample(Exp::new(1e12 * f64::from(remaining)).unwrap());
    }
    assert_eq!(restored.checkpoint(), reference.checkpoint());
    for _ in 0..4 {
        let distribution = Exp::new(1.).unwrap();
        assert_eq!(
            restored.sample(distribution),
            reference.sample(distribution)
        );
    }
}

#[test]
fn des_checkpoint_resumes_with_original_history_and_rng() {
    let mut payload = json!({"inputs":{"time":[0,1,2,4]},
        "parameters":{"p":1e12,"q":0,"m":4},
        "simulation_options":{"engine":"kairos_des","seed":42,"population":4,"ticks_per_unit":1}});
    let kernel = KernelBinding::new();
    let run = |payload| kernel.simulate_model("bass", payload);
    let uninterrupted = run(payload.clone()).unwrap().result.unwrap();
    payload["simulation_options"]["pause_after_adoptions"] = json!(1);
    let paused = run(payload.clone()).unwrap().result.unwrap();
    assert_eq!(paused["status"], "paused");
    assert_eq!(paused["checkpoint"]["events"], json!(["1"]));

    payload["simulation_options"]["pause_after_adoptions"] = json!(2);
    payload["simulation_options"]["resume_checkpoint"] = paused["checkpoint"].clone();
    let paused_again = run(payload.clone()).unwrap().result.unwrap();
    assert_eq!(paused_again["checkpoint"]["events"], json!(["1", "2"]));

    payload["simulation_options"]
        .as_object_mut()
        .unwrap()
        .remove("pause_after_adoptions");
    payload["simulation_options"]["resume_checkpoint"] = paused_again["checkpoint"].clone();
    assert_eq!(run(payload.clone()).unwrap().result.unwrap(), uninterrupted);
    payload["simulation_options"]["seed"] = json!(43);
    assert!(run(payload).is_err());
}
