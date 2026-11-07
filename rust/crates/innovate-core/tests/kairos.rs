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
