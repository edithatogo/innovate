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
