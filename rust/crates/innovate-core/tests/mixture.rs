use innovate_core::{KernelBinding, json};
fn parameters() -> serde_json::Value {
    json!({"model_0_L":100.,"model_0_k":0.4,"model_0_x0":6.,"model_1_L":200.,"model_1_k":0.3,"model_1_x0":10.,"weight_0":0.4,"weight_1":0.6,"noise_scale":2.})
}
#[test]
fn mixture_prediction_and_posterior_responsibilities_match_closed_forms() {
    let a=KernelBinding::new().predict_model("mixture",json!({"inputs":{"time":[0.,6.,10.]},"constructor_kwargs":{"components":["logistic","logistic"]},"parameters":parameters()})).unwrap().result.unwrap();
    let expected = 0.4 * 50. + 0.6 * 200. / (1. + 1.2_f64.exp());
    assert!((a["values"][1].as_f64().unwrap() - expected).abs() < 1e-10);
    assert_eq!(a["component_rows"][1][0], 50.);
    let d=KernelBinding::new().diagnose_model("mixture",json!({"inputs":{"time":[6.],"observed":[50.]},"constructor_kwargs":{"components":["logistic","logistic"]},"parameters":parameters()})).unwrap().result.unwrap();
    let expected =
        0.4 / (0.4 + 0.6 * (-0.5 * ((50. - 200. / (1. + 1.2_f64.exp())) / 2.).powi(2)).exp());
    assert!((d["responsibilities"][0][0].as_f64().unwrap() - expected).abs() < 1e-12);
    assert!(d["diagnostics"]["metrics"]["AIC"].is_null());
}
#[test]
fn gaussian_em_updates_likelihood_and_saves_component_state() {
    let t: Vec<_> = (0..50).map(|i| i as f64 * 0.5).collect();
    let y: Vec<_> = t
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i % 2 == 0 {
                100. / (1. + (-0.4 * (t - 6.)).exp())
            } else {
                200. / (1. + (-0.3 * (t - 10.)).exp())
            }
        })
        .collect();
    let f=KernelBinding::new().fit_model("mixture",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"components":["logistic","logistic"],"max_iter":30,"noise_floor":0.001},"fit_options":{"initial_parameters":parameters()}})).unwrap().result.unwrap();
    assert_eq!(f["optimizer"]["converged"], true);
    let history = f["optimizer"]["objective_history"].as_array().unwrap();
    assert!(
        history
            .windows(2)
            .all(|p| p[1].as_f64().unwrap() + 1e-8 >= p[0].as_f64().unwrap())
    );
    assert!((f["parameters"]["model_0_L"].as_f64().unwrap() - 100.).abs() < 0.05);
    assert!((f["parameters"]["model_1_L"].as_f64().unwrap() - 200.).abs() < 0.05);
    let a = KernelBinding::new()
        .predict_model(
            "mixture",
            json!({"inputs":{"time":[40.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!(a["values"][0].as_f64().unwrap() > 140.);
}

#[test]
fn historical_adoption_mode_preserves_normalized_level_responsibilities() {
    let t: Vec<_> = (0..20).map(|i| i as f64).collect();
    let y: Vec<_> = t
        .iter()
        .map(|t| 100. / (1. + (-0.4 * (t - 6.)).exp()))
        .collect();
    let f=KernelBinding::new().fit_model("mixture",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"components":["logistic","logistic"],"mode":"historical_adoption","weights":[0.25,0.75],"max_iter":3}})).unwrap().result.unwrap();
    assert_eq!(f["optimizer"]["converged"], true);
    assert_eq!(
        f["optimizer"]["objective_name"],
        "adoption_log_score_not_observation_likelihood"
    );
    assert!((f["parameters"]["weight_0"].as_f64().unwrap() - 0.25).abs() < 1e-10);
    assert!(
        f["responsibilities"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| (r[0].as_f64().unwrap() - 0.25).abs() < 1e-10)
    );
    let summary = KernelBinding::new()
        .summarize_model("mixture", json!({"state":f["state"]}))
        .unwrap()
        .result
        .unwrap();
    assert!(summary.get("predictions").is_none());
}
#[test]
fn failed_component_updates_are_rolled_back_and_never_report_convergence() {
    let t: Vec<_> = (0..20).map(|i| i as f64).collect();
    let y: Vec<_> = t
        .iter()
        .map(|t| 100. / (1. + (-0.4 * (t - 6.)).exp()))
        .collect();
    let initial =
        json!({"model_0_L":20.,"model_0_k":0.2,"model_0_x0":10.,"weight_0":1.,"noise_scale":2.});
    let f=KernelBinding::new().fit_model("mixture",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"components":["logistic"]},"fit_options":{"initial_parameters":initial,"component_max_iterations":1}})).unwrap().result.unwrap();
    assert_eq!(f["optimizer"]["converged"], false);
    assert_eq!(
        f["optimizer"]["termination_reason"],
        "component_update_failed_rolled_back"
    );
    assert_eq!(f["parameters"], initial);
    assert_eq!(f["component_updates"][0]["accepted"], false);
    assert_eq!(
        f["optimizer"]["objective_history"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn invalid_mixture_options_weights_and_noise_fail_closed() {
    let base = json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":{"components":["logistic","logistic"]},"parameters":parameters()});
    for o in [
        json!({"components":[]}),
        json!({"components":["unknown"]}),
        json!({"components":["logistic","logistic"],"weights":[0.,0.]}),
        json!({"components":["logistic","logistic"],"weights":[1.]}),
        json!({"components":["logistic","logistic"],"mode":"unknown"}),
        json!({"components":["logistic","logistic"],"noise_floor":0.}),
    ] {
        let mut p = base.clone();
        p["constructor_kwargs"] = o;
        assert!(KernelBinding::new().predict_model("mixture", p).is_err());
    }
    for (key, value) in [
        ("weight_0", json!(-0.1)),
        ("noise_scale", json!(0.)),
        ("model_0_k", json!(-1.)),
    ] {
        let mut p = base.clone();
        p["parameters"][key] = value;
        assert!(KernelBinding::new().predict_model("mixture", p).is_err());
    }
}

#[test]
fn gaussian_default_initialization_recovers_separated_components_without_a_warm_start() {
    let t: Vec<_> = (0..50).map(|i| i as f64 * 0.5).collect();
    let y: Vec<_> = t
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i % 2 == 0 {
                100. / (1. + (-0.4 * (t - 6.)).exp())
            } else {
                200. / (1. + (-0.3 * (t - 10.)).exp())
            }
        })
        .collect();
    let f=KernelBinding::new().fit_model("mixture",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"components":["logistic","logistic"],"noise_floor":0.001}})).unwrap().result.unwrap();
    assert_eq!(
        f["optimizer"]["converged"],
        true,
        "optimizer: {}; final update: {:?}",
        f["optimizer"],
        f["component_updates"]
            .as_array()
            .and_then(|updates| updates.last())
    );
    let mut capacities = [
        f["parameters"]["model_0_L"].as_f64().unwrap(),
        f["parameters"]["model_1_L"].as_f64().unwrap(),
    ];
    capacities.sort_by(f64::total_cmp);
    assert!((capacities[0] - 100.).abs() < 0.05);
    assert!((capacities[1] - 200.).abs() < 0.05);
}
