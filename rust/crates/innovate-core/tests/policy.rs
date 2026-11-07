use innovate_core::{KernelBinding, json};
#[test]
fn policy_events_use_exact_boundaries_decay_and_historical_cumulative_max() {
    let time = [0_f64, 2., 3., 4., 10.];
    let a=KernelBinding::new().predict_model("policy_hazard",json!({"inputs":{"time":time},"constructor_kwargs":{"base_model":"logistic","event_times":[2.,4.],"event_effects":[0.5,-0.2]},"parameters":{"decay":0.5,"base_L":100.,"base_k":0.5,"base_x0":3.}})).unwrap().result.unwrap();
    let mut maximum = 0_f64;
    for (i, t) in time.iter().enumerate() {
        let effect = if *t >= 2. {
            0.5 * (-0.5 * (t - 2.)).exp()
        } else {
            0.
        } + if *t >= 4. {
            -0.2 * (-0.5 * (t - 4.)).exp()
        } else {
            0.
        };
        maximum = maximum.max(100. / (1. + (-0.5 * (t - 3.)).exp()) * (1. + effect));
        assert!((a["values"][i].as_f64().unwrap() - maximum).abs() < 1e-10);
    }
}
#[test]
fn historical_policy_fit_holds_decay_and_preserves_forecast_state() {
    let time: Vec<_> = (0..26).map(|i| i as f64).collect();
    let y: Vec<_> = time
        .iter()
        .map(|t| {
            let d = (-0.33 * t).exp();
            100. * (1. - d) / (1. + 0.3 / 0.03 * d)
        })
        .collect();
    let fit = KernelBinding::new()
        .fit_model(
            "policy_hazard",
            json!({"inputs":{"time":time,"observed":y},"constructor_kwargs":{"decay":0.3}}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["parameters"]["decay"], 0.3);
    assert_eq!(fit["base_diagnostics"]["optimizer"]["converged"], true);
    let r = KernelBinding::new()
        .predict_model(
            "policy_hazard",
            json!({"inputs":{"time":[30.]},"state":fit["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let d = (-0.33_f64 * 30.).exp();
    assert!((r["values"][0].as_f64().unwrap() - 100. * (1. - d) / (1. + 10. * d)).abs() < 0.01);
}
#[test]
fn no_time_policy_summary_retains_events_and_clips_negative_multipliers() {
    let p = json!({"constructor_kwargs":{"base_model":"logistic","event_times":[0.],"event_effects":[-2.],"event_labels":["negative"]},"parameters":{"decay":0.,"base_L":100.,"base_k":1.,"base_x0":2.}});
    let summary = KernelBinding::new()
        .summarize_model("policy_hazard", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(summary["details"]["event_count"], 1);
    assert_eq!(summary["details"]["average_effect"], -2.);
    assert!(summary.get("predictions").is_none());
    let mut p = p;
    p["inputs"] = json!({"time":[0.,2.,4.]});
    assert_eq!(
        KernelBinding::new()
            .predict_model("policy_hazard", p)
            .unwrap()
            .result
            .unwrap()["values"],
        json!([0., 0., 0.])
    );
}
#[test]
fn invalid_event_shapes_decay_and_unknown_fields_fail() {
    for kw in [
        json!({"event_times":[1.],"event_effects":[]}),
        json!({"event_times":[1.],"event_effects":[0.2],"event_labels":["a","b"]}),
        json!({"unknown":true}),
    ] {
        assert!(KernelBinding::new().predict_model("policy_hazard",json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":kw,"parameters":{"decay":0.5,"base_p":0.03,"base_q":0.3,"base_m":100.}})).is_err());
    }
    assert!(KernelBinding::new().predict_model("policy_hazard",json!({"inputs":{"time":[0.,1.]},"parameters":{"decay":-0.5,"base_p":0.03,"base_q":0.3,"base_m":100.}})).is_err());
}
