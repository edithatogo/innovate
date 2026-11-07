use innovate_core::{KernelBinding, json};
fn payload(mode: &str) -> serde_json::Value {
    json!({"inputs":{"time":[0.,1.,2.,3.,4.]},"constructor_kwargs":{"base_model":"logistic","prediction_mode":mode,"split_time":2.},"parameters":{"changepoint_index":2.,"regime_0_L":100.,"regime_0_k":1.,"regime_0_x0":2.,"regime_1_L":200.,"regime_1_k":0.5,"regime_1_x0":2.}})
}
#[test]
fn historical_index_split_retains_batch_dependent_offset_and_singletons() {
    let p = payload("historical_index");
    let a = KernelBinding::new()
        .predict_model("regime_switching", p.clone())
        .unwrap()
        .result
        .unwrap();
    let first = 100. / (1. + 1_f64.exp());
    assert!((a["values"][1].as_f64().unwrap() - first).abs() < 1e-10);
    assert_eq!(a["values"][1], a["values"][2]);
    let mut p = p;
    p["inputs"] = json!({"time":[10.]});
    let a = KernelBinding::new()
        .predict_model("regime_switching", p)
        .unwrap()
        .result
        .unwrap();
    assert!((a["values"][0].as_f64().unwrap() - 100. / (1. + (-8_f64).exp())).abs() < 1e-10);
}
#[test]
fn fixed_time_switch_is_continuous_and_does_not_reset_on_forecast_batches() {
    let p = payload("fixed_time");
    let a = KernelBinding::new()
        .predict_model("regime_switching", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(a["values"][2], 50.);
    let mut p = p;
    p["inputs"] = json!({"time":[3.,4.]});
    let b = KernelBinding::new()
        .predict_model("regime_switching", p)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(b["values"], json!([a["values"][3], a["values"][4]]));
}
#[test]
fn fitted_regimes_retain_fixed_boundary_and_segment_parameters() {
    let time: Vec<_> = (0..30).map(|i| i as f64).collect();
    let y: Vec<_> = time
        .iter()
        .map(|t| {
            if *t < 15. {
                100. / (1. + (-0.4 * (t - 6.)).exp())
            } else {
                200. / (1. + (-0.3 * (t - 18.)).exp())
            }
        })
        .collect();
    let f=KernelBinding::new().fit_model("regime_switching",json!({"inputs":{"time":time,"observed":y},"constructor_kwargs":{"base_model":"logistic","prediction_mode":"fixed_time","split_time":14.5}})).unwrap().result.unwrap();
    assert_eq!(f["parameters"]["changepoint_index"], 15);
    assert_eq!(f["state"]["constructor_kwargs"]["split_time"], 14.5);
    assert!((f["parameters"]["regime_0_k"].as_f64().unwrap() - 0.4).abs() < 0.001);
    assert!((f["parameters"]["regime_1_k"].as_f64().unwrap() - 0.3).abs() < 0.001);
    assert!(f["diagnostics"]["metrics"]["AIC"].is_null());
    let a = KernelBinding::new()
        .predict_model(
            "regime_switching",
            json!({"inputs":{"time":[40.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!(a["values"][0].as_f64().unwrap() > 100.);
}

#[test]
fn automatic_detection_and_short_single_regime_are_explicit() {
    let time: Vec<_> = (0..30).map(|i| i as f64).collect();
    let y: Vec<_> = time
        .iter()
        .map(|t| 100. / (1. + (-0.4 * (t - 8.)).exp()))
        .collect();
    let f=KernelBinding::new().fit_model("regime_switching",json!({"inputs":{"time":time,"observed":y},"constructor_kwargs":{"base_model":"logistic"}})).unwrap().result.unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/regime_ruptures.json")).unwrap();
    let expected = &fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "smooth_curve")
        .unwrap()["breakpoints"];
    assert_eq!(&f["details"]["detected_breakpoints"], expected);
    assert_eq!(f["parameters"]["changepoint_index"], expected[0]);
    let f=KernelBinding::new().fit_model("regime_switching",json!({"inputs":{"time":[0.,1.],"observed":[0.11920292202211755,0.2689414213699951]},"constructor_kwargs":{"base_model":"fisher_pry"}})).unwrap().result.unwrap();
    assert_eq!(f["details"]["regime_count"], 1);
    assert_eq!(f["parameters"]["changepoint_index"], 1);
    let s = KernelBinding::new()
        .summarize_model("regime_switching", json!({"state":f["state"]}))
        .unwrap()
        .result
        .unwrap();
    assert!(s.get("predictions").is_none());
    assert_eq!(s["details"]["regime_count"], 1);
}
#[test]
fn malformed_regime_state_and_options_fail_closed() {
    for kw in [
        json!({"prediction_mode":"unknown"}),
        json!({"regime_count":3}),
        json!({"detector_jump":0}),
        json!({"detector_penalty":-1.}),
        json!({"unknown":true}),
    ] {
        let mut p = payload("fixed_time");
        p["constructor_kwargs"] = kw;
        assert!(
            KernelBinding::new()
                .predict_model("regime_switching", p)
                .is_err()
        );
    }
    for i in [0., -1., 1.5, 1e12] {
        let mut p = payload("fixed_time");
        p["parameters"]["changepoint_index"] = json!(i);
        assert!(
            KernelBinding::new()
                .predict_model("regime_switching", p)
                .is_err()
        );
    }
    let mut p = payload("fixed_time");
    p["constructor_kwargs"]
        .as_object_mut()
        .unwrap()
        .remove("split_time");
    assert!(
        KernelBinding::new()
            .predict_model("regime_switching", p)
            .is_err()
    );
    let mut p = payload("historical_index");
    p["inputs"]["covariates"] = json!({});
    assert!(
        KernelBinding::new()
            .predict_model("regime_switching", p)
            .is_err()
    );
    let mut p = payload("fixed_time");
    p["parameters"]["regime_1_k"] = json!(-1.);
    assert!(
        KernelBinding::new()
            .predict_model("regime_switching", p)
            .is_err()
    );
}

#[test]
fn two_point_fisher_pry_fit_is_identified_but_has_no_residual_degrees_of_freedom() {
    let f = KernelBinding::new()
        .fit_model(
            "fisher_pry",
            json!({"inputs":{"time":[0.,1.],"observed":[0.11920292202211755,0.2689414213699951]}}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((f["parameters"]["alpha"].as_f64().unwrap() - 1.).abs() < 1e-8);
    assert!((f["parameters"]["t0"].as_f64().unwrap() - 2.).abs() < 1e-8);
    assert!(f["diagnostics"]["metrics"]["AIC"].is_null());
    assert!(
        f["diagnostics"]["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v
                .as_str()
                .unwrap()
                .contains("no residual degrees of freedom"))
    );
}
