use innovate_core::{KernelBinding, json};
#[test]
fn historical_and_centered_group_predictions_retain_distinct_parameter_meanings() {
    let p = json!({"inputs":{"time":[0.,2.,4.]},"constructor_kwargs":{"base_model":"logistic","groups":["a","b"]},"parameters":{"global_L":100.,"global_k":0.5,"global_x0":2.,"a_L":20.,"a_k":0.1,"a_x0":1.,"b_L":-20.,"b_k":-0.1,"b_x0":-1.}});
    let a = KernelBinding::new()
        .predict_model("hierarchical", p)
        .unwrap()
        .result
        .unwrap();
    let expected = 120. / (1. + 0.6_f64.exp()) + 80. / (1. + (-0.4_f64).exp());
    assert!((a["values"][1].as_f64().unwrap() - expected).abs() < 1e-10);
    assert_eq!(a["group_rows"][1].as_array().unwrap().len(), 2);
}
#[test]
fn centered_group_fits_recover_independent_curves_and_saved_group_output() {
    let t: Vec<_> = (0..25).map(|i| i as f64).collect();
    let y: Vec<Vec<_>> = t
        .iter()
        .map(|t| {
            vec![
                100. / (1. + (-0.4 * (t - 7.)).exp()),
                200. / (1. + (-0.3 * (t - 9.)).exp()),
            ]
        })
        .collect();
    let f=KernelBinding::new().fit_model("hierarchical",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"base_model":"logistic","groups":["a","b"],"output":"groups"}})).unwrap().result.unwrap();
    assert!((f["parameters"]["global_L"].as_f64().unwrap() - 150.).abs() < 0.01);
    assert!((f["parameters"]["a_L"].as_f64().unwrap() + 50.).abs() < 0.01);
    let a = KernelBinding::new()
        .predict_model(
            "hierarchical",
            json!({"inputs":{"time":[30.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((a["rows"][0][0].as_f64().unwrap() - 100. / (1. + (-9.2_f64).exp())).abs() < 0.01);
    assert!((a["aggregate_values"][0].as_f64().unwrap() - 300.).abs() < 0.5);
}
#[test]
fn shared_observations_fit_once_with_zero_group_offsets() {
    let t: Vec<_> = (0..20).map(|i| i as f64).collect();
    let y: Vec<_> = t
        .iter()
        .map(|t| 100. / (1. + (-0.5 * (t - 6.)).exp()))
        .collect();
    let f=KernelBinding::new().fit_model("hierarchical",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"base_model":"logistic","groups":["a","b"]}})).unwrap().result.unwrap();
    assert_eq!(f["parameters"]["a_L"], 0.);
    assert_eq!(f["parameters"]["b_k"], 0.);
    let a = KernelBinding::new()
        .predict_model(
            "hierarchical",
            json!({"inputs":{"time":[6.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((a["values"][0].as_f64().unwrap() - 100.).abs() < 0.01);
}

#[test]
fn historical_group_mapping_keeps_absolute_estimates_and_adds_global_means() {
    let t: Vec<_> = (0..25).map(|i| i as f64).collect();
    let a: Vec<_> = t
        .iter()
        .map(|t| 100. / (1. + (-0.4 * (t - 7.)).exp()))
        .collect();
    let b: Vec<_> = t
        .iter()
        .map(|t| 200. / (1. + (-0.3 * (t - 9.)).exp()))
        .collect();
    let f=KernelBinding::new().fit_model("hierarchical",json!({"inputs":{"time":t,"observed":{"a":a,"b":b}},"constructor_kwargs":{"base_model":"logistic","groups":["a","b"],"parameterization":"historical_additive"}})).unwrap().result.unwrap();
    assert!((f["parameters"]["a_L"].as_f64().unwrap() - 100.).abs() < 0.01);
    assert!((f["parameters"]["global_L"].as_f64().unwrap() - 150.).abs() < 0.01);
    let a = KernelBinding::new()
        .predict_model(
            "hierarchical",
            json!({"inputs":{"time":[8.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let p = &f["parameters"];
    let expected = ["a", "b"]
        .iter()
        .map(|g| {
            let v = |name: &str| {
                p[format!("global_{name}")].as_f64().unwrap()
                    + p[format!("{g}_{name}")].as_f64().unwrap()
            };
            v("L") / (1. + (-v("k") * (8. - v("x0"))).exp())
        })
        .sum::<f64>();
    assert!((a["values"][0].as_f64().unwrap() - expected).abs() < 1e-10);
    let summary = KernelBinding::new()
        .summarize_model("hierarchical", json!({"state":f["state"]}))
        .unwrap()
        .result
        .unwrap();
    assert!(summary.get("predictions").is_none());
    assert!(f["diagnostics"]["metrics"]["AIC"].is_null());
}
#[test]
fn invalid_groups_parameters_and_observation_shapes_fail_closed() {
    let base = json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":{"base_model":"logistic","groups":["a"]},"parameters":{"global_L":100.,"global_k":1.,"global_x0":2.,"a_L":0.,"a_k":0.,"a_x0":0.}});
    for groups in [json!([]), json!(["a", "a"]), json!(["global"]), json!([""])] {
        let mut p = base.clone();
        p["constructor_kwargs"]["groups"] = groups;
        assert!(
            KernelBinding::new()
                .predict_model("hierarchical", p)
                .is_err()
        );
    }
    for y in [
        json!({"missing":[1.,2.]}),
        json!([[1., 2.], [3., 4.]]),
        json!([1.]),
        json!([1., -1.]),
    ] {
        let mut p = base.clone();
        p["inputs"]["observed"] = y;
        assert!(KernelBinding::new().fit_model("hierarchical", p).is_err());
    }
    for (key, value) in [
        ("output", json!("unknown")),
        ("parameterization", json!("unknown")),
        ("unknown", json!(true)),
    ] {
        let mut p = base.clone();
        p["constructor_kwargs"][key] = value;
        assert!(
            KernelBinding::new()
                .predict_model("hierarchical", p)
                .is_err()
        );
    }
    let mut p = base.clone();
    p["parameters"]["a_k"] = json!(-2.);
    assert!(
        KernelBinding::new()
            .predict_model("hierarchical", p)
            .is_err()
    );
    let mut p = base.clone();
    p["parameters"]["global_L"] = json!(1e308);
    p["parameters"]["a_L"] = json!(1e308);
    assert!(
        KernelBinding::new()
            .predict_model("hierarchical", p)
            .is_err()
    );
}
#[test]
fn large_constant_diagnostics_do_not_invent_explained_variation() {
    let p = json!({"inputs":{"time":[0.,1.],"observed":[1e308,1e308]},"constructor_kwargs":{"base_model":"logistic","groups":["a"]},"parameters":{"global_L":1e308,"global_k":1.,"global_x0":-100.,"a_L":0.,"a_k":0.,"a_x0":0.}});
    let d = KernelBinding::new()
        .diagnose_model("hierarchical", p)
        .unwrap()
        .result
        .unwrap();
    assert!(d["diagnostics"]["metrics"]["R_squared"].is_null());
}
