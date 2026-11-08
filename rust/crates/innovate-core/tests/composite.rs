use innovate_core::{KernelBinding, json};
#[test]
fn independent_composite_growth_matches_closed_forms_and_rhs_rates() {
    let time = [0_f64, 3., 10., 20.];
    let a=KernelBinding::new().predict_model("composite",json!({"inputs":{"time":time},"constructor_kwargs":{"components":["bass","logistic"],"initial_values":[0.,10.]},"parameters":{"p_1":0.03,"q_1":0.35,"m_1":100.,"L_2":200.,"k_2":0.4,"x0_2":99.,"alpha_1_2":0.,"alpha_2_1":0.}})).unwrap().result.unwrap();
    for (i, t) in time.iter().enumerate() {
        let e = (-0.38 * t).exp();
        let bass = 100. * (1. - e) / (1. + 0.35 / 0.03 * e);
        let logistic = 200. / (1. + 19. * (-0.4 * t).exp());
        assert!((a["rows"][i][0].as_f64().unwrap() - bass).abs() < 1e-6);
        assert!((a["rows"][i][1].as_f64().unwrap() - logistic).abs() < 1e-6);
        assert!(
            (a["rates"][i][1].as_f64().unwrap() - 0.4 * logistic * (1. - logistic / 200.)).abs()
                < 1e-6
        );
    }
}
#[test]
fn composite_fit_retains_conditioned_initial_state_and_forecasts_from_origin() {
    let time: Vec<_> = (0..25).map(|i| i as f64).collect();
    let y: Vec<_> = time
        .iter()
        .map(|t| {
            let e = (-0.38 * t).exp();
            vec![100. * (1. - e) / (1. + 0.35 / 0.03 * e)]
        })
        .collect();
    let f=KernelBinding::new().fit_model("composite",json!({"inputs":{"time":time,"observed":y},"constructor_kwargs":{"components":["bass"]},"fit_options":{"initial_parameters":{"p_1":0.025,"q_1":0.32,"m_1":110.}}})).unwrap().result.unwrap();
    assert_eq!(f["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(
        f["diagnostics"]["initial_condition"]["excluded_conditioning_rows"],
        1
    );
    let a = KernelBinding::new()
        .predict_model(
            "composite",
            json!({"inputs":{"time":[30.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let e = (-0.38_f64 * 30.).exp();
    assert!(
        (a["rows"][0][0].as_f64().unwrap() - 100. * (1. - e) / (1. + 0.35 / 0.03 * e)).abs()
            < 0.005
    );
}
#[test]
fn historical_batch_reset_is_explicit_and_fixed_origin_batches_match() {
    let p = json!({"inputs":{"time":[0.,10.,20.]},"constructor_kwargs":{"components":["bass"]},"parameters":{"p_1":0.03,"q_1":0.35,"m_1":100.}});
    let a = KernelBinding::new()
        .predict_model("composite", p.clone())
        .unwrap()
        .result
        .unwrap();
    let mut p = p;
    p["inputs"] = json!({"time":[10.,20.]});
    let b = KernelBinding::new()
        .predict_model("composite", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert!((a["rows"][1][0].as_f64().unwrap() - b["rows"][0][0].as_f64().unwrap()).abs() < 1e-6);
    p["constructor_kwargs"]["origin_mode"] = json!("historical_batch");
    let c = KernelBinding::new()
        .predict_model("composite", p)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(c["rows"][0][0], 0.);
}

#[test]
fn heterogeneous_signed_interactions_match_independent_scipy_and_absent_parameters_stay_fixed() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/composite_scipy.json")).unwrap();
    let p = json!({"inputs":{"time":f["time"]},"constructor_kwargs":{"components":["bass","logistic","gompertz"],"initial_values":f["initial_values"]},"parameters":f["parameters"]});
    let a = KernelBinding::new()
        .predict_model("composite", p.clone())
        .unwrap()
        .result
        .unwrap();
    for key in ["rows", "rates"] {
        for (actual, expected) in a[key]
            .as_array()
            .unwrap()
            .iter()
            .zip(f[key].as_array().unwrap())
        {
            for (x, y) in actual
                .as_array()
                .unwrap()
                .iter()
                .zip(expected.as_array().unwrap())
            {
                assert!((x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-6);
            }
        }
    }
    let mut p = p;
    p["parameters"]["x0_2"] = json!(-100.);
    p["parameters"]["b_3"] = json!(0.);
    assert_eq!(
        a["rows"],
        KernelBinding::new()
            .predict_model("composite", p)
            .unwrap()
            .result
            .unwrap()["rows"]
    );
}
#[test]
fn signed_negative_levels_are_returned_and_fisher_pry_ode_has_a_native_adapter() {
    let a=KernelBinding::new().predict_model("composite",json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":{"components":["logistic","logistic"],"initial_values":[0.,5.]},"parameters":{"L_1":100.,"k_1":0.,"x0_1":0.,"L_2":100.,"k_2":0.,"x0_2":0.,"alpha_1_2":1.,"alpha_2_1":0.}})).unwrap().result.unwrap();
    assert!((a["rows"][1][0].as_f64().unwrap() + 5.).abs() < 1e-10);
    assert_eq!(a["rates"][1][0], -5.);
    assert_eq!(a["metadata"]["warnings"].as_array().unwrap().len(), 1);
    let a=KernelBinding::new().predict_model("composite",json!({"inputs":{"time":[0.,10.]},"constructor_kwargs":{"components":["fisher_pry"],"initial_values":[0.05]},"parameters":{"alpha_1":0.4,"t0_1":99.}})).unwrap().result.unwrap();
    assert!((a["rows"][1][0].as_f64().unwrap() - 1. / (1. + 19. * (-4_f64).exp())).abs() < 1e-8);
}
#[test]
fn composite_invalid_shapes_origins_domains_and_unknown_options_fail() {
    let base = json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":{"components":["bass"]},"parameters":{"p_1":0.03,"q_1":0.35,"m_1":100.}});
    for o in [
        json!({"components":[]}),
        json!({"components":["unknown"]}),
        json!({"components":["bass"],"initial_values":[0.,1.]}),
        json!({"components":["bass"],"initial_time":2.}),
        json!({"components":["bass"],"alpha":[[1.]]}),
        json!({"components":["bass"],"origin_mode":"historical_batch","initial_values":[0.]}),
        json!({"components":["bass"],"unknown":true}),
    ] {
        let mut p = base.clone();
        p["constructor_kwargs"] = o;
        assert!(KernelBinding::new().predict_model("composite", p).is_err());
    }
    let mut p = base.clone();
    p["parameters"]["m_1"] = json!(0.);
    assert!(KernelBinding::new().predict_model("composite", p).is_err());
    let summary=KernelBinding::new().summarize_model("composite",json!({"constructor_kwargs":base["constructor_kwargs"],"parameters":base["parameters"]})).unwrap().result.unwrap();
    assert!(summary.get("predictions").is_none());
}

#[test]
fn coupled_fit_estimates_signed_interactions_and_reports_absent_parameters() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/composite_scipy.json")).unwrap();
    let mut initial = f["parameters"].clone();
    initial["alpha_1_2"] = json!(0.);
    let fit=KernelBinding::new().fit_model("composite",json!({"inputs":{"time":f["time"],"observed":f["rows"]},"constructor_kwargs":{"components":["bass","logistic","gompertz"],"initial_values":f["initial_values"]},"fit_options":{"initial_parameters":initial}})).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-5);
    assert_eq!(fit["parameters"]["x0_2"], 99.);
    assert_eq!(fit["parameters"]["b_3"], 77.);
    assert_eq!(
        fit["diagnostics"]["structural_identifiability"]["absent_from_rhs"],
        json!(["x0_2", "b_3"])
    );
}

#[test]
fn stimulated_capacity_fit_can_use_positive_bounds_or_explicit_historical_floors() {
    let parameters = json!({"p_1":0.03,"q_1":0.35,"m_1":100.,"L_2":100.,"k_2":0.,"x0_2":0.,"alpha_1_2":-1.,"alpha_2_1":0.});
    let options = json!({"components":["bass","logistic"],"initial_values":[0.,5.]});
    let time = json!([0., 5., 10., 20., 30.]);
    let a = KernelBinding::new()
        .predict_model(
            "composite",
            json!({"inputs":{"time":time},"constructor_kwargs":options,"parameters":parameters}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!(a["rows"][4][0].as_f64().unwrap() > 100.);
    let mut p = json!({"inputs":{"time":time,"observed":a["rows"]},"constructor_kwargs":options,"fit_options":{"initial_parameters":parameters,"max_iterations":1}});
    assert!(
        KernelBinding::new()
            .fit_model("composite", p.clone())
            .is_ok()
    );
    p["constructor_kwargs"]["capacity_bounds"] = json!("historical_observed");
    let error = KernelBinding::new().fit_model("composite", p).unwrap_err();
    assert!(error.message.contains("initial capacities"));
}
