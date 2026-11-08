use innovate_core::{KernelBinding, json};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/competitive_scipy.json")).unwrap()
}
#[test]
fn competitive_native_odes_match_independent_scipy_and_forecast_from_saved_origin() {
    let f = fixture();
    for model in ["lotka_volterra", "multi_product"] {
        let c = &f["cases"][model];
        let options = json!({"initial_time":0.,"initial_values":c["initial"]});
        let r=KernelBinding::new().predict_model(model,json!({"inputs":{"time":c["time"]},"parameters":c["parameters"],"constructor_kwargs":options})).unwrap().result.unwrap();
        for key in ["rows", "rates"] {
            for (a, b) in r[key]
                .as_array()
                .unwrap()
                .iter()
                .zip(c[key].as_array().unwrap())
            {
                for (a, b) in a.as_array().unwrap().iter().zip(b.as_array().unwrap()) {
                    assert!(
                        (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-5,
                        "{model} {key}"
                    );
                }
            }
        }
        let forecast=KernelBinding::new().predict_model(model,json!({"inputs":{"time":[25.]},"parameters":c["parameters"],"constructor_kwargs":options})).unwrap().result.unwrap();
        for (a, b) in forecast["rows"][0]
            .as_array()
            .unwrap()
            .iter()
            .zip(c["rows"][8].as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-5);
        }
    }
}
#[test]
fn competitive_fits_report_quality_and_keep_the_initial_condition_for_forecasts() {
    let f = fixture();
    for model in ["lotka_volterra", "multi_product"] {
        let c = &f["cases"][model];
        let fit=KernelBinding::new().fit_model(model,json!({"inputs":{"time":c["time"],"observed":c["rows"]},"fit_options":{"initial_parameters":c["parameters"]}})).unwrap().result.unwrap();
        assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true, "{fit}");
        assert_eq!(
            fit["diagnostics"]["initial_condition"]["excluded_conditioning_rows"],
            1
        );
        assert_eq!(fit["diagnostics"]["observation_shape"], json!([8, 2]));
        assert_eq!(
            fit["diagnostics"]["identifiability"]["rank"],
            if model == "lotka_volterra" { 4 } else { 8 }
        );
        for (a, b) in fit["state"]["constructor_kwargs"]["initial_values"]
            .as_array()
            .unwrap()
            .iter()
            .zip(c["initial"].as_array().unwrap())
        {
            assert_eq!(a.as_f64(), b.as_f64());
        }
        let r = KernelBinding::new()
            .predict_model(model, json!({"inputs":{"time":[25.]},"state":fit["state"]}))
            .unwrap()
            .result
            .unwrap();
        for (a, b) in r["rows"][0]
            .as_array()
            .unwrap()
            .iter()
            .zip(c["rows"][8].as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-4);
        }
    }
}
#[test]
fn competitive_domain_and_unsupported_covariates_are_explicit_errors() {
    let f = fixture();
    let c = &f["cases"]["lotka_volterra"];
    for options in [
        json!({}),
        json!({"initial_values":[0.1]}),
        json!({"initial_values":[0.1,-0.1]}),
        json!({"initial_values":[0.1,0.1],"unknown":true}),
        json!({"initial_values":[0.1,0.1],"covariates":["x"]}),
    ] {
        assert!(KernelBinding::new().predict_model("lotka_volterra",json!({"inputs":{"time":[0.,1.]},"parameters":c["parameters"],"constructor_kwargs":options})).is_err());
    }
}

#[test]
fn lotka_volterra_fits_without_seed_and_decoupled_products_match_bass() {
    let f = fixture();
    let c = &f["cases"]["lotka_volterra"];
    let fit = KernelBinding::new()
        .fit_model(
            "lotka_volterra",
            json!({"inputs":{"time":c["time"],"observed":c["rows"]}}),
        )
        .unwrap()
        .result
        .unwrap();
    for (k, v) in c["parameters"].as_object().unwrap() {
        assert!(
            (fit["parameters"][k].as_f64().unwrap() - v.as_f64().unwrap()).abs() < 1e-4,
            "{k}: {fit}"
        );
    }
    for op in [
        innovate_core::KernelOperation::SummarizeModel,
        innovate_core::KernelOperation::DiagnoseModel,
        innovate_core::KernelOperation::SimulateModel,
    ] {
        let mut request = innovate_core::KernelRequest::predict_model(
            "lotka_volterra",
            json!({"inputs":{"time":c["time"],"observed":c["rows"]},"state":fit["state"]}),
        );
        request.operation = op;
        let result = KernelBinding::new()
            .invoke(&request)
            .unwrap()
            .result
            .unwrap();
        if op != innovate_core::KernelOperation::SimulateModel {
            assert!(result["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10);
        }
    }
    let mut p = json!({});
    for i in 1..=3 {
        p[format!("p{i}")] = json!(0.03);
        p[format!("q{i}")] = json!(0.35);
        p[format!("m{i}")] = json!(100. * i as f64);
        for j in 1..=3 {
            if i != j {
                p[format!("alpha_{i}_{j}")] = json!(0.);
            }
        }
    }
    let r = KernelBinding::new()
        .predict_model(
            "multi_product",
            json!({"inputs":{"time":[0.,1.,5.,20.]},"parameters":p}),
        )
        .unwrap()
        .result
        .unwrap();
    for (row, t) in r["rows"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0_f64, 1., 5., 20.])
    {
        let f = (1. - (-0.38 * t).exp()) / (1. + 0.35 / 0.03 * (-0.38 * t).exp());
        for i in 0..3 {
            assert!((row[i].as_f64().unwrap() - 100. * (i + 1) as f64 * f).abs() < 1e-6);
        }
    }
}

#[test]
fn competitive_fit_reports_limits_and_rejects_invalid_data() {
    let c = &fixture()["cases"]["lotka_volterra"];
    let fit=KernelBinding::new().fit_model("lotka_volterra",json!({"inputs":{"time":c["time"],"observed":c["rows"]},"fit_options":{"max_iterations":1}})).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], false);
    for options in [json!({"max_iterations":0}), json!({"unknown":true})] {
        assert!(
            KernelBinding::new()
                .fit_model(
                    "lotka_volterra",
                    json!({"inputs":{"time":c["time"],"observed":c["rows"]},"fit_options":options})
                )
                .is_err()
        );
    }
    for observed in [
        json!([[0., 0.], [0., 0.], [0., 0.], [0., 0.]]),
        json!([[0.1, 0.1], [-0.1, 0.2], [0.3, 0.3], [0.4, 0.4]]),
        json!([[1.], [2.], [3.], [4.]]),
    ] {
        assert!(
            KernelBinding::new()
                .fit_model(
                    "lotka_volterra",
                    json!({"inputs":{"time":[0.,1.,2.,3.],"observed":observed}})
                )
                .is_err()
        );
    }
    let p = c["parameters"].clone();
    for options in [
        json!({"initial_values":[0.1,0.2],"initial_time":1.}),
        json!({"initial_values":[0.1,0.2],"n_products":2}),
        json!({"initial_values":[0.1,1.2]}),
    ] {
        let e = KernelBinding::new()
            .predict_model(
                "lotka_volterra",
                json!({"inputs":{"time":[0.,1.]},"parameters":p,"constructor_kwargs":options}),
            )
            .unwrap_err();
        assert_eq!(e.code, "invalid_request");
    }
}

#[test]
fn lotka_volterra_forcing_matches_integrated_logistic_growth_and_saved_history() {
    let request = json!({
        "inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},
        "parameters":{"alpha1":0.2,"beta1":0.,"alpha2":0.1,"beta2":0.,
            "beta_alpha1_price":0.1,"beta_beta1_price":0.,
            "beta_alpha2_price":-0.02,"beta_beta2_price":0.},
        "constructor_kwargs":{"initial_values":[0.1,0.2],"covariates":["price"]}
    });
    let result = KernelBinding::new()
        .predict_model("lotka_volterra", request)
        .unwrap()
        .result
        .unwrap();
    let expected = 1. / (1. + 9. * (-0.6_f64).exp());
    assert!((result["rows"][2][0].as_f64().unwrap() - expected).abs() < 1e-8);
    assert!(
        (result["rates"][2][0].as_f64().unwrap() - 0.4 * expected * (1. - expected)).abs() < 1e-8
    );
    let forecast = KernelBinding::new()
        .predict_model(
            "lotka_volterra",
            json!({
                "inputs":{"time":[3.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert!(
        (forecast["rows"][0][0].as_f64().unwrap() - 1. / (1. + 9. * (-1.0_f64).exp())).abs() < 1e-8
    );
    assert!(
        KernelBinding::new()
            .predict_model(
                "lotka_volterra",
                json!({
                    "inputs":{"time":[1.],"covariates":{"price":[0.]}}, "state":result["state"]
                })
            )
            .is_err()
    );
}

#[test]
fn interacting_lotka_forcing_fits_independent_reference_and_forecasts() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/lotka_forcing_scipy.json")).unwrap();
    let options = json!({"covariates":["price"],"initial_values":fixture["initial"]});
    let payload = json!({"inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
        "parameters":fixture["parameters"],"constructor_kwargs":options});
    let binding = KernelBinding::new();
    let prediction = binding
        .predict_model("lotka_volterra", payload)
        .unwrap()
        .result
        .unwrap();
    for quantity in ["rows", "rates"] {
        for (actual, expected) in prediction[quantity]
            .as_array()
            .unwrap()
            .iter()
            .zip(fixture[quantity].as_array().unwrap())
        {
            for (actual, expected) in actual
                .as_array()
                .unwrap()
                .iter()
                .zip(expected.as_array().unwrap())
            {
                assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-6);
            }
        }
    }
    let mut start = fixture["parameters"].clone();
    start["alpha1"] = json!(0.29);
    start["beta_beta1_price"] = json!(-0.025);
    let fit = binding.fit_model("lotka_volterra",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
        "constructor_kwargs":options,"fit_options":{"initial_parameters":start,"max_iterations":150}
    })).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10);
    assert_eq!(fit["parameters"].as_object().unwrap().len(), 8);
    let forecast = binding
        .predict_model(
            "lotka_volterra",
            json!({
                "inputs":{"time":fixture["forecast_time"]},"state":fit["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    for (actual, expected) in forecast["rows"][0]
        .as_array()
        .unwrap()
        .iter()
        .zip(fixture["forecast_rows"][0].as_array().unwrap())
    {
        assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
    }
    let automatic = binding.fit_model("lotka_volterra",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
        "constructor_kwargs":options
    }));
    let fit = automatic.unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10);
    let optimizer = &fit["diagnostics"]["optimizer"];
    let reports = optimizer["start_reports"].as_array().unwrap();
    assert_eq!(reports.len(), 3);
    let selected = optimizer["selected_start_index"].as_u64().unwrap() as usize;
    assert_eq!(reports[selected]["scaled_rss"], optimizer["scaled_rss"]);
    assert!(
        reports
            .iter()
            .filter_map(|r| r["scaled_rss"].as_f64())
            .all(|rss| rss >= optimizer["scaled_rss"].as_f64().unwrap())
    );
}

#[test]
fn multi_product_forcing_layouts_are_explicit_and_capacity_zero_is_safe() {
    let mut payload = json!({
        "inputs":{"time":[0.],"covariates":{"price":[1.]}},
        "parameters":{"p1":0.1,"p2":0.2,"q1":0.3,"q2":0.4,"m1":100.,"m2":200.,"alpha_1_2":0.,"alpha_2_1":0.,
            "beta_p1_price":0.01,"beta_p2_price":0.02,"beta_q1_price":0.03,"beta_q2_price":0.04,
            "beta_m1_price":1.,"beta_m2_price":2.,"beta_alpha_1_2_price":0.1,"beta_alpha_2_1_price":0.2},
        "constructor_kwargs":{"n_products":2,"initial_values":[10.,20.],"covariates":["price"]}
    });
    let run = |payload| {
        KernelBinding::new()
            .predict_model("multi_product", payload)
            .unwrap()
            .result
            .unwrap()
    };
    let corrected = run(payload.clone());
    let first = (0.11 + 0.33 * 10. / 101.) * (101. - 10. - 0.1 * 20.);
    assert!((corrected["rates"][0][0].as_f64().unwrap() - first).abs() < 1e-10);
    payload["constructor_kwargs"]["coefficient_layout"] = json!("historical");
    let historical = run(payload.clone());
    let first = (0.11 + 0.32 * 10. / 100.03) * (100.03 - 10. - 0.1 * 20.);
    assert!((historical["rates"][0][0].as_f64().unwrap() - first).abs() < 1e-10);
    assert_ne!(historical["rates"], corrected["rates"]);
    payload["constructor_kwargs"]["coefficient_layout"] = json!("corrected");
    payload["parameters"]["beta_m1_price"] = json!(-100.);
    assert_eq!(run(payload.clone())["rates"][0][0], 0.);
    payload["constructor_kwargs"]["coefficient_layout"] = json!("unknown");
    assert!(
        KernelBinding::new()
            .predict_model("multi_product", payload)
            .is_err()
    );
}

#[test]
fn multi_product_forcing_layouts_match_independent_scipy_and_saved_forecasts() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/multi_forcing_scipy.json")).unwrap();
    let binding = KernelBinding::new();
    for layout in ["corrected", "historical"] {
        let result = binding.predict_model("multi_product",json!({
            "inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
            "parameters":fixture["parameters"],
            "constructor_kwargs":{"n_products":2,"covariates":["price"],"coefficient_layout":layout}
        })).unwrap().result.unwrap();
        for quantity in ["rows", "rates"] {
            for (actual, expected) in result[quantity]
                .as_array()
                .unwrap()
                .iter()
                .zip(fixture["cases"][layout][quantity].as_array().unwrap())
            {
                for (actual, expected) in actual
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected.as_array().unwrap())
                {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "{layout} {quantity}"
                    );
                }
            }
        }
        let forecast = binding
            .predict_model(
                "multi_product",
                json!({
                    "inputs":{"time":fixture["forecast_time"]},"state":result["state"]
                }),
            )
            .unwrap()
            .result
            .unwrap();
        for (actual, expected) in forecast["rows"][0].as_array().unwrap().iter().zip(
            fixture["cases"][layout]["forecast_rows"][0]
                .as_array()
                .unwrap(),
        ) {
            assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
        }
        assert_eq!(
            forecast["state"]["constructor_kwargs"]["coefficient_layout"],
            layout
        );
    }
}

#[test]
fn multi_product_covariate_fit_preserves_signed_coefficients() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/multi_forcing_scipy.json")).unwrap();
    let mut start = fixture["parameters"].clone();
    start["p1"] = json!(0.0305);
    let result = KernelBinding::new().fit_model("multi_product",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["cases"]["corrected"]["rows"]},
        "constructor_kwargs":{"n_products":2,"covariates":["price"]},
        "fit_options":{"initial_parameters":start,"max_iterations":200}
    })).unwrap().result.unwrap();
    assert_eq!(result["diagnostics"]["optimizer"]["converged"], true);
    assert!(result["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
    assert_eq!(result["parameters"].as_object().unwrap().len(), 16);
    let forecast = KernelBinding::new()
        .predict_model(
            "multi_product",
            json!({
                "inputs":{"time":fixture["forecast_time"]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    for (actual, expected) in forecast["rows"][0].as_array().unwrap().iter().zip(
        fixture["cases"]["corrected"]["forecast_rows"][0]
            .as_array()
            .unwrap(),
    ) {
        assert!(
            (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-3,
            "actual={actual} expected={expected} fit={result}"
        );
    }
}

#[test]
fn fixed_parameter_events_split_exactly_and_use_right_hand_rates() {
    let discovered = KernelBinding::new().discover_models().unwrap();
    for model in discovered.models {
        assert_eq!(
            model.supports_parameter_events,
            [
                "lotka_volterra",
                "complementary_goods",
                "lock_in",
                "multi_product",
                "norton_bass"
            ]
            .contains(&model.key.as_str())
        );
    }
    let payload = json!({
        "inputs":{"time":[0.,1.,2.,3.]},
        "parameters":{"alpha1":0.2,"beta1":0.,"alpha2":0.1,"beta2":0.},
        "constructor_kwargs":{"initial_values":[0.1,0.2],
            "parameter_events":[{"time":1.,"parameters":{"alpha1":0.4}},
                                {"time":2.,"parameters":{"alpha1":0.}}]}
    });
    let result = KernelBinding::new()
        .predict_model("lotka_volterra", payload)
        .unwrap()
        .result
        .unwrap();
    let at_one = 1. / (1. + 9. * (-0.2_f64).exp());
    let at_two = 1. / (1. + 9. * (-0.6_f64).exp());
    assert!((result["rows"][1][0].as_f64().unwrap() - at_one).abs() < 1e-8);
    assert!((result["rows"][2][0].as_f64().unwrap() - at_two).abs() < 1e-8);
    assert!((result["rows"][3][0].as_f64().unwrap() - at_two).abs() < 1e-8);
    assert!((result["rates"][1][0].as_f64().unwrap() - 0.4 * at_one * (1. - at_one)).abs() < 1e-8);
    assert_eq!(result["rates"][2][0], 0.);
    let forecast = KernelBinding::new()
        .predict_model(
            "lotka_volterra",
            json!({
                "inputs":{"time":[4.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((forecast["rows"][0][0].as_f64().unwrap() - at_two).abs() < 1e-8);
}

#[test]
fn fixed_events_before_origin_apply_and_invalid_events_are_rejected() {
    let mut payload = json!({
        "inputs":{"time":[1.,2.]},
        "parameters":{"alpha1":0.2,"beta1":0.,"alpha2":0.1,"beta2":0.},
        "constructor_kwargs":{"initial_time":1.,"initial_values":[0.1,0.2],
            "parameter_events":[{"time":0.,"parameters":{"alpha1":0.4}}]}
    });
    let result = KernelBinding::new()
        .predict_model("lotka_volterra", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert!(
        (result["rows"][1][0].as_f64().unwrap() - 1. / (1. + 9. * (-0.4_f64).exp())).abs() < 1e-8
    );
    for invalid in [
        json!([{"time":-1.,"parameters":{"alpha1":0.4}}]),
        json!([{"time":1.,"parameters":{}}]),
        json!([{"time":1.,"parameters":{"unknown":0.4}}]),
        json!([{"time":1.,"parameters":{"alpha1":-0.4}}]),
        json!([{"time":1.,"parameters":{"alpha1":0.4}},{"time":1.,"parameters":{"alpha1":0.2}}]),
    ] {
        payload["constructor_kwargs"]["parameter_events"] = invalid;
        assert!(
            KernelBinding::new()
                .predict_model("lotka_volterra", payload.clone())
                .is_err()
        );
    }
}

#[test]
fn ecosystem_events_compose_with_controls_and_preserve_lock_in_projection() {
    let binding = KernelBinding::new();
    let result = binding
        .predict_model(
            "complementary_goods",
            json!({
                "inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},
                "parameters":{"k1":0.2,"k2":0.1,"c1":0.,"c2":0.,
                    "beta_k1_price":0.1,"beta_k2_price":0.,"beta_c1_price":0.,"beta_c2_price":0.},
                "constructor_kwargs":{"initial_values":[0.1,0.2],"covariates":["price"],
                    "parameter_events":[{"time":1.,"parameters":{"k1":0.4,"beta_k1_price":-0.1}}]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert!(
        (result["rows"][2][0].as_f64().unwrap() - 1. / (1. + 9. * (-0.5_f64).exp())).abs() < 1e-8
    );
    let projected = binding.predict_model("lock_in",json!({
        "inputs":{"time":[0.,1.,2.,3.]},
        "parameters":{"alpha1":0.,"alpha2":0.,"beta1":0.,"beta2":0.,"gamma1":0.,"gamma2":0.,"m":100.},
        "constructor_kwargs":{"initial_values":[95.,0.],
            "parameter_events":[{"time":1.,"parameters":{"m":90.}},
                {"time":2.,"parameters":{"m":80.}},{"time":3.,"parameters":{"m":100.}}]}
    })).unwrap().result.unwrap();
    assert_eq!(
        projected["rows"],
        json!([[95., 0.], [90., 0.], [80., 0.], [95., 0.]])
    );
    assert_eq!(
        projected["rates"],
        json!([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    );
}

#[test]
fn ecosystem_events_match_independent_references_and_conditional_fit() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/ecosystem_events_scipy.json")).unwrap();
    let binding = KernelBinding::new();
    for model in [
        "lotka_volterra",
        "complementary_goods",
        "lock_in",
        "multi_product",
    ] {
        let case = &fixture["cases"][model];
        let result = binding
            .predict_model(
                model,
                json!({
                    "inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
                    "parameters":case["parameters"],"constructor_kwargs":case["constructor_kwargs"]
                }),
            )
            .unwrap()
            .result
            .unwrap();
        for quantity in ["rows", "rates"] {
            for (actual, expected) in result[quantity]
                .as_array()
                .unwrap()
                .iter()
                .zip(case[quantity].as_array().unwrap())
            {
                for (actual, expected) in actual
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected.as_array().unwrap())
                {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "{model} {quantity}"
                    );
                }
            }
        }
        let forecast = binding
            .predict_model(
                model,
                json!({
                    "inputs":{"time":fixture["forecast_time"]},"state":result["state"]
                }),
            )
            .unwrap()
            .result
            .unwrap();
        for (actual, expected) in forecast["rows"][0]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["forecast_rows"][0].as_array().unwrap())
        {
            assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
        }
    }
    for model in ["lotka_volterra", "lock_in"] {
        let case = &fixture["cases"][model];
        let mut start = case["parameters"].clone();
        start["alpha1"] = json!(0.29);
        let fit = binding.fit_model(model,json!({
            "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":case["rows"]},
            "constructor_kwargs":case["constructor_kwargs"],
            "fit_options":{"initial_parameters":start,"max_iterations":200}
        })).unwrap().result.unwrap();
        assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
        assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
        assert_eq!(
            fit["diagnostics"]["parameter_events"]["parameters_estimated"],
            false
        );
        if model == "lock_in" {
            assert_eq!(
                fit["diagnostics"]["structural_identifiability"]["status"],
                "requires_event_specific_analysis"
            );
        }
    }
}

#[test]
fn weighted_competition_fit_excludes_all_outputs_on_a_zero_weight_row() {
    let f: Value = serde_json::from_str(include_str!("fixtures/lotka_forcing_scipy.json")).unwrap();
    let mut observed = f["rows"].as_array().unwrap().clone();
    observed[5] = json!([1e100, 1e100]);
    let mut weights = vec![2.; observed.len()];
    weights[5] = 0.;
    let fit = KernelBinding::new()
        .fit_model(
            "lotka_volterra",
            json!({
                "inputs":{"time":f["time"],"observed":observed,"covariates":f["covariates"]},
                "constructor_kwargs":{"covariates":["price"],"initial_values":f["initial"]},
                "fit_options":{"weights":weights}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(
        fit["diagnostics"]["weighting"]["positive_observations"],
        observed.len() - 1
    );
    assert!(
        fit["diagnostics"]["weighting"]["weighted_rss_normalized_by_max"]
            .as_f64()
            .unwrap()
            < 1e-10
    );
}

#[test]
fn automatic_starts_converge_on_unforced_multi_product_and_lock_in() {
    let multi: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/competitive_scipy.json")).unwrap();
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lock_in_scipy.json")).unwrap();
    for (model, reference, options) in [
        (
            "multi_product",
            &multi["cases"]["multi_product"],
            json!({"n_products":2}),
        ),
        ("lock_in", &lock, json!({"initial_values":lock["initial"]})),
    ] {
        let fit = KernelBinding::new()
            .fit_model(
                model,
                json!({
                    "inputs":{"time":reference["time"],"observed":reference["rows"]},
                    "constructor_kwargs":options,"fit_options":{"max_iterations":200}
                }),
            )
            .unwrap()
            .result
            .unwrap();
        assert_eq!(
            fit["diagnostics"]["optimizer"]["converged"], true,
            "{model}"
        );
        assert!(
            fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10,
            "{model}"
        );
    }
}

#[test]
fn forced_automatic_starts_converge_and_recover_held_out_trajectories() {
    let multi: Value =
        serde_json::from_str(include_str!("fixtures/multi_forcing_scipy.json")).unwrap();
    let lock: Value =
        serde_json::from_str(include_str!("fixtures/lock_forcing_scipy.json")).unwrap();
    let events: Value =
        serde_json::from_str(include_str!("fixtures/ecosystem_events_scipy.json")).unwrap();
    let kernel = KernelBinding::new();
    for (model, fixture, reference, options) in [
        (
            "multi_product",
            &multi,
            &multi["cases"]["corrected"],
            json!({"n_products":2,"covariates":["price"]}),
        ),
        (
            "lock_in",
            &lock,
            &lock,
            json!({"initial_values":lock["initial"],"covariates":["price"]}),
        ),
        (
            "multi_product",
            &events,
            &events["cases"]["multi_product"],
            events["cases"]["multi_product"]["constructor_kwargs"].clone(),
        ),
        (
            "lock_in",
            &events,
            &events["cases"]["lock_in"],
            events["cases"]["lock_in"]["constructor_kwargs"].clone(),
        ),
    ] {
        let fit = kernel.fit_model(model, json!({
            "inputs":{"time":fixture["time"],"observed":reference["rows"],"covariates":fixture["covariates"]},
            "constructor_kwargs":options,"fit_options":{"max_iterations":200}
        })).unwrap().result.unwrap();
        assert_eq!(
            fit["diagnostics"]["optimizer"]["converged"], true,
            "{model}"
        );
        let forecast = kernel
            .predict_model(
                model,
                json!({
                    "inputs":{"time":fixture["forecast_time"]},"state":fit["state"]
                }),
            )
            .unwrap()
            .result
            .unwrap();
        let rows = forecast["rows"].as_array().unwrap();
        let expected = reference["forecast_rows"].as_array().unwrap();
        assert_eq!(rows.len(), expected.len());
        for (row, expected) in rows.iter().zip(expected) {
            let row = row.as_array().unwrap();
            let expected = expected.as_array().unwrap();
            assert_eq!(row.len(), expected.len());
            for (actual, expected) in row.iter().zip(expected) {
                assert!(
                    (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-3,
                    "{model}"
                );
            }
        }
    }
}

#[test]
fn ecosystem_models_profile_parameter_event_times() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/ecosystem_events_scipy.json")).unwrap();
    for model in [
        "lotka_volterra",
        "complementary_goods",
        "lock_in",
        "multi_product",
    ] {
        let case = &fixture["cases"][model];
        let mut options = case["constructor_kwargs"].clone();
        let target = options["parameter_events"][0]["time"].as_f64().unwrap();
        options["event_candidates"] = json!([target - 0.125, target, target + 0.125]);
        options["event_index"] = json!(0);
        let result = KernelBinding::new().fit_model(model, json!({
            "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":case["rows"]},
            "constructor_kwargs":options,
            "fit_options":{"initial_parameters":case["parameters"],"max_iterations":200}
        })).unwrap().result.unwrap();
        assert_eq!(
            result["state"]["constructor_kwargs"]["parameter_events"],
            case["constructor_kwargs"]["parameter_events"],
            "{model}"
        );
        assert_eq!(result["event_profile"]["event_index"], 0);
        assert!(
            result["event_profile"]["candidates"][1]["objective"]
                .as_f64()
                .unwrap()
                < 1e-8
        );
    }
}

#[test]
fn lotka_volterra_profiles_event_time_and_override() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/ecosystem_events_scipy.json")).unwrap();
    let case = &fixture["cases"]["lotka_volterra"];
    let mut options = case["constructor_kwargs"].clone();
    options["event_candidates"] = json!([2., 2.125]);
    options["event_index"] = json!(0);
    options["event_override_parameter"] = json!("alpha1");
    options["event_override_candidates"] = json!([0.35, 0.4, 0.45]);
    let result = KernelBinding::new().fit_model("lotka_volterra", json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":case["rows"]},
        "constructor_kwargs":options,
        "fit_options":{"initial_parameters":case["parameters"],"max_iterations":200}
    })).unwrap().result.unwrap();
    assert_eq!(
        result["state"]["constructor_kwargs"]["parameter_events"],
        case["constructor_kwargs"]["parameter_events"]
    );
    let profile = &result["event_profile"];
    assert_eq!(profile["selected_index"], 4);
    assert_eq!(profile["selected"], profile["candidates"][4]);
    assert_eq!(profile["override_parameter"], "alpha1");
    assert_eq!(profile["candidates"].as_array().unwrap().len(), 6);
    assert!(profile["candidates"][4]["objective"].as_f64().unwrap() < 1e-8);
}

#[test]
fn historical_competition_equilibrium_summaries_match_closed_forms() {
    let run = |model, payload| {
        let request = json!({"schema_version":"2.0","operation":"summarize_competition_equilibrium","model_key":model,"payload":payload});
        serde_json::from_str::<Value>(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let shares = run("multi_product", json!({"market_potentials":[1e308,1e308]}));
    assert_eq!(
        shares["result"]["shares"],
        json!({"product_0":0.5,"product_1":0.5})
    );
    let coexist = run(
        "lotka_volterra",
        json!({"carrying_capacity_1":120,
        "carrying_capacity_2":90,"competition_coeff_12":0.5,"competition_coeff_21":0.25}),
    );
    assert_eq!(coexist["result"]["N1_eq"], 75. / 0.875);
    assert_eq!(coexist["result"]["N2_eq"], 60. / 0.875);
    let boundary = run("lotka_volterra", json!({}));
    assert_eq!(boundary["result"]["N1_eq"], 1000.);
    assert_eq!(boundary["result"]["boundary_fallback"], true);
    assert_eq!(
        run("multi_product", json!({"market_potentials":[0]}))["error"]["code"],
        "invalid_request"
    );
}

#[test]
fn historical_cross_elasticity_matches_independent_exponential_path() {
    let mut request = json!({"schema_version":"2.0","operation":"competition_cross_elasticity","payload":{
        "time":[0,1,2],"p":[0.1,0.2],"q":[[0,0.4],[0,0]],"market_potentials":[50,100]}});
    let run = |request: &Value| {
        serde_json::from_str::<Value>(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let output = &response["result"]["elasticity"];
    for (index, time) in [0_f64, 1., 2.].into_iter().enumerate() {
        let expected = 0.4 * 100. * (1. - (-0.2 * time).exp()) / (50. + 1e-10);
        assert!((output[0][index].as_f64().unwrap() - expected).abs() < 1e-8);
        assert_eq!(output[1][index], 0.);
    }
    request["payload"]["q"] = json!([[0, 0.4]]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn historical_interaction_matrix_predicts_levels_and_instantaneous_rates() {
    let request = json!({"schema_version":"2.0","operation":"predict_competition_interaction_matrix",
        "payload":{"time":[3,4,5],"p":[0.1,0.2],"q":[[0,-0.05],[0,0]],"market_potentials":[50,100]}});
    let response: Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert_eq!(result["axis_order"], json!(["time", "product"]));
    for (index, elapsed) in [0_f64, 1., 2.].into_iter().enumerate() {
        let survival = (-0.2 * elapsed).exp();
        let force = 0.1 - 0.05 * (1. - survival);
        let remaining = 50. * (-0.05 * elapsed - 0.25 * (1. - survival)).exp();
        for (product, level, rate) in [
            (0, 50. - remaining, force * remaining),
            (1, 100. * (1. - survival), 20. * survival),
        ] {
            assert!((result["rows"][index][product].as_f64().unwrap() - level).abs() < 1e-6);
            assert!((result["rates"][index][product].as_f64().unwrap() - rate).abs() < 1e-6);
        }
    }
}

#[test]
fn historical_interaction_matrix_scores_pooled_observations() {
    let mut request = json!({"schema_version":"2.0","operation":"predict_competition_interaction_matrix",
        "payload":{"time":[0,1],"p":[0,0],"q":[[0,0],[0,0]],"market_potentials":[50,100],
        "observed":[[1,2],[3,4]]}});
    let run = |request: &Value| {
        serde_json::from_str::<Value>(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    assert_eq!(run(&request)["result"]["r_squared"], -5.);
    request["payload"]["observed"] = json!([[0.1, 0.1], [0.1, 0.1]]);
    assert_eq!(run(&request)["result"]["r_squared"], 0.);
    request["payload"]["observed"] = json!([[1, 2]]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn historical_interaction_matrix_fit_recovers_independent_trajectory() {
    let time: Vec<_> = (0..24).map(|i| i as f64 * 0.5).collect();
    let observed: Vec<_> = time
        .iter()
        .map(|t| {
            [
                50. * (1. - (-0.1 * t).exp()),
                100. * (1. - (-0.2 * t).exp()),
            ]
        })
        .collect();
    let request = json!({"schema_version":"2.0","operation":"fit_competition_interaction_matrix",
        "payload":{"time":time,"p":[0.11,0.19],"q":[[0.001,-0.001],[0.001,0.001]],
        "market_potentials":[49,101],"observed":observed}});
    let response: Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert!(
        result["predictions"]["r_squared"]
            .as_f64()
            .unwrap_or_default()
            > 0.999999,
        "{response}"
    );
    assert_eq!(result["identifiability"]["rank"], 8);
}
