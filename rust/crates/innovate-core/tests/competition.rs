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
