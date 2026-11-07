use innovate_core::{KernelBinding, json};
#[test]
fn complementary_equations_match_independent_scipy_and_fit() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/complementary_scipy.json")).unwrap();
    let a=KernelBinding::new().predict_model("complementary_goods",json!({"inputs":{"time":f["time"]},"parameters":f["parameters"],"constructor_kwargs":{"initial_values":f["initial"]}})).unwrap().result.unwrap();
    for (a, b) in a["rows"]
        .as_array()
        .unwrap()
        .iter()
        .zip(f["rows"].as_array().unwrap())
    {
        for j in 0..2 {
            assert!((a[j].as_f64().unwrap() - b[j].as_f64().unwrap()).abs() < 1e-7);
        }
    }
    let fit=KernelBinding::new().fit_model("complementary_goods",json!({"inputs":{"time":f["time"],"observed":f["rows"]},"fit_options":{"initial_parameters":{"k1":0.28,"k2":0.38,"c1":0.015,"c2":0.025}}})).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(fit["diagnostics"]["identifiability"]["rank"], 4);
    for (name, v) in f["parameters"].as_object().unwrap() {
        assert!((fit["parameters"][name].as_f64().unwrap() - v.as_f64().unwrap()).abs() < 1e-4);
    }
}
#[test]
fn decoupled_complementary_goods_match_logistic_and_allow_levels_above_one() {
    let a=KernelBinding::new().predict_model("complementary_goods",json!({"inputs":{"time":[0.,3.]},"parameters":{"k1":0.3,"k2":0.4,"c1":0.,"c2":0.},"constructor_kwargs":{"initial_values":[0.05,0.02]}})).unwrap().result.unwrap();
    assert!((a["rows"][1][0].as_f64().unwrap() - 1. / (1. + 19. * (-0.9_f64).exp())).abs() < 1e-8);
    let a=KernelBinding::new().predict_model("complementary_goods",json!({"inputs":{"time":[0.]},"parameters":{"k1":0.3,"k2":0.4,"c1":0.02,"c2":0.03},"constructor_kwargs":{"initial_values":[1.1,1.2]}})).unwrap().result.unwrap();
    assert_eq!(a["rows"][0][0], 1.1);
}

#[test]
fn complementary_controls_preserve_signed_effects_and_saved_history() {
    let payload = json!({
        "inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},
        "parameters":{"k1":0.2,"k2":0.1,"c1":0.,"c2":0.,
            "beta_k1_price":0.1,"beta_k2_price":-0.02,"beta_c1_price":0.,"beta_c2_price":0.},
        "constructor_kwargs":{"initial_values":[0.1,0.2],"covariates":["price"]}
    });
    let result = KernelBinding::new()
        .predict_model("complementary_goods", payload)
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
            "complementary_goods",
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
}

#[test]
fn complementary_forcing_fits_independent_reference_and_forecasts() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/complementary_forcing_scipy.json")).unwrap();
    let options = json!({"initial_values":fixture["initial"],"covariates":["price"]});
    let binding = KernelBinding::new();
    let result = binding
        .predict_model(
            "complementary_goods",
            json!({
                "inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
                "parameters":fixture["parameters"],"constructor_kwargs":options
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
    start["k1"] = json!(0.29);
    start["beta_c2_price"] = json!(-0.007);
    for fit_options in [json!({"initial_parameters":start,"max_iterations":150}), json!({"max_iterations":150})] {
        let fit = binding.fit_model("complementary_goods",json!({
            "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
            "constructor_kwargs":options,"fit_options":fit_options
        })).unwrap().result.unwrap();
        assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
        assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10);
        let forecast = binding
            .predict_model(
                "complementary_goods",
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
    }
}
