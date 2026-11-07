use innovate_core::{KernelBinding, KernelRequest, json};
use serde_json::Value;

fn assert_close(got: &Value, expected: &Value, tolerance: f64) {
    let a = got.as_array().unwrap();
    let b = expected.as_array().unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        if x.is_array() {
            assert_close(x, y, tolerance);
        } else {
            assert!(
                (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < tolerance,
                "{x} != {y}"
            );
        }
    }
}

#[test]
fn legacy_three_generation_ode_matches_independent_scipy_oracle() {
    let f: Value = serde_json::from_str(include_str!("fixtures/norton_legacy_scipy.json")).unwrap();
    let payload = json!({"inputs":{"time":f["time"]},"parameters":f["parameters"],"constructor_kwargs":f["constructor_kwargs"]});
    let response = KernelBinding::new()
        .predict_model("norton_bass", payload)
        .unwrap();
    let result = response.result.unwrap();
    assert_close(&result["rows"], &f["rows"], 1e-5);
    assert_close(&result["rates"], &f["rates"], 1e-5);
    assert_eq!(result["metadata"]["variant"], "legacy_cannibalization");
}

#[test]
fn successive_generation_closed_form_launches_and_conservation() {
    let params = json!({"p1":0.03,"q1":0.2,"m1":1000.,"p2":0.02,"q2":0.3,"m2":1500.,"p3":0.04,"q3":0.25,"m3":600.});
    let time = [0., 1., 4., 8., 12., 30., 1000.];
    let payload = json!({"inputs":{"time":time},"parameters":params,"constructor_kwargs":{"n_generations":3,"variant":"successive_generations","launch_times":[0.,4.,8.]}});
    let result = KernelBinding::new()
        .predict_model("norton_bass", payload)
        .unwrap()
        .result
        .unwrap();
    let fraction = |t: f64, p: f64, q: f64| {
        if t <= 0. {
            0.
        } else {
            (1. - (-(p + q) * t).exp()) / (1. + q / p * (-(p + q) * t).exp())
        }
    };
    for (i, t) in time.into_iter().enumerate() {
        let f1 = fraction(t, 0.03, 0.2);
        let f2 = fraction(t - 4., 0.02, 0.3);
        let f3 = fraction(t - 8., 0.04, 0.25);
        let expected = json!([
            1000. * f1 * (1. - f2),
            (1500. + 1000. * f1) * f2 * (1. - f3),
            (600. + (1500. + 1000. * f1) * f2) * f3
        ]);
        assert_close(&result["rows"][i], &expected, 1e-9);
        let total: f64 = result["rows"][i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert!((total - (1000. * f1 + 1500. * f2 + 600. * f3)).abs() < 1e-9);
    }
}

#[test]
fn multi_output_fit_reports_convergence_and_predicts_held_out_times() {
    let time: Vec<f64> = (0..41).map(|i| f64::from(i) * 0.5).collect();
    let fraction = |t: f64, p: f64, q: f64| {
        if t <= 0. {
            0.
        } else {
            (1. - (-(p + q) * t).exp()) / (1. + q / p * (-(p + q) * t).exp())
        }
    };
    let observed: Vec<Vec<f64>> = time
        .iter()
        .map(|&t| {
            let f1 = fraction(t, 0.03, 0.2);
            let f2 = fraction(t - 4., 0.02, 0.3);
            vec![1000. * f1 * (1. - f2), (1500. + 1000. * f1) * f2]
        })
        .collect();
    let options =
        json!({"n_generations":2,"variant":"successive_generations","launch_times":[0.,4.]});
    let fit = KernelBinding::new()
        .fit_model(
            "norton_bass",
            json!({"inputs":{"time":time,"observed":observed},"constructor_kwargs":options}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(fit["diagnostics"]["identifiability"]["rank"], 6);
    for (key, value) in [
        ("p1", 0.03),
        ("q1", 0.2),
        ("m1", 1000.),
        ("p2", 0.02),
        ("q2", 0.3),
        ("m2", 1500.),
    ] {
        let got = fit["parameters"][key].as_f64().unwrap();
        assert!((got / value - 1.).abs() < 0.001, "{key}: {got}");
    }
    let future = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({"inputs":{"time":[21.,25.,30.]},"state":fit["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let expected: Vec<Vec<f64>> = [21., 25., 30.]
        .into_iter()
        .map(|t| {
            let f1 = fraction(t, 0.03, 0.2);
            let f2 = fraction(t - 4., 0.02, 0.3);
            vec![1000. * f1 * (1. - f2), (1500. + 1000. * f1) * f2]
        })
        .collect();
    assert_close(&future["rows"], &json!(expected), 0.01);
}

#[test]
fn invalid_shapes_parameters_and_unknown_options_are_rejected() {
    let binding = KernelBinding::new();
    for kwargs in [
        json!({"n_generations":0}),
        json!({"n_generations":2,"variant":"typo"}),
        json!({"n_generations":2,"launch_times":[0.]}),
        json!({"n_generations":2,"unrecognised":true}),
    ] {
        let r = KernelRequest::predict_model(
            "norton_bass",
            json!({"inputs":{"time":[0.,1.]},"parameters":{"p1":0.03,"q1":0.2,"m1":100.,"p2":0.02,"q2":0.3,"m2":80.},"constructor_kwargs":kwargs}),
        );
        assert!(binding.invoke(&r).is_err());
    }
    assert!(binding.fit_model("norton_bass",json!({"inputs":{"time":[0.,1.,2.,3.],"observed":[[1.,2.],[3.],[4.,5.],[6.,7.]]},"constructor_kwargs":{"n_generations":2}})).is_err());
}

#[test]
fn legacy_fit_recovers_independent_ode_fixture_and_rates_keep_their_sign() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/norton_legacy_scipy.json")).unwrap();
    let fit = KernelBinding::new()
        .fit_model(
            "norton_bass",
            json!({
                "inputs":{"time":fixture["time"],"observed":fixture["rows"]},
                "constructor_kwargs":fixture["constructor_kwargs"],
                "fit_options":{"initial_parameters":fixture["parameters"]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_close(&fit["predictions"]["rows"], &fixture["rows"], 1e-4);
    assert!(fit["predictions"]["rates"][8][0].as_f64().unwrap() < 0.0);
}

#[test]
fn convergence_limits_and_unobserved_generations_are_reported_without_overclaiming() {
    let time: Vec<f64> = (0..21).map(f64::from).collect();
    let observed: Vec<Vec<f64>> = time
        .iter()
        .map(|t| {
            let d = (-0.23 * t).exp();
            vec![1000. * (1. - d) / (1. + 0.2 / 0.03 * d), 0.]
        })
        .collect();
    let payload = json!({"inputs":{"time":time,"observed":observed},"constructor_kwargs":{"n_generations":2,"variant":"successive_generations","launch_times":[0.,100.]}});
    let fit = KernelBinding::new()
        .fit_model("norton_bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        fit["diagnostics"]["identifiability"]["status"],
        "rank_deficient"
    );
    assert_eq!(fit["diagnostics"]["identifiability"]["rank"], 3);
    assert!(
        !fit["diagnostics"]["warnings"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let mut limited = payload;
    limited["fit_options"] = json!({"max_iterations":1});
    let fit = KernelBinding::new()
        .fit_model("norton_bass", limited)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], false);
    assert_eq!(
        fit["diagnostics"]["optimizer"]["termination"],
        "iteration_limit"
    );
}

#[test]
fn unknown_generation_parameters_and_nonfinite_domains_do_not_silently_disappear() {
    for parameters in [
        json!({"p1":0.03,"q1":0.2,"m1":100.,"p2":-0.1,"q2":0.2,"m2":80.}),
        json!({"p1":0.03,"q1":0.2,"m1":100.,"p2":0.1,"q2":-0.2,"m2":80.}),
        json!({"p1":0.03,"q1":0.2,"m1":100.,"p2":0.1,"q2":0.2,"m2":0.}),
        json!({"p1":0.03,"q1":0.2,"m1":100.,"p3":0.1,"q3":0.2,"m3":80.}),
    ] {
        assert!(
            KernelBinding::new()
                .predict_model(
                    "norton_bass",
                    json!({"inputs":{"time":[0.,1.]},"parameters":parameters})
                )
                .is_err()
        );
    }
}

#[test]
fn legacy_origin_is_explicit_and_forecast_batching_does_not_reset_state() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/norton_legacy_scipy.json")).unwrap();
    let result=KernelBinding::new().predict_model("norton_bass",json!({"inputs":{"time":[9.,16.,25.]},"parameters":fixture["parameters"],"constructor_kwargs":fixture["constructor_kwargs"]})).unwrap().result.unwrap();
    for (i, j) in [6, 7, 8].into_iter().enumerate() {
        assert_close(&result["rows"][i], &fixture["rows"][j], 1e-5);
    }
    assert!(KernelBinding::new().predict_model("norton_bass",json!({"inputs":{"time":[0.,1.]},"parameters":{"p1":0.03,"q1":0.2,"m1":100.},"constructor_kwargs":{"initial_time":2.}})).is_err());
}

#[test]
fn covariate_history_drives_both_historical_generations_and_replays() {
    let binding = KernelBinding::new();
    let parameters = json!({"p1":0.1,"q1":0.,"m1":100.,"p2":0.1,"q2":0.,"m2":100.,
        "beta_p1_price":0.05,"beta_q1_price":0.,"beta_m1_price":0.,
        "beta_p2_price":0.05,"beta_q2_price":0.,"beta_m2_price":0.});
    let options = json!({"n_generations":2,"covariates":["price"]});
    let result = binding
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},
                "constructor_kwargs":options,"parameters":parameters
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let replay = binding
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[3.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let check = |row: &serde_json::Value, hazard: f64| {
        let decay = (-hazard).exp();
        assert!((row[0].as_f64().unwrap() - 100. * hazard * decay).abs() < 1e-5);
        assert!((row[1].as_f64().unwrap() - 100. * (1. - decay)).abs() < 1e-5);
    };
    check(&result["rows"][2], 0.3);
    check(&replay["rows"][0], 0.5);
    assert!((result["rates"][2][1].as_f64().unwrap() - 20. * (-0.3_f64).exp()).abs() < 1e-5);
    assert!(
        binding
            .predict_model(
                "norton_bass",
                json!({
                    "inputs":{"time":[1.],"covariates":{"price":[0.]}},
                    "state":result["state"]
                })
            )
            .is_err()
    );
}

#[test]
fn conditional_covariate_fit_retains_all_named_coefficients() {
    let binding = KernelBinding::new();
    let time: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let values: Vec<f64> = time.iter().map(|t| (t / 3.).sin()).collect();
    let parameters = json!({"p1":0.03,"q1":0.3,"m1":100.,
        "beta_p1_price":0.005,"beta_q1_price":0.02,"beta_m1_price":2.});
    let options = json!({"n_generations":1,"covariates":["price"]});
    let inputs = json!({"time":time,"covariates":{"price":values}});
    let generated = binding
        .predict_model(
            "norton_bass",
            json!({
                "inputs":inputs,"constructor_kwargs":options,"parameters":parameters
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let mut training = inputs;
    training["observed"] = generated["rows"].clone();
    let mut start = parameters.clone();
    start["p1"] = json!(0.032);
    start["beta_p1_price"] = json!(0.0045);
    let fit = binding
        .fit_model(
            "norton_bass",
            json!({
                "inputs":training,"constructor_kwargs":options,
                "fit_options":{"initial_parameters":start,"max_iterations":150}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(fit["parameters"].as_object().unwrap().len(), 6);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-10);
    let automatic = binding
        .fit_model(
            "norton_bass",
            json!({"inputs":training,"constructor_kwargs":options}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(automatic["diagnostics"]["optimizer"]["converged"], true);
    assert!(automatic["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
}

#[test]
fn historical_norton_events_preserve_continuous_generations_and_controls() {
    let result = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.,"p2":0.1,"q2":0.,"m2":100.,
                    "beta_p1_price":0.05,"beta_q1_price":0.,"beta_m1_price":0.,
                    "beta_p2_price":0.05,"beta_q2_price":0.,"beta_m2_price":0.},
                "constructor_kwargs":{"n_generations":2,"covariates":["price"],
                    "parameter_events":[{"time":1.,"parameters":{"p1":0.2,"p2":0.2}},
                        {"time":2.,"parameters":{"beta_p1_price":0.,"beta_p2_price":0.}}]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((result["rows"][2][0].as_f64().unwrap() - 40. * (-0.4_f64).exp()).abs() < 1e-5);
    assert!((result["rates"][1][1].as_f64().unwrap() - 25. * (-0.125_f64).exp()).abs() < 1e-5);
    let forecast = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[3.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((forecast["rows"][0][0].as_f64().unwrap() - 60. * (-0.6_f64).exp()).abs() < 1e-5);
}

#[test]
fn scalar_norton_events_extend_to_both_variants() {
    let mut payload = json!({
        "inputs":{"time":[0.,1.,2.]},
        "parameters":{"p1":0.1,"q1":0.,"m1":100.},
        "constructor_kwargs":{"n_generations":1,
            "parameter_events":[{"time":0.,"parameters":{"p1":0.2}},
                {"time":1.,"parameters":{"p1":0.}}]}
    });
    let result = KernelBinding::new()
        .predict_model("norton_bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert!((result["rows"][2][0].as_f64().unwrap() - 100. * (1. - (-0.2_f64).exp())).abs() < 1e-6);
    assert_eq!(result["rates"][1][0], 0.);
    payload["constructor_kwargs"]["variant"] = json!("successive_generations");
    let successive = KernelBinding::new()
        .predict_model("norton_bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert!(
        (successive["rows"][2][0].as_f64().unwrap() - 100. * (1. - (-0.2_f64).exp())).abs() < 1e-6
    );
}

#[test]
fn historical_norton_events_match_independent_reference_and_fit() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/norton_events_scipy.json")).unwrap();
    let binding = KernelBinding::new();
    let result = binding.predict_model("norton_bass",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
        "parameters":fixture["parameters"],"constructor_kwargs":fixture["constructor_kwargs"]
    })).unwrap().result.unwrap();
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
                assert!(
                    (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                    "{quantity}"
                );
            }
        }
    }
    let mut start = fixture["parameters"].clone();
    start["p1"] = json!(0.0305);
    let fit = binding.fit_model("norton_bass",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
        "constructor_kwargs":fixture["constructor_kwargs"],
        "fit_options":{"initial_parameters":start,"max_iterations":200}
    })).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
    assert_eq!(
        fit["diagnostics"]["parameter_events"]["parameters_estimated"],
        false
    );
    let forecast = binding
        .predict_model(
            "norton_bass",
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
        assert!(
            (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-3,
            "actual={actual} expected={expected} fit={fit}"
        );
    }
}

#[test]
fn successive_controls_include_capacity_slopes_and_launch_rates() {
    let result = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[0.,1.,2.,3.],"covariates":{"price":[0.,1.,2.,3.]}},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.,"p2":0.1,"q2":0.,"m2":200.,
                    "beta_p1_price":0.05,"beta_q1_price":0.,"beta_m1_price":10.,
                    "beta_p2_price":0.05,"beta_q2_price":0.,"beta_m2_price":-5.},
                "constructor_kwargs":{"n_generations":2,"variant":"successive_generations",
                    "launch_times":[0.,1.],"covariates":["price"]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let first = 1. - (-0.3_f64).exp();
    let second = 1. - (-0.175_f64).exp();
    let first_rate = 0.2 * (1. - first);
    let second_rate = 0.2 * (1. - second);
    let inherited = 120. * first;
    let inherited_rate = 120. * first_rate + 10. * first;
    let expected = [inherited * (1. - second), second * (190. + inherited)];
    let rates = [
        inherited_rate * (1. - second) - inherited * second_rate,
        second_rate * (190. + inherited) + second * (-5. + inherited_rate),
    ];
    for index in 0..2 {
        assert!((result["rows"][2][index].as_f64().unwrap() - expected[index]).abs() < 1e-5);
        assert!((result["rates"][2][index].as_f64().unwrap() - rates[index]).abs() < 1e-5);
    }
    assert_eq!(result["rows"][1][1], 0.);
    let forecast = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[4.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let first = 1. - (-0.775_f64).exp();
    let second = 1. - (-0.65_f64).exp();
    assert!((forecast["rows"][0][0].as_f64().unwrap() - 130. * first * (1. - second)).abs() < 1e-5);
}

#[test]
fn successive_forcing_matches_independent_fraction_reference_and_fit() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/successive_forcing_scipy.json")).unwrap();
    let binding = KernelBinding::new();
    let result = binding.predict_model("norton_bass",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"]},
        "parameters":fixture["parameters"],"constructor_kwargs":fixture["constructor_kwargs"]
    })).unwrap().result.unwrap();
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
                assert!(
                    (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                    "{quantity} actual={actual} expected={expected}"
                );
            }
        }
    }
    let fit = binding.fit_model("norton_bass",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
        "constructor_kwargs":fixture["constructor_kwargs"],
        "fit_options":{"max_iterations":200}
    })).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
    let forecast = binding
        .predict_model(
            "norton_bass",
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
        assert!(
            (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-3,
            "actual={actual} expected={expected} fit={fit}"
        );
    }
}

#[test]
fn successive_capacity_events_jump_levels_and_invalid_hazards_are_rejected() {
    let mut payload = json!({
        "inputs":{"time":[0.,1.,2.]},
        "parameters":{"p1":0.1,"q1":0.,"m1":100.},
        "constructor_kwargs":{"variant":"successive_generations","n_generations":1,
            "parameter_events":[{"time":1.,"parameters":{"m1":200.}}]}
    });
    let result = KernelBinding::new()
        .predict_model("norton_bass", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert!((result["rows"][1][0].as_f64().unwrap() - 200. * (1. - (-0.1_f64).exp())).abs() < 1e-6);
    assert!((result["rates"][1][0].as_f64().unwrap() - 20. * (-0.1_f64).exp()).abs() < 1e-6);
    payload["constructor_kwargs"]["covariates"] = json!(["price"]);
    payload["inputs"]["covariates"] = json!({"price":[0.,1.,2.]});
    payload["parameters"]["beta_p1_price"] = json!(-0.2);
    payload["parameters"]["beta_q1_price"] = json!(0.);
    payload["parameters"]["beta_m1_price"] = json!(0.);
    assert!(
        KernelBinding::new()
            .predict_model("norton_bass", payload.clone())
            .is_err()
    );
    payload["constructor_kwargs"]["launch_times"] = json!([5.]);
    payload["inputs"] = json!({"time":[0.,1.],"covariates":{"price":[0.,1.]}});
    let unlaunched = KernelBinding::new()
        .predict_model("norton_bass", payload)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(unlaunched["rows"], json!([[0.], [0.]]));
    assert_eq!(unlaunched["rates"], json!([[0.], [0.]]));
}

#[test]
fn capacity_rate_scaling_avoids_unrepresentable_intermediate_control_slopes() {
    let binding = KernelBinding::new();
    for beta in [0., 1e-308] {
        let result = binding.predict_model("norton_bass",json!({
            "inputs":{"time":[0.,0.5,1.],"covariates":{"price":[-1e308,0.,1e308]}},
            "parameters":{"p1":0.1,"q1":0.,"m1":100.,
                "beta_p1_price":0.,"beta_q1_price":0.,"beta_m1_price":beta},
            "constructor_kwargs":{"n_generations":1,"variant":"successive_generations","covariates":["price"]}
        })).unwrap().result.unwrap();
        let fraction = 1. - (-0.05_f64).exp();
        let capacity_rate = if beta == 0. { 0. } else { 2. };
        assert!((result["rows"][1][0].as_f64().unwrap() - 100. * fraction).abs() < 1e-6);
        assert!(
            (result["rates"][1][0].as_f64().unwrap()
                - (10. * (1. - fraction) + capacity_rate * fraction))
                .abs()
                < 1e-6
        );
    }
    let result = binding.predict_model("norton_bass",json!({
        "inputs":{"time":[0.,1.,2.]},
        "parameters":{"p1":0.1,"q1":0.,"m1":100.,
            "beta_p1_price":0.,"beta_q1_price":0.,"beta_m1_price":1e308},
        "constructor_kwargs":{"n_generations":1,"variant":"successive_generations","covariates":["price"],
            "covariate_history":{"time":[0.,1e308],"values":{"price":[0.,1e-308]}},
            "parameter_events":[{"time":1.,"parameters":{"p1":0.}}]}
    })).unwrap().result.unwrap();
    let rate = result["rates"][2][0].as_f64().unwrap();
    assert!(rate > 0.);
    assert!((rate / 1e-308 - (1. - (-0.1_f64).exp())).abs() < 1e-6);
}

#[test]
fn successive_control_history_can_start_at_the_first_launch() {
    let result = KernelBinding::new()
        .predict_model(
            "norton_bass",
            json!({
                "inputs":{"time":[2.,3.],"covariates":{"price":[0.,0.]}},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.,
                    "beta_p1_price":0.,"beta_q1_price":0.,"beta_m1_price":0.},
                "constructor_kwargs":{"n_generations":1,"variant":"successive_generations",
                    "launch_times":[2.],"covariates":["price"]}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["rows"][0][0], 0.);
    assert!((result["rows"][1][0].as_f64().unwrap() - 100. * (1. - (-0.1_f64).exp())).abs() < 1e-6);
}

#[test]
fn capacity_event_uses_each_intervals_active_control_derivative() {
    let result = KernelBinding::new().predict_model("norton_bass",json!({
        "inputs":{"time":[0.,1.,1.5],"covariates":{"price":[0.,1.,1e308]}},
        "parameters":{"p1":0.1,"q1":0.,"m1":100.,
            "beta_p1_price":0.,"beta_q1_price":0.,"beta_m1_price":1.},
        "constructor_kwargs":{"n_generations":1,"variant":"successive_generations","covariates":["price"],
            "parameter_events":[{"time":1.,"parameters":{"beta_m1_price":0.}}]}
    })).unwrap().result.unwrap();
    assert!((result["rows"][1][0].as_f64().unwrap() - 100. * (1. - (-0.1_f64).exp())).abs() < 1e-6);
    assert!((result["rates"][1][0].as_f64().unwrap() - 10. * (-0.1_f64).exp()).abs() < 1e-6);
}

#[test]
fn threshold_summary_uses_the_named_generation_parameters() {
    let response = KernelBinding::new()
        .summarize_model("norton_bass", json!({
            "inputs":{"time":[0.]},
            "parameters":{"p1":0.1,"q1":0.9,"m1":100.,"p2":0.2,"q2":0.,"m2":50.},
            "constructor_kwargs":{"n_generations":2,"variant":"successive_generations","launch_times":[0.,4.]}
        }))
        .unwrap().result.unwrap();
    let rows = &response["threshold_diagnostics"]["generations"];
    assert_eq!(rows[0]["generation"], 1);
    assert!((rows[0]["half_life"].as_f64().unwrap() - 9_f64.ln()).abs() < 1e-12);
    assert!(rows[0]["takeoff"].as_f64().unwrap().abs() < 1e-12);
    assert_eq!(rows[1]["reference_time"], 4.);
    assert_eq!(rows[1]["status"], "degenerate_rates");
    assert_eq!(rows[1]["saturation"], 0.);
}

#[test]
fn forced_threshold_summary_reports_sampled_net_rate_peaks() {
    let response = KernelBinding::new()
        .summarize_model(
            "norton_bass",
            json!({
                "inputs":{"time":[0.,1.,2.,3.]},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.,"p2":0.1,"q2":0.,"m2":100.},
                "constructor_kwargs":{
                    "n_generations":2,"variant":"successive_generations","launch_times":[0.,4.],
                    "parameter_events":[{"time":2.,"parameters":{"p1":0.5}}]
                }
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let trajectory = &response["threshold_diagnostics"]["trajectory"];
    assert_eq!(trajectory["window"], json!([0., 3.]));
    assert_eq!(trajectory["generations"][0]["peak_time"], 2.);
    assert_eq!(
        trajectory["generations"][0]["peak_rate"],
        response["predictions"]["rates"][2][0]
    );
    assert_eq!(trajectory["generations"][1]["status"], "no_active_samples");
    let milestones = &response["threshold_diagnostics"]["intrinsic_adoption_milestones"];
    assert_eq!(milestones["generations"][1]["reference_time"], 4.);
    assert!(milestones["generations"][1]["takeoff"].is_null());
    assert!(milestones["generations"][0]["saturation"].is_null());
}

#[test]
fn forced_successive_milestones_resolve_between_requested_samples() {
    let result = KernelBinding::new()
        .summarize_model(
            "norton_bass",
            json!({
                "inputs":{"time":[3.,7.]},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.},
                "constructor_kwargs":{
                    "variant":"successive_generations","launch_times":[1.],
                    "parameter_events":[{"time":2.,"parameters":{"p1":0.5,"m1":300.}}]
                }
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let row = &result["threshold_diagnostics"]["intrinsic_adoption_milestones"]["generations"][0];
    for (key, fraction) in [("takeoff", 0.1_f64), ("median", 0.5), ("saturation", 0.9)] {
        let expected = 2. + (-(-fraction).ln_1p() - 0.1) / 0.5;
        assert!((row[key].as_f64().unwrap() - expected).abs() < 1e-7);
    }
}

#[test]
fn historical_milestones_use_fixed_base_market_even_after_capacity_override() {
    let response = KernelBinding::new()
        .summarize_model(
            "norton_bass",
            json!({
                "inputs":{"time":[1.,2.,3.]},
                "parameters":{"p1":0.1,"q1":0.,"m1":100.},
                "constructor_kwargs":{
                    "initial_time":1.,"initial_values":[12.],
                    "parameter_events":[{"time":2.,"parameters":{"m1":1000.}}]
                }
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let diagnostic = &response["threshold_diagnostics"]["net_level_milestones"];
    let row = &diagnostic["generations"][0];
    assert_eq!(diagnostic["window"], json!([1., 3.]));
    assert_eq!(row["reference_market"], 100.);
    assert_eq!(row["takeoff"], 1.);
    assert_eq!(row["median"], 3.);
    assert_eq!(row["saturation"], 3.);
}

#[test]
fn weighted_norton_fit_excludes_both_outputs_and_preserves_forecast() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/norton_events_scipy.json")).unwrap();
    let mut rows = fixture["rows"].as_array().unwrap().clone();
    let mut weights = vec![2.; rows.len()];
    rows[5] = json!([1e100, 1e100]);
    weights[5] = 0.;
    let binding = KernelBinding::new();
    let fit = binding.fit_model("norton_bass",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":rows},
        "constructor_kwargs":fixture["constructor_kwargs"],
        "fit_options":{"initial_parameters":fixture["parameters"],"weights":weights}
    })).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(
        fit["diagnostics"]["weighting"]["positive_observations"],
        rows.len() - 1
    );
    assert!(
        fit["diagnostics"]["weighting"]["weighted_rss_normalized_by_max"]
            .as_f64()
            .unwrap()
            < 1e-8
    );
    let p = binding
        .predict_model(
            "norton_bass",
            json!({"inputs":{"time":fixture["forecast_time"]},"state":fit["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    for (actual, expected) in p["rows"][0]
        .as_array()
        .unwrap()
        .iter()
        .zip(fixture["forecast_rows"][0].as_array().unwrap())
    {
        assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-3);
    }
}
