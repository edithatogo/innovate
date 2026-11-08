use innovate_core::{KernelBinding, KernelRequest, json};

#[test]
fn gompertz_uses_all_three_parameters() {
    let request = KernelRequest::predict_model(
        "gompertz",
        json!({
            "inputs":{"time":[0.0,1.0]}, "parameters":{"a":100.0,"b":2.0,"c":0.5}
        }),
    );
    let r = KernelBinding::new()
        .invoke(&request)
        .unwrap()
        .result
        .unwrap();
    assert!((r["values"][0].as_f64().unwrap() - 100.0 * (-2.0_f64).exp()).abs() < 1e-12);
}

#[test]
fn diagnostics_do_not_fabricate_uncomputed_significance_tests() {
    let time: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-(t - 2.0)).exp()))
        .collect();
    let r = KernelBinding::new()
        .fit_model(
            "fisher_pry",
            json!({"inputs":{"time":time,"observed":observed}}),
        )
        .unwrap()
        .result
        .unwrap();
    let residuals = &r["diagnostics"]["residual_analysis"];
    assert!(residuals["shapiro_wilk_p"].is_null());
    assert!(residuals["breusch_pagan_p"].is_null());
    assert_eq!(residuals["significance_tests"], "not_implemented");
}

#[test]
fn invalid_parameters_and_unsorted_times_fail_before_computation() {
    for payload in [
        json!({"inputs":{"time":[0.0,1.0]},"parameters":{"L":-1.0,"k":1.0,"x0":0.0}}),
        json!({"inputs":{"time":[1.0,0.0]},"parameters":{"L":1.0,"k":1.0,"x0":0.0}}),
    ] {
        assert!(
            KernelBinding::new()
                .invoke(&KernelRequest::predict_model("logistic", payload))
                .is_err()
        );
    }
}

#[test]
fn native_fits_recover_synthetic_logistic_and_gompertz_parameters() {
    let time: Vec<f64> = (0..20).map(f64::from).collect();
    for (model, parameters, observed) in [
        (
            "logistic",
            json!({"L":100.0,"k":0.6,"x0":8.0}),
            time.iter()
                .map(|t| 100.0 / (1.0 + (-0.6 * (t - 8.0)).exp()))
                .collect::<Vec<_>>(),
        ),
        (
            "gompertz",
            json!({"a":100.0,"b":3.0,"c":0.3}),
            time.iter()
                .map(|t| 100.0 * (-3.0 * (-0.3 * t).exp()).exp())
                .collect::<Vec<_>>(),
        ),
    ] {
        let result = KernelBinding::new()
            .fit_model(model, json!({"inputs":{"time":time,"observed":observed}}))
            .unwrap()
            .result
            .unwrap();
        for (key, value) in parameters.as_object().unwrap() {
            assert!(
                (result["parameters"][key].as_f64().unwrap() - value.as_f64().unwrap()).abs()
                    < 1e-4,
                "{model} {key}: {result}"
            );
        }
    }
}

#[test]
fn bass_forecasts_use_absolute_time_without_needing_the_origin_in_every_batch() {
    let request = KernelRequest::predict_model(
        "bass",
        json!({
            "inputs":{"time":[3.0,4.0]},
            "state":{"model_key":"bass","parameters":{"p":0.03,"q":0.35,"m":1000.0},"constructor_kwargs":{},"predict_kwargs":{}}
        }),
    );
    let values = KernelBinding::new()
        .invoke(&request)
        .unwrap()
        .result
        .unwrap();
    let decay = (-0.38_f64 * 3.0).exp();
    let expected = 1000.0 * (1.0 - decay) / (1.0 + 0.35 / 0.03 * decay);
    assert!((values["values"][0].as_f64().unwrap() - expected).abs() < 1e-10);
}

#[test]
fn bass_fit_recovers_parameters_on_irregular_times_and_rescaled_units() {
    for (time_scale, adoption_scale) in [(1.0, 1.0), (365.0, 1000.0)] {
        let time: Vec<f64> = [0.0, 0.3, 1.1, 2.0, 4.7, 7.0, 9.5, 13.2, 20.0, 30.0]
            .iter()
            .map(|t| t * time_scale)
            .collect();
        let p = 0.03 / time_scale;
        let q = 0.35 / time_scale;
        let m = 100.0 * adoption_scale;
        let observed: Vec<f64> = time
            .iter()
            .map(|t| {
                let decay = (-(p + q) * t).exp();
                m * (1.0 - decay) / (1.0 + q / p * decay)
            })
            .collect();
        let fit = KernelBinding::new()
            .fit_model("bass", json!({"inputs":{"time":time,"observed":observed}}))
            .unwrap()
            .result
            .unwrap();
        for (key, expected) in [("p", p), ("q", q), ("m", m)] {
            let actual = fit["parameters"][key].as_f64().unwrap();
            assert!(
                (actual / expected - 1.0).abs() < 1e-5,
                "{key}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn information_criteria_use_consistent_gaussian_parameter_counts() {
    let time: Vec<f64> = (0..21).map(f64::from).collect();
    for (model, count, observed) in [
        (
            "fisher_pry",
            3.0,
            time.iter()
                .enumerate()
                .map(|(i, t)| {
                    1.0 / (1.0 + (-0.6 * (t - 8.0)).exp())
                        + if i % 2 == 0 { 0.0001 } else { -0.0001 }
                })
                .collect::<Vec<_>>(),
        ),
        (
            "gompertz",
            4.0,
            time.iter()
                .enumerate()
                .map(|(i, t)| {
                    100.0 * (-3.0 * (-0.3 * t).exp()).exp() + if i % 2 == 0 { 0.01 } else { -0.01 }
                })
                .collect::<Vec<_>>(),
        ),
    ] {
        let binding = KernelBinding::new();
        let fit = binding
            .fit_model(model, json!({"inputs":{"time":time,"observed":observed}}))
            .unwrap()
            .result
            .unwrap();
        let summary = binding
            .summarize_model(
                model,
                json!({"inputs":{"time":time,"observed":observed},"state":fit["state"]}),
            )
            .unwrap()
            .result
            .unwrap();
        for result in [&fit, &summary] {
            let metrics = &result["diagnostics"]["metrics"];
            let rss = metrics["RSS"].as_f64().unwrap();
            let expected_aic = 21.0 * (rss / 21.0).ln() + 2.0 * count;
            let expected_bic = 21.0 * (rss / 21.0).ln() + count * 21.0_f64.ln();
            assert!(
                (metrics["AIC"].as_f64().unwrap() - expected_aic).abs() < 1e-9,
                "{model}: {metrics}"
            );
            assert!((metrics["BIC"].as_f64().unwrap() - expected_bic).abs() < 1e-9);
        }
    }
}

#[test]
fn bass_prediction_recovers_after_rejected_parameters() {
    let binding = KernelBinding::new();
    let predict = |parameters| {
        binding.invoke(&KernelRequest::predict_model(
            "bass",
            json!({"inputs":{"time":[1.0,2.0,3.0]},"parameters":parameters}),
        ))
    };
    for parameters in [json!({}), json!({"p":-0.03,"q":0.38,"m":1000.0})] {
        assert!(predict(parameters).is_err());
    }
    let result = predict(json!({"p":0.03,"q":0.38,"m":1000.0}))
        .unwrap()
        .result
        .unwrap();
    let values = result["values"].as_array().unwrap();
    assert_eq!(values.len(), 3);
    for (index, value) in values.iter().enumerate() {
        let time = (index + 1) as f64;
        let decay = (-0.41 * time).exp();
        let expected = 1000.0 * (1.0 - decay) / (1.0 + 0.38 / 0.03 * decay);
        assert!((value.as_f64().unwrap() - expected).abs() < 1e-10);
    }
}
