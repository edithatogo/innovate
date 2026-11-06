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
