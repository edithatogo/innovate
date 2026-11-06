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
