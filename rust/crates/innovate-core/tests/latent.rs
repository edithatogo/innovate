use innovate_core::{KernelBinding, json};
#[test]
fn saved_latent_state_interpolates_and_extends_endpoints() {
    let time = [0_f64, 1., 2., 3., 4., 5., 6.];
    let a=KernelBinding::new().predict_model("latent_process",json!({"inputs":{"time":time},"constructor_kwargs":{"base_model":"logistic","latent_time":[1.,3.,5.],"latent_values":[2.,-4.,6.]},"parameters":{"base_L":100.,"base_k":0.5,"base_x0":3.,"latent_smoothing":0.3,"latent_noise":2.}})).unwrap().result.unwrap();
    let offsets = [2., 2., -1., -4., 1., 6., 6.];
    let mut maximum = 0_f64;
    for (i, t) in time.iter().enumerate() {
        maximum = maximum.max(100. / (1. + (-0.5 * (t - 3.)).exp()) + offsets[i]);
        assert!((a["values"][i].as_f64().unwrap() - maximum).abs() < 1e-10);
    }
}
#[test]
fn fitted_latent_state_matches_independent_residual_recursion_and_forecast() {
    let time: Vec<_> = (0..21).map(|i| i as f64).collect();
    let y: Vec<_> = time
        .iter()
        .map(|t| 100. / (1. + (-0.5 * (t - 8.)).exp()) + 2. + 0.1 * t.sin())
        .collect();
    let f=KernelBinding::new().fit_model("latent_process",json!({"inputs":{"time":time,"observed":y},"constructor_kwargs":{"base_model":"logistic","smoothing":0.4}})).unwrap().result.unwrap();
    assert!(f["diagnostics"]["metrics"]["AIC"].is_null());
    assert_eq!(f["diagnostics"]["information_criteria"]["available"], false);
    let p = &f["parameters"];
    let mut previous = 0.;
    let mut residuals = vec![];
    for (i, t) in time.iter().enumerate() {
        let baseline = p["base_L"].as_f64().unwrap()
            / (1. + (-p["base_k"].as_f64().unwrap() * (t - p["base_x0"].as_f64().unwrap())).exp());
        let residual = y[i] - baseline;
        residuals.push(residual);
        previous = if i == 0 {
            residual
        } else {
            0.4 * residual + 0.6 * previous
        };
        assert!(
            (f["state"]["constructor_kwargs"]["latent_values"][i]
                .as_f64()
                .unwrap()
                - previous)
                .abs()
                < 1e-10
        );
    }
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let noise = (residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (residuals.len() - 1) as f64)
        .sqrt();
    assert!((p["latent_noise"].as_f64().unwrap() - noise).abs() < 1e-10);
    let a = KernelBinding::new()
        .predict_model(
            "latent_process",
            json!({"inputs":{"time":[30.]},"state":f["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let expected = p["base_L"].as_f64().unwrap()
        / (1. + (-p["base_k"].as_f64().unwrap() * (30. - p["base_x0"].as_f64().unwrap())).exp())
        + previous;
    assert!((a["values"][0].as_f64().unwrap() - expected.max(0.)).abs() < 1e-10);
}
#[test]
fn invalid_latent_arrays_and_parameter_domains_fail() {
    for kw in [
        json!({"latent_time":[1.],"latent_values":[]}),
        json!({"latent_time":[1.,1.],"latent_values":[0.,1.]}),
        json!({"smoothing":-0.1}),
        json!({"unknown":true}),
    ] {
        assert!(KernelBinding::new().predict_model("latent_process",json!({"inputs":{"time":[1.]},"constructor_kwargs":kw,"parameters":{"base_p":0.03,"base_q":0.3,"base_m":100.,"latent_smoothing":0.3,"latent_noise":0.}})).is_err());
    }
}

#[test]
fn latent_summary_clipping_and_singleton_increment_preserve_history() {
    let mut p = json!({"constructor_kwargs":{"base_model":"logistic","latent_time":[0.],"latent_values":[-200.]},"parameters":{"base_L":100.,"base_k":1.,"base_x0":2.,"latent_smoothing":0.,"latent_noise":0.}});
    let s = KernelBinding::new()
        .summarize_model("latent_process", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(s["details"]["latent_state_length"], 1);
    assert!(s.get("predictions").is_none());
    p["inputs"] = json!({"time":[1.,2.,3.]});
    assert_eq!(
        KernelBinding::new()
            .predict_model("latent_process", p.clone())
            .unwrap()
            .result
            .unwrap()["values"],
        json!([0., 0., 0.])
    );
    p["constructor_kwargs"]["latent_values"] = json!([2.]);
    p["inputs"] = json!({"time":[2.]});
    let a = KernelBinding::new()
        .simulate_model("latent_process", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(a["values"], json!([52.]));
    assert_eq!(a["increments"], json!([52.]));
    p["inputs"]["observed"] = json!([52.]);
    assert!(
        KernelBinding::new()
            .diagnose_model("latent_process", p)
            .unwrap()
            .result
            .unwrap()["diagnostics"]
            .is_object()
    );
}
#[test]
fn latent_invalid_parameters_unsupported_options_and_overflow_fail() {
    let base = json!({"inputs":{"time":[1.]},"parameters":{"base_p":0.03,"base_q":0.3,"base_m":100.,"latent_smoothing":0.3,"latent_noise":0.}});
    for (key, value) in [
        ("latent_noise", json!(-1.)),
        ("latent_smoothing", json!(1.1)),
        ("base_p", json!(-0.1)),
        ("extra", json!(1.)),
    ] {
        let mut p = base.clone();
        p["parameters"][key] = value;
        assert!(
            KernelBinding::new()
                .predict_model("latent_process", p)
                .is_err()
        );
    }
    for (key, value) in [
        ("simulation_options", json!({"seed":42})),
        ("predict_kwargs", json!({"noise":1.})),
        ("constructor_kwargs", json!({"base_model":"latent_process"})),
        ("inputs", json!({"time":[1.],"covariates":{}})),
    ] {
        let mut p = base.clone();
        p[key] = value;
        assert!(
            KernelBinding::new()
                .predict_model("latent_process", p)
                .is_err()
        );
    }
    let p = json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":{"base_model":"logistic","latent_time":[0.,1.],"latent_values":[1e308,1e308]},"parameters":{"base_L":1.7e308,"base_k":1.,"base_x0":-100.,"latent_smoothing":1.,"latent_noise":0.}});
    assert!(
        KernelBinding::new()
            .predict_model("latent_process", p)
            .is_err()
    );
}
