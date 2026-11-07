use innovate_core::{KernelBinding, json};
#[test]
fn historical_logistic_covariates_and_event_boundary_match_closed_forms() {
    let p =
        json!({"L":100.,"k":0.5,"x0":2.,"beta_L_price":10.,"beta_k_price":0.1,"beta_x0_price":0.2});
    let a=KernelBinding::new().predict_model("logistic",json!({"inputs":{"time":[0.,1.,2.],"covariates":{"price":[0.,1.,2.]}},"parameters":p,"constructor_kwargs":{"covariates":["price"]}})).unwrap().result.unwrap();
    for (i, t) in [0_f64, 1., 2.].iter().enumerate() {
        let x = *t;
        let y = (100. + 10. * x) / (1. + (-(0.5 + 0.1 * x) * (t - (2. + 0.2 * x))).exp());
        assert!((a["values"][i].as_f64().unwrap() - y).abs() < 1e-10);
    }
    let r=KernelBinding::new().predict_model("logistic",json!({"inputs":{"time":[1.,2.,3.]},"parameters":{"L":100.,"k":0.5,"x0":2.,"L_post":200.,"k_post":1.,"x0_post":2.},"constructor_kwargs":{"t_event":2.}})).unwrap().result.unwrap();
    assert_eq!(r["values"][1], 100.);
    assert_eq!(r["rates"][1][0], 50.);
    let fixed_covariate_rate =
        110. * 0.6 * (1. + (0.72_f64).exp()).recip() * (1. + (-0.72_f64).exp()).recip();
    assert!((a["rates"][1][0].as_f64().unwrap() - fixed_covariate_rate).abs() < 1e-10);
}
#[test]
fn historical_event_fit_forecasts_and_reports_rank() {
    let time: Vec<_> = (0..30).map(|i| i as f64).collect();
    let observed: Vec<_> = time
        .iter()
        .map(|t| {
            if *t < 12. {
                100. / (1. + (-0.5 * (t - 5.)).exp())
            } else {
                200. / (1. + (-0.3 * (t - 18.)).exp())
            }
        })
        .collect();
    for fit_options in [
        json!({"initial_parameters":{"L":110.,"k":0.4,"x0":6.,"L_post":220.,"k_post":0.4,"x0_post":17.}}),
        json!({}),
    ] {
        let fit=KernelBinding::new().fit_model("logistic",json!({"inputs":{"time":time,"observed":observed},"constructor_kwargs":{"t_event":12.},"fit_options":fit_options})).unwrap().result.unwrap();
        assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
        assert_eq!(fit["diagnostics"]["identifiability"]["rank"], 6);
        let a = KernelBinding::new()
            .predict_model(
                "logistic",
                json!({"inputs":{"time":[35.]},"state":fit["state"]}),
            )
            .unwrap()
            .result
            .unwrap();
        assert!(
            (a["values"][0].as_f64().unwrap() - 200. / (1. + (-0.3_f64 * 17.).exp())).abs() < 0.001
        );
    }
}
#[test]
fn historical_event_covariate_policy_is_explicit() {
    let mut p = json!({"inputs":{"time":[2.],"covariates":{"price":[1.]}},"constructor_kwargs":{"t_event":2.,"covariates":["price"]},"parameters":{"L":100.,"k":1.,"x0":2.,"L_post":200.,"k_post":1.,"x0_post":2.,"beta_L_price":20.,"beta_k_price":0.,"beta_x0_price":0.}});
    let a = KernelBinding::new()
        .predict_model("logistic", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(a["values"][0], 100.);
    p["constructor_kwargs"]["event_covariates"] = json!("additive");
    assert_eq!(
        KernelBinding::new()
            .predict_model("logistic", p)
            .unwrap()
            .result
            .unwrap()["values"][0],
        110.
    );
}
#[test]
fn advanced_shapes_and_options_fail_clearly() {
    let base = json!({"inputs":{"time":[0.,1.],"covariates":{"x":[0.,1.]}},"constructor_kwargs":{"covariates":["x"]},"parameters":{"L":100.,"k":0.5,"x0":2.,"beta_L_x":0.,"beta_k_x":0.,"beta_x0_x":0.}});
    for (key, value) in [
        ("constructor_kwargs", json!({"covariates":["x","x"]})),
        ("predict_kwargs", json!({"unknown":true})),
        ("inputs", json!({"time":[0.,0.],"covariates":{"x":[0.,1.]}})),
        ("inputs", json!({"time":[0.,1.],"covariates":{"x":[0.]}})),
    ] {
        let mut p = base.clone();
        p[key] = value;
        assert!(KernelBinding::new().predict_model("logistic", p).is_err());
    }
}
#[test]
fn additive_covariate_fit_recovers_independent_curve() {
    let t: Vec<_> = (0..30).map(|i| i as f64).collect();
    let x: Vec<_> = (0..30).map(|i| ((i * 7) % 5) as f64 - 2.).collect();
    let y: Vec<_> = t
        .iter()
        .zip(&x)
        .map(|(t, x)| (100. + 10. * x) / (1. + (-(0.4 + 0.02 * x) * (t - (10. + 0.1 * x))).exp()))
        .collect();
    for fit_options in [
        json!({"initial_parameters":{"L":110.,"k":0.3,"x0":9.,"beta_L_x":8.,"beta_k_x":0.,"beta_x0_x":0.}}),
        json!({}),
    ] {
        let f=KernelBinding::new().fit_model("logistic",json!({"inputs":{"time":t,"observed":y,"covariates":{"x":x}},"constructor_kwargs":{"covariates":["x"]},"fit_options":fit_options})).unwrap().result.unwrap();
        assert_eq!(f["diagnostics"]["optimizer"]["converged"], true);
        assert_eq!(f["diagnostics"]["identifiability"]["rank"], 6);
        assert!((f["parameters"]["beta_L_x"].as_f64().unwrap() - 10.).abs() < 0.001);
    }
}
