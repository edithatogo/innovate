use innovate_core::{KernelBinding, json};
#[test]
fn historical_lock_in_matches_independent_ode_reference() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lock_in_scipy.json")).unwrap();
    let a=KernelBinding::new().predict_model("lock_in",json!({"inputs":{"time":f["time"]},"parameters":f["parameters"],"constructor_kwargs":{"initial_values":f["initial"]}})).unwrap().result.unwrap();
    for (a, b) in a["rows"]
        .as_array()
        .unwrap()
        .iter()
        .zip(f["rows"].as_array().unwrap())
    {
        for i in 0..2 {
            assert!((a[i].as_f64().unwrap() - b[i].as_f64().unwrap()).abs() < 1e-5);
        }
    }
}
#[test]
fn lock_in_fitting_preserves_trajectories_and_reports_structural_ambiguity() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lock_in_scipy.json")).unwrap();
    let a=KernelBinding::new().fit_model("lock_in",json!({"inputs":{"time":f["time"],"observed":f["rows"]},"fit_options":{"initial_parameters":f["parameters"]}})).unwrap().result.unwrap();
    assert_eq!(
        a["diagnostics"]["structural_identifiability"]["rank_upper_bound"],
        6
    );
    assert!(a["diagnostics"]["metrics"]["RMSE"].as_f64().unwrap() < 1e-4);
}
#[test]
fn historical_lock_in_clips_reported_levels() {
    let a=KernelBinding::new().predict_model("lock_in",json!({"inputs":{"time":[0.]},"parameters":{"alpha1":0.3,"alpha2":0.2,"beta1":0.1,"beta2":0.1,"gamma1":0.1,"gamma2":0.1,"m":100.},"constructor_kwargs":{"initial_values":[120.,150.]}})).unwrap().result.unwrap();
    assert_eq!(a["rows"][0], json!([100., 100.]));
}
#[test]
fn lock_in_forecasts_from_saved_origin_and_labels_historical_rates() {
    let parameters =
        json!({"alpha1":0.3,"alpha2":0.2,"beta1":0.,"beta2":0.,"gamma1":0.,"gamma2":0.,"m":1000.});
    let r=KernelBinding::new().predict_model("lock_in",json!({"inputs":{"time":[10.,20.]},"parameters":parameters,"constructor_kwargs":{"initial_values":[10.,0.]}})).unwrap().result.unwrap();
    assert!((r["rows"][0][0].as_f64().unwrap() - 1000. / (1. + 99. * (-3_f64).exp())).abs() < 1e-5);
    assert!(
        r["metadata"]["rate_semantics"]
            .as_str()
            .unwrap()
            .contains("not derivative of output clipping")
    );
}
#[test]
fn lock_in_fitter_rejects_initial_market_below_either_observed_series() {
    let p = json!({"inputs":{"time":[0.,1.,2.,3.],"observed":[[1.,10.],[2.,20.],[3.,30.],[4.,40.]]},"fit_options":{"initial_parameters":{"alpha1":0.3,"alpha2":0.2,"beta1":0.02,"beta2":0.03,"gamma1":0.04,"gamma2":0.05,"m":20.},"max_iterations":1}});
    let e = KernelBinding::new().fit_model("lock_in", p).unwrap_err();
    assert!(e.message.contains("maximum observed"));
}

#[test]
fn forced_lock_in_projects_to_current_capacity_without_resetting_raw_state() {
    let mut payload = json!({
        "inputs":{"time":[0.,1.,2.,3.],"covariates":{"price":[0.,1.,2.,0.]}},
        "parameters":{"alpha1":0.,"alpha2":0.,"beta1":0.,"beta2":0.,"gamma1":0.,"gamma2":0.,"m":100.,
            "beta_alpha1_price":0.,"beta_alpha2_price":0.,"beta_beta1_price":0.,"beta_beta2_price":0.,
            "beta_gamma1_price":0.,"beta_gamma2_price":0.,"beta_m_price":-10.},
        "constructor_kwargs":{"initial_values":[95.,0.],"covariates":["price"]}
    });
    let result = KernelBinding::new()
        .predict_model("lock_in", payload.clone())
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        result["rows"],
        json!([[95., 0.], [90., 0.], [80., 0.], [95., 0.]])
    );
    assert_eq!(
        result["rates"],
        json!([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    );
    let forecast = KernelBinding::new()
        .predict_model(
            "lock_in",
            json!({
                "inputs":{"time":[4.]},"state":result["state"]
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(forecast["rows"], json!([[95., 0.]]));
    payload["inputs"]["covariates"]["price"] = json!([0., 1., 10., 0.]);
    assert!(
        KernelBinding::new()
            .predict_model("lock_in", payload)
            .is_err()
    );
}

#[test]
fn lock_in_forcing_matches_independent_reference_and_retains_confounding() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lock_forcing_scipy.json")).unwrap();
    let options = json!({"initial_values":fixture["initial"],"covariates":["price"]});
    let binding = KernelBinding::new();
    let result = binding
        .predict_model(
            "lock_in",
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
                assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
            }
        }
    }
    let mut start = fixture["parameters"].clone();
    start["alpha1"] = json!(0.29);
    let fit = binding.fit_model("lock_in",json!({
        "inputs":{"time":fixture["time"],"covariates":fixture["covariates"],"observed":fixture["rows"]},
        "constructor_kwargs":options,"fit_options":{"initial_parameters":start,"max_iterations":200}
    })).unwrap().result.unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-8);
    assert_eq!(
        fit["diagnostics"]["structural_identifiability"]["parameters"],
        14
    );
    assert_eq!(
        fit["diagnostics"]["structural_identifiability"]["rank_upper_bound"],
        13
    );
    let forecast = binding
        .predict_model(
            "lock_in",
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
fn forced_market_baseline_can_be_below_observed_levels() {
    let binding = KernelBinding::new();
    let parameters = json!({"alpha1":0.2,"alpha2":0.1,"beta1":0.02,"beta2":0.01,"gamma1":0.03,"gamma2":0.02,"m":20.,
        "beta_alpha1_price":0.,"beta_alpha2_price":0.,"beta_beta1_price":0.,"beta_beta2_price":0.,
        "beta_gamma1_price":0.,"beta_gamma2_price":0.,"beta_m_price":80.});
    let inputs = json!({"time":[0.,1.,2.,3.],"covariates":{"price":[1.,1.,1.,1.]}});
    let options = json!({"initial_values":[30.,20.],"covariates":["price"]});
    let result = binding
        .predict_model(
            "lock_in",
            json!({
                "inputs":inputs,"parameters":parameters,"constructor_kwargs":options
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let mut training = inputs;
    training["observed"] = result["rows"].clone();
    let fit = binding
        .fit_model(
            "lock_in",
            json!({
                "inputs":training,"constructor_kwargs":options,
                "fit_options":{"initial_parameters":parameters,"max_iterations":2}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert!(fit["parameters"]["m"].as_f64().unwrap() < 30.);
}
