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
