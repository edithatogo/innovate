use innovate_core::{KernelBinding, json};
#[test]
fn zero_weights_exclude_outliers_and_common_weight_scaling_is_invariant() {
    let time: Vec<_> = (0..25).map(|i| i as f64).collect();
    let mut y: Vec<_> = time
        .iter()
        .map(|t| 100. / (1. + (-0.4 * (t - 7.)).exp()))
        .collect();
    y[20] = 300.;
    let mut weights = vec![1.; 25];
    weights[20] = 0.;
    let p = json!({"inputs":{"time":time,"observed":y},"fit_options":{"weights":weights,"initial_parameters":{"L":110.,"k":0.35,"x0":8.}}});
    let a = KernelBinding::new()
        .fit_model("logistic", p.clone())
        .unwrap()
        .result
        .unwrap();
    assert!((a["parameters"]["L"].as_f64().unwrap() - 100.).abs() < 1e-5);
    assert_eq!(a["diagnostics"]["weighting"]["positive_observations"], 24);
    assert!(a["diagnostics"]["metrics"]["AIC"].is_null());
    let mut p = p;
    p["fit_options"]["weights"] = json!(weights.iter().map(|w| w * 1e200).collect::<Vec<_>>());
    let b = KernelBinding::new()
        .fit_model("logistic", p)
        .unwrap()
        .result
        .unwrap();
    assert_eq!(a["parameters"], b["parameters"]);
    assert!(
        (a["predictions"]["values"][20].as_f64().unwrap() - 100. / (1. + (-5.2_f64).exp())).abs()
            < 1e-5
    );
}
#[test]
fn invalid_weights_and_insufficient_positive_support_fail() {
    for w in [
        json!([1.]),
        json!([0., 0., 0., 0.]),
        json!([1., -1., 1., 1.]),
        json!([1., 1., 0., 0.]),
        json!([1., null, 1., 1.]),
    ] {
        assert!(KernelBinding::new().fit_model("logistic",json!({"inputs":{"time":[0.,1.,2.,3.],"observed":[1.,2.,3.,4.]},"fit_options":{"weights":w}})).is_err());
    }
}

#[test]
fn weighted_logistic_matches_independent_scipy_reference() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/weighted_scipy.json")).unwrap();
    let a=KernelBinding::new().fit_model("logistic",json!({"inputs":{"time":fixture["time"],"observed":fixture["observed"]},"fit_options":{"weights":fixture["weights"],"initial_parameters":{"L":110.,"k":0.35,"x0":8.}}})).unwrap().result.unwrap();
    for name in ["L", "k", "x0"] {
        assert!(
            (a["parameters"][name].as_f64().unwrap()
                - fixture["parameters"][name].as_f64().unwrap())
            .abs()
                < 0.0001
        );
    }
    let maximum = fixture["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .fold(0., f64::max);
    assert!(
        (a["diagnostics"]["weighting"]["weighted_rss_normalized_by_max"]
            .as_f64()
            .unwrap()
            * maximum
            - fixture["weighted_rss"].as_f64().unwrap())
        .abs()
            < 1e-6
    );
}
