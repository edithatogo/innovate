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
