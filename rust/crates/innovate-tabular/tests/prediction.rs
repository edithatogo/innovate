use innovate_tabular::predict;
use polars::prelude::*;
use serde_json::json;

#[test]
fn prediction_preserves_identity_and_adds_native_values() {
    let frame = df!("id" => ["a", "b", "c"], "time" => [0.0, 1.0, 2.0]).unwrap();
    let out = predict(frame, "logistic", json!({"L":100.0,"k":1.0,"x0":1.0})).unwrap();
    assert_eq!(out.column("id").unwrap().str().unwrap().get(1), Some("b"));
    assert_eq!(
        out.column("prediction").unwrap().f64().unwrap().get(1),
        Some(50.0)
    );
}

#[test]
fn invalid_time_contract_fails_in_rust() {
    for times in [vec![0.0, f64::NAN], vec![1.0, 0.0], vec![0.0, 0.0]] {
        let frame = df!("time" => times).unwrap();
        assert!(predict(frame, "logistic", json!({"L":100.0,"k":1.0,"x0":1.0})).is_err());
    }
}
