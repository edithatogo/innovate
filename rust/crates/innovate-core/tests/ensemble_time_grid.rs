//! Public native boundary checks for ordered regime-ensemble observations.

use innovate_core::execute_json;
use serde_json::{Value, json};

fn request(time: Value) -> Value {
    json!({
        "schema_version": "2.0",
        "operation": "compose_regime_ensemble",
        "payload": {
            "time": time,
            "predictions": {
                "low": [2.0, 4.0, 8.0],
                "high": [6.0, 8.0, 12.0]
            },
            "weights": {"low": 1.0, "high": 3.0},
            "observed": [4.0, 7.0, 12.0]
        }
    })
}

fn execute(request: &Value) -> Value {
    serde_json::from_str(&execute_json(&request.to_string())).unwrap()
}

fn assert_invalid_time(time: Value) {
    let response = execute(&request(time));
    assert_eq!(response["error"]["code"], "invalid_request");
    assert_eq!(
        response["error"]["message"],
        "time must be finite and strictly increasing"
    );
    assert!(response["result"].is_null());
    assert_eq!(response["metadata"]["runtime"], "rust_native");
}

#[test]
fn duplicate_ensemble_times_are_rejected_including_signed_zero() {
    assert_invalid_time(json!([0.0, 0.0, 2.0]));
    assert_invalid_time(json!([-0.0, 0.0, 2.0]));
}

#[test]
fn descending_ensemble_times_are_rejected() {
    assert_invalid_time(json!([2.0, 1.0, 0.0]));
}

#[test]
fn locally_reversed_ensemble_times_are_rejected_without_sorting_rows() {
    assert_invalid_time(json!([0.0, 2.0, 1.0]));
}

#[test]
fn irregular_increasing_times_preserve_rows_and_weighted_diagnostics() {
    let time = json!([-2.0, 0.5, 7.0]);
    let response = execute(&request(time.clone()));
    assert!(response["error"].is_null(), "{response}");
    let result = &response["result"];
    assert_eq!(result["time"], time);
    // Independent arithmetic: (low + 3 * high) / 4.
    assert_eq!(result["mean"], json!([5.0, 7.0, 11.0]));
    assert_eq!(result["metadata"]["weights"]["low"], 0.25);
    assert_eq!(result["metadata"]["weights"]["high"], 0.75);
    let diagnostics = &result["diagnostics"];
    assert!((diagnostics["mae"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-14);
    let expected_rmse = (2.0_f64 / 3.0).sqrt();
    assert!((diagnostics["rmse"].as_f64().unwrap() - expected_rmse).abs() < 1e-14);
}

#[test]
fn singleton_ensemble_time_is_valid() {
    let response = execute(&json!({
        "schema_version": "2.0",
        "operation": "compose_regime_ensemble",
        "payload": {
            "time": [3.0],
            "predictions": {"low": [2.0], "high": [6.0]},
            "weights": {"low": 1.0, "high": 3.0}
        }
    }));
    assert!(response["error"].is_null(), "{response}");
    assert_eq!(response["result"]["time"], json!([3.0]));
    assert_eq!(response["result"]["mean"], json!([5.0]));
}

#[test]
fn positive_affine_time_change_preserves_ensemble_values() {
    let original = execute(&request(json!([-2.0, 0.5, 7.0])));
    let transformed_time = json!([1.0, 6.0, 19.0]);
    let transformed = execute(&request(transformed_time.clone()));
    assert!(original["error"].is_null(), "{original}");
    assert!(transformed["error"].is_null(), "{transformed}");
    assert_eq!(transformed["result"]["time"], transformed_time);
    assert_eq!(original["result"]["mean"], transformed["result"]["mean"]);
    assert_eq!(
        original["result"]["diagnostics"],
        transformed["result"]["diagnostics"]
    );
}
