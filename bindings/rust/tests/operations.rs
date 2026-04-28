use innovate_rust::{json, KernelBinding, KernelOperation, KernelResponse};

#[test]
fn diagnostics_summary_extracts_support_information() {
    let response = KernelResponse {
        schema_version: "1.0".to_string(),
        operation: KernelOperation::DiagnoseModel,
        model_key: Some("logistic".to_string()),
        result: Some(json!({
            "diagnostics": {
                "support_level": "supported",
                "provenance": "deterministic",
                "comparison_family": "fitted",
                "warnings": [
                    {
                        "code": "nonlinear_fit_warning"
                    }
                ],
                "metrics": {
                    "RMSE": 0.12,
                    "MAE": 0.08
                },
                "model_name": "LogisticModel"
            },
            "state": {
                "model_key": "logistic"
            }
        })),
        error: None,
        metadata: json!({}),
    };

    let summary = response
        .diagnostics_summary()
        .expect("diagnostics summary should be present");
    assert_eq!(summary.support_level, "supported");
    assert_eq!(summary.provenance, "deterministic");
    assert_eq!(summary.comparison_family, "fitted");
    assert_eq!(summary.warning_count, 1);
    assert_eq!(summary.metric_count, 2);
    assert_eq!(summary.model_name.as_deref(), Some("LogisticModel"));
}

#[test]
fn request_builders_preserve_model_identity_and_payload_shape() {
    let binding = KernelBinding::new();
    let payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let predict_request = binding.predict_model_request("logistic", payload.clone());
    let simulate_request = binding.simulate_model_request("logistic", payload.clone());
    let summarize_request = binding.summarize_model_request("logistic", payload.clone());
    let diagnose_request = binding.diagnose_model_request("logistic", payload);

    assert_eq!(predict_request.operation, KernelOperation::PredictModel);
    assert_eq!(simulate_request.operation, KernelOperation::SimulateModel);
    assert_eq!(summarize_request.operation, KernelOperation::SummarizeModel);
    assert_eq!(diagnose_request.operation, KernelOperation::DiagnoseModel);

    assert_eq!(predict_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(simulate_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(summarize_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(diagnose_request.model_key.as_deref(), Some("logistic"));
}

#[test]
fn native_logistic_fit_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let true_params = json!({
        "L": 100.0,
        "k": 0.65,
        "x0": 3.0
    });
    let time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let observed = vec![
        12.45533581839509,
        21.416501696862853,
        34.29895373044305,
        50.0,
        65.70104626955695,
        78.58349830313714,
    ];
    let payload = json!({
        "inputs": {
            "time": time,
            "observed": observed,
        }
    });

    let request = binding.fit_model_request("logistic", payload.clone());
    let native = binding
        .fit_model_native(&request)
        .expect("native logistic fit should succeed");
    let bridged = binding
        .fit_model_via_bridge("logistic", payload)
        .expect("Python bridge fit should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::FitModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native.result.expect("native fit should include a result");
    let bridged_result = bridged.result.expect("bridged fit should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);

    for key in ["L", "k", "x0"] {
        let native_value = native_result["parameters"][key]
            .as_f64()
            .expect("native parameter should be numeric");
        let bridged_value = bridged_result["parameters"][key]
            .as_f64()
            .expect("bridged parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-3,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }

    let native_predictions = native_result["predictions"]["values"]
        .as_array()
        .expect("native predictions should be an array");
    let bridged_predictions = bridged_result["predictions"]["values"]
        .as_array()
        .expect("bridged predictions should be an array");
    assert_eq!(native_predictions.len(), bridged_predictions.len());

    for (native_value, bridged_value) in native_predictions.iter().zip(bridged_predictions) {
        let native_value = native_value
            .as_f64()
            .expect("native prediction should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged prediction should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-2);
    }

    for key in ["L", "k", "x0"] {
        let native_value = native_result["state"]["parameters"][key]
            .as_f64()
            .expect("native state parameter should be numeric");
        let bridged_value = bridged_result["state"]["parameters"][key]
            .as_f64()
            .expect("bridged state parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-3,
            "{key} mismatch in state parameters: native={native_value}, bridged={bridged_value}"
        );
    }
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );
    let _ = true_params;
}

#[test]
fn native_logistic_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "logistic",
            "model_name": "LogisticModel",
            "constructor_kwargs": {},
            "parameters": {
                "L": 100.0,
                "k": 0.65,
                "x0": 3.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        }
    });

    let request = binding.predict_model_request("logistic", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native logistic prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("logistic", payload)
        .expect("Python bridge prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-12,
            "native value {native_value} should match bridged value {bridged_value}"
        );
    }
}

#[test]
fn native_fit_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let response = binding
        .fit_model("fisher_pry", payload)
        .expect("non-native fit should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::FitModel);
    assert_eq!(response.metadata["model_key"], "fisher_pry");
    assert!(response.result.is_some());
}

#[test]
fn native_logistic_simulation_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "logistic",
            "model_name": "LogisticModel",
            "constructor_kwargs": {},
            "parameters": {
                "L": 120.0,
                "k": 0.55,
                "x0": 2.5
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        }
    });

    let request = binding.simulate_model_request("logistic", payload.clone());
    let native = binding
        .simulate_model_native(&request)
        .expect("native logistic simulation should succeed");
    let bridged = binding
        .simulate_model_via_bridge("logistic", payload)
        .expect("Python bridge simulation should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SimulateModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native simulation response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged simulation response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-12);
    }
}

#[test]
fn native_prediction_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "fisher_pry",
            "model_name": "FisherPryModel",
            "constructor_kwargs": {},
            "parameters": {
                "alpha": 1.6,
                "t0": 2.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0]
        }
    });

    let response = binding
        .predict_model("fisher_pry", payload)
        .expect("non-native prediction should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::PredictModel);
    assert_eq!(response.metadata["model_key"], "fisher_pry");
    assert!(response.result.is_some());
}

#[test]
fn native_simulation_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "fisher_pry",
            "model_name": "FisherPryModel",
            "constructor_kwargs": {},
            "parameters": {
                "alpha": 1.6,
                "t0": 2.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0]
        }
    });

    let response = binding
        .simulate_model("fisher_pry", payload)
        .expect("non-native simulation should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::SimulateModel);
    assert_eq!(response.metadata["model_key"], "fisher_pry");
    assert!(response.result.is_some());
}
