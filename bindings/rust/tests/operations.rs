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
fn native_logistic_summary_matches_python_bridge_contract() {
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
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "observed": [
                12.45533581839509,
                21.416501696862853,
                34.29895373044305,
                50.0,
                65.70104626955695,
                78.58349830313714
            ]
        }
    });

    let request = binding.summarize_model_request("logistic", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native logistic summary should succeed");
    let bridged = binding
        .summarize_model_via_bridge("logistic", payload)
        .expect("Python bridge summary should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native summary should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged summary should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);
    assert_eq!(
        native_result["parameter_names"],
        bridged_result["parameter_names"]
    );
    assert_eq!(
        native_result["constructor_kwargs"],
        bridged_result["constructor_kwargs"]
    );
    assert_eq!(native_result["state"], bridged_result["state"]);

    let native_diagnostics = native_result["diagnostics"]
        .as_object()
        .expect("native diagnostics");
    let bridged_diagnostics = bridged_result["diagnostics"]
        .as_object()
        .expect("bridged diagnostics");

    assert_eq!(native_diagnostics["support_level"], "supported");
    assert_eq!(native_diagnostics["provenance"], "deterministic");
    assert_eq!(native_diagnostics["comparison_family"], "fitted");
    assert_eq!(native_diagnostics["model_name"], "LogisticModel");
    assert_eq!(
        native_diagnostics["uncertainty"]["report_type"],
        "point_estimate"
    );
    assert_eq!(
        native_diagnostics["warnings"],
        bridged_diagnostics["warnings"]
    );

    for key in ["MSE", "RMSE", "MAE", "R-squared", "R_squared", "RSS"] {
        let native_value = native_diagnostics["metrics"][key]
            .as_f64()
            .expect("native metric should be numeric");
        let bridged_value = bridged_diagnostics["metrics"][key]
            .as_f64()
            .expect("bridged metric should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }

    let native_residuals = native_diagnostics["residuals"]
        .as_array()
        .expect("native residuals should be an array");
    let bridged_residuals = bridged_diagnostics["residuals"]
        .as_array()
        .expect("bridged residuals should be an array");
    assert_eq!(native_residuals.len(), bridged_residuals.len());
    for (native_value, bridged_value) in native_residuals.iter().zip(bridged_residuals) {
        let native_value = native_value
            .as_f64()
            .expect("native residual should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged residual should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-6);
    }

    let native_analysis = native_diagnostics["residual_analysis"]
        .as_object()
        .expect("native residual analysis should be an object");
    let bridged_analysis = bridged_diagnostics["residual_analysis"]
        .as_object()
        .expect("bridged residual analysis should be an object");
    for key in [
        "mean_residual",
        "max_abs_residual",
        "std_residual",
        "durbin_watson",
    ] {
        let native_value = native_analysis[key]
            .as_f64()
            .expect("native analysis metric");
        let bridged_value = bridged_analysis[key]
            .as_f64()
            .expect("bridged analysis metric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch"
        );
    }
}

#[test]
fn native_logistic_diagnose_matches_python_bridge_contract() {
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
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "observed": [
                12.45533581839509,
                21.416501696862853,
                34.29895373044305,
                50.0,
                65.70104626955695,
                78.58349830313714
            ]
        }
    });

    let request = binding.diagnose_model_request("logistic", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native logistic diagnostics should succeed");
    let bridged = binding
        .diagnose_model_via_bridge("logistic", payload)
        .expect("Python bridge diagnostics should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native diagnostics should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged diagnostics should include a result");

    assert_eq!(native_result["state"], bridged_result["state"]);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );

    for key in ["MSE", "RMSE", "MAE", "R-squared", "R_squared", "RSS"] {
        let native_value = native_result["diagnostics"]["metrics"][key]
            .as_f64()
            .expect("native metric should be numeric");
        let bridged_value = bridged_result["diagnostics"]["metrics"][key]
            .as_f64()
            .expect("bridged metric should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }
}

#[test]
fn native_summary_and_diagnose_fall_back_to_bridge_for_non_native_models() {
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
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let summary = binding
        .summarize_model("fisher_pry", payload.clone())
        .expect("non-native summary should fall back to the bridge");
    let diagnose = binding
        .diagnose_model("fisher_pry", payload)
        .expect("non-native diagnostics should fall back to the bridge");

    assert_eq!(summary.operation, KernelOperation::SummarizeModel);
    assert_eq!(diagnose.operation, KernelOperation::DiagnoseModel);
    assert_eq!(summary.metadata["model_key"], "fisher_pry");
    assert_eq!(diagnose.metadata["model_key"], "fisher_pry");
    assert!(summary.result.is_some());
    assert!(diagnose.result.is_some());
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
