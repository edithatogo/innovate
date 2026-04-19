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
