use innovate_rust::{json, KernelBinding};

#[test]
fn rust_binding_can_run_the_stable_kernel_workflow_end_to_end() {
    let binding = KernelBinding::new();
    let discovery = binding
        .discover_models()
        .expect("discover_models should succeed");
    assert_eq!(discovery.schema_version, "1.0");
    assert!(discovery
        .models
        .iter()
        .any(|record| record.key == "logistic"));

    let fit_payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "observed": [0.05, 0.12, 0.3, 0.6, 0.85]
        }
    });
    let fit_response = binding
        .fit_model("logistic", fit_payload)
        .expect("fit_model should succeed");

    let fit_summary = fit_response
        .diagnostics_summary()
        .expect("fit response should include diagnostics");
    assert_eq!(fit_summary.support_level, "supported");
    assert_eq!(fit_summary.provenance, "deterministic");

    let state = fit_response
        .result_object()
        .and_then(|result| result.get("state"))
        .cloned()
        .expect("fit response should include serialized model state");

    let predict_payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0]
        },
        "state": state
    });
    let predict_response = binding
        .predict_model("logistic", predict_payload.clone())
        .expect("predict_model should succeed");
    assert!(predict_response
        .result_object()
        .expect("predict response should contain an object payload")
        .contains_key("values"));

    let simulate_response = binding
        .simulate_model("logistic", predict_payload.clone())
        .expect("simulate_model should succeed");
    assert!(simulate_response
        .result_object()
        .expect("simulate response should contain an object payload")
        .contains_key("values"));

    let analyze_payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "observed": [0.05, 0.12, 0.3, 0.6, 0.85]
        },
        "state": fit_response
            .result_object()
            .and_then(|result| result.get("state"))
            .cloned()
            .expect("fit response should include serialized model state")
    });

    let summarize_response = binding
        .summarize_model("logistic", analyze_payload.clone())
        .expect("summarize_model should succeed");
    assert!(summarize_response
        .result_object()
        .expect("summarize response should contain an object payload")
        .contains_key("diagnostics"));

    let diagnose_response = binding
        .diagnose_model("logistic", analyze_payload)
        .expect("diagnose_model should succeed");
    let diagnose_summary = diagnose_response
        .diagnostics_summary()
        .expect("diagnose response should include diagnostics");
    assert_eq!(diagnose_summary.support_level, "supported");
    assert!(diagnose_response
        .result_object()
        .expect("diagnose response should contain an object payload")
        .contains_key("state"));
}
