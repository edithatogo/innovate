use innovate_rust::{json, KernelBinding, KernelOperation, KernelRequest};

#[test]
fn kernel_binding_exposes_the_expected_contract_surface() {
    let binding = KernelBinding::new();

    assert_eq!(binding.schema_version(), "1.0");
    assert_eq!(
        binding.bridge_script_path(),
        std::path::PathBuf::from("inst/python/kernel_bridge.py")
    );
    assert!(binding.bridge_script_exists());
    assert_eq!(
        binding.available_operations(),
        [
            KernelOperation::DiscoverModels,
            KernelOperation::FitModel,
            KernelOperation::PredictModel,
            KernelOperation::SimulateModel,
            KernelOperation::SummarizeModel,
            KernelOperation::DiagnoseModel,
        ]
    );
    assert_eq!(
        binding.discover_models_request(),
        KernelRequest::discover_models()
    );

    let fit_request = binding.fit_model_request(
        "logistic",
        json!({
            "inputs": {
                "time": [0.0, 1.0, 2.0],
                "observed": [0.05, 0.12, 0.3]
            }
        }),
    );
    assert_eq!(fit_request.operation, KernelOperation::FitModel);
    assert_eq!(fit_request.model_key.as_deref(), Some("logistic"));
}
