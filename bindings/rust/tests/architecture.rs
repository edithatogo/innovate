use innovate_rust::{KernelBinding, KernelOperation};

#[test]
fn kernel_binding_exposes_the_expected_contract_surface() {
    let binding = KernelBinding::new();

    assert_eq!(binding.schema_version(), "1.0");
    assert_eq!(
        binding.bridge_script_path(),
        std::path::PathBuf::from("inst/python/kernel_bridge.py")
    );
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
}
