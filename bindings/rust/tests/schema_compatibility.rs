use std::fs;
use std::path::PathBuf;

use innovate_rust::{KernelBinding, KernelOperation, KernelRequest};

fn python_kernel_schema_version() -> String {
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("src")
        .join("innovate")
        .join("kernel.py");
    let source = fs::read_to_string(&source_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", source_path.display()));

    let major = source
        .lines()
        .find(|line| line.starts_with("KERNEL_SCHEMA_MAJOR_VERSION"))
        .and_then(|line| line.split('=').nth(1))
        .and_then(|value| value.trim().parse::<u32>().ok())
        .expect("expected KERNEL_SCHEMA_MAJOR_VERSION in Python kernel");

    let minor = source
        .lines()
        .find(|line| line.starts_with("KERNEL_SCHEMA_MINOR_VERSION"))
        .and_then(|line| line.split('=').nth(1))
        .and_then(|value| value.trim().parse::<u32>().ok())
        .expect("expected KERNEL_SCHEMA_MINOR_VERSION in Python kernel");

    format!("{major}.{minor}")
}

#[test]
fn schema_compatibility_guard_matches_the_python_kernel_contract() {
    let binding = KernelBinding::new();
    let python_schema_version = python_kernel_schema_version();

    assert_eq!(binding.schema_version(), python_schema_version);
    assert_eq!(
        KernelRequest::discover_models().schema_version,
        python_schema_version
    );

    let want_operations = [
        KernelOperation::DiscoverModels,
        KernelOperation::FitModel,
        KernelOperation::PredictModel,
        KernelOperation::SimulateModel,
        KernelOperation::SummarizeModel,
        KernelOperation::DiagnoseModel,
    ];
    assert_eq!(binding.available_operations(), want_operations);
    assert!(binding.bridge_script_exists());
    assert_eq!(
        want_operations
            .iter()
            .map(|operation| operation.as_str())
            .collect::<Vec<_>>(),
        vec![
            "discover_models",
            "fit_model",
            "predict_model",
            "simulate_model",
            "summarize_model",
            "diagnose_model",
        ]
    );
}
