use std::path::{Path, PathBuf};

pub const KERNEL_SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOperation {
    DiscoverModels,
    FitModel,
    PredictModel,
    SimulateModel,
    SummarizeModel,
    DiagnoseModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelBindingError {
    pub code: &'static str,
    pub message: &'static str,
}

impl KernelBindingError {
    pub fn unimplemented(message: &'static str) -> Self {
        Self {
            code: "unimplemented",
            message,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct KernelBinding;

impl KernelBinding {
    pub fn new() -> Self {
        Self
    }

    pub fn schema_version(&self) -> &'static str {
        KERNEL_SCHEMA_VERSION
    }

    pub fn bridge_script_path(&self) -> PathBuf {
        PathBuf::from("inst")
            .join("python")
            .join("kernel_bridge.py")
    }

    pub fn bridge_script_exists(&self) -> bool {
        Path::new("inst")
            .join("python")
            .join("kernel_bridge.py")
            .exists()
    }

    pub fn available_operations(&self) -> [KernelOperation; 6] {
        [
            KernelOperation::DiscoverModels,
            KernelOperation::FitModel,
            KernelOperation::PredictModel,
            KernelOperation::SimulateModel,
            KernelOperation::SummarizeModel,
            KernelOperation::DiagnoseModel,
        ]
    }

    pub fn discover_models(&self) -> Result<Vec<String>, KernelBindingError> {
        Err(KernelBindingError::unimplemented(
            "Rust kernel discovery is not implemented yet",
        ))
    }
}
