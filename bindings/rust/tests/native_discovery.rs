use innovate_rust::KernelBinding;
use std::fs;
use std::path::PathBuf;

#[test]
fn native_discovery_manifest_is_packaged_and_decodable() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inst")
        .join("discovery_manifest.json");
    let manifest = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", manifest_path.display()));
    let decoded: innovate_rust::KernelDiscoveryResponse =
        serde_json::from_str(&manifest).expect("manifest should decode");

    let native = KernelBinding::new().discover_models_native();
    assert_eq!(decoded, native);
}

#[test]
fn native_discovery_matches_python_bridge_metadata() {
    let binding = KernelBinding::new();

    let native = binding.discover_models_native();
    let bridged = binding
        .discover_models_via_bridge()
        .expect("Python bridge discovery should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.models.len(), bridged.models.len());

    for native_record in &native.models {
        let bridged_record = bridged
            .models
            .iter()
            .find(|record| record.key == native_record.key)
            .unwrap_or_else(|| panic!("missing bridged record for {}", native_record.key));

        assert_eq!(native_record.family, bridged_record.family);
        assert_eq!(native_record.import_path, bridged_record.import_path);
        assert_eq!(native_record.stability, bridged_record.stability);
        assert_eq!(
            native_record.supports_covariates,
            bridged_record.supports_covariates
        );
        assert_eq!(
            native_record.supports_multivariate_output,
            bridged_record.supports_multivariate_output
        );
        assert_eq!(
            native_record.supported_backends,
            bridged_record.supported_backends
        );
        assert_eq!(
            native_record.optional_dependencies,
            bridged_record.optional_dependencies
        );
        assert_eq!(
            native_record.supports_simulation,
            bridged_record.supports_simulation
        );
        assert_eq!(
            native_record.supports_summarize,
            bridged_record.supports_summarize
        );
    }
}

#[test]
fn native_discovery_reports_structured_decode_errors() {
    let response = innovate_rust::KernelResponse {
        schema_version: "1.0".to_string(),
        operation: innovate_rust::KernelOperation::DiscoverModels,
        model_key: None,
        result: Some(serde_json::json!({
            "schema_version": "1.0",
            "models": "not-an-array",
            "metadata": {}
        })),
        error: None,
        metadata: serde_json::json!({}),
    };

    let error = innovate_rust::KernelDiscoveryResponse::from_response(response)
        .expect_err("invalid discovery payload should fail deterministically");

    assert_eq!(error.code, "bridge_command_failed");
    assert!(error
        .message
        .contains("failed to decode discovery response"));
}

#[test]
fn native_discovery_reports_missing_results_as_bridge_failures() {
    let response = innovate_rust::KernelResponse {
        schema_version: "1.0".to_string(),
        operation: innovate_rust::KernelOperation::DiscoverModels,
        model_key: None,
        result: None,
        error: None,
        metadata: serde_json::json!({}),
    };

    let error = innovate_rust::KernelDiscoveryResponse::from_response(response)
        .expect_err("missing discovery result should be a structured bridge failure");

    assert_eq!(error.code, "bridge_command_failed");
    assert!(error.message.contains("did not return a result"));
}
