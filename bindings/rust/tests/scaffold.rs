use std::path::Path;

#[test]
fn package_scaffold_includes_the_bridge_entrypoint() {
    assert!(
        Path::new("Cargo.toml").exists(),
        "expected Cargo.toml in bindings/rust"
    );
    assert!(
        Path::new("README.md").exists(),
        "expected README in bindings/rust"
    );
    assert!(
        Path::new("inst/python/kernel_bridge.py").exists(),
        "expected kernel bridge entrypoint in bindings/rust/inst/python"
    );
}
