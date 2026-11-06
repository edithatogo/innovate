# Innovate JULIA native binding

This 0.7 development package invokes the Rust C ABI, without Python. Build the
core with `cargo build --locked -p innovate-ffi` from the repository root and set
`INNOVATE_NATIVE_LIBRARY` to the absolute shared-library path under `target/debug`.
Only bounded native capabilities are currently exposed. Advanced features and
installed native distribution bundles remain open in the root migration ledger.
The JSON request schema is 2.0. Inputs are borrowed for the call; the wrapper
copies response text and releases the Rust-owned result exactly once.
