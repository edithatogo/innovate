# Innovate R native binding

This 0.7 development package invokes the Rust C ABI, without Python. Build the
core with `cargo build --locked -p innovate-ffi` from the repository root and set
`INNOVATE_NATIVE_LIBRARY` to the absolute shared-library path under `target/debug`.
Only bounded native capabilities are currently exposed. Advanced features and
installed native distribution bundles remain open in the root migration ledger.
The JSON request schema is 2.0. Inputs are borrowed for the call; the wrapper
copies response text and releases the Rust-owned result exactly once.

### Local macOS binary package

Build the locked Rust FFI library and copy `bindings/r` to a staging
directory. Place `libinnovate_ffi.dylib` in the staged package's
`inst/lib` directory. From a separate output directory, run
`R CMD INSTALL --build` on the staged package. Install the resulting
binary archive into a fresh R library and unset `INNOVATE_NATIVE_LIBRARY`
to exercise installed-library discovery. The binary is specific to its
platform and architecture.
