# innovate.rs Rust Bindings

This crate provides a thin Rust-facing binding surface for the shared Innovate kernel.
The user-facing language suffix is `innovate.rs`; the crates.io package name is
`innovate-rs` because Cargo crate names do not use dots.
Model discovery metadata is available through a Rust-native path and is parity
tested against the Python bridge. Simple logistic and Bass `predict_model`
requests with fitted state payloads now run through Rust-native paths with
Python bridge fallback for unsupported execution shapes. `simulate_model`
follows the same native fitted-state slices and bridge fallback. `fit_model`
now has a native logistic slice as well, with Python bridge fallback for
unsupported model families. `summarize_model` and `diagnose_model` also have
native logistic slices for fitted-state payloads with the same bridge fallback
for other models.

## Layout

- `src/lib.rs`: core Rust API and kernel contract helpers
- `benches/native_kernel.rs`: Criterion benchmarks for the native logistic
  kernel paths
- `examples/profile_memory_native_kernels.rs`: DHAT memory profiling driver for
  the native logistic kernel paths
- `inst/discovery_manifest.json`: embedded native discovery metadata, parity
  tested against the Python kernel
- `inst/python/kernel_bridge.py`: kernel bridge entrypoint used by the bindings
- `scripts/profile_native_kernels.sh`: repeatable profiling entrypoint for the
  native Rust benchmarks
- `scripts/profile_memory_native_kernels.sh`: repeatable memory profiling
  entrypoint for the native Rust benchmarks
- `tests/`: contract, scaffold, and architecture checks

## Development

- `cargo test`
- `cargo fmt --check`
- `cargo clippy --all-targets --all-features`
- `cargo bench --bench native_kernel`
- `cargo check --example profile_memory_native_kernels`

## Benchmarking and profiling

- `bindings/rust/benches/native_kernel.rs` measures the native logistic
  `fit_model`, `predict_model`, `simulate_model`, `summarize_model`, and
  `diagnose_model` paths with Criterion.
- `bindings/rust/scripts/profile_native_kernels.sh` profiles the same native
  benchmark group with `cargo flamegraph` and writes
  `flamegraph-native-kernels.svg`.
- `bindings/rust/scripts/profile_memory_native_kernels.sh` profiles the same
  native execution paths with DHAT and writes `dhat-heap.json`. Set
  `INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS` to increase or reduce the loop
  count.
- The benchmark and profiling surface intentionally stays on the Rust-native
  execution path; the Python bridge remains the fallback implementation for
  unsupported shapes.
- Fallback paths and bridge failures emit structured `tracing` events for
  debugging and regression triage.
- GPU profiling is not yet a Rust crate responsibility because this crate does
  not own a Rust-native GPU execution backend. GPU and XLA device profiling
  belongs with the optional JAX/XLA backend until a Rust GPU backend is promoted
  behind the kernel contract.

## Compatibility checks

- `tests/schema_compatibility.rs` guards the shared schema version and stable
  operation list against drift.
- `tests/native_discovery.rs` verifies that Rust-native discovery metadata
  matches the Python bridge response.
- `tests/operations.rs` verifies Rust-native logistic and Bass prediction
  against the Python bridge contract, verifies the same pattern for simulation,
  fitting, summary, and diagnostics where implemented, and confirms fallback
  for non-native or unsupported shapes.
- `tests/end_to_end.rs` exercises the live Python bridge against a stable
  kernel model.
- `tests/architecture.rs` verifies the package scaffold and bridge entrypoint.

## Runtime expectations

- The crate uses Rust-native metadata for `discover_models`.
- Simple fitted-state logistic and Bass `predict_model` requests use
  Rust-native execution.
- Simple fitted-state logistic and Bass `simulate_model` requests use the same
  native execution paths.
- Simple fitted-state logistic `fit_model` requests use the same native
  execution path.
- Simple fitted-state logistic `summarize_model` and `diagnose_model` requests
  use the same native execution path.
- Unsupported prediction shapes and other execution operations remain thin
  bridges over the shared Python kernel.
- The core is not yet entirely Rust: Rust-native execution covers the documented
  slices above, while unsupported model families and payload shapes still use the
  Python bridge fallback.
- Rust-native operations do not require Python.
- Bridge fallback operations require the Python `innovate` package to be
  available to the configured Python command.
- `uv run python` is the default bridge launcher; override it with
  `INNOVATE_PYTHON_COMMAND`, for example
  `INNOVATE_PYTHON_COMMAND="uv run --with innovate==0.5.0 python"`.

## Support boundaries

- The package only wraps the stable kernel contract.
- Rust-native execution is intentionally narrow and must remain parity tested
  against the Python reference semantics.
- It is intended for publication as the `innovate-rs` crate once the
  maintainer-owned crates.io token is configured.
