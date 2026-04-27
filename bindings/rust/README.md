# Innovate Rust Bindings

This crate provides a thin Rust-facing binding surface for the shared Innovate kernel.
Model discovery metadata is available through a Rust-native path and is parity
tested against the Python bridge. Model execution remains Python-backed.

## Layout

- `src/lib.rs`: core Rust API and kernel contract helpers
- `inst/discovery_manifest.json`: embedded native discovery metadata, parity
  tested against the Python kernel
- `inst/python/kernel_bridge.py`: kernel bridge entrypoint used by the bindings
- `tests/`: contract, scaffold, and architecture checks

## Development

- `cargo test`
- `cargo fmt --check`
- `cargo clippy --all-targets --all-features`

## Compatibility checks

- `tests/schema_compatibility.rs` guards the shared schema version and stable
  operation list against drift.
- `tests/native_discovery.rs` verifies that Rust-native discovery metadata
  matches the Python bridge response.
- `tests/end_to_end.rs` exercises the live Python bridge against a stable
  kernel model.
- `tests/architecture.rs` verifies the package scaffold and bridge entrypoint.

## Runtime expectations

- The crate uses Rust-native metadata for `discover_models`.
- Non-discovery operations remain thin bridges over the shared Python kernel.
- The repository `src/` tree must be available at runtime.
- `uv run python` is the default bridge launcher; override it with
  `INNOVATE_PYTHON_COMMAND` when needed.

## Support boundaries

- The package only wraps the stable kernel contract.
- It does not reimplement diffusion semantics in Rust.
- Rust-native discovery is metadata-only and must remain parity tested against
  the Python capability registry.
- It is intended for local development and direct repository use, not registry
  publication.
