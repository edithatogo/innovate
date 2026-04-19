# Innovate Rust Bindings

This crate provides a thin Rust-facing binding surface for the shared Innovate kernel.

## Layout

- `src/lib.rs`: core Rust API and kernel contract helpers
- `inst/python/kernel_bridge.py`: kernel bridge entrypoint used by the bindings
- `tests/`: contract, scaffold, and architecture checks

## Development

- `cargo test`
- `cargo fmt --check`
- `cargo clippy --all-targets --all-features`

## Compatibility checks

- `tests/schema_compatibility.rs` guards the shared schema version and stable
  operation list against drift.
- `tests/end_to_end.rs` exercises the live Python bridge against a stable
  kernel model.
- `tests/architecture.rs` verifies the package scaffold and bridge entrypoint.

## Runtime expectations

- The crate remains a thin bridge over the shared Python kernel.
- The repository `src/` tree must be available at runtime.
- `uv run python` is the default bridge launcher; override it with
  `INNOVATE_PYTHON_COMMAND` when needed.

## Support boundaries

- The package only wraps the stable kernel contract.
- It does not reimplement diffusion semantics in Rust.
- It is intended for local development and direct repository use, not registry
  publication.
