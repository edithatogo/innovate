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
