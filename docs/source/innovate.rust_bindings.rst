Rust Bindings
=============

The Rust bindings expose the stable `innovate` kernel through a thin adapter
layer. Discovery metadata is available through a Rust-native path and is parity
tested against the Python bridge. Model execution still shells out to the shared
Python kernel bridge instead of reimplementing diffusion semantics in Rust.

Package layout
--------------

- `bindings/rust/src/lib.rs`: core Rust API and kernel contract helpers
- `bindings/rust/inst/discovery_manifest.json`: embedded native discovery
  metadata, parity tested against the Python kernel
- `bindings/rust/inst/python/kernel_bridge.py`: bridge entrypoint used by the
  Rust crate
- `bindings/rust/tests/`: architecture, compatibility, operation, and
  end-to-end checks

Installation and development
----------------------------

From the repository root:

.. code-block:: bash

   cd bindings/rust
   cargo test
   cargo fmt --check
   cargo clippy --all-targets --all-features

The Rust crate is intended for direct repository use. It is not published as a
standalone registry package.

Compatibility and drift checks
------------------------------

The Rust layer keeps its schema version aligned with the Python kernel
contract. The automated test suite checks that:

- the Rust schema version matches the Python kernel schema version
- the stable operation list matches the exported wrapper surface
- the Rust-native discovery response matches the live Python bridge response
- the discovery response remains decodable from the live bridge
- the end-to-end example still runs in automated test contexts

Runtime expectations
--------------------

- The crate remains thin and contract-driven.
- `discover_models` uses Rust-native metadata.
- `fit_model`, `predict_model`, `simulate_model`, `summarize_model`, and
  `diagnose_model` remain Python-backed.
- The repository `src/` tree must be available at runtime.
- `uv run python` is the default bridge launcher.
- `INNOVATE_PYTHON_COMMAND` can override the Python launcher when needed.

Support boundaries
------------------

- The package only wraps the stable kernel contract.
- It does not reimplement diffusion semantics in Rust.
- Rust-native discovery is metadata-only and must remain parity tested against
  the Python capability registry.
- Future FFI or SDK hardening should extend the same contract boundary rather
  than replacing it with Rust-native model logic.
