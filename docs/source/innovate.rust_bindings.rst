Rust Bindings
=============

The Rust bindings expose the stable `innovate` kernel through a thin adapter
layer. Discovery metadata is available through a Rust-native path and is parity
tested against the Python bridge. The first model-execution slice is also
Rust-native for simple logistic payloads; unsupported operations and payload
shapes continue to use the shared Python kernel bridge.

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
- `fit_model` uses Rust-native logistic fitting only for simple positive
  observed values without covariates, events, or custom fitter options.
- `predict_model` and `simulate_model` use Rust-native logistic execution only
  for simple fitted states without covariates or event splits.
- `summarize_model` and `diagnose_model` use Rust-native logistic reporting
  only for simple fitted states with the required explicit inputs.
- The crate emits structured `tracing` events when native paths fall back or
  the Python bridge fails.
- Non-native model families, unsupported payload shapes, probabilistic
  runtimes, richer diagnostics, and model-specific Python internals remain
  Python-backed.
- The repository `src/` tree must be available at runtime.
- `uv run python` is the default bridge launcher.
- `INNOVATE_PYTHON_COMMAND` can override the Python launcher when needed.

Fallback and errors
-------------------

Native entrypoints return `unsupported_native_operation` when the request is
outside the documented Rust slice. Public wrapper methods treat that as a
recoverable condition, emit a `tracing` event, and retry the original request
through the Python bridge. Bridge launch, status, or decoding failures are
reported as `bridge_command_failed` so callers can distinguish unsupported
native coverage from transport failure.

Support boundaries
------------------

- The package only wraps the stable kernel contract.
- It does not create a second public modelling API around Rust internals.
- Rust-native discovery and logistic execution must remain parity tested
  against the Python capability registry and Python reference semantics.
- Future FFI or SDK hardening should extend the same contract boundary rather
  than replacing it with Rust-native model logic.
