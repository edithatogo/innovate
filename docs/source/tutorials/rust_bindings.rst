Rust Bindings
=============

The Rust bindings provide a thin adapter over the shared `innovate` kernel.
They expose a Rust-facing request and response surface for stable kernel
operations while keeping Python reference semantics authoritative. Native Rust
execution is currently limited to discovery metadata and simple logistic
``fit_model``, ``predict_model``, ``simulate_model``, ``summarize_model``, and
``diagnose_model`` slices; unsupported payloads fall back to the Python bridge.

Installation
------------

From the repository root, run the Rust package tests from the module
directory:

.. code-block:: bash

   cd bindings/rust
   cargo test

The binding uses ``uv run python`` by default. If your environment requires a
different launcher, set the ``INNOVATE_PYTHON_COMMAND`` environment variable
before calling into the crate.

Basic usage
-----------

.. code-block:: rust

   use innovate_rust::{json, KernelBinding};

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       let binding = KernelBinding::new();
       let discovery = binding.discover_models()?;
       let logistic = discovery
           .models
           .iter()
           .find(|record| record.key == "logistic")
           .expect("logistic must remain discoverable");

       let fit = binding.fit_model(
           logistic.key.clone(),
           json!({
               "inputs": {
                   "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                   "observed": [0.05, 0.12, 0.3, 0.6, 0.85]
               }
           }),
       )?;

       let diagnostics = fit
           .diagnostics_summary()
           .expect("fit response should expose diagnostics");

       println!("{}", binding.schema_version());
       println!("{}", diagnostics.support_level);
       Ok(())
   }

Compatibility and drift checks
------------------------------

The Rust package keeps its schema version aligned with the Python kernel
contract. The automated test suite checks that the shared schema version and
stable operation list do not drift, and it exercises the live end-to-end
example during package tests.

Support boundaries
------------------

- The Rust layer remains thin and contract-driven.
- Only the stable kernel operations are wrapped.
- Native execution is operation-slice-specific rather than model-family-wide.
- `unsupported_native_operation` triggers wrapper-level Python bridge fallback;
  `bridge_command_failed` reports transport or bridge execution failure.
- The package is intended for local development and direct repository use, not
  registry publication.
- Future transport or FFI work should extend the same contract boundary rather
  than replacing it with Rust-native model logic.
