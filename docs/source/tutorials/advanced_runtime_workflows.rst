Advanced runtime workflows
==========================

The advanced runtime layer provides opt-in workflows for ensemble forecasting,
policy scenario comparison, streaming updates, calibrated intervals, and
accelerator-aware execution. These surfaces are additive and return
JSON-friendly ``AdvancedResult`` payloads.

Stable surfaces
---------------

``policy_scenario`` and ``uncertainty_calibration`` are stable result contracts.
They include explicit ``schema_version``, ``capability``, ``metadata``, and
``diagnostics`` fields.

Experimental surfaces
---------------------

``regime_ensemble`` and ``streaming_update`` are experimental. They are safe to
use in examples and validation lanes, but should not be treated as permanent
statistical APIs until more external evidence exists.

Runnable example
----------------

The end-to-end example lives at ``examples/advanced_runtime_workflows.py`` and
is validated by the unit test suite. It builds:

* a weighted regime ensemble;
* a policy intervention scenario;
* a streaming update payload;
* calibrated prediction intervals with holdout coverage.

Accelerator evidence
--------------------

Machine-readable smoke evidence is stored in
``docs/source/_static/advanced_runtime/performance_evidence.json``. The current
evidence records dependency-free NumPy execution and safe fallback behavior for
optional JAX and Rust-native routing.
