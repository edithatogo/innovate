innovate.benchmarks.mars_surrogate module
=========================================

The MARS surrogate benchmark gate records whether the sibling ``mars`` package
should become an optional backend for adoption-curve surrogate workflows. The
current decision is **defer**: ``mars`` remains outside base and optional
package metadata until benchmark evidence justifies promotion.

Fast CI behavior is ``metadata_validation_only``. The fast gate validates the
candidate scenarios, reference baseline, XLA-backed alternative, tolerances,
promotion thresholds, failure modes, and decision outcome without importing or
running ``mars``.

Candidate comparisons
---------------------

Each candidate must compare:

- ``numpy_scipy`` reference behavior for correctness and baseline runtime
- ``mars`` surrogate behavior for surrogate-specific speed and dependency cost
- ``jax_xla_surrogate_candidate`` for eligible XLA-backed alternatives

Benchmark reports must attribute any observed gain to the surrogate, XLA, their
interaction, neither, or unknown. XLA compile cost must stay separate from XLA
steady-state runtime.

Promotion thresholds
--------------------

The gate records these initial thresholds:

- ``max_rmse_ratio`` no greater than ``1.05`` against the reference path
- ``min_surrogate_speedup`` at least ``1.5`` before promotion is considered
- ``max_xla_compile_to_steady_state_ratio`` no greater than ``5.0`` when XLA is
  eligible

Opt-in artifact dry run
-----------------------

.. code-block:: bash

   uv run python -m innovate.benchmarks.mars_surrogate --write-json benchmark-results/mars-surrogate-gate.json

This command writes metadata and validation status only. It does not import
``mars`` and does not execute timing benchmarks.

Module contents
---------------

.. automodule:: innovate.benchmarks.mars_surrogate
   :members:
   :show-inheritance:
   :undoc-members:
