Diagnostics and Uncertainty Artifacts
=====================================

Innovate exposes diagnostics through two compatible surfaces:

* ``DiagnosticsContract`` keeps the existing Python-facing metrics, residual
  analysis, warnings, and uncertainty summary.
* ``DiagnosticsArtifactPayload`` adds a versioned, binding-friendly artifact
  envelope under ``diagnostics["artifacts"]``.

The artifact envelope is intended for Python, R, Julia, TypeScript, Go, Rust,
and C# consumers that need stable JSON or Arrow-compatible tabular payloads
without depending on private Python objects.

Artifact Contract
-----------------

The first schema version is ``1.0``. It includes:

* ``schema_version``: diagnostics artifact schema version.
* ``model_name``: model class name reported by the diagnostics contract.
* ``support_level``: ``supported``, ``partial``, or ``unsupported``.
* ``provenance``: ``deterministic``, ``bootstrap``, ``bayesian``, or
  ``unknown``.
* ``backend``: the backend used to assemble the artifact payload.
* ``xla``: eligibility metadata and rationale.
* ``promotion_criteria``: requirements before new diagnostics move into the
  stable artifact contract.
* ``artifacts``: named residual, calibration, uncertainty, and comparison
  payloads.

Implemented Artifacts
---------------------

``residuals``
  Stable residual diagnostics with ``index``, ``residual``, and
  ``standardized_residual`` columns plus summary statistics such as
  Durbin-Watson and maximum absolute residual.

``calibration``
  Initial calibration slice based on residual bias and residual magnitude.
  This is marked ``partial`` because richer calibration curves are still
  planned.

``uncertainty``
  Interval-shaped uncertainty rows with ``parameter``, ``lower``, ``median``,
  and ``upper`` columns. Deterministic point estimates can validly have no
  interval rows.

``model_comparison``
  Fit metrics represented as ``metric`` and ``value`` rows for binding and
  Arrow consumers.

Arrow Compatibility
-------------------

Tabular artifacts can be converted to existing ``KernelTablePayload`` objects
with ``DiagnosticsArtifactPayload.to_table_payloads()`` and then passed through
the Arrow interchange helpers.

The representative fixture
``tests/fixtures/diagnostics_artifact_payload.json`` is used to verify that
thin bindings preserve the nested artifact payload without interpreting it as a
private Python object.

XLA Eligibility
---------------

The first implemented artifact slice is assembled from deterministic NumPy and
SciPy diagnostics, so it is not marked XLA-backed. Array-heavy residual,
interval, and simulation summaries are eligible for future JAX/XLA promotion
once they have parity tests, deterministic PRNG handling, benchmark evidence,
and stable schema fixtures.

Promotion Criteria
------------------

New diagnostics should not appear stable until they have:

* a schema-compatible payload,
* deterministic or tolerance-bounded tests,
* representative binding fixture coverage,
* documented support tier and model-family scope,
* XLA benchmark evidence or an explicit rejection rationale when acceleration
  is relevant.
