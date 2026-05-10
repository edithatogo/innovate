Rust core roadmap
=================

Purpose
-------

The Rust core roadmap turns the contract-first architecture into an execution
plan. Python remains the primary ergonomic and reference surface. Rust is the
strategic long-term runtime for robust, efficient, portable kernel execution,
but the core is not fully Rust-owned today.

The roadmap is governed by ADR 0004 and follows four rules:

* Python reference semantics remain authoritative until a Rust path passes
  parity tests.
* Rust-backed execution must use the same kernel schema compatibility rules as
  every binding.
* Bindings stay thin and must not duplicate model logic.
* Rust components are promoted only behind benchmark gates backed by parity,
  compatibility, and profiling evidence.

Audited status
--------------

The core is not entirely Rust. This roadmap is intentionally tied to the
current source layout so that documentation drift can be checked by tests:

* ``src/innovate/kernel.py`` remains the Python reference owner for
  ``KERNEL_OPERATIONS`` and the canonical ``discover_models``, ``fit_model``,
  ``predict_model``, ``simulate_model``, ``summarize_model``, and
  ``diagnose_model`` functions.
* ``bindings/rust/src/lib.rs`` exposes Rust-native execution only for the
  documented slices: packaged discovery metadata, logistic, Fisher-Pry, and
  Gompertz ``fit_model``, logistic, Fisher-Pry, and Gompertz
  ``summarize_model`` and ``diagnose_model``, and logistic, Fisher-Pry,
  Gompertz, and Bass
  ``predict_model``/``simulate_model`` fitted-state execution. In short:
  Rust-native execution only for the documented slices.
* The Rust binding still contains the Python bridge fallback path through
  ``invoke``, ``bridge_script_absolute_path``, ``kernel_pythonpath``, and
  ``python_command_segments``. Unsupported native slices therefore remain
  bridge-backed rather than Rust-owned.
* The remaining ownership gap is tracked as the Conductor follow-on track
  ``Rust Core Migration Completion and Polyglot Claim Closure`` so the
  residual bridge-backed slices and Python-only reference areas stay explicit
  rather than becoming implicit roadmap drift.
* A full Rust core must not be claimed until every canonical operation, every
  Python registry model family, and every stable payload shape has a
  Rust-native implementation or an explicitly promoted non-Python backend.
  This means every Python registry model family must be covered before claiming
  full Rust ownership.
  Today, model families such as ``network_diffusion`` and ``policy_hazard``
  remain outside the Rust-native slice, while ``fisher_pry`` and ``gompertz``
  have moved into the Rust-native substitution/diffusion slices. Covariates,
  event splits, probabilistic runtimes, custom fitter options, and incomplete
  fitted states still require fallback or Python-only reference behavior.

Candidate operations
--------------------

The first Rust-backed candidates are stable, schema-driven operations whose
behavior is already explicit in the functional kernel contract:

* ``discover_models``: low-risk metadata discovery driven by the capability
  registry and schema version. This now has a Rust-native path in the Rust
  binding with parity tests against the Python bridge.
* ``predict_model``: deterministic execution against fitted state payloads once
  model-state schemas are stable. The first implemented slices are Rust-native
  logistic, Fisher-Pry, Gompertz, and Bass prediction for simple fitted states,
  with Python bridge fallback for unsupported shapes such as covariates, event
  splits, and non-native model families.
* ``simulate_model``: deterministic or seeded simulation paths where payload
  shapes, dtypes, and error mapping can be verified without Python object
  identity. The logistic, Fisher-Pry, Gompertz, and Bass native slices cover
  simulation for simple fitted states.
* ``fit_model``: bounded fitting workflows where the parameter search is
  deterministic enough to reproduce with the same response contract. The first
  implemented slices are Rust-native logistic, Fisher-Pry, and Gompertz fitting
  for simple fitted states, with Python bridge fallback for unsupported
  families and payload shapes.
* ``summarize_model`` and ``diagnose_model``: fitted-state reporting paths that
  can reuse native parameters, residuals, and diagnostics contract fields. The
  first implemented slices are Rust-native logistic, Fisher-Pry, and Gompertz
  summary and diagnostics for simple fitted states, with Python bridge fallback
  for unsupported families and payload shapes.

Operations that require broad Python-backed fitting behavior, optional
probabilistic runtimes, or model-specific class internals should remain
Python-backed initially. In particular, broader model families should remain
Python-backed unless their state, diagnostics, and uncertainty payloads can
pass parity checks without relying on hidden Python objects.

Operation support inventory
---------------------------

The current Rust core status is operation- and model-slice-level, not
model-family-wide. Native Rust support exists only where the request payload is
stable, the response shape is covered by parity tests, and unsupported cases can
return to the Python bridge without changing the public kernel contract.

The canonical machine-readable inventory is
:download:`rust_core_migration_inventory.json <_static/rust_core_migration_inventory.json>`.
It records each slice's ``current_owner`` as one of ``rust_native``,
``python_bridge``, or ``python_reference``; its fallback status; profiling
requirements; promotion blockers; operation-level dependencies; promotion
gates; evidence commands; and binding smoke requirements. Release and CI tooling
should consume that fixture rather than scraping this prose.

The table below is a human summary of the current default slices.

.. list-table::
   :header-rows: 1
   :widths: 16 28 25 25 18

   * - Operation
     - Native Rust scope
     - Bridge fallback scope
     - Python-only reference scope
     - Rust vs JAX/XLA eligibility
   * - ``discover_models``
     - Native metadata discovery from the packaged manifest, parity checked
       against the Python capability registry.
     - Bridge discovery remains available for parity and drift checks.
     - Python registry remains authoritative for model metadata generation.
     - Low. Discovery is metadata I/O, so XLA is not useful.
   * - ``fit_model``
     - Native logistic, Fisher-Pry, and Gompertz fitting for simple positive
       observations without covariates, events, or custom fitter options.
     - Unsupported model families, covariates, event splits, and custom fitter options fall back to the Python bridge.
     - Broader fitters, optional probabilistic runtimes, uncertainty-aware
       fitting, and model-specific class internals remain Python-backed.
     - Medium. Batched or differentiable fitting can be JAX/XLA-eligible, but
       the current scalar logistic slice favors Rust for packaging, predictable
       CPU latency, and no accelerator dependency.
   * - ``predict_model``
     - Native logistic, Fisher-Pry, Gompertz, and Bass prediction for simple
       fitted states with explicit parameters and time arrays.
     - Unsupported families, covariate payloads, event splits, and incomplete
       fitted states fall back to the Python bridge.
     - Model-specific prediction semantics that depend on Python objects remain
       Python-backed.
     - High. Large batched prediction can be JAX/XLA-eligible; default
       promotion must compare XLA compile cost, steady-state runtime, and Rust
       native CPU latency.
   * - ``simulate_model``
     - Native logistic, Fisher-Pry, Gompertz, and Bass simulation for the same
       simple fitted-state payload used by prediction.
     - Unsupported families, stochastic policies that are not represented in
       the stable payload, covariates, and event splits fall back to the bridge.
     - Probabilistic simulation, DES-style event queues, and model-specific
       stochastic internals remain Python-backed until their schemas stabilize.
     - High for bounded array simulation; low for dynamic DES/event-queue
       semantics. JAX/XLA is suitable only when randomness and shapes can be
       expressed through explicit PRNG keys and bounded arrays.
   * - ``summarize_model``
     - Native logistic, Fisher-Pry, and Gompertz summary for simple fitted
       states and deterministic summary fields.
     - Unsupported families, custom diagnostics, covariates, and event splits
       fall back to the bridge.
     - Rich model cards, uncertainty reports, and backend-specific summaries
       remain Python-backed.
     - Medium. Array-heavy summaries may be JAX/XLA-eligible, while small
       schema assembly is better kept Rust-native or Python-backed.
   * - ``diagnose_model``
     - Native logistic, Fisher-Pry, and Gompertz diagnostics for simple fitted
       states with explicit observed and time arrays.
     - Unsupported families, missing diagnostic inputs, covariates, and event
       splits fall back to the bridge when the wrapper path is used.
     - Rich residual diagnostics, calibration workflows, posterior diagnostics,
       and optional backend diagnostics remain Python-backed.
     - Medium. Vectorized diagnostic metrics can be JAX/XLA-eligible; promotion
       requires parity, benchmark evidence, and a clear deployment rationale.

Execution backlog by operation family
-------------------------------------

The machine-readable inventory is the execution backlog. It groups every slice
into phases that can be worked in parallel while preserving Python reference
semantics and bridge fallback behavior.

``phase_0_native_guardrails``
  Keep packaged discovery metadata native and guarded by schema, manifest, and
  binding smoke checks. Discovery has no CPU flamegraph metadata or DHAT memory
  profile gate because it is metadata I/O.

``phase_1_default_hardening``
  Harden the current Bass ``predict_model`` and ``simulate_model`` native
  slices. Capture parity, fallback-rate evidence, CPU benchmark output, CPU
  flamegraph metadata, and a promotion dossier before expanding those defaults.

``phase_2_logistic_expansion``
  Widen the logistic ``fit_model``, ``predict_model``, ``simulate_model``,
  ``summarize_model``, and ``diagnose_model`` slices only after unsupported
  payload shapes such as covariates, event splits, custom fitter options, and
  incomplete fitted states have explicit schema fixtures and error mappings.

``phase_3_model_family_migration``
  Migrate bridge-default model families operation by operation. Each candidate
  must declare dependencies on stable request and response schemas, parity
  fixtures, error mapping, benchmark evidence, memory evidence when relevant,
  and a binding smoke matrix before any Rust-default claim.

``phase_4_reference_boundary_review``
  Keep probabilistic runtimes, uncertainty reports, backend-specific diagnostics,
  and Python object internals Python-reference-owned until a stable schema
  boundary exists. A later dossier can promote Rust-native, XLA-backed, or
  continued Python ownership. Rule: No Rust-default claim exists without
  evidence.

Promotion dossier capture remains mandatory for any default change. The dossier
must link raw Criterion output, Python reference timings, fallback-rate evidence,
CPU flamegraph metadata, DHAT memory profile output or a not-applicable
rationale, XLA CPU/GPU evidence when eligible, and binding smoke results for R,
Julia, TypeScript, Go, Rust, C#, and Python.

Fallback and error behavior
---------------------------

Native Rust entrypoints return ``unsupported_native_operation`` when the
operation or payload is outside the documented native slice. Public wrapper
methods treat that code as recoverable and dispatch the original request to the
Python bridge, emitting structured ``tracing`` events for observability. Invalid
payloads that violate the stable request schema remain hard errors and should
not be silently rewritten into fallback requests.

Bridge execution failures return ``bridge_command_failed`` with the operation,
message, retryability, and details preserved through the kernel error mapping.
Bindings should expose those stable error codes rather than transport-specific
exceptions so R, Python, Rust, Julia, C#, TypeScript, and Go consumers see the
same failure contract.

Rust vs JAX/XLA promotion criteria
----------------------------------

Rust-native execution and JAX/XLA-backed execution should compete only for
slices where both are technically eligible. Rust is the default candidate for
portable CPU execution and packaging-sensitive paths. JAX/XLA is the default
candidate to evaluate for accelerator-oriented array kernels with static shapes,
explicit randomness, and acceptable compile cost. The promotion decision for
each operation must record:

* the NumPy/SciPy or Python reference result and tolerance policy;
* whether a JAX/XLA implementation is eligible, rejected, or complementary;
* XLA compile cost, steady-state runtime, accelerator target, and dependency
  cost when XLA is eligible;
* Rust-native CPU runtime, packaging impact, memory behavior where measurable,
  and bridge fallback rate;
* schema compatibility, error mapping, and binding smoke-test results;
* the explicit promotion decision: keep Python-backed, keep experimental,
  promote Rust-native, promote XLA-backed, or keep both behind runtime
  capability gates.

Promotion gates
---------------

Rust-backed execution can become the default for a kernel operation only after
all of the following gates pass:

* Parity tests compare Rust results against Python reference semantics for the
  same request payloads.
* Schema compatibility tests prove the Rust request and response payloads use
  the same ``KERNEL_SCHEMA_VERSION`` and operation names.
* Error mapping tests prove Rust errors round-trip through the same stable
  kernel error codes.
* Benchmark gates show a material CPU latency, packaging, portability, or
  robustness gain without changing public semantics.
* XLA eligibility checks document whether JAX/XLA is unsuitable,
  complementary, or a stronger default candidate than Rust-native execution.
* Binding smoke tests prove R, Julia, TypeScript, Go, Rust, and future C#
  surfaces can call the promoted operation through the same contract.

Benchmark gates must include a benchmark promotion dossier before defaults
change. The dossier should include local Criterion output for Rust-native CPU
paths, Python reference timings, XLA compile cost and steady-state runtime when
eligible, memory evidence for allocation-sensitive slices, fallback-rate evidence,
binding smoke evidence, and a regression threshold that CI or release checks can
enforce. Use
:download:`rust_core_promotion_dossier_example.json <_static/rust_core_promotion_dossier_example.json>`
as the machine-readable template/example for a current Rust-native slice.
Use
:download:`rust_core_promotion_dossier_bass_example.json <_static/rust_core_promotion_dossier_bass_example.json>`
for the current Bass native slice.

Binding policy
--------------

Language bindings may expose idiomatic package shapes, but they should remain
thin contract surfaces. R, Rust, Julia, TypeScript, Go, C#, and future bindings
should call or mirror the shared kernel contract rather than independently
implementing model behavior.

C# should be added as a planned binding after the schema-compatibility and
operation-dispatch rules are documented for the existing bindings. Its initial
scope should match the other thin bindings: discovery, request construction,
response conversion, error mapping, and schema drift checks.

Benchmarking and profiling tooling
----------------------------------

The Rust migration is supported by a native performance toolchain so benchmark
gates and regressions can be evaluated on the Rust side instead of only through
the Python harness. The ``criterion`` benchmarks and ``cargo-flamegraph`` cover
Rust-native CPU hot paths.
In this repository that is implemented by
``bindings/rust/benches/native_kernel.rs`` and
``bindings/rust/scripts/profile_native_kernels.sh``.
Memory profiling is implemented separately through the DHAT-backed
``bindings/rust/examples/profile_memory_native_kernels.rs`` driver and
``bindings/rust/scripts/profile_memory_native_kernels.sh`` wrapper.
GPU profiling is not currently part of the Rust crate because Rust does not yet
own a GPU execution backend in this project; GPU and XLA device profiling should
remain attached to the optional JAX/XLA backend until a Rust GPU backend is
promoted behind the kernel contract. Use the benchmark workflow's
``JAX_PLATFORM_NAME=cpu`` and ``JAX_PLATFORM_NAME=gpu`` commands for XLA CPU/GPU
evidence, and keep those artifacts separate from Rust CPU flamegraphs and DHAT
heap profiles.

The core is therefore not entirely written in Rust yet. The Rust crate owns
native metadata discovery and selected logistic and Bass execution slices.
Unsupported model families, covariate/event payloads, probabilistic runtimes,
and broader model operations still fall back to the shared Python kernel.
Promotion remains operation by operation. A slice can move from experimental to
default only when parity, schema compatibility, stable error mapping, binding
smoke tests, CPU benchmark evidence, memory evidence when relevant, and XLA
eligibility or rejection rationale are recorded.

This work should stay narrower than the Python testing stack:

* use Rust-native benchmarking for the Rust core paths that matter most;
* use a repeatable profiling workflow for hotspot analysis;
* use Rust memory profiling for allocation-sensitive native slices;
* keep GPU profiling with the active GPU/XLA backend until Rust owns a GPU
  execution path;
* keep mutation testing as a later, lower-priority consideration rather than a
  required Rust-side mirror of the Python tooling.
