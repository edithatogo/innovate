Rust core roadmap
=================

Purpose
-------

The Rust core roadmap turns the contract-first architecture into an execution
plan. Python remains the primary ergonomic and reference surface, while Rust is
the strategic long-term runtime for robust, efficient, portable kernel
execution.

The roadmap is governed by ADR 0004 and follows four rules:

* Python reference semantics remain authoritative until a Rust path passes
  parity tests.
* Rust-backed execution must use the same kernel schema compatibility rules as
  every binding.
* Bindings stay thin and must not duplicate model logic.
* Rust components are promoted only behind benchmark gates and compatibility
  checks.

Candidate operations
--------------------

The first Rust-backed candidates are stable, schema-driven operations whose
behavior is already explicit in the functional kernel contract:

* ``discover_models``: low-risk metadata discovery driven by the capability
  registry and schema version. This now has a Rust-native path in the Rust
  binding with parity tests against the Python bridge.
* ``predict_model``: deterministic execution against fitted state payloads once
  model-state schemas are stable. The first implemented slice is Rust-native
  logistic prediction for simple fitted states, with Python bridge fallback for
  unsupported shapes such as covariates, event splits, and non-native model
  families.
* ``simulate_model``: deterministic or seeded simulation paths where payload
  shapes, dtypes, and error mapping can be verified without Python object
  identity. The same logistic-native slice now covers simulation for simple
  fitted states.
* ``fit_model``: bounded fitting workflows where the parameter search is
  deterministic enough to reproduce with the same response contract. The first
  implemented slice is Rust-native logistic fitting for simple fitted states,
  with Python bridge fallback for unsupported families and payload shapes.
* ``summarize_model`` and ``diagnose_model``: fitted-state reporting paths that
  can reuse native parameters, residuals, and diagnostics contract fields. The
  first implemented slice is Rust-native logistic summary and diagnostics for
  simple fitted states, with Python bridge fallback for unsupported families
  and payload shapes.

Operations that require broad Python-backed fitting behavior, optional
probabilistic runtimes, or model-specific class internals should remain
Python-backed initially. In particular, broader model families should remain
Python-backed unless their state, diagnostics, and uncertainty payloads can
pass parity checks without relying on hidden Python objects.

Operation support inventory
---------------------------

The current Rust core status is operation-level, not model-family-wide. Native
Rust support exists only where the request payload is stable, the response shape
is covered by parity tests, and unsupported cases can return to the Python
bridge without changing the public kernel contract.

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
     - Native logistic fitting for simple positive observations without
       covariates, events, or custom fitter options.
     - Unsupported model families, covariates, event splits, and custom fitter
       options fall back to the Python bridge.
     - Broader fitters, optional probabilistic runtimes, uncertainty-aware
       fitting, and model-specific class internals remain Python-backed.
     - Medium. Batched or differentiable fitting can be JAX/XLA-eligible, but
       the current scalar logistic slice favors Rust for packaging, predictable
       CPU latency, and no accelerator dependency.
   * - ``predict_model``
     - Native logistic prediction for simple fitted states with explicit
       parameters and time arrays.
     - Unsupported families, covariate payloads, event splits, and incomplete
       fitted states fall back to the Python bridge.
     - Model-specific prediction semantics that depend on Python objects remain
       Python-backed.
     - High. Large batched prediction can be JAX/XLA-eligible; default
       promotion must compare XLA compile cost, steady-state runtime, and Rust
       native CPU latency.
   * - ``simulate_model``
     - Native logistic simulation for the same simple fitted-state payload used
       by prediction.
     - Unsupported families, stochastic policies that are not represented in
       the stable payload, covariates, and event splits fall back to the bridge.
     - Probabilistic simulation, DES-style event queues, and model-specific
       stochastic internals remain Python-backed until their schemas stabilize.
     - High for bounded array simulation; low for dynamic DES/event-queue
       semantics. JAX/XLA is suitable only when randomness and shapes can be
       expressed through explicit PRNG keys and bounded arrays.
   * - ``summarize_model``
     - Native logistic summary for simple fitted states and deterministic
       summary fields.
     - Unsupported families, custom diagnostics, covariates, and event splits
       fall back to the bridge.
     - Rich model cards, uncertainty reports, and backend-specific summaries
       remain Python-backed.
     - Medium. Array-heavy summaries may be JAX/XLA-eligible, while small
       schema assembly is better kept Rust-native or Python-backed.
   * - ``diagnose_model``
     - Native logistic diagnostics for simple fitted states with explicit
       observed and time arrays.
     - Unsupported families, missing diagnostic inputs, covariates, and event
       splits fall back to the bridge when the wrapper path is used.
     - Rich residual diagnostics, calibration workflows, posterior diagnostics,
       and optional backend diagnostics remain Python-backed.
     - Medium. Vectorized diagnostic metrics can be JAX/XLA-eligible; promotion
       requires parity, benchmark evidence, and a clear deployment rationale.

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
slices where both are technically eligible. The promotion decision for each
operation must record:

* the NumPy/SciPy or Python reference result and tolerance policy;
* whether a JAX/XLA implementation is eligible, rejected, or complementary;
* XLA compile cost, steady-state runtime, accelerator target, and dependency
  cost when XLA is eligible;
* Rust-native runtime, packaging impact, memory behavior where measurable, and
  bridge fallback rate;
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
* Benchmark gates show a material performance, packaging, or robustness gain
  without changing public semantics.
* XLA eligibility checks document whether JAX/XLA is unsuitable,
  complementary, or a stronger default candidate than Rust-native execution.
* Binding smoke tests prove R, Julia, TypeScript, Go, Rust, and future C#
  surfaces can call the promoted operation through the same contract.

Benchmark gates must include a benchmark promotion dossier before defaults
change. The dossier should include local Criterion output for Rust-native
paths, Python reference timings, XLA compile cost and steady-state runtime when
eligible, and a regression threshold that CI or release checks can enforce.

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

The Rust migration should be supported by a native performance toolchain so
benchmark gates and regressions can be evaluated on the Rust side instead of
only through the Python harness. The next planned step is a dedicated Rust
benchmark and profiling track that introduces a benchmark harness for native
kernel paths, records benchmark results for the promoted slices, and provides a
repeatable local profiling workflow for hot paths. The intended tooling path
is criterion-based benchmarking together with a native profiling helper such
as ``cargo-flamegraph``.
In this repository that is implemented by
``bindings/rust/benches/native_kernel.rs`` and
``bindings/rust/scripts/profile_native_kernels.sh``.

This work should stay narrower than the Python testing stack:

* use Rust-native benchmarking for the Rust core paths that matter most;
* use a repeatable profiling workflow for hotspot analysis;
* keep mutation testing as a later, lower-priority consideration rather than a
  required Rust-side mirror of the Python tooling.
