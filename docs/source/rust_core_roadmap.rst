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

Operations that require broad Python-backed fitting behavior, optional
probabilistic runtimes, or model-specific class internals should remain
Python-backed initially. In particular, ``fit_model``, ``summarize_model``, and
``diagnose_model`` should move later unless their state, diagnostics, and
uncertainty payloads can pass parity checks without relying on hidden Python
objects.

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
* Binding smoke tests prove R, Julia, TypeScript, Go, Rust, and future C#
  surfaces can call the promoted operation through the same contract.

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
