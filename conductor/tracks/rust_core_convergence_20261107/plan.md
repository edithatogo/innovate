# Rust-Core Convergence Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Make Rust authoritative for the declared stable scientific scope.

**Architecture:** Incrementally extract the existing native code into a core independent of bindings; invert language dispatch without discarding independent references.

**Tech Stack:** Cargo workspace, qualified numerical crates, Arrow, optional Kairos, PyO3 and versioned FFI.

**Spec:** [spec.md](spec.md).

## Global constraints

Preserve supported behaviour through an explicit migration. Coordinate pins with the existing alignment owner. Native promotion requires the integrity track's relevant acceptance tests.

## Review focus

Python-free execution, hidden subprocess fallback, Arrow ownership/lifetimes, unsupported rich payloads and cross-language error semantics each require explicit tests.

## Task 1 — Ownership inventory and first extraction

Files: `bindings/rust/src/lib.rs`, `bindings/rust/Cargo.toml`, `src/innovate/kernel.py`, `docs/source/_static/rust_final_ownership_ledger.json`; proposed `Cargo.toml`, `crates/innovate-core/`, `crates/innovate-arrow/`.

- [ ] Inventory model x operation x payload x backend ownership and reconcile historical ledgers with actual dispatch.
- [ ] Write a Python-absent installed-native test and independent numerical parity tests for one bounded deterministic slice; run them before extraction.
- [ ] Extract that slice and its domain/error types into the core; rerun tests and commit without claiming other slices are native.

## Task 2 — Invert bindings and isolate simulation

Files: proposed `bindings/python/`, `bindings/ffi/`, `crates/innovate-kairos/`; existing `bindings/rust/inst/python/kernel_bridge.py`, `src/innovate/abm/kairos_adapter.py` and language contract tests.

- [ ] Write dispatch tests that detect a Python subprocess and reject unsupported native requests.
- [ ] Implement direct native dispatch for the accepted slice; feature-gate Kairos and retain explicitly named external reference paths.
- [ ] Test installed Python and native consumers, Arrow round trips, resource ownership, errors and package isolation; commit.

## Task 3 — Expand with evidence

- [ ] Qualify numerical dependencies with accuracy, memory, warm/cold runtime and maintenance/licence receipts before pinning.
- [ ] Promote remaining stable slices one at a time, including covariates, event splits and uncertainty only after their semantics are verified.
- [ ] Add root `cargo test --workspace --locked` only once the workspace exists; retain current manifest-specific tests during migration.
- [ ] Generate capability claims from executable evidence, run required repository gates and review installed consumers before archival. Full-native status remains false while any advertised stable native combination depends on Python.
