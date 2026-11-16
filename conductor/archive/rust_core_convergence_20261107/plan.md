# Rust-Core Convergence Implementation Plan

> For agentic workers: execute named packets using superpowers:executing-plans or superpowers:subagent-driven-development. Do not invent fallback engines.

**Goal:** Native-only innovation computation and supported Polars/Arrow interfaces without pandas/xarray.

**Architecture:** Binding-independent Rust crates, optional adapters and direct native interfaces.

**Tech Stack:** Qualified Rust/Polars/Arrow, PyO3/FFI and Kairos.

**Spec:** [spec.md](spec.md).

## Global Constraints

Follow [execution contract](../../execution-contract-20261107.md), [candidate qualification](../../dependency-candidates-20261107.md) and the mandatory support retirement. Every new operation needs an independent oracle and installed execution.

## Review Focus

Python subprocess fallback; pandas/xarray transitive extras; malformed/lifetime-unsafe Arrow buffers; unqualified Git/root patches; model names overstating native payload coverage.

## Executable packets

The exact paths, interfaces, tests/assertions and commands are in the [taskbook](../../innovation-taskbook-20261107.md). Record receipts here only after implementation; no competing abbreviated task definition.

- [x] R08 — qualify common dependency closure (`conductor/qualification/dependency-candidates.json`, `conductor/qualification/dependency-receipts/rust-closure-20261116.json`).
- [x] R01 — extract a binding-independent native workspace and first model slice (`bass_cumulative`, `tests/bass_reference.rs`).
- [x] R02 — typed observation and parameter contracts (`rust/crates/innovate-core/src/domain.rs`, `tests/domain_contract.rs`).
- [x] R03 — native Polars ingestion and explicit column bindings (`bind_panel` in `innovate-tabular`, `tests/panel_contract.rs`).
- [x] R04 — Arrow/joint-draw schema and ownership (`specs/joint-draw-v1.json`, `encode_joint_draws`/`decode_joint_draws`, `tests/joint_draw_roundtrip.rs`).
- [x] R05 — direct Python native dispatch (`tests/native/test_convergence_dispatch.py`).
- [x] R06 — cease supported pandas/xarray and host-engine paths (`pyproject.toml`, `bindings/python/innovate/__init__.py`, `tests/native/test_convergence_dispatch.py`).
- [x] R07 — CLI and Python-absent consumer (`rust/crates/innovate-core/src/bin/innovate-native.rs`, `tests/native/test_convergence_dispatch.py`).
- [x] R09 — direct FFI and each existing binding's accepted scope (`rust/crates/innovate-ffi/tests/ffi_lifecycle.rs`).
- [x] R10 — executable capability matrix (`conductor/capabilities.json`, `scripts/validate_capability_claims.py`, `tests/unit/test_capability_claims.py`).

Do not archive until relevant I/Q gates and current installed-consumer evidence pass. Source extraction alone is not full Rust ownership.
