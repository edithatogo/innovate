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

## Issue #661 — R07 CI follow-up

Chat `01a1e3d8-0464-7e93-a2f9-1ec89a9b9aef` owns this bounded follow-up
in the managed worktree `/Users/doughnut/.codex/worktrees/innovate-issue-661/innovate`,
based on feature-branch commit `035f2c7b253a9eee58b97cbb22de5637a289ad88`.
The initial allowed files are this plan and
`tests/native/test_convergence_dispatch.py`. No shared manifest, workflow,
native service, or authentication scope is assigned to this lane. The next
operation is to reproduce and repair the Python-free CLI test in a clean
worktree, then record focused checks and hosted evidence. PR #656 retains
ownership of feature-branch integration and its other outstanding gates.

The latest CI run also stops at one Rust formatting difference in
`rust/crates/innovate-core/src/inference/mod.rs`. This file is added to the
allowed scope solely for the `cargo fmt` correction needed by the same CI
gate. The separate documentation-quality failures remain with PR #656.

Local evidence on the issue branch, 16 November 2026:

- The repository-local interpreter reports CPython 3.15.2; `rustc` and `cargo`
  report 1.99.1. With an initially absent `target/debug/innovate-native`,
  `PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN=1.99.1 .venv/bin/python -m pytest tests/native/test_convergence_dispatch.py -q`
  passed all four cases. The CLI is built with `--locked`, selected from
  Cargo's artifact messages, and executed with an empty `PATH`.
- The focused file plus
  `tests/native/test_registry_readiness.py::test_real_packet_is_checked_by_the_active_ci_suite`
  passed five cases after the formatting correction. This confirms the real
  digest packet locally; the earlier branch commit `d4721698` restored its
  `pyproject.toml` input.
- `cargo fmt --all --check` (Rust 1.99.1), Ruff check and format for the
  changed test, its basedpyright check, `git diff --check`, and
  `.venv/bin/python scripts/check_repository_policy.py` passed. The policy
  check did not inspect live settings.
- `RUSTDOCFLAGS='--cfg docsrs -D warnings' cargo doc --locked --no-deps -p innovate-core -p innovate-tabular -p innovate-ffi -j 2`
  passed under Rust 1.99.1 with existing Cargo/compiler warnings. This is the
  CI docs.rs build, not the stricter separate documentation-coverage gate.

Hosted run `43845616128` at base `035f2c7b` had the Windows executable
failure and the Linux/macOS formatting failure. The independent documentation
workflow also reported many missing public Rust item descriptions, release
readiness remained blocked by incomplete scientific migration obligations,
and offline AI security was queued. Those results are not passes or release
acceptance. Hosted verification of this follow-up remains pending.
