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

- [ ] R08 — qualify common dependency closure; do not overwrite alignment owners' pins without coordination.
- [ ] R01 — extract a binding-independent native workspace and first model slice.
- [ ] R02 — typed observation and parameter contracts.
- [ ] R03 — native Polars ingestion and explicit column bindings.
- [ ] R04 — Arrow/joint-draw schema and ownership.
- [ ] R05 — direct Python native dispatch and installed-wheel test.
- [ ] R06 — cease supported pandas/xarray and host-engine paths.
- [ ] R07 — CLI and Python-absent consumer.
- [ ] R09 — direct FFI and each existing binding's accepted scope.
- [ ] R10 — executable capability matrix.

Do not archive until relevant I/Q gates and current installed-consumer evidence pass. Source extraction alone is not full Rust ownership.
