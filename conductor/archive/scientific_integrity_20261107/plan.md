# Scientific Integrity Implementation Plan

> For agentic workers: execute the named taskbook packets with failing tests, minimum changes, review and receipts.

**Goal:** No failed/unsupported model calculation may appear as valid scientific evidence.

**Architecture:** Typed inference status and honest method dispatch across native results and bindings; legacy safety repairs precede retirement.

**Tech Stack:** Rust scientific core and isolated independent references.

**Spec:** [spec.md](spec.md), updated by the mandatory [product contract](../../product.md).

## Global Constraints

The previous Python/JAX production-backend direction is superseded. I01/I02 repair misleading current behaviour but do not authorise maintaining a parallel host engine after native migration.

## Review Focus

Failed refit retaining old posterior; synthetic fallback uncertainty; ignored confounders/method selectors; untested gradients; calibration failures hidden from denominators.

## Executable packets

Read [taskbook](../../innovation-taskbook-20261107.md).

- [x] I01 — fail-closed inference and stale-result tests (`tests/native/test_scientific_integrity.py`; legacy Python fitters already retired).
- [x] I02 — descriptive/causal method dispatch and explicit unsupported errors (`tests/native/test_scientific_integrity.py`; `rust/crates/innovate-core/src/causal.rs`).
- [x] I03 — native status/log-density/gradient contracts (`rust/crates/innovate-core/src/inference/status.rs`, `tests/inference_status.rs`, `tests/gradient_oracle.rs`).
- [x] I04 — simulation-based calibration and repeated coverage harness (`validation/inference/protocol.json`, `validation/inference/run.py`, `tests/validation/test_inference_protocol.py`).

Review status through Arrow/kernel/reports. No invalid posterior is promoted merely because a type/schema test passes.
