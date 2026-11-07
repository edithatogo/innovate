# Ecosystem Decision Integration Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Support uncertainty-preserving adoption–capacity and implementation/research decisions.

**Architecture:** Compose existing library responsibilities through typed artefacts; application models own valuation and clinical assumptions.

**Tech Stack:** Rust Innovate, pinned Kairos, Arrow, qualified Voiage consumers and optional MARS.

**Spec:** [spec.md](spec.md).

## Global constraints

Reuse existing accepted fixtures and ownership policy. No cross-repository writes or upstream-pin changes without coordination. Native claims require actual installed native execution.

## Review focus

Broken joint-draw dependence, duplicate simulation clocks, mixed units/time bases, unidentifiable demand/capacity and surrogate-induced decision reversals each need explicit tests.

## Task 1 — Qualify shared artefacts

Files: `src/innovate/arrow_interchange.py`, `src/innovate/scenario/schemas.py`, `src/innovate/abm/kairos_adapter.py`; proposed `tests/integration/test_adoption_decision_interchange.py`.

- [ ] Inspect and version current repository-standards, Kairos and Voiage contracts; reconcile existing consumer plans before adding a second adapter.
- [ ] Write failing round-trip tests for correlated draws, unit/time mismatches, invalid inference status and unsupported schema versions.
- [ ] Implement the minimal native handoff; qualify installed consumers and record runtime ownership; commit.

## Task 2 — Adoption–capacity demonstration

- [ ] Write analytic/resource-conservation and deterministic-replay tests; require one Kairos clock/state/RNG authority.
- [ ] Build an application-level known-truth fixture with limited delivery capacity and identifiable versus non-identifiable demand cases.
- [ ] Compare component and coupled alternatives on prediction, calibration, net benefit and regret; preserve explicit economic assumptions.

## Task 3 — Learning decisions and surrogate gates

- [ ] Define feasible evidence-collection designs, simulate their observations and verify Voiage results against a small independent decision oracle.
- [ ] Compare full-simulator and optional surrogate decisions on held-out parameter regimes; reject extrapolation failures or material decision reversals.
- [ ] Record exact versions, seeds, resource costs and external handoff requirements; review and run installed-consumer/repository gates before archival.
