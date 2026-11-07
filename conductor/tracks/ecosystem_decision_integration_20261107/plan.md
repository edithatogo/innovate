# Ecosystem Integration Implementation Plan

> For agentic workers: execute named taskbook packets; provider work stays in its owning repository.

**Goal:** Native simulation integration and uncertainty-preserving innovation/decision workflows without duplicate libraries or cycles.

**Architecture:** Innovate -> optional Kairos/game providers; optional Voiage adapter consumes Innovate; application artefacts carry feedback.

**Tech Stack:** Rust, Polars/Arrow, accepted Kairos/Voiage/game provider pins and optional MARS.

**Spec:** [spec.md](spec.md).

## Global Constraints

One clock/state/RNG; no hidden host engine; no library depends on an application; VOP is a POC whose reusable decision capabilities belong in Voiage. Retain source licences and historical evidence.

## Review Focus

Draw independence introduced by export; wrong perspective/units; cyclic imports; duplicate generic algorithms; surrogate-induced decision reversal.

## Executable packets

Read [taskbook](../../innovation-taskbook-20261107.md).

- [ ] E01 — actual native Kairos execution.
- [ ] E02 — explicit decision binding and joint-draw Voiage handoff.
- [ ] E03 — study/policy feedback through artefacts, not mutual core imports.
- [ ] E04 — VOP capability audit, Voiage promotion and POC consumer migration.
- [ ] E05 — application-promotion ledger and optional surrogate acceptance.

Coordinate G01–G04 with Kairos track110 S01–S04 and the existing medicines-game incubator. Provider PRs must pass before the installed-consumer task closes.
