# Innovation Models and Strategic Competition Implementation Plan

> For agentic workers: execute the named packets individually; exact interfaces/tests are in the taskbook.

**Goal:** Verified innovation models spanning diffusion, utilisation, substitution, competition and strategic behaviour.

**Architecture:** Native dynamics/observation/inference modules with optional externally owned game/runtime adapters.

**Tech Stack:** Qualified Rust numerical stack, Polars/Arrow, Kairos and the extracted generic game capability.

**Spec:** [spec.md](spec.md).

## Global Constraints

Do not implement generic game/VOI/simulation infrastructure inside Innovate. No vague all-models-at-once implementation task. Each scientific slice retains independent validation and unsupported cases.

## Review Focus

Unidentified early saturation; cumulative-count likelihood misuse; unbalanced switching/stock flows; endogenous price/attention; equilibrium selection and duplicate solver ownership.

## Executable packets

Read [taskbook](../../innovation-taskbook-20261107.md). A08 requires the `innovate-games` adapter crate introduced by G03 in addition to the listed G02 solver evidence; create no second crate/solver.

- [x] A01 — classical native fit/predict (`models/{mod,bass,logistic,gompertz,fisher_pry}.rs`, `tests/classical_fit.rs`).
- [x] A02 — observation likelihoods (`likelihood.rs`, `tests/likelihood_reference.rs`).
- [x] A03 — hierarchical partial pooling (`models/hierarchical.rs`, `tests/hierarchical_recovery.rs`).
- [x] A04 — multi-generation and competing-product dynamics (`models/{norton_bass,competition}.rs`, `tests/competition_conservation.rs`).
- [x] A05 — repeat use, switching and abandonment (`models/repeat_use.rs`, `tests/repeat_use_balance.rs`).
- [x] A06 — regimes/shocks/market potential (`models/regimes.rs`, `tests/regime_boundaries.rs`).
- [~] A07 — networks/complements/path dependence (pure-core `models/network.rs` implemented and tested; `innovate-kairos` runtime adapter pending `E01`).
- [ ] G01 — provider ownership inventory and consolidation (external `edithatogo/paignia`).
- [ ] G02 — independently consumable verified game solver slices (external `edithatogo/paignia`).
- [ ] G03 — optional dynamic Kairos adapter and Innovate game boundary (pending `G02` & `E01`).
- [~] A08 — structural demand and innovation-specific strategic games (pure-core `models/demand.rs` `choice_shares` implemented and tested; `innovate-games` adapter pending `G02`).
- [ ] G04 — strategic innovation validation/research cases (pending `A08`, `G02`, `V02`).

Coordinate provider work through explicit handoffs. Do not close from an issue/plan alone.
