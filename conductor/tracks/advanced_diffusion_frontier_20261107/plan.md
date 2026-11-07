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

Read [taskbook](../../innovation-taskbook-20261107.md). G02's real static-consumer
leaf creates `innovate-games` using qualified Paignia artifacts without Kairos.
A08 uses that boundary; G03 later extends it for optional dynamic execution.
Static G04 cases require A08/G02/V02, while cases declaring dynamic execution
also require G03. Preserve one crate/solver owner and all 52 top-level packets.

- [ ] A01 — classical native fit/predict.
- [ ] A02 — observation likelihoods.
- [ ] A03 — hierarchical partial pooling, including the qualified native sampler adapter required by that method.
- [ ] A04 — multi-generation and competing-product dynamics.
- [ ] A05 — repeat use, switching and abandonment.
- [ ] A06 — regimes/shocks/market potential.
- [ ] A07 — networks/complements/path dependence.
- [ ] G01 — provider ownership inventory and consolidation.
- [ ] G02 — independently consumable verified game solver slices.
- [ ] G03 — optional dynamic Kairos adapter extending the existing game boundary.
- [ ] A08 — structural demand and innovation-specific strategic games.
- [ ] G04 — strategic innovation validation/research cases, with static and dynamic results scoped independently.

Coordinate provider work through explicit handoffs. Do not close from an issue/plan alone.

## 8 November 2026 reconciliation check

Merged main `301abcb2a9c5b3e3e48200965ddcd77f71bbbd17` into the existing programme
candidate while preserving the I07/I10 plan and specification byte-for-byte.
All 52 packet IDs remain unchanged; the static G04/A08 dependency closure
excludes G03/E01. Ten existing Conductor workflow/registry tests,
`python scripts/conductor_registry_hygiene.py --json`, inline dependency checks
and `git diff --check` passed with Python 3.12.14 compatibility tooling.
These are planning checks; native, scientific, installed-consumer and release
acceptance remain pending under their existing owners.
