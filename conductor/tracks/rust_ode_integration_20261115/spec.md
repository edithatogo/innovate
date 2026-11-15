# Specification: Rust Adaptive ODE/IVP Integration Modernization

## Objective
Replace the hand-rolled fixed-step `numerics::rk4` integrator in `rust/crates/innovate-core/src/numerics.rs` (consumed by `competition.rs`, `substitution.rs`, `mechanistic.rs`, and `composite.rs`) with a published `crates.io` adaptive ODE/IVP solver (`ode_solvers` or `differential-equations`).

## Scope
- Evaluate `ode_solvers` (Dopri5 / RK4 / Dop853) vs `differential-equations` (adaptive RK45 / BDF) for irregular time grids, dense output, and stiff coupled systems.
- Preserve deterministic evaluation and exact numerical parity on existing fixture suites (`competition.rs`, `scientific_contract.rs`, `tests/native`).
- Ensure zero `git` dependencies and clean `docs.rs` / `cargo audit` compliance.

## Non-Goals
- Altering discrete-event / agent-based stepping owned by `kairo-ecs-des` and `kairo-ecs-abm`.
