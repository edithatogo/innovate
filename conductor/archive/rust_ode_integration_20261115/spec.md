# Specification: Rust Adaptive ODE/IVP Integration Modernization

## Objective
Replace the hand-rolled fixed-step `numerics::rk4` integrator in `rust/crates/innovate-core/src/numerics.rs` (consumed by `competition.rs`, `substitution.rs`, `mechanistic.rs`, and `composite.rs`) with a published `crates.io` adaptive ODE/IVP solver (`ode_solvers` or `differential-equations`).

## Scope
- Evaluate `ode_solvers` (Dopri5 / RK4 / Dop853) vs `differential-equations` (adaptive RK45 / BDF) for irregular time grids, dense output, and stiff coupled systems.
- Preserve deterministic evaluation and exact numerical parity on existing fixture suites (`competition.rs`, `scientific_contract.rs`, `tests/native`).
- Ensure zero `git` dependencies and clean `docs.rs` / `cargo audit` compliance.

## Non-Goals
- Altering discrete-event / agent-based stepping owned by `kairo-ecs-des` and `kairo-ecs-abm`.

## Crate Selection Decision
- **Selected:** `ode_solvers = "=0.6.2"` (`Apache-2.0`, `Dopri5` Dormand-Prince 5(4) with `ContinuousOutputModel` dense output over `nalgebra::DVector<f64>`).
- **Rationale:** Reuses `nalgebra 0.34.x` / `simba 0.9.x` / `serde` already in `Cargo.lock` with zero extra dependency bloat, supports arbitrary-length runtime state vectors via `DVector<f64>`, and provides exact 4th-order dense interpolation at caller-requested irregular `times` without re-integrating per point. All existing ODE callers (`competition.rs`, `composite.rs`, `events.rs`, `mechanistic.rs`) route through `numerics::integrate`.
