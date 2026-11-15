# Specification: Rust HMC/NUTS Posterior Sampling Migration (`nuts-rs`)

## Objective
Extend `rust/crates/innovate-core/src/bayesian.rs` from bounded 1D/2D posterior quadrature (`grid`, capped at 10,000 cells) to general multi-parameter Hamiltonian Monte Carlo / No-U-Turn Sampling using `nuts-rs` from `crates.io`.

## Scope
- Implement `nuts_rs::CpuLogpFunc` for diffusion model log-posterior (`log_likelihood` + `GaussianPriors` / `statrs` priors) with unconstrained-to-constrained parameter transforms (e.g., log/logit transforms for positive/bounded rates `p`, `q`, `m`) and finite-difference or analytic gradients.
- Support deterministic seeded multi-chain execution via `nuts-rs` (`DiagAdaptExpSettings` / `Settings`), returning draws, divergence counts, energy/BFMI, and existing Vehtari split-R-hat (`ranked_rhat`) and bulk-ESS (`paired_ess`) diagnostics.
- Retain `bayesian::grid` as a backward-compatible deterministic quadrature mode for 1D/2D requests while routing `sampler: "nuts"` (or >2 parameters) through `nuts-rs`.

## Non-Goals
- Breaking existing 1D/2D quadrature JSON fixtures or removing split-R-hat / ESS diagnostic contracts.

## Phase 1 Design Record (`nuts-rs = "=0.19.1"`)
- **Verified Crate:** `nuts-rs = "=0.19.1"` (`cargo check -p innovate-core` and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-core` pass on Rust 1.99.1).
- **Unconstrained Bijection & Jacobian:** For each sampled parameter $\theta_j \in (l_j, u_j)$ with width $w_j = u_j - l_j$, map $u_j \in \mathbb{R}$ via numerically stable logistic sigmoid $p_j = \sigma(u_j) \in (0, 1)$, $\theta_j(u_j) = l_j + w_j p_j$, with inverse $u_j = \ln(p_j / (1 - p_j))$ (clamping initial boundary values strictly inside $(0, 1)$ by $10^{-12}$). The log-Jacobian determinant is $\ln |d\theta_j / du_j| = \ln w_j + \ln p_j + \ln(1 - p_j)$.
- **Gradient Evaluation:** Central finite differences in unconstrained coordinates ($h = 10^{-6}$) over `log_likelihood + priors.log_density + log_jacobian`, returning a recoverable `LogpError` if any evaluation is non-finite.
