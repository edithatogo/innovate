# Plan: Rust Adaptive ODE/IVP Integration Modernization

- [ ] Evaluate `ode_solvers` vs `differential-equations` on irregular-grid and stiff coupled diffusion systems (`competition.rs`, `substitution.rs`, `mechanistic.rs`).
- [ ] Add characterization and metamorphic tests comparing fixed-step `rk4` vs adaptive Dormand–Prince / BDF tolerances across extreme rate scales.
- [ ] Replace `numerics::rk4` callers with the selected adaptive ODE solver while preserving deterministic dense output at requested time points.
- [ ] Verify full Rust (`cargo test --locked`), Python (`pytest tests/native`), `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc`, and `cargo audit` checks before review and archive.
