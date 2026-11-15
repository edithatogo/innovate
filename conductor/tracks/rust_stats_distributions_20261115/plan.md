# Plan: Rust Statistical Distributions and Quantiles Modernization

- [ ] Audit all custom quantile, CDF, and log-PDF implementations across `bayesian.rs`, `causal.rs`, and `bootstrap.rs`.
- [ ] Add `statrs` (published `crates.io` release) to `rust/crates/innovate-core/Cargo.toml` and verify `Cargo.lock` and `cargo audit`.
- [ ] Replace `bayesian::normal_quantile` and custom distribution/likelihood boilerplate with `statrs::distribution` while preserving split-R-hat and bulk-ESS contracts.
- [ ] Verify full Rust (`cargo test --locked`), Python (`pytest tests/native`), and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc` before review and archive.
