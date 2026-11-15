# Specification: Rust `kairo-ecs-*` `crates.io` Publishing & Pinning

## Problem Statement
`rust/crates/innovate-core/Cargo.toml` currently pins `kairo-ecs-core`, `kairo-ecs-rng`, `kairo-ecs-types`, `kairo-ecs-abm`, and `kairo-ecs-des` to `git = "https://github.com/edithatogo/kairos.git", rev = "e76b97be2ea3712645933d903822ff6381c8883e"`. `crates.io` forbids publishing crates with git dependencies, blocking `innovate-core` from `crates-publish.yml`.

## Goals
1. Verify/coordinate published `crates.io` releases for `kairo-ecs-{types,rng,core,abm,des}` matching or superseding `rev = "e76b97be2ea3712645933d903822ff6381c8883e"`.
2. Replace the 5 git dependency entries in `rust/crates/innovate-core/Cargo.toml` with `crates.io` version pins.
3. Verify `cargo package --locked -p innovate-core` (or `cargo publish --dry-run`) succeeds once `rust_tabular_crates_io_20261115` is also complete.

## Acceptance Criteria
- Zero `git =` dependencies remain in `rust/crates/innovate-core/Cargo.toml`.
- `cargo test -p innovate-core --locked`, `cargo audit`, and `cargo publish --dry-run -p innovate-core --locked` succeed.
