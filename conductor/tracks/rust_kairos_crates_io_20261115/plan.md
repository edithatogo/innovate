# Plan: Rust `kairo-ecs-*` `crates.io` Publishing & Pinning

> **Execution Order:** #3 (Parallelizable with #1–#2; depends externally on `edithatogo/kairos` publishing `kairo-ecs-*` to `crates.io`).

## Phase 1: Upstream `crates.io` Readiness Check
- [ ] **1.1 Check `crates.io` availability:** Run `cargo search kairo-ecs` (or query `crates.io` API) for `kairo-ecs-types`, `kairo-ecs-rng`, `kairo-ecs-core`, `kairo-ecs-abm`, and `kairo-ecs-des` and record available versions in `spec.md`.
- [ ] **1.2 Verify upstream parity (if not yet published, stop and report external blocker):** Confirm the published `kairo-ecs-*` versions include all symbols used by `innovate-core` at `rev = "e76b97be2ea3712645933d903822ff6381c8883e"`.
- [ ] **1.3 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 findings and confirm readiness before editing `Cargo.toml`.

## Phase 2: Switch `innovate-core` to `crates.io` Semver Pins
- [ ] **2.1 Update `rust/crates/innovate-core/Cargo.toml`:** Replace the 5 `git = "https://github.com/edithatogo/kairos.git", rev = "..."` entries with exact/semver `crates.io` version specs.
- [ ] **2.2 Refresh lockfile and compile:** Run `cargo update -p kairo-ecs-core -p kairo-ecs-rng -p kairo-ecs-types -p kairo-ecs-abm -p kairo-ecs-des` and `cargo test -p innovate-core --locked`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Publish Dry-Run & Workspace Gates
- [ ] **3.1 Verify `crates.io` packaging:** Run `cargo publish --dry-run -p innovate-core --locked` (or `cargo package --locked -p innovate-core`) and confirm zero git-dependency errors.
- [ ] **3.2 Run full workspace & docs checks:** Execute `cargo test --workspace --locked`, `cargo audit`, and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-core`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
