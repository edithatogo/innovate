# Semi-Markov Competing-Risks Diffusion and Cohort Kernels Implementation Plan

**Goal:** Deliver Rust-native competing-risks hazard generators, semi-Markov duration tunnel states, and Polars/Arrow trajectory interchange for multi-product switching and `tobac` dynamic cohort projections.

**Spec:** [spec.md](spec.md).

## Executable packets

- [x] **SM01** — Continuous-time competing-risks generator $Q$ and uniformization / Padé transition step $P(\Delta t) = \exp(Q\,\Delta t)$ with deterministic semigroup subdivision and row-stochastic conservation property tests (`models/semi_markov.rs`, `tests/semi_markov_kernels.rs`).
- [x] **SM02** — Semi-Markov duration-since-transition tunnel states $\tau \in \{0, \dots, K\}$ and bi-exponential frailty decay kernels reducing to the time-homogeneous Markov baseline when decay parameters vanish (`models/semi_markov.rs`, `tests/semi_markov_kernels.rs`).
- [~] **SM03** — Polars/Arrow cohort state-occupancy interchange (`cohort_occupancy_ipc`/`decode_cohort_occupancy_ipc` in `innovate-tabular`, `tests/cohort_interchange.rs` complete; downstream `edithatogo/tobac` consumer parity remains in external `tobac` repo).

## PR 656 review follow-up — 16 November 2026

Writer: chat `01a1e3f3-81ee-7bc3-896b-d7e42296c8ff`; isolated worktree
`/Users/doughnut/.codex/worktrees/innovate-pr656-review/innovate`; branch
`codex/pr-656-review-corrections`; base
`8244d1eb23fec915c0ce93f4da93e3cae44d6ddb`. Scope: review findings on
SM01 uniformization, SM03 cohort IPC nulls, deterministic-fit posterior access,
and minimum-length regime detection. Allowed source and tests: the named SM01
and SM03 files and their tests; `rust/crates/innovate-core/src/regime.rs` test
module; focused classical, hype, hierarchical, and inference-status tests or
API documentation; and removal of now-unused imports from the three deterministic
fit implementations. This plan is the lane's handoff; the repository integrator
retains ownership of the PR branch, shared records, and promotion. No native
service object or authentication scope is assigned. Next: reproduce the review
examples and edge cases, apply any bounded correction, and record results.

### Results and handoff

The original review's deterministic-fit, null-extraction and minimum-segment
changes are present at the base. This packet adds direct negative regressions
for those changes and replaces the large-rate subdivision after finding that the
base can still produce an infinite matrix for finite inputs.

- Red: `rustup run 1.99.1 cargo test --locked -j 2 -p innovate-core --test
  semi_markov_kernels -- --nocapture` at the base plus the new tests returned
  2 failures: the two-state transition at `dt=1e25` contained infinities, and
  a row with two `f64::MAX` hazards was accepted with an infinite exit rate.
  The review's `dt=1000` absorbing example already passed at the base.
- Green: the same native test target now passes 5 tests, including the original
  absorbing example, a closed-form reversible two-state chain through an
  overflowing rate-time product, a mixed fast/slow chain, zero rates, invalid
  timesteps and invalid/overflowing hazards.
- Green: `rustup run 1.99.1 cargo test --locked -j 2 -p innovate-core --lib
  regime::tests` passed 4 tests. The new check compares the objective over
  88,533 small-series, penalty and grid combinations with exhaustive dynamic
  programming, including the review's six-observation example.
- Green: `rustup run 1.99.1 cargo test --locked -j 2 -p innovate-core --test
  inference_status --test classical_fit --test hype_fit --test hype_posterior
  --test hierarchical_recovery --test regime --test regime_boundaries` passed
  16 tests. A deterministic refit rejects posterior access and clears an earlier
  posterior; Bayesian sampler coverage remains separate.
- Green: `rustup run 1.99.1 cargo test --locked -j 2 -p innovate-tabular --test
  cohort_interchange` passed 2 tests, including a null in each of nine columns.
- `cargo fmt --all -- --check` with Cargo/Rustfmt 1.100.0, `git diff --check`,
  and `.venv-native/bin/python scripts/check_repository_policy.py` passed. Local
  Python was 3.15.2. Native tests used Rust/Cargo 1.99.1, a worktree-local target,
  `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_PROFILE_DEV_DEBUG=0`, and
  `CARGO_INCREMENTAL=0`. `Cargo.lock` SHA-256:
  `2fbada0a3db416e66a1fb338ce82c4e77a53cecfed5529c20b98cc3228f00a44`.
  Source Git blobs: SM01 `212a553709cef2f0122e5f42ef280e29742d212e`;
  regime `939c30883242b7a5a7ec2b4a54ae7890d785cc2c`; cohort test
  `9ea756a465befc33e789d894abd076f8718fca63`.

Self-reviewed against analytic transition formulas and an exhaustive detector
oracle. This is focused local evidence, not full coverage, hosted CI, installed
`tobac` consumer parity, scientific sign-off or release acceptance. Next: the
existing integrator reviews and integrates the scoped packet into PR 656;
its exact-head hosted checks and external consumer qualification remain open.
