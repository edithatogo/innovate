# Plan: CI & Release Supply-Chain Hardening

> **Dependencies:** `deny.toml` `[sources] allow-git = []` strict mode requires `rust_tabular_crates_io_20261115` and `rust_kairos_crates_io_20261115` to complete first (allow those two URLs temporarily in `deny.toml` if implementing earlier).

## Phase 1: `cargo-deny` & `poutine` in Security Quality Workflow
- [ ] **1.1 Create `deny.toml` at repo root:** Configure `[advisories]`, `[licenses]` (SPDX allowlist), `[bans]` (duplicate/wildcards), and `[sources]` (`unknown-registry = "deny"`, `unknown-git = "deny"`). Run `cargo deny check` locally (or via `taiki-e/install-action` in CI) and verify baseline pass.
- [ ] **1.2 Wire `cargo-deny` into `.github/workflows/security-quality.yml`:** Add a `cargo-deny` step alongside `cargo-audit` using SHA-pinned `EmbarkStudios/cargo-deny-action` or `taiki-e/install-action`.
- [ ] **1.3 Add `poutine` workflow supply-chain check:** Add a SHA-pinned `poutine` scan step in `.github/workflows/security-quality.yml` alongside `zizmor`.
- [ ] **1.4 Verify workflow lint & policy tests:** Run `uvx zizmor .github/workflows/security-quality.yml`, `python3 scripts/check_repository_policy.py`, and `pytest -o minversion=9.0 tests/workflows/test_repository_policy.py`.
- [ ] **1.5 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-verify, and commit checkpoint.

## Phase 2: `Harden-Runner` on Publish Workflows
- [ ] **2.1 Audit outbound endpoints for publish workflows:** Inspect `.github/workflows/pypi-publish.yml`, `crates-publish.yml`, `bindings-publish.yml`, and `conda-publish.yml` and list required allowed endpoints (GitHub, PyPI, crates.io, conda/anaconda, rustup).
- [ ] **2.2 Add SHA-pinned `step-security/harden-runner`:** Add `harden-runner` (`egress-policy: audit` initially, then `block` with `allowed-endpoints` once verified) as the first step in each publish job.
- [ ] **2.3 Verify workflow lint & release tests:** Run `uvx zizmor .github/workflows/*.yml` and `pytest -o minversion=9.0 tests/workflows/test_release_workflows.py`.
- [ ] **2.4 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-verify, and commit checkpoint.

## Phase 3: Sigstore / in-toto / SLSA Provenance & Syft / Grype SBOM
- [ ] **3.1 Add `actions/attest-build-provenance` (Sigstore + in-toto + SLSA):** Wire SHA-pinned `actions/attest-build-provenance` in `.github/workflows/bindings-publish.yml` (with `attestations: write` and `id-token: write` job permissions) for built FFI shared-library archives, and document `gh attestation verify` / `slsa-verifier` verification in release docs.
- [ ] **3.2 Add `Syft` SBOM generation + `Grype` scan:** Add SHA-pinned `anchore/sbom-action` (`Syft`) and `anchore/scan-action` (`Grype`) in `.github/workflows/bindings-publish.yml` / release packaging, uploading SPDX/CycloneDX SBOMs as release artifacts.
- [ ] **3.3 Run full workflow & policy verification:** Execute `uvx zizmor .github/workflows/*.yml`, `python3 scripts/check_repository_policy.py`, `pytest -o minversion=9.0 tests/workflows/test_release_workflows.py`, and `pytest -o minversion=9.0 tests/workflows/test_repository_policy.py`.
- [ ] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-verify, and mark track complete in `conductor/tracks.md`.
