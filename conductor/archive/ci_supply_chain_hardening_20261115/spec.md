# Specification: CI & Release Supply-Chain Hardening

## Problem Statement
The repository already runs `CodeQL`, `Gitleaks`, `zizmor`, `cargo-audit`, and `scorecard.yml`, but lacks:
1. Rust license/duplicate/source policy enforcement (`cargo-deny` via `deny.toml`).
2. Runtime network-egress auditing on publish/release workflows (`step-security/harden-runner`).
3. Verifiable `Sigstore` / `in-toto` / `SLSA` provenance attestations and SBOM vulnerability scans (`Syft` + `Grype`) on released FFI/binary artifacts.
4. Complementary workflow supply-chain linting (`poutine`).

## Out of Scope (Explicitly Excluded)
- Redundant or low-ROI tools for this repo: `cargo-vet`, `Kani`, `TruffleHog`, `Semgrep`, `Cimon`, `Gato`/`Raven`/`SmokedMeat`, `Checkov`/`OPA`/`Conftest`, and `Trivy` (using `Syft` + `Grype` instead).

## Acceptance Criteria
- `deny.toml` passes `cargo deny check` in `.github/workflows/security-quality.yml`.
- Publish workflows (`pypi-publish.yml`, `crates-publish.yml`, `bindings-publish.yml`, `conda-publish.yml`) run `step-security/harden-runner` (SHA-pinned).
- Release artifacts in `bindings-publish.yml` emit `actions/attest-build-provenance` (Sigstore/in-toto/SLSA) and `Syft`/`Grype` SBOM/scan outputs.
- `uvx zizmor .github/workflows/*.yml`, `python3 scripts/check_repository_policy.py`, and workflow tests pass.
