# Release Maturity and Hardening

## Overview

This track turns the current release-candidate posture into a mature release posture. It hardens CI, supply-chain provenance, security gates, reproducibility, release dry-runs, compatibility checks, and artifact validation so a public release is defensible beyond basic feature completion.

## Functional Requirements

1. Add a single release-readiness gate that aggregates Python, Rust, docs, bindings, packaging, security, and provenance checks.
2. Enforce coverage, mutation-test sampling, type checks, linting, docs build, Rust tests, binding smoke tests, and package dry-runs.
3. Add supply-chain hardening: SBOM, SLSA-style provenance where available, dependency audit, license inventory, and artifact checksum publication.
4. Add reproducibility gates for benchmark fixtures, seeded simulations, and generated docs artifacts.
5. Add compatibility tests for supported Python versions, Rust MSRV, Node/pnpm docs builds, and binding package manifests.
6. Add release-candidate evidence artifacts consumed by docs and Conductor.

## Non-Functional Requirements

1. CI must remain non-interactive and deterministic.
2. Expensive checks must be staged into appropriate fast, release, nightly, or manual workflows.
3. Secrets must never be required for local verification or pull-request checks.
4. Release gates must fail closed when evidence is missing.

## Acceptance Criteria

1. A maintainer can run one documented local command to produce a release-readiness report.
2. GitHub Actions exposes an equivalent release-readiness workflow.
3. Security, provenance, packaging, docs, and language-binding checks are represented in machine-readable artifacts.
4. Release docs clearly distinguish release candidate, public release, and external acceptance states.
5. Final CI pass proves the mature release gate is operational.

## Required Operational Cadence

Every task requires a task implementation commit, a separate plan-status commit, phase review with `conductor-review`, push plus GitHub Actions monitoring, final track review, final push, and passing GitHub Actions before archive.

## Out of Scope

1. Publishing a release without maintainer approval.
2. Introducing paid security or signing services without a documented maintainer decision.
3. Weakening existing release gates to improve speed.
