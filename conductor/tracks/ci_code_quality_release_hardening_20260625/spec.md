# CI, Code Quality, and Release Hardening

## Overview

Raise the repository from feature-complete to mature release-ready by enforcing
CI, code quality, automation, observability, security, mutation testing,
coverage, release evidence freshness, and GitHub Actions monitoring.

## Functional Requirements

- Consolidate local and CI quality gates around `nox`.
- Enforce basedpyright strict, ruff, tests, coverage, docs, package dry-run,
  security audit, dependency dashboards, mutation testing, and polyglot checks.
- Add release-readiness freshness gates for all required artifacts.
- Ensure conductor-review, push, and GitHub Actions monitoring are embedded in
  every Conductor phase and track.
- Add observability evidence for runtime diagnostics and release automation.

## Non-Functional Requirements

- CI should be deterministic and non-interactive.
- Security and dependency findings must fail closed unless explicitly waived.
- Slow gates may be scheduled or opt-in, but release readiness must require
  fresh evidence.

## Acceptance Criteria

- `uv run nox -s lint types tests docs package` passes locally.
- CI runs the same required gates or documented equivalents.
- Coverage and mutation evidence are produced and checked.
- Release-readiness cannot be `release_ready` with stale or missing evidence.
- GitHub Actions pass on the pushed branch before track completion.

## Out Of Scope

- External publication using credentials.
- Major runtime feature work outside hardening gaps.
