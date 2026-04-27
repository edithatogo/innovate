# Specification: Quality Gates, CI, and Release Hardening

## Overview

Harden the repository so local quality commands, CI workflows, docs builds, and release workflows are trustworthy and fail hard on regressions. This track treats maturity gaps as product issues, not just tooling issues.

## Functional Requirements

1. Fix local quality-gate configuration so linting, formatting, typing, testing, packaging, and docs builds can be run deterministically.
2. Remove masked failures from CI workflows and make required jobs fail hard on real errors.
3. Ensure base test collection succeeds in the default environment.
4. Align release and publish workflows with the active branching and packaging strategy.
5. Add documentation or governance for quality gates, compatibility, and release expectations.

## Non-Functional Requirements

1. CI should remain reasonably fast while becoming stricter.
2. Workflow permissions must follow least-privilege defaults.
3. The repo should be buildable and testable on a clean machine using documented commands.
4. Changes must improve trustworthiness without bundling unrelated feature work.

## Acceptance Criteria

1. `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and the project's documented type-check commands run as intended.
2. CI no longer uses `|| true` to suppress failures on required jobs.
3. Docs build and package build are part of the verified quality story.
4. Release workflow assumptions are consistent with the repo's active branch and publishing policy.

## Out of Scope

1. Adding new scientific models.
2. Rewriting the compute core.
3. Creating non-Python bindings.
