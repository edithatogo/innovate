# Product Guidelines

## Current architecture and scope

The user-authorized 0.7 development line uses a Rust scientific core, Polars tables
and Python, Julia and R bindings only. See [product context](product.md),
[technology decision](tech-stack.md) and the
[native migration guide](../docs/astro-site/src/content/docs/native-migration.md).
Python uses the explicitly selected repo-local 3.15.2 interpreter. Requalify the
newest patch before promotion; do not replace an environment under an active job.
No pandas/xarray runtime or implicit Python numerical fallback is permitted.
Project-owned code uses Apache-2.0; preserve upstream notices and inherited terms.

All current and planned scientific features remain required by
[migration-ledger.json](../migration-ledger.json). Bounded native capabilities,
local checks and archived completion records do not prove full scientific,
platform, distribution or release acceptance.

## Prose and documentation

- Write clear, concrete, active prose and explain scientific conventions/limits.
- Python public APIs use fully documented Parameters, Returns and Examples
  sections. NumPy-style docstring formatting does not require NumPy execution.
- Document Rust public APIs and the Julia/R interfaces against the same native
  contracts, including shape, units, parameter domains and error behavior.
- Leave a blank source line before Markdown headings and follow renderer syntax.
- Preserve historical evidence, citations, external licences and user edits.

## API design

- Use consistent native fit, predict, simulate, summarize and diagnose operations;
  public bindings remain thin adapters with explicit options and typed surfaces.
- Keep simple defaults usable and expose advanced options explicitly. Distinguish
  historical semantics from redesigned defaults, with named variants and examples.
- The approved prerelease migration permits breaking APIs. Stable public breaking
  releases require semantic versioning and a migration guide; no public release is
  authorized by local development qualification.
- Never infer stochastic simulation, calibrated uncertainty, causal validity,
  identifiability or global optimality from a deterministic point path or local fit.

## Visualization

Preserve publication-ready, labelled, accessible and customizable plotting
requirements. Optional consumer adapters, including legacy Matplotlib-Axes use
cases, remain migration obligations; plotting must not become a Python numerical
execution backend. No optional adapter is qualified merely by this guideline.

## Testing and scientific evidence

- Rust tests live with crates and under `rust/crates/*/tests`. Installed Python
  consumer/policy tests live in `tests/native`; Julia/R native conformance lives
  under their binding directories. Retired Python test tiers remain in Git history.
- Use failing behavioral tests before implementation, independent analytic or
  numerical references where meaningful, and both valid and invalid input cases.
- Run focused affected selectors during edits and reuse unchanged receipts. Keep
  pytest cache enabled and use failure-only selectors only for actual cached failures.
- Preserve full integration, installed-consumer, clean-package, coverage, security
  and release gates at coherent milestones. Partial checks are not full acceptance.
- Maintain >80% overall coverage; critical scientific modules target >90%. State
  whether evidence measures lines, branches or mutations and name its source scope.
- Use Hypothesis for relevant binding/contract properties and native property
  checks for numerical invariants. Rust mutation qualification, the >70% score
  target and weekly automation remain required; do not substitute Python-only
  mutation coverage for Rust computation or claim scheduled execution without proof.

## Performance

Rust owns numerical loops, solver work and parallel execution. Polars owns tabular
operations; measure allocations, conversions, binding overhead and IPC copies.
Profile the native hot paths before optimization using qualified native tools.
Accelerator/HPC capability remains required, with explicit backend qualification;
NumPy/JAX fallbacks do not satisfy Rust ownership. Preserve reproducible benchmark
and profiling requirements without treating retired Python profilers as current
native evidence.

## Errors and dependencies

Validate shapes, domains and options early and return structured native errors.
Unsupported features must fail explicitly. Keep warnings and scientific limits in
responses and binding exceptions; never silently ignore an active option.

Cargo/uv locks, pinned native Git revisions and the documentation lock govern their
respective graphs. Use uv for Python environment/dependency changes, Cargo for Rust
and pnpm for the documentation site. Qualify frontier candidates in isolated
owned environments, record retained-version exceptions and keep audit warnings
visible. Dependency updates do not imply promotion, merge or public publication.

## Style and typing

Use Ruff and Cargo fmt/Clippy as configured; ordinary Git hooks stay enabled.
The [Python style guide](code_styleguides/python.md) and `pyproject.toml` define
binding formatting. Do not introduce Black, isort, Pylint or parallel formatters.

Python typing uses 3.15 semantics and the selected 3.15.2 interpreter. Strict typing
of the public binding surface remains the target. The current bounded migration
uses the configured `basedpyright` standard mode and a recorded native-source
resolution warning; a passing current check is not strict-mode qualification.
Restore and qualify strict coverage before claiming that obligation complete.
Rust type safety and Julia/R contract checks remain separate requirements.

## Versioning, CI and review

Use conventional commits and semantic versions; retain the versioned migration
contract for approved prerelease breaking changes. Release automation and
changelogs do not authorize a release while required gates remain blocked.

Required workflows are defined under `.github/workflows`; do not claim a single
pipeline or an active schedule unless the current definitions prove it. Use
Python 3.15.2 in current Python jobs. After an authorized push, monitor exact-head
CI and repair failures without weakening gates. Codecov/coverage reporting must
identify the measured scope. Native cross-platform and scientific qualification
remain separate from local macOS conformance.

Run `conductor-review` automatically at phase checkpoints, retain findings/fixes
and source-bound evidence, and preserve genuine unresolved integration/release
approvals. No routine manual review is needed for already authorized local work.

## Citation

Preserve `CITATION.cff` and cite the exact library version, authors and DOI when
available. Retain model/reference provenance and do not conflate code conformance
with scientific validation or operational acceptance.
