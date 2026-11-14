# Working on Innovate

## Canonical context

Start with [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md) and the [Conductor index](conductor/index.md).
Resolve the owning track through [conductor/tracks.md](conductor/tracks.md),
then read its index, specification, plan and recorded acceptance evidence.
Follow the [workflow](conductor/workflow.md) and
[bounded execution contract](conductor/execution-contract-20261107.md).
The [migration ledger](migration-ledger.json) and
[release gates](conductor/release-gates-20261107.md) distinguish implementation,
scientific qualification and release acceptance.

## Scope and ownership

- Inspect the current branch, commit, worktree status and related open pull
  requests before writing. Select one bounded issue or plan entry and record
  its base commit and allowed files. Reconcile existing work before duplicating it.
- Preserve unrelated changes, historical receipts and other writers' environments.
  Use an isolated branch/worktree for independent work. Do not reset, clean,
  rewrite shared history or discard unpublished work as part of a routine fix.
- One integrator owns shared manifests, dependency pins, schemas, root workflows,
  registries and promotion. Coordinate changes to those files with active owners;
  module work does not confer ownership of shared interfaces.
- Update the owning Conductor plan with actual commands, revisions, results and
  blockers. Do not advance a planning-only track or a dependent task whose
  upstream acceptance remains open.

## Architecture and evidence

Rust owns scientific computation and validation. Python, Julia and R are thin
bindings; missing native operations return explicit errors. Polars is the public
table interface, with versioned Arrow/Parquet interchange. Do not add a Python,
NumPy/JAX, pandas or xarray production fallback. Separate reference implementations
may serve as test oracles without becoming production dependencies.

Kairos owns simulation scheduling, state and replay; Innovate owns innovation
model semantics. Voiage owns value-of-information and economic analysis. Respect
the provider and consumer boundaries in the owning track instead of copying a
second implementation into Innovate.

Test changed behavior with independent expected results and negative cases.
Keep required scientific capabilities, tolerances and quality thresholds intact.
Skipped, cancelled, queued or unexecuted checks are not passes. A source check,
local numerical result or generic code review does not establish scientific
validity, complete platform coverage or release readiness.

## Validation

Use the interpreter and toolchain specified in [CONTRIBUTING.md](CONTRIBUTING.md).
From the repository root, the dependency-free repository-policy check is:

```sh
python scripts/check_repository_policy.py
```

This checks repository policy and drift within its reported scope. It does not
replace native tests, installed-consumer checks, security scans, measured coverage
or live GitHub-setting verification. The full development command and separate
migration/release checks are documented in CONTRIBUTING.md. Run affected checks
during edits and the applicable complete gates at coherent source milestones.

## Solo-maintainer operation

Keep zero mandatory human approvals, code-owner approvals and team gates.
CODEOWNERS may provide routing information; it must not become a required approval
gate. Required automated checks must pass on the commit being promoted. Inventory
bot behavior before selecting stable required checks, and retain owner recovery
without using it to bypass failed checks during routine merges.

Preserve existing hooks and security gates. A conditional offline-security lane
remains unqualified until its dedicated runner and required local assets are
verified; do not provision substitutes, expose private material or waive that
boundary to make a check green. Keep credentials and private inputs out of Git,
logs and public issue reports. Publication, deployment and release acceptance
remain separate from completing a source change.
