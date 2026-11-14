# Voiage alignment: dependency and automation reconciliation

Date: 14 November 2026. Owning issue: [Innovate #559](https://github.com/edithatogo/innovate/issues/559).
Source baseline: [`276c2e505a0ce26910820fe8a94a6a498f505700`](https://github.com/edithatogo/innovate/tree/276c2e505a0ce26910820fe8a94a6a498f505700).

This is the existing follow-up track's source audit. It distinguishes declared
dependencies and configuration from executed installed-consumer, security and
signing acceptance. It does not qualify a release, replace the scientific
ecosystem track, or modify the shared toolchain pins owned by the #556 lane.

## Current source contract

| Surface | Observed source | Disposition |
| --- | --- | --- |
| Python tabular runtime | `polars==2.1.0` is the only base dependency | Meets the issue's Polars minimum. |
| Arrow interoperability | Optional `arrow` extra pins `pyarrow==26.0.1` | Meets the issue's PyArrow minimum when that extra is installed; preserve the optional boundary. |
| Application Python | `.python-version` is `3.15.2`; package requires `>=3.15.2,<3.16` | Later approved native migration supersedes the issue's 3.15.1 checkpoint; do not downgrade. |
| Rust | Workspace `rust-version = "1.99.1"`, edition 2024; active CI selects 1.99.1 | Later native qualification target supersedes the issue's 1.92 checkpoint. |
| Python package tooling | uv-first CI and nox commands; setup-uv action pinned to a commit | Record the actual uv executable version in each executable receipt; an action SHA alone does not establish it. |
| Lint | `ruff==0.17.4` | Existing gate retained. |
| Types | `basedpyright==1.40.2`, standard mode on the native Python binding; no `ty` dependency | Retained checker exception below; ty acceptance remains open. |
| Documentation | Astro manifest `^7.4.7`; active pnpm lock resolves 7.4.7 | Exceeds the issue's 7.3.3 floor. |
| Security tooling dependencies | Active uv lock resolves `virtualenv==21.15.5` and `filelock==4.2.9` | Inventory only; inspect actual alerts before claiming remediation. |
| Pillow and tornado | Neither package is present in the active uv lock | Not applicable to this locked Python closure; live alert inventory remains unverified. |

Source references at the inspected revision:

- [Python manifest](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/pyproject.toml), [uv lock](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/uv.lock), [.python-version](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/.python-version).
- [Rust workspace manifest](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/Cargo.toml), [CI](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/.github/workflows/ci.yml), [nox tasks](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/noxfile.py).
- [Documentation manifest](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/docs/astro-site/package.json), [pnpm lock](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/docs/astro-site/pnpm-lock.yaml). This lock contains multiple YAML documents; the application importer is in its second document.
- [Scientific ecosystem plan](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/conductor/tracks/scientific_ecosystem_alignment_20261029/plan.md), [current development workflow](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/conductor/workflow.md).

## Tabular boundary and retained exceptions

The active uv lock contains neither pandas nor xarray. The
[public Python binding](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/bindings/python/innovate/__init__.py)
accepts Polars frames and explicitly marshalled IPC bytes. Its `_ipc` boundary
rejects objects that are not Polars DataFrame or LazyFrame instances. Existing
[consumer tests](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/tests/native/test_python_consumer.py)
check that prohibited scientific Python modules have not been imported.
These source observations do not replace a clean installed-artifact execution.

There is no active runtime pandas/xarray exception. The
[tech-stack contract](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/conductor/tech-stack.md)
retains isolated external scientific test oracles and historical evidence;
those are outside supported production dispatch and cannot supply a fallback.

Retain basedpyright while equivalent ty coverage of the current Python 3.15
binding and native extension stub is unqualified. No ty incompatibility was
reproduced in this audit. The reason for retaining the existing checker is to
preserve its active gate during consumer qualification, not to claim the
issue's ty requirement has passed. Revisit with the shared toolchain owner after
running a pinned ty candidate against the same public source and deliberately
invalid type fixtures; record actual diagnostics before promoting or replacing
a checker.

The required hosted check name `unit-tests-base (3.15.1)` remains unchanged;
the current workflow explicitly executes Python 3.15.2 under that name. This
preserves branch protection while recording the real application interpreter.

The existing [privacy workflow](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/.github/workflows/privacy-secrets.yml)
records isolated Python 3.12 for Presidio because that scanner's current release
excludes Python 3.15. This is an existing scanner exception, separate from the
application interpreter. Revisit through the privacy workflow owner when the
scanner supports the selected Python release.

## Dependency alert scope

The tracked dependency locks are the active `uv.lock`, root `Cargo.lock`, docs
`pnpm-lock.yaml`, and the retained
[`kairos-candidate.Cargo.lock`](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/conductor/tracks/scientific_ecosystem_alignment_20261029/kairos-candidate.Cargo.lock).
There is no retained historical Python dependency lock in the current tracked
tree. The candidate Cargo lock is scientific qualification evidence and is not
used by the active root Cargo build. Do not edit that evidence or dismiss an
alert merely because its path is historical: first inspect the alert, dependency
path and active reachability, then record an alert-specific reason.

The [7 November pip-audit receipt](https://github.com/edithatogo/innovate/blob/276c2e505a0ce26910820fe8a94a6a498f505700/conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/native-cutover-20261107/innovate-python-audit-final-20261107.json)
reports no known vulnerabilities for the audited virtualenv and filelock
versions and skips the unpublished local Innovate package. It remains historical
evidence, not a current Dependabot alert inventory or general security clearance.

The available GitHub connector does not expose Dependabot alert listing or
dismissal operations. Its documented generic fetch allowlist does not include
the Dependabot alert endpoint family. No unsupported endpoint was called and no
alert was dismissed. Live alert count, remediation and any justified historical
alert disposition remain unverified.

A permitted read-only browser fallback reached the repository's Dependabot URL
but displayed a GitHub 404 with **Sign in / Sign up** in the header and a sign-in
return target pointing to that security page. The browser session was signed
out. This supplies no alert count or disposition evidence. No login, credential
request or security-setting change was initiated.

## Renovate signed commits

This increment adds only `"platformCommit": "enabled"` to
`.github/renovate.json`. The
[official Renovate option](https://docs.renovatebot.com/configuration-options/#platformcommit)
uses GitHub API commits and GitHub's signing support when Renovate runs as a
GitHub App. It is not a Personal Access Token signing mechanism, and a
`Signed-off-by` trailer is not a cryptographic signature.

Hosted app activity is evidenced by
[issue #475](https://github.com/edithatogo/innovate/issues/475), authored by
`renovate[bot]`. Its invalid schedule was repaired by
[PR #594](https://github.com/edithatogo/innovate/pull/594), merged as
[`2c6c1185c3ba4fafe9f8a35d2fe60c5f6cca6978`](https://github.com/edithatogo/innovate/commit/2c6c1185c3ba4fafe9f8a35d2fe60c5f6cca6978).
The corrected Monday UTC schedule, branch selection, automerge configuration
and every other existing setting are preserved.

Read-only GitHub observations for this audit:

- `GET /repos/edithatogo/innovate/rulesets?includes_parents=true` returned `[]`.
  This does not establish the absence of classic branch protection.
- `GET /repos/edithatogo/innovate/branches/main/protection` returned HTTP 403,
  `Resource not accessible by integration`. The signed-commit rule could not be
  inspected through this connection. No protection setting was changed.
- PR search scoped to this repository and `author:app/renovate` returned no
  results. This is not proof that the app is absent or that its commits are
  verified; it supplies no signature acceptance evidence.

Signing acceptance remains open until a real Renovate update has a commit with
GitHub `verification.verified == true` and passes the repository's actual
protected checks. No credentials, permissions, app installation or required
check was changed by this increment.

## Validation of this configuration increment

The JSON parses successfully. A structural comparison with
`git show HEAD:.github/renovate.json` removes the added `platformCommit` property
and asserts that the remaining JSON objects are equal. That check exited 0:
`PASS: valid JSON; platformCommit=enabled is the only semantic change`.

Strict Renovate validation is recorded below separately from application and
installed-consumer acceptance.

Attempted command, from the repository root, with Node `v24.22.1`:

```sh
npx --fetch-retries=0 --fetch-timeout=20000 --yes --package=renovate@44.239.0 renovate-config-validator --strict --no-global .github/renovate.json
```

Exit status: **137**. npm emitted dependency deprecation warnings before the
process terminated; no validator result was produced. The requested Renovate
version was 44.239.0, matching the version previously used in PR #594, but its
installation and execution did not complete in this environment. No cached or
global Renovate validator was available. The termination cause was not
established. **Strict configuration validation remains unexecuted**; JSON
syntax and structural-diff checks are not represented as a substitute.
