# Empirical validation integration review

- Owner: Codex chat `01a1e3d8-fb04-7010-a038-2a5478662a5d`.
- Worktree: `/Users/doughnut/.codex/worktrees/innovate-empirical-integration/innovate`.
- Branch: `codex/innovate-empirical-integration-scanned-20261117`.
- Base: `cc2e4da4af9c848043b31f1191c9d96212048276`.
- Scope: receive the reviewed M01 and Q04 source packets, propose honest
  empirical-track status and exercise their offline tests in CI and Nox.
- Allowed files: the two source packets and their lane receipts; this receipt;
  `conductor/tracks.md`; this track's `index.md`, `metadata.json` and `plan.md`;
  `.github/workflows/ci.yml`; and `noxfile.py`.
- Native object scope: none. This branch is an integration proposal; the
  existing sole integrator retains shared-branch promotion, shared workflow
  ownership and any required acceptance of the common registry changes.
- Next operation: validate the combined source and track-status proposal and
  preserve the exact results for the existing integrator.

The original M01 branch was `054074beca4062f63be77b905b815503a1103168`,
based at `18c467e18b71d972060365c68865aa405a69949c`; it supplied three scoped
commits ending in the [source handoff](m01-source-qualification-20261116.md).
The original Q04 packet was `1f9e6ae6cd2c6d8264335326812f7fd83c83236b`, based
at `57d2916586d107c8affe90355e453abe27f01c26`; its findings and limitations are
in the [claim review](q04-claim-review-20261116.md). Read-only integration
preflights immediately before each cherry-pick found clean source and target
worktrees and no overlapping dirty paths, staged deletions or missing files.
The reports do not establish hosted checks, acceptance or mergeability.

Initial combined focused verification with the branch-local CPython 3.15.2
environment: **39 passed**, comprising the 27 M01 cases and 12 Q04 cases.
Ruff check and format check passed for the five source and test files. The
repository-policy check passed its local scope; live settings were not checked.
No hosted result is inferred for this branch.

## Integration review and corrections

- The original Q04 historical-revision test relied on an older commit being
  present in the surrounding repository. A synthetic checkout containing only
  a new source commit reproduced a failure, as the CI checkout is shallow by
  default. The test now creates a separate temporary Git repository with a
  historical source commit and a newer commit. The same synthetic-checkout
  reproduction then passed, retaining the historical-revision assertion and
  the requirement that self-declared receipts remain unqualified.
- `uv run --no-sync nox -s validation` now executes the three M01/Q04 test
  files using this worktree's `.venv/bin/python` (CPython 3.15.2): **39 passed**
  on macOS arm64. CI proposes this same session for each existing native-Python
  platform and separate lint/format checks for the affected source and tests.
  This is not a hosted-platform or installed-artifact receipt.
- `ruff check` and `ruff format --check` passed for the five M01/Q04 files and
  `noxfile.py`; the complete local `nox -s lint` session, including
  `cargo fmt --all --check`, also passed. `scripts/check_repository_policy.py`,
  `scripts/conductor_registry_hygiene.py`, the 20 selected repository-policy
  and workflow-documentation tests, and `git diff --check` passed locally.
  An actionlint 1.7.12 run without external integrations parsed
  the changed workflow successfully. The full local actionlint attempt did not
  complete and was interrupted; a separate ShellCheck run passed the three new
  run commands. The required hosted workflow/security checks remain outstanding
  and must not be represented by the partial local results.
- A local Gitleaks scan reported the unmodified upstream receipt's
  `idempotency_key` as a generic API key. The Atlas replay implementation
  derives that value as SHA-256 of the partition and snapshot IDs; it is a
  deterministic identity, not an access credential. The fixture now stores a
  null placeholder and the test reconstructs the original metadata bytes from
  those public IDs, checking the original pinned receipt SHA-256 before use.
  The source payload is not included. This avoids a scanner exemption while
  retaining the exact-byte receipt check; the original local candidate branch
  is preserved for its history and is not the proposed branch to publish.
- The proposed registry, plan, index and metadata describe V01–V04, Q01 and
  Q04 as partial, rather than completed. Q01 still needs its complete corpus,
  fuzz and sanitizer evidence assessed; this review does not say it failed.
  M01 is partial until the selected CI and integration are accepted. See the
  individual source receipts for the original findings and review limits.

## Open acceptance

The migration release checker still cannot bind a self-declared qualified
ledger entry to substantive scientific and exact-artifact acceptance. Its
existing synthetic test demonstrates that arbitrary hashed text can satisfy
the current migration-only check. The currently checked-in ledger continues
to block release with 38 unqualified obligations. A shared evidence contract
and the Q02/Q03/Q04 gates remain open; this proposal does not create a receipt
schema or assert release readiness. M02–M05, V05, scientific review and
publication depend on further source, consumer and validation work.
