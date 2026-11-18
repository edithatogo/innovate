# Issue #414 implementation plan

## Alert #44 CodeQL language-check gap — 18 November 2026

- Owner: Codex chat `01a1e448-3a28-7653-a681-cd3fdb41f498`.
- Isolated worktree: `/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`,
  branch `codex/alert44-codeql-language-gate`, based on main commit
  `0f86ee6bc40600d19df467b2c395741c89ad7f92`.
- Allowed files: this plan, `scripts/check_repository_policy.py`, and
  `tests/workflows/test_repository_policy.py`. Native scope is only the required
  status checks on `main`; preserve all other branch-protection settings.
- Next operation: qualify the two existing CodeQL language-job contexts against
  current pull-request and bot histories, prepare the drift-checker change, and
  add both app-bound contexts to protection only if they do not replace an
  existing check or disturb an in-flight pull request.

On 18 November this chat continued the requested success-only qualification in
the same worktree. The allowed source scope additionally includes
`scripts/check_codeql_success.py` and `tests/workflows/test_codeql_success.py`.
The next bounded operation is a read-only verifier for an exact PR head, using
GitHub's default-setup workflow, latest run, both job conclusions and uploaded
analysis identities. It does not publish a status or change native protection.
Any later automated status producer requires separate threat and event-race
qualification before it is made required.

The allowed scope also includes `.github/workflows/codeql-evidence-canary.yml`:
a non-required, read-only hosted canary may exercise the verifier with its
actual restricted GitHub token. It does not authorize a required status or
checkout of PR code in a privileged workflow.

The verifier requires an exact open PR head, the current GitHub default-setup
run, successful Python and Rust jobs on its latest attempt, matching uploaded
analyses with completed SARIF processing, a successful GitHub CodeQL aggregate,
and no open CodeQL findings on that head. It checks again for concurrent head,
run or aggregate changes. This is a stricter point-in-time observation; it is
not continuous enforcement. In particular, a later same-head rerun needs an
independently qualified event/status design before any result can be required.

With repo-local Python 3.15.2, 11 verifier cases and 19 policy cases passed,
including negative controls for skipped/neutral checks, wrong app or ref,
missing or stale analyses, unfinished SARIF, findings and rerun races. A
separate review exposed an older-run rerun and inconsistent-metadata bypass;
both were fixed and their controls passed. Read-only live probes returned
success for PR #678 head `0c9ecfd3`, pending for PR #682 head `dc9ce084` while
Rust was running, and failure for a wrong PR head. Ruff, actionlint, Zizmor on
the canary, repository policy and Conductor hygiene passed locally. Hosted
canary execution and its restricted-token evidence remain pending.

The required `CodeQL` aggregate was neutral on PR #671 while the Rust analysis
had not finished. GitHub treats neutral and skipped required status checks as
passing. Requiring each default-setup language job adds a missing/pending/failure
barrier, but does not by itself reject a future neutral or skipped job. Keep
that residual gap open and do not claim complete enforcement from this change.

On 18 November, the exact current heads of open PRs #662, #678 and #681, and
merged PRs #671, #672, #670, #667 and #666, each reported successful
`Analyze (python)` and `Analyze (rust)` checks from GitHub Actions app 15368.
Those samples include StepSecurity and Renovate updates; they do not guarantee
that future configurations or bot changes cannot skip a language. PR #662's
existing auto-merge request is owned elsewhere and was not changed. Its current
head already has both successes and its merge state is behind.

The drift-checker regression first failed for both absent contexts and both
wrong-app contexts. With the two app-bound requirements added to the checker,
all 19 repository-policy tests pass on repo-local Python 3.15.2. The local
policy check, Conductor hygiene, Ruff and `git diff --check` also pass; the
local policy check does not inspect live settings.

The live classic protection was then updated using the exact reviewed baseline
and read back: strict mode and the seven pre-existing app-bound contexts remain,
and `Analyze (python)` and `Analyze (rust)` are additionally required from app
15368. The full protection response outside required status checks was unchanged,
including enforced administrators and zero required approvals. PR #662 remained
open, behind and auto-merge enabled, with its existing head unchanged. This
additive change does not reject a skipped or neutral language job and is not
acceptance of the complete CodeQL requirement.

The updated checker accepted the freshly read classic protection after the
separate restrictions endpoint explicitly returned HTTP 404 `Push restrictions
not enabled`. The protection response itself omitted that field, so evaluating
it without the second read correctly failed as incomplete. This targeted
classic-protection check does not certify the separate ruleset proposal.

### Isolated native rule qualification

The same chat owns a temporary `codex/codeql-rule-probe-base` branch and a
branch-scoped test ruleset for this task. The probe starts at
`b74de7e678966865ea91fe00c5fe0eb15d31c20f` and advances normally to
`0f86ee6bc40600d19df467b2c395741c89ad7f92`. It does not target `main` or alter
another pull request. First enable only a non-fast-forward rule and scan the
protected test base. Then test a CodeQL code-scanning rule against a temporary
pull request using this branch's source, read back the resulting rule and
checks, and close the owned probe after its terminal evidence is recorded.
The probe will not authorize skipping findings or altering the existing checks.
GitHub documents that its code-scanning rule does not apply to Dependabot pull
requests analyzed by default setup, so this limitation must remain explicit.
The first fast-forward to the already-scanned main commit did not create a
CodeQL run or analysis for the test ref. A normal empty commit in a separate
test worktree will check whether a distinct commit triggers the baseline scan;
an absent baseline is a blocked probe, not a pass.

Rule `27519086` applied only `non_fast_forward` to the test branch. The normal
fast-forward to `0f86ee6b` and the distinct empty commit
`4b635ae775690b5278a60e72e034623a85ddd922` produced no default-setup CodeQL
workflow runs or code-scanning analyses for that protected ref. The expected
baseline was Python and Rust analyses; only offline-security jobs queued.
Because both targeted baseline attempts failed, the proposed code-scanning
rule was not exercised or applied to main. Its missing-analysis and Dependabot
behavior remain unqualified here. The owned test rule, branch and queued test
jobs are removed after readback; this does not change main's protections.

## PR #672 workflow-lint repair — 17 November 2026

- Integrator: Codex chat `01a1e448-3a28-7653-a681-cd3fdb41f498`.
- Isolated app-managed worktree:
  `/Users/doughnut/.codex/worktrees/innovate-pr672-security-pin/innovate`,
  branch `codex/pr672-workflow-lint`, based on PR head
  `49602f622e19a04fa99e3ad872c6b277d484db95`.
- Scope: reconcile the PR with current main, pin the bot-added
  `step-security/harden-runner@v2` uses to the verified upstream commit already
  used by this repository, remove its newly introduced competing Dependabot
  version schedule, refresh the workflow-bound documentation inventory, and
  record validation. Allowed paths are this plan, affected `.github/workflows/`
  files, `.github/dependabot.yml`, and
  `docs/contracts/polyglot-api-documentation.json`. Preserve existing security
  controls and Git hooks.
- Native object/authentication scope: none. The next operation is a read-only
  baseline audit, then a standard merge of verified main `cc2e4da4` into the
  isolated candidate and focused local and exact-head hosted checks.

The linked workflow-lint job reported 30 high `unpinned-uses` findings for the
new floating Harden-Runner references. The upstream `v2` and `v2.22.4` tags
both resolved to `1d88384fd173af6ecb012dfd0264a306b711241a`, already pinned in
six release workflow steps on this base. The repository-policy checker also
reproduced the bot's competing Dependabot version schedule; Renovate's existing
dashboard #595 is open and was updated on 17 November. The bot's CI workflow
edit has made the recorded polyglot structural inventory stale. These are
separate from scientific release acceptance and the offline-security lane.

The isolated repo-local Python 3.15.2 environment also ran the bot-added
ESLint 10.8.0 pre-commit hook against an existing documentation script. It
failed because this repository has no ESLint configuration. The allowed scope
therefore includes `.pre-commit-config.yaml` solely to omit this new, unqualified
hook; retain the existing hook configuration and the bot's other hook additions.
Node documentation tests and existing lint/security checks remain unchanged.

On the merged candidate, repo-local CPython 3.15.2 with the workflow's Zizmor
1.32.0 reports zero active findings (the same offline default used by the
failed hosted job); the original 30 unpinned-use failures are gone. The existing
two ignored and 61 suppressed results were not changed. Actionlint 1.7.12,
16 Node workflow-control cases, 19 repository-policy cases, 11 release-workflow
cases, 38 polyglot documentation cases, Ruff and the local policy check passed.
The refreshed polyglot inventory changes only the CI workflow hash. The newly
proposed Gitleaks and ShellCheck hooks each passed an isolated representative
run; the incompatible ESLint hook was removed from the proposal, not from main.

Poutine 1.1.4 ran with the workflow's existing arguments: it returned zero and
reported 70 `github_action_from_unverified_creator_used` findings, the same
count observed on current main. This tool invocation does not use
`--fail-on-violation`, so exit zero is not a clean vulnerability assessment.
Exact-head hosted execution, including the original job's skipped steps, remains
required. Local policy success does not establish live GitHub settings.

Base: `276c2e505a0ce26910820fe8a94a6a498f505700`.
Branch: `chatgpt/issue-414-maintainer-security`.
PR: [#606](https://github.com/edithatogo/innovate/pull/606).

- [x] Reconcile issue, main, active PRs, Conductor and bot/check evidence.
  - #600 owns native ensemble changes; #590 remains a separate offline-security
    draft. Neither is modified by this track.
  - At the initial inventory, hosted Renovate dashboard #595 detected all four
    ecosystems in the existing Dependabot configuration, and main had no Codecov
    status/check. That dashboard establishes existing app activity, not evaluation
    of this PR's new central-preset inheritance.
- [x] Implement canonical context, solo-maintainer ruleset proposal and a tested
  read-only policy/drift checker.
  - Python 3.15.2: 19 stdlib policy tests passed. Local validation exits 0 with
    live state `not_checked`; the actual drift receipt exits 1 for the genuine
    missing ruleset and unavailable full protection. No live pass is inferred.
  - Independent review reproduced and fixed three defects: preserving every
    existing protected check/app, handling malformed enum values as structured
    failures, and accepting receipt metadata without trusting it as enforcement.
    The reviewer independently reran all 19 tests and targeted regressions.
  - Ruff 0.17.4 check/format passed; Zizmor 1.32.0 reports no new findings on the
    changed workflows under the existing policy. Conductor hygiene passes.
  - Actionlint 1.7.9 passed all three changed/new workflows; downloaded archive
    SHA-256 matched `233b280d05e100837f4af1433c7b40a5dcb306e3aa68fb4f17f8a7f45a7df7b4`.
    The unchanged CI-monitor suite also passed all 13 tests.
- [~] Adopt the central Renovate preset, validate the repository configuration,
  qualify merged inheritance, and retire duplicate Dependabot version scheduling.
  - The source configuration inherits `github>edithatogo/renovate-config` and
    removes the overlapping version-update schedule after confirming existing
    hosted Renovate activity.
  - Independent inspection of pinned Renovate 44.239.0 established that the
    successful first hosted validator invocation used local global-config mode.
    It did not resolve top-level `extends`; it is not evidence of validated
    repository-mode configuration or evaluation of the merged inherited preset.
  - The corrected invocation uses `--no-global` for local repository-config
    validation and does not receive an unnecessary Renovate token. Its execution
    on the next head remains pending. Even a successful local repository-mode
    result will not establish merged-inheritance evaluation; retain that gap
    until separate hosted-app or resolver evidence is recorded for this config.
- [x] Implement bounded Rust/binding coverage jobs, OIDC upload and report retention.
  - Codecov's validation endpoint returned `Valid!` for the configuration.
    Shell/YAML/pin/permission checks passed.
  - Hosted Rust and Python binding coverage, both 80% line floors, OIDC uploads
    and artifact retention passed on `b8020005`; see the source-bound evidence
    below. This is separate from the unsuccessful local build attempts.
  - Local native builds were killed by SIGKILL while compiling tokio, polars-arrow
    and (on the serialized retry) chrono-tz. No completed tests, coverage percentage
    or upload is claimed from these attempts. Pins and thresholds were retained.
- [~] Review the scoped diff and execute focused local and exact-head hosted
  checks; record outcomes without treating skips or missing tools as passes.
  - Prior-head hosted evidence is recorded below. The Renovate command correction
    and documentation changes require verification on the resulting final head.
- [!] Activate and read back the ruleset and combined branch protection.
  - GitHub connector reports HTTP 403 for protected-branch administration and
    exposes no settings mutation operation. The browser was signed out of GitHub;
    its settings page returned the signed-out 404. No settings were changed.
  - No active repository ruleset has been observed. The limited branch summary
    cannot establish approval, force-push, deletion or bypass settings.
- [x] Verify Codecov service activity and an exact commit's service status.
  - Codecov-app check `codecov/patch` succeeded on `b8020005`, establishing actual
    service activity for the uploaded PR coverage. Project contexts have not been
    observed; no additional context is declared stable or made required.
- [ ] Integrate after required automated checks and reconcile issue acceptance.

The issue stays open pending live enforcement, the corrected repository-mode
validator run, separate merged-inheritance evidence and final-head integration
checks. The 38 scientific obligations, manual scenario checks and release
readiness remain outside this track's completion claim.

## Hosted evidence for PR #606

These results belong to head
`b8020005153b76f3c0049b7f478bc29fcd0a7c0a`. They do not certify a later head.

- [Security and Quality run 43602887270](https://github.com/edithatogo/innovate/actions/runs/43602887270)
  passed all five jobs: Dependency Review, RustSec advisory audit, Secret Scan,
  Workflow and policy lint, and Python quality harness. The policy job executed
  all 19 repository-policy tests successfully.
- [Privacy and secrets run 43602887280](https://github.com/edithatogo/innovate/actions/runs/43602887280)
  and [Binding Conformance run 43602887282](https://github.com/edithatogo/innovate/actions/runs/43602887282)
  passed.
- [Rust coverage job 132471231966](https://github.com/edithatogo/innovate/actions/runs/43602887328/job/132471231966)
  executed 372 passing tests and one ignored live OECD test. It measured
  20,728/23,071 lines, **89.84%**. The 80% line floor, OIDC upload and artifact
  retention passed.
- [Python binding coverage job 132471231666](https://github.com/edithatogo/innovate/actions/runs/43602887328/job/132471231666)
  executed 158 passing tests and one skipped live OECD test. It measured
  205/237 binding lines, **86.50%**. The 80% line floor, OIDC upload and artifact
  retention passed.
- The aggregate Rust result does not establish the critical-module 90% target.
  Reported line coverage includes FFI **29.67%**, tabular **69.98%**, contracts
  **65.66%** and kernel **75.31%**. No critical-module, branch or mutation coverage
  acceptance is claimed from these line reports.
- [Codecov-app patch check 132476340576](https://github.com/edithatogo/innovate/runs/132476340576)
  completed successfully and links to
  [Codecov's PR #606 page](https://app.codecov.io/gh/edithatogo/innovate/pull/606).
  This is service-activation evidence for this tested source, beyond a successful
  uploader exit. Codecov project contexts have not yet been observed.
- [Renovate configuration run 43602887341](https://github.com/edithatogo/innovate/actions/runs/43602887341)
  completed successfully, but ran the local validator in global-config mode.
  Its prior job name overstated inherited-preset validation. The repository-mode
  correction must run on the next head; merged-inheritance evaluation remains
  separately unverified.
- [CI run 43602887274](https://github.com/edithatogo/innovate/actions/runs/43602887274)
  was still in progress at the recorded inspection. No complete required-check
  pass or merge acceptance is inferred from the completed jobs above.
  [Release Readiness run 43602887242](https://github.com/edithatogo/innovate/actions/runs/43602887242)
  retains the failure for the 38 unresolved scientific obligations.

## Integration reconciliation

Open PR #603 now owns #413's native metamorphic tests, workflow time bounds and
pinned actionlint installer. This PR changes only the policy-environment/install
and validation steps in their shared `security-quality.yml`; the hunks are
disjoint. Preserve both changes during integration. #602 owns tabular outcomes,
#604 owns registry readiness, #600 owns ensemble validation, and #590 remains an
unqualified offline-security draft. The existing Innovate integration coordinator
owns shared promotion; this lane supplies the focused PR and exact-head evidence
without racing other owners' main updates.

Local installation of the pinned Renovate validator was killed twice with exit
137; an inspected partial installation lacked a module. These are unexecuted
validation, not configuration failures or passing evidence. The dedicated
read-only hosted validator is intended to check local repository configuration
with `--no-global`; it does not resolve or evaluate top-level inherited presets.
It is deliberately not a required context because its triggers are path-filtered.
