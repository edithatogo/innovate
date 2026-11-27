# Issue #414 implementation plan

## Alert #56 and StepSecurity PR #681 follow-up, 18 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the repository protection
readback and the bounded PR #681 policy repair. The worktree
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`
is on `codex/innovate-1118-security`. It started at
`bdc8558fd468c9d40b42b99267ba141bf4b10332` and fast-forwarded to
`20fe0b41b815bc926fc3d61b9672c3d4b2feb289` after PR #678 merged.
The allowed source scope is this plan, the policy snapshot checker and its
tests, the bot-added `.github/dependabot.yml` and its ESLint hook addition
in `.pre-commit-config.yaml`. Preserve the existing hook entries. The native
scope is the existing tracked solo-maintainer ruleset and the disposition of
PR #681. Other PRs and their auto-merge settings have separate owners.

The Scorecard alert reports missing explicit pull-request protection and
mandatory human and code-owner reviews. The existing
`.github/rulesets/main.json` proposal defines an explicit pull-request rule
with zero required approvals, force-push and deletion protection, and two
strict automated contexts already required by classic branch protection.
Verify the live baseline and activate that proposal. Preserve the classic
protection and the existing CodeQL results ruleset, then rerun Scorecard and
read the alert back. The solo-maintainer policy requires zero approvals.

The saved ruleset `27532448` is active on main. Its four rules are effective
alongside CodeQL rule `27522878`. The entire classic protection response was
unchanged across the activation. GitHub added its default
`require_extra_approval_for_unattributed_changes: true` and an empty
`required_reviewers` list to the saved response. GitHub documents that this
default adds an approval only when the required approval count is above zero.
The native and tracked configurations both specify zero, so retain the
default and validate its type in the live snapshot checker.

PR #681's job `134656576567` at
`3305d565642beb950d505d0e2842199c8dfe0032` failed when the policy checker
detected the bot-added weekly GitHub Actions update schedule. GitHub reports
an active Renovate dashboard covering Actions and enabled Dependabot
vulnerability alerts and security fixes. The other proposed change repeats
the ESLint 10.8.0 hook that previously failed qualification in PR #672.
Reproduce the policy failure on the exact bot source and qualify the minimal
repair without promoting changes while the supply-chain checks are failing.
Use the isolated standard Git worktree `/Users/doughnut/.codex/worktrees/innovate-pr681-policy/innovate`,
branch `codex/pr681-policy-repair`, for the exact bot-head comparison.

On the bot head, repo-local Python 3.15.2 reproduced the policy exit 1.
The proposed ESLint 10.8.0 hook returned exit 2 on the documentation generator
because the repository has no ESLint configuration. Removing the two bot
additions made the policy command and all 21 policy tests pass. The repaired
tree has zero differences from the bot PR's base `bdc8558f`. GitHub confirmed
PR #681 closed without a merge at `2026-11-18T08:21:29Z`. The isolated
verification worktree was removed after restoring the two trial paths.

Scorecard run `44265672565` completed on `20fe0b41` after the ruleset activation.
Branch Protection rose from 3 to 4 and stopped reporting a missing pull-request
requirement. Alert #56 remains open for Scorecard's review and bypass criteria.
Native classic protection still applies to administrators, and the separate
CodeQL rule has no bypass actors. The activated tracked ruleset retains the
owner's pull-request recovery exception. The combined live settings snapshot
passed the policy checker with both rulesets, strict classic checks, and zero
required approvals. The checker now validates the GitHub-added boolean, and
all 21 policy tests and Ruff pass on the selected Python 3.15.2 environment.

## Dependabot #166 documentation dependency mitigation, 18 November 2026

The same integration branch owns a bounded docs workspace follow-up from
`20fe0b41b815bc926fc3d61b9672c3d4b2feb289`. Allowed files are this plan, the
site's pnpm workspace and lock, the two scoped package patches, and a focused
documentation test. Keep the existing dependency release-age policy and full
site checks. The advisory lists all versions through 4.0.7 and no fixed
release. The lock includes 4.0.7 via Astro, Starlight, Vite and tinyglobby, and
2.3.2 via anymatch. Its unchanged baseline SHA-256 is
`17bef22df127cab558cae199a180afbf3effa63a5f7f73fb7bb976d5da7f0826`.
Prepare a temporary, version-bound guard rejecting repeated extglob syntax
in both package parsers and qualify the actual installed docs build. This
is an interim mitigation. Keep the advisory open pending an upstream release
and successful scanner readback.

The updated lock has SHA-256
`c3983b770f2dca6cfb1dc506ec7cfab416a04d8ddb452a512e6c420d516f41c3`.
All seven resolved `picomatch` references carry the applicable patch hash.
Removing the patch references restores the original lock byte for byte.
With Node 26.13.1 and pnpm 12.21.0, `pnpm install --frozen-lockfile`,
`pnpm test`, `pnpm build` and `pnpm check` passed on the final patches.
The suite exercised both installed dependency paths and passed all 58 tests.
The site built 211 pages, with valid internal links and a passing production
artifact check. Astro reported zero errors, warnings or hints. GitHub still
reports Dependabot alert #166 open and lists no patched version.

## Alert #44 dependency follow-up, 18 November 2026

GitHub still reports `RUSTSEC-2026-0543` and `RUSTSEC-2026-0555` on main
`20fe0b41` after PR #678 merged the temporal compatibility tests. The locked
Innovate consumer graph still contains `chrono` 0.4.45 and `chrono-tz` 0.10.4
through Polars 0.55.2. The upstream Polars change in PR #29110 is an unmerged
draft. A released Polars dependency that removes these paths and preserves
the public `DataFrame` type remains necessary. Keep both advisories visible
in RustSec. The release and offline security lanes still require successful
qualification before promotion.

## PR #684 base update, 18 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns this draft branch in
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`.
Align its head `ecaccce231a1aa53b46418df52bd2cf3cc4e8d85` with main
`fd7e259bb9a685d620f80907c6815d4837da3e53` through a normal merge.
The preview confirmed the same eight PR paths against the new base.
The only additional edit in this operation is this record. Native scope is
the existing draft's head branch and check readback. Its next required
promotion evidence is successful checks on the updated head.

The new main includes the scientific evidence and Arrow/MSRV workflow from
PRs #662 and #685. Its hosted Release Readiness run `44288977642` still
reports all 38 incomplete scientific obligations. The Rust dependency
fixtures and both documentation patches retain their existing bytes.
On the integrated tree the local repository-policy check, all 21 policy
tests, both installed `picomatch` controls and the diff checks passed.

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
canary `44235275341` passed on PR #682 head
`e572893aa7ded32aba1753d77cb6243a88d56432` with its restricted token. The
same hash-verified Opengrep 2.0.0 local rule-engine diagnostic scanned the
verifier and its test file with no findings or errors. This focused scan is
not the separate complete consumer SAST or parser gate. Later commits require
their own hosted checks. A subsequent read-only verifier run also passed for
PR #678 head `a498d1df54d6e6353dadf6ea531281e7145e898e`, using CodeQL run
`44237683850`; that head's RustSec and release-readiness jobs failed.

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

On 18 November the same integrator tests a new hypothesis: the two prior test
pushes had no Rust or Python source change. An isolated `codex/codeql-probe2-base`
branch may receive one harmless Rust comment after its branch-scoped protection
is active. Its only purpose is to request a real default-setup baseline before
testing the native code-scanning rule; it does not target main or change another
pull request. Record the exact outcomes and remove only these owned probe
resources. No skipped, neutral or absent result is treated as a pass.

Ruleset `27521435` made the isolated branch report protected. Source-bearing
commit `a49fbe0b9e0918bb6d96bf5b199a48a30a83a2ce` still produced no CodeQL
run or analysis on that ref. A final bounded comparison adds classic branch
protection to this same owned test branch and pushes a second harmless comment
change; the classic-protection endpoint previously returned 404 for this ref.
This tests whether default setup distinguishes the protection mechanisms. Both
temporary protections and only this branch's queued jobs must be removed after
the result is read back.

Classic protection followed by comment-only commit
`b59da1d3b3a0ce4f0d16c0016c74cf56567bef07` started default-setup run
`44238449463`; this is a trigger observation, not a completed analysis. The
same integrator may create a separate `codex/codeql-probe2-head` branch with a
harmless source comment and an explicitly temporary pull request targeting
only the protected test branch. Add the native CodeQL code-scanning rule only
to ruleset `27521435`; observe its behavior while the test analyses are
pending and after they finish. After both baseline languages succeed, a single
rerun of the owned test PR's Rust job may test whether a same-head rerun blocks
the native rule. The test PR has no auto-merge and must never be
merged. Close it and remove both owned test branches, both temporary protections
and their queued offline jobs after recording the terminal evidence.

The base run and PR run eventually completed successfully for both languages,
and both uploaded analyses were present on each exact ref. GitHub rejected the
owned PR's Rust-job-only rerun with HTTP 403: jobs in this dynamic workflow run
cannot be rerun. The separately scoped full-run rerun endpoint may be tried
once; its outcome must be recorded and must not be generalized to other runs.
The full-run request was also rejected as not retryable. The integrator may
instead advance only the test PR head with a second harmless comment change,
then observe the native rule against the now-complete test base while the new
head's analyses are pending. This tests a new head, not a same-head rerun.
On new test head `1dff04125f59428f5f8c0e7a0f11ebd57c96f1e5`, with the
completed Python and Rust baseline at `b59da1d3`, GitHub's native PR view
displayed “Merge is blocked” and said it was waiting for CodeQL results while
both new-head language jobs were still running. This establishes the initial
missing-result block for this test, not that the rule requires each language
or rejects skipped/neutral results after one language has completed.
After the new-head Python job succeeded, its Rust job was still running and
the aggregate was neutral. GitHub continued to block the test PR, explicitly
reporting one expected CodeQL result. The earlier head had both successful
analyses and an `UNSTABLE`, rather than `BLOCKED`, merge state after the base
finished. This does not qualify the documented Dependabot exception or a
same-head rerun, which GitHub rejected for this generated workflow.

The authorized native scope may also include one additive main-only CodeQL
code-scanning ruleset with no bypass actors, error-level non-security and
high-or-higher security thresholds. Before creation, current main `0f86ee6b`
has both CodeQL baseline analyses. Separately owned PR #662 is behind with
auto-merge enabled, both language jobs and aggregate successful, and no open
CodeQL findings on its current head; PR #681 also has both successes and no
open CodeQL findings. Preserve the classic protection, existing checks and
PR #662's auto-merge setting, and read them back after any additive rule.
This partial strengthening is not a success-only guarantee for Dependabot or
an unqualified skipped/neutral language result.

Ruleset `27522878` was then created and read back active on `refs/heads/main`
with only the CodeQL results rule, the stated thresholds and no bypass actors.
The effective main rules endpoint confirms it applies. Classic strict mode,
enforced administrators, zero required approvals and all nine app-bound status
checks remain present. PR #662 remained open at
`d98ede3b4daa1a3f1a1fcec74e015ed77e24c3d5`, behind and auto-merge enabled;
PR #681 remained open at its prior head and blocked. This is an additional
native barrier for applicable pull requests, not complete success-only
enforcement for every actor or every skipped result.
After activation, PR #682 at `e572893a` with both CodeQL analyses successful
remained `UNSTABLE` rather than rule-blocked; PR #678 remained behind. The
disposable `1dff041` PR run was only needed to observe the missing-Rust block;
if stopped during cleanup, its cancelled Rust job is not a successful scan.
The owned test PR #683 was closed unmerged, run `44241877647` was cancelled
with its Rust job cancelled, and its seven queued offline jobs and second
release-drafter run were cancelled. The isolated worktree, both test branches
and both test protections were removed. Readback leaves the new main ruleset
active and PR #662's head, behind status and auto-merge unchanged.

The policy checker's new live-snapshot control first failed because it did not
recognize this additional no-bypass rule. It now requires an active main
CodeQL results rule with at least the observed thresholds and rejects missing,
disabled, excluded, bypassable, wrong-tool or weakened variants. It accepts
stronger thresholds and additional tools. The actual rule `27522878` passed the
targeted ruleset parser readback. On repo-local Python 3.15.2, the combined
policy and verifier suites passed 32 tests; Ruff, the local repository-policy
check, Conductor hygiene and `git diff --check` passed. The shared staged
Gitleaks 8.30.1 wrapper reported zero findings; a local Presidio scan was not
run in these lean environments. This targeted readback does not certify every
other live repository setting.

The native rule is active on the test ref with CodeQL `errors` and
`high_or_higher` thresholds. Test PR `#683` has no auto-merge, base
`b59da1d3b3a0ce4f0d16c0016c74cf56567bef07`, and head
`90b9b688d6673f925692cc784c9a139f125af368`. At approximately 04:39 UTC,
the internal GitHub PR view displayed “Able to merge” while both base run
`44238449463` and PR run `44238616323` had successful Python jobs but Rust
jobs still in progress. REST returned `mergeable: true` and `unstable` at the
same observation. The base Rust analysis had not yet established a baseline,
so this proves only that the rule did not block that incomplete-baseline
state; it does not establish behavior after both base languages are present.
No merge was attempted. The test PR also triggered release-drafter run
`44238619427`; its log reported forced dry-run and disabled publish for the
ephemeral PR merge ref. The owned run was cancelled and its result was
`cancelled`.

For the hosted Opengrep failure, a separate clean detached diagnostic worktree
at exact consumer `e2a3ce65f330252c9ad74b182a19a0e8cdb19418` was scanned
with the SHA-256-verified macOS Opengrep 2.0.0 release and the candidate's
unchanged rules. The local rule-engine run reported all 891 parent-index source
paths, seven Python egress-rule findings and one partial Rust parse error; one
uninitialized submodule was outside that inventory. The seven rule matches
occur in four developer-reporting scripts. They have not been classified as
actual leaks or resolved, and this diagnostic did not rerun the separate
per-file parser gate. No match contents were retained or sent to CI. The clean
diagnostic worktree was removed. A macOS diagnostic is not the failed hosted
Linux consumer run or proof of complete SAST coverage.

On current PR #682 head `e572893aa7ded32aba1753d77cb6243a88d56432`, a
separate real Opengrep 2.0.0 strict per-file Rust parser probe tested all 746
tracked Rust files. It rejected 46: 26 under vendored faer, 18 under vendored
nalgebra, and one each under vendored faer-traits and pulp. No source text or
scanner diagnostics were printed. Vendored source remains within the scanner's
tracked-source inventory; these failures are not excluded or treated as clean.
The local macOS parser result is not the hosted Linux qualification.

The shared standards draft's later exact-head hosted run `44239253142` on
`8e303d11a81ce821ffb37893030ee0d91d07b93f` again passed image construction
and controls but failed the consumer Opengrep step. Its artifact `14399107739`
has verified archive SHA-256
`6d3b9b9d2f805423a996d4fa4d57527f107dbcc862a46118a1ae8b571914e732`.
The manifest and SBOM/vulnerability receipt hashes match; the all-layer report
retains 451 matches including 11 critical and 129 high, and one ignored match.
The tested image and release remain unqualified.

A separate temporary inventory checked the frozen source ZIP hashes and root
licence/notice file hashes for all 65 compiled Go modules in the candidate
Gitleaks binary. Eleven ZIPs were retrieved from their frozen Go-proxy sources;
the other 54 were already cached. Each module had at least one root notice.
The inventory SHA-256 is
`04a14db9dc8d1c3bc3c4c157861998e22733e376505e096bd730e97312fdcc49`.
This does not classify the licences, cover all embedded or noncompiled source,
or establish redistribution compliance; the licence review remains open.

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
