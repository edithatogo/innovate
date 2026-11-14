# Issue #414 implementation plan

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
