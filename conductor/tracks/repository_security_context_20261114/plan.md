# Issue #414 implementation plan

Base: `276c2e505a0ce26910820fe8a94a6a498f505700`.
Branch: `chatgpt/issue-414-maintainer-security`.

- [x] Reconcile issue, main, active PRs, Conductor and bot/check evidence.
  - #600 owns native ensemble changes; #590 remains a separate offline-security
    draft. Neither is modified by this track.
  - Hosted Renovate dashboard #595 detects all four ecosystems in the current
    Dependabot configuration. Current main has no Codecov status/check.
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
- [~] Adopt the central Renovate preset, validate the resolved configuration, and
  retire duplicate Dependabot version-update scheduling.
- [x] Implement bounded Rust/binding coverage jobs, OIDC upload and report retention.
  - Codecov's validation endpoint returned `Valid!` for the configuration.
    Shell/YAML/pin/permission checks passed. Executed coverage is still pending.
  - Local native builds were killed by SIGKILL while compiling tokio, polars-arrow
    and (on the serialized retry) chrono-tz. No completed tests, coverage percentage
    or upload is claimed from these attempts. Pins and thresholds were retained.
- [ ] Review the scoped diff and execute focused local and exact-head hosted
  checks; record outcomes without treating skips or missing tools as passes.
- [!] Activate and read back the ruleset and combined branch protection.
  - GitHub connector reports HTTP 403 for protected-branch administration and
    exposes no settings mutation operation. The browser is signed out of GitHub;
    its settings page returns the signed-out 404. No settings were changed.
- [ ] Verify Codecov repository activation and the exact commit's service status.
- [ ] Integrate after required automated checks and reconcile issue acceptance.

The issue stays open until remote enforcement and coverage delivery have actual
evidence. The 38 scientific obligations, manual scenario checks and release
readiness remain outside this track's completion claim.

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
read-only hosted validator job resolves the inherited preset and is deliberately
not a required context because its triggers are path-filtered.
