# Repository security and solo-maintainer context

## Problem and scope

At main `276c2e505a0ce26910820fe8a94a6a498f505700`, issue #414 still has no
root agent context, no repository ruleset, competing dependency-update schedules,
and no current automated native coverage upload. The archived June hardening
track describes historical gates and cannot establish current enforcement.

This track implements issue #414 for the existing maintainer. It does not alter
scientific execution, runtime dependencies, release criteria, other agents'
native work, or the dedicated offline-security runner in draft PR #590.

## Requirements and acceptance

1. Provide a reviewable main-branch ruleset blocking deletion and force pushes,
   requiring pull requests with zero mandatory human, code-owner or team
   approvals, and requiring only observed stable automated checks. Preserve
   owner recovery and existing protections; no bot bypass.
2. Inventory current bots and their effects before selecting required contexts.
   Document rejected conditional or unqualified contexts. A local JSON proposal
   is not live enforcement; activation requires an authenticated administrator
   and read-back evidence of the combined ruleset and branch protection.
3. Add canonical agent guidance and reconcile contribution/security guidance.
   Provide a dependency-free, read-only repository-policy validation command and
   a timestamped drift receipt distinguishing local checks from live settings.
   Negative tests must reject weakening, malformed or unavailable evidence.
4. Inherit `github>edithatogo/renovate-config`, retaining explained repository
   overrides. Remove Dependabot version-update scheduling only after hosted
   Renovate health and coverage of the same ecosystems are verified. Preserve
   GitHub vulnerability alerts and security-update settings.
5. Generate actual Rust and Python-binding coverage, retain source-bound reports,
   and upload with Codecov OIDC and job-scoped permissions. Verify the service
   commit result before claiming delivery. Do not require an unproven Codecov
   context in branch rules, infer Rust coverage from Python, or infer scientific,
   branch, mutation or release acceptance from line coverage.

## Implementation boundaries and verification

Use existing pinned toolchains and immutable Actions. No runtime dependency or
public API changes. The checker consumes local files and optional sanitized
GitHub GET snapshots; it never changes settings or handles credentials.
Keep the existing native CI checks and failing release gate intact. Use focused
policy tests, upstream Renovate validation, workflow validation, independent
review, and exact-head hosted checks. Record any external access blocker rather
than inventing a successful setting change. Revert this scoped PR to undo source
changes; retain the baseline settings receipt before any future activation.
