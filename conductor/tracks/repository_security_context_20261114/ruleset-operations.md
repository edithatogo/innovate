# Main-branch ruleset: application, drift and recovery

## Current status

[The JSON proposal](../../../.github/rulesets/main.json) is source-controlled but
**has not been applied**. [The receipt](drift-receipt.json) records no repository
rulesets and a permission failure reading full classic protection. The branch
summary confirms seven existing required contexts, but cannot establish approval,
force-push, deletion or bypass settings. Keep #414 open until these live settings
are verified.

## Why these rules

The proposal targets the default branch, blocks deletion and non-fast-forward
updates, and requires pull requests without mandatory approving reviews,
CODEOWNERS approval, last-pusher approval or team approval. The existing
CODEOWNERS file supplies routing information only.

The ruleset adds the two existing CI aggregate contexts `lint` and
`unit-tests-base (3.15.1)`, bound to the GitHub Actions app (`15368`) and strict
up-to-date checking. Both have successful exact-baseline runs; their jobs run
after every native matrix result, including failure, rather than disappearing
under a path filter. The legacy Python version in the second name is an existing
compatibility context; its tests execute Python 3.15.2.

**Preserve all seven existing classic-protection contexts:** `lint`, `package`,
`CodeQL`, `Secret Scan`, `Workflow and policy lint`, `Python quality harness`, and
`unit-tests-base (3.15.1)`. This additive proposal does not replace or remove
them. GitHub combines rulesets with classic branch protection, so adding a
zero-approval ruleset cannot cancel an existing human-approval requirement.

The admin role (`RepositoryRole`, ID `5`) has pull-request-only recovery. No
integration, deploy-key, team, writer or maintainer bypass is granted. This
cannot authorize a direct force push or branch deletion; the repository owner
retains administrative control of the settings. Recovery is exceptional and
recorded, never a substitute for required checks during routine integration.

## Bot inventory and settings decisions

Hosted Renovate created [dashboard #595](https://github.com/edithatogo/innovate/issues/595)
and enumerated all four ecosystems previously scheduled by Dependabot. The
recent-PR inventory also contains 14 Dependabot PRs, confirming overlap. The
updated Renovate config inherits `github>edithatogo/renovate-config`; the local
Monday cron remains because [#594](https://github.com/edithatogo/innovate/pull/594)
repaired an invalid schedule. Existing main-branch targeting, production/dev
grouping, digest pinning, limits, semantic commits and major-update handling
remain to preserve the established rollout behavior. No GitHub vulnerability
alert or security-update setting is changed by removing the version-update file.
Three narrowly scoped regex managers keep the inline Renovate validator,
cargo-llvm-cov and Codecov CLI versions visible to Renovate; the Actions manager
owns the immutable uploader-action pin. The dedicated configuration workflow
resolves the public inherited preset with the pinned upstream validator. Its
path-filtered job is not a required branch context.

Release Please, Release Drafter, TagBot and the incident monitor use their
existing GitHub Actions permissions. They receive no ruleset exemption.
Conditional deployment, release, monitoring and offline-security jobs are not
new required PR contexts. Codecov must produce a verified service result before
its contexts are considered stable enough for protection. The separate release
readiness failure and offline-security draft #590 remain explicit gates in their
own scopes.

## Apply only through an authenticated administrator

1. Read the current default branch, full branch protection, and all rulesets with
   inherited rules included. Retain the exact pre-change responses privately if
   they contain account details. Check for a same-name ruleset before creating
   one, and reconcile any new owner changes rather than overwriting them.
2. Run `python scripts/check_repository_policy.py`. Review the JSON against the
   current bot/check inventory. Verify the seven existing protected contexts,
   their app IDs and zero mandatory human-approval settings in the full response.
3. With the existing authenticated GitHub CLI administrator connection, create
   the reviewed ruleset once:

   ```sh
   gh api --method POST repos/edithatogo/innovate/rulesets \
     --input .github/rulesets/main.json
   ```

   If the same-name ruleset already exists, compare its full definition and use
   the appropriate update operation only after reconciling the differences. Do
   not remove classic protection, weaken checks or add an app bypass.
4. Read back the full created ruleset and full classic protection, along with
   the current default-branch commit. Populate a fresh sanitized snapshot using
   the shape of `drift-receipt.json`, with complete objects and an empty `errors`
   list only when every read succeeded. Validate it:

   ```sh
   python scripts/check_repository_policy.py --snapshot path/to/fresh-snapshot.json --json
   ```

   The existing receipt intentionally fails live validation. A local policy pass
   does not turn inaccessible, stale or missing remote evidence into a pass.
5. Record the ruleset ID/URL, observation time, effective rules, application
   identity and check results. Mark activation complete only after read-back.

## Recovery and rollback

If the new ruleset blocks an intended change because a named check was renamed,
repair the check/context mapping through a reviewed PR. In an exceptional
administrative recovery, record the reason, preserve the prior definition and
make the smallest reversible ruleset edit; retain classic protection throughout.
Never force-push or delete `main` to recover CI. Reverting this source PR does not
change a live ruleset, and changing a live rule does not rewrite the receipt.

## Authoritative references

- [GitHub ruleset REST API](https://docs.github.com/en/rest/repos/rules)
- [GitHub rule layering](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [Creating rulesets and restricting bypass to pull requests](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
- [Current central Renovate preset inspected for this change](https://github.com/edithatogo/renovate-config/blob/e3689e25be30b8eeb4799d442dbc9f6fdffed1c3/default.json)
