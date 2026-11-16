# Repair of observed browser qualification failures

## Executed evidence

[Documentation run 43653989247](https://github.com/edithatogo/innovate/actions/runs/43653989247)
executed all 142 browser cases in
[build job 132628376222](https://github.com/edithatogo/innovate/actions/runs/43653989247/job/132628376222).
There were 61 passes and 81 failures. Both evidence uploads succeeded.
Native examples and deployment were skipped because the browser gate failed.

The PR head was 15606717b8b534bb7c95f47b6837907bd4dba6da. The actual checkout
was GitHub's merge d3ec27806cee229a9578f4163c1ddff4622877ad, with main
663002b03ccb92abee2ece934d69116cb4bf3415. All 233 Python contract tests,
source documentation gates, frozen installation, Node tests, Astro build and
type checking, content gate, and production artifact verification passed before
the browser step.

The failure classes overlap:

| Finding | Cases containing the finding | Repair |
| --- | ---: | --- |
| Search button visible label missing from accessible name | 66 | Reference the localized label and visible platform shortcut together |
| Insufficient light-theme accent contrast | 14 | Change the accent from #0a7c66 to #086b57 |
| Scrollable tables without keyboard access | 5 | Render semantic tables with tabindex=0 and a visible focus outline |
| Search test expects results after Escape clears the query | 2 | Retype and recheck the query after keyboard reopening |

No complete resource-budget pass is inferred from the absence of budget errors:
those assertions occur after axe and were not reached in failing cases.

## Bounded repair

The search wrapper reuses the existing version-aware component from
starlight-versions. It preserves localized text, the Ctrl/Command shortcut,
version filters, the archive notice and modal behavior. Its explicit
aria-labelledby relationship includes both existing text nodes; it does not hide
the visible shortcut or exclude the button from accessibility checks.

The table transform runs in the Markdown processor and preserves semantic table
markup. It reaches Markdown and MDX. The existing current-page browser cases now
also focus overflowing tables and verify that an arrow key changes their scroll
position with a visible focus indicator. This checks native scrolling after
programmatic focus; it is not described as a complete Tab-order test.

The pinned Pagefind input handler clears its value on Escape. The revised search
journey explicitly checks that empty state, types the original query again,
checks the same first result, and then uses Tab and Enter to navigate.

All 142 cases, axe tags, zero-retry requirements and resource limits remain.
No new dependency or lockfile change is required.

## Verification after the repair

On the restored local workspace, Node 26.13.1 and pnpm 12.18.0 ran:

- All ten existing Node tests: passed.
- Astro build: 202 HTML pages; internal link validation passed.
- Astro check: four files, zero errors, warnings or hints.
- Production artifact verification: 11,699 local links, zero errors;
  335 external URLs explicitly unverified.
- Scientific review consistency/freshness: all 27 inputs passed.
- Browser discovery: all 142 cases retained.
- Built-HTML inspection: focusable tables present in ordinary Markdown and the
  causal/scenario MDX tutorials; current and archived search version attributes
  remain distinct.
- Diff whitespace check: passed.

The build retains the versioning plugin's advisory about a custom Search
component. The custom component manually renders VersionSearch as that advisory
requires. This advisory is separate from Astro check's zero-warning result.

The repaired working tree is based on the actual hosted merge tree
4f6d2156e6f3065a68cb421fb47eae57fa27d155. The sole accepted-main change relative
to the prior branch is the Renovate action version update from PR #612.
The repair commit integrates that main parent while preserving branch history.

Actual browser execution and fresh native-language qualification must pass on
the new revision. The local checks above do not replace them. The earlier
local-evidence record retains its original source identity; its measurements
are not relabeled as results for this repair.
