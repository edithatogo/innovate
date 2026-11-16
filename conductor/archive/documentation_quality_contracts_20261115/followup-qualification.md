# Follow-up documentation qualification

This increment continues [PR #610](https://github.com/edithatogo/innovate/pull/610)
and [PR #611](https://github.com/edithatogo/innovate/pull/611) in
[PR #613](https://github.com/edithatogo/innovate/pull/613).
Receiving main was `384ed3b0f42d967f520dc66ef536d8fa0d92df7d`.
The local qualification below ran on the integrated working tree based on
`b14c05077e83223fce13353b958a323ae02ea82d`, including the aggregate repair,
review-checker formatting and PR concurrency fix committed with this record.

[followup-local-evidence.json](followup-local-evidence.json) records exact
fingerprints for nineteen relevant source, policy, runner and test inputs.
Results in earlier track documents retain their original source identity.

## What changed

Twelve pages now describe the current native API: core concepts, the kernel and
Arrow contracts, fitting, forecasting, backend selection, the tutorial index,
Norton–Bass, Bayesian inference, causal contrasts, Arrow results and dataset
provenance. Six tutorial pages import the complete registered example source and
its declared expected output. Earlier Markdown counterparts were removed while
retaining the same public routes.

Current material now appears under current guide, tutorial and API navigation.
All 61 remaining historical pages at current paths retain their visible
pre-0.7 notice and declare `pagefind: false`. The historical map accounts for
every route exactly once: eight current task replacements, 37 partial
replacements, three unsupported interfaces and thirteen tasks without a
qualified replacement. The 80-page versioned archive is preserved.

The search contract rejects historical pages that re-enter bundled Pagefind,
and current pages that opt out. Its regressions distinguish real frontmatter
from prose, comments, code fences, quoted booleans and duplicate keys.
An external Algolia index remains a separate qualification obligation.

Scientific explanations now distinguish valid inputs from suitable scientific
models, levels from rates, stored draws from effective sample size, regression
tolerances from empirical diagnostic policy, and scenario arithmetic from causal
identification. The scoped review records fifteen pages, 27 source fingerprints,
twelve resolved findings, 21 worked reader tasks, ten primary or official source
records and twelve independent agent page reviews. No human participants or
reader trials are claimed.

The new review gate checks source freshness, imported-source links, reference
identifiers, expected answers and bounded completion claims. It rejects duplicate
JSON keys and unsupported human-validation claims. Seven tests cover its material
failure modes, including the nested invocation directory used by CI.

## Remaining native aggregation defect

The [PR #611 documentation run](https://github.com/edithatogo/innovate/actions/runs/43647879490)
passed its build and all individual native runners, then failed aggregation in
[job 132614368807](https://github.com/edithatogo/innovate/actions/runs/43647879490/job/132614368807).

Julia reported 24 exported docstrings, seven examples and nine observed
assertions. R reported 24 exports, fourteen manuals, six examples and ten observed
assertions. Both receipts were rejected with:

> Inventory example source hash is invalid.

PR #611 had repaired the outer source-map hash check. A later comparison still
compared a bare example digest with a source-map value carrying the exact
`sha256:` label. The new repair normalizes only that source-map value for this
comparison and validates both resulting digests. It preserves the original
inventory and receipt maps, their exact equality, native binary checks and all
other receipt requirements.

The new full-receipt regression reproduced the failure for both languages before
the repair. Five new regressions cover the real mixed encoding, malformed and
changed example digests, unsupported source-map prefixes, exact receipt-map
equality and changed source bytes. The two affected modules then passed all
78 tests on Python 3.15.2.

The Julia runner also retains its assertion counter locally to avoid reading a
newly defined module binding from an older world. The documentation job now uses
`--depwarn=error`. The inventory records the changed runner hash. Historical
Julia successes do not qualify this new runner; a fresh hosted execution remains
required.

## Local evidence

The recovered workspace uses Python **3.15.2**, Node **26.13.1** and pnpm
**12.18.0**. Astro resolves to **7.4.11** and Starlight to **0.42.11**.

| Check | Observed result |
| --- | --- |
| Frozen dependency installation | Passed; all 567 lock entries passed the existing supply-chain policy |
| Python documentation contract suites | 233 tests passed; zero failures, errors or skips |
| Node transforms and browser-contract helpers | Ten tests passed; zero failures or skips |
| Python source docstrings | Strict gate passed with no gaps |
| Installed native-extension docstrings | All ten public callables passed; installed Python source matches the checkout |
| Julia/R source inventory | Exact generator check passed |
| Generated API reference | 34 pages generated; freshness check passed |
| Current content | 60 current pages, ten example files, zero unbound Python blocks, zero violations and zero accepted exceptions |
| Python execution | All ten example files and eighteen API doctest statements passed with observed assertions and expected output |
| Review record | All 27 declared inputs passed source-freshness and consistency checks |
| Astro build | 202 HTML pages built; internal link validator passed |
| Astro type checking | Zero errors, warnings or hints |
| Production artifact verifier | 11,699 local references checked; zero errors; 335 external URLs explicitly unverified |
| Static resource budget | 12,524,286 artifact bytes across 377 files, below the declared limits |
| Browser discovery | 142 cases enumerated across desktop and 390-pixel mobile Chromium projects |
| Scoped Ruff / formatting | Passed |
| Workflow syntax and diff hygiene | `actionlint` and `git diff --check` passed |

Coverage across the seven Python documentation checkers is **84.02% of
statements**, **77.48% of branches**, and **81.87% combined**. These are measured
values, not a claim that every error path is exercised.

The full local language command deliberately returned nonzero: Julia and R
runtimes and fresh native receipts were unavailable locally. Its report retains
passing Python results while leaving complete language qualification false.
The installed native binary was reused from the task environment; the runtime
receipt does not attest that this binary was rebuilt from this checkout.

The local workflow-security checker was not rerun because it was unavailable
in the recovered tooling environment. Hosted security checks remain separate.

## Browser qualification

The configured suite checks every current page in Chromium at desktop and
mobile sizes, plus six dark-theme entry pages. Separate cases cover keyboard
skip navigation and focus, Pagefind searches, version switching, current sidebar
groups, horizontal overflow, internal resource failures and deterministic
resource budgets.

Search results must belong to the current inventory. Sidebar assertions require
the migrated routes in their current groups and reject classified historical
routes from primary navigation. The reporter rejects incomplete, filtered,
skipped, duplicated and retried runs. Receipt validation also rejects invalid,
oversized or substituted byte/request measurements and limits.

The browser reports capture source/artifact identities before and after the run,
browser versions, served-resource hashes, axe findings and per-case observations.
They preserve incomplete axe findings for manual investigation.

**Discovery and helper tests are not browser execution.** No local Chromium run,
manual accessibility review, human reader trial, WCAG conformance certification,
other-browser qualification or real-user performance measurement is claimed
here. The hosted browser step must pass on the final revision.

## Hosted execution and publication

Superseded PR documentation runs may now be cancelled within their existing
PR-specific concurrency group. Main push/dispatch publication retains its
ordered queue. Both build and native-example dependencies, frozen installation,
explicit receipt allowlists and the Pages activation gate remain required.

The temporary lock-recovery workflow was removed after the original local lock
was recovered and independently hash-checked. No non-frozen installation remains
in the final documentation workflow.

At the live-site inspection during this increment, the public URL still
identified Sphinx 8.2.3 and served the earlier API. Publishing and verifying the
qualified Astro artifact requires the existing coordinator-owned integration
and Pages activation process. No deployment setting was changed.

Full scientific release obligations, qualification of retained historical
content, human reader testing and live publication remain explicitly separate
from the passing local checks above. Latest exact-head hosted results belong in
the PR status and workflow artifacts; they must not be inferred from this record.

