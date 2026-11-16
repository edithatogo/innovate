# Specification: documentation as an executable product contract

## Direction and observed starting point

The maintainer requested Astro, complete docstrings and exceptional documentation,
examples and tutorials on 15 November 2026 (Australia/Brisbane). At inspected main
`13bdcf10ccb3e1cd06da1df4d98fc5da831768db`, Astro already exists under
`docs/astro-site`. The public URL still serves Sphinx. The deployment workflow is
explicitly gated by `ENABLE_PAGES_ACTIONS_DEPLOY`; this work preserves that setting
and the integration coordinator's merge ownership. Other active PRs own numerical,
security, registry and ecosystem work.

The current quickstart uses removed Python APIs and pandas; the maintained API
uses Rust and Polars. The examples validator delegates to the scientific release
gate without executing examples. Historical JSON reports cannot qualify this build.

## Required outcomes

1. Preserve Astro/Starlight as the single active site; build and type-check the
   real workspace with its pinned lockfile and supported runtime.
2. Discover the complete maintained Python source/stub API, including undocumented
   symbols. Require useful docstrings and matching parameters, returns and direct
   errors. Use compiler/rustdoc checks for Rust rather than regex coverage claims.
   Inventory Julia/R public exports, signatures and argument documentation, and
   check runtime examples in their installed language environments.
3. Inventory all documentation pages and example files. Distinguish current,
   historical and generated surfaces explicitly. Report the denominator and gaps.
4. Tutorials specify learning outcomes, prerequisites, executable steps, expected
   results, interpretation and limitations. Display the actual checked example
   source. Examples use synthetic data and assert independent results and failures.
5. Separate strict completion from an explicitly bounded no-regression check.
   An accepted backlog never becomes a claim that the documentation is complete.
6. Verify the built HTML, canonical/base paths, links/fragments, search assets and
   sitemap from this build. Retain source hashes and test outcomes in CI artifacts.
7. Preserve scientific migration and release gates. Content presence, executable
   correctness, scientific validity, usability and live deployment are distinct.

## Acceptance and limits

Passing structural rules is necessary but cannot establish pedagogy or scientific
validity. The target additionally requires substantive technical review, independent
scientific references, accessible rendering, browser/keyboard/mobile/search checks,
and reader task testing. No `frontier` or `beyond-frontier` acceptance is inferred
from framework choice, generated prose, historical reports or passing headings.

The implementation repairs the current entry path, completes maintained binding
documentation, qualifies all current Python examples and establishes enforceable
discovery, source-derived reference output and regression gates. Rust compiler
documentation and its two examples have passed locally. Julia/R runtime evidence,
historical content, scientific appraisal, browser accessibility and verified live
publication stay open until actually completed. No runtime scientific behavior
changes are in scope.
