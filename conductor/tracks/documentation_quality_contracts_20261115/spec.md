# Specification: documentation as an executable product contract

## Historical direction and starting point — 15 November 2026

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

## Planning baseline — 16 November 2026

Dylan requested assessment and incorporation of marimo notebooks, maximal useful
Astro features, and adequate beginner-to-advanced tutorial coverage with public
datasets. The [learning programme](learning-plan-20261116.md) and its linked
coverage, design and dataset records are the current specification extension.
The earlier Sphinx and removed-quickstart observations above describe the starting
point; the current site is Astro with source-bound native examples.

At inspected main 06b16671b654b4e51f158bf7d605c4286f156755, the maintained content
contract has eight tutorial pages using six distinct example sources, ten native
Python examples overall, and no registered public-dataset case. Notebook support,
the expanded curriculum and the proposed Astro enhancements are not implemented
by this planning revision. Existing documentation and integration ownership remain.

## Authorised implementation extension — 16 November 2026

The maintainer subsequently authorised implementation, maximal useful docstrings
and language conventions, updated executable contracts, removal of obsolete
content, and alignment with the open-science communities identified in issues.
The [implementation record](implementation-20261116.md) supersedes planning-only
status; the original coverage targets and unfinished acceptance remain binding.

The implementation is based on main
`69b674814513e09a683279c679b1584af8bb4214`. It strengthens Python, Rust, Julia and
R documentation in each language's existing conventions; derives Astro discovery
from current native/content records; adds an offline WDI/ITU analysis and optional
marimo adapter; promotes four advanced native examples; and removes obsolete
documentation sources with immutable history and checked migration notices.

Documentation conventions are repository policy informed by community guidance.
The issue-linked standards contract distinguishes current commitments, adopted
guidance and historical aspirations. It must not infer JOSS acceptance, registry
identifiers, community membership or scientific validity from structural checks.

The optional locked notebook project is confined to `notebooks/`, as recorded in
[ADR 0008](../../../docs/adr/0008-current-documentation-and-learning-resources.md).
Shared root dependencies, CI, registry and publication remain with the existing
integrator. Runtime scientific algorithms and public native interfaces are outside
this documentation increment. All maintained examples still use the public
Rust-backed interface and Polars.

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
   source. Synthetic examples assert independent results and failures. Public-data
   tutorials are additionally required by the 16 November extension, with explicit
   dataset admission, attribution, frozen inputs and study-specific interpretation;
   they supplement rather than replace independent synthetic correctness checks.
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

The initial implementation repairs the entry path, completes maintained binding
documentation, qualifies current Python examples and establishes enforceable
discovery, source-derived reference output and regression gates. The earlier local
record qualified Rust documentation and two examples while Julia/R execution
remained open. At the 16 November inspected revision, hosted run 43804385173 adds
native documentation execution evidence, including Julia/R, within that job's
scope. Historical content, wider scientific appraisal, browser qualification,
public artifact identity and full platform acceptance remain separate. No runtime
scientific behavior changes are authorised by this documentation scope.

For the extended scope, completeness additionally requires the declared model,
operation, task and language coverage described in the learning programme. Every
notebook is inventoried; each promoted current notebook is executed against the
intended native installation.
An HTML export, a new framework, a public dataset, a model-family checkbox or a
source hash alone cannot establish that requirement. Browser-native execution,
new hosting, native scientific changes and external empirical validation retain
their separate prerequisites and owners.
