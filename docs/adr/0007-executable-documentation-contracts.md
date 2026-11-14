# ADR 0007: Executable documentation contracts for Astro

Accepted by the maintainer's Astro and documentation-quality direction on
15 November 2026. Implementation and full acceptance are tracked separately in
`documentation_quality_contracts_20261115`.

Astro/Starlight remains the sole active documentation site. Inspection found a
working site build with a stale current quickstart, incomplete binding docstrings,
and a vendored API plugin that constructed pages in memory but never persisted
them into the content collection. Historical verification JSON could not establish
the quality of the current artifact. The public site still served Sphinx.

The build now generates the Python reference from syntax and documented source
exports before Astro loads content. A deterministic manifest binds generated
pages to source hashes and public symbols. No native package import is required
for extraction; actual example execution uses the installed native package in a
separate gate. A generator failure, missing export or altered generated file fails.

Page and example discovery precedes coverage. Tutorial code and expected output
are imported from checked sources. Execution requires observed assertions and
expected results, not a `classified` status. Python structural completeness and
Rust compiler documentation checks remain distinct from scientific and editorial
review. Native PyO3 runtime docstrings are checked separately from Rust visibility.

The contract supports an explicit source-bound no-regression baseline for retained
debt. This is an incremental integration mechanism, never a completion claim.
Strict whole-site acceptance retains historical pages, unsupported tutorials,
scientific review, accessibility, browser behavior and live publication obligations.

Rendered artifact checks inspect this build's HTML, routes, local fragments,
canonical/base paths, search index, sitemap, semantics and size. Source and output
fingerprints are retained as contemporaneous evidence; they alone are not a signed
build attestation. The existing Pages activation boundary and coordinator merge
ownership remain in place.

Alternatives considered were another site migration, manually maintained API
pages, and trusting the existing plugin's generation log. Each would preserve or
increase drift. Repairing the build boundary and qualifying actual output provides
a smaller change while making the missing evidence visible.
