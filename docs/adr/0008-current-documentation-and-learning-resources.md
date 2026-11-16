# ADR 0008: Current documentation and executable learning resources

Date: 2026-11-16
Status: Accepted for the documentation implementation; external publication and
scientific acceptance remain separate.

## Context

The Astro site mixed current native guidance with 61 historical pages and an
80-page versioned archive. Sphinx instructions and old notebooks also remained
in the checkout. They described removed interfaces and could be mistaken for
current examples. The user authorized their deprecation and removal, expanded
tutorial coverage, marimo evaluation, native language conventions and Conductor
reconciliation.

## Decision

1. Maintain one current Astro/Starlight documentation line. Retire obsolete prose
   and notebooks from the tree, retain exact source identities in a machine
   readable contract, and render small non-searchable migration notices at
   former web URLs with Starlight's supported custom-page component. Preserve
   historical ADRs and acceptance evidence in place.
2. Treat docstrings, rustdoc, Julia documentation and R roxygen/help sources as
   canonical API documentation. Enforce language-specific conventions and
   reject ambiguous, missing or stale reference material. Structural, installed
   runtime and scientific validation remain separate gates.
3. Build the model atlas and tutorial catalogue from discovery and the existing
   content inventory with typed Astro content collections. Display coverage
   gaps explicitly. Do not infer operation/variant combinations or scientific
   support from a model name alone.
4. Use marimo as optional native Python learning tooling. A notebook must share
   the ordinary example's computation, execute in the supported environment,
   and provide a deterministic headless path. Static exports and browser/WASM
   execution require their own qualification; an HTML file alone is not success.
5. Map adopted community documentation requirements to sources, evidence and
   pending decisions. Software Heritage, RRID and JOSS issue commitments remain
   distinct from pyOpenSci/rOpenSci guidance and historical community aspirations.

## Consequences

Git history becomes the authority for obsolete instructions; the live site
provides current guidance and migration routes. This reduces contradictory
instructions without erasing scientific obligations. New catalogue entries and
source metadata cannot silently claim executed or scientifically qualified
coverage. Optional notebook dependencies stay outside the core runtime.

Retirement, source/reference generation, standards mapping, current examples,
HTML output and browser behavior are checked independently. Remaining work,
runtime limitations and external review are recorded in the active Conductor
documentation track. The existing integration owner retains merge/publication
coordination and shared scientific/CI ownership.

## References

- [ADR 0006](0006-rust-polars-native-only.md)
- [ADR 0007](0007-executable-documentation-contracts.md)
- [Starlight custom pages](https://starlight.astro.build/guides/pages/#custom-pages)
- [marimo export formats](https://docs.marimo.io/guides/exporting/)
- [Research registry issues](https://github.com/edithatogo/innovate/issues/398)
