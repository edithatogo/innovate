---
title: Binding Conformance CI
description: Shared binding conformance gate and local fallback policy.
slug: latest/maintainers/binding-conformance
---

# Binding conformance CI

The binding conformance gate validates shared evidence before language-native
package checks are interpreted as release evidence. The workflow runs the
machine-readable conformance inventory, golden fixture, and hardening tests,
then uploads the evidence JSON payloads as a CI artifact for maintainer review.

## Local Fallback

Run the shared contract gate locally with:

```bash
uv run pytest \
  tests/unit/test_polyglot_binding_conformance.py \
  tests/unit/test_polyglot_binding_golden_fixtures.py \
  tests/unit/test_polyglot_binding_hardening.py \
  -q
```

If a language-native package checks toolchain is unavailable, use this shared
contract gate as the fallback evidence and record the missing toolchain in the
release notes or submission checklist. The fallback does not replace native
package checks such as `cargo test`, `npm test`, `R CMD check`, `Pkg.test()`,
`go test ./...`, or `dotnet test` before publication.

## Uploaded Artifacts

The `Binding Conformance` workflow uploads:

- `docs/source/_static/binding_conformance_inventory.json`
- `docs/source/_static/binding_golden_fixtures.json`
- `docs/source/_static/binding_hardening_evidence.json`

Canonical source:

- `docs/astro-site/src/content/docs/maintainers/binding-conformance.md`
