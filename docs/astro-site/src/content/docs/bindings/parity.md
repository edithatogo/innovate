---
title: Binding parity
description: Evidence-backed parity status for supported language bindings.
---

# Binding parity

Innovate treats Python as the canonical reference surface and keeps Rust, R,
Julia, TypeScript, Go, and C# aligned through shared kernel contracts rather
than duplicated model logic.

| Language | Status | Evidence |
| --- | --- | --- |
| Python | Supported reference package | `binding_conformance_inventory.json` and package checks in CI |
| Rust | Supported native runtime and crate | `binding_hardening_evidence.json`, `binding_golden_fixtures.json`, and cargo package checks |
| R | Supported package surface | `binding_hardening_evidence.json`, R examples, and `R CMD check` evidence |
| Julia | Supported package surface | `binding_hardening_evidence.json`, package tests, and installed-package smoke evidence |
| TypeScript | Supported npm surface | `binding_hardening_evidence.json`, type checks, and npm dry-run pack evidence |
| Go | Supported module surface | `binding_hardening_evidence.json`, module path checks, and Go package tests |
| C# | Supported NuGet surface | `binding_hardening_evidence.json`, .NET test and pack evidence |

Machine-readable evidence:

- `docs/source/_static/binding_conformance_inventory.json`
- `docs/source/_static/binding_golden_fixtures.json`
- `docs/source/_static/binding_hardening_evidence.json`

CI evidence:

- The `Binding Conformance` workflow uploads the
  `binding-conformance-evidence` artifact for maintainer review.
- Language-native package checks remain the release gate for registry
  submission and publication readiness.
- Package-manager receipts and deferred maintainer handoffs are tracked in
  `registry_submission_receipts`.
