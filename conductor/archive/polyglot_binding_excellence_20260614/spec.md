# Polyglot Binding Excellence

## Overview

The current bindings have publication and smoke evidence, but maturity requires idiomatic SDK surfaces, conformance suites, examples, version alignment, and release-grade package checks across R, Julia, TypeScript, Go, C#, Rust, and Python.

## Functional Requirements

1. Define a shared binding conformance suite that validates capability metadata, canonical operations, serialization, errors, examples, and version identity.
2. Add or refresh idiomatic examples for each language binding.
3. Add package-level checks for each language: R checks, Julia package tests, TypeScript type/build tests, Go tests, NuGet pack/test checks, and Rust cargo checks.
4. Add cross-language golden fixtures for operations and payload round trips.
5. Add capability documentation that identifies each binding as supported, experimental, or explicitly limited.
6. Add release artifacts showing binding parity and package-manager readiness.

## Non-Functional Requirements

1. Binding behavior must be contract-first and not duplicate divergent model logic.
2. Failures must be actionable and language-specific.
3. Conformance data must be machine-readable and consumed by docs.
4. Optional dependencies should be isolated to language-specific workflows.

## Acceptance Criteria

1. Every supported binding has a conformance status and package-check evidence.
2. Golden fixtures verify parity for all promoted canonical operations.
3. Documentation no longer relies on generic binding claims.
4. CI checks prevent version and capability drift.
5. Examples run or are validated by language-native tooling.

## Required Operational Cadence

Every task requires a task implementation commit, a separate plan-status commit, phase review with `conductor-review`, push plus GitHub Actions monitoring, final track review, final push, and passing GitHub Actions before archive.

## Out of Scope

1. Publishing new package versions without maintainer approval.
2. Adding unsupported language bindings not already present in the product roadmap.
3. Making every advanced feature available in every language before the binding contract supports it.
