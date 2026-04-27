# Specification: Binding Publication and Multi-Language CI

## Overview

Ensure each language binding is treated as a packageable product surface with language-native CI and a publication path to the relevant package manager or registry.

## Functional Requirements

1. Document the publication target for each binding: npm for TypeScript, crates.io for Rust, R-universe/CRAN for R, Julia General for Julia, versioned Go modules for Go, and NuGet for planned C#.
2. Add CI jobs for implemented language bindings so Rust, TypeScript, Go, Julia, and R code is validated on pull requests and pushes.
3. Add a release-gated binding publication workflow that performs package checks and only publishes with explicit release/manual intent and registry secrets.
4. Ensure TypeScript package metadata does not block npm publication.

## Non-Functional Requirements

1. Publication must not bypass schema compatibility or binding tests.
2. Registry publishing must be gated and must not run automatically on ordinary pull requests.
3. C# remains planned until package scaffolding exists.

## Acceptance Criteria

1. Binding publication docs list the relevant registry or package manager for every target language.
2. Main CI contains binding jobs for Rust, TypeScript, Go, Julia, and R.
3. A release/manual workflow exists for binding package publication checks and gated publishing.
4. Tests guard the publication docs and CI workflow coverage.

## Out of Scope

1. Publishing packages during this track.
2. Creating the C# package implementation.
3. Completing CRAN, Julia General, or NuGet onboarding.
