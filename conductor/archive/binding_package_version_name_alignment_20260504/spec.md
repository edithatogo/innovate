# Specification: Binding Package Version and Language-Suffix Name Alignment

## Overview

All language binding packages should share the highest current release version
and use the `innovate` core name with language suffixes following package
manager conventions.

## Functional Requirements

- Align binding package versions with the primary Python package version
  `0.5.0`.
- Rename package metadata to the requested language-suffix policy where the
  target registry supports it.
- Preserve registry-valid alternatives where dotted names are not accepted:
  Rust publishes as `innovate-rs` for the user-facing `innovate.rs` suffix, and
  Julia keeps the valid `Innovate` package/module name for the user-facing
  `innovate.jl` suffix.
- Update publication documentation and CI/package validation expectations.
- Fix package metadata found during dry-run checks.

## Acceptance Criteria

- Python, TypeScript, Rust, R, Julia, and C# metadata all report version
  `0.5.0`.
- TypeScript uses `innovate.ts`, R uses `innovate.R`, C# uses `innovate.cs`,
  Rust uses registry-valid `innovate-rs`, and Julia uses registry-valid
  `Innovate`.
- Publication docs state the naming policy and version alignment.
- Focused package metadata tests pass.
- Dry-run package checks pass for TypeScript, Rust, R, Julia, Go, and local
  C# net10.0 packaging.

## Out of Scope

- Live publication to external registries.
- Registry credentials or package ownership setup.
- Renaming C# project folders or namespaces.
