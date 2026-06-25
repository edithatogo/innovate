---
title: Runtime Logging
description: Runtime observability and runtime diagnostic logging policy.
---

# Runtime Logging and Instrumentation

The repository does not use a custom logging framework. Runtime code should use
standard logging primitives and structured error payloads rather than ad hoc
`print` calls.

## Guidance

- Library modules should create module-level loggers and emit context-rich
  warnings or errors when a failure needs operator attention.
- Bridge scripts should keep stdout machine-readable and reserve stderr for
  diagnostics or unrecoverable failures.
- Tests, examples, and intentionally human-facing scripts may still use
  `print`.
- Rust-native runtime observability should move toward structured
  instrumentation rather than text-only debugging output as the core expands.

## Scope

This page is governance, not a public API. It describes the repo's default
observability stance so Python and Rust work can stay consistent while the core
continues to move native.

Canonical source:

- `docs/astro-site/src/content/docs/maintainers/runtime-logging.md`
