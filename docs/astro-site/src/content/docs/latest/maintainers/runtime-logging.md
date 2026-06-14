---
title: Runtime Logging
description: Runtime observability and runtime diagnostic logging policy.
slug: latest/maintainers/runtime-logging
---

# Runtime Logging and Instrumentation

The repository favors standard logging primitives and structured error payloads.

Guidance:

* Modules should emit context-rich warnings and errors for operator-relevant failures.
* Machine logs should remain machine-readable in automated paths.
* Print statements are reserved for scripts and intentionally human-facing examples.

Runtime observability guidance is expected to become more structured as native Rust execution expands.

Migration source:

* `docs/source/runtime_logging.rst`
