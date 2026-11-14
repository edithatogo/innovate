---
title: Native bindings
description: Python, Julia and R over the Innovate Rust core
---

Only Python, Julia and R bindings are maintained for the 0.7 development line.
Rust owns computation; it is the core library. Python uses PyO3 and Polars/Arrow
IPC. Julia and R currently call the versioned JSON contract through the C ABI.
Native library bundles and complete tabular adapters remain under qualification.
See the [native migration guide](/innovate/native-migration/) for current scope.

## Current API documentation

- [Python reference](/innovate/api/python/): generated from the maintained binding
  and checked against the installed native extension.
- [Julia reference](/innovate/api/generated/julia/):
  source-derived docstrings for the exported API, including native library discovery and
  JSON precision boundaries.
- [R reference](/innovate/api/generated/r/):
  usage signatures, arguments and return contracts for the exported functions.

These references identify the maintained native implementation and link to their
source documentation. The pre-0.7 Julia page describes a different bridge.
Structural documentation checks and the language-specific installed-package
tests provide separate evidence; neither substitutes for scientific or platform
qualification.

TypeScript, Go and C# packages are removed. Historical Conductor and 0.6 reference
documentation record the earlier implementation and do not establish current support.
