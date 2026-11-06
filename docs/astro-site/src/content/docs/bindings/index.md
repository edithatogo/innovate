---
title: Native bindings
description: Python, Julia and R over the Innovate Rust core
---

Only Python, Julia and R bindings are maintained for the 0.7 development line.
Rust owns computation; it is the core library. Python uses PyO3 and Polars/Arrow
IPC. Julia and R currently call the versioned JSON contract through the C ABI.
Native library bundles and complete tabular adapters remain under qualification.
See the [native migration guide](/innovate/native-migration/) for current scope.

TypeScript, Go and C# packages are removed. Historical Conductor and 0.6 reference
documentation record the earlier implementation and do not establish current support.
