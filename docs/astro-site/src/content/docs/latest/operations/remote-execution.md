---
title: Remote Execution
description: Remote service boundary for kernel requests and responses.
slug: latest/operations/remote-execution
---

# Remote Execution

Remote execution is a hosted-service boundary around the functional kernel and does not introduce a second API family.

Remote requests and responses preserve the kernel schema plus context metadata for provenance and observability.

Eligible operations currently include:

* `discover_models`
* `predict_model`
* `simulate_model`
* `summarize_model`
* `diagnose_model`

`fit_model` remains local-only by default because it can be long-running and sensitive.

Migration source:

* `docs/source/remote_execution.rst`
