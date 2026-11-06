---
title: Innovate
description: Current development API and feature migration status
---

Innovate 0.7 is a breaking development line. Rust owns computation and scientific
validation. Python, Julia and R are the supported bindings; Polars owns tables.
The installed Python package requires Polars and its native extension. There is
no Python numerical fallback, pandas or xarray runtime.

## Current scope

Bounded native Bass, Logistic, Gompertz, Fisher–Pry and one-generation Norton–Bass
operations are exposed. Advanced models, covariates, events, uncertainty and
native Kairos simulation remain migration obligations. A development test pass
is not full scientific acceptance. Release is blocked until the feature ledger
and installed/platform qualification gates close.

## Python

```python
import polars as pl
from innovate import predict, fit, capabilities

frame = pl.DataFrame({"time": [0.0, 1.0, 2.0]})
result = predict("logistic", frame, {"L": 100.0, "k": 1.0, "x0": 1.0})
print(result)
```

Inputs require finite, strictly increasing time. Nulls, duplicate times and
invalid domains are rejected in Rust. IPC currently copies; LazyFrames collect
at the execution boundary. Fit data additionally needs an `observed` column.

## R and Julia

Both call the Rust library through the C ABI and need no Python interpreter.
Build with `cargo build --locked -p innovate-ffi`; set `INNOVATE_NATIVE_LIBRARY`
to the resulting shared library. Julia uses JSON 1.13.4 and R uses jsonlite 2.0.0.
Native distribution bundles and tabular IPC wrappers remain open qualification.

## Scientific contract

Schema 2.0 makes Gompertz explicit: `a * exp(-b * exp(-c*t))`. The previous
implementation ignored `b`. Unimplemented significance tests return null with
`not_implemented`. Synthetic recovery checks do not establish identifiability,
calibrated intervals or all-model parity. The full pre-migration source/test
inventory remains recoverable from Git and is tracked by the active Conductor plan.

## Ecosystem alignment

Voiage owns economic/VOI calculations and consumes Polars/Arrow scenario tables.
Kairos owns scheduling, DES/ABM state and deterministic replay. Innovate retains
diffusion fitting and diagnostics. Do not copy the upstream scheduler or assume
its optional languages, experimental backends or hosted proofs apply here.
