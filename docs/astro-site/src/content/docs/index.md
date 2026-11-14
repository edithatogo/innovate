---
title: Innovate
description: Learn the current Rust and Polars diffusion API through executable examples and explicit scientific limits.
---

Innovate models innovation adoption, technology substitution and policy
diffusion. Its **0.7 development line** runs scientific computation in Rust,
uses Polars tables, and exposes Python, Julia and R bindings.

## Start with a checked result

The [native prediction tutorial](/innovate/user-guide/getting-started/) takes you
from an installed development checkout to a three-point Logistic prediction.
It checks an independent analytic answer, preserves row identity, and exercises
invalid-input handling. The code displayed in the tutorial is the file that
the example runner executes.

- [Install the development package](/innovate/user-guide/installation/)
- [Run the native prediction tutorial](/innovate/user-guide/getting-started/)
- [Read the current Python API](/innovate/api/python/)
- [Read the Julia API](/innovate/api/generated/julia/) or [R API](/innovate/api/generated/r/)
- [Check scientific conventions and supported capabilities](/innovate/native-migration/)

## Understand the development boundary

Sixteen model families have bounded native implementations. Coverage differs by
operation and model; use capability discovery and the native migration guide.
Unported capabilities fail explicitly. Full scientific, installed-artifact and
platform qualification remain release requirements.

Predictions require finite, strictly increasing time. Polars LazyFrames collect
at the native boundary, and Arrow IPC currently copies buffers. Synthetic
recovery or local numerical convergence does not establish empirical validity,
global identifiability or calibrated uncertainty.

## Know which documentation you are reading

The current entry path and generated binding references describe the development
checkout. Pages labelled **Pre-0.7 reference**, and the **0.6 Reference** version,
retain historical designs and APIs. They are not evidence that those features
are available in 0.7.

The [documentation quality contract](/innovate/maintainers/documentation-quality/)
records inventory, source-bound examples, remaining content debt and the separate
requirements for scientific review and usability. A working Astro build alone
does not establish completion.

## Ecosystem boundaries

Kairos owns event scheduling and simulation state. Voiage consumes adoption and
scenario tables for economic and value-of-information work. Innovate owns its
diffusion semantics, fitting and diagnostics; other projects' acceptance does not
automatically qualify this library.
