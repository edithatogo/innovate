---
title: Rust Core
description: Rust runtime migration roadmap and ownership closure.
---

# Rust Core

The Rust roadmap tracks the transition from thin binding execution to native
runtime ownership for promoted slices.

Current status:

- Rust owns the promoted native slices for discovery metadata, logistic,
  Fisher-Pry, Gompertz, Bass, and narrow Norton-Bass paths documented in the
  repository.
- Remaining bridge-backed slices stay explicit rather than implicit.
- Full Rust ownership is not claimed while any canonical operation, Python
  registry model family, or stable payload shape remains bridge-backed,
  Python-reference-owned, or explicitly promoted elsewhere.
- The migration record is mirrored in the Astro/Starlight site and preserved in
  legacy Sphinx source as archival evidence during cutover cleanup.

Migration goals:

- keep canonical operations, model families, and payload shapes explicit;
- preserve parity and profiling evidence;
- retain stable fallback behavior for non-native slices during the parallel
  run.

Active ownership tracks:

- `conductor/tracks/rust_native_operation_completion_20260614/`
- `conductor/tracks/rust_native_payload_model_coverage_20260614/`
