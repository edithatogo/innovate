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
- The migration record is preserved in the Sphinx docs until the Astro site
  fully replaces it.

Migration goals:

- keep canonical operations, model families, and payload shapes explicit;
- preserve parity and profiling evidence;
- retain stable fallback behavior for non-native slices during the parallel
  run.
