# Specification: Rust Core Benchmarking and Profiling Tooling

## Overview

Add a Rust-side benchmarking and profiling toolchain so native kernel paths can
be measured and compared independently of the Python bridge. The goal is to
make benchmark gates and profiling evidence first-class inputs to future Rust
core promotion decisions.

## Functional Requirements

1. Add a Rust benchmark harness for the native kernel paths that have already
   moved into the Rust binding.
2. Provide a profiling workflow for Rust hot paths so performance regressions
   can be investigated without relying on Python-only tools.
3. Document which Rust operations are benchmarked, how to run the benchmarks,
   and how to capture profiles.
4. Keep the benchmark and profiling surface focused on native Rust execution
   rather than duplicating the existing Python benchmark corpus.
5. Leave mutation testing as a lower-priority future consideration rather than
   a mandatory Rust-side requirement.

## Non-Functional Requirements

1. Benchmark and profiling tooling must be reproducible in CI or local
   development with minimal setup.
2. The tooling must stay scoped to performance validation and not become a
   second public API.
3. The new tooling must preserve the stable kernel contract and not alter
   runtime semantics.

## Acceptance Criteria

1. The Rust crate has a benchmark harness suitable for native kernel paths.
2. The Rust docs explain how to run benchmarks and profiling for the native
   Rust execution path.
3. The project tech stack explicitly records the chosen Rust benchmark and
   profiling tools.
4. The recommendation to add Rust benchmarking/profiling is visible in the
   project roadmap or equivalent governance docs.

## Out of Scope

1. Porting the Python mutation-testing stack one-for-one into Rust.
2. Replacing the Python benchmark corpus or Scalene-based profiling workflow.
3. Changing the kernel schema or public API semantics.
