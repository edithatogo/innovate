# ADR 0004: Stabilize the Core API, Expose Thin Bindings, and Evolve Toward a Rust Core

- Status: Accepted
- Date: 2026-04-28

## Context

`innovate` is moving from a Python-only modeling library toward a portable modeling platform. The project already has or is planning language surfaces for Python, R, Rust, Julia, TypeScript, Go, and C#. It also has a functional kernel contract, capability metadata, and Arrow-oriented interchange work that make a language-neutral architecture practical.

The project needs a clear ordering principle so future tracks do not fragment the implementation:

1. Stabilize the core API and kernel contract first.
2. Surface that same contract through thin language bindings.
3. Move performance-critical and portability-critical core execution toward Rust over time.

Without this decision, each binding could drift into its own behavior, and Rust could be treated as only another client binding rather than the strategic core runtime direction.

## Decision

`innovate` will adopt a contract-first, Rust-core trajectory:

1. The canonical Python public API, capability registry, stable schemas, and functional kernel operations define the initial core contract.
2. R, Rust, Julia, TypeScript, Go, C#, and future bindings should expose the same contract without independently reimplementing model logic.
3. Python remains the primary ergonomic research and documentation interface.
4. Rust is the preferred long-term implementation language for robust, efficient, portable kernel execution.
5. Rust components may be introduced incrementally behind the existing kernel contract, but they must pass parity and schema-compatibility tests before becoming the default execution path.
6. Binding-specific APIs may feel idiomatic in their host language, but their semantics must remain traceable to the shared core contract.

## Consequences

### Positive

- Users get consistent model behavior across languages.
- Bindings stay smaller, easier to test, and less likely to drift.
- The project has a credible path to a fast, portable core without abandoning Python usability.
- Rust can improve correctness boundaries, packaging portability, and execution performance while still serving the same public API.

### Negative

- Core schema and compatibility work must precede broad binding expansion.
- Rust migration requires parity tests, benchmark gates, and careful packaging design.
- Some language-specific convenience features may need to wait until the shared contract can support them cleanly.

## Alternatives Considered

### Treat each language binding as an independent implementation

Rejected because it would multiply maintenance cost, create behavioral drift, and make correctness difficult to verify across languages.

### Keep Python as the only durable core indefinitely

Rejected because Python remains excellent for research ergonomics but is not the best long-term foundation for portable, high-performance shared execution.

### Make Rust the immediate primary API and de-emphasize Python

Rejected because Python is the current product surface, documentation center, and expected research interface. Rust should strengthen the core without forcing users away from Python.

## Follow-Up Work

1. Add C# as a planned binding track after the current API and binding contracts are stable.
2. Define Rust parity tests against the Python reference semantics.
3. Identify the first kernel operations suitable for Rust implementation behind the existing contract.
4. Add benchmark gates that compare Python reference behavior with Rust-backed execution.
