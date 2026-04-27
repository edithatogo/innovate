# ADR 0002: JAX Is an Optional Accelerator Backend, Not the Public ABI

- Status: Accepted
- Date: 2026-04-16

## Context

JAX is already present in `innovate` for accelerator-backed fitting and advanced inference. It is valuable for JIT compilation, vectorization, and probabilistic tooling. However, JAX's own documentation makes clear that:

- JAX is still in a `0.x` phase and uses effort-based versioning.
- JAX follows a three-month deprecation policy for breaking changes.
- JAX export compatibility is version-bounded rather than evergreen.
- Only a subset of custom calls are guaranteed stable for export compatibility.

These properties make JAX an excellent accelerator and research backend, but a poor choice for the library's durable public ABI.

## Decision

`innovate` will treat JAX as an optional accelerator backend.

This means:

1. JAX-backed fitters, simulation paths, and inference features remain supported where they provide clear value.
2. Public contracts for the kernel, bindings, and interchange must not rely on XLA lowering details, jaxlib internals, or JAX export artifacts.
3. The NumPy/SciPy path remains the reference implementation for correctness and portability.

## Consequences

### Positive

- The project keeps access to accelerator performance and modern probabilistic tooling.
- Users who do not need JAX keep a simpler and more stable base environment.
- The public contract is insulated from JAX-specific compatibility churn.

### Negative

- Dual-path testing and capability metadata become necessary.
- Some advanced features may remain experimental until their JAX-backed implementations are stabilized.
- Performance-critical code must be designed to degrade gracefully when JAX is absent.

## Alternatives Considered

### Make JAX/XLA the primary public runtime contract

Rejected because JAX compatibility guarantees are deliberately narrower than what `innovate` needs for a long-lived, multi-language public surface.

### Remove JAX entirely

Rejected because it would sacrifice useful acceleration and inference capability without solving the portability problem as cleanly as keeping JAX optional.

## References

- JAX API compatibility: https://docs.jax.dev/en/latest/api_compatibility.html
- JAX exporting and serialization: https://docs.jax.dev/en/latest/export/export.html
