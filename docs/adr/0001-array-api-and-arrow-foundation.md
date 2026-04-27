# ADR 0001: Adopt Array API Semantics and Arrow-Compatible Interchange as the Long-Lived Foundation

- Status: Accepted
- Date: 2026-04-16

## Context

`innovate` is expected to grow beyond a Python-only library into a platform with a functional kernel, stable plugin boundaries, and bindings for R, Julia, TypeScript, Go, and Rust.

The project therefore needs a foundation that is:

- portable across numerical backends
- durable across language bindings
- not coupled to Python class internals
- not coupled to any one accelerator runtime

Official upstream documentation points toward two standards-friendly layers that fit this need:

- NumPy's main namespace is compatible with the Python Array API standard, providing a portable numerical target for downstream libraries.
- Arrow's C Data Interface freezes the C ABI once the specification is supported in an official Arrow release, making it suitable as a long-lived interchange layer.

## Decision

`innovate` will adopt the following foundation:

1. Public numerical semantics will target Array API-compatible behavior where practical.
2. Public structured and tabular interchange will use Arrow-compatible schemas and encodings.
3. The Python OO API will remain important, but it will sit on top of a language-neutral execution and interchange layer rather than define it.

## Consequences

### Positive

- The project can support multiple backends without changing the durable public contract.
- Future language bindings can target documented schemas rather than Python internals.
- The architecture aligns with the broader scientific-Python portability direction.

### Negative

- More upfront schema design and compatibility work is required.
- Some current implementation details will need refactoring to avoid backend-specific assumptions.
- The project must invest in validation and round-trip testing for the interchange layer.

## Alternatives Considered

### Use XLA or JAX export artifacts as the public contract

Rejected because this would tie the public surface to accelerator-specific lowering behavior and compatibility windows rather than to a stable, general-purpose modeling contract.

### Keep Python objects as the only stable boundary

Rejected because it does not support the project's interoperability and multi-language goals.

### Use ad hoc JSON-only interchange

Rejected because it is too weak for typed columnar data and would duplicate functionality already standardized by Arrow.

## References

- NumPy Array API standard compatibility: https://numpy.org/doc/2.3/reference/array_api.html
- SciPy support for the array API standard: https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
- Arrow C Data Interface: https://arrow.apache.org/docs/dev/format/CDataInterface.html
