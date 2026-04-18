# Specification: Arrow Interchange and Schema Layer

## Overview

Define an Arrow-compatible interchange layer for `innovate` so kernel payloads, diagnostics, and future non-Python bindings can exchange typed columnar data through a durable, versioned contract rather than through Python object internals.

## Functional Requirements

1. Define canonical Arrow-compatible schemas for the tabular payloads used by the kernel, diagnostics, and model metadata.
2. Define conversion rules between the Python-facing pandas API and the Arrow interchange layer.
3. Implement validation and round-trip helpers for the supported interchange payloads.
4. Document versioning, compatibility, and error semantics for the interchange layer.
5. Provide examples showing how the interchange layer fits the functional kernel and future bindings.

## Non-Functional Requirements

1. The interchange layer must not depend on XLA, jaxlib internals, or Python object identity.
2. Schema evolution should be backward-compatible where practical and explicitly versioned when not.
3. The Python surface should remain readable and ergonomic even when Arrow is the underlying interchange boundary.
4. The design should permit low-copy or zero-copy paths where the participating libraries support them.

## Acceptance Criteria

1. A documented Arrow-compatible interchange spec exists in the repository.
2. Tests cover round-tripping of at least the core stable payload shapes.
3. Documentation explains how pandas, PyArrow, and the kernel interact at the boundary.
4. Binding authors have a single documented reference for tabular and structured payload encoding.

## Out of Scope

1. Publishing remote services or RPC transport.
2. Rewriting the entire Python surface around Polars.
3. Completing all downstream bindings in this track.
