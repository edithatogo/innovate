---
title: ABI and Binary Compatibility
description: Native and binary boundary strategy for the packaging stack.
---

# ABI and Binary Compatibility Strategy

The strategy keeps public users on schema-versioned payloads and public API surfaces while allowing internal native implementation changes under strict compatibility gates.

Core boundaries include:

- Public API preservation for Python operations and binding wrappers.
- Kernel-schema compatibility by versioned request/response contracts.
- Explicit native ABI boundaries with backend capability metadata.
- Arrow C Data / Stream interfaces as preferred cross-language transport.

XLA internals, private Rust structs, and binding-specific native implementations are not part of public ABI claims.

Migration source:

- `docs/source/abi_binary_compatibility_strategy.rst`

