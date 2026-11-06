---
title: Diagnostics Contract
description: Stable diagnostics payloads for fitted-state reporting.
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


# Diagnostics Contract

The diagnostics contract documents the fitted-state reporting fields that stay
stable across bindings and runtime implementations.

This page records the stable diagnostics boundary:

- diagnostics stay tied to the shared kernel contract;
- unsupported shapes continue to fail explicitly;
- future route changes must preserve the Astro documentation path.
