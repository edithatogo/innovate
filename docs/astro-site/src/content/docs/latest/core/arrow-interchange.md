---
title: Arrow Interchange
description: Columnar and tabular interchange for cross-language payloads.
slug: latest/core/arrow-interchange
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


# Arrow Interchange

The docs-site migration preserves the Arrow interchange boundary as a core
contract surface. This page explains the payload shapes, versioning rules, and
interoperability guarantees for the polyglot library.

Key migration points:

* Arrow remains the stable interchange boundary for language bindings.
* The site keeps the interchange contract visible alongside the kernel pages.
* Astro/Starlight routes are the supported documentation URLs after cutover.
