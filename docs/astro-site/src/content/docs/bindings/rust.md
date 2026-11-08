---
title: Rust core
description: Scientific execution is owned by Rust
---

Rust is the native core, not a client binding. The workspace separates
`innovate-core`, `innovate-tabular`, `innovate-python` and `innovate-ffi`.
Run `cargo test --locked -p innovate-core -p innovate-tabular -p innovate-ffi`.
Scientific feature coverage and release acceptance remain tracked in the
[native migration guide](/innovate/native-migration/).
