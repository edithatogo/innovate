---
title: Rust core
description: Scientific execution is owned by Rust
---

Rust is the native core, not a client binding. The workspace separates
`innovate-core`, `innovate-tabular`, `innovate-python`, `innovate-ffi` and
`innovate-kairos`. The Kairos crate exposes a typed simulation adapter over the
core dispatcher. Run `cargo test --workspace --locked` from the repository root.

`innovate-games` is a standalone static game adapter with its own workspace and
lockfile. The root workspace explicitly excludes it: `cargo doc --workspace` and
`cargo test --workspace --doc` do not qualify its API or examples. Its separate
manifest is `rust/crates/innovate-games/Cargo.toml`; record any execution against
that manifest separately. See the [language conventions](/innovate/maintainers/language-conventions/)
for the strict workspace rustdoc and doctest commands.

Scientific feature coverage and release acceptance remain tracked in the
[native migration guide](/innovate/native-migration/).
