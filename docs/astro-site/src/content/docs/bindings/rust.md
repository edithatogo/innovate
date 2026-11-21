---
title: Rust core
description: Scientific execution is owned by Rust
---

Rust is the native core. The workspace includes `innovate-core`,
`innovate-demand`, `innovate-tabular`, `innovate-python`, `innovate-ffi` and
`innovate-kairos`. The demand crate provides the shared native choice model.
The Kairos crate exposes a typed simulation adapter over the core dispatcher.
Run `cargo test --workspace --locked` from the repository root.

`innovate-games` is a standalone static game adapter with its own workspace and
lockfile. It uses `innovate-demand` and the version-pinned Paignia core and solver.
The root workspace excludes `innovate-games`, so test it directly with
`cargo test --locked --manifest-path rust/crates/innovate-games/Cargo.toml`.
Root `--workspace` commands select workspace members. Doctests run for eligible
targets, including native libraries with doctests enabled.
[Language conventions](/innovate/maintainers/language-conventions/)

Scientific feature coverage and release acceptance remain tracked in the
[native migration guide](/innovate/native-migration/).
