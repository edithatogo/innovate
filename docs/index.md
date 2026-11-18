# Innovate documentation

The maintained site is [Innovate documentation](https://edithatogo.github.io/innovate/).
Author its pages in `docs/astro-site/src/content/docs`; public API reference is
generated from the maintained language source. The build and execution policies
are in `docs/contracts` and [ADR 0007](adr/0007-executable-documentation-contracts.md).

- [Native architecture](adr/0006-rust-polars-native-only.md)
- [Documentation retirement and learning resources](adr/0008-current-documentation-and-learning-resources.md)
- [Architecture decision history](adr/index.md)
- [Research registry preparation](registry-readiness.md)

Sphinx content and pre-0.7 notebooks were retired from the maintained tree.
`docs/contracts/retired-documentation.json` records their exact Git history.
`docs/source/_static` contains retained historical evidence, not current pages.
