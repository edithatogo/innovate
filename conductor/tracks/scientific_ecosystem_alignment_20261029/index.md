# innovate: scientific ecosystem alignment

- [New evidence acquisition and concrete integration handoff](../research_software_registry_readiness_20260721/evidence-acquisition-20261116.md)
- [Current-main installed qualification](evidence/evidence-acquisition-20261116/README.md)
- [Complete recurring Arrow/MSRV workflow candidate](integration/README.md)

- [Current reconciliation and remaining acceptance — 16 November 2026](reconciliation-20261116.md)
- [Qualification after the main-branch frontier merge](evidence/arrow-integrated-20261116/README.md)
- [Earlier installed Arrow qualification](evidence/arrow-installed-20261116/README.md)
- [Integrated scientific/journal source reassessment](../research_software_registry_readiness_20260721/integration-20261116.md)
- [Specification](spec.md)
- [Plan](plan.md)
- [Metadata](metadata.json)
- [Coordinator](https://github.com/edithatogo/repository-standards/tree/7874f05132447c77f681fbca7ad5394066564aaf/conductor/tracks/scientific_ecosystem_alignment_20261029)
- [Research-software registry and journal reconciliation](../research_software_registry_readiness_20260721/reconciliation-20261116.md)

This active track coordinates native ecosystem qualification for
[issue #556](https://github.com/edithatogo/innovate/issues/556). The current
reconciliation takes precedence over historical next-action and ownership
statements in the chronological plan; it preserves their original evidence.

[PR #554](https://github.com/edithatogo/innovate/pull/554) delivered the earlier
Python/Arrow candidate. The current 0.7 development architecture has a Rust
scientific core, Polars tables and thin Python, Julia and R bindings.
[PR #602](https://github.com/edithatogo/innovate/pull/602) merged the scenario
Arrow outcome-shape correction. Neither merge nor the reported 38-test pass on
`44cb9635` establishes current-source provider, optional-consumer or release
acceptance. All 38 full-feature scientific obligations remain open.

The current follow-up implements a repeatable installed-artifact qualifier and
records 42 passing Linux consumer cases with all four optional PyArrow cases
executed and no skips. Its artifact/source/compiler scope is explicit in the
linked receipt; shared CI adoption and broader acceptance remain pending.
