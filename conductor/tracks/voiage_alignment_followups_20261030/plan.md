# Plan

- [x] Inventory pandas/xarray use and the Polars/PyArrow boundary (alignment-audit-20261114.md).
- [~] Add a failing consumer test for normalized-bundle round trip, then implement.
- [x] Record toolchain exceptions against the shared matrix (alignment-audit-20261114.md).
- [~] Triage open Dependabot alerts; fix or dismiss with a recorded reason.
- [~] Installed-artifact evidence; run required hosted gates (qualification-20261114.md; local contract tests pass, native hosted evidence pending).
- [x] Owner authorisation: Dylan requested implementation of issue #559 on 14 November 2026.
- [ ] Archive only after the remaining acceptance gates pass.

## 14 November 2026 continuation

Reconciled GitHub main at `276c2e505a0ce26910820fe8a94a6a498f505700`.
This continuation owns the existing Voiage qualification harness, focused
consumer tests, the Renovate signing option and this follow-up's records.
It reuses the parent scientific ecosystem programme. Shared toolchain promotion
and the broader #556 lane retain their owners; PR #600's ensemble guard and
the unqualified offline security draft #590 are separate work.

Extend `scripts/qualify_voiage_arrow.py` to read normalized IPC/Parquet bundles,
check source digests before Polars conversion, and exercise installed native
Innovate with independent synthetic numerical expectations. Record actual
import origins, artifact/source hashes, command and exit status outside the
checkout. Existing scientific and hosted gates remain required.

Source checkpoint: `ef5a0737f987d331513f5239f74a8e168a32af37`,
[PR #605](https://github.com/edithatogo/innovate/pull/605). Nineteen focused
contract/artifact/launcher tests pass from outside the checkout. A fresh audit
of 99 locked installed dependencies reports no known vulnerabilities or skips.
Native installed qualification and protected hosted gates are pending. Live
Dependabot/signing acceptance remains unverified; see the linked audit and
qualification records. The PR is a focused integration proposal for the
existing repository coordinator, not a completed release or issue closure.
