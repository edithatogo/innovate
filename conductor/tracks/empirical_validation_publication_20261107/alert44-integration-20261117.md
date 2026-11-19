# Alert 44 empirical candidate receiving scope

- Integrator: Codex chat `01a1e448-3a28-7653-a681-cd3fdb41f498`.
- Receiving worktree: `/Volumes/PortableSSD/GitHub/innovate-security-alert-44`,
  branch `codex/security-alert-44`, base
  `92519488c56c7f618c1e555b353c5fc2a5a52070`.
- Frozen source: `/Users/doughnut/.codex/worktrees/innovate-empirical-review/innovate`,
  branch `codex/innovate44-empirical-evidence`, tip
  `bb69d1b4a465521e42b55ff0599c4f951e5c69f7`, common base
  `cc2e4da4af9c848043b31f1191c9d96212048276`.
- Scope: receive the exact empirical source-validation and capability guards,
  their tests, track and registry records, the fail-closed migration guard and
  its tests, and their associated CI/Nox and documentation inventory changes.
  The integrator owns shared workflows, registry and promotion. This receipt and
  a refreshed workflow-bound documentation inventory are also in scope.
- Native object/authentication scope: none. The source checkout and its hosted
  run remain independent and are not modified by receiving this packet.
- Next operation: rerun the exact-range integration preflight, reconcile the
  independently modified workflow, Nox file and source inventory without
  dropping either branch's checks, then run focused and hosted checks. Keep the
  38 ledger obligations, scientific acceptance and release gate open.

The first read-only preflight found clean source and target worktrees, no
staged deletions, missing target files or dirty overlaps. It did not establish
semantic mergeability. The shared CI workflow, Nox file and workflow-bound
documentation inventory differ on both branches and require explicit review.

The repeated preflight at receiving commit `bc62b5811f49036c16c10892c39c1d81225ca9f7`
was clean. The standard merge retained the target's vendored-format and security
checks and the source's evidence-coverage checks. The only textual conflict was
the workflow SHA in the polyglot inventory; regeneration changed that hash alone
to match the combined workflow and its structural check passed.

On the combined staged source, repo-local Python 3.15.2 passed **87** migration
policy tests and **69** source/capability tests. The required Nox validation
session retained its two 90% branch/line gates and measured **94.39%** and
**95.45%**. The **38** polyglot documentation contract cases, targeted Ruff
lint and format, local repository policy, actionlint parsing, `git diff --check`,
and staged Gitleaks/Presidio scans passed. Hosted checks on the resulting merge
remain required; none of these checks qualifies the scientific obligations or
the offline security image.

## Diagnostic finding follow-up

At receiving base `1d5b45fc3b723f7476c24941cf77ada7f6f5b1d3`, the
alert-44 integrator owns a bounded investigation of two diagnostic data flows.
The independent source review found that unverified export literals and
docstring-only parameter names can enter API gap IDs, generator errors, stdout
and receipts; scheduler job/step environment values can enter shard receipts.
The permitted source and test paths are `scripts/check_api_documentation.py`,
`scripts/generate_api_reference.py`, `scripts/run_native_benchmarks.py`, `tests/test_api_documentation_contract.py`,
`tests/test_api_reference_generation.py`, `tests/native/test_benchmark_shards.py`
and `docs/astro-site/src/content/docs/maintainers/documentation-quality.md` and
this record. Preserve valid symbol inventory, distinct gap/ratchet identities,
same-step verification, and the existing independent checks. Review any changed
diagnostic identity and scheduler formats explicitly; hosted or real Slurm
execution is not implied by local synthetic tests.

The independent candidate reviewer also reproduced source-controlled imported
export aliases in gap details and a non-renderable exported constant in the
generator's error. The generator is included above to protect those diagnostic
copies while retaining public inventory and the existing failure behavior.

The reviewer additionally reproduced raw legacy baseline identities in stale
gap and removed-symbol output, a newline in a source filename forging a
generator diagnostic line, and legitimate signed scheduler IDs rejected by an
initial unsigned-only candidate. The final candidate fingerprints unqualified
gap and stale-baseline identities, JSON-escapes displayed diagnostic text, and
retains signed and unsigned 32-bit Slurm IDs plus the named special steps.
[Slurm's task environment source](https://github.com/SchedMD/slurm/blob/slurm-26-05-5-1/src/common/env.c)
documents signed formatting; this is source evidence, not a live Slurm run.
External API ratchet baselines containing prior raw gap IDs will need explicit
review and refresh. Fingerprints of guessable names are not anonymity.

The pre-fix regressions failed on both disclosure paths and the signed-ID
control. After the corrections, repo-local Python 3.15.2 passed **34** API
checker, **27** generator, **25** production-docs and **12** native benchmark
shard cases. Those shard tests used the separately installed local wheel above;
they do not establish operation on a real scheduler. API generation and its
freshness check, targeted Ruff and local repository-policy checks passed. The
offline SAST control and image review remain separate and unqualified.
The real local Opengrep rerun over the four originally reported scripts still
reported all seven taint findings, with four targets scanned and no scanner
errors in that focused run. The reviewer traced those reports primarily to
overtainted counts and generated identities; they are retained as open findings,
not suppressed or described as a passing security control.

The filename-control regression now feeds a synthetic newline-bearing inventory
ID into the generator boundary rather than creating a filename containing a
control character. This preserves the diagnostic assertion on Windows filesystems
that cannot create such names; the separate reviewer reproduced the real-path
case on macOS. All **27** generator tests and targeted Ruff checks still pass.
## GitHub-hosted offline check

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the hosted caller in
the PR #684 worktree from `fcda61d8a9cc1e78c381c5fcb091abd47bbc2e77`.
This packet covers the offline workflow, security policy and this record.
The native scope is Innovate PR #684 and shared policy PR #152.

The prior hosted canary denied scanner egress and passed the scanner
controls on the exact Innovate source. The new caller selects GitHub-hosted
Ubuntu through the immutable shared workflow. It keeps the scanner controls,
adds the Wolfi review to the check, and records image advisories on every run.
RustSec and the 38 release obligations remain gates on promotion.

## Direct calendar dependency

The same chat owns Innovate's direct calendar dependency from
`1703840d7c3b70cae6475206ec29e5ad94fed484`. Scope is the core crate's private
date and timestamp implementation, manifest and lock, and focused consumer
tests. Preserve the timestamp wire format, the supported calendar range,
and the Polars DataFrame API. Cargo still resolves the two unmaintained
crates through published Polars, so the RustSec check must continue to fail
until that dependency path is replaced too.

Rust 1.100.0 passed the eight temporal consumer cases and both calendar
unit cases. A standalone comparison against the former implementation
matched 13 dates, 23 timestamp inputs and 1,407 weekly comparisons with
the same private adapter. The locked core dependency tree selects
`time 0.3.55`. The workspace tree retains `chrono-tz 0.10.4` through
`polars-arrow 0.55.2`.

## Published Polars consumer check

From `60c0593988f522495bbc867d91db3949c8086f4c`, a separate Cargo root
selected crates.io `polars-core 0.55.2` with default features disabled.
It compiled and passed a Bass prediction using the same published
`DataFrame` type on input and output. The inverse dependency trees still
reported `chrono 0.4.45` and `chrono-tz 0.10.4` through Polars.
The upstream Jiff PR is a draft. Polars' own Cargo policy now ignores the
two advisories, while Innovate's RustSec check continues to enforce them.
[Polars replacement](https://github.com/pola-rs/polars/pull/29110) ·
[Polars policy change](https://github.com/pola-rs/polars/pull/30859)

## Standalone games lockfile in the local Ubuntu replay

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns this follow-up in
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`
from `1fb5055bda849ec886900afc938bb2f80e4a1965`. Scope covers
`rust/crates/innovate-games/Cargo.lock` and this record.

The isolated Tart and Sand Ubuntu guest ran the existing games ban command
on the exact PR source. Cargo rejected the standalone lock with `--locked`
because it still listed `chrono` under `innovate-core`. The core manifest had
already moved to `time`. The locked root graph independently reported the
two RustSec findings through Polars. The published-DataFrame prediction
consumer passed in the same guest.

Run `cargo metadata --manifest-path rust/crates/innovate-games/Cargo.toml
--offline --all-features --format-version 1` to refresh the standalone
lock. The resulting delta removes Chrono and selects the same six `time`
packages as the root lock. Paignia and Kairos pins retain their existing
revisions. Repeat the games ban check with `--locked` in the guest.
