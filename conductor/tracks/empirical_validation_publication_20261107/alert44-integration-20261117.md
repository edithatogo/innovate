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
