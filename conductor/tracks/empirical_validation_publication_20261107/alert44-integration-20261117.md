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
