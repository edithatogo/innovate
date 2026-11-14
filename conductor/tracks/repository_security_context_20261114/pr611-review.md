# PR #611 release-control repair

## Scope and reconciliation

Parent: `74156aa3eb89fb6bd8e71c18f096b390145ac96b`, tree
`d6c7ab762954d31af7582c6e29ddaced83f8a5c5`. The original source review used
`41a8a136ae736b5af5fdf97738672fc5216c2530`; while work was in progress the
PR author fixed dry-run failure propagation for both crates in `74156aa3`.
That failure-propagation correction is preserved, not claimed as new work here.
The two crates are now selected together for packaging and upload so Cargo can
validate their workspace dependency graph before either has been published.

The bounded repair covers only `crates-publish.yml`, `release-readiness.yml`,
`security-quality.yml`, `tests/workflows/test_release_workflows.py` and this
track's index/evidence. No production Rust, runtime pins, branch protections,
scientific acceptance data, real publication or offline infrastructure changes.
Existing review fixes for stage aliases and the fuzz workspace remain intact.

## Plan and execution

- [x] Reconcile source hashes, author changes, review findings and release policy.
- [x] Reproduce remaining release-control defects before applying the repair.
- [x] Preserve `check_migration.py --release` as a failing gate for every
  readiness event; do not convert incomplete scientific acceptance into success.
- [x] Require the scientific gate and a successful joint dry run of both crates before
  requesting any registry credential or uploading. Reject unknown execution
  modes, mask the returned token, serialize publication and bound job duration.
- [x] Wire executable stdlib regressions and their lint/format checks into the
  existing required Workflow and policy lint job on Python 3.15.2.
- [ ] Complete independent review and exact-final-head hosted checks.
- [ ] Verify merge and receiving-main checks, then record final integration.

## Local verification

The isolated sparse source snapshot was fetched through the GitHub connector;
all three original workflow files were verified against their Git blob hashes.
The reconciled publishing source matched blob
`60ba86964dee8d09be532dcd8054e00e2bd1fa2c` at `74156aa3`.

Command:

```sh
python -m unittest discover -s tests/workflows -p 'test_release_workflows.py' -v
```

Actual local interpreter: CPython 3.13.5. Ten tests against the reconciled
unrepaired source exposed 16 failing subcases, with zero test errors. The
repaired source passes all ten tests with zero failures, errors or skips.
The shell-control suite uses inert process-boundary stubs for Cargo, the
migration checker, curl and jq: it does not use live credentials or a registry.
A test-harness pipe-consumption defect was corrected before the final runs;
command stubs now consume pipeline input instead of causing spurious SIGPIPE.

Seven one-at-a-time negative controls were all rejected by the suite:
removing the scientific gate, replacing its release check with a development
check, swallowing core packaging failure, omitting the tabular preflight,
allowing a dry run to upload, weakening readiness to development mode, and
swallowing readiness failure. Independent YAML parsing agreed with the test's
stdlib extraction of both workflow shell blocks. Both blocks passed `bash -n`.

A second red/green cycle exposed five failing subcases for separate per-crate
commands before joint package selection was implemented. Tabular depends on
core; Cargo 1.90 introduced native workspace publication, including joint
dry-run verification of unpublished dependencies. The existing pinned Rust
1.99.1 is retained. The command selects only innovate-core and innovate-tabular,
not every workspace member. Cargo manages dependency ordering and index waits;
the previous fixed sleep is removed. These semantics are documented by the
[Rust 1.90 announcement](https://blog.rust-lang.org/2025/09/18/Rust-1.90.0/)
and [Cargo publish reference](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
Actual packaging is still unqualified here, and publication remains blocked by
the scientific gate and existing nonregistry dependency constraints.

These are executable control-flow regressions, not actual Cargo packaging,
OIDC, scanner, native, platform or scientific qualification. The local container
has no required Python 3.15.2/Rust 1.99.1 environment, and no workstation was
connected. Full native validation and pinned Ruff/actionlint/zizmor execution
remain hosted obligations; local success does not replace them. No release
workflow was dispatched and no publication was attempted.

## Remaining boundaries

The 38 scientific migration obligations and full-release restrictions remain
open. The intentionally failing, nonrequired release-readiness report is not
an integration-CI defect and must not be suppressed to obtain a green display.

Offline caller PR #590 depends on unqualified repository-standards PR #126.
Its dedicated runner, immutable image, model assets and actual scanner/agent
execution require independent evidence. This repair neither substitutes
infrastructure nor advances that acceptance. The caller must remain draft.
