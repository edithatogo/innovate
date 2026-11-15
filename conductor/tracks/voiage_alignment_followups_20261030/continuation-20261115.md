# Hosted Voiage qualification and remaining acceptance

Recorded 15 November 2026 in Australia/Brisbane; the hosted executions and
inspection occurred on 14 November UTC. This continues
[issue #559](https://github.com/edithatogo/innovate/issues/559) and
[PR #605](https://github.com/edithatogo/innovate/pull/605) within the existing
scientific ecosystem programme.

The installed native consumer is now **executed and passing**. Strict Renovate
configuration validation also passes. Live Dependabot disposition, a real
verified Renovate update, and checks on the branch after incorporating newer
main remain separate acceptance items.

## Exact hosted source and artifact

[Consumer run 43603039393](https://github.com/edithatogo/innovate/actions/runs/43603039393),
attempt 1, completed successfully at `2026-11-14T14:24:04Z`. Its job was
`132471687381`. The workflow event identifies PR head
`7075f24ef2b6dcc3e49cdfe07df659afec522475`. Checkout and the receipt identify
the synthetic test merge `6d132d6379cfe7f7ba08456b7fc0563cab696fc5`, with parents
`276c2e505a0ce26910820fe8a94a6a498f505700` and that PR head. Its tree
`b7ea97300f10d1b60033336b3318b9ee9bda74ef` equals the PR head's tree.

The job ran `bash scripts/build_release_wheel.sh .venv`, creating an sdist from
that checkout and building the extracted source with Maturin's release profile
and the locked Cargo closure. It installed the resulting Innovate wheel and the
downloaded Voiage 3.1.0 wheel in a fresh environment outside the checkout.
Observed tools were Python **3.15.2**, Rust **1.99.1** and uv **0.13.8**; consumer
dependencies included Polars **2.1.0** and PyArrow **26.0.1**. The dependency
compatibility check passed for all 34 installed consumer packages.

Artifact `14117822045` was downloaded and its full ZIP SHA-256 independently
matched GitHub's artifact digest:

```text
e9fce853715cba333207b884b13012600a0ca11d21b3f8c91f1ee17383fcc6c2
```

The unmodified [qualification JSON](evidence/hosted-consumer-43603039393/qualification.json)
and [focused test output](evidence/hosted-consumer-43603039393/tests.txt) are
retained with [source, build and archive provenance](evidence/hosted-consumer-43603039393/provenance.json).
The JSON records the actual isolated worker command, exit status **0**, installed
import paths, wheel/native/RECORD hashes, and input/output hashes. The worker's
directory was `/tmp/innovate-installed-consumer-xjwyqs4w`; imported packages were
under `/home/runner/work/_temp/voiage-consumer.fuZu80`, outside the checkout.
Qualification stderr was empty. Imported package members matched both supplied
wheel files; pandas and xarray were absent from the consumer environment.

## Executed numerical and interchange results

The focused suite passed **19 tests in 1.73 seconds**. The independent installed
probe then passed all six native normalized-bundle cases:

| Fixture | Formats | Persisted contract | Installed native result |
| --- | --- | --- | --- |
| Simple table | IPC and Parquet | Original/restored digest and schema equal | `[1.0, 2.0, 3.0]` |
| Multiple chunks | IPC and Parquet | Digest equal; `[2, 1]` chunk lengths restored for both columns | `[1.0, 2.0, 3.0]` |
| Parquet-normalized types | IPC and Parquet | Digest equal; timestamp seconds, nested field metadata and schema restored | `[1.0, 2.0, 3.0]` |

All native cases explicitly projected the verified frame to `time` and
`sample_id`. These results qualify those numerical columns; preserving the full
Arrow schema does not qualify native datetime or covariance operations.

The pre-existing Bass analytic and Logistic midpoint checks also passed. The
subsequent payoff handoff produced the following EVPI results, within the
unchanged `1e-12` relative and absolute tolerances:

| Oracle | Independent reference | Installed Voiage |
| --- | ---: | ---: |
| Equal-weight EVPI | 5.641720719927869 | 5.641720719927868 |
| Empirical 3:1 EVPI | 3.7317144196428957 | 3.731714419642895 |

This remains a synthetic CPU consumer qualification. It does not complete the
broader scientific `voiage_arrow_handoff` obligation or imply HPC, clinical or
release acceptance. The earlier local compilation terminations remain accurately
recorded; the successful hosted build supplies the missing native evidence.

## Protected checks and the release boundary

All seven protected contexts succeeded for head `7075f24`:

| Required context | Check/job ID |
| --- | ---: |
| `lint` | 132485994077 |
| `package` | 132477551155 |
| `CodeQL` | 132475275506 |
| `Secret Scan` | 132471687552 |
| `Workflow and policy lint` | 132471687951 |
| `Python quality harness` | 132471687845 |
| `unit-tests-base (3.15.1)` | 132485994050 |

The historical required context name is preserved. That job aggregates the
native Python platform jobs, which installed and asserted Python 3.15.2.
The [machine-readable gate observation](evidence/hosted-gates-7075f24.json)
retains all check and workflow URLs, including CodeQL, which is omitted by the
connector's pull-request-only workflow wrapper. Binding conformance, the three
native Python platform jobs, RustSec and Dependency Review also passed.

[Release Readiness 43603039387](https://github.com/edithatogo/innovate/actions/runs/43603039387/job/132471686568)
failed at `.venv-release/bin/python scripts/check_migration.py --release`, exit
**1**, listing exactly the existing **38 scientific obligations**. The full list
and exact log lines are retained in the gate observation. None were marked
complete, waived, or removed. The separate release-lane job was skipped by its
existing manual/scheduled trigger condition.

## Renovate validation and remaining security evidence

The exact `.github/renovate.json` with `platformCommit: enabled` now passes
Renovate **44.239.0** in strict repository mode, on Node **24.22.1**:

```sh
renovate-config-validator --strict --no-global .github/renovate.json
```

The [validation receipt](evidence/renovate-validation-20261115.json) records the
actual executable path, bounded installation, command, exit status, output and
configuration SHA-256. The isolated install omitted optional dependencies;
Renovate warned that RE2 was unavailable and used its built-in RegExp fallback.
This exact configuration has no custom regex managers or patterns. Future
custom-regex configuration needs RE2-capable validation. The previous exit-137
installation remains historical negative evidence. Repository-mode validation covers this configuration and
preset-reference syntax; it does not evaluate a fully merged remote preset or
demonstrate a bot's cryptographic signature.

The active Renovate [dashboard #595](https://github.com/edithatogo/innovate/issues/595)
lists updates awaiting the configured schedule. The accessible PR, commit and
branch observations supplied no actual Renovate update commit to verify.
`platformCommit: enabled` must be preserved when integrating the separately
owned central-preset/custom-manager work in [PR #606](https://github.com/edithatogo/innovate/pull/606).
Its validator workflow remains separate; no duplicate workflow or required check
was added here.

Live Dependabot alerts remain unavailable through the connector and the prior
browser fallback was signed out. Full classic signing-rule details returned 403.
The earlier 99-package audit with no reported vulnerabilities remains valid for
its recorded Python closure, but it is not an alert inventory. No unseen alert
was dismissed and no protection rule changed. These acceptance items remain open.

## Main reconciliation and integration

Main advanced beyond the tested base while the consumer run was underway, to
`13bdcf10ccb3e1cd06da1df4d98fc5da831768db`, the separate ensemble time-grid fix
from PR #600. This continuation incorporates that revision without changing its
four files. The Voiage harness, focused tests, consumer workflow and Renovate
configuration remain byte-identical to their qualified source checkpoint.

The retained successful runs are attributed only to their actual earlier head
and test merge. The resulting updated branch must receive its own hosted checks;
their current status is maintained in PR #605 rather than rewriting this original
receipt. Integration remains with the existing repository coordinator. Issue
#559 and its track remain open for security acceptance; no release is authorized
by these results.
