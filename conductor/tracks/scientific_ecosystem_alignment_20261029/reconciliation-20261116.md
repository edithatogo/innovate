# Scientific ecosystem qualification reconciliation — 16 November 2026

**Executed follow-up:** [installed/native qualification evidence](evidence/arrow-installed-20261116/README.md)
records a fresh Linux development wheel, 42 passing installed cases including
all four optional Arrow cases, and the separate successful Rust 1.99.1 workspace
compile check. The linked packet and current plan supersede the initial
unexecuted-command status below only within those source/artifact scopes.
Shared CI adoption, broader scientific/platform/provider acceptance and release
promotion remain with #624 and the existing owners.

## Scope and authority

This record reconciles [issue #556](https://github.com/edithatogo/innovate/issues/556)
against main `69b674814513e09a683279c679b1584af8bb4214`, inspected for the
scientific/journal update on branch `docs/scientific-journal-reconciliation-20261116`.
It is the current operational summary for this track. The chronological
[plan](plan.md), original receipts and failed checks remain intact; their old
next-action or assignment statements do not create new work ownership.

Allowed files for this bounded pass are this record plus `index.md`, `spec.md`,
`plan.md` and `metadata.json` in this track. Sole integration coordinator
`6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53` retains shared source, manifests,
provider pins, CI, registries and promotion through
[#624](https://github.com/edithatogo/innovate/issues/624) and the
[ownership record](../../dependency-migration-ownership-20261115.md).
No existing scientific, empirical, documentation or security owner is replaced.

## Current architecture and evidence

| Item | Observed source or receipt | Current disposition |
| --- | --- | --- |
| Scientific execution | Rust core, Polars tables, thin Python/Julia/R bindings; `README.md` and `AGENTS.md` | Current architecture; no Python numerical fallback |
| Python and tables | `pyproject.toml`: Python `>=3.15.2,<3.16`, Polars `2.1.0`, optional PyArrow `26.0.1` | Declared versions; each acceptance receipt must identify actual runtime and installed artifacts |
| Rust | `Cargo.toml`: minimum `1.99.1`; `.github/workflows/ci.yml`: compiler `1.100.0` | Minimum and tested compiler are distinct; exact minimum-version execution remains a shared qualification question |
| Kairos | Five dependencies in `rust/crates/innovate-core/Cargo.toml` and the lock point to `a1f6283575ad26918230adc122a99b5c08fc4da3` after #657 | Pin integrated; provider-specific acceptance and publishability are separate |
| Original Python/Arrow candidate | [#554](https://github.com/edithatogo/innovate/pull/554), merged as `e0b764896704c27562f66c8891e74b5f8582b9ed` | Historical bounded evidence, predating the native development architecture |
| Arrow outcome correction | [#602](https://github.com/edithatogo/innovate/pull/602), head `ae592570f1d3de7364ef936695a4ca35084f6c01`, merged as `2c9ce0fda410161961209ddbf19d99515732e3c9` | Source integrated; explicit outcome-shape validation, not scientific-result validation |
| Arrow regression history | [Red receipt](issue-556-arrow-qualification.md): 33 expected invalid-input failures and five positive controls against source-equivalent retained code | Preserved historical red evidence; its pending-green wording is superseded only by later scoped receipts |
| Later Arrow pass | [Comment 6907498117](https://github.com/edithatogo/innovate/issues/556#issuecomment-6907498117): 38 passed, zero skipped on `44cb9635`, including all four optional cases | Reported earlier-source qualification; not a fresh run on this record's source or provider pin |
| Historical optional tools | [qualification.md](qualification.md): ArviZ API incompatibility and pytest-benchmark breakage | Historical exceptions retained; retired Python numerical routes are not restored by this track |
| Scientific release scope | `migration-ledger.json`: 16 required models and 22 required feature areas, all `bounded_native`; `release_ready: false` | All 38 full-feature obligations remain; bounded implementations are not full scientific acceptance |

An artifact-level receipt must include the source/tree identity, lock hashes,
provider revision, runtime/compiler, OS/architecture, wheel/shared-library hashes,
installed import locations, command, exit and pass/fail/skip counts. Reuse a
historical receipt only with a documented source/dependency equivalence argument
within its original scope. Never transfer a whole-track pass from a merge.

## Current quality and release observations

At the inspected source, hosted
[Release Readiness run 43844964697](https://github.com/edithatogo/innovate/actions/runs/43844964697)
failed in job `133222714028` at the migration-policy step. The inspected log uses
Python 3.15.2 and lists all 38 outstanding obligations. The reconciliation
coordinator also executed
`.venv-registry-reconciliation/bin/python scripts/check_migration.py --release`
under Python 3.15.2 locally: exit 1, with the same 38 obligations. This is an
expected unresolved release gate, not evidence that native tests failed.

At the hosted snapshot, CI `43844964716`, Security and Quality `43844964709`,
documentation `43844964699` and offline security `43844965140` were queued;
Voiage installed-consumer `43844964682` and coverage `43844964696` were in
progress. Privacy/secrets `43844964366` passed. Re-read these exact-source runs
before integration; queued or running work is not qualified.

The coordinator's repository-policy check passed. Conductor registry hygiene
reported the pre-existing completed ODE, NUTS and supply-chain tracks remaining
in active directories. Shared registry reconciliation belongs to the integrator.
This documentation pass ran no Rust numerical or installed-native test suite.

## Remaining work and order

The IDs below organise existing obligations; they do not create parallel source
owners or replace the taskbook. Begin with current ownership and source checks.

| Packet | Existing owner and dependencies | Required outcome and evidence |
| --- | --- | --- |
| ALIGN-01: current status | This bounded documentation lane; fresh main and PR inspection | Current source, historical receipts and unresolved obligations recorded; source review complete; integrator acceptance remains pending |
| ALIGN-02: optional Arrow consumer | Existing tabular/scenario owner with integrator-controlled artifact build; depends on current installed artifact | Focused Rust export checks and installed Python export/producer cases, all four optional PyArrow cases executed, recovery and nested-data preservation retained, exact artifact receipt |
| ALIGN-03: Kairos consumer and distribution | Existing provider/native owners and sole integrator; depends on accepted immutable provider interface and candidate | Real DES/ABM/network replay and binding evidence at the selected pin; registry publication/pinning and crate publishability disposition remain separately owned |
| ALIGN-04: native quality and supply chain | Existing Q01–Q04/security/platform owners; depends on the exact candidate and provider closure | Actual advertised platform consumers, FFI/input tests, current coverage/mutation/security/license and artifact evidence; scheduled-only gates receive their own executed receipts |
| ALIGN-05: research and paper evidence | Existing empirical V01–V05 and documentation owners; depends on truthful supported scope and qualified sources | Independent oracles, frozen comparative protocols, reproducible installed examples, research-use evidence and claim-specific limitations; no empirical claim inferred from registry metadata |
| ALIGN-06: release and journal decisions | Integrator/Q05 and author; depends on all mandatory scoped receipts | Explicit exact-artifact release disposition and separate journal readiness/author decisions; unresolved mandatory gates block promotion |

ALIGN-02, provider inspection within ALIGN-03, and registry/documentation
preparation can proceed independently within existing owners. Empirical jobs
must respect their native and source-rights prerequisites. Integration and
release acceptance remain serial decisions after current evidence is reviewed.

### Existing executable targets

These are next qualification commands, not runs performed by this documentation
pass. Use the selected qualified toolchain and an owned environment. Build and
install an identified wheel before the Python checks; execute its copied tests
outside the checkout without a source-path override, and record imports resolving
to that installed package.

```sh
cargo test --locked -p innovate-tabular --test scenario_exports
python -m pytest tests/native/test_scenario_exports.py
python -m pytest tests/native/test_python_consumer.py -k 'scenario or kairos'
cargo test --locked -p innovate-core --test kairos --test network
python scripts/check_migration.py
python scripts/check_migration.py --release
```

For the Arrow packet, the installed environment must include the locked `arrow`
extra, Python 3.15.2 and the configured pytest version. An unbuilt source tree,
import-only check, runtime-version override or skipped optional test cannot stand
in for the installed qualification. Additional Julia/R and platform checks use
their existing qualified consumer procedures. A source or dependency change
invalidates affected results until reviewed and rerun where necessary.

Ordinary CI does not install the optional Arrow extra. It also runs source-
distribution consumers, remote-container checks, native benchmarks and MPI
benchmark checks only on scheduled/manual events. Routine PR success therefore
does not establish those results. Native speed, uncertainty and empirical claims
use the existing [V01–V05/Q01–Q05 taskbook](../../innovation-taskbook-20261107.md),
with actual targets reconciled by their owners before execution. Several older
taskbook/provider plans name retired paths; a planned filename is not proof that
the target exists.

## Shared integration intake

Send these requirements with this packet to the existing #624 integrator. They
are requested follow-up work, not implemented CI or dependency changes here.

1. **Optional Arrow recurrence:** qualify a maintained job installing the existing
   locked extra and identified package; require the four optional round-trip cases
   to execute and retain test counts, artifact hashes and runtime identity. Keep
   the minimal-install lane and all existing gates intact.
2. **Rust minimum-version evidence:** decide and record whether the current
   `1.99.1` minimum is actually tested alongside CI's `1.100.0`. Execute the
   applicable locked minimum-version build/tests or explicitly retain the
   qualification gap; do not silently change the declared minimum to obtain green.
3. **Provider/publishability:** retain the five-crate exact-pin receipt and route
   upstream publication through `rust_kairos_crates_io_20261115`. Its older
   `e76b97b...` comparison baseline and the runtime-boundary plan's retired paths
   need owner reconciliation before execution. A lawful Git dependency build is
   not proof of crates.io publishability.
4. **Active PR reconciliation:** [#656](https://github.com/edithatogo/innovate/pull/656)
   changes shared contracts and the empirical plan and proposes track archival;
   [#659](https://github.com/edithatogo/innovate/pull/659) adds unfinished tutorial
   work. Preserve active work and re-resolve links/ownership when either integrates.
   Neither PR's proposed status transfers scientific or publication acceptance.

## Journal and registry handoff

The [registry reconciliation](../research_software_registry_readiness_20260721/reconciliation-20261116.md)
coordinates #398–#401. [JOSS readiness #401](https://github.com/edithatogo/innovate/issues/401)
consumes a defensible software contribution, truthful support boundaries,
independent installation/reproduction, research-use evidence and current journal
requirements. Archival identifiers and citation metadata identify software; they
do not qualify its numerical or empirical claims.

The [empirical publication specification](../empirical_validation_publication_20261107/spec.md)
and its V05 checkpoint govern each paper's frozen inputs, claims, review and
author decisions. The software paper remains distinct from forecasting,
causal, innovation/hype, medicines and decision studies. All 38 scientific
obligations, the four historical manual scenario checks, provider acceptance,
optional-consumer qualification and supply-chain/release holds are retained.

## Validation of this documentation increment

Source and hosted evidence were inspected read-only before these edits. Local
`git diff --check` passed. Python standard-library checks parsed the metadata,
resolved all new/changed relative-link targets and compared the original plan
from its first historical section to EOF byte for byte with the inspected
baseline: unchanged. The five allowed documentation files passed root and independent source review;
integration and current-head hosted qualification remain pending.

No scientific result, runtime implementation, shared CI change, release,
registration or journal submission is claimed by this reconciliation.
