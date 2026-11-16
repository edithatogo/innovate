# innovate delivery contract

Approved by user on 29 October 2026. This bounded track implements the
[coordinated programme](https://github.com/edithatogo/repository-standards/tree/7874f05132447c77f681fbca7ad5394066564aaf/conductor/tracks/scientific_ecosystem_alignment_20261029); it does not replace existing native
acceptance or other active track owners.

- Qualify the current declared Python, Arrow/Polars and native dependency
  candidates. The original Python 3.15.1 candidate is historical; the current
  source contract is recorded below and in the dated reconciliation.
- Inventory actual Kairos calls and qualify the accepted upstream pin without copying a scheduler.
- Retain diffusion fitting and diagnostics here; qualify installed forecasts and adoption artifact handoff.

Use frontier-first qualified candidates and explicit retained-version exceptions.
Separate core, binding, backend, build, test-oracle and tooling dependencies.
Acceptance requires exact-source numerical and consumer tests, measured builds,
agent correctness/security review, fixes and repository integration. Planning or
static source checks are not native acceptance. Existing promises are retained
until a versioned migration passes; no paid execution or public release implied.

## User-authorised scope expansion — 6 November 2026

The user now requests a Rust-owned library, latest qualified dependencies,
Polars without pandas/xarray, and only Python/Julia/R bindings. API redesign
and interim feature unavailability are allowed; all current and planned
scientific features remain required in the final library. This supersedes the
original interim API-preservation restriction above. The detailed migration
plan, feature-source inventory and root migration ledger govern this proposal.
No main integration, public release, native deployment or acceptance is implied.

## Current qualification contract — 16 November 2026

The [current reconciliation](reconciliation-20261116.md) binds this contract to
main `69b674814513e09a683279c679b1584af8bb4214`. Rust owns scientific
computation and validation; Polars is the public table interface, with versioned
Arrow/Parquet interchange. Python, Julia and R remain thin bindings. Kairos owns
scheduling, state and replay; Voiage owns value-of-information analysis.

The inspected manifest selects Python `>=3.15.2,<3.16`, Polars `2.1.0` and the
optional PyArrow `26.0.1` extra. Cargo declares a Rust minimum of `1.99.1`, while
the inspected native CI uses `1.100.0`. Each receipt must record the actual
interpreter/compiler and dependency closure; a declared minimum is not an
executed compatibility result. The five Kairos dependencies are pinned to
`a1f6283575ad26918230adc122a99b5c08fc4da3`. A dependency update requires
provider-specific evidence before earlier acceptance can be reused.

Acceptance must distinguish these outcomes:

1. **Source integration:** the native Arrow outcome-shape correction is merged
   through #602. It validates explicit result/error envelopes and preserves
   supported outcome data; it does not validate every scientific result.
2. **Installed-consumer qualification:** identified built artifacts execute the
   declared positive, invalid-input and recovery cases outside the checkout.
   The optional Arrow extra must be installed and its four readback/round-trip
   cases must execute. An optional-test skip is not a passing consumer result.
3. **Provider and platform qualification:** receipts identify the Innovate source,
   Kairos revision, locks, features, artifacts and tested platforms. Actual native
   execution, deterministic replay and installed binding results are required.
4. **Scientific and release acceptance:** all required migration obligations,
   relevant independent-oracle and empirical evidence, current quality/security
   gates and distributable-artifact checks retain their existing requirements.
   Historical receipts, documentation checks and merges cannot close these gates.

Historical ArviZ and pytest-benchmark compatibility exceptions remain in
[qualification.md](qualification.md). Their retired Python computational routes
are not production fallbacks. Native uncertainty and performance obligations
remain with their existing scientific owners and the
[V01–V05/Q01–Q05 taskbook](../../innovation-taskbook-20261107.md).

The [registry/journal track](../research_software_registry_readiness_20260721/index.md)
consumes accurate software scope, reproducible installed examples and research-use
evidence. It does not confer scientific acceptance. A JOSS software paper,
comparative prediction study, causal study and innovation/hype application retain
separate claims and evidence. Submission, release and external registration remain
separate actions with their own authority and verified outcomes.

This reconciliation changes only the five named documentation files in this
track. Shared manifests, provider pins, schemas, CI, registries and promotion
remain with the [sole integrator](../../dependency-migration-ownership-20261115.md).
Existing native, empirical and documentation owners retain their work.
