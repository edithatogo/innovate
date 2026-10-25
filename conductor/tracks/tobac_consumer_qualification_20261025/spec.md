# tobac uptake and diffusion consumer qualification

**Approved planning only; implementation NOT authorised.**
Base: `83d556ba4995fd5f5238f4953e9a8af34976f464`. [Master track](https://github.com/edithatogo/tobac/tree/codex/ecosystem-planning-20261025/conductor/tracks/scientific_ecosystem_20261025).

## Scope

Qualify existing uptake/diffusion, fitting, functional-kernel and Arrow interfaces for tobac. Preserve Rust-core direction and native/public bindings. Equinox, Diffrax and Optimistix are conditional optional adapters after capability and numerical review, not mandatory replacements.

The new deliverable is a named tobac consumer qualification record, not another general
modernisation programme. Registry and all-state issue inspection found no matching tobac
consumer track. Source capability labels and completed generic tracks do not prove downstream
compatibility. tobac owns domain-specific decisions and scientific applicability.

## Reuse

- [Existing owner 1](../../../conductor/archive/functional_kernel_contract_20260415/)
- [Existing owner 2](../../../conductor/archive/arrow_interchange_schema_20260416/)
- [Existing owner 3](../../../conductor/archive/optional_backends_stabilization_20260415/)

## Acceptance

1. Map the exact producer API/schema/artifact revision to each named consumer requirement;
   distinguish existing capability from a demonstrated gap before modifying code.
2. Write consumer and independent-oracle tests before semantic changes. Matched trajectories, units/time origin, ordering, parameter-domain rejection, synthetic fit recovery, missing data and conditional float64 backend parity.
3. Exercise real installed artifacts outside the checkout/sibling editable sources; record
   command, exit status, imported paths, revisions, input/output hashes and schema versions.
4. Qualify explicitly selected repo-local Python 3.15.1 before future Python implementation,
   retaining native binding ownership and honest dependency blockers. No downgrade/build bypass.
5. Preserve canonical reference/evidence hashes, precision contracts, schema migration and
   rollback. Stochastic agreement is not established by matching backend seeds alone.
6. Reuse existing engineering/harness/security owners and the master tool matrix. Assess
   pytest-gremlins and other tools only for demonstrated gaps; preserve current required gates.
7. Run relevant full native, installed-consumer, coverage, security and hosted gates at coherent
   milestones. Contract fixtures do not establish hosted execution or scientific calibration.

## Authority and exclusions

Registration permits only planning files and coordination. No implementation, environment,
lock, dependency, workflow, scientific model, release or publication action is authorised.
No automatic progression, Rust-to-Python rewrite, duplicate statistical engine or blanket
new tool stack. Existing workstreams retain their scopes. A future assessment can conclude
with an evidenced defer decision without claiming the deferred capability is implemented.
