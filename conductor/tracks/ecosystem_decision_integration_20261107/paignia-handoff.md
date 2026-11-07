# Paignia, Kairos and Voiage integration supplement

Date: 2026-11-08. Read the existing specification, plan and 52-packet programme first. Status: planned; no model, dependency or consumer acceptance is claimed. This supplement resolves names and provider handoffs without replacing those packets.

## Provider identity

**Paignia** is the chosen reusable Rust game library at [edithatogo/paignia](https://github.com/edithatogo/paignia). Its repository is created; transfer of the renamed `paignia-core`/`paignia-solve` seed from `new-drug-reimbursement-game` is being coordinated. Package publication, installed-consumer and numerical qualification remain pending. Historical `uogto-game-*` IDs remain in their original evidence; optional `paignia-kairos`, `paignia-uogto` and `paignia-arrow` packages are still planned.

Provider Track T17 transfers with Paignia while retaining its seventeen IDs and the reimbursement-game issue #100/PR #101 history. Paignia owns T17.01–T17.08/T17.12–T17.17; T17.09 is the first application-consumer handoff, T17.10 the second distinct application/Innovate handoff, and T17.11 remains reimbursement-specific valuation work. Those application implementations and completion receipts stay with their owning repositories, linked from the provider track. Kairos S01-S04 scientific obligations remain open per method until independent tests pass; they do not all gate a smaller static slice. Do not introduce a second generic solver in Innovate.

## Scope preserved

Innovate remains the general library for modelling and predicting innovation: policy diffusion, adoption, competition, substitution, complements, hype effects, entry, repeat use, switching and abandonment. Medicines utilisation is a flagship consumer, not the product boundary. Hype means expectations/attention and their impact, not an assumption of exaggerated claims.

Innovation-specific game construction, likelihoods, estimation, payoff-model assumptions and forecast interpretation belong here. Generic strategic algorithms belong in Paignia; execution time/state/resources/RNG/replay belongs in Kairos. Static innovation games need not use Kairos. Dynamic interaction uses the optional adapter with exactly one simulation authority.

## Packet mapping and implementation steps

These PI labels refine existing packets, not additional completion counters:
PI01 maps to G01/G02; PI02 to G02/A08 and static G04; PI03 to E01/G03 and dynamic
G04; PI04 to E02–E04. Keep all task status in the owning plans.

### PI01 — resolve the provider for G01-G04/A08

No new solver implementation. First inspect the actual native workspace and current T17 public contract. Add a failing test in the relevant existing boundary suite asserting that the chosen consumer artifact identifies Paignia, the exact current package IDs, source/artifact hashes, supported game class, algorithm identity and numerical qualification receipt. Reject an unpublished planned package name presented as an installed dependency. Implement only the identity/receipt mapping, then rerun the test and record a nonzero count.

Use the existing proposed `crates/innovate-games` adapter boundary after its native-workspace prerequisite exists. Resolve and freeze its real Cargo package/test target before implementation; a nonexistent or filtered-out target is not acceptance. Keep each leaf to one behavior and at most five files.

### PI02 — qualify one innovation game, not a renamed toy

Depends on accepted T17 package/algorithm contracts. Build an Innovate-owned entry/launch/competition case with independently derived expected equilibria for a small parameter slice. Test action/player relabelling, dominated actions, multiple equilibria and absent convergence. Forward algorithm/selection/residual diagnostics unchanged. Consume the exact same packaged artifacts as a second real application; do not count two invocations of one harness as two consumers. Freeze exact paths and the `cargo test --locked -p <actual-package> --test <actual-target>` command in the implementing leaf before execution.

Static G02 qualification establishes the `innovate-games` domain adapter used
by A08. It does not wait for G03; the same crate gains optional dynamic support
later. Reimbursement and Innovate can be T17's two distinct real consumers.

### PI03 — native dynamic execution and dependency isolation

Link E01 and existing Kairos alignment acceptance. Write a test that fails if the game adapter advances a second clock, forks simulation state or uses an undisclosed host/subprocess path. Prove actual Kairos dispatch, resource conservation, deterministic event/RNG replay and unsupported-operation rejection. Separately prove static solving works without Kairos and production scientific execution without Python/pandas/xarray. Preserve isolated external numerical references only as test oracles.

### PI04 — valued joint outcomes and feedback

Link E02-E04 and Voiage issue #1286. Test joint draw/scenario/strategy IDs, correlations, units, population, time basis, discounting, perspective and fit validity through Arrow. Reject an adoption count or arbitrary game payoff supplied as utility without an explicit application valuation mapping. An independently calculated two-strategy EVPI case must match within its declared tolerance. Selected policies/study designs return through versioned artefacts and application orchestration; no Innovate-Voiage core dependency cycle.

## Existing research and hardening requirements remain

The composite hype joint-likelihood/native-Bayesian, impact/feedback, medicines stock-flow and empirical validation packets retain their definitions. This naming work does not modify model assumptions, raw source interpretation or priors. Preserve frontier/post-frontier hypotheses and failure reporting rather than making superiority claims from package identity.

Use Polars for supported tabular computation and Arrow/Parquet/versioned metadata for interchange; native numerical arrays remain appropriate internally. The next breaking supported release excludes pandas/xarray and host-compute fallback. Dependency alignment follows exact provider/consumer qualification, not name-only version matching or a fresh pin inferred from this document.

Acceptance receipts require exact source and dependency closure, artifact hashes, seeds, commands, nonzero test counts, residual/calibration tolerances and installed-native consumers. Beta/RC/hardened gates and all historical evidence remain unchanged. Provider fixes belong upstream; this supplement authorizes no automatic publication or deletion of compatibility paths.
