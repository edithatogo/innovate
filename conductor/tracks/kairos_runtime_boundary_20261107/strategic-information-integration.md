# Strategic evidence, incentives and adoption integration

Date: 8 November 2026. Status: planning incorporated; implementation and scientific
acceptance pending. Owners remain I07 and the PR570 G01–G04/A08 programme. This
addendum does not take over PR570, alter its branch or qualify a native consumer.

## Parallel-owner reconciliation

Read the [shared reconciliation](https://github.com/edithatogo/paignia/blob/87b4bda345b85310e85239d4845b0f85bf1c6439/docs/integrations/parallel-plan-reconciliation-20261108.md)
and the current [PR #570](https://github.com/edithatogo/innovate/pull/570). Its 52
packets and broad innovation/policy-diffusion, competition, substitution and hype
programme remain authoritative; medicines and the pilot here are applications,
not a narrower product definition. Its new supported scientific core targets Rust
and Polars/Arrow without a host-compute or pandas/xarray fallback; this note does
not claim current hybrid source already meets that target.

G02 admits a qualified static game adapter. A08 and static G04 do not depend on
optional G03 dynamic execution. I10 remains independent of optional I07. Paignia
is the provider: its bounded native seed merged in PR #2 at
`12b22ce7571b9c590405bf53c2c1c5862b49cd64`; released artifacts and real-consumer
qualification remain separate. T17.10 is the selected second application's
receipt, not another local provider task. Conditional Voiage feedback uses
versioned study/policy artifacts through application orchestration, never reverse
core imports. Existing provider pins and the #570 branch are untouched.

Reuse Voiage's O01–O08 paired-input contracts and SI tests. Parameter/model,
measurement, simulation-replicate and solver identities must survive export.
The shared one-parameter-state noise control A=[-1,1], B=[1,-1] has parameter
EVPI zero, not one; Voiage owns that arithmetic and Innovate owns truthful inputs.
Forecast fit, causal intervention evidence and equilibrium assumptions remain
separate. Source links here are planning references, not dependency pins.

## Independent boundaries

Innovate owns adoption/diffusion/incentive models, inference and diagnostics.
The reusable Rust game packages supply qualified strategic response methods;
UOGTO supplies semantics. Kairos is optional when the model needs dynamic execution.
Closed-form forecasts and static games must not acquire a required scheduler.
Voiage evaluates valued evidence outcomes; HEOR valuation stays separately declared.
No copied solver, VOI algorithm or host-language simulation fallback is introduced.

Preserve I10's explicit native-runtime prerequisites and its independence from I07.
An installed native-runtime receipt does not qualify a game; static game acceptance
does not prove dynamic execution. T17 can qualify a bounded static release with
reimbursement plus Innovate as two real applications using the same artifacts.
Do not require CareOps or full strategic VOI to release a non-game consumer.

## Pending work within the existing game and outcome leaves

- [ ] Choose one bounded, independently solvable innovation-adoption/incentive game.
  Declare actors, information sets, response concept, selection policy, payoffs and
  available evidence/disclosure actions. Keep empirical priors separate from private
  observations: an analyst's uncertainty draw is not shared knowledge among agents.
- [ ] Use the accepted provider representation and result contract. Preserve solver
  version, status, residuals and catalogue scope. Unknown concepts, nonconvergence
  and stale artifacts fail explicitly; do not substitute forecasts or printed imports
  for actual game execution. Bayesian/CE/CFR labels require their own qualification.
- [ ] Reuse the paired-outcome contract for strategy/draw IDs, weights, common-world
  coupling, adoption trajectories, time units, horizon and provenance. Keep adoption
  counts, resource use, costs, payments and health/value outcomes distinguishable.
  Retain the source and assumptions for any trajectory-to-payoff transformation.
- [ ] Qualify a thin optional Voiage adapter against installed artifacts outside both
  source trees. Preserve actor and social signed values, transfer accounting and
  declared welfare weights. Do not add a default Voiage/game dependency to generic
  forecasting or a reverse dependency from a provider into Innovate.
- [ ] For dynamic models only, require actual Rust Kairos dispatch, permitted policy
  observations, solver/exogenous RNG isolation and joint policy/native replay. Changing
  solver iterations must not advance unrelated adoption/random-event streams.

## Pilot and evidence

Extend the existing Bass-to-Voiage pilot rather than duplicating it, but retain its
bounded synthetic scope. Compare fixed responses and recomputed responses under
research-only, disclosure-only, incentive-only and joint designs where well defined.
Forecasting fit against uptake time series does not identify an intervention's causal
effect. State causal/behavioural assumptions and test sensitivity or label the result
as scenario analysis. Keep held-out cases separate from tuning.

Use exact small games first: matching pennies for supported mixed solving, a
coordination game for selection/multiplicity, and indistinguishable hidden states
for observation discipline. Check broken pairings, incompatible time/unit metadata,
negative information value and solver failure. Report decision regret, sampling and
solver errors separately; measure the full composed workflow at matched accuracy.

Before implementation, freeze mapping/version and concrete tests in the existing
owner's bounded packet; demonstrate failure, implement, run focused and required
repository checks, and retain source/artifact hashes. No pin changes, capability
promotion, publication or completed-task marks occur in this planning addition.

## T17.10 static price commitment

The bounded consumer case starts from `fd7e259bb9a685d620f80907c6815d4837da3e53`.
It adds a sequential game to the existing `innovate-games` adapter. The first
firm commits to one of its offers. The second firm observes the commitment
and chooses its own offer. Innovate's native multinomial-logit shares and
margin calculations supply both firms' terminal profits. Paignia returns
a contingent policy with its versioned tolerance and selection rule.

For equal qualities, the synthetic case uses 100 demand units and a price
coefficient of `−ln(2)`. The offers at prices 1 and 3 receive weights `1/2`
and `1/8`. With the outside-option weight 1, the independent shares are
`1/4` each at low prices, `1/10` each at high prices, and `4/13` and `1/13`
when prices differ. The follower chooses the same price as the leader.
The leader chooses the premium price for profit 27.50 and follower profit
29.50. A premium fixed cost of 20 changes the commitment to the value price,
with profits 18.75 and 23.75. The controls check all four payoff vectors,
action identity under menu reordering, duplicate offers and the declared
policy selection when the tolerance covers a payoff gap. The assumptions
are for static scenario analysis using synthetic inputs.

Rust 1.99.1 passed eight adapter tests through
`cargo test --locked --manifest-path rust/crates/innovate-games/Cargo.toml --all-targets`.
The existing competition and public contract tests also passed. The adapter shares
the native demand implementation through `innovate-demand`. The core
reexports the same types and calculation for its existing callers. The
adapter consumes `bd880dd` and its locked normal and build closure excludes
the simulation runtime. Hosted CI tests the closure and keeps the moved
demand source in the existing coverage gate. Qualification of the exact
packaged archives remains separate.

The artifact runner uses Rust 1.99.1 and the repository's CPython 3.15.2
environment. Put the qualified Paignia archives and sparse registry cache
within reach of the runner and choose a staging directory outside the
application and provider checkouts. Retrieve the pinned admission helper
from the reimbursement checkout. The runner checks the helper's SHA-256.

```sh
git -C "$reimbursement_checkout" show \
  b669bd00e78757029e2b9e7d26d86a07e20bed43:scripts/validate_paignia_consumer.py \
  > "$evidence_directory/reimbursement-verifier.py"
"$project_python" scripts/qualify_paignia_consumer.py \
  --application "$PWD" --revision "$application_revision" \
  --provider "$provider_checkout" --handoff "$handoff_json" \
  --archives "$qualified_archives" --registry "$cargo_registry_cache" \
  --work-dir "$evidence_directory" \
  --admission-helper "$evidence_directory/reimbursement-verifier.py"
```

The runner passes the locked package resolution and native test commands
through one macOS sandbox. It records the archive pair, resolved metadata,
lockfile, source-read controls and test output in the new staging directory.

## Coordinated Voiage plan

See the [shared SI01–SI04 plan and SI-V01–SI-V08 cases](https://github.com/edithatogo/voiage/blob/c414752a228e7af5d689d9d5876f8f7372c42447/conductor/tracks/scientific_ecosystem_alignment_20261029/strategic-information-integration.md).
That source commit is a planning proposal, not a dependency pin or an acceptance receipt.
