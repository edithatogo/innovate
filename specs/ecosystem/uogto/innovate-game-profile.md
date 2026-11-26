# Innovate finite pricing commitment

I07 reuses Paignia’s static perfect-information profile and the
[pinned UOGTO vocabulary](https://github.com/edithatogo/UOGTO/tree/cc0c84201929996580bb4af98774cf2145b224ae).
The shared intake key is `finite-perfect-information`. The specification
identity is `urn:innovate:price-commitment:1`. The paired fixtures preserve
distinct state, action, transition, instance and run identities.

Firm A commits to a posted value or premium price. Firm B observes that price
and chooses its own offer. Each of Firm B’s decisions has its own information
set. Buyers may select either offer or an outside option. Innovate’s
multinomial-logit demand and contribution margins produce both firms’ static
profits. The synthetic case uses 100 units of demand, equal quality and
a price coefficient of `−ln(2)`. The stress case adds 20 to Firm A’s premium
launch cost. [Independent profit derivation and native test scope](../../../conductor/tracks/kairos_runtime_boundary_20261107/strategic-information-integration.md#pricing-commitment-intake)

The semantic disposition is `reuse_existing`. The prefix `gt:` denotes
`https://w3id.org/uogto/core#`. Paignia's projection terms use
`https://github.com/edithatogo/paignia#`.

| Model role | Declaration |
| --- | --- |
| Game and sequence | `gt:GameSpecification`, `gt:ExtensiveFormRepresentation`, `gt:hasRepresentation`, `gt:hasInitialState`, `gt:hasState` |
| Firms and observations | `gt:Player`, `gt:hasPlayer`, `gt:controlsNode`, `gt:DecisionNode`, `gt:InformationSet`, `gt:belongsToInformationSet` |
| Prices and continuations | `gt:Action`, `gt:hasAvailableAction`, `gt:Transition`, `gt:hasTransition`, `gt:transitionsFrom`, `gt:transitionsTo`. The projection attaches `position` and `action` to the transition. |
| Terminal profits | `gt:TerminalNode`, `gt:PayoffProfile`, `gt:PlayerPayoffLink`, `gt:hasPayoffForPlayer`, `gt:payoffPlayer`, `gt:payoffValue`. The projection's `payoffProfile` attaches the full payoff vector to its terminal. |

The provider disposition is `reuse_existing`. The standalone native adapter
uses `backward_induction_result` at signed Paignia `v0.1.0` source
`a04a19da06cc50e310a172574ed66f62000b891a`. Its normal and build dependencies
exclude the simulation runtime. The rollback is the tested `bd880dd` pin
and its matching game crate lock. The verified policy chooses the premium
price for profits 27.50 and 29.50 in the baseline. The launch-cost case
chooses value for profits 18.75 and 23.75. Both include Firm B’s reply to
every observed commitment.

[Native CI](https://github.com/edithatogo/innovate/actions/workflows/ci.yml)
checks the application’s own payoff construction and replays the two
declared graphs with the signed provider. The retained evidence covers all
choices, both profits, source bindings and asserted RDF output. The
perfect-information projection rejects a shared follower information set.
[Baseline graph](pricing-v1/base.nt) · [Launch-cost graph](pricing-v1/cost.nt)

## Caller dispositions

| Caller family | Disposition |
| --- | --- |
| `solve_normal_form_innovation_game` | Finite Nash pricing uses Paignia's pure solver. Innovate owns ordered offers and demand-derived payoffs. |
| `solve_extensive_innovation_game` | The observed-price commitment uses Paignia’s contingent policy. This receipt covers the two synthetic cost cases. |
| Demand, Bass and logistic adoption | The application owns demand estimation and diffusion trajectories. Their runtime and independent prediction contracts are distinct from the finite equilibrium interface. |
| Continuous competing adoption | The separate trajectory reference needs its own matched state, rate and time contract. [Innovate #724](https://github.com/edithatogo/innovate/issues/724) |
