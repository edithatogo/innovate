//! Static strategic innovation game adapter backed by `paignia-core` and `paignia-solve`.
//!
//! This crate intentionally has no dependency on `innovate-kairos` or any
//! Kairos simulation runtime (`G02` / `A08` static adapter).

use std::collections::BTreeMap;
use std::fmt;

use innovate_demand::{ChoiceSet, InnovationError, PreferenceParameters, choice_shares};
use paignia_core::{
    ActionEdge, ActionId, GAME_SPEC_SCHEMA_VERSION, Game, GameSpecification, GameSpecificationId,
    Node, NodeId, PayoffVector, PlayerId,
};
use paignia_solve::{
    NormalFormGame, SolveError, SolverConfig, StrategyProfile, TiePolicy,
    backward_induction_with_config, pure_nash_equilibria,
};

/// Solver provider name recorded in [`AlgorithmIdentity`].
pub const PROVIDER_NAME: &str = "paignia-solve";
/// Version of the `innovate-games` adapter crate recorded in solution identity.
///
/// The resolved Paignia dependency revision is recorded separately by Cargo.
pub const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Finite game representation associated with a reported solution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameClass {
    /// Simultaneous choices represented by a strategy-profile payoff table.
    NormalForm,
    /// Sequential decisions represented by a perfect-information game graph.
    ExtensiveFormPerfectInformation,
}

/// Information available to players in the reported game class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InformationStructure {
    /// Players choose actions simultaneously.
    Simultaneous,
    /// Each decision node identifies the state observed by its acting player.
    PerfectInformation,
}

/// Requested equilibrium concept, including concepts rejected explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolutionConcept {
    /// Enumerate pure-strategy Nash equilibria of a finite normal-form game.
    PureNash,
    /// Obtain one perfect-information policy by backward induction under the solver tolerance.
    SubgamePerfectBackwardInduction,
    /// Mixed-strategy Nash request; unsupported by this adapter.
    MixedNash,
    /// Correlated-equilibrium request; unsupported by this adapter.
    CorrelatedEquilibrium,
}

/// Extent of the solution search declared by the selected adapter route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletenessClaim {
    /// Enumerate the pure profiles in the declared finite normal-form game.
    ///
    /// The scope is pure-strategy equilibria under the supplied solver tolerance.
    ExhaustiveWithinDeclaredScope,
    /// Return one backward-induction solution under the supplied tie policy.
    OneFound,
}

/// Outcome of a completed supported solver call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolutionStatus {
    /// At least one pure equilibrium or a backward-induction solution was returned.
    Solved,
    /// The pure-strategy search returned no equilibrium in its declared scope.
    NoEquilibriumFound,
}

/// Provider, algorithm and scope metadata attached to a static solution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmIdentity {
    /// Solver provider identifier, set to [`PROVIDER_NAME`].
    pub provider: &'static str,
    /// Provider entry point, either `pure_nash_equilibria` or `backward_induction`.
    pub algorithm: &'static str,
    /// Adapter crate version from [`PROVIDER_VERSION`].
    pub version: &'static str,
    /// Representation used for the solved game.
    pub game_class: GameClass,
    /// Information structure corresponding to the solver route.
    pub information_structure: InformationStructure,
    /// Equilibrium concept actually requested from the provider.
    pub solution_concept: SolutionConcept,
    /// Scope of the returned search result.
    pub completeness_claim: CompletenessClaim,
}

/// One firm's product, price and cost choice in a static pricing game.
///
/// Use a consistent monetary unit for price and costs. The preference
/// coefficients convert quality and price into dimensionless utility.
#[derive(Clone, Debug, PartialEq)]
pub struct PricingOffer {
    /// Strategy label; choose distinct labels within each firm's offer list.
    ///
    /// The builder uses these labels as payoff-table keys.
    pub action: String,
    /// Finite product quality in the unit used by the quality preference coefficient.
    pub quality: f64,
    /// Finite price per unit of demand, in the caller's monetary unit.
    pub price: f64,
    /// Finite variable cost per unit of demand, in the same unit as `price`.
    pub unit_cost: f64,
    /// Finite total fixed cost for selecting this offer, in the monetary unit.
    pub fixed_cost: f64,
}

/// Two firms' offers and shared demand assumptions for a simultaneous game.
#[derive(Clone, Debug, PartialEq)]
pub struct DuopolyPricingSpec {
    /// Positive finite potential demand, in units consistent with per-unit prices.
    pub market_size: f64,
    /// Shared coefficients for quality and price in multinomial-logit utility.
    pub preferences: PreferenceParameters,
    /// Nonempty action list assigned to player `firm_a`.
    pub firm_a_offers: Vec<PricingOffer>,
    /// Nonempty action list assigned to player `firm_b`.
    pub firm_b_offers: Vec<PricingOffer>,
}

/// Static solver result with an explicit solution concept and search scope.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticGameSolution {
    /// Provider entry point and the scope attached to this result.
    pub identity: AlgorithmIdentity,
    /// Whether the completed supported search returned a solution.
    pub status: SolutionStatus,
    /// Numeric comparison tolerance reported for the provider call.
    pub tolerance: f64,
    /// Provider tie-resolution policy used by the call.
    pub tie_policy: TiePolicy,
    /// Enumerated pure profiles for normal-form games; empty for extensive games.
    pub pure_equilibria: Vec<StrategyProfile>,
    /// Selected action by decision node for extensive games; empty for normal form.
    pub extensive_choices: BTreeMap<NodeId, ActionId>,
    /// Payoffs of the first returned pure profile, or the extensive-game root result.
    ///
    /// A normal-form result with no pure equilibrium has an empty map. Multiple
    /// pure equilibria are retained in `pure_equilibria`; their payoffs are not
    /// averaged into this field. Units are inherited from the supplied game.
    pub expected_payoffs: PayoffVector,
    /// Original normal-form payoff table; empty for an extensive-game solution.
    pub payoff_table: BTreeMap<StrategyProfile, PayoffVector>,
}

/// Input, demand-model, provider or unsupported-concept failure.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticGameError {
    /// Invalid market size, empty offer lists or non-finite offer costs.
    InvalidInput(String),
    /// Error returned by the native multinomial-logit demand calculation.
    DemandModel(InnovationError),
    /// Validation or execution error returned by Paignia.
    Solver(SolveError),
    /// A requested solution concept has no supported adapter route for this game.
    Unsupported {
        /// Rejected solution concept.
        concept: SolutionConcept,
        /// Explanation of the unsupported concept or representation combination.
        reason: String,
    },
}

impl fmt::Display for StaticGameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid static game input: {msg}"),
            Self::DemandModel(err) => write!(f, "demand model error: {err}"),
            Self::Solver(err) => write!(f, "solver error: {err}"),
            Self::Unsupported { concept, reason } => {
                write!(f, "unsupported solution concept {concept:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for StaticGameError {}

impl From<InnovationError> for StaticGameError {
    fn from(err: InnovationError) -> Self {
        Self::DemandModel(err)
    }
}

impl From<SolveError> for StaticGameError {
    fn from(err: SolveError) -> Self {
        Self::Solver(err)
    }
}

/// Build a two-firm simultaneous innovation pricing game backed by [`choice_shares`].
///
/// The players are `firm_a` and `firm_b`. For every pair of offers, the demand
/// calculation includes an outside option with utility zero. Firm `i` receives
/// payoff `market_size * share_i * (price_i - unit_cost_i) - fixed_cost_i`, in the
/// caller's monetary unit. The returned table contains the offer combinations;
/// equilibrium search is a separate call to [`solve_normal_form_innovation_game`].
///
/// # Errors
/// Returns [`StaticGameError::InvalidInput`] for a nonpositive or non-finite market
/// size, empty offer lists, or non-finite costs. Non-finite quality, price or
/// preference inputs are reported through [`StaticGameError::DemandModel`].
pub fn build_duopoly_pricing_game(
    spec: &DuopolyPricingSpec,
) -> Result<NormalFormGame, StaticGameError> {
    if !spec.market_size.is_finite() || spec.market_size <= 0.0 {
        return Err(StaticGameError::InvalidInput(
            "market_size must be finite and positive".to_owned(),
        ));
    }
    if spec.firm_a_offers.is_empty() || spec.firm_b_offers.is_empty() {
        return Err(StaticGameError::InvalidInput(
            "both firms must have at least one pricing offer".to_owned(),
        ));
    }

    let firm_a = PlayerId("firm_a".to_owned());
    let firm_b = PlayerId("firm_b".to_owned());

    let strategies_a: Vec<ActionId> = spec
        .firm_a_offers
        .iter()
        .map(|o| ActionId(o.action.clone()))
        .collect();
    let strategies_b: Vec<ActionId> = spec
        .firm_b_offers
        .iter()
        .map(|o| ActionId(o.action.clone()))
        .collect();

    let mut strategies = BTreeMap::new();
    strategies.insert(firm_a.clone(), strategies_a);
    strategies.insert(firm_b.clone(), strategies_b);

    let mut payoffs = BTreeMap::new();
    for offer_a in &spec.firm_a_offers {
        if !offer_a.unit_cost.is_finite() || !offer_a.fixed_cost.is_finite() {
            return Err(StaticGameError::InvalidInput(
                "offer costs must be finite".to_owned(),
            ));
        }
        for offer_b in &spec.firm_b_offers {
            if !offer_b.unit_cost.is_finite() || !offer_b.fixed_cost.is_finite() {
                return Err(StaticGameError::InvalidInput(
                    "offer costs must be finite".to_owned(),
                ));
            }
            let choices = ChoiceSet {
                quality: vec![offer_a.quality, offer_b.quality],
                price: vec![offer_a.price, offer_b.price],
            };
            let shares = choice_shares(&choices, &spec.preferences)?;
            let profit_a =
                spec.market_size * shares[0] * (offer_a.price - offer_a.unit_cost)
                    - offer_a.fixed_cost;
            let profit_b =
                spec.market_size * shares[1] * (offer_b.price - offer_b.unit_cost)
                    - offer_b.fixed_cost;

            let mut profile = StrategyProfile::new();
            profile.insert(firm_a.clone(), ActionId(offer_a.action.clone()));
            profile.insert(firm_b.clone(), ActionId(offer_b.action.clone()));

            let mut vec = PayoffVector::new();
            vec.insert(firm_a.clone(), profit_a);
            vec.insert(firm_b.clone(), profit_b);
            payoffs.insert(profile, vec);
        }
    }

    Ok(NormalFormGame {
        players: vec![firm_a, firm_b],
        strategies,
        payoffs,
    })
}

/// Build a perfect-information pricing commitment game from Innovate demand.
///
/// Firm A commits to a price and offer, then firm B observes that choice and
/// chooses its response. Terminal profits are the same demand and cost
/// valuations used by [`build_duopoly_pricing_game`]. Actions are ordered by
/// identity to keep the decision graph independent of menu presentation.
/// The returned specification carries the caller's identity for versioned
/// result admission through [`paignia_solve::backward_induction_result`].
///
/// # Errors
/// Returns a [`StaticGameError`] for invalid demand, offer or payoff data.
pub fn build_price_commitment_game(
    spec: &DuopolyPricingSpec,
    specification_id: GameSpecificationId,
) -> Result<GameSpecification, StaticGameError> {
    let mut market = build_duopoly_pricing_game(spec)?;
    market.validate()?;
    let leader = &market.players[0];
    let follower = &market.players[1];
    let mut leader_actions = market.strategies[leader].clone();
    let mut follower_actions = market.strategies[follower].clone();
    leader_actions.sort_unstable();
    follower_actions.sort_unstable();

    let root = NodeId("price-commitment".into());
    let mut nodes = BTreeMap::new();
    let mut commitments = Vec::with_capacity(leader_actions.len());
    for (leader_index, leader_action) in leader_actions.into_iter().enumerate() {
        let response = NodeId(format!("price-response/{leader_index}"));
        let mut responses = Vec::with_capacity(follower_actions.len());
        for (follower_index, follower_action) in follower_actions.iter().enumerate() {
            let profile = StrategyProfile::from([
                (leader.clone(), leader_action.clone()),
                (follower.clone(), follower_action.clone()),
            ]);
            let payoffs = market.payoffs.remove(&profile).ok_or_else(|| {
                StaticGameError::InvalidInput("missing profits for the proposed offers".into())
            })?;
            let target = NodeId(format!("pricing-profits/{leader_index}/{follower_index}"));
            nodes.insert(target.clone(), Node::Terminal { payoffs });
            responses.push(ActionEdge {
                action: follower_action.clone(),
                target,
            });
        }
        nodes.insert(
            response.clone(),
            Node::Decision {
                player: follower.clone(),
                edges: responses,
            },
        );
        commitments.push(ActionEdge {
            action: leader_action,
            target: response,
        });
    }
    nodes.insert(
        root.clone(),
        Node::Decision {
            player: leader.clone(),
            edges: commitments,
        },
    );
    Ok(GameSpecification {
        schema_version: GAME_SPEC_SCHEMA_VERSION.into(),
        id: specification_id.0,
        game: Game { root, nodes },
    })
}

/// Solve a static normal-form innovation game under the requested [`SolutionConcept`].
///
/// The supported route is [`SolutionConcept::PureNash`], delegated to Paignia.
/// All returned pure profiles are retained. An empty result returns
/// [`SolutionStatus::NoEquilibriumFound`] with an empty `expected_payoffs` map;
/// it does not establish the absence of mixed equilibria. With multiple pure
/// profiles, `expected_payoffs` contains the first profile's payoffs.
///
/// # Errors
/// Returns [`StaticGameError::Unsupported`] for mixed Nash, correlated equilibrium
/// or backward induction on a normal-form game. Provider validation and execution
/// failures are propagated through [`StaticGameError::Solver`].
pub fn solve_normal_form_innovation_game(
    game: &NormalFormGame,
    concept: SolutionConcept,
    config: SolverConfig,
) -> Result<StaticGameSolution, StaticGameError> {
    match concept {
        SolutionConcept::PureNash => {
            let equilibria = pure_nash_equilibria(game, config)?;
            let status = if equilibria.is_empty() {
                SolutionStatus::NoEquilibriumFound
            } else {
                SolutionStatus::Solved
            };
            let expected_payoffs = equilibria
                .first()
                .and_then(|profile| game.payoffs.get(profile).cloned())
                .unwrap_or_default();
            Ok(StaticGameSolution {
                identity: AlgorithmIdentity {
                    provider: PROVIDER_NAME,
                    algorithm: "pure_nash_equilibria",
                    version: PROVIDER_VERSION,
                    game_class: GameClass::NormalForm,
                    information_structure: InformationStructure::Simultaneous,
                    solution_concept: SolutionConcept::PureNash,
                    completeness_claim: CompletenessClaim::ExhaustiveWithinDeclaredScope,
                },
                status,
                tolerance: config.tolerance,
                tie_policy: config.tie_policy,
                pure_equilibria: equilibria,
                extensive_choices: BTreeMap::new(),
                expected_payoffs,
                payoff_table: game.payoffs.clone(),
            })
        }
        SolutionConcept::MixedNash | SolutionConcept::CorrelatedEquilibrium => {
            Err(StaticGameError::Unsupported {
                concept,
                reason: "paignia-solve does not yet qualify mixed-Nash or correlated equilibrium solvers".to_owned(),
            })
        }
        SolutionConcept::SubgamePerfectBackwardInduction => Err(StaticGameError::Unsupported {
            concept,
            reason: "backward induction requires an extensive-form perfect-information game".to_owned(),
        }),
    }
}

/// Select a policy for a finite, acyclic, perfect-information game through
/// tolerance-aware backward induction.
///
/// Returns the provider's chosen action map and root payoff vector, with
/// [`CompletenessClaim::OneFound`]. The supplied tie policy determines choices
/// when the provider regards alternatives as tied; this route does not enumerate
/// all equilibria. Normal-form profile and payoff-table fields are empty.
///
/// # Errors
/// Provider validation and execution failures, including invalid games, invalid
/// tolerances and exceeded solver limits, are propagated through
/// [`StaticGameError::Solver`].
pub fn solve_extensive_innovation_game(
    game: &Game,
    config: SolverConfig,
) -> Result<StaticGameSolution, StaticGameError> {
    let solved = backward_induction_with_config(game, config)?;
    Ok(StaticGameSolution {
        identity: AlgorithmIdentity {
            provider: PROVIDER_NAME,
            algorithm: "backward_induction",
            version: PROVIDER_VERSION,
            game_class: GameClass::ExtensiveFormPerfectInformation,
            information_structure: InformationStructure::PerfectInformation,
            solution_concept: SolutionConcept::SubgamePerfectBackwardInduction,
            completeness_claim: CompletenessClaim::OneFound,
        },
        status: SolutionStatus::Solved,
        tolerance: solved.tolerance,
        tie_policy: solved.tie_policy,
        pure_equilibria: Vec::new(),
        extensive_choices: solved.choices,
        expected_payoffs: solved.expected_payoffs,
        payoff_table: BTreeMap::new(),
    })
}
