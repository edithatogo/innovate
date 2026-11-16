//! Static strategic innovation game adapter backed by `paignia-core` and `paignia-solve`.
//!
//! This crate intentionally has no dependency on `innovate-kairos` or any
//! Kairos simulation runtime (`G02` / `A08` static adapter).

use std::collections::BTreeMap;
use std::fmt;

use innovate_core::inference::InnovationError;
use innovate_core::models::demand::{ChoiceSet, PreferenceParameters, choice_shares};
use paignia_core::{ActionId, Game, NodeId, PayoffVector, PlayerId};
use paignia_solve::{
    NormalFormGame, SolveError, SolverConfig, StrategyProfile, TiePolicy,
    backward_induction_with_config, pure_nash_equilibria,
};

pub const PROVIDER_NAME: &str = "paignia-solve";
pub const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameClass {
    NormalForm,
    ExtensiveFormPerfectInformation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InformationStructure {
    Simultaneous,
    PerfectInformation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolutionConcept {
    PureNash,
    SubgamePerfectBackwardInduction,
    MixedNash,
    CorrelatedEquilibrium,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletenessClaim {
    ExhaustiveWithinDeclaredScope,
    OneFound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolutionStatus {
    Solved,
    NoEquilibriumFound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmIdentity {
    pub provider: &'static str,
    pub algorithm: &'static str,
    pub version: &'static str,
    pub game_class: GameClass,
    pub information_structure: InformationStructure,
    pub solution_concept: SolutionConcept,
    pub completeness_claim: CompletenessClaim,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PricingOffer {
    pub action: String,
    pub quality: f64,
    pub price: f64,
    pub unit_cost: f64,
    pub fixed_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DuopolyPricingSpec {
    pub market_size: f64,
    pub preferences: PreferenceParameters,
    pub firm_a_offers: Vec<PricingOffer>,
    pub firm_b_offers: Vec<PricingOffer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StaticGameSolution {
    pub identity: AlgorithmIdentity,
    pub status: SolutionStatus,
    pub tolerance: f64,
    pub tie_policy: TiePolicy,
    pub pure_equilibria: Vec<StrategyProfile>,
    pub extensive_choices: BTreeMap<NodeId, ActionId>,
    pub expected_payoffs: PayoffVector,
    pub payoff_table: BTreeMap<StrategyProfile, PayoffVector>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StaticGameError {
    InvalidInput(String),
    DemandModel(InnovationError),
    Solver(SolveError),
    Unsupported {
        concept: SolutionConcept,
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

/// Build a two-firm simultaneous innovation pricing game backed by `innovate_core::models::demand::choice_shares`.
///
/// # Errors
/// Returns [`StaticGameError`] on empty strategy sets, non-finite market size/costs, or invalid demand parameters.
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

/// Solve a static normal-form innovation game under the requested [`SolutionConcept`].
///
/// # Errors
/// Returns [`StaticGameError::Unsupported`] for concepts not yet qualified in `paignia-solve`
/// (e.g. `MixedNash`, `CorrelatedEquilibrium`, or `SubgamePerfectBackwardInduction` on a normal-form game).
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

/// Solve a finite, acyclic, perfect-information extensive-form innovation game via backward induction.
///
/// # Errors
/// Returns [`StaticGameError`] on invalid games, non-finite tolerances, or solver limits.
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
