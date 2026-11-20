//! Independent finite price-commitment controls using Innovate's demand model.

use innovate_demand::PreferenceParameters;
use innovate_games::{DuopolyPricingSpec, PricingOffer, build_price_commitment_game};
use paignia_core::{
    ActionId, BACKWARD_INDUCTION_ALGORITHM_VERSION, CatalogueClaim, Game, GameInstanceId,
    GameSpecification, GameSpecificationId, Node, NodeId, PayoffVector, PlayerId,
    RESULT_IDENTITY_SCHEMA_VERSION, ResultAlgorithm, ResultEnvelope, ResultIdentityRecord,
    ResultSelectionRule, ResultSolutionConcept, ResultStatus, STATIC_GAME_PROFILE_V1, SolverRunId,
};
use paignia_solve::{SolverConfig, backward_induction_result};

fn spec(premium_launch_cost: f64) -> DuopolyPricingSpec {
    let offer = |action: &str, price, unit_cost, fixed_cost| PricingOffer {
        action: action.to_owned(),
        quality: 0.0,
        price,
        unit_cost,
        fixed_cost,
    };
    DuopolyPricingSpec {
        market_size: 100.0,
        preferences: PreferenceParameters {
            alpha_quality: 0.0,
            beta_price: -std::f64::consts::LN_2,
        },
        firm_a_offers: vec![
            offer("value", 1.0, 0.25, 0.0),
            offer("premium", 3.0, 0.25, premium_launch_cost),
        ],
        firm_b_offers: vec![
            offer("value", 1.0, 0.05, 0.0),
            offer("premium", 3.0, 0.05, 0.0),
        ],
    }
}

fn build(spec: &DuopolyPricingSpec) -> GameSpecification {
    let id = GameSpecificationId("urn:innovate:price-commitment:1".into());
    build_price_commitment_game(spec, id).unwrap()
}

fn solve(specification: &GameSpecification, case: &str, config: SolverConfig) -> ResultEnvelope {
    let identities = ResultIdentityRecord {
        schema_version: RESULT_IDENTITY_SCHEMA_VERSION.into(),
        specification_id: GameSpecificationId(specification.id.clone()),
        game_instance_id: GameInstanceId(format!("urn:innovate:price-commitment:{case}")),
        solver_run_id: SolverRunId(format!("urn:innovate:benchmark:{case}:1")),
        session_id: None,
        policy_artifact_id: None,
        solver_trace_id: None,
    };
    backward_induction_result(specification, STATIC_GAME_PROFILE_V1, identities, config)
        .unwrap()
        .envelope
}

fn decision_target<'a>(game: &'a Game, node: &NodeId, player: &str, action: &str) -> &'a NodeId {
    let Node::Decision {
        player: acting,
        edges,
    } = &game.nodes[node]
    else {
        panic!("expected player {player} to act");
    };
    assert_eq!(acting.0, player);
    &edges
        .iter()
        .find(|edge| edge.action.0 == action)
        .unwrap()
        .target
}

fn assert_price_payoffs(payoffs: &PayoffVector, leader: f64, follower: f64) {
    assert!((payoffs[&PlayerId("firm_a".into())] - leader).abs() < 1e-10);
    assert!((payoffs[&PlayerId("firm_b".into())] - follower).abs() < 1e-10);
}

#[test]
fn follower_prices_contingently_and_leader_takes_the_premium_opportunity() {
    let specification = build(&spec(0.0));
    let game = &specification.game;
    let solved = solve(&specification, "base", SolverConfig::default());

    // At prices 1 and 3, unnormalized choice weights are 1/2 and 1/8.
    // At equal offers each firm sells 25 units at value and 10 units at premium.
    // The low and high shares are 4/13 and 1/13 when prices differ.
    for (commitment, response, leader, follower) in [
        ("value", "value", 75.0 / 4.0, 95.0 / 4.0),
        ("value", "premium", 300.0 / 13.0, 295.0 / 13.0),
        ("premium", "value", 275.0 / 13.0, 380.0 / 13.0),
        ("premium", "premium", 55.0 / 2.0, 59.0 / 2.0),
    ] {
        let continuation = decision_target(game, &game.root, "firm_a", commitment);
        let terminal = decision_target(game, continuation, "firm_b", response);
        let Node::Terminal { payoffs } = &game.nodes[terminal] else {
            panic!("expected terminal profits for {commitment}/{response}");
        };
        assert_price_payoffs(payoffs, leader, follower);
    }

    let policy = &solved.solutions[0];
    for action in ["value", "premium"] {
        let continuation = decision_target(game, &game.root, "firm_a", action);
        assert_eq!(policy.choices[continuation], ActionId(action.into()));
    }
    assert_eq!(policy.choices[&game.root], ActionId("premium".into()));
    assert_price_payoffs(&policy.expected_payoffs, 27.5, 29.5);
    assert_eq!(solved.semantic_profile, STATIC_GAME_PROFILE_V1);
    assert_eq!(solved.status, ResultStatus::Completed);
    assert_eq!(solved.catalogue, CatalogueClaim::OneFound);
    assert_eq!(
        solved.algorithm.algorithm,
        ResultAlgorithm::BackwardInduction
    );
    assert_eq!(
        solved.algorithm.version,
        BACKWARD_INDUCTION_ALGORITHM_VERSION
    );
    assert_eq!(solved.identities.specification_id.0, specification.id);
}

#[test]
fn premium_launch_cost_changes_the_leaders_commitment() {
    let specification = build(&spec(20.0));
    let solved = solve(&specification, "cost", SolverConfig::default());
    let policy = &solved.solutions[0];

    // The same continuation responses yield 18.75 at value and 27.5 - 20 at premium.
    assert_eq!(
        policy.choices[&specification.game.root],
        ActionId("value".into())
    );
    assert_price_payoffs(&policy.expected_payoffs, 18.75, 23.75);
}

#[test]
fn price_commitment_rejects_an_ambiguous_action_menu() {
    let mut invalid = spec(0.0);
    invalid.firm_b_offers[1].action = invalid.firm_b_offers[0].action.clone();
    let id = GameSpecificationId("urn:innovate:price-commitment:1".into());
    assert!(build_price_commitment_game(&invalid, id).is_err());
}

#[test]
fn reversing_the_menus_preserves_the_game_and_action_identities() {
    let original = spec(0.0);
    let mut reordered = original.clone();
    reordered.firm_a_offers.reverse();
    reordered.firm_b_offers.reverse();
    assert_eq!(build(&original), build(&reordered));
}

#[test]
fn tolerated_follower_choice_retains_the_providers_selection_contract() {
    let specification = build(&spec(0.0));
    let game = &specification.game;
    let config = SolverConfig {
        tolerance: 2.0,
        ..SolverConfig::default()
    };
    let solved = solve(&specification, "tolerance", config);
    let response = decision_target(game, &game.root, "firm_a", "value");

    // A tolerance of 2 admits the 55/52 continuation-profit gap, so the
    // provider's lexicographic tie rule selects premium after value.
    assert_eq!(
        solved.solutions[0].choices[response],
        ActionId("premium".into())
    );
    assert_eq!(solved.tolerance.to_bits(), 2.0_f64.to_bits());
    assert_eq!(
        solved.solution_concept,
        ResultSolutionConcept::ToleranceAwareBackwardInductionPolicy
    );
    assert_eq!(
        solved.selection_rule,
        ResultSelectionRule::InputOrderPairwiseToleranceThenLexicographic
    );
}
