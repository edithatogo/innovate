//! Independent finite price-commitment controls using Innovate's demand model.

use innovate_demand::PreferenceParameters;
use innovate_games::{
    DuopolyPricingSpec, PricingOffer, build_price_commitment_game, solve_extensive_innovation_game,
};
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

fn assert_action_order(game: &Game, node: &NodeId) {
    let Node::Decision { edges, .. } = &game.nodes[node] else {
        panic!("expected a decision at {node:?}");
    };
    let actions: Vec<_> = edges.iter().map(|edge| edge.action.0.as_str()).collect();
    assert_eq!(actions, ["premium", "value"]);
}

fn assert_price_tree(game: &Game, case: &str) {
    // At prices 1 and 3, choice weights are 1/2 and 1/8. Each firm
    // sells 25 or 10 units when offers match. The mixed shares are 4/13
    // and 1/13, which give the other profits after costs.
    let rows = include_str!("fixtures/price_commitment_payoffs.tsv");
    let mut profiles = std::collections::BTreeSet::new();
    assert_eq!(game.nodes.len(), 7);
    assert_action_order(game, &game.root);
    for row in rows.lines().skip(1) {
        let columns: Vec<_> = row.split('\t').collect();
        let [fixture_case, commitment, response, leader, follower] = columns[..] else {
            panic!("invalid pricing oracle row: {row}");
        };
        if fixture_case != case {
            continue;
        }
        assert!(profiles.insert((commitment, response)));
        let continuation = decision_target(game, &game.root, "firm_a", commitment);
        assert_action_order(game, continuation);
        let terminal = decision_target(game, continuation, "firm_b", response);
        let Node::Terminal { payoffs } = &game.nodes[terminal] else {
            panic!("expected terminal profits for {commitment}/{response}");
        };
        assert_eq!(payoffs.len(), 2);
        assert_price_payoffs(payoffs, leader.parse().unwrap(), follower.parse().unwrap());
    }
    assert_eq!(profiles.len(), 4);
}

fn assert_complete_policy(
    specification: &GameSpecification,
    solved: &ResultEnvelope,
    root_action: &str,
    leader_profit: f64,
    follower_profit: f64,
) {
    let game = &specification.game;
    assert_eq!(solved.solutions.len(), 1);
    let policy = &solved.solutions[0];
    assert_eq!(policy.choices.len(), 3);
    for action in ["value", "premium"] {
        let continuation = decision_target(game, &game.root, "firm_a", action);
        assert_eq!(policy.choices[continuation], ActionId(action.into()));
    }
    assert_eq!(policy.choices[&game.root], ActionId(root_action.into()));
    assert_price_payoffs(&policy.expected_payoffs, leader_profit, follower_profit);
    let wrapper = solve_extensive_innovation_game(game, SolverConfig::default()).unwrap();
    assert_eq!(wrapper.extensive_choices, policy.choices);
    assert_eq!(wrapper.expected_payoffs, policy.expected_payoffs);
}

#[test]
fn follower_prices_contingently_and_leader_takes_the_premium_opportunity() {
    let specification = build(&spec(0.0));
    let solved = solve(&specification, "base", SolverConfig::default());
    assert_price_tree(&specification.game, "base");
    assert_complete_policy(&specification, &solved, "premium", 27.5, 29.5);
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
    // The same continuation responses yield 18.75 at value and 27.5 - 20 at premium.
    assert_price_tree(&specification.game, "cost");
    assert_complete_policy(&specification, &solved, "value", 18.75, 23.75);
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
