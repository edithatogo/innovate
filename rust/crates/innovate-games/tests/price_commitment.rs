//! Independent finite price-commitment controls using Innovate's demand model.

use innovate_core::models::demand::PreferenceParameters;
use innovate_games::{
    CompletenessClaim, DuopolyPricingSpec, PricingOffer, SolutionConcept, StaticGameSolution,
    build_price_commitment_game, solve_extensive_innovation_game,
};
use paignia_core::{ActionId, Game, Node, PlayerId};
use paignia_solve::SolverConfig;

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

fn assert_response(game: &Game, solved: &StaticGameSolution, commitment: &str, response: &str) {
    let Node::Decision { edges, .. } = &game.nodes[&game.root] else {
        panic!("expected the leader to commit to a price");
    };
    let edge = edges
        .iter()
        .find(|edge| edge.action.0 == commitment)
        .unwrap();
    assert_eq!(
        solved.extensive_choices[&edge.target],
        ActionId(response.into())
    );
}

fn assert_price_payoffs(solved: &StaticGameSolution, leader: f64, follower: f64) {
    assert!((solved.expected_payoffs[&PlayerId("firm_a".into())] - leader).abs() < 1e-10);
    assert!((solved.expected_payoffs[&PlayerId("firm_b".into())] - follower).abs() < 1e-10);
}

#[test]
fn follower_prices_contingently_and_leader_takes_the_premium_opportunity() {
    let game = build_price_commitment_game(&spec(0.0)).unwrap();
    let solved = solve_extensive_innovation_game(&game, SolverConfig::default()).unwrap();

    // At prices 1 and 3, unnormalized choice weights are 1/2 and 1/8.
    // At equal offers each firm sells 25 units at value and 10 units at premium.
    // At either asymmetric profile the shares are 4/13 and 1/13. These give
    // firm B's four continuation profits and distinguish both responses.
    assert_response(&game, &solved, "value", "value");
    assert_response(&game, &solved, "premium", "premium");
    assert_eq!(
        solved.extensive_choices[&game.root],
        ActionId("premium".into())
    );
    assert_price_payoffs(&solved, 27.5, 29.5);
    assert_eq!(
        solved.identity.solution_concept,
        SolutionConcept::SubgamePerfectBackwardInduction
    );
    assert_eq!(
        solved.identity.completeness_claim,
        CompletenessClaim::OneFound
    );
}

#[test]
fn premium_launch_cost_changes_the_leaders_commitment() {
    let game = build_price_commitment_game(&spec(20.0)).unwrap();
    let solved = solve_extensive_innovation_game(&game, SolverConfig::default()).unwrap();

    // The same continuation responses yield 18.75 at value and 27.5 - 20 at premium.
    assert_eq!(
        solved.extensive_choices[&game.root],
        ActionId("value".into())
    );
    assert_price_payoffs(&solved, 18.75, 23.75);
}

#[test]
fn price_commitment_rejects_an_ambiguous_action_menu() {
    let mut invalid = spec(0.0);
    invalid.firm_b_offers[1].action = invalid.firm_b_offers[0].action.clone();
    assert!(build_price_commitment_game(&invalid).is_err());
}

#[test]
fn reversing_the_menus_preserves_the_game_and_action_identities() {
    let original = spec(0.0);
    let mut reordered = original.clone();
    reordered.firm_a_offers.reverse();
    reordered.firm_b_offers.reverse();
    assert_eq!(
        build_price_commitment_game(&original).unwrap(),
        build_price_commitment_game(&reordered).unwrap(),
    );
}
