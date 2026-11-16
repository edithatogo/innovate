use std::collections::BTreeMap;

use innovate_core::models::demand::PreferenceParameters;
use innovate_games::{
    CompletenessClaim, DuopolyPricingSpec, GameClass, InformationStructure, PROVIDER_NAME,
    PricingOffer, SolutionConcept, SolutionStatus, StaticGameError, build_duopoly_pricing_game,
    solve_extensive_innovation_game, solve_normal_form_innovation_game,
};
use paignia_core::{ActionEdge, ActionId, Game, Node, NodeId, PayoffVector, PlayerId};
use paignia_solve::SolverConfig;

#[test]
fn static_duopoly_pricing_game_reports_honest_identity_and_pure_nash() {
    let spec = DuopolyPricingSpec {
        market_size: 1000.0,
        preferences: PreferenceParameters {
            alpha_quality: 1.5,
            beta_price: -1.2,
        },
        firm_a_offers: vec![
            PricingOffer {
                action: "low_price".to_owned(),
                quality: 1.0,
                price: 1.0,
                unit_cost: 0.4,
                fixed_cost: 50.0,
            },
            PricingOffer {
                action: "high_price".to_owned(),
                quality: 1.0,
                price: 2.5,
                unit_cost: 0.4,
                fixed_cost: 50.0,
            },
        ],
        firm_b_offers: vec![
            PricingOffer {
                action: "low_price".to_owned(),
                quality: 1.0,
                price: 1.0,
                unit_cost: 0.4,
                fixed_cost: 50.0,
            },
            PricingOffer {
                action: "high_price".to_owned(),
                quality: 1.0,
                price: 2.5,
                unit_cost: 0.4,
                fixed_cost: 50.0,
            },
        ],
    };

    let game = build_duopoly_pricing_game(&spec).expect("valid pricing spec");
    let sol = solve_normal_form_innovation_game(
        &game,
        SolutionConcept::PureNash,
        SolverConfig::default(),
    )
    .expect("pure Nash solver succeeds");

    assert_eq!(sol.identity.provider, PROVIDER_NAME);
    assert_eq!(sol.identity.algorithm, "pure_nash_equilibria");
    assert_eq!(sol.identity.game_class, GameClass::NormalForm);
    assert_eq!(
        sol.identity.information_structure,
        InformationStructure::Simultaneous
    );
    assert_eq!(sol.identity.solution_concept, SolutionConcept::PureNash);
    assert_eq!(
        sol.identity.completeness_claim,
        CompletenessClaim::ExhaustiveWithinDeclaredScope
    );
    assert_eq!(sol.status, SolutionStatus::Solved);
    assert!(!sol.pure_equilibria.is_empty());
    assert_eq!(sol.payoff_table.len(), 4);
}

#[test]
fn unsupported_solution_concepts_fail_explicitly() {
    let spec = DuopolyPricingSpec {
        market_size: 100.0,
        preferences: PreferenceParameters {
            alpha_quality: 1.0,
            beta_price: -1.0,
        },
        firm_a_offers: vec![PricingOffer {
            action: "launch".to_owned(),
            quality: 1.0,
            price: 1.5,
            unit_cost: 0.5,
            fixed_cost: 0.0,
        }],
        firm_b_offers: vec![PricingOffer {
            action: "launch".to_owned(),
            quality: 1.0,
            price: 1.5,
            unit_cost: 0.5,
            fixed_cost: 0.0,
        }],
    };
    let game = build_duopoly_pricing_game(&spec).expect("valid spec");

    for unsupported in [
        SolutionConcept::MixedNash,
        SolutionConcept::CorrelatedEquilibrium,
        SolutionConcept::SubgamePerfectBackwardInduction,
    ] {
        let err = solve_normal_form_innovation_game(&game, unsupported, SolverConfig::default())
            .expect_err("unsupported concept must error explicitly");
        assert!(matches!(
            err,
            StaticGameError::Unsupported { concept, .. } if concept == unsupported
        ));
    }
}

#[test]
fn sequential_launch_tree_solves_via_backward_induction() {
    let incumbent = PlayerId("incumbent".to_owned());
    let entrant = PlayerId("entrant".to_owned());

    let mut nodes = BTreeMap::new();
    nodes.insert(
        NodeId("root".to_owned()),
        Node::Decision {
            player: entrant.clone(),
            edges: vec![
                ActionEdge {
                    action: ActionId("stay_out".to_owned()),
                    target: NodeId("out".to_owned()),
                },
                ActionEdge {
                    action: ActionId("enter".to_owned()),
                    target: NodeId("incumbent_turn".to_owned()),
                },
            ],
        },
    );
    nodes.insert(
        NodeId("incumbent_turn".to_owned()),
        Node::Decision {
            player: incumbent.clone(),
            edges: vec![
                ActionEdge {
                    action: ActionId("accommodate".to_owned()),
                    target: NodeId("duopoly".to_owned()),
                },
                ActionEdge {
                    action: ActionId("price_war".to_owned()),
                    target: NodeId("war".to_owned()),
                },
            ],
        },
    );

    let mut out_payoffs = PayoffVector::new();
    out_payoffs.insert(entrant.clone(), 0.0);
    out_payoffs.insert(incumbent.clone(), 10.0);
    nodes.insert(
        NodeId("out".to_owned()),
        Node::Terminal {
            payoffs: out_payoffs,
        },
    );

    let mut duopoly_payoffs = PayoffVector::new();
    duopoly_payoffs.insert(entrant.clone(), 4.0);
    duopoly_payoffs.insert(incumbent.clone(), 5.0);
    nodes.insert(
        NodeId("duopoly".to_owned()),
        Node::Terminal {
            payoffs: duopoly_payoffs,
        },
    );

    let mut war_payoffs = PayoffVector::new();
    war_payoffs.insert(entrant.clone(), -2.0);
    war_payoffs.insert(incumbent.clone(), 1.0);
    nodes.insert(
        NodeId("war".to_owned()),
        Node::Terminal {
            payoffs: war_payoffs,
        },
    );

    let game = Game {
        root: NodeId("root".to_owned()),
        nodes,
    };

    let sol = solve_extensive_innovation_game(&game, SolverConfig::default())
        .expect("backward induction succeeds");

    assert_eq!(sol.identity.algorithm, "backward_induction");
    assert_eq!(
        sol.identity.game_class,
        GameClass::ExtensiveFormPerfectInformation
    );
    assert_eq!(
        sol.identity.information_structure,
        InformationStructure::PerfectInformation
    );
    assert_eq!(
        sol.identity.solution_concept,
        SolutionConcept::SubgamePerfectBackwardInduction
    );
    assert_eq!(
        sol.identity.completeness_claim,
        CompletenessClaim::OneFound
    );
    assert_eq!(
        sol.extensive_choices.get(&NodeId("root".to_owned())),
        Some(&ActionId("enter".to_owned()))
    );
    assert_eq!(
        sol.extensive_choices
            .get(&NodeId("incumbent_turn".to_owned())),
        Some(&ActionId("accommodate".to_owned()))
    );
    assert_eq!(sol.expected_payoffs.get(&entrant), Some(&4.0));
    assert_eq!(sol.expected_payoffs.get(&incumbent), Some(&5.0));
}
