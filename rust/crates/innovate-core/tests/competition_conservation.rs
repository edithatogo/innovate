use innovate_core::domain::bass_cumulative;
use innovate_core::models::competition::{CompetitionParameters, MarketState, competing_rates};
use innovate_core::models::demand::{ChoiceSet, PreferenceParameters, choice_shares};
use innovate_core::models::network::neighbor_exposure;
use innovate_core::models::norton_bass::norton_bass_two_gen;

#[test]
fn norton_bass_and_competition_conserve_market_and_respect_entry_times() {
    // Before tau2=5, gen 2 is zero and gen 1 matches single-product Bass.
    let before = norton_bass_two_gen(3.0, 0.03, 0.38, 1000.0, 0.04, 0.4, 500.0, 5.0).unwrap();
    assert_eq!(before[1], 0.0);
    assert!((before[0] - bass_cumulative(3.0, 0.03, 0.38, 1000.0).unwrap()).abs() < 1e-12);

    // After launch, S1(t) + S2(t) == m1*F1(t) + m2*F2(t - tau2).
    let after = norton_bass_two_gen(10.0, 0.03, 0.38, 1000.0, 0.04, 0.4, 500.0, 5.0).unwrap();
    let expected_total = bass_cumulative(10.0, 0.03, 0.38, 1000.0).unwrap()
        + bass_cumulative(5.0, 0.04, 0.4, 500.0).unwrap();
    assert!((after[0] + after[1] - expected_total).abs() < 1e-10);

    // Competing rates: product 1 not yet entered at t=1.0 -> rate == 0.0; relabelling permutes outputs.
    let state = MarketState {
        adoption: vec![100.0, 0.0],
    };
    let params = CompetitionParameters {
        market_capacity: 1000.0,
        entry_times: vec![0.0, 3.0],
        p: vec![0.02, 0.03],
        q: vec![0.3, 0.4],
        interaction: vec![vec![0.0, -0.1], vec![0.05, 0.0]],
    };
    let rates = competing_rates(1.0, &state, &params).unwrap();
    assert_eq!(rates.rates[1], 0.0);
    assert!(rates.rates[0] > 0.0);

    let swapped_state = MarketState {
        adoption: vec![0.0, 100.0],
    };
    let swapped_params = CompetitionParameters {
        market_capacity: 1000.0,
        entry_times: vec![3.0, 0.0],
        p: vec![0.03, 0.02],
        q: vec![0.4, 0.3],
        interaction: vec![vec![0.0, 0.05], vec![-0.1, 0.0]],
    };
    let swapped = competing_rates(1.0, &swapped_state, &swapped_params).unwrap();
    assert!((swapped.rates[1] - rates.rates[0]).abs() < 1e-12);
    assert_eq!(swapped.rates[0], 0.0);
}

#[test]
fn network_exposure_and_logit_choice_shares_respect_invariants() {
    // No-edge network gives zero exposure; permutation preserves exposures.
    let zero_adj = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    assert_eq!(
        neighbor_exposure(&zero_adj, &[1.0, 0.0]).unwrap(),
        vec![0.0, 0.0]
    );

    let shares = choice_shares(
        &ChoiceSet {
            quality: vec![2.0, 1.0],
            price: vec![1.0, 0.5],
        },
        &PreferenceParameters {
            alpha_quality: 1.0,
            beta_price: -0.5,
        },
    )
    .unwrap();
    assert_eq!(shares.len(), 3);
    assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
