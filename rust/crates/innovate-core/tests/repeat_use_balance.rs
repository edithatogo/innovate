use innovate_core::models::repeat_use::{
    ActiveUseState, SupplyIntensity, UseRates, expected_transactions, step_active_use,
};

#[test]
fn repeat_use_conserves_stock_flows_and_separates_transactions() {
    let state = ActiveUseState {
        active: vec![100.0, 50.0],
        cumulative_starts: vec![100.0, 50.0],
        cumulative_stops: vec![0.0, 0.0],
    };
    let rates = UseRates {
        starts: vec![10.0, 5.0],
        stops: vec![4.0, 2.0],
        switch: vec![vec![0.0, 3.0], vec![1.0, 0.0]],
    };
    let next = step_active_use(&state, &rates, 2.0).unwrap();
    // Product 0: 100 + (10 + 1 - 4 - 3)*2 = 108
    // Product 1: 50 + (5 + 3 - 2 - 1)*2 = 60
    assert!((next.active[0] - 108.0).abs() < 1e-12);
    assert!((next.active[1] - 60.0).abs() < 1e-12);

    // Total stock balances total starts minus total stops across switching.
    let delta_total: f64 = next.active.iter().sum::<f64>() - state.active.iter().sum::<f64>();
    let net_starts: f64 = next.cumulative_starts.iter().sum::<f64>()
        - state.cumulative_starts.iter().sum::<f64>()
        - (next.cumulative_stops.iter().sum::<f64>() - state.cumulative_stops.iter().sum::<f64>());
    assert!((delta_total - net_starts).abs() < 1e-12);

    // Expected transactions are scaled by supply intensity, distinct from actor counts.
    let tx = expected_transactions(
        &next,
        &SupplyIntensity {
            prescriptions_per_active: vec![2.5, 0.0],
        },
    )
    .unwrap();
    assert!((tx[0] - 270.0).abs() < 1e-12);
    assert_eq!(tx[1], 0.0);
}
