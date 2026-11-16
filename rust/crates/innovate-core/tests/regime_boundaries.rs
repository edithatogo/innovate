use innovate_core::domain::TimeGrid;
use innovate_core::models::regimes::{DatedShock, RegimeRequest, predict_regimes};
use innovate_core::models::{ClassicalFamily, PredictionRequest, predict_classical};

#[test]
fn regimes_no_shock_matches_baseline_and_never_uses_future_shocks() {
    let time = TimeGrid::try_from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let base_params = vec![0.03, 0.38, 1000.0];
    let baseline = predict_classical(&PredictionRequest {
        family: ClassicalFamily::Bass,
        time: time.clone(),
        parameters: base_params.clone(),
    })
    .unwrap();

    let no_shock = predict_regimes(&RegimeRequest {
        family: ClassicalFamily::Bass,
        time: time.clone(),
        base_parameters: base_params.clone(),
        shocks: vec![],
    })
    .unwrap();
    assert_eq!(no_shock.values, baseline.values);

    // Shock at t=3.0 must not alter t < 3.0; decreasing capacity must not produce negative adoption.
    let shocked = predict_regimes(&RegimeRequest {
        family: ClassicalFamily::Bass,
        time,
        base_parameters: base_params,
        shocks: vec![DatedShock {
            effective_time: 3.0,
            p_multiplier: 1.5,
            q_multiplier: 1.0,
            capacity_multiplier: 0.5,
        }],
    })
    .unwrap();
    assert_eq!(shocked.values[0..3], baseline.values[0..3]);
    assert!(shocked.values.windows(2).all(|w| w[1] >= w[0]));
}
