use innovate_core::models::expectation_feedback::{DelayedStateModel, ExternalLagSignal};

#[test]
fn external_lag_and_delayed_state_feedback_match_baselines() {
    let zero_lag = ExternalLagSignal {
        lag: 0.0,
        baseline_value: 0.0,
    };
    let f = |t: f64| 2.0 * t + 1.0;
    assert_eq!(zero_lag.evaluate(3.0, f).unwrap(), 7.0);

    let lagged = ExternalLagSignal {
        lag: 2.0,
        baseline_value: 0.0,
    };
    assert_eq!(lagged.evaluate(1.0, f).unwrap(), 0.0);
    assert_eq!(lagged.evaluate(5.0, f).unwrap(), f(3.0));

    let state_model = DelayedStateModel {
        delay_steps: 2,
        feedback_gain: 0.5,
        decay: 0.2,
    };
    let history = [1.0, 2.0, 4.0];
    let next = state_model.step_hype(1.0, &history, 0.3, 1.0).unwrap();
    // lagged_adoption = history[3 - 1 - 2] = history[0] = 1.0
    // next = 1.0 + (-0.2*1.0 + 0.5*1.0 + 0.3)*1.0 = 1.6
    assert!((next - 1.6).abs() < 1e-12);

    assert!(state_model.step_hype(1.0, &[1.0, 2.0], 0.0, 1.0).is_err());
}
