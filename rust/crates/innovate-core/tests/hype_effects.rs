use innovate_core::models::hype_effects::{
    HypeEffectParameters, HypeSignal, InnovationState, innovation_rates,
};

#[test]
fn hype_effects_separate_baseline_inhibition_and_active_use_cessation() {
    let state = InnovationState {
        cumulative_first_adoption: 100.0,
        active_use: 80.0,
    };
    let base = HypeEffectParameters {
        p0: 0.03,
        q0: 0.38,
        m: 1000.0,
        beta_p: 0.0,
        beta_q: 0.0,
        h0_stop: 0.1,
        beta_stop: 0.0,
    };
    let r0 = innovation_rates(1.0, &state, &HypeSignal { level: 2.0 }, &base).unwrap();
    let expected_first = (0.03 + 0.38 * 0.1) * 900.0;
    assert!((r0.first_adoption_rate - expected_first).abs() < 1e-12);
    assert!((r0.active_use_rate - (expected_first - 8.0)).abs() < 1e-12);

    // Negative beta_p and beta_q inhibit uptake while keeping first_adoption_rate >= 0.
    let inhibited = HypeEffectParameters {
        beta_p: -0.5,
        beta_q: -0.5,
        ..base
    };
    let r_inh = innovation_rates(1.0, &state, &HypeSignal { level: 2.0 }, &inhibited).unwrap();
    assert!(r_inh.first_adoption_rate >= 0.0);
    assert!(r_inh.first_adoption_rate < r0.first_adoption_rate);
}
