use innovate_core::models::hype::{
    HypeConfiguration, HypeParameters, HypePulse, hype_derivative, hype_mean,
};

#[test]
fn hype_no_pulse_limit_and_derivative_match_finite_difference() {
    let no_pulse = HypeParameters {
        baseline: 1.5,
        maturity_amplitude: 4.0,
        growth: 0.8,
        t_maturity: 5.0,
        pulses: vec![],
        configuration: HypeConfiguration::UnrestrictedWave,
    };
    for t in [0.0_f64, 2.5, 5.0, 10.0] {
        let expected = 1.5 + 4.0 / (1.0 + (-0.8 * (t - 5.0)).exp());
        assert!((hype_mean(t, &no_pulse).unwrap() - expected).abs() < 1e-12);
    }

    // Composite curve with positive and negative pulses; no hard clipping.
    let wave = HypeParameters {
        baseline: -1.0,
        maturity_amplitude: 3.0,
        growth: 0.5,
        t_maturity: 6.0,
        pulses: vec![
            HypePulse {
                amplitude: 5.0,
                center: 2.0,
                width: 1.0,
            },
            HypePulse {
                amplitude: -3.0,
                center: 4.5,
                width: 0.8,
            },
        ],
        configuration: HypeConfiguration::OrderedCycle,
    };
    assert!(hype_mean(4.5, &wave).unwrap() < 0.0);
    let eps = 1e-6;
    for t in [1.0, 2.2, 4.5, 6.0] {
        let deriv = hype_derivative(t, &wave).unwrap();
        let fd =
            (hype_mean(t + eps, &wave).unwrap() - hype_mean(t - eps, &wave).unwrap()) / (2.0 * eps);
        assert!((deriv - fd).abs() < 1e-7);
    }
}

#[test]
fn hype_rejects_invalid_widths_nan_and_unordered_cycles() {
    let bad_width = HypeParameters {
        baseline: 0.0,
        maturity_amplitude: 1.0,
        growth: 1.0,
        t_maturity: 1.0,
        pulses: vec![HypePulse {
            amplitude: 1.0,
            center: 1.0,
            width: 0.0,
        }],
        configuration: HypeConfiguration::UnrestrictedWave,
    };
    assert!(hype_mean(1.0, &bad_width).is_err());

    let unordered = HypeParameters {
        baseline: 0.0,
        maturity_amplitude: 1.0,
        growth: 1.0,
        t_maturity: 1.0,
        pulses: vec![
            HypePulse {
                amplitude: 1.0,
                center: 4.0,
                width: 1.0,
            },
            HypePulse {
                amplitude: 1.0,
                center: 2.0,
                width: 1.0,
            },
        ],
        configuration: HypeConfiguration::OrderedCycle,
    };
    assert!(hype_mean(1.0, &unordered).is_err());
    assert!(hype_mean(f64::NAN, &bad_width).is_err());
}
