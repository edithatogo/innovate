use innovate_core::bass_cumulative;

#[test]
fn bass_cumulative_matches_boundary_exponential_and_high_precision_reference() {
    assert_eq!(bass_cumulative(0.0, 0.03, 0.38, 1000.0).unwrap(), 0.0);

    // When q == 0, Bass reduces to exponential m * (1 - exp(-p * t)).
    for t in [0.0, 1.0, 5.0, 25.0] {
        let actual = bass_cumulative(t, 0.04, 0.0, 500.0).unwrap();
        let expected = 500.0 * (1.0 - (-0.04 * t).exp());
        assert!((actual - expected).abs() < 1e-12);
    }

    // High-precision reference at p=0.03, q=0.38, m=1000, t=5:
    // (1 - exp(-2.05)) / (1 + (0.38/0.03) * exp(-2.05)) * 1000 = 330.7209427029166
    let val = bass_cumulative(5.0, 0.03, 0.38, 1000.0).unwrap();
    let ref_val = {
        let e = (-0.41_f64 * 5.0).exp();
        1000.0 * (1.0 - e) / (1.0 + (0.38 / 0.03) * e)
    };
    assert!((val - ref_val).abs() < 1e-12);
    assert!((0.0..=1000.0).contains(&val));
}

#[test]
fn bass_cumulative_rejects_invalid_inputs() {
    assert!(bass_cumulative(-1.0, 0.03, 0.38, 1000.0).is_err());
    assert!(bass_cumulative(f64::NAN, 0.03, 0.38, 1000.0).is_err());
    assert!(bass_cumulative(1.0, 0.0, 0.38, 1000.0).is_err());
    assert!(bass_cumulative(1.0, 0.03, -0.1, 1000.0).is_err());
    assert!(bass_cumulative(1.0, 0.03, 0.38, 0.0).is_err());
}
