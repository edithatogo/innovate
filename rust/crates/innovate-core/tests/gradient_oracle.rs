use innovate_core::inference::{ObservationSeries, log_density};

#[test]
fn gaussian_log_density_and_gradient_match_independent_oracle_and_finite_differences() {
    let ys = vec![1.2, 0.7, 1.5, 2.1, 0.9];
    let sigma = 0.65_f64;
    let series = ObservationSeries::new(ys.clone(), sigma).expect("valid series");

    for mu in [-1.5, 0.0, 1.28, 3.4] {
        let (ll, grad) = log_density(&[mu], &series).expect("valid log_density");
        assert_eq!(grad.len(), 1);

        // Independent closed-form Gaussian log-likelihood oracle.
        let n = ys.len() as f64;
        let expected_ll = -0.5 * n * (2.0 * std::f64::consts::PI * sigma * sigma).ln()
            - ys.iter().map(|y| (y - mu).powi(2)).sum::<f64>() / (2.0 * sigma * sigma);
        let expected_grad = ys.iter().map(|y| (y - mu) / (sigma * sigma)).sum::<f64>();

        assert!((ll - expected_ll).abs() < 1e-12);
        assert!((grad[0] - expected_grad).abs() < 1e-12);

        // Central finite-difference check.
        let eps = 1e-6;
        let (ll_plus, _) = log_density(&[mu + eps], &series).unwrap();
        let (ll_minus, _) = log_density(&[mu - eps], &series).unwrap();
        let fd = (ll_plus - ll_minus) / (2.0 * eps);
        assert!((grad[0] - fd).abs() < 1e-8);
    }
}

#[test]
fn log_density_rejects_invalid_variance_empty_data_and_non_finite_coordinates() {
    assert!(ObservationSeries::new(vec![], 1.0).is_err());
    assert!(ObservationSeries::new(vec![1.0, f64::NAN], 1.0).is_err());
    assert!(ObservationSeries::new(vec![1.0], 0.0).is_err());
    assert!(ObservationSeries::new(vec![1.0], -0.5).is_err());
    assert!(ObservationSeries::new(vec![1.0], f64::INFINITY).is_err());

    let series = ObservationSeries::new(vec![1.0, 2.0], 0.5).unwrap();
    assert!(log_density(&[], &series).is_err());
    assert!(log_density(&[1.0, 2.0], &series).is_err());
    assert!(log_density(&[f64::NAN], &series).is_err());
    assert!(log_density(&[f64::INFINITY], &series).is_err());
}
