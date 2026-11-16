use innovate_core::models::semi_markov::{
    BiExponentialKernel, CompetingRisksGenerator, TunnelStateModel, mat_mul,
};

#[test]
fn competing_risks_uniformization_conserves_rows_and_semigroup_composition() {
    let generator = CompetingRisksGenerator::new(vec![
        vec![0.0, 0.15, 0.05],
        vec![0.02, 0.0, 0.08],
        vec![0.0, 0.0, 0.0],
    ])
    .unwrap();

    // Every row of Q sums to 0.
    for row in generator.matrix() {
        assert!(row.iter().sum::<f64>().abs() < 1e-12);
    }

    let p1 = generator.uniformization_step(0.7).unwrap();
    let p2 = generator.uniformization_step(1.3).unwrap();
    let p_sum = generator.uniformization_step(2.0).unwrap();
    let composed = mat_mul(&p1, &p2);

    for i in 0..3 {
        assert!((p_sum[i].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for j in 0..3 {
            assert!(p_sum[i][j] >= 0.0);
            assert!((composed[i][j] - p_sum[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn bi_exponential_and_tunnel_states_reduce_to_markov_and_conserve_cohort() {
    let kernel = BiExponentialKernel {
        alpha: 0.6,
        lambda1: 0.5,
        lambda2: 0.1,
    };
    assert!((kernel.weight(0.0).unwrap() - 1.0).abs() < 1e-12);
    assert!(kernel.weight(2.0).unwrap() < 1.0);

    let markov_limit = TunnelStateModel {
        max_tunnel: 3,
        baseline_relapse: 0.05,
        frailty_amplitude: 0.0,
        decay_rate: 0.4,
    };
    for tau in 0..=3 {
        assert!((markov_limit.hazard_at_duration(tau).unwrap() - 0.05).abs() < 1e-12);
    }

    let initial = vec![100.0, 50.0, 25.0, 10.0];
    let (next, relapsed) = markov_limit.step_cohort(&initial, 15.0).unwrap();
    let total_before: f64 = initial.iter().sum::<f64>() + 15.0;
    let total_after: f64 = next.iter().sum::<f64>() + relapsed;
    assert!((total_before - total_after).abs() < 1e-10);
}

#[test]
fn large_rate_time_products_match_closed_form_and_preserve_cohort_mass() {
    // The original review reproducer has state 1 absorbing.
    let absorbing = CompetingRisksGenerator::new(vec![vec![0.0, 1.0], vec![0.0, 0.0]])
        .unwrap()
        .uniformization_step(1000.0)
        .unwrap();
    for (row, expected) in absorbing.iter().zip([[0.0, 1.0], [0.0, 1.0]]) {
        for (actual, expected) in row.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{absorbing:?}");
        }
    }

    // For a two-state chain 0 <-> 1, the exact transient law is known.
    // The final time makes max_exit_rate * dt overflow, although both are finite.
    let generator = CompetingRisksGenerator::new(vec![vec![0.0, 4.0], vec![1.0, 0.0]]).unwrap();
    for dt in [0.125_f64, 250.0, 1e25, 1e100, f64::MAX] {
        let decay = (-5.0 * dt).exp();
        let expected = [
            [0.2 + 0.8 * decay, 0.8 * (1.0 - decay)],
            [0.2 * (1.0 - decay), 0.8 + 0.2 * decay],
        ];
        let p = generator.uniformization_step(dt).unwrap();
        for (row, expected_row) in p.iter().zip(expected) {
            assert!(
                (row.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                "dt={dt}: {p:?}"
            );
            for (actual, expected) in row.iter().zip(expected_row) {
                assert!(actual.is_finite() && *actual >= 0.0);
                assert!((actual - expected).abs() < 1e-12, "dt={dt}: {p:?}");
            }
        }
    }

    // A much slower transition must remain visible when another row sets the
    // uniformization rate. Its exact survival probability at this time is e^-1.
    let mixed_rates = CompetingRisksGenerator::new(vec![
        vec![0.0, 1e-25, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ])
    .unwrap()
    .uniformization_step(1e25)
    .unwrap();
    assert!(
        (mixed_rates[0][0] - (-1.0_f64).exp()).abs() < 1e-12,
        "{mixed_rates:?}"
    );
    assert!((mixed_rates[0][1] - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
}

#[test]
fn generator_rejects_overflowing_exit_rate() {
    let result = CompetingRisksGenerator::new(vec![
        vec![0.0, f64::MAX, f64::MAX],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ]);
    assert_eq!(result.unwrap_err().code, "invalid_hazard");
}

#[test]
fn uniformization_preserves_zero_rates_and_rejects_invalid_inputs() {
    let generator = CompetingRisksGenerator::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap();
    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(generator.uniformization_step(0.0).unwrap(), identity);
    assert_eq!(generator.uniformization_step(f64::MAX).unwrap(), identity);
    for dt in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            generator.uniformization_step(dt).unwrap_err().code,
            "invalid_timestep"
        );
    }
    for hazard in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            CompetingRisksGenerator::new(vec![vec![0.0, hazard], vec![0.0, 0.0]])
                .unwrap_err()
                .code,
            "invalid_hazard"
        );
    }
}
