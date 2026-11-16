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
