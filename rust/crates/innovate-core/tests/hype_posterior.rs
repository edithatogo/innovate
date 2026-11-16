use innovate_core::domain::TimeGrid;
use innovate_core::inference::hype_posterior::{
    HypePosteriorRequest, sample_hype_posterior, transform_unconstrained_hype,
};
use innovate_core::inference::posterior;
use innovate_core::models::hype::{HypeConfiguration, HypeParameters, HypePulse, hype_mean};

#[test]
fn hype_ordered_transform_jacobian_and_posterior_sampler_work() {
    // Check ordered pulse centers and positive log-Jacobian.
    let u = vec![
        0.5,
        2.0,
        0.0_f64.exp().ln(),
        6.0,
        3.0,
        2.0,
        0.0,
        1.5,
        0.5,
        0.2,
    ];
    let (params, log_jac) = transform_unconstrained_hype(&u, 2, true).unwrap();
    assert!(params.pulses[1].center > params.pulses[0].center);
    assert!((log_jac - (u[2] + u[6] + u[8] + u[9])).abs() < 1e-12);

    // Sample posterior on synthetic single-pulse series.
    let true_p = HypeParameters {
        baseline: 0.5,
        maturity_amplitude: 2.0,
        growth: 1.0,
        t_maturity: 6.0,
        pulses: vec![HypePulse {
            amplitude: 3.0,
            center: 2.5,
            width: 1.0,
        }],
        configuration: HypeConfiguration::OrderedCycle,
    };
    let time = TimeGrid::try_from((0..12).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
    let observed: Vec<f64> = time
        .as_slice()
        .iter()
        .map(|&t| hype_mean(t, &true_p).unwrap())
        .collect();

    let outcome = sample_hype_posterior(&HypePosteriorRequest {
        time: time.clone(),
        observed: observed.clone(),
        initial_unconstrained: vec![0.5, 2.0, 0.0, 6.0, 3.0, 2.5, 0.0],
        n_pulses: 1,
        ordered_centers: true,
        sigma: 0.25,
        n_draws: 80,
        seed: 42,
    });
    let draws = posterior(&outcome).unwrap();
    assert_eq!(draws.draws().len(), 80);

    // Invalid sigma must fail closed.
    let invalid = sample_hype_posterior(&HypePosteriorRequest {
        time,
        observed,
        initial_unconstrained: vec![0.5, 2.0, 0.0, 6.0, 3.0, 2.5, 0.0],
        n_pulses: 1,
        ordered_centers: true,
        sigma: 0.0,
        n_draws: 20,
        seed: 42,
    });
    assert!(posterior(&invalid).is_err());
}
