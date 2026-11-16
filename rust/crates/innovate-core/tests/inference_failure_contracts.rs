use innovate_core::domain::TimeGrid;
use innovate_core::inference::FitOutcome;
use innovate_core::inference::hype_fit::{HypeFitRequest, fit_hype};
use innovate_core::inference::hype_posterior::{
    HypePosteriorRequest, sample_hype_posterior, transform_unconstrained_hype,
};
use innovate_core::models::hierarchical::{
    PanelFitRequest, fit_hierarchical, unseen_group_predictive,
};
use innovate_core::models::hype::{HypeConfiguration, HypeParameters};
use innovate_core::models::semi_markov::{
    BiExponentialKernel, CompetingRisksGenerator, TunnelStateModel,
};

fn failure_code(outcome: &FitOutcome) -> &str {
    match outcome {
        FitOutcome::Invalid(failure) => &failure.code,
        other => panic!("expected an invalid fit, got {other:?}"),
    }
}

#[test]
fn hype_inference_rejects_bad_coordinates_and_failed_starts() {
    assert_eq!(
        transform_unconstrained_hype(&[0.0], 0, false)
            .unwrap_err()
            .code,
        "invalid_unconstrained_vector"
    );
    assert_eq!(
        transform_unconstrained_hype(&[0.0, 0.0, f64::NAN, 0.0], 0, false)
            .unwrap_err()
            .code,
        "invalid_unconstrained_vector"
    );
    let time = TimeGrid::try_from(vec![0.0, 1.0]).unwrap();
    let request = HypePosteriorRequest {
        time: time.clone(),
        observed: vec![0.0, 0.0],
        initial_unconstrained: vec![0.0],
        n_pulses: 0,
        ordered_centers: false,
        sigma: 1.0,
        n_draws: 2,
        seed: 7,
    };
    assert_eq!(
        failure_code(&sample_hype_posterior(&request)),
        "invalid_initial_state"
    );
    let invalid_guess = HypeParameters {
        baseline: 0.0,
        maturity_amplitude: 1.0,
        growth: 0.0,
        t_maturity: 0.0,
        pulses: vec![],
        configuration: HypeConfiguration::UnrestrictedWave,
    };
    let fit = fit_hype(&HypeFitRequest {
        time,
        observed: vec![0.0, 0.0],
        initial_guesses: vec![invalid_guess],
        max_iterations: 1,
    });
    assert_eq!(failure_code(&fit), "hype_fit_failed");
}

#[test]
fn hierarchical_fit_fails_closed_for_invalid_panels_and_predictive_access() {
    let mut request = PanelFitRequest {
        group_ids: vec!["a".into()],
        values: vec![1.0],
        prior_mean: 0.0,
        prior_sd: 1.0,
        obs_sd: 1.0,
        tau: 1.0,
    };
    request.obs_sd = 0.0;
    let bad_scale = fit_hierarchical(&request);
    assert_eq!(failure_code(&bad_scale), "invalid_hierarchical_request");
    assert_eq!(
        unseen_group_predictive(&bad_scale).unwrap_err().code,
        "invalid_inference"
    );
    request.obs_sd = 1.0;
    request.group_ids[0] = " ".into();
    assert_eq!(
        failure_code(&fit_hierarchical(&request)),
        "invalid_group_id"
    );
}

#[test]
fn tunnel_and_competing_risk_inputs_are_checked() {
    assert_eq!(
        CompetingRisksGenerator::new(vec![]).unwrap_err().code,
        "invalid_generator"
    );
    let kernel = BiExponentialKernel {
        alpha: 1.1,
        lambda1: 0.5,
        lambda2: 0.1,
    };
    assert_eq!(kernel.weight(1.0).unwrap_err().code, "invalid_kernel");
    let model = TunnelStateModel {
        max_tunnel: 1,
        baseline_relapse: 2.0,
        frailty_amplitude: 0.0,
        decay_rate: 0.0,
    };
    assert_eq!(
        model.hazard_at_duration(2).unwrap_err().code,
        "invalid_tunnel_duration"
    );
    assert_eq!(
        model.step_cohort(&[1.0], 0.0).unwrap_err().code,
        "invalid_tunnel_occupancy"
    );
    let (next, relapsed) = model.step_cohort(&[10.0, 0.0], 0.0).unwrap();
    let expected_survival = 10.0 * (-2.0_f64).exp();
    assert!((next[1] - expected_survival).abs() < 1e-12);
    assert!((relapsed - (10.0 - expected_survival)).abs() < 1e-12);
    let invalid = TunnelStateModel {
        decay_rate: -1.0,
        ..model
    };
    assert_eq!(
        invalid.hazard_at_duration(0).unwrap_err().code,
        "invalid_tunnel_model"
    );
}
