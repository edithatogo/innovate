use innovate_core::inference::{
    DeterministicFitResult, FitFailure, FitOutcome, FitResult, InferenceSession, PosteriorDraws,
    posterior,
};

fn valid_outcome() -> FitOutcome {
    let draws = PosteriorDraws::new(
        vec!["mu".to_string()],
        vec![vec![1.0], vec![1.2], vec![0.8], vec![1.1]],
    )
    .expect("valid posterior draws");
    FitOutcome::Valid(FitResult {
        posterior: draws,
        diagnostics: "converged".into(),
    })
}

#[test]
fn invalid_outcome_never_exposes_posterior_draws_or_intervals() {
    let invalid = FitOutcome::Invalid(FitFailure {
        code: "sampler_failed".into(),
        message: "warmup diverged".into(),
    });
    let err = posterior(&invalid).unwrap_err();
    assert_eq!(err.code, "invalid_inference");
    assert!(err.message.contains("warmup diverged"));
}

#[test]
fn failed_refit_and_reset_clear_prior_posterior_draws() {
    let mut session = InferenceSession::new();
    assert_eq!(session.posterior().unwrap_err().code, "invalid_inference");

    session.record(valid_outcome());
    let draws = session.posterior().expect("valid draws after fit");
    assert_eq!(draws.draws().len(), 4);
    let (lo, hi) = draws.credible_interval(0, 0.90).expect("valid interval");
    assert!(lo <= hi);

    // Failed refit must overwrite and clear the prior posterior.
    session.record(FitOutcome::Invalid(FitFailure {
        code: "refit_failed".into(),
        message: "non-finite logp".into(),
    }));
    let err = session.posterior().unwrap_err();
    assert_eq!(err.code, "invalid_inference");
    assert!(err.message.contains("refit_failed"));

    // Explicit reset must also clear any valid posterior.
    session.record(valid_outcome());
    assert!(session.posterior().is_ok());
    session.reset();
    assert_eq!(session.posterior().unwrap_err().code, "invalid_inference");
}

#[test]
fn deterministic_refit_clears_previous_posterior() {
    let mut session = InferenceSession::new();
    session.record(valid_outcome());
    session.record(FitOutcome::Deterministic(DeterministicFitResult {
        names: vec!["mu".into()],
        values: vec![1.0],
        diagnostics: "point estimate".into(),
    }));
    assert_eq!(session.posterior().unwrap_err().code, "deterministic_fit");
    assert!(matches!(session.outcome(), FitOutcome::Deterministic(_)));
}

#[test]
fn posterior_draws_reject_empty_mismatched_and_non_finite_inputs() {
    assert!(PosteriorDraws::new(vec![], vec![vec![1.0]]).is_err());
    assert!(PosteriorDraws::new(vec!["mu".into()], vec![]).is_err());
    assert!(PosteriorDraws::new(vec!["mu".into()], vec![vec![1.0, 2.0]]).is_err());
    assert!(PosteriorDraws::new(vec!["mu".into()], vec![vec![f64::NAN]]).is_err());
}
