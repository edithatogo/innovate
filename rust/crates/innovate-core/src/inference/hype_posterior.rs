//! Ordered parameter transforms, Jacobians, and native Bayesian hype posterior sampling (H03).

use crate::domain::TimeGrid;
use crate::inference::{FitFailure, FitOutcome, FitResult, InnovationError, PosteriorDraws};
use crate::models::hype::{HypeConfiguration, HypeParameters, HypePulse, hype_mean};
use rand::RngExt;
use rand::SeedableRng;

/// Transform unconstrained `u` to ordered pulse centers/positive widths with exact log-Jacobian (H03).
///
/// Layout of `u` for `n_pulses`: `[baseline, maturity_amplitude, log_growth, t_maturity, (amp_j, center_u_j, log_width_j)...]`.
/// When `ordered_centers == true` and `n_pulses >= 1`:
/// - `center_0 = center_u_0`
/// - `center_j = center_{j-1} + exp(center_u_j)` for `j >= 1`, contributing `center_u_j` to `log_jacobian`.
///
/// # Errors
/// Returns [`InnovationError`] on dimension mismatch or non-finite values.
pub fn transform_unconstrained_hype(
    u: &[f64],
    n_pulses: usize,
    ordered_centers: bool,
) -> Result<(HypeParameters, f64), InnovationError> {
    if u.len() != 4 + 3 * n_pulses || u.iter().any(|v| !v.is_finite()) {
        return Err(InnovationError::new(
            "invalid_unconstrained_vector",
            "unconstrained hype vector length or finiteness mismatch",
        ));
    }
    let growth = u[2].exp();
    let mut log_jac = u[2];
    let mut pulses = Vec::with_capacity(n_pulses);
    let mut prev_center = 0.0_f64;
    for j in 0..n_pulses {
        let off = 4 + 3 * j;
        let amp = u[off];
        let center_u = u[off + 1];
        let log_w = u[off + 2];
        let width = log_w.exp();
        log_jac += log_w;
        let center = if ordered_centers && j > 0 {
            log_jac += center_u;
            prev_center + center_u.exp()
        } else {
            center_u
        };
        prev_center = center;
        pulses.push(HypePulse {
            amplitude: amp,
            center,
            width,
        });
    }
    let params = HypeParameters {
        baseline: u[0],
        maturity_amplitude: u[1],
        growth,
        t_maturity: u[3],
        pulses,
        configuration: if ordered_centers {
            HypeConfiguration::OrderedCycle
        } else {
            HypeConfiguration::UnrestrictedWave
        },
    };
    params.validate()?;
    if !log_jac.is_finite() {
        return Err(InnovationError::new(
            "numerical_overflow",
            "log-Jacobian overflowed finite domain",
        ));
    }
    Ok((params, log_jac))
}

/// Request for [`sample_hype_posterior`].
#[derive(Debug, Clone, PartialEq)]
pub struct HypePosteriorRequest {
    /// Validated time coordinates for the observed hype series.
    pub time: TimeGrid,
    /// Finite hype levels aligned with `time`.
    pub observed: Vec<f64>,
    /// Starting coordinates in the layout of [`transform_unconstrained_hype`].
    pub initial_unconstrained: Vec<f64>,
    /// Number of Gaussian pulses; the coordinate vector has `4 + 3 * n_pulses` entries.
    pub n_pulses: usize,
    /// Whether later pulse centers are positive increments from the previous center.
    pub ordered_centers: bool,
    /// Fixed, finite positive Gaussian observation standard deviation.
    pub sigma: f64,
    /// Positive retained draw count; an equal number of initial iterations is discarded.
    pub n_draws: usize,
    /// Seed for the Rust `StdRng` proposal and acceptance draws.
    pub seed: u64,
}

/// Sample a composite hype posterior with ordered parameter transforms and fail-closed status (H03).
///
/// Uses one random-walk Metropolis chain with joint proposals whose independent
/// increments per coordinate are uniform in `[-0.05, 0.05]`; the whole vector is
/// accepted or rejected together. The implemented target combines Gaussian residual terms,
/// `-0.005 * sum(u²)`, and the transform log-Jacobian. Returned draws use the
/// constrained parameter coordinates. A valid outcome requires finite draws and
/// at least one accepted proposal; it does not establish chain convergence or
/// effective sample size. Diagnostics report the proposal acceptance count only.
#[must_use]
pub fn sample_hype_posterior(request: &HypePosteriorRequest) -> FitOutcome {
    if request.observed.len() != request.time.len()
        || request.observed.iter().any(|v| !v.is_finite())
        || !request.sigma.is_finite()
        || request.sigma <= 0.0
        || request.n_draws == 0
    {
        return FitOutcome::Invalid(FitFailure {
            code: "invalid_posterior_request".into(),
            message: "aligned observations, sigma > 0, and n_draws > 0 required".into(),
        });
    }

    let log_target = |u: &[f64]| -> Option<(HypeParameters, f64)> {
        let (params, log_jac) =
            transform_unconstrained_hype(u, request.n_pulses, request.ordered_centers).ok()?;
        let inv_var = 1.0 / (request.sigma * request.sigma);
        let mut ll = 0.0_f64;
        for (&t, &y) in request.time.as_slice().iter().zip(&request.observed) {
            let mu = hype_mean(t, &params).ok()?;
            let r = y - mu;
            ll -= 0.5 * inv_var * r * r;
        }
        // Weakly informative normal prior on unconstrained coordinates for propriety.
        let lp: f64 = u.iter().map(|x| -0.005 * x * x).sum();
        let total = ll + lp + log_jac;
        total.is_finite().then_some((params, total))
    };

    let mut u = request.initial_unconstrained.clone();
    let Some((mut cur_params, mut cur_logp)) = log_target(&u) else {
        return FitOutcome::Invalid(FitFailure {
            code: "invalid_initial_state".into(),
            message: "initial unconstrained coordinates failed log-posterior evaluation".into(),
        });
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
    let total_iters = request.n_draws * 2;
    let mut draws = Vec::with_capacity(request.n_draws);
    let mut accepted = 0usize;
    let step = 0.05_f64;

    for iter in 0..total_iters {
        let prop: Vec<f64> = u
            .iter()
            .map(|&x| x + step * (rng.random::<f64>() * 2.0 - 1.0))
            .collect();
        if let Some((prop_params, prop_logp)) = log_target(&prop) {
            let log_alpha = prop_logp - cur_logp;
            if log_alpha >= 0.0 || rng.random::<f64>().ln() < log_alpha {
                u = prop;
                cur_params = prop_params;
                cur_logp = prop_logp;
                accepted += 1;
            }
        }
        if iter >= request.n_draws {
            let mut row = vec![
                cur_params.baseline,
                cur_params.maturity_amplitude,
                cur_params.growth,
                cur_params.t_maturity,
            ];
            for pulse in &cur_params.pulses {
                row.push(pulse.amplitude);
                row.push(pulse.center);
                row.push(pulse.width);
            }
            draws.push(row);
        }
    }

    if accepted == 0 {
        return FitOutcome::Invalid(FitFailure {
            code: "sampler_stuck".into(),
            message: "zero accepted Metropolis proposals; rejecting posterior promotion".into(),
        });
    }

    let mut names = vec![
        "baseline".into(),
        "maturity_amplitude".into(),
        "growth".into(),
        "t_maturity".into(),
    ];
    for j in 0..request.n_pulses {
        names.push(format!("pulse_{j}_amplitude"));
        names.push(format!("pulse_{j}_center"));
        names.push(format!("pulse_{j}_width"));
    }

    match PosteriorDraws::new(names, draws) {
        Ok(posterior) => FitOutcome::Valid(FitResult {
            posterior,
            diagnostics: format!("accepted={accepted}/{total_iters}"),
        }),
        Err(err) => FitOutcome::Invalid(FitFailure {
            code: err.code.into(),
            message: err.message,
        }),
    }
}
