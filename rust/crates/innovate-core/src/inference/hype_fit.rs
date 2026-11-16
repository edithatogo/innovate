//! Joint constrained multistart fitting for composite hype curves (H02).

use crate::domain::TimeGrid;
use crate::inference::{FitFailure, FitOutcome, FitResult, PosteriorDraws};
use crate::models::hype::{HypeConfiguration, HypeParameters, HypePulse, hype_mean};

/// Request for [`fit_hype`].
#[derive(Debug, Clone, PartialEq)]
pub struct HypeFitRequest {
    pub time: TimeGrid,
    pub observed: Vec<f64>,
    pub initial_guesses: Vec<HypeParameters>,
    pub max_iterations: usize,
}

fn flatten_params(p: &HypeParameters) -> Vec<f64> {
    let mut v = vec![p.baseline, p.maturity_amplitude, p.growth, p.t_maturity];
    for pulse in &p.pulses {
        v.push(pulse.amplitude);
        v.push(pulse.center);
        v.push(pulse.width);
    }
    v
}

fn unflatten_params(
    theta: &[f64],
    n_pulses: usize,
    configuration: HypeConfiguration,
) -> HypeParameters {
    let mut pulses = Vec::with_capacity(n_pulses);
    for j in 0..n_pulses {
        let off = 4 + 3 * j;
        pulses.push(HypePulse {
            amplitude: theta[off],
            center: theta[off + 1],
            width: theta[off + 2],
        });
    }
    HypeParameters {
        baseline: theta[0],
        maturity_amplitude: theta[1],
        growth: theta[2],
        t_maturity: theta[3],
        pulses,
        configuration,
    }
}

fn sse(time: &[f64], observed: &[f64], p: &HypeParameters) -> Option<f64> {
    let mut total = 0.0_f64;
    for (&t, &y) in time.iter().zip(observed) {
        let pred = hype_mean(t, p).ok()?;
        let r = pred - y;
        total += r * r;
    }
    total.is_finite().then_some(total)
}

/// Jointly fit composite hype parameters from multiple starting points (H02).
///
/// Stores training-only time normalization metadata and flags overlapping/non-identifiable
/// pulse decompositions in `diagnostics` while preserving predictive total accuracy.
#[must_use]
pub fn fit_hype(request: &HypeFitRequest) -> FitOutcome {
    if request.observed.len() != request.time.len()
        || request.observed.iter().any(|v| !v.is_finite())
        || request.initial_guesses.is_empty()
        || request.max_iterations == 0
    {
        return FitOutcome::Invalid(FitFailure {
            code: "invalid_hype_fit_request".into(),
            message:
                "aligned finite observations, initial guesses, and max_iterations > 0 required"
                    .into(),
        });
    }

    let time = request.time.as_slice();
    let t_min = time[0];
    let t_span = (time[time.len() - 1] - t_min).max(1.0);

    let mut best_params: Option<HypeParameters> = None;
    let mut best_sse = f64::INFINITY;

    for guess in &request.initial_guesses {
        if guess.validate().is_err() {
            continue;
        }
        let n_pulses = guess.pulses.len();
        let config = guess.configuration;
        let mut theta = flatten_params(guess);
        let Some(mut cur_sse) = sse(time, &request.observed, guess) else {
            continue;
        };

        let mut step_sizes = vec![0.1_f64; theta.len()];
        for _ in 0..request.max_iterations {
            let mut improved = false;
            for idx in 0..theta.len() {
                let h = step_sizes[idx];
                if h < 1e-8 {
                    continue;
                }
                let orig = theta[idx];
                let mut best_cand = (orig, cur_sse);
                for dir in [-1.0_f64, 1.0_f64] {
                    theta[idx] = orig + dir * h;
                    let cand = unflatten_params(&theta, n_pulses, config);
                    if let Some(s) = sse(time, &request.observed, &cand)
                        && s < best_cand.1
                    {
                        best_cand = (theta[idx], s);
                    }
                }
                theta[idx] = best_cand.0;
                if best_cand.1 < cur_sse {
                    cur_sse = best_cand.1;
                    step_sizes[idx] *= 1.15;
                    improved = true;
                } else {
                    step_sizes[idx] *= 0.5;
                }
            }
            if !improved && step_sizes.iter().all(|&s| s < 1e-6) {
                break;
            }
        }

        let fitted = unflatten_params(&theta, n_pulses, config);
        if cur_sse < best_sse && fitted.validate().is_ok() {
            best_sse = cur_sse;
            best_params = Some(fitted);
        }
    }

    let Some(fitted) = best_params else {
        return FitOutcome::Invalid(FitFailure {
            code: "hype_fit_failed".into(),
            message: "no valid initial guess converged to finite composite SSE".into(),
        });
    };

    let mut overlapping = false;
    for i in 0..fitted.pulses.len() {
        for j in (i + 1)..fitted.pulses.len() {
            let dist = (fitted.pulses[i].center - fitted.pulses[j].center).abs();
            let min_w = fitted.pulses[i].width.min(fitted.pulses[j].width);
            if dist < min_w {
                overlapping = true;
            }
        }
    }

    let mut names = vec![
        "baseline".into(),
        "maturity_amplitude".into(),
        "growth".into(),
        "t_maturity".into(),
    ];
    for j in 0..fitted.pulses.len() {
        names.push(format!("pulse_{j}_amplitude"));
        names.push(format!("pulse_{j}_center"));
        names.push(format!("pulse_{j}_width"));
    }
    let vals = flatten_params(&fitted);
    let diag = format!(
        "converged; sse={best_sse:.6}; norm_origin={t_min:.6}; norm_span={t_span:.6}; overlapping_pulses={overlapping}"
    );
    match PosteriorDraws::new(names, vec![vals]) {
        Ok(posterior) => FitOutcome::Valid(FitResult {
            posterior,
            diagnostics: diag,
        }),
        Err(err) => FitOutcome::Invalid(FitFailure {
            code: err.code.into(),
            message: err.message,
        }),
    }
}
