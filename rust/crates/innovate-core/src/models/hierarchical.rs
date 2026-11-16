//! Hierarchical partial-pooling model with non-centred group effects (A03).

use crate::inference::{FitFailure, FitOutcome, FitResult, InnovationError, PosteriorDraws};

/// Group-level observation panel for [`fit_hierarchical`].
#[derive(Debug, Clone, PartialEq)]
pub struct PanelFitRequest {
    pub group_ids: Vec<String>,
    pub values: Vec<f64>,
    pub prior_mean: f64,
    pub prior_sd: f64,
    pub obs_sd: f64,
    pub tau: f64,
}

/// Fit a conjugate hierarchical normal partial-pooling model with non-centred effects (A03).
///
/// For group $g$ with $n_g$ observations and sample mean $\bar{y}_g$:
/// - Precision $w_g = n_g / \sigma^2$, group prior precision $1 / \tau^2$
/// - Shrinkage weight $\lambda_g = \frac{w_g}{w_g + 1/\tau^2}$
/// - Hypermean $\hat{\mu} = \frac{\mu_0/\sigma_0^2 + \sum_g \frac{w_g \bar{y}_g}{1 + w_g \tau^2}}{1/\sigma_0^2 + \sum_g \frac{w_g}{1 + w_g \tau^2}}$
/// - Group estimate $\hat{\theta}_g = \lambda_g \bar{y}_g + (1 - \lambda_g)\hat{\mu}$.
#[must_use]
pub fn fit_hierarchical(request: &PanelFitRequest) -> FitOutcome {
    if request.group_ids.is_empty()
        || request.group_ids.len() != request.values.len()
        || !request.prior_mean.is_finite()
        || !request.prior_sd.is_finite()
        || request.prior_sd <= 0.0
        || !request.obs_sd.is_finite()
        || request.obs_sd <= 0.0
        || !request.tau.is_finite()
        || request.tau < 0.0
        || request.values.iter().any(|v| !v.is_finite())
    {
        return FitOutcome::Invalid(FitFailure {
            code: "invalid_hierarchical_request".into(),
            message: "groups, values, and positive finite scales are required".into(),
        });
    }

    let mut groups: std::collections::BTreeMap<String, (usize, f64)> =
        std::collections::BTreeMap::new();
    for (gid, &val) in request.group_ids.iter().zip(&request.values) {
        if gid.trim().is_empty() {
            return FitOutcome::Invalid(FitFailure {
                code: "invalid_group_id".into(),
                message: "group_id must be nonempty".into(),
            });
        }
        let entry = groups.entry(gid.clone()).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += val;
    }

    let obs_var = request.obs_sd * request.obs_sd;
    let tau_var = request.tau * request.tau;
    let prior_prec = 1.0 / (request.prior_sd * request.prior_sd);

    let mut num = prior_prec * request.prior_mean;
    let mut den = prior_prec;
    for &(n_g, sum_g) in groups.values() {
        let w_g = (n_g as f64) / obs_var;
        let eff_prec = w_g / (1.0 + w_g * tau_var);
        let mean_g = sum_g / (n_g as f64);
        num += eff_prec * mean_g;
        den += eff_prec;
    }
    let mu_hat = num / den;
    let mu_sd = (1.0 / den).sqrt();

    let mut names = vec!["hypermean".to_string(), "hypermean_sd".to_string()];
    let mut vals = vec![mu_hat, mu_sd];
    for (gid, &(n_g, sum_g)) in &groups {
        let mean_g = sum_g / (n_g as f64);
        let theta_g = if tau_var == 0.0 {
            mu_hat
        } else {
            let w_g = (n_g as f64) / obs_var;
            let group_prec = 1.0 / tau_var;
            (w_g * mean_g + group_prec * mu_hat) / (w_g + group_prec)
        };
        names.push(format!("group_{gid}"));
        vals.push(theta_g);
    }
    // Unseen group predictive mean equals hypermean with predictive sd sqrt(mu_sd^2 + tau^2).
    names.push("unseen_group_sd".to_string());
    vals.push(mu_sd.hypot(request.tau));

    if vals.iter().any(|v| !v.is_finite()) {
        return FitOutcome::Invalid(FitFailure {
            code: "non_finite_estimate".into(),
            message: "fitted parameters contain non-finite values".into(),
        });
    }
    FitOutcome::Deterministic(crate::inference::DeterministicFitResult {
        names,
        values: vals,
        diagnostics: format!("converged; groups={}", groups.len()),
    })
}

/// Draw a deterministic predictive mean/sd tuple for an unseen group (A03).
///
/// # Errors
/// Returns [`InnovationError`] if `outcome` is invalid.
pub fn unseen_group_predictive(outcome: &FitOutcome) -> Result<(f64, f64), InnovationError> {
    let row = match outcome {
        FitOutcome::Valid(res) => &res.posterior.draws()[0],
        FitOutcome::Deterministic(res) => &res.values,
        FitOutcome::Invalid(failure) => {
            return Err(InnovationError::new(
                "invalid_inference",
                format!("{}: {}", failure.code, failure.message),
            ));
        }
    };
    let mu = row[0];
    let unseen_sd = *row.last().unwrap();
    Ok((mu, unseen_sd))
}
