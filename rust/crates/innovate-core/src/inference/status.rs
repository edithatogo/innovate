//! Fail-closed inference status and analytic Gaussian log-density/gradient contracts.

pub use innovate_demand::InnovationError;

/// Validated observation series with strictly positive finite noise scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationSeries {
    values: Vec<f64>,
    sigma: f64,
}

impl ObservationSeries {
    /// Validate and construct an [`ObservationSeries`].
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `values` is empty, contains non-finite entries,
    /// or `sigma` is non-finite or non-positive.
    pub fn new(values: Vec<f64>, sigma: f64) -> Result<Self, InnovationError> {
        if values.is_empty() {
            return Err(InnovationError::new(
                "invalid_data",
                "observation series must be nonempty",
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(InnovationError::new(
                "invalid_data",
                "observation series must contain only finite values",
            ));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(InnovationError::new(
                "invalid_variance",
                "noise scale sigma must be finite and strictly positive",
            ));
        }
        Ok(Self { values, sigma })
    }

    /// Borrow the validated observation values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Return the validated noise standard deviation.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

/// Validated posterior draws with non-empty finite chains.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorDraws {
    parameter_names: Vec<String>,
    draws: Vec<Vec<f64>>,
}

impl PosteriorDraws {
    /// Construct validated [`PosteriorDraws`].
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `parameter_names` or `draws` are empty,
    /// dimensions mismatch, or any draw is non-finite.
    pub fn new(
        parameter_names: Vec<String>,
        draws: Vec<Vec<f64>>,
    ) -> Result<Self, InnovationError> {
        if parameter_names.is_empty() {
            return Err(InnovationError::new(
                "invalid_posterior",
                "posterior must name at least one parameter",
            ));
        }
        if draws.is_empty() {
            return Err(InnovationError::new(
                "invalid_posterior",
                "posterior must contain at least one draw",
            ));
        }
        for draw in &draws {
            if draw.len() != parameter_names.len() || draw.iter().any(|v| !v.is_finite()) {
                return Err(InnovationError::new(
                    "invalid_posterior",
                    "posterior draws must be finite and match parameter_names length",
                ));
            }
        }
        Ok(Self {
            parameter_names,
            draws,
        })
    }

    /// Parameter names corresponding to each coordinate of a draw.
    #[must_use]
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Matrix of posterior draws (`[n_draws][n_params]`).
    #[must_use]
    pub fn draws(&self) -> &[Vec<f64>] {
        &self.draws
    }

    /// Compute an equal-tailed empirical credible interval for parameter `index`.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `index` is out of bounds or `mass` is not in `(0, 1)`.
    pub fn credible_interval(
        &self,
        index: usize,
        mass: f64,
    ) -> Result<(f64, f64), InnovationError> {
        if index >= self.parameter_names.len() {
            return Err(InnovationError::new(
                "invalid_index",
                "parameter index out of bounds",
            ));
        }
        if !mass.is_finite() || mass <= 0.0 || mass >= 1.0 {
            return Err(InnovationError::new(
                "invalid_mass",
                "credible_mass must be strictly between 0 and 1",
            ));
        }
        let mut column: Vec<f64> = self.draws.iter().map(|row| row[index]).collect();
        column.sort_by(f64::total_cmp);
        let tail = (1.0 - mass) / 2.0;
        let lo_idx = ((column.len() - 1) as f64 * tail).floor() as usize;
        let hi_idx = ((column.len() - 1) as f64 * (1.0 - tail)).ceil() as usize;
        Ok((column[lo_idx], column[hi_idx]))
    }
}

/// Successful inference payload.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Validated posterior draws.
    pub posterior: PosteriorDraws,
    /// Human-readable convergence/diagnostic note.
    pub diagnostics: String,
}

/// Failed inference payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitFailure {
    /// Machine-readable failure reason.
    pub code: String,
    /// Human-readable diagnostic message.
    pub message: String,
}

/// Successful deterministic point-estimate fit payload (not a posterior).
#[derive(Debug, Clone, PartialEq)]
pub struct DeterministicFitResult {
    /// Parameter names in order.
    pub names: Vec<String>,
    /// Fitted point-estimate values in the same order as `names`.
    pub values: Vec<f64>,
    /// Human-readable convergence/diagnostic note.
    pub diagnostics: String,
}

/// Explicit inference status preventing invalid or stale posteriors from being exposed.
#[derive(Debug, Clone, PartialEq)]
pub enum FitOutcome {
    /// Inference succeeded and produced validated posterior draws.
    Valid(FitResult),
    /// Deterministic optimization succeeded and produced a point estimate (no posterior draws).
    Deterministic(DeterministicFitResult),
    /// Inference failed or was reset; no posterior draws are available.
    Invalid(FitFailure),
}

/// Stateful inference session that clears prior draws on reset or failed refit.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSession {
    outcome: FitOutcome,
}

impl Default for InferenceSession {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceSession {
    /// Create a new uninitialized [`InferenceSession`] in the [`FitOutcome::Invalid`] state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            outcome: FitOutcome::Invalid(FitFailure {
                code: "uninitialized".into(),
                message: "no inference has been completed".into(),
            }),
        }
    }

    /// Record a new [`FitOutcome`], replacing any prior result (including on failure).
    pub fn record(&mut self, outcome: FitOutcome) {
        self.outcome = outcome;
    }

    /// Reset the session to [`FitOutcome::Invalid`], clearing any prior posterior draws.
    pub fn reset(&mut self) {
        self.outcome = FitOutcome::Invalid(FitFailure {
            code: "reset".into(),
            message: "inference session was reset".into(),
        });
    }

    /// Borrow the current [`FitOutcome`].
    #[must_use]
    pub fn outcome(&self) -> &FitOutcome {
        &self.outcome
    }

    /// Fail-closed posterior accessor; rejects invalid and deterministic states.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if the current outcome is invalid or deterministic.
    pub fn posterior(&self) -> Result<&PosteriorDraws, InnovationError> {
        posterior(&self.outcome)
    }
}

/// Fail-closed accessor that exposes posterior draws only for [`FitOutcome::Valid`].
///
/// # Errors
/// Returns [`InnovationError`] when `outcome` is invalid or deterministic.
pub fn posterior(outcome: &FitOutcome) -> Result<&PosteriorDraws, InnovationError> {
    match outcome {
        FitOutcome::Valid(result) => Ok(&result.posterior),
        FitOutcome::Invalid(failure) => Err(InnovationError::new(
            "invalid_inference",
            format!("{}: {}", failure.code, failure.message),
        )),
        FitOutcome::Deterministic(_) => Err(InnovationError::new(
            "deterministic_fit",
            "deterministic point-estimate fit has no posterior draws",
        )),
    }
}

/// Evaluate exact Gaussian log-density and analytic gradient for `theta = [mu]`.
///
/// Computes:
/// $$\ell(\mu) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mu)^2$$
/// and $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (y_i - \mu)$.
///
/// # Errors
/// Returns [`InnovationError`] if `theta.len() != 1`, `theta[0]` is non-finite,
/// `data` has invalid variance/values, or the resulting log-density/gradient overflows.
pub fn log_density(
    theta: &[f64],
    data: &ObservationSeries,
) -> Result<(f64, Vec<f64>), InnovationError> {
    if theta.len() != 1 {
        return Err(InnovationError::new(
            "invalid_dimension",
            "Gaussian location model requires exactly 1 parameter (mu)",
        ));
    }
    let mu = theta[0];
    if !mu.is_finite() {
        return Err(InnovationError::new(
            "invalid_parameter",
            "theta[0] must be finite",
        ));
    }
    let sigma = data.sigma();
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(InnovationError::new(
            "invalid_variance",
            "noise scale sigma must be finite and strictly positive",
        ));
    }
    let values = data.values();
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(InnovationError::new(
            "invalid_data",
            "observation series must be nonempty and finite",
        ));
    }

    let n = values.len() as f64;
    let inv_var = 1.0 / (sigma * sigma);
    let mut sum_sq = 0.0_f64;
    let mut sum_diff = 0.0_f64;
    for &y in values {
        let diff = y - mu;
        sum_diff += diff;
        sum_sq += diff * diff;
    }
    let norm_const = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + 2.0 * sigma.ln());
    let ll = norm_const - 0.5 * inv_var * sum_sq;
    let grad = inv_var * sum_diff;
    if !ll.is_finite() || !grad.is_finite() {
        return Err(InnovationError::new(
            "numerical_overflow",
            "log-density or gradient overflowed finite domain",
        ));
    }
    Ok((ll, vec![grad]))
}
