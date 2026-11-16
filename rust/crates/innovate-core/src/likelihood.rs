//! Count and continuous observation likelihoods with explicit parametrisation (A02).

use crate::domain::DomainObservationSeries;
use crate::inference::InnovationError;
use statrs::function::gamma::ln_gamma;

/// Declared observation likelihood family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LikelihoodFamily {
    /// Poisson count likelihood parameterized by its mean.
    Poisson,
    /// Negative-binomial count likelihood parameterized by mean and dispersion.
    NegativeBinomial {
        /// Positive dispersion (size) parameter.
        dispersion: f64,
    },
    /// Gaussian likelihood parameterized by mean and standard deviation.
    Gaussian {
        /// Positive observation standard deviation.
        sigma: f64,
    },
    /// Student-t likelihood parameterized by location, scale and degrees of freedom.
    StudentT {
        /// Positive observation scale.
        sigma: f64,
        /// Positive degrees of freedom.
        df: f64,
    },
}

/// Observation scale guard preventing cumulative counts from being treated as independent increments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationScale {
    /// Counts for separate observation intervals.
    IncidenceCounts,
    /// Cumulative counts, which cannot be treated as independent count increments.
    CumulativeCounts,
    /// Observations measured on a continuous scale.
    Continuous,
}

/// Configuration for [`log_likelihood`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LikelihoodConfig {
    /// Distribution used to evaluate each present observation.
    pub family: LikelihoodFamily,
    /// Declared observation scale used by the cumulative-count guard.
    pub scale: ObservationScale,
}

/// Evaluate observation log-likelihood for `obs` given predicted `mean` values.
///
/// # Errors
/// Returns [`InnovationError`] if lengths mismatch, `mean` is non-finite, dispersion/scale/df
/// are invalid, `CumulativeCounts` is supplied to a count likelihood, or count observations
/// are negative/non-integer.
pub fn log_likelihood(
    obs: &DomainObservationSeries,
    mean: &[f64],
    config: &LikelihoodConfig,
) -> Result<f64, InnovationError> {
    let values = obs.values();
    if values.len() != mean.len() {
        return Err(InnovationError::new(
            "length_mismatch",
            "observation series and mean vector must have equal length",
        ));
    }
    if matches!(
        config.family,
        LikelihoodFamily::Poisson | LikelihoodFamily::NegativeBinomial { .. }
    ) && config.scale == ObservationScale::CumulativeCounts
    {
        return Err(InnovationError::new(
            "cumulative_count_misuse",
            "cumulative counts cannot be treated as independent increments in count likelihoods",
        ));
    }

    let mut total = 0.0_f64;
    for (&maybe_y, &mu) in values.iter().zip(mean) {
        let Some(y) = maybe_y else {
            continue;
        };
        if !y.is_finite() || !mu.is_finite() {
            return Err(InnovationError::new(
                "non_finite_input",
                "observations and means must be finite",
            ));
        }
        let term = match config.family {
            LikelihoodFamily::Poisson => {
                if y < 0.0 || (y - y.round()).abs() > 1e-9 {
                    return Err(InnovationError::new(
                        "invalid_count",
                        "Poisson likelihood requires nonnegative integer counts",
                    ));
                }
                if mu < 0.0 || (mu == 0.0 && y > 0.0) {
                    return Err(InnovationError::new(
                        "invalid_mean",
                        "Poisson mean must be positive for positive counts",
                    ));
                }
                if y == 0.0 {
                    -mu
                } else {
                    y * mu.ln() - mu - ln_gamma(y + 1.0)
                }
            }
            LikelihoodFamily::NegativeBinomial { dispersion } => {
                if !dispersion.is_finite() || dispersion <= 0.0 {
                    return Err(InnovationError::new(
                        "invalid_dispersion",
                        "NegativeBinomial dispersion must be finite and strictly positive",
                    ));
                }
                if y < 0.0 || (y - y.round()).abs() > 1e-9 {
                    return Err(InnovationError::new(
                        "invalid_count",
                        "NegativeBinomial likelihood requires nonnegative integer counts",
                    ));
                }
                if mu < 0.0 || (mu == 0.0 && y > 0.0) {
                    return Err(InnovationError::new(
                        "invalid_mean",
                        "NegativeBinomial mean must be positive for positive counts",
                    ));
                }
                let r = dispersion;
                if y == 0.0 {
                    r * (r / (r + mu)).ln()
                } else {
                    ln_gamma(y + r) - ln_gamma(r) - ln_gamma(y + 1.0)
                        + r * (r / (r + mu)).ln()
                        + y * (mu / (r + mu)).ln()
                }
            }
            LikelihoodFamily::Gaussian { sigma } => {
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(InnovationError::new(
                        "invalid_scale",
                        "Gaussian sigma must be finite and strictly positive",
                    ));
                }
                let z = (y - mu) / sigma;
                -0.5 * ((2.0 * std::f64::consts::PI).ln() + 2.0 * sigma.ln() + z * z)
            }
            LikelihoodFamily::StudentT { sigma, df } => {
                if !sigma.is_finite() || sigma <= 0.0 || !df.is_finite() || df <= 0.0 {
                    return Err(InnovationError::new(
                        "invalid_scale",
                        "StudentT sigma and df must be finite and strictly positive",
                    ));
                }
                let z = (y - mu) / sigma;
                ln_gamma(0.5 * (df + 1.0))
                    - ln_gamma(0.5 * df)
                    - 0.5 * (df * std::f64::consts::PI).ln()
                    - sigma.ln()
                    - 0.5 * (df + 1.0) * (1.0 + (z * z) / df).ln()
            }
        };
        if !term.is_finite() {
            return Err(InnovationError::new(
                "numerical_overflow",
                "log-likelihood term overflowed finite domain",
            ));
        }
        total += term;
    }
    Ok(total)
}
