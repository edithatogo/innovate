//! Multinomial-logit structural demand shared by Innovate's native callers.
#![forbid(unsafe_code)]

use std::fmt;

/// Typed error returned when model state or inputs violate scientific contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnovationError {
    /// Machine-readable error category.
    pub code: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl InnovationError {
    /// Construct a typed `InnovationError`.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for InnovationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for InnovationError {}

/// Choice set of inside alternatives (with observable attributes and prices).
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceSet {
    /// Finite quality attributes in alternative order, excluding the outside option.
    pub quality: Vec<f64>,
    /// Finite prices in the same alternative order and caller-defined currency.
    pub price: Vec<f64>,
}

/// Preference parameters for [`choice_shares`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreferenceParameters {
    /// Signed coefficient of quality in utility `alpha_quality * quality + beta_price * price`.
    pub alpha_quality: f64,
    /// Signed price coefficient in utility; a negative value represents price aversion.
    pub beta_price: f64,
}

/// Compute multinomial logit choice shares $[s_1, \dots, s_J, s_0]$ including the outside option $s_0$ (A08).
///
/// Outside option utility is normalized to $0$.
///
/// # Errors
/// Returns [`InnovationError`] on empty/mismatched choice sets or non-finite inputs.
pub fn choice_shares(
    choices: &ChoiceSet,
    params: &PreferenceParameters,
) -> Result<Vec<f64>, InnovationError> {
    if choices.quality.is_empty()
        || choices.quality.len() != choices.price.len()
        || !params.alpha_quality.is_finite()
        || !params.beta_price.is_finite()
        || choices.quality.iter().any(|v| !v.is_finite())
        || choices.price.iter().any(|v| !v.is_finite())
    {
        return Err(InnovationError::new(
            "invalid_choice_set",
            "choice set and preference parameters must be nonempty and finite",
        ));
    }
    let utilities: Vec<f64> = choices
        .quality
        .iter()
        .zip(&choices.price)
        .map(|(&q, &p)| params.alpha_quality * q + params.beta_price * p)
        .collect();
    let max_u = utilities.iter().copied().fold(0.0_f64, f64::max);
    let mut exps: Vec<f64> = utilities.iter().map(|&u| (u - max_u).exp()).collect();
    let outside = (-max_u).exp();
    let denom = outside + exps.iter().sum::<f64>();
    for e in &mut exps {
        *e /= denom;
    }
    exps.push(outside / denom);
    Ok(exps)
}
