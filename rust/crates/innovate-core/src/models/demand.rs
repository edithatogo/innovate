//! Multinomial logit structural demand choice shares with outside option (A08 core slice).

use crate::inference::InnovationError;

/// Choice set of inside alternatives (with observable attributes and prices).
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceSet {
    pub quality: Vec<f64>,
    pub price: Vec<f64>,
}

/// Preference parameters for [`choice_shares`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreferenceParameters {
    pub alpha_quality: f64,
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
