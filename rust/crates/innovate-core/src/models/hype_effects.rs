//! Separate hype effects on innovation ($p$), imitation ($q$), and cessation (H04).

use crate::inference::InnovationError;

/// State separating cumulative first adoption from active use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnovationState {
    /// Finite nonnegative cumulative amount adopting for the first time.
    pub cumulative_first_adoption: f64,
    /// Finite nonnegative amount currently in use, on the same scale as adoption.
    pub active_use: f64,
}

/// Exogenous or observed hype level at time `t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HypeSignal {
    /// Finite signal value multiplying each log-linear effect coefficient.
    pub level: f64,
}

/// Separate log-linear hype effect coefficients on `p`, `q`, and cessation hazard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HypeEffectParameters {
    /// Finite positive baseline innovation coefficient per time unit.
    pub p0: f64,
    /// Finite nonnegative baseline imitation coefficient per time unit.
    pub q0: f64,
    /// Finite positive market capacity on the cumulative first-adoption scale.
    pub m: f64,
    /// Finite signed effect on innovation: `p = p0 * exp(beta_p * level)`.
    pub beta_p: f64,
    /// Finite signed effect on imitation: `q = q0 * exp(beta_q * level)`.
    pub beta_q: f64,
    /// Finite nonnegative baseline cessation hazard per time unit, applied to each unit of active use.
    pub h0_stop: f64,
    /// Finite signed effect on cessation: `h_stop = h0_stop * exp(beta_stop * level)`.
    pub beta_stop: f64,
}

/// Instantaneous rates returned by [`innovation_rates`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnovationRates {
    /// Instantaneous increase in cumulative first adoption per time unit.
    pub first_adoption_rate: f64,
    /// Net active-use change per time unit after cessation; this can be negative.
    pub active_use_rate: f64,
}

/// Compute separate hype-modulated first-adoption and active-use rates (H04).
///
/// Zero coefficients (`beta_p = beta_q = beta_stop = 0`) recover baseline Bass uptake and cessation.
/// Each multiplier is `exp(beta * level)`: a negative product reduces the
/// corresponding baseline coefficient, and a positive product increases it.
/// `beta_stop` acts on cessation, so reducing that multiplier lowers active-use
/// losses rather than inhibiting first adoption.
///
/// # Errors
/// Returns [`InnovationError`] if parameters or state are non-finite or out of domain.
pub fn innovation_rates(
    t: f64,
    state: &InnovationState,
    hype: &HypeSignal,
    p: &HypeEffectParameters,
) -> Result<InnovationRates, InnovationError> {
    if !t.is_finite()
        || !hype.level.is_finite()
        || !state.cumulative_first_adoption.is_finite()
        || state.cumulative_first_adoption < 0.0
        || !state.active_use.is_finite()
        || state.active_use < 0.0
        || !p.p0.is_finite()
        || p.p0 <= 0.0
        || !p.q0.is_finite()
        || p.q0 < 0.0
        || !p.m.is_finite()
        || p.m <= 0.0
        || !p.h0_stop.is_finite()
        || p.h0_stop < 0.0
        || !p.beta_p.is_finite()
        || !p.beta_q.is_finite()
        || !p.beta_stop.is_finite()
    {
        return Err(InnovationError::new(
            "invalid_hype_effect_input",
            "hype effect state, signal, and parameters must be finite and valid",
        ));
    }
    let p_eff = p.p0 * (p.beta_p * hype.level).exp();
    let q_eff = p.q0 * (p.beta_q * hype.level).exp();
    let h_stop = p.h0_stop * (p.beta_stop * hype.level).exp();

    let remaining = (p.m - state.cumulative_first_adoption).max(0.0);
    let first_adoption_rate =
        ((p_eff + q_eff * (state.cumulative_first_adoption / p.m)) * remaining).max(0.0);
    let active_use_rate = first_adoption_rate - h_stop * state.active_use;
    Ok(InnovationRates {
        first_adoption_rate,
        active_use_rate,
    })
}
