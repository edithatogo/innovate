//! Additive composite hype expectation curve, ordered/unrestricted configs, and analytic derivatives (H01).

use crate::inference::InnovationError;

/// Single Gaussian pulse in a composite hype expectation curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HypePulse {
    /// Finite signed height; negative values represent downward pulses.
    pub amplitude: f64,
    /// Finite pulse-center coordinate in the evaluation time unit.
    pub center: f64,
    /// Finite positive Gaussian width in the same time unit as `center`.
    pub width: f64,
}

/// Structural configuration separating ordered hype cycles from unrestricted waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypeConfiguration {
    /// Require strictly increasing pulse centers; this alone does not enforce a canonical hype-cycle shape.
    OrderedCycle,
    /// Permit any order of finite pulse centers.
    UnrestrictedWave,
}

/// Parameters for [`hype_mean`] and [`hype_derivative`].
#[derive(Debug, Clone, PartialEq)]
pub struct HypeParameters {
    /// Finite additive level in the units of the observed hype signal.
    pub baseline: f64,
    /// Finite signed amplitude of the logistic maturity component.
    pub maturity_amplitude: f64,
    /// Finite positive logistic growth coefficient per time unit.
    pub growth: f64,
    /// Finite midpoint time of the logistic maturity component.
    pub t_maturity: f64,
    /// Additive Gaussian pulses, whose ordering is controlled by `configuration`.
    pub pulses: Vec<HypePulse>,
    /// Structural rule governing pulse-center order.
    pub configuration: HypeConfiguration,
}

impl HypeParameters {
    /// Validate finiteness, positive pulse widths, positive growth, and ordering if `OrderedCycle`.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if any parameter is non-finite, widths/growth are non-positive,
    /// or `OrderedCycle` pulse centers are not strictly increasing.
    pub fn validate(&self) -> Result<(), InnovationError> {
        if !self.baseline.is_finite()
            || !self.maturity_amplitude.is_finite()
            || !self.growth.is_finite()
            || self.growth <= 0.0
            || !self.t_maturity.is_finite()
        {
            return Err(InnovationError::new(
                "invalid_hype_parameters",
                "baseline, maturity_amplitude, growth (>0), and t_maturity must be finite",
            ));
        }
        for pulse in &self.pulses {
            if !pulse.amplitude.is_finite()
                || !pulse.center.is_finite()
                || !pulse.width.is_finite()
                || pulse.width <= 0.0
            {
                return Err(InnovationError::new(
                    "invalid_hype_pulse",
                    "pulse amplitude, center, and positive width must be finite",
                ));
            }
        }
        if self.configuration == HypeConfiguration::OrderedCycle
            && self.pulses.windows(2).any(|w| w[1].center <= w[0].center)
        {
            return Err(InnovationError::new(
                "unordered_hype_cycle",
                "OrderedCycle requires strictly increasing pulse centers",
            ));
        }
        Ok(())
    }
}

/// Evaluate additive composite hype mean $\mu(t)$ without hard clipping (H01).
///
/// # Errors
/// Returns [`InnovationError`] if `t` is non-finite or `p` fails validation.
pub fn hype_mean(t: f64, p: &HypeParameters) -> Result<f64, InnovationError> {
    if !t.is_finite() {
        return Err(InnovationError::new("invalid_time", "t must be finite"));
    }
    p.validate()?;
    let logistic = p.maturity_amplitude / (1.0 + (-p.growth * (t - p.t_maturity)).exp());
    let pulse_sum: f64 = p
        .pulses
        .iter()
        .map(|pulse| {
            let z = (t - pulse.center) / pulse.width;
            pulse.amplitude * (-0.5 * z * z).exp()
        })
        .sum();
    Ok(p.baseline + logistic + pulse_sum)
}

/// Evaluate exact analytic derivative $\mu'(t)$ of [`hype_mean`] (H01).
///
/// # Errors
/// Returns [`InnovationError`] if `t` is non-finite or `p` fails validation.
pub fn hype_derivative(t: f64, p: &HypeParameters) -> Result<f64, InnovationError> {
    if !t.is_finite() {
        return Err(InnovationError::new("invalid_time", "t must be finite"));
    }
    p.validate()?;
    let sig = 1.0 / (1.0 + (-p.growth * (t - p.t_maturity)).exp());
    let logistic_deriv = p.maturity_amplitude * p.growth * sig * (1.0 - sig);
    let pulse_deriv: f64 = p
        .pulses
        .iter()
        .map(|pulse| {
            let z = (t - pulse.center) / pulse.width;
            let gauss = (-0.5 * z * z).exp();
            -pulse.amplitude * gauss * (t - pulse.center) / (pulse.width * pulse.width)
        })
        .sum();
    Ok(logistic_deriv + pulse_deriv)
}
