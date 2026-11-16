//! Explicit separation of external lag forcing vs genuine delayed state feedback (H05).

use crate::inference::InnovationError;

/// External lag signal evaluated at `t - lag` without using future information.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalLagSignal {
    /// Finite nonnegative delay in the same time unit as the evaluation coordinate.
    pub lag: f64,
    /// Finite value used when `t - lag` precedes zero, without calling the signal closure.
    pub baseline_value: f64,
}

impl ExternalLagSignal {
    /// Evaluate external signal `f(t - lag)` using provided closure `signal_fn`.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `t` or `lag < 0` is invalid or `signal_fn` produces non-finite output.
    pub fn evaluate<F>(&self, t: f64, signal_fn: F) -> Result<f64, InnovationError>
    where
        F: Fn(f64) -> f64,
    {
        if !t.is_finite()
            || !self.lag.is_finite()
            || self.lag < 0.0
            || !self.baseline_value.is_finite()
        {
            return Err(InnovationError::new(
                "invalid_lag_signal",
                "t and nonnegative lag must be finite",
            ));
        }
        let shifted = t - self.lag;
        let val = if shifted < 0.0 {
            self.baseline_value
        } else {
            signal_fn(shifted)
        };
        if !val.is_finite() {
            return Err(InnovationError::new(
                "non_finite_signal",
                "evaluated lag signal is not finite",
            ));
        }
        Ok(val)
    }
}

/// Coupled hype-adoption delayed state step using explicit state history (H05).
#[derive(Debug, Clone, PartialEq)]
pub struct DelayedStateModel {
    /// Number of stored observations back from the latest history entry;
    /// this is an index delay, with no interpolation of irregular time coordinates.
    pub delay_steps: usize,
    /// Finite signed coefficient multiplying the selected lagged adoption value.
    pub feedback_gain: f64,
    /// Finite nonnegative decay coefficient per time unit in `step_hype`.
    pub decay: f64,
}

impl DelayedStateModel {
    /// Advance hype state `h[t]` given adoption history `adoption_history` and exogenous shock.
    ///
    /// Uses one explicit Euler step; the caller chooses `dt` and its time unit.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if history is shorter than `delay_steps + 1` or inputs are non-finite.
    pub fn step_hype(
        &self,
        current_hype: f64,
        adoption_history: &[f64],
        exogenous_shock: f64,
        dt: f64,
    ) -> Result<f64, InnovationError> {
        if !current_hype.is_finite()
            || !exogenous_shock.is_finite()
            || !dt.is_finite()
            || dt < 0.0
            || !self.feedback_gain.is_finite()
            || !self.decay.is_finite()
            || self.decay < 0.0
        {
            return Err(InnovationError::new(
                "invalid_delayed_state",
                "delayed state inputs must be finite with nonnegative dt and decay",
            ));
        }
        if adoption_history.len() <= self.delay_steps {
            return Err(InnovationError::new(
                "insufficient_history",
                "adoption_history must cover at least delay_steps + 1 entries",
            ));
        }
        let lagged_adoption = adoption_history[adoption_history.len() - 1 - self.delay_steps];
        if !lagged_adoption.is_finite() {
            return Err(InnovationError::new(
                "non_finite_history",
                "lagged adoption must be finite",
            ));
        }
        let dh =
            (-self.decay * current_hype + self.feedback_gain * lagged_adoption + exogenous_shock)
                * dt;
        Ok(current_hype + dh)
    }
}
