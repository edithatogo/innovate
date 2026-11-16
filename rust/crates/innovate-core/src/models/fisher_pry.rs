use crate::inference::InnovationError;

/// Evaluate Fisher-Pry share $1 / (1 + \exp(-\alpha (t - t_0)))$ over `time`.
///
/// # Errors
/// Returns [`InnovationError`] if `alpha <= 0` or any input is non-finite.
pub fn evaluate(time: &[f64], alpha: f64, t0: f64) -> Result<Vec<f64>, InnovationError> {
    if !alpha.is_finite() || alpha <= 0.0 || !t0.is_finite() {
        return Err(InnovationError::new(
            "invalid_parameter",
            "FisherPry requires finite alpha > 0 and finite t0",
        ));
    }
    time.iter()
        .map(|&t| {
            if !t.is_finite() {
                return Err(InnovationError::new("invalid_time", "time must be finite"));
            }
            let val = 1.0 / (1.0 + (-alpha * (t - t0)).exp());
            Ok(val.clamp(0.0, 1.0))
        })
        .collect()
}
