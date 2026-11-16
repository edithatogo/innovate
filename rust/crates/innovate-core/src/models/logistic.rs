use crate::inference::InnovationError;

/// Evaluate Logistic curve $L / (1 + \exp(-k(t - x_0)))$ over `time`.
///
/// # Errors
/// Returns [`InnovationError`] if `L <= 0`, `k <= 0`, or any parameter/time is non-finite.
pub fn evaluate(time: &[f64], l: f64, k: f64, x0: f64) -> Result<Vec<f64>, InnovationError> {
    if !l.is_finite() || l <= 0.0 || !k.is_finite() || k <= 0.0 || !x0.is_finite() {
        return Err(InnovationError::new(
            "invalid_parameter",
            "Logistic requires finite L > 0, k > 0, and finite x0",
        ));
    }
    time.iter()
        .map(|&t| {
            if !t.is_finite() {
                return Err(InnovationError::new("invalid_time", "time must be finite"));
            }
            let val = l / (1.0 + (-k * (t - x0)).exp());
            Ok(val.clamp(0.0, l))
        })
        .collect()
}
