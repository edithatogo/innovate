use crate::inference::InnovationError;

/// Evaluate Gompertz curve $a \exp(-b \exp(-c t))$ over `time`.
///
/// # Errors
/// Returns [`InnovationError`] if `a <= 0`, `b <= 0`, `c <= 0`, or any input is non-finite.
pub fn evaluate(time: &[f64], a: f64, b: f64, c: f64) -> Result<Vec<f64>, InnovationError> {
    if !a.is_finite() || a <= 0.0 || !b.is_finite() || b <= 0.0 || !c.is_finite() || c <= 0.0 {
        return Err(InnovationError::new(
            "invalid_parameter",
            "Gompertz requires finite positive a, b, c",
        ));
    }
    time.iter()
        .map(|&t| {
            if !t.is_finite() {
                return Err(InnovationError::new("invalid_time", "time must be finite"));
            }
            let val = a * (-b * (-c * t).exp()).exp();
            Ok(val.clamp(0.0, a))
        })
        .collect()
}
