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
    let predict = curve(a, b, c);
    time.iter()
        .map(|&t| {
            if !t.is_finite() {
                return Err(InnovationError::new("invalid_time", "time must be finite"));
            }
            Ok(predict(t).0)
        })
        .collect()
}

/// Evaluate Gompertz growth and its analytic derivative for validated parameters.
pub(crate) fn curve(a: f64, b: f64, c: f64) -> impl Fn(f64) -> (f64, f64) {
    let log_a = a.ln();
    let log_b = b.ln();
    let log_c = c.ln();
    move |t| {
        let decay = (-c * t).exp();
        let log_argument = log_b - c * t;
        let argument = if decay.is_normal() {
            b * decay
        } else {
            log_argument.exp()
        };
        if argument == f64::INFINITY {
            return (0.0, 0.0);
        }
        let fraction = (-argument).exp();
        let level = if fraction.is_normal() {
            a * fraction
        } else {
            (log_a - argument).exp()
        };
        let scaled_rate = level * argument;
        let rate = if argument.is_normal() && scaled_rate.is_normal() && c.is_normal() {
            scaled_rate * c
        } else {
            (log_a + log_c + log_argument - argument).exp()
        };
        (level, rate)
    }
}
