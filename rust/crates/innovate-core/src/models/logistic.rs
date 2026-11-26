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
    let predict = curve(l, k, x0);
    time.iter()
        .map(|&t| {
            if !t.is_finite() {
                return Err(InnovationError::new("invalid_time", "time must be finite"));
            }
            Ok(predict(t).0)
        })
        .collect()
}

/// Evaluate the scaled sigmoid and its derivative for validated parameters.
pub(crate) fn curve(l: f64, k: f64, x0: f64) -> impl Fn(f64) -> (f64, f64) {
    let log_l = l.ln();
    let log_k = k.ln();
    move |t| {
        let elapsed = t - x0;
        let logit = if elapsed.is_finite() {
            k * elapsed
        } else {
            k * t - k * x0
        };
        let tail = (-logit.abs()).exp();
        let denominator = 1.0 + tail;
        // A small sigmoid tail can underflow before multiplication by the market size.
        let scaled_tail = if tail.is_normal() {
            l * tail
        } else {
            (log_l - logit.abs()).exp()
        };
        let level = if logit >= 0.0 { l } else { scaled_tail } / denominator;
        let rate = if tail.is_normal() && scaled_tail.is_normal() && k.is_normal() {
            (scaled_tail / denominator) * (k / denominator)
        } else {
            (log_l + log_k - logit.abs() - 2.0 * tail.ln_1p()).exp()
        };
        (level, rate)
    }
}
