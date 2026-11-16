use crate::domain::bass_cumulative;
use crate::inference::InnovationError;

/// Evaluate Bass cumulative adoption over `time`.
///
/// # Errors
/// Returns [`InnovationError`] if any parameter or time is invalid.
pub fn evaluate(time: &[f64], p: f64, q: f64, m: f64) -> Result<Vec<f64>, InnovationError> {
    time.iter().map(|&t| bass_cumulative(t, p, q, m)).collect()
}
