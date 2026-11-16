//! Network exposure and complement-effect helpers (A07 core slice).

use crate::inference::InnovationError;

/// Compute permutation-invariant weighted neighbor exposure for each node.
///
/// # Errors
/// Returns [`InnovationError`] on dimension mismatch, non-finite weights, or invalid states.
pub fn neighbor_exposure(
    adjacency: &[Vec<f64>],
    adopted: &[f64],
) -> Result<Vec<f64>, InnovationError> {
    let n = adopted.len();
    if n == 0
        || adjacency.len() != n
        || adjacency.iter().any(|row| row.len() != n)
        || adopted.iter().any(|v| !v.is_finite() || *v < 0.0)
        || adjacency
            .iter()
            .flatten()
            .any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(InnovationError::new(
            "invalid_network",
            "adjacency and adopted states must be square, finite, and nonnegative",
        ));
    }
    let mut exposure = Vec::with_capacity(n);
    for (i, row) in adjacency.iter().enumerate() {
        let mut weight_sum = 0.0_f64;
        let mut active_sum = 0.0_f64;
        for (j, &w) in row.iter().enumerate() {
            if i != j {
                weight_sum += w;
                active_sum += w * adopted[j];
            }
        }
        exposure.push(if weight_sum > 0.0 {
            active_sum / weight_sum
        } else {
            0.0
        });
    }
    Ok(exposure)
}
