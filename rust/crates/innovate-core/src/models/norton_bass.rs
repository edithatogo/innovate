//! Norton-Bass multi-generation substitution trajectories (A04).

use crate::domain::bass_cumulative;
use crate::inference::InnovationError;

/// Evaluate 2-generation Norton-Bass cumulative active adoption $[S_1(t), S_2(t)]$.
///
/// Generation 2 launches at `tau2`; before `tau2`, $S_2(t) = 0$ and $S_1(t) = m_1 F_1(t)$.
/// At $t \ge \tau_2$, $S_1(t) = m_1 F_1(t)(1 - F_2(t - \tau_2))$ and
/// $S_2(t) = (m_2 + m_1 F_1(t)) F_2(t - \tau_2)$, conserving total active base
/// $S_1(t) + S_2(t) = m_1 F_1(t) + m_2 F_2(t - \tau_2)$.
///
/// # Errors
/// Returns [`InnovationError`] if parameters or time are invalid.
pub fn norton_bass_two_gen(
    t: f64,
    p1: f64,
    q1: f64,
    m1: f64,
    p2: f64,
    q2: f64,
    m2: f64,
    tau2: f64,
) -> Result<[f64; 2], InnovationError> {
    if !tau2.is_finite() || tau2 < 0.0 {
        return Err(InnovationError::new(
            "invalid_launch_time",
            "tau2 must be finite and nonnegative",
        ));
    }
    let s1_base = bass_cumulative(t, p1, q1, m1)?;
    if t < tau2 {
        return Ok([s1_base, 0.0]);
    }
    let f2 = bass_cumulative(t - tau2, p2, q2, 1.0)?;
    let s1 = s1_base * (1.0 - f2);
    let s2 = (m2 + s1_base) * f2;
    Ok([s1.max(0.0), s2.max(0.0)])
}
