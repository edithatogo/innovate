//! Dated regime, shock, and time-varying market-potential predictions (A06).

use crate::domain::TimeGrid;
use crate::inference::InnovationError;
use crate::models::{ClassicalFamily, PredictionBundle, PredictionRequest, predict_classical};

/// Dated shock or market-potential shift applied from `effective_time` onward.
#[derive(Debug, Clone, PartialEq)]
pub struct DatedShock {
    /// Finite time from which this shock participates, inclusive of the boundary.
    pub effective_time: f64,
    /// Positive multiplier of Bass `p`, logistic `k`, Gompertz `c`, or Fisher–Pry `alpha`.
    pub p_multiplier: f64,
    /// Positive multiplier of Bass `q`; validated but unused by the other families.
    pub q_multiplier: f64,
    /// Positive multiplier of Bass `m`, logistic `L`, or Gompertz `a`;
    /// validated but unused for Fisher–Pry shares.
    pub capacity_multiplier: f64,
}

/// Request for [`predict_regimes`].
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeRequest {
    /// Classical family whose parameters receive the applicable multipliers.
    pub family: ClassicalFamily,
    /// Validated coordinates at which to evaluate the regime-adjusted curve.
    pub time: TimeGrid,
    /// Baseline values in the positional layout of [`PredictionRequest::parameters`].
    pub base_parameters: Vec<f64>,
    /// Dated multipliers; every applicable entry is multiplied into the baseline
    /// at each query time, irrespective of the order of this vector.
    pub shocks: Vec<DatedShock>,
}

/// Predict cumulative diffusion across dated regime shocks without using future shocks early (A06).
///
/// Guarantees:
/// - No-shock limit reproduces [`predict_classical`] identically.
/// - Decreasing market capacity never produces negative cumulative adoption (monotone cumulative floor).
/// - No shock with `effective_time > t` influences predictions at `t`.
///
/// # Errors
/// Returns [`InnovationError`] if shock multipliers are non-positive/non-finite or base parameters fail.
pub fn predict_regimes(request: &RegimeRequest) -> Result<PredictionBundle, InnovationError> {
    for shock in &request.shocks {
        if !shock.effective_time.is_finite()
            || !shock.p_multiplier.is_finite()
            || shock.p_multiplier <= 0.0
            || !shock.q_multiplier.is_finite()
            || shock.q_multiplier <= 0.0
            || !shock.capacity_multiplier.is_finite()
            || shock.capacity_multiplier <= 0.0
        {
            return Err(InnovationError::new(
                "invalid_shock",
                "shock effective_time must be finite and multipliers strictly positive",
            ));
        }
    }

    let baseline = predict_classical(&PredictionRequest {
        family: request.family,
        time: request.time.clone(),
        parameters: request.base_parameters.clone(),
    })?;
    if request.shocks.is_empty() {
        return Ok(baseline);
    }

    let mut out = Vec::with_capacity(request.time.len());
    let mut prev = 0.0_f64;
    for &t in request.time.as_slice() {
        let mut params = request.base_parameters.clone();
        for shock in &request.shocks {
            if t >= shock.effective_time {
                match request.family {
                    ClassicalFamily::Bass => {
                        params[0] *= shock.p_multiplier;
                        params[1] *= shock.q_multiplier;
                        params[2] *= shock.capacity_multiplier;
                    }
                    ClassicalFamily::Logistic => {
                        params[0] *= shock.capacity_multiplier;
                        params[1] *= shock.p_multiplier;
                    }
                    ClassicalFamily::Gompertz => {
                        params[0] *= shock.capacity_multiplier;
                        params[2] *= shock.p_multiplier;
                    }
                    ClassicalFamily::FisherPry => {
                        params[0] *= shock.p_multiplier;
                    }
                }
            }
        }
        let step = predict_classical(&PredictionRequest {
            family: request.family,
            time: TimeGrid::try_from(vec![t])?,
            parameters: params,
        })?;
        // Decreasing capacity must never create negative cumulative adoption.
        let val = step.values[0].max(prev);
        prev = val;
        out.push(val);
    }

    Ok(PredictionBundle {
        family: format!("{}_regimes", request.family.as_str()),
        time: request.time.as_slice().to_vec(),
        values: out,
        weakly_identified: false,
    })
}
