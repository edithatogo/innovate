//! Classical, hierarchical, competing, repeat-use, regime, network, and demand models (A01–A08).

pub mod bass;
pub mod competition;
pub mod demand;
pub mod expectation_feedback;
pub mod fisher_pry;
pub mod gompertz;
pub mod hierarchical;
pub mod hype;
pub mod hype_effects;
pub mod logistic;
pub mod network;
pub mod norton_bass;
pub mod regimes;
pub mod repeat_use;
pub mod semi_markov;

use crate::KernelBinding;
use crate::domain::TimeGrid;
use crate::inference::{FitFailure, FitOutcome, InnovationError};
use serde_json::json;

/// Supported classical scalar diffusion families (A01).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassicalFamily {
    Bass,
    Logistic,
    Gompertz,
    FisherPry,
}

impl ClassicalFamily {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bass => "bass",
            Self::Logistic => "logistic",
            Self::Gompertz => "gompertz",
            Self::FisherPry => "fisher_pry",
        }
    }
}

/// Request for [`fit_classical`].
#[derive(Debug, Clone, PartialEq)]
pub struct FitRequest {
    pub family: ClassicalFamily,
    pub time: TimeGrid,
    pub observed: Vec<f64>,
}

/// Request for [`predict_classical`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionRequest {
    pub family: ClassicalFamily,
    pub time: TimeGrid,
    pub parameters: Vec<f64>,
}

/// Prediction output bundle returned by [`predict_classical`] and [`regimes::predict_regimes`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionBundle {
    pub family: String,
    pub time: Vec<f64>,
    pub values: Vec<f64>,
    pub weakly_identified: bool,
}

/// Fit a classical scalar diffusion curve via the shared `innovate-core` optimizer (A01).
#[must_use]
pub fn fit_classical(request: &FitRequest) -> FitOutcome {
    if request.observed.len() != request.time.len()
        || request.observed.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return FitOutcome::Invalid(FitFailure {
            code: "invalid_data".into(),
            message: "observed values must align with time and be finite nonnegative".into(),
        });
    }
    let payload = json!({
        "inputs": {
            "time": request.time.as_slice(),
            "observed": request.observed,
        }
    });
    let response = match KernelBinding::new().fit_model(request.family.as_str(), payload) {
        Ok(resp) => resp,
        Err(err) => {
            return FitOutcome::Invalid(FitFailure {
                code: "fit_failed".into(),
                message: err.to_string(),
            });
        }
    };
    let Some(result) = response.result else {
        return FitOutcome::Invalid(FitFailure {
            code: "missing_result".into(),
            message: "native fit returned no result".into(),
        });
    };
    let Some(params_obj) = result["parameters"].as_object() else {
        return FitOutcome::Invalid(FitFailure {
            code: "missing_parameters".into(),
            message: "native fit returned no parameters".into(),
        });
    };
    let mut names = Vec::with_capacity(params_obj.len());
    let mut vals = Vec::with_capacity(params_obj.len());
    for (k, v) in params_obj {
        let Some(num) = v.as_f64().filter(|x| x.is_finite()) else {
            return FitOutcome::Invalid(FitFailure {
                code: "non_finite_parameter".into(),
                message: format!("parameter {k} is not finite"),
            });
        };
        names.push(k.clone());
        vals.push(num);
    }
    let fitted_capacity = params_obj
        .get("m")
        .or_else(|| params_obj.get("L"))
        .or_else(|| params_obj.get("a"))
        .and_then(|v| v.as_f64());
    let last_observed = request.observed.last().copied().unwrap_or(0.0);
    let is_pre_inflection = request.observed.len() < 5
        || fitted_capacity.is_some_and(|cap| cap.is_finite() && last_observed < 0.5 * cap);
    let diag = if is_pre_inflection {
        "converged; warning: weakly_identified pre-inflection or short series"
    } else {
        "converged"
    };
    if vals.iter().any(|v| !v.is_finite()) {
        return FitOutcome::Invalid(FitFailure {
            code: "non_finite_estimate".into(),
            message: "fitted parameters contain non-finite values".into(),
        });
    }
    FitOutcome::Deterministic(crate::inference::DeterministicFitResult {
        names,
        values: vals,
        diagnostics: diag.into(),
    })
}

/// Predict a classical scalar diffusion curve (A01).
///
/// # Errors
/// Returns [`InnovationError`] if parameters are invalid or out of domain.
pub fn predict_classical(request: &PredictionRequest) -> Result<PredictionBundle, InnovationError> {
    let values = match request.family {
        ClassicalFamily::Bass => {
            if request.parameters.len() != 3 {
                return Err(InnovationError::new(
                    "invalid_dimension",
                    "Bass requires [p, q, m]",
                ));
            }
            bass::evaluate(
                request.time.as_slice(),
                request.parameters[0],
                request.parameters[1],
                request.parameters[2],
            )?
        }
        ClassicalFamily::Logistic => {
            if request.parameters.len() != 3 {
                return Err(InnovationError::new(
                    "invalid_dimension",
                    "Logistic requires [L, k, x0]",
                ));
            }
            logistic::evaluate(
                request.time.as_slice(),
                request.parameters[0],
                request.parameters[1],
                request.parameters[2],
            )?
        }
        ClassicalFamily::Gompertz => {
            if request.parameters.len() != 3 {
                return Err(InnovationError::new(
                    "invalid_dimension",
                    "Gompertz requires [a, b, c]",
                ));
            }
            gompertz::evaluate(
                request.time.as_slice(),
                request.parameters[0],
                request.parameters[1],
                request.parameters[2],
            )?
        }
        ClassicalFamily::FisherPry => {
            if request.parameters.len() != 2 {
                return Err(InnovationError::new(
                    "invalid_dimension",
                    "FisherPry requires [alpha, t0]",
                ));
            }
            fisher_pry::evaluate(
                request.time.as_slice(),
                request.parameters[0],
                request.parameters[1],
            )?
        }
    };
    Ok(PredictionBundle {
        family: request.family.as_str().into(),
        time: request.time.as_slice().to_vec(),
        values,
        weakly_identified: false,
    })
}
