//! Typed Kairos simulation adapter over `innovate-core` (E01).

use innovate_core::domain::TimeGrid;
use innovate_core::inference::InnovationError;
use innovate_core::{KernelBinding, json};

/// Explicit Kairos runtime configuration enforcing one clock, RNG seed, and population.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KairosRuntime {
    pub engine: String,
    pub seed: u64,
    pub population: u64,
    pub ticks_per_unit: u64,
}

/// Typed innovation simulation request for [`run_innovation`].
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationRequest {
    pub model_key: String,
    pub time: TimeGrid,
    pub p: f64,
    pub q: f64,
    pub m: f64,
}

/// Typed simulation output returned by [`run_innovation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationResult {
    pub engine: String,
    pub counts: Vec<u64>,
    pub dispatched_events: u64,
}

/// Execute a native Kairos simulation via `innovate-core` (E01).
///
/// # Errors
/// Returns [`InnovationError`] if `engine != "kairos_des"`, parameters/population are invalid,
/// or native execution fails.
pub fn run_innovation(
    request: &SimulationRequest,
    runtime: &KairosRuntime,
) -> Result<SimulationResult, InnovationError> {
    if runtime.engine != "kairos_des" {
        return Err(InnovationError::new(
            "unsupported_engine",
            format!("unsupported simulation engine '{}'", runtime.engine),
        ));
    }
    if runtime.population == 0 || runtime.ticks_per_unit == 0 {
        return Err(InnovationError::new(
            "invalid_runtime",
            "population and ticks_per_unit must be positive",
        ));
    }
    let payload = json!({
        "inputs": {"time": request.time.as_slice()},
        "parameters": {"p": request.p, "q": request.q, "m": request.m},
        "simulation_options": {
            "engine": runtime.engine,
            "seed": runtime.seed,
            "population": runtime.population,
            "ticks_per_unit": runtime.ticks_per_unit,
        }
    });
    let response = KernelBinding::new()
        .simulate_model(&request.model_key, payload)
        .map_err(|err| InnovationError::new("simulation_failed", err.to_string()))?;
    let result = response
        .result
        .ok_or_else(|| InnovationError::new("missing_result", "missing simulation result"))?;
    let counts: Vec<u64> = result["counts"]
        .as_array()
        .ok_or_else(|| InnovationError::new("invalid_output", "missing counts array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| InnovationError::new("invalid_output", "count must be u64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dispatched_events = result["metadata"]["kairos_dispatched_events"]
        .as_u64()
        .ok_or_else(|| {
            InnovationError::new(
                "invalid_output",
                "missing kairos_dispatched_events metadata",
            )
        })?;
    Ok(SimulationResult {
        engine: runtime.engine.clone(),
        counts,
        dispatched_events,
    })
}
