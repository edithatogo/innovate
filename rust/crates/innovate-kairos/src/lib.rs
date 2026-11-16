//! Typed Kairos simulation adapter over `innovate-core` (E01).
//!
//! [`run_innovation`] exposes the Bass finite-population pure-birth process with
//! the `kairos_des` engine. Innovate supplies the adoption hazard; Kairos owns
//! the event clock, dispatch and random stream. The returned trajectory contains
//! cumulative actor counts, with zero initial adopters.

use innovate_core::domain::TimeGrid;
use innovate_core::inference::InnovationError;
use innovate_core::{KernelBinding, json};

/// Explicit Kairos runtime configuration enforcing one clock, RNG seed, and population.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KairosRuntime {
    /// Engine identifier; this adapter accepts `"kairos_des"`.
    pub engine: String,
    /// Run seed for the Kairos random stream.
    ///
    /// Replay requires the same request, clock resolution and pinned provider
    /// implementation; this field does not promise stability across revisions.
    pub seed: u64,
    /// Number of potential adopters, in `1..=100_000`.
    ///
    /// This actor count supplies the hazard denominator and bounds cumulative
    /// adoption independently of [`SimulationRequest::m`].
    pub population: u64,
    /// Integer clock ticks per caller-selected time unit, in `1..=1_000_000_000`.
    ///
    /// Native waiting times round upward to at least one tick. Observation
    /// coordinates round downward to their corresponding tick.
    pub ticks_per_unit: u64,
}

/// Typed innovation simulation request for [`run_innovation`].
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationRequest {
    /// Native model key; this adapter's supported simulation model is `"bass"`.
    pub model_key: String,
    /// Observation coordinates in one consistent time unit.
    ///
    /// The grid is nonempty, finite and strictly increasing. Simulation also
    /// requires nonnegative coordinates whose tick values fit the native clock.
    pub time: TimeGrid,
    /// Finite, nonnegative innovation rate per unit of [`Self::time`].
    pub p: f64,
    /// Finite, nonnegative imitation coefficient per unit of [`Self::time`].
    pub q: f64,
    /// Positive finite scale for the native adopted-level quantity.
    ///
    /// The native level is `m * count / population`; this adapter returns the
    /// integer counts, whose denominator is [`KairosRuntime::population`].
    pub m: f64,
}

/// Typed simulation output returned by [`run_innovation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationResult {
    /// Executed engine identifier, `"kairos_des"`.
    pub engine: String,
    /// Cumulative adopted actors at each request time, in the same order.
    ///
    /// Counts are nondecreasing and bounded by the runtime population.
    pub counts: Vec<u64>,
    /// Number of adoption events dispatched by Kairos through the final time.
    ///
    /// This is native `kairos_dispatched_events` metadata, not the number of
    /// observations or the size of the potential population.
    pub dispatched_events: u64,
}

/// Execute a native Kairos simulation via `innovate-core` (E01).
///
/// Returns cumulative actor counts for [`SimulationRequest::time`] and the
/// dispatched-event count. The model hazard at `n` adopted actors out of `N`
/// potential adopters is `(p + q * n / N) * (N - n)` per time unit.
///
/// # Errors
/// Returns [`InnovationError`] with code `unsupported_engine` for any engine
/// other than `"kairos_des"`, or `invalid_runtime` for zero population or clock
/// resolution. Native model, parameter, clock-domain and population-limit errors
/// are propagated as `simulation_failed`. A response without a result produces
/// `missing_result`; missing or non-unsigned counts/event metadata produce
/// `invalid_output`.
///
/// # Examples
///
/// Repeated execution with the same configuration replays within the pinned
/// implementation. The assertions check actor counts without assuming a
/// particular random trajectory.
///
/// ```
/// use innovate_core::domain::TimeGrid;
/// use innovate_kairos::{KairosRuntime, SimulationRequest, run_innovation};
///
/// let request = SimulationRequest {
///     model_key: "bass".into(),
///     time: TimeGrid::try_from(vec![0.0, 1.0, 5.0])?,
///     p: 0.2,
///     q: 0.3,
///     m: 100.0,
/// };
/// let runtime = KairosRuntime {
///     engine: "kairos_des".into(),
///     seed: 42,
///     population: 100,
///     ticks_per_unit: 1_000_000,
/// };
/// let result = run_innovation(&request, &runtime)?;
/// assert_eq!(result, run_innovation(&request, &runtime)?);
/// assert_eq!(result.counts.len(), request.time.len());
/// assert!(result.counts.iter().all(|&count| count <= runtime.population));
/// # Ok::<(), innovate_core::inference::InnovationError>(())
/// ```
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
