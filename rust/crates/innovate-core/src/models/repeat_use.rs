//! Repeat-use, switching, and abandonment stock-flow conservation (A05).

use crate::inference::InnovationError;

/// Active-user stock vector across products/states.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveUseState {
    pub active: Vec<f64>,
    pub cumulative_starts: Vec<f64>,
    pub cumulative_stops: Vec<f64>,
}

/// Flow rates per unit time: starts, abandonment/stops, and pairwise switching matrix `switch[from][to]`.
#[derive(Debug, Clone, PartialEq)]
pub struct UseRates {
    pub starts: Vec<f64>,
    pub stops: Vec<f64>,
    pub switch: Vec<Vec<f64>>,
}

/// Supply intensity (transactions per active user per unit time) separating transactions from actor counts.
#[derive(Debug, Clone, PartialEq)]
pub struct SupplyIntensity {
    pub prescriptions_per_active: Vec<f64>,
}

/// Advance [`ActiveUseState`] by `dt`, enforcing exact stock-flow balance (A05).
///
/// $\Delta A_i = (\text{starts}_i + \sum_{j \ne i} \text{switch}_{j \to i} - \text{stops}_i - \sum_{j \ne i} \text{switch}_{i \to j}) \Delta t$.
///
/// # Errors
/// Returns [`InnovationError`] on negative rates/stocks, dimension mismatches, or negative resulting stock.
pub fn step_active_use(
    state: &ActiveUseState,
    rates: &UseRates,
    dt: f64,
) -> Result<ActiveUseState, InnovationError> {
    let k = state.active.len();
    if k == 0
        || state.cumulative_starts.len() != k
        || state.cumulative_stops.len() != k
        || rates.starts.len() != k
        || rates.stops.len() != k
        || rates.switch.len() != k
        || rates.switch.iter().any(|row| row.len() != k)
    {
        return Err(InnovationError::new(
            "invalid_dimension",
            "active use state and rates dimensions must match",
        ));
    }
    if !dt.is_finite()
        || dt < 0.0
        || state.active.iter().any(|x| !x.is_finite() || *x < 0.0)
        || rates.starts.iter().any(|x| !x.is_finite() || *x < 0.0)
        || rates.stops.iter().any(|x| !x.is_finite() || *x < 0.0)
        || rates
            .switch
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(InnovationError::new(
            "invalid_input",
            "stocks, rates, and dt must be finite and nonnegative",
        ));
    }

    let mut next_active = Vec::with_capacity(k);
    let mut next_starts = Vec::with_capacity(k);
    let mut next_stops = Vec::with_capacity(k);
    for i in 0..k {
        let in_switch: f64 = (0..k).filter(|&j| j != i).map(|j| rates.switch[j][i]).sum();
        let out_switch: f64 = (0..k).filter(|&j| j != i).map(|j| rates.switch[i][j]).sum();
        let updated =
            state.active[i] + (rates.starts[i] + in_switch - rates.stops[i] - out_switch) * dt;
        if updated < -1e-12 {
            return Err(InnovationError::new(
                "negative_stock",
                "active use stock cannot become negative",
            ));
        }
        next_active.push(updated.max(0.0));
        next_starts.push(state.cumulative_starts[i] + rates.starts[i] * dt);
        next_stops.push(state.cumulative_stops[i] + rates.stops[i] * dt);
    }
    Ok(ActiveUseState {
        active: next_active,
        cumulative_starts: next_starts,
        cumulative_stops: next_stops,
    })
}

/// Convert active user stock to expected transaction/prescription counts (A05).
///
/// # Errors
/// Returns [`InnovationError`] if lengths mismatch or intensities are negative/non-finite.
pub fn expected_transactions(
    state: &ActiveUseState,
    supply: &SupplyIntensity,
) -> Result<Vec<f64>, InnovationError> {
    if state.active.len() != supply.prescriptions_per_active.len() {
        return Err(InnovationError::new(
            "invalid_dimension",
            "supply intensity must match active stock dimension",
        ));
    }
    state
        .active
        .iter()
        .zip(&supply.prescriptions_per_active)
        .map(|(&a, &s)| {
            if !s.is_finite() || s < 0.0 {
                return Err(InnovationError::new(
                    "invalid_supply_intensity",
                    "supply intensity must be finite and nonnegative",
                ));
            }
            Ok(a * s)
        })
        .collect()
}
