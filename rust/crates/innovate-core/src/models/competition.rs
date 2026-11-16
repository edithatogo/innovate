//! Multi-product competing diffusion rates with shared market boundary (A04).

use crate::inference::InnovationError;

/// Current market state for competing products.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketState {
    pub adoption: Vec<f64>,
}

/// Parameters for competing-product diffusion rates.
#[derive(Debug, Clone, PartialEq)]
pub struct CompetitionParameters {
    pub market_capacity: f64,
    pub entry_times: Vec<f64>,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    /// Interaction matrix `alpha[i][j]` (product `j` effect on product `i`).
    pub interaction: Vec<Vec<f64>>,
}

/// Instantaneous adoption rates returned by [`competing_rates`].
#[derive(Debug, Clone, PartialEq)]
pub struct MarketRates {
    pub rates: Vec<f64>,
}

/// Compute instantaneous adoption rates for competing products under a shared market capacity (A04).
///
/// # Errors
/// Returns [`InnovationError`] on mismatched dimensions, negative states, or non-finite inputs.
pub fn competing_rates(
    t: f64,
    state: &MarketState,
    params: &CompetitionParameters,
) -> Result<MarketRates, InnovationError> {
    let k = state.adoption.len();
    if k == 0
        || params.entry_times.len() != k
        || params.p.len() != k
        || params.q.len() != k
        || params.interaction.len() != k
        || params.interaction.iter().any(|row| row.len() != k)
    {
        return Err(InnovationError::new(
            "invalid_dimension",
            "competition state and parameter dimensions must match",
        ));
    }
    if !t.is_finite()
        || !params.market_capacity.is_finite()
        || params.market_capacity <= 0.0
        || state.adoption.iter().any(|x| !x.is_finite() || *x < 0.0)
        || params.p.iter().any(|x| !x.is_finite() || *x < 0.0)
        || params.q.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(InnovationError::new(
            "invalid_parameter",
            "competition inputs must be finite and nonnegative",
        ));
    }
    let total: f64 = state.adoption.iter().sum();
    let remaining = (params.market_capacity - total).max(0.0);
    let mut rates = vec![0.0_f64; k];
    for i in 0..k {
        if t < params.entry_times[i] {
            rates[i] = 0.0;
            continue;
        }
        let own_share = state.adoption[i] / params.market_capacity;
        let mut pressure = params.p[i] + params.q[i] * own_share;
        for j in 0..k {
            if i != j && t >= params.entry_times[j] {
                pressure += params.interaction[i][j] * (state.adoption[j] / params.market_capacity);
            }
        }
        rates[i] = (pressure * remaining).max(0.0);
    }
    Ok(MarketRates { rates })
}
