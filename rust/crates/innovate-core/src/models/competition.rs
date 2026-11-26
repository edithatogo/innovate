//! Multi-product competing diffusion rates with shared market boundary (A04).

use crate::inference::InnovationError;

/// Current market state for competing products.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketState {
    /// Nonnegative adopted amounts in the shared product order.
    pub adoption: Vec<f64>,
}

/// Parameters for competing-product diffusion rates.
#[derive(Debug, Clone, PartialEq)]
pub struct CompetitionParameters {
    /// Finite positive capacity shared by all products.
    pub market_capacity: f64,
    /// Entry time for each product; its rate is zero before this time.
    pub entry_times: Vec<f64>,
    /// Nonnegative external-adoption coefficients per product and time unit.
    pub p: Vec<f64>,
    /// Nonnegative imitation coefficients multiplying each product's own share.
    pub q: Vec<f64>,
    /// Interaction matrix `alpha[i][j]` (product `j` effect on product `i`).
    pub interaction: Vec<Vec<f64>>,
}

/// Instantaneous adoption rates returned by [`competing_rates`].
#[derive(Debug, Clone, PartialEq)]
pub struct MarketRates {
    /// Adoption amounts per time unit, aligned with [`MarketState::adoption`].
    pub rates: Vec<f64>,
}

/// Compute instantaneous adoption rates for competing products under a shared market capacity (A04).
///
/// # Errors
/// Returns [`InnovationError`] on mismatched dimensions, non-finite time, invalid
/// capacity, or non-finite/negative adoption and `p`/`q` values. Entry times and
/// interaction coefficients are caller-supplied; this function does not
/// independently validate their finiteness.
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{CompetitionParameters, MarketState, competing_rates};
    use crate::numerics::integrate;

    #[test]
    fn symmetric_trajectory_matches_the_closed_form_logistic_reference() {
        let q = 3.0_f64.ln();
        let params = CompetitionParameters {
            market_capacity: 100.0,
            entry_times: vec![0.0, 0.0],
            p: vec![0.0, 0.0],
            q: vec![q, q],
            interaction: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        };
        let initial = MarketState {
            adoption: vec![25.0, 25.0],
        };
        let rates = competing_rates(0.0, &initial, &params).unwrap();
        for rate in rates.rates {
            assert!((rate - 12.5 * q).abs() < 1e-12);
        }

        let evaluations = Cell::new(0);
        let times = [0.0, 0.5, 1.0, 2.0];
        let rows = integrate(&times, 0.0, &initial.adoption, |t, adoption| {
            evaluations.set(evaluations.get() + 1);
            let state = MarketState {
                adoption: adoption.to_vec(),
            };
            competing_rates(t, &state, &params).unwrap().rates
        })
        .unwrap();

        assert_eq!(rows.len(), times.len());
        let mut max_total_error = 0.0_f64;
        for (time, row) in times.into_iter().zip(&rows) {
            let [first, second] = row.as_slice() else {
                panic!("expected two products at time {time}");
            };
            // Summing the two rates yields logistic growth with initial total 50.
            let expected_total = 100.0 / (1.0 + (-q * time).exp());
            let observed_total = first + second;
            assert_eq!(first, second);
            assert!((50.0..=100.0).contains(&observed_total));
            max_total_error = max_total_error.max((observed_total - expected_total).abs());
        }
        assert!((rows[2][0] - 37.5).abs() < 5e-7);
        assert!(max_total_error < 1e-6, "total error {max_total_error:e}");
        assert!((1..=512).contains(&evaluations.get()));
        eprintln!(
            "symmetric A04 maximum total error {max_total_error:e}, {} rate evaluations",
            evaluations.get()
        );
    }
}
