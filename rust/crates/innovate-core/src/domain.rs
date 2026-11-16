//! Typed time-grid, observation, unit, and denominator contracts (R02).

use crate::inference::InnovationError;

/// Strictly increasing, finite, non-empty time grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeGrid {
    times: Vec<f64>,
}

impl TimeGrid {
    /// Borrow the validated strictly increasing time coordinates.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.times
    }

    /// Number of time points in the grid.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the grid is empty (never true for a constructed `TimeGrid`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

impl TryFrom<Vec<f64>> for TimeGrid {
    type Error = InnovationError;

    fn try_from(times: Vec<f64>) -> Result<Self, Self::Error> {
        if times.is_empty() {
            return Err(InnovationError::new(
                "empty_time_grid",
                "time grid must be nonempty",
            ));
        }
        if times.iter().any(|t| !t.is_finite()) {
            return Err(InnovationError::new(
                "non_finite_time",
                "time grid must contain only finite values",
            ));
        }
        if times.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InnovationError::new(
                "unsorted_or_duplicate_time",
                "time grid must be strictly increasing without duplicates",
            ));
        }
        Ok(Self { times })
    }
}

/// Semantic kind of observed series; observations do not infer an underlying actor count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationKind {
    /// New adoption during each observation interval.
    NewAdoption,
    /// Actors or units currently active at each observation time.
    ActiveStock,
    /// Transactions observed at each time point.
    Transactions,
    /// A rate measured per unit time or expressed as a proportion.
    Rate,
    /// A share of an explicitly defined population or market.
    Share,
    /// An attention or expectation measure, distinct from adoption counts.
    Attention,
}

/// Measurement units for an observation series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Units {
    /// A count of observed events or units.
    Count,
    /// A dimensionless proportion.
    Proportion,
    /// A rate expressed per unit of time.
    RatePerUnitTime,
    /// A monetary amount in the caller's declared currency.
    Currency,
    /// An index on the caller's declared measurement scale.
    Index,
}

/// Optional explicit denominator accompanying normalized or share observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Denominator {
    /// Size of the population underlying a normalized observation.
    Population(f64),
    /// Market potential underlying a normalized observation.
    MarketPotential(f64),
}

impl Denominator {
    fn validate(self) -> Result<(), InnovationError> {
        let value = match self {
            Self::Population(v) | Self::MarketPotential(v) => v,
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(InnovationError::new(
                "invalid_denominator",
                "denominator must be finite and strictly positive",
            ));
        }
        Ok(())
    }
}

/// Typed observation series preserving missingness (`None`) distinct from zero (`Some(0.0)`).
#[derive(Debug, Clone, PartialEq)]
pub struct DomainObservationSeries {
    grid: TimeGrid,
    values: Vec<Option<f64>>,
    kind: ObservationKind,
    units: Units,
    denominator: Option<Denominator>,
}

impl DomainObservationSeries {
    /// Construct and validate a [`DomainObservationSeries`].
    ///
    /// # Errors
    /// Returns [`InnovationError`] on length mismatch, non-finite values, negative counts,
    /// shares outside `[0, 1]`, or incompatible units/denominators.
    pub fn try_new(
        grid: TimeGrid,
        values: Vec<Option<f64>>,
        kind: ObservationKind,
        units: Units,
        denominator: Option<Denominator>,
    ) -> Result<Self, InnovationError> {
        if values.len() != grid.len() {
            return Err(InnovationError::new(
                "length_mismatch",
                "values length must match time grid length",
            ));
        }
        if let Some(denom) = denominator {
            denom.validate()?;
        }
        match (kind, units) {
            (ObservationKind::Share, Units::Proportion)
            | (
                ObservationKind::NewAdoption
                | ObservationKind::ActiveStock
                | ObservationKind::Transactions,
                Units::Count | Units::Currency,
            )
            | (ObservationKind::Rate, Units::RatePerUnitTime | Units::Proportion)
            | (ObservationKind::Attention, Units::Index | Units::Count | Units::Proportion) => {}
            _ => {
                return Err(InnovationError::new(
                    "incompatible_units",
                    "observation kind and units combination is incompatible",
                ));
            }
        }
        if matches!(
            kind,
            ObservationKind::NewAdoption
                | ObservationKind::ActiveStock
                | ObservationKind::Transactions
        ) && denominator.is_some()
        {
            return Err(InnovationError::new(
                "incompatible_denominator",
                "raw count series cannot carry a share denominator",
            ));
        }
        for val in values.iter().flatten() {
            if !val.is_finite() {
                return Err(InnovationError::new(
                    "non_finite_observation",
                    "observation values must be finite when present",
                ));
            }
            match kind {
                ObservationKind::NewAdoption
                | ObservationKind::ActiveStock
                | ObservationKind::Transactions
                    if *val < 0.0 =>
                {
                    return Err(InnovationError::new(
                        "negative_count",
                        "count observations cannot be negative",
                    ));
                }
                ObservationKind::Share if *val < 0.0 || *val > 1.0 => {
                    return Err(InnovationError::new(
                        "invalid_share",
                        "share observations must lie in [0, 1]",
                    ));
                }
                _ => {}
            }
        }
        Ok(Self {
            grid,
            values,
            kind,
            units,
            denominator,
        })
    }

    /// Borrow the validated time grid associated with the observations.
    #[must_use]
    pub fn grid(&self) -> &TimeGrid {
        &self.grid
    }

    /// Borrow the observations, preserving missing entries as `None`.
    #[must_use]
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }

    /// Return the declared semantic kind of the observations.
    #[must_use]
    pub fn kind(&self) -> ObservationKind {
        self.kind
    }

    /// Return the declared measurement units.
    #[must_use]
    pub fn units(&self) -> Units {
        self.units
    }

    /// Return the optional validated denominator.
    #[must_use]
    pub fn denominator(&self) -> Option<Denominator> {
        self.denominator
    }
}

/// Evaluate cumulative Bass diffusion $F(t) = m \frac{1 - e^{-(p+q)t}}{1 + (q/p)e^{-(p+q)t}}$ (R01).
///
/// # Errors
/// Returns [`InnovationError`] if `t < 0`, `p <= 0`, `q < 0`, `m <= 0`, or any input is non-finite.
pub fn bass_cumulative(t: f64, p: f64, q: f64, m: f64) -> Result<f64, InnovationError> {
    if !t.is_finite() || t < 0.0 {
        return Err(InnovationError::new(
            "invalid_time",
            "t must be finite and nonnegative",
        ));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(InnovationError::new(
            "invalid_parameter",
            "p must be finite and strictly positive",
        ));
    }
    if !q.is_finite() || q < 0.0 {
        return Err(InnovationError::new(
            "invalid_parameter",
            "q must be finite and nonnegative",
        ));
    }
    if !m.is_finite() || m <= 0.0 {
        return Err(InnovationError::new(
            "invalid_parameter",
            "m must be finite and strictly positive",
        ));
    }
    if t == 0.0 {
        return Ok(0.0);
    }
    let rate = p + q;
    let exp_term = (-rate * t).exp();
    let value = m * (1.0 - exp_term) / (1.0 + (q / p) * exp_term);
    Ok(value.clamp(0.0, m))
}
