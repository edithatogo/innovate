//! Continuous-time competing-risks generators, uniformization, and semi-Markov tunnel states (SM01/SM02).

use crate::inference::InnovationError;

/// Continuous-time competing-risks generator matrix $Q$ with zero row sums (SM01).
#[derive(Debug, Clone, PartialEq)]
pub struct CompetingRisksGenerator {
    q: Vec<Vec<f64>>,
    max_exit_rate: f64,
}

impl CompetingRisksGenerator {
    /// Construct a validated generator $Q$ from an $S \times S$ off-diagonal hazard matrix.
    ///
    /// Diagonal entries are overwritten with $-\sum_{j \ne i} q_{ij}$ so rows sum to zero.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `rates` is empty, non-square, has
    /// negative/non-finite off-diagonal hazards, or an exit-rate sum overflows.
    pub fn new(mut rates: Vec<Vec<f64>>) -> Result<Self, InnovationError> {
        let s = rates.len();
        if s == 0 || rates.iter().any(|row| row.len() != s) {
            return Err(InnovationError::new(
                "invalid_generator",
                "generator rates must be nonempty and square",
            ));
        }
        let mut max_exit_rate = 0.0_f64;
        for (i, row) in rates.iter_mut().enumerate() {
            let mut off_sum = 0.0_f64;
            for (j, val) in row.iter().enumerate() {
                if i != j {
                    if !val.is_finite() || *val < 0.0 {
                        return Err(InnovationError::new(
                            "invalid_hazard",
                            "off-diagonal hazards must be finite and nonnegative",
                        ));
                    }
                    off_sum += *val;
                    if !off_sum.is_finite() {
                        return Err(InnovationError::new(
                            "invalid_hazard",
                            "sum of off-diagonal hazards must be finite",
                        ));
                    }
                }
            }
            row[i] = -off_sum;
            max_exit_rate = max_exit_rate.max(off_sum);
        }
        Ok(Self {
            q: rates,
            max_exit_rate,
        })
    }

    /// Borrow the $S \times S$ generator matrix $Q$.
    #[must_use]
    pub fn matrix(&self) -> &[Vec<f64>] {
        &self.q
    }

    /// Compute $P(\Delta t) = \exp(Q\,\Delta t)$ via uniformization (SM01).
    ///
    /// Produces non-negative entries and conserves row sums to floating-point accuracy.
    /// Semigroup composition is subject to rounding and Poisson-series truncation.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `dt < 0`, `dt` is non-finite, or numerical
    /// normalization fails.
    pub fn uniformization_step(&self, dt: f64) -> Result<Vec<Vec<f64>>, InnovationError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(InnovationError::new(
                "invalid_timestep",
                "dt must be finite and nonnegative",
            ));
        }
        let s = self.q.len();
        let mut identity = vec![vec![0.0; s]; s];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        if dt == 0.0 || self.max_exit_rate == 0.0 {
            return Ok(identity);
        }

        let lambda = self.max_exit_rate;
        // Halve the duration until the Poisson series is safely representable.
        // Unlike a float-to-usize step count, this also handles an overflowing
        // product of two otherwise finite inputs.
        let mut sub_dt = dt;
        let mut squarings = 0;
        while lambda * sub_dt > 20.0 {
            sub_dt *= 0.5;
            squarings += 1;
        }
        let alpha = lambda * sub_dt;
        let mut r_mat = vec![vec![0.0; s]; s];
        for i in 0..s {
            for j in 0..s {
                r_mat[i][j] = identity[i][j] + self.q[i][j] / lambda;
            }
        }

        let mut p_mat = vec![vec![0.0; s]; s];
        let mut r_pow = identity;
        let mut pois_weight = (-alpha).exp();
        let mut total_weight = 0.0_f64;

        let max_terms = ((alpha + 12.0 * alpha.sqrt() + 40.0).ceil() as usize).max(64);
        for k in 0..max_terms {
            for i in 0..s {
                for j in 0..s {
                    p_mat[i][j] += pois_weight * r_pow[i][j];
                }
            }
            total_weight += pois_weight;
            if 1.0 - total_weight < 1e-15 {
                break;
            }
            pois_weight *= alpha / ((k + 1) as f64);
            r_pow = mat_mul(&r_pow, &r_mat);
        }
        normalize_rows(&mut p_mat)?;
        // Normalize after each square so tiny rounding errors in row sums do
        // not compound exponentially over very large rate-time products.
        for _ in 0..squarings {
            p_mat = mat_mul(&p_mat, &p_mat);
            normalize_rows(&mut p_mat)?;
        }
        Ok(p_mat)
    }
}

/// Multiply two square matrices of identical dimension.
#[must_use]
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let s = a.len();
    let mut out = vec![vec![0.0; s]; s];
    for i in 0..s {
        for k in 0..s {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..s {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn normalize_rows(matrix: &mut [Vec<f64>]) -> Result<(), InnovationError> {
    for row in matrix {
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || sum <= 0.0 || row.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(InnovationError::new(
                "numerical_overflow",
                "transition probabilities could not be normalized",
            ));
        }
        for value in row {
            *value /= sum;
        }
    }
    Ok(())
}

/// Bi-exponential excess-risk / frailty decay kernel $w(\tau) = \alpha e^{-\lambda_1 \tau} + (1 - \alpha) e^{-\lambda_2 \tau}$ (SM02).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiExponentialKernel {
    /// Mixture weight of the first exponential, between zero and one.
    pub alpha: f64,
    /// Nonnegative decay rate of the first exponential.
    pub lambda1: f64,
    /// Nonnegative decay rate of the second exponential.
    pub lambda2: f64,
}

impl BiExponentialKernel {
    /// Evaluate $w(\tau)$ at duration $\tau \ge 0$.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `alpha` is outside `[0, 1]` or rates/tau are negative/non-finite.
    pub fn weight(&self, tau: f64) -> Result<f64, InnovationError> {
        if !self.alpha.is_finite()
            || !(0.0..=1.0).contains(&self.alpha)
            || !self.lambda1.is_finite()
            || self.lambda1 < 0.0
            || !self.lambda2.is_finite()
            || self.lambda2 < 0.0
            || !tau.is_finite()
            || tau < 0.0
        {
            return Err(InnovationError::new(
                "invalid_kernel",
                "alpha in [0,1] and nonnegative finite rates/tau required",
            ));
        }
        Ok(self.alpha * (-self.lambda1 * tau).exp()
            + (1.0 - self.alpha) * (-self.lambda2 * tau).exp())
    }
}

/// Semi-Markov duration-since-transition tunnel-state model $\tau \in \{0, \dots, K\}$ (SM02).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TunnelStateModel {
    /// Maximum represented duration; the final tunnel retains longer durations.
    pub max_tunnel: usize,
    /// Nonnegative baseline relapse hazard.
    pub baseline_relapse: f64,
    /// Nonnegative excess hazard at duration zero.
    pub frailty_amplitude: f64,
    /// Nonnegative exponential decay rate of the excess hazard.
    pub decay_rate: f64,
}

impl TunnelStateModel {
    /// Relapse/transition hazard at tunnel duration `tau`; reduces to `baseline_relapse` when `frailty_amplitude == 0`.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if parameters are invalid or `tau > max_tunnel`.
    pub fn hazard_at_duration(&self, tau: usize) -> Result<f64, InnovationError> {
        if !self.baseline_relapse.is_finite()
            || self.baseline_relapse < 0.0
            || !self.frailty_amplitude.is_finite()
            || self.frailty_amplitude < 0.0
            || !self.decay_rate.is_finite()
            || self.decay_rate < 0.0
        {
            return Err(InnovationError::new(
                "invalid_tunnel_model",
                "tunnel model parameters must be finite and nonnegative",
            ));
        }
        if tau > self.max_tunnel {
            return Err(InnovationError::new(
                "invalid_tunnel_duration",
                "tau exceeds max_tunnel",
            ));
        }
        Ok(self.baseline_relapse + self.frailty_amplitude * (-self.decay_rate * tau as f64).exp())
    }

    /// Advance tunnel occupancy vector of length `max_tunnel + 1` by one discrete step (`dt = 1`),
    /// returning `(next_occupancy, total_relapsed)`.
    ///
    /// # Errors
    /// Returns [`InnovationError`] for invalid occupancy, entry flow, or model parameters.
    pub fn step_cohort(
        &self,
        occupancy: &[f64],
        entry_flow: f64,
    ) -> Result<(Vec<f64>, f64), InnovationError> {
        if occupancy.len() != self.max_tunnel + 1
            || !entry_flow.is_finite()
            || entry_flow < 0.0
            || occupancy.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(InnovationError::new(
                "invalid_tunnel_occupancy",
                "occupancy length must equal max_tunnel + 1 with nonnegative finite values",
            ));
        }
        let mut next = vec![0.0_f64; self.max_tunnel + 1];
        next[0] = entry_flow;
        let mut relapsed = 0.0_f64;
        for (tau, &count) in occupancy.iter().enumerate() {
            let p_relapse = 1.0 - (-self.hazard_at_duration(tau)?).exp();
            let out_flow = count * p_relapse;
            let stay = count - out_flow;
            relapsed += out_flow;
            let next_tau = (tau + 1).min(self.max_tunnel);
            next[next_tau] += stay;
        }
        Ok((next, relapsed))
    }
}
