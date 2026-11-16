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
    /// Returns [`InnovationError`] if `rates` is empty, non-square, or has negative/non-finite off-diagonal hazards.
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
    /// Guarantees non-negative entries, exact row-sum conservation, and semigroup composition.
    ///
    /// # Errors
    /// Returns [`InnovationError`] if `dt < 0` or `dt` is non-finite.
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
        let alpha = lambda * dt;
        // Uniformized discrete-time transition matrix R = I + Q / lambda.
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
        // Renormalize tiny truncation remainder (<1e-15) to preserve exact row sum 1.0.
        for row in &mut p_mat {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for v in row {
                    *v /= sum;
                }
            }
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

/// Bi-exponential excess-risk / frailty decay kernel $w(\tau) = \alpha e^{-\lambda_1 \tau} + (1 - \alpha) e^{-\lambda_2 \tau}$ (SM02).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiExponentialKernel {
    pub alpha: f64,
    pub lambda1: f64,
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
    pub max_tunnel: usize,
    pub baseline_relapse: f64,
    pub frailty_amplitude: f64,
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
    /// Returns [`InnovationError`] on length mismatch, negative occupancy/entry, or hazard > 1.
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
