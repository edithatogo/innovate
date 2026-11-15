//! Small deterministic numerical routines. No language runtime or scheduler dependency.
use std::cell::RefCell;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{ColPivQR, DMatrix, DVector, Dyn, storage::Owned};
use serde_json::{Value, json};

pub(crate) struct LeastSquares {
    pub parameters: Vec<f64>,
    pub report: Value,
    pub identifiability: Value,
}

struct Candidate {
    error: f64,
    parameters: Vec<f64>,
    predictions: Vec<f64>,
    iterations: usize,
    converged: bool,
    reason: &'static str,
}

fn jacobian<F>(theta: &[f64], prediction: &[f64], evaluate: &F) -> Result<Vec<Vec<f64>>, String>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, String>,
{
    let mut columns = Vec::with_capacity(theta.len());
    for j in 0..theta.len() {
        let step = 1e-5 * theta[j].abs().max(1.0);
        let mut plus = theta.to_vec();
        let mut minus = theta.to_vec();
        plus[j] += step;
        minus[j] -= step;
        let plus = evaluate(&plus);
        let minus = evaluate(&minus);
        // Near a domain boundary, differentiate from the feasible side.
        let (a, b, width) = match (&plus, &minus) {
            (Ok(a), Ok(b)) => (a.as_slice(), b.as_slice(), 2. * step),
            (Ok(a), Err(_)) => (a.as_slice(), prediction, step),
            (Err(_), Ok(b)) => (prediction, b.as_slice(), step),
            (Err(error), Err(_)) => return Err(error.clone()),
        };
        if a.len() != prediction.len()
            || b.len() != prediction.len()
            || a.iter().chain(b).any(|x| !x.is_finite())
        {
            return Err("Jacobian shape or finite-value violation".into());
        }
        let column: Vec<f64> = a.iter().zip(b).map(|(a, b)| (a - b) / width).collect();
        if column.iter().any(|v| !v.is_finite()) {
            return Err("Jacobian exceeded the finite numerical domain".into());
        }
        columns.push(column);
    }
    Ok(columns)
}

#[cfg(test)]
fn solve(a: Vec<Vec<f64>>, b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let matrix = DMatrix::from_fn(n, n, |i, j| a[i][j]);
    let rhs = DVector::from_column_slice(&b);
    if matrix.iter().chain(rhs.iter()).any(|x| !x.is_finite()) {
        return None;
    }
    let lu = matrix.lu();
    if lu
        .u()
        .diagonal()
        .iter()
        .any(|d| !d.is_finite() || d.abs() < 1e-20)
    {
        return None;
    }
    let sol = lu.solve(&rhs)?;
    sol.iter()
        .all(|x| x.is_finite())
        .then(|| sol.as_slice().to_vec())
}

fn local_rank(columns: Vec<Vec<f64>>) -> Value {
    const MIN_COLUMN_NORM: f64 = 1e-14;
    let n = columns.len();
    let nrows = columns.first().map_or(0, Vec::len);
    let mut norms = Vec::with_capacity(n);
    let mut normalized = DMatrix::<f64>::zeros(nrows, n);

    for (j, column) in columns.iter().enumerate() {
        let scale = column.iter().map(|x| x.abs()).fold(0., f64::max);
        if scale == 0. {
            norms.push(0.);
            continue;
        }
        let scaled_norm = column
            .iter()
            .map(|&x| {
                let s = x / scale;
                s * s
            })
            .sum::<f64>()
            .sqrt();
        let magnitude = scale * scaled_norm;
        norms.push(magnitude);
        if magnitude > MIN_COLUMN_NORM {
            let denom = scale * scaled_norm;
            for (i, &x) in column.iter().enumerate() {
                // Divide via scale first to avoid intermediate underflow/overflow when scale ~ 1e308.
                normalized[(i, j)] = (x / scale) / scaled_norm;
            }
            let _ = denom;
        }
    }

    let diagonals: Vec<f64> = if nrows > 0 && n > 0 {
        let qr = ColPivQR::new(normalized);
        let r = qr.r();
        let k = nrows.min(n);
        (0..k)
            .map(|i| r[(i, i)].abs())
            .take_while(|&d| d >= 1e-7)
            .collect()
    } else {
        Vec::new()
    };

    let rank = diagonals.len();
    let excluded: Vec<_> = norms
        .iter()
        .enumerate()
        .filter_map(|(index, &norm)| (norm <= MIN_COLUMN_NORM).then_some(index))
        .collect();
    json!({"rank":rank,"parameters":n,"status":if rank==n {"local_full_rank"} else {"rank_deficient"},
        "method":"column_scaled_pivoted_qr", "relative_rank_tolerance":1e-7,
        "qr_diagonal_ratio":if rank==n && n>0 {Some(diagonals[0]/diagonals[n-1])} else {None},
        "sensitivity_column_norms":norms,
        "sensitivity_column_norm_cutoff":MIN_COLUMN_NORM,
        "excluded_sensitivity_column_indices":excluded,
        "scope":"local transformed-parameter sensitivity; not global uniqueness or calibrated uncertainty"})
}

struct LeastSquaresProblemAdapter<'a, F> {
    observed: &'a [f64],
    evaluate: &'a F,
    params: DVector<f64>,
    max_iterations: usize,
    iterations: RefCell<usize>,
    iteration_limit_reached: RefCell<bool>,
    domain_error: RefCell<Option<String>>,
}

impl<'a, F> LeastSquaresProblem<f64, Dyn, Dyn> for LeastSquaresProblemAdapter<'a, F>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, String>,
{
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params = x.clone();
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        if self.params.iter().any(|v| !v.is_finite() || v.abs() > 35.0) {
            return Some(DVector::from_element(self.observed.len(), 1e75));
        }
        match (self.evaluate)(self.params.as_slice()) {
            Ok(prediction)
                if prediction.len() == self.observed.len()
                    && prediction.iter().all(|x| x.is_finite()) =>
            {
                Some(DVector::from_iterator(
                    self.observed.len(),
                    prediction
                        .into_iter()
                        .zip(self.observed)
                        .map(|(p, y)| p - y),
                ))
            }
            // Treat infeasible or non-finite trial steps as high-cost rejections so LM shrinks the trust region.
            _ => Some(DVector::from_element(self.observed.len(), 1e75)),
        }
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let next = *self.iterations.borrow() + 1;
        if next > self.max_iterations {
            *self.iteration_limit_reached.borrow_mut() = true;
            return None;
        }
        *self.iterations.borrow_mut() = next;
        let prediction = (self.evaluate)(self.params.as_slice()).ok()?;
        let columns = match jacobian(self.params.as_slice(), &prediction, self.evaluate) {
            Ok(columns) => columns,
            Err(err) => {
                *self.domain_error.borrow_mut() = Some(err);
                return None;
            }
        };
        let m = prediction.len();
        let n = self.params.len();
        let matrix = DMatrix::from_fn(m, n, |i, j| columns[j][i]);
        let normal = matrix.transpose() * &matrix;
        if normal.iter().any(|x| !x.is_finite()) {
            *self.domain_error.borrow_mut() =
                Some("optimizer exceeded the finite numerical domain".into());
            return None;
        }
        Some(matrix)
    }
}

/// Damped nonlinear least squares in caller-supplied transformed parameters.
pub(crate) fn least_squares<F>(
    observed: &[f64],
    starts: &[Vec<f64>],
    evaluate: F,
    max_iterations: usize,
) -> Result<LeastSquares, String>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, String>,
{
    let loss = |prediction: &[f64]| {
        if prediction.len() == observed.len() && prediction.iter().all(|x| x.is_finite()) {
            prediction
                .iter()
                .zip(observed)
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
        } else {
            f64::INFINITY
        }
    };
    let mut best: Option<(usize, Candidate)> = None;
    let mut best_converged: Option<(usize, Candidate)> = None;
    let mut start_reports = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().enumerate() {
        let initial_prediction = match evaluate(start) {
            Ok(prediction) => prediction,
            Err(error) => {
                start_reports.push(json!({"termination":"invalid_initial_prediction",
                    "initial_transformed_parameters":start,"error":error}));
                continue;
            }
        };
        let initial_error = loss(&initial_prediction);
        if !initial_error.is_finite() {
            start_reports.push(json!({"termination":"nonfinite_initial_loss","initial_transformed_parameters":start}));
            continue;
        }
        let (theta, prediction, error, iterations, converged, reason) = if max_iterations == 0 {
            (
                start.clone(),
                initial_prediction,
                initial_error,
                0,
                false,
                "iteration_limit",
            )
        } else {
            let problem = LeastSquaresProblemAdapter {
                observed,
                evaluate: &evaluate,
                params: DVector::from_column_slice(start),
                max_iterations,
                iterations: RefCell::new(0),
                iteration_limit_reached: RefCell::new(false),
                domain_error: RefCell::new(None),
            };
            let (solved, lm_report) = LevenbergMarquardt::new()
                .with_ftol(1e-12)
                .with_xtol(1e-12)
                .with_gtol(1e-9)
                .with_stepbound(3.0)
                .with_patience(max_iterations)
                .minimize(problem);
            if let Some(err) = solved.domain_error.into_inner() {
                return Err(err);
            }
            let iteration_limit_reached = solved.iteration_limit_reached.into_inner();
            let iterations = solved.iterations.into_inner();
            let candidate_theta = solved.params.as_slice().to_vec();
            let candidate_pred = evaluate(&candidate_theta).ok();
            let candidate_err = candidate_pred.as_deref().map_or(f64::INFINITY, loss);
            let (theta, prediction, error) = if candidate_err <= initial_error {
                (
                    candidate_theta,
                    candidate_pred.unwrap_or(initial_prediction),
                    candidate_err,
                )
            } else {
                (start.clone(), initial_prediction, initial_error)
            };
            let (converged, reason) = if iteration_limit_reached {
                (false, "iteration_limit")
            } else {
                match lm_report.termination {
                    TerminationReason::ResidualsZero | TerminationReason::Orthogonal => {
                        (true, "gradient_or_residual_tolerance")
                    }
                    TerminationReason::Converged { .. } if max_iterations > 1 => {
                        (true, "objective_tolerance")
                    }
                    TerminationReason::Converged { .. }
                    | TerminationReason::NoImprovementPossible(_)
                    | TerminationReason::LostPatience => (false, "iteration_limit"),
                    TerminationReason::User(_)
                    | TerminationReason::NoParameters
                    | TerminationReason::NoResiduals
                    | TerminationReason::WrongDimensions(_) => (false, "damping_limit"),
                    TerminationReason::Numerical(_) => {
                        return Err("optimizer exceeded the finite numerical domain".into());
                    }
                }
            };
            (theta, prediction, error, iterations, converged, reason)
        };
        start_reports.push(json!({
            "iterations": iterations,
            "converged": converged,
            "termination": reason,
            "initial_scaled_rss": initial_error,
            "scaled_rss": error,
            "initial_transformed_parameters": start,
            "final_transformed_parameters": theta
        }));
        let slot = if converged {
            &mut best_converged
        } else {
            &mut best
        };
        if slot
            .as_ref()
            .is_none_or(|(_, candidate)| error < candidate.error)
        {
            *slot = Some((
                index,
                Candidate {
                    error,
                    parameters: theta,
                    predictions: prediction,
                    iterations,
                    converged,
                    reason,
                },
            ));
        }
    }
    // A difference at floating-point resolution is insufficient to prefer
    // a stalled start over one that met the optimizer's termination criteria.
    let best = match (best, best_converged) {
        (Some(stalled), Some(converged)) => {
            let roundoff = 8. * f64::EPSILON * stalled.1.error.max(converged.1.error);
            Some(if converged.1.error - stalled.1.error <= roundoff {
                converged
            } else {
                stalled
            })
        }
        (stalled, converged) => stalled.or(converged),
    };
    let (selected_start, best) = best.ok_or("no finite optimizer candidate")?;
    let Candidate {
        error,
        parameters,
        predictions,
        iterations,
        converged,
        reason,
    } = best;
    let columns = jacobian(&parameters, &predictions, &evaluate)?;
    let gradient = columns
        .iter()
        .map(|c| {
            c.iter()
                .zip(predictions.iter().zip(observed))
                .map(|(j, (p, y))| j * (p - y))
                .sum::<f64>()
                .abs()
        })
        .try_fold(0.0_f64, |maximum, component| {
            component.is_finite().then(|| maximum.max(component))
        });
    let converged_starts = start_reports
        .iter()
        .filter(|report| report["converged"] == true)
        .count();
    let alternative_loss = start_reports
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != selected_start)
        .filter_map(|(_, report)| report["scaled_rss"].as_f64())
        .min_by(f64::total_cmp);
    let identifiability = local_rank(columns);
    Ok(LeastSquares {
        parameters,
        identifiability,
        report: json!({"method":"deterministic_multistart_damped_least_squares",
        "converged":converged,"termination":reason,"iterations":iterations,"max_iterations":max_iterations,
        "starts":starts.len(),"converged_starts":converged_starts,"alternative_start_scaled_rss_gap":alternative_loss.map(|loss| loss - error),"start_reports":start_reports,"selected_start_index":selected_start,"scaled_rss":error,"gradient_max_abs":gradient,
        "interpretation":"numerical termination only; not a global optimum or statistical acceptance"}),
    })
}

struct ClosureSystem<'a, F>(&'a F);

impl<F> ode_solvers::System<f64, ode_solvers::DVector<f64>> for ClosureSystem<'_, F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    fn system(&self, x: f64, y: &ode_solvers::DVector<f64>, dy: &mut ode_solvers::DVector<f64>) {
        let deriv = (self.0)(x, y.as_slice());
        for (dst, src) in dy.iter_mut().zip(deriv) {
            *dst = src;
        }
    }
}

/// Adaptive Dormand-Prince 5(4) IVP solver via `ode_solvers::Dopri5`.
fn rk4_single_step<F>(f: &F, t: f64, y: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let k1 = f(t, y);
    let half_h = 0.5 * h;
    let y2: Vec<f64> = y
        .iter()
        .zip(&k1)
        .map(|(yi, k1i)| yi + half_h * k1i)
        .collect();
    let k2 = f(t + half_h, &y2);
    let y3: Vec<f64> = y
        .iter()
        .zip(&k2)
        .map(|(yi, k2i)| yi + half_h * k2i)
        .collect();
    let k3 = f(t + half_h, &y3);
    let y4: Vec<f64> = y.iter().zip(&k3).map(|(yi, k3i)| yi + h * k3i).collect();
    let k4 = f(t + h, &y4);
    y.iter()
        .enumerate()
        .map(|(i, yi)| yi + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Integrate from an explicit origin, never from the first requested sample.
pub(crate) fn integrate<F>(
    times: &[f64],
    origin: f64,
    initial: &[f64],
    f: F,
) -> Result<Vec<Vec<f64>>, String>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    use ode_solvers::{DVector, Dopri5, OutputType};

    if times.is_empty() {
        return Ok(Vec::new());
    }
    if initial.iter().any(|x| !x.is_finite()) || !origin.is_finite() {
        return Err("ODE left the finite numerical domain".into());
    }
    if initial.is_empty() {
        for &target in times {
            if !target.is_finite() {
                return Err("ODE left the finite numerical domain".into());
            }
            if target < origin {
                return Err("observation time precedes initial_time".into());
            }
        }
        return Ok(vec![Vec::new(); times.len()]);
    }
    let mut t_prev = origin;
    let mut state = DVector::from_column_slice(initial);
    let mut rows = Vec::with_capacity(times.len());
    for &target in times {
        if !target.is_finite() {
            return Err("ODE left the finite numerical domain".into());
        }
        if target < t_prev {
            return Err("observation time precedes initial_time".into());
        }
        if target > t_prev {
            let span = target - t_prev;
            let tiny_tol = 64.0 * f64::EPSILON * target.abs().max(t_prev.abs()).max(1.0);
            if span <= tiny_tol {
                let next = rk4_single_step(&f, t_prev, state.as_slice(), span);
                if next.iter().any(|x| !x.is_finite()) {
                    return Err("ODE left the finite numerical domain".into());
                }
                state = DVector::from_vec(next);
                t_prev = target;
            } else {
                let mut solver = Dopri5::from_param(
                    ClosureSystem(&f),
                    t_prev,
                    target,
                    0.0,
                    state.clone(),
                    1e-8,
                    1e-8,
                    0.9,
                    0.04,
                    0.2,
                    10.0,
                    span,
                    0.0,
                    100000,
                    1000,
                    OutputType::Sparse,
                );
                solver
                    .integrate()
                    .map_err(|err| format!("ODE integration failed: {err}"))?;
                let next = solver
                    .y_out()
                    .last()
                    .ok_or_else(|| "ODE solver produced no final state".to_string())?;
                if next.iter().any(|x| !x.is_finite()) {
                    return Err("ODE left the finite numerical domain".into());
                }
                state = next.clone();
                t_prev = target;
            }
        }
        rows.push(state.as_slice().to_vec());
    }
    Ok(rows)
}

pub(crate) fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let rank = probability * (sorted.len() - 1) as f64;
    sorted[rank.floor() as usize] * (1. - rank.fract())
        + sorted[rank.ceil() as usize] * rank.fract()
}

#[cfg(test)]
#[test]
fn converged_start_wins_only_within_objective_roundoff() {
    for (offset, expected_convergence) in [(f64::EPSILON, true), (1e-6, false)] {
        let evaluate = |theta: &[f64]| {
            let x = theta[0];
            if x.abs() < 0.1 {
                Ok(vec![1. + offset])
            } else if (x - 1.).abs() < 0.1 {
                Ok(vec![x])
            } else {
                Err("outside the test domains".into())
            }
        };
        for starts in [[vec![0.], vec![1.]], [vec![1.], vec![0.]]] {
            let fit = least_squares(&[0.], &starts, evaluate, 1).unwrap();
            assert_eq!(fit.report["converged"], expected_convergence);
            assert_eq!(
                fit.parameters,
                vec![if expected_convergence { 0. } else { 1. }]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_rank_normalizes_large_finite_sensitivities() {
        let report = local_rank(vec![vec![1e308, 1e308], vec![1e308, -1e308]]);
        assert_eq!(report["rank"], 2);
        assert_eq!(report["status"], "local_full_rank");
        assert!(
            report["sensitivity_column_norms"][0]
                .as_f64()
                .unwrap()
                .is_finite()
        );
        let overflow = local_rank(vec![vec![f64::MAX, f64::MAX]]);
        assert_eq!(overflow["rank"], 1);
        assert!(overflow["sensitivity_column_norms"][0].is_null());
    }

    #[test]
    fn local_rank_distinguishes_collinear_columns_at_extreme_scales() {
        let report = local_rank(vec![
            vec![1e308, -1e308, 0.],
            vec![1., -1., 0.],
            vec![0., 0., 1e-20],
        ]);
        assert_eq!(report["rank"], 1);
        assert_eq!(report["status"], "rank_deficient");
        assert!(report["qr_diagonal_ratio"].is_null());
        assert_eq!(report["sensitivity_column_norms"][2], 1e-20);
        assert_eq!(report["sensitivity_column_norm_cutoff"], 1e-14);
        assert_eq!(report["excluded_sensitivity_column_indices"], json!([2]));
    }

    #[test]
    fn multistart_report_distinguishes_equal_objectives() {
        let fit =
            least_squares(&[1.], &[vec![-1.], vec![1.]], |p| Ok(vec![p[0].powi(2)]), 4).unwrap();
        assert_eq!(fit.parameters, [-1.]);
        assert_eq!(fit.report["converged_starts"], 2);
        assert_eq!(fit.report["alternative_start_scaled_rss_gap"], 0.);
        assert_eq!(
            fit.report["start_reports"][0]["final_transformed_parameters"],
            json!([-1.])
        );
        assert_eq!(
            fit.report["start_reports"][1]["final_transformed_parameters"],
            json!([1.])
        );
    }

    #[test]
    fn multistart_report_retains_initial_objective() {
        let fit = least_squares(&[2.], &[vec![0.], vec![2.]], |p| Ok(vec![p[0]]), 12).unwrap();
        let reports = &fit.report["start_reports"];
        assert_eq!(reports[0]["initial_scaled_rss"], 4.);
        assert!(reports[0]["scaled_rss"].as_f64().unwrap() < 1e-16);
        assert_eq!(reports[1]["initial_scaled_rss"], 0.);
        assert_eq!(reports[1]["scaled_rss"], 0.);
    }

    #[test]
    fn multistart_report_retains_rejected_start_coordinates() {
        let fit = least_squares(
            &[1.],
            &[vec![-2.], vec![1.]],
            |p| {
                if p[0] < 0. {
                    Err("negative start".into())
                } else {
                    Ok(vec![p[0]])
                }
            },
            4,
        )
        .unwrap();
        let reports = &fit.report["start_reports"];
        assert_eq!(fit.report["selected_start_index"], 1);
        assert_eq!(reports[0]["termination"], "invalid_initial_prediction");
        assert_eq!(reports[0]["error"], "negative start");
        assert_eq!(reports[0]["initial_transformed_parameters"], json!([-2.]));
        assert_eq!(reports[1]["initial_transformed_parameters"], json!([1.]));
    }

    #[test]
    fn nonfinite_final_gradient_is_unavailable() {
        let fit = least_squares(
            &[-2., -2.],
            &[vec![0.]],
            |p| Ok(vec![p[0] * 1e308, -p[0] * 1e308]),
            0,
        )
        .unwrap();
        assert!(fit.report["gradient_max_abs"].is_null());
        assert_eq!(fit.report["converged"], false);
    }

    #[test]
    fn unrepresentable_sensitivities_never_report_convergence() {
        let error = least_squares(
            &[0.],
            &[vec![0.]],
            |p| {
                Ok(vec![if p[0] == 0. {
                    0.
                } else {
                    p[0].signum() * 1e308
                }])
            },
            10,
        )
        .err()
        .unwrap();
        assert!(error.contains("finite numerical domain"));
    }

    #[test]
    fn solve_handles_well_conditioned_and_singular_systems() {
        let x = solve(
            vec![
                vec![2.0, 1.0, -1.0],
                vec![-3.0, -1.0, 2.0],
                vec![-2.0, 1.0, 2.0],
            ],
            vec![8.0, -11.0, -3.0],
        )
        .expect("well-conditioned 3x3 system must solve");
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - (-1.0)).abs() < 1e-12);

        let singular = solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![3.0, 6.0]);
        assert!(singular.is_none());
    }

    #[test]
    fn local_rank_reports_full_rank_and_near_collinearity() {
        let full = local_rank(vec![vec![3.0, 0.0], vec![0.0, 4.0]]);
        assert_eq!(full["rank"], 2);
        assert_eq!(full["status"], "local_full_rank");
        assert_eq!(full["qr_diagonal_ratio"], 1.0);

        let collinear = local_rank(vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-9]]);
        assert_eq!(collinear["rank"], 1);
        assert_eq!(collinear["status"], "rank_deficient");
        assert!(collinear["qr_diagonal_ratio"].is_null());
    }

    #[test]
    fn least_squares_recovers_nonlinear_exponential_fit() {
        let times = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let observed: Vec<f64> = times.iter().map(|&t| 2.5 * (-0.4 * t).exp()).collect();
        let fit = least_squares(
            &observed,
            &[vec![1.0, -0.1], vec![4.0, -1.0]],
            |theta| {
                Ok(times
                    .iter()
                    .map(|&t| theta[0] * (theta[1] * t).exp())
                    .collect())
            },
            50,
        )
        .expect("nonlinear exponential fit should succeed");
        assert_eq!(fit.report["converged"], true);
        assert!((fit.parameters[0] - 2.5).abs() < 1e-6);
        assert!((fit.parameters[1] - (-0.4)).abs() < 1e-6);
        assert!(fit.report["scaled_rss"].as_f64().unwrap() < 1e-12);
        assert_eq!(fit.identifiability["status"], "local_full_rank");
    }

    #[test]
    fn least_squares_enforces_iteration_budget() {
        let times = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let observed: Vec<f64> = times.iter().map(|&t| 2.5 * (-0.4 * t).exp()).collect();
        let fit = least_squares(
            &observed,
            &[vec![0.5, -1.5]],
            |theta| {
                Ok(times
                    .iter()
                    .map(|&t| theta[0] * (theta[1] * t).exp())
                    .collect())
            },
            1,
        )
        .expect("single-iteration fit should complete without error");
        let start = &fit.report;
        assert!(start["iterations"].as_u64().unwrap() <= 1);
        assert_eq!(start["converged"], false);
        assert_eq!(start["termination"], "iteration_limit");
    }

    #[test]
    fn least_squares_rejects_out_of_bounds_steps_consistently() {
        let observed = [100.0_f64, 100.0, 100.0];
        let fit = least_squares(
            &observed,
            &[vec![34.5, 34.5]],
            |theta| Ok(vec![theta[0], theta[1], theta[0] + theta[1]]),
            20,
        )
        .expect("bounded fit should complete without desynchronizing");
        assert!(fit.parameters.iter().all(|v| v.abs() <= 35.0));
        assert!(fit.report["scaled_rss"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn dopri5_matches_coupled_linear_analytical_solution_on_irregular_grid() {
        let times = [0.15, 0.37, 1.25, 3.8, 7.5];
        let rows = integrate(&times, 0.0, &[1.0, 0.0], |_t, y| {
            vec![-0.7 * y[0], 0.7 * y[0] - 0.3 * y[1]]
        })
        .expect("irregular grid integration should succeed");
        assert_eq!(rows.len(), times.len());
        for (&t, row) in times.iter().zip(&rows) {
            let exact_y0 = (-0.7 * t).exp();
            let exact_y1 = (0.7 / (0.7 - 0.3)) * ((-0.3 * t).exp() - (-0.7 * t).exp());
            assert!(
                (row[0] - exact_y0).abs() < 1e-7,
                "y0 mismatch at t={t}: got {}, expected {exact_y0}",
                row[0]
            );
            assert!(
                (row[1] - exact_y1).abs() < 1e-7,
                "y1 mismatch at t={t}: got {}, expected {exact_y1}",
                row[1]
            );
        }

        let err_before_origin = integrate(&[-0.1], 0.0, &[1.0], |_t, y| vec![-y[0]])
            .expect_err("times before origin must error");
        assert!(err_before_origin.contains("precede"));

        let err_non_finite = integrate(&[1.0], 0.0, &[f64::NAN], |_t, y| vec![-y[0]])
            .expect_err("non-finite initial state must error");
        assert!(err_non_finite.contains("finite"));
    }

    #[test]
    fn dopri5_handles_zero_length_origin_step_and_extreme_rates() {
        let times = [0.0, 1e-4, 0.05, 0.5];
        let k = 250.0;
        let rows = integrate(&times, 0.0, &[2.0], |_t, y| vec![-k * y[0]])
            .expect("extreme-rate integration should succeed");
        assert_eq!(rows.len(), times.len());
        assert_eq!(rows[0], vec![2.0]);
        for (&t, row) in times.iter().zip(&rows) {
            let exact = 2.0 * (-k * t).exp();
            assert!(
                (row[0] - exact).abs() < 1e-7,
                "stiff decay mismatch at t={t}: got {}, expected {exact}",
                row[0]
            );
        }
    }
}
