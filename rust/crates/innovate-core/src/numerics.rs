//! Small deterministic numerical routines. No language runtime or scheduler dependency.
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

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if !a[pivot][k].is_finite() || a[pivot][k].abs() < 1e-20 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            let pivot_values = a[k].clone();
            for (value, pivot) in a[i].iter_mut().zip(pivot_values).skip(k) {
                *value -= factor * pivot;
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x.iter().all(|x| x.is_finite()).then_some(x)
}

fn local_rank(mut columns: Vec<Vec<f64>>) -> Value {
    const MIN_COLUMN_NORM: f64 = 1e-14;
    let n = columns.len();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut norms = Vec::with_capacity(n);
    for column in &mut columns {
        let scale = column.iter().map(|x| x.abs()).fold(0., f64::max);
        if scale == 0. {
            norms.push(0.);
            continue;
        }
        for value in column.iter_mut() {
            *value /= scale;
        }
        let scaled_norm = norm(column);
        let magnitude = scale * scaled_norm;
        norms.push(magnitude);
        if magnitude > MIN_COLUMN_NORM {
            for value in column {
                *value /= scaled_norm;
            }
        } else {
            column.fill(0.);
        }
    }
    let mut diagonals = Vec::new();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| norm(&columns[i]).total_cmp(&norm(&columns[j])))
            .unwrap();
        columns.swap(k, pivot);
        let d = norm(&columns[k]);
        if d < 1e-7 {
            break;
        }
        diagonals.push(d);
        let basis: Vec<_> = columns[k].iter().map(|x| x / d).collect();
        for c in columns.iter_mut().skip(k + 1) {
            // Reorthogonalise to reduce numerical loss of rank near collinearity.
            for _ in 0..2 {
                let dot = c.iter().zip(&basis).map(|(a, b)| a * b).sum::<f64>();
                for (x, q) in c.iter_mut().zip(&basis) {
                    *x -= dot * q;
                }
            }
        }
    }
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
    let mut best: Option<Candidate> = None;
    let mut start_reports = Vec::with_capacity(starts.len());
    let mut selected_start = 0;
    for (index, start) in starts.iter().enumerate() {
        let mut theta = start.clone();
        let Ok(mut prediction) = evaluate(&theta) else {
            start_reports.push(json!({"termination":"invalid_initial_prediction"}));
            continue;
        };
        let mut error = loss(&prediction);
        if !error.is_finite() {
            start_reports.push(json!({"termination":"nonfinite_initial_loss"}));
            continue;
        }
        let mut damping = 1e-3_f64;
        let mut converged = false;
        let mut reason = "iteration_limit";
        let mut iterations = 0;
        for iteration in 0..max_iterations {
            iterations = iteration + 1;
            let columns = jacobian(&theta, &prediction, &evaluate)?;
            let n = theta.len();
            let mut normal = vec![vec![0.0; n]; n];
            let mut rhs = vec![0.0; n];
            for a in 0..n {
                rhs[a] = -columns[a]
                    .iter()
                    .zip(prediction.iter().zip(observed))
                    .map(|(j, (p, y))| j * (p - y))
                    .sum::<f64>();
                for b in 0..n {
                    normal[a][b] = columns[a].iter().zip(&columns[b]).map(|(a, b)| a * b).sum();
                }
            }
            let gradient_norm = rhs.iter().copied().map(f64::abs).fold(0.0, f64::max);
            if rhs
                .iter()
                .chain(normal.iter().flatten())
                .any(|x| !x.is_finite())
            {
                return Err("optimizer exceeded the finite numerical domain".into());
            }
            if error < 1e-20 || gradient_norm < 1e-9 {
                converged = true;
                reason = "gradient_or_residual_tolerance";
                break;
            }
            for (i, row) in normal.iter_mut().enumerate() {
                row[i] += damping * row[i].max(1e-8);
            }
            let delta = solve(normal, rhs);
            let candidate = delta.map(|d| {
                theta
                    .iter()
                    .zip(d)
                    .map(|(t, d)| (t + d.clamp(-3.0, 3.0)).clamp(-35.0, 35.0))
                    .collect::<Vec<_>>()
            });
            let attempt = candidate.as_ref().and_then(|c| evaluate(c).ok());
            if let (Some(candidate), Some(candidate_prediction)) = (candidate, attempt) {
                let candidate_error = loss(&candidate_prediction);
                if candidate_error < error {
                    let improvement = error - candidate_error;
                    theta = candidate;
                    prediction = candidate_prediction;
                    error = candidate_error;
                    damping = (damping * 0.25).max(1e-12);
                    if improvement < 1e-12 * (error + improvement) {
                        converged = true;
                        reason = "objective_tolerance";
                        break;
                    }
                    continue;
                }
            }
            damping *= 10.0;
            if damping > 1e14 {
                reason = "damping_limit";
                break;
            }
        }
        start_reports.push(json!({
            "iterations": iterations,
            "converged": converged,
            "termination": reason,
            "scaled_rss": error
        }));
        if best.as_ref().is_none_or(|b| error < b.error) {
            selected_start = index;
            best = Some(Candidate {
                error,
                parameters: theta,
                predictions: prediction,
                iterations,
                converged,
                reason,
            });
        }
    }
    let Candidate {
        error,
        parameters,
        predictions,
        iterations,
        converged,
        reason,
    } = best.ok_or("no finite optimizer candidate")?;
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
    let identifiability = local_rank(columns);
    Ok(LeastSquares {
        parameters,
        identifiability,
        report: json!({"method":"deterministic_multistart_damped_least_squares",
        "converged":converged,"termination":reason,"iterations":iterations,"max_iterations":max_iterations,
        "starts":starts.len(),"start_reports":start_reports,"selected_start_index":selected_start,"scaled_rss":error,"gradient_max_abs":gradient,
        "interpretation":"numerical termination only; not a global optimum or statistical acceptance"}),
    })
}

fn rk4<F>(time: f64, y: &[f64], h: f64, f: &F) -> Vec<f64>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let k1 = f(time, y);
    let k2 = f(
        time + h / 2.0,
        &y.iter()
            .zip(&k1)
            .map(|(y, k)| y + h * k / 2.0)
            .collect::<Vec<_>>(),
    );
    let k3 = f(
        time + h / 2.0,
        &y.iter()
            .zip(&k2)
            .map(|(y, k)| y + h * k / 2.0)
            .collect::<Vec<_>>(),
    );
    let k4 = f(
        time + h,
        &y.iter()
            .zip(&k3)
            .map(|(y, k)| y + h * k)
            .collect::<Vec<_>>(),
    );
    (0..y.len())
        .map(|i| y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
        .collect()
}

/// Adaptive step-doubling RK4. Integrate from an explicit origin, never from the first requested sample.
pub(crate) fn integrate<F>(
    times: &[f64],
    origin: f64,
    initial: &[f64],
    f: F,
) -> Result<Vec<Vec<f64>>, String>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut t = origin;
    let mut state = initial.to_vec();
    let mut step = 0.1_f64;
    let mut rows = Vec::new();
    let mut attempts = 0;
    for &target in times {
        if target < origin {
            return Err("observation time precedes initial_time".into());
        }
        while t < target {
            attempts += 1;
            if attempts > 200_000 {
                return Err("ODE step budget exhausted".into());
            }
            let h = step.min(target - t);
            if t + h == t {
                return Err("ODE step underflow".into());
            }
            let coarse = rk4(t, &state, h, &f);
            let half = rk4(t, &state, h / 2.0, &f);
            let fine = rk4(t + h / 2.0, &half, h / 2.0, &f);
            let error = coarse
                .iter()
                .zip(&fine)
                .zip(&state)
                .map(|((a, b), y)| (a - b).abs() / (15.0 * (1e-10 + 1e-10 * y.abs().max(b.abs()))))
                .fold(0.0_f64, f64::max);
            if !error.is_finite() || fine.iter().any(|x| !x.is_finite()) {
                return Err("ODE left the finite numerical domain".into());
            }
            if error <= 1.0 {
                state = fine
                    .iter()
                    .zip(&coarse)
                    .map(|(fine, coarse)| fine + (fine - coarse) / 15.0)
                    .collect();
                t += h;
            }
            step = h * if error == 0.0 {
                4.0
            } else {
                (0.9 * error.powf(-0.2)).clamp(0.1, 4.0)
            };
        }
        rows.push(state.clone());
    }
    Ok(rows)
}

pub(crate) fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let rank = probability * (sorted.len() - 1) as f64;
    sorted[rank.floor() as usize] * (1. - rank.fract())
        + sorted[rank.ceil() as usize] * rank.fract()
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
}
