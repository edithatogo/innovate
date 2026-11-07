//! Case and centered residual resampling for native model fits.
use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn fit(payload: &Value) -> Result<Value, &'static str> {
    let model = payload["fit_model"]
        .as_str()
        .ok_or("bootstrap requires fit_model")?;
    let scalar = matches!(model, "bass" | "logistic" | "gompertz" | "fisher_pry");
    if !scalar
        && !matches!(
            model,
            "norton_bass" | "lotka_volterra" | "complementary_goods" | "lock_in" | "multi_product"
        )
    {
        return Err("bootstrap requires a supported model");
    }
    let method = payload
        .get("method")
        .map_or(Some("paired_case_resampling"), Value::as_str)
        .filter(|method| {
            matches!(
                *method,
                "paired_case_resampling" | "centered_residual_resampling"
            )
        })
        .ok_or("unknown bootstrap resampling method")?;
    let draws = payload["draws"]
        .as_u64()
        .filter(|n| (2..=200).contains(n))
        .ok_or("bootstrap draws must be between 2 and 200")?;
    let interval_mass = payload
        .get("interval_mass")
        .map_or(Some(0.95), Value::as_f64)
        .filter(|mass| *mass > 0. && *mass < 1.)
        .ok_or("bootstrap interval_mass must be between zero and one")?;
    let seed = payload["seed"]
        .as_u64()
        .ok_or("bootstrap requires an unsigned seed")?;
    let fit = payload["fit"]
        .as_object()
        .ok_or("bootstrap requires a fit payload")?;
    let time = fit
        .get("inputs")
        .and_then(|inputs| inputs["time"].as_array())
        .filter(|time| (4..=500).contains(&time.len()))
        .ok_or("bootstrap requires 4 to 500 time samples")?;
    let mut sample = Value::Object(fit.clone());
    let options = sample
        .as_object_mut()
        .unwrap()
        .entry("fit_options")
        .or_insert_with(|| json!({}));
    let options = options
        .as_object_mut()
        .ok_or("fit_options must be an object")?;
    if options.contains_key("weights") {
        return Err("bootstrap fit_options must omit weights");
    }
    let kernel = crate::KernelBinding::new();
    let baseline = kernel
        .fit_model(model, sample.clone())
        .map_err(|_| "baseline fit failed")?
        .result
        .ok_or("baseline fit returned no result")?;
    if baseline["diagnostics"]["identifiability"]["status"] == "rank_deficient" {
        return Err("parameter bootstrap requires a baseline fit without detected rank deficiency");
    }
    let forecast_inputs = payload.get("forecast_inputs");
    let predict = |inputs: &Value, fit: &Value| -> Option<(Vec<String>, Vec<Vec<f64>>)> {
        let result = kernel
            .predict_model(model, json!({"inputs":inputs,"state":fit["state"]}))
            .ok()?
            .result?;
        if scalar {
            let values: Vec<f64> = serde_json::from_value(result["values"].clone()).ok()?;
            Some((
                vec!["prediction".into()],
                values.into_iter().map(|value| vec![value]).collect(),
            ))
        } else {
            Some((
                serde_json::from_value(result["columns"].clone()).ok()?,
                serde_json::from_value(result["rows"].clone()).ok()?,
            ))
        }
    };
    let baseline_forecast = if let Some(inputs) = forecast_inputs {
        let count = inputs["time"]
            .as_array()
            .filter(|time| (1..=100).contains(&time.len()))
            .ok_or("forecast_inputs requires 1 to 100 time samples")?
            .len();
        Some(
            predict(inputs, &baseline)
                .filter(|(columns, rows)| {
                    !columns.is_empty()
                        && rows.len() == count
                        && rows.iter().all(|row| row.len() == columns.len())
                })
                .ok_or("baseline forecast failed")?,
        )
    } else {
        None
    };
    let mut forecast_draws = Vec::new();
    let profile = ["event_profile", "launch_profile", "joint_time_profile"]
        .into_iter()
        .find(|key| baseline[*key].is_object());
    let mut event_counts = profile
        .and_then(|key| baseline[key]["candidates"].as_array())
        .map(|candidates| vec![0_u64; candidates.len()]);
    let residual_design = if method == "centered_residual_resampling" {
        if baseline["diagnostics"]["optimizer"]["converged"] != true {
            return Err("residual bootstrap requires a converged baseline fit");
        }
        let rows = |value: &Value| -> Result<Vec<Vec<f64>>, &'static str> {
            if scalar {
                serde_json::from_value::<Vec<f64>>(value.clone())
                    .map(|values| values.into_iter().map(|value| vec![value]).collect())
            } else {
                serde_json::from_value(value.clone())
            }
            .map_err(|_| "residual bootstrap requires numeric observed and fitted rows")
        };
        let fitted = rows(&baseline["predictions"][if scalar { "values" } else { "rows" }])?;
        let observed = rows(&fit["inputs"]["observed"])?;
        let width = fitted.first().map_or(0, Vec::len);
        if width == 0
            || fitted.len() != time.len()
            || observed.len() != time.len()
            || fitted.iter().chain(&observed).any(|row| row.len() != width)
        {
            return Err("residual bootstrap requires aligned fitted and observed rows");
        }
        let offset = usize::from(
            baseline["diagnostics"]["initial_condition"]["excluded_conditioning_rows"] == 1,
        );
        let n = (time.len() - offset) as f64;
        let mut residuals: Vec<Vec<f64>> = observed[offset..]
            .iter()
            .zip(&fitted[offset..])
            .map(|(y, f)| y.iter().zip(f).map(|(y, f)| y - f).collect())
            .collect();
        for col in 0..width {
            let mean = residuals.iter().map(|r| r[col] / n).sum::<f64>();
            residuals.iter_mut().for_each(|r| r[col] -= mean);
        }
        Some((fitted, observed, residuals, offset))
    } else {
        None
    };
    let range = Uniform::new(0, time.len()).expect("validated nonempty time");
    let mut rng = DeterministicStream::from_run_seed(seed);
    let mut estimates = BTreeMap::<String, Vec<f64>>::new();
    let mut succeeded = 0;
    let mut invalid_synthetic = 0;
    let mut fit_errors = 0;
    let mut nonconverged = 0;
    let mut rank_deficient = 0;
    for _ in 0..draws {
        if let Some((fitted, observed, residuals, offset)) = &residual_design {
            let residual_range = Uniform::new(0, residuals.len()).expect("validated residual rows");
            let mut synthetic = observed[..*offset].to_vec();
            synthetic.extend(fitted[*offset..].iter().map(|f| {
                let residual = &residuals[rng.sample(residual_range)];
                f.iter()
                    .zip(residual)
                    .map(|(f, r)| f + r)
                    .collect::<Vec<_>>()
            }));
            if synthetic.iter().flatten().any(|value| !value.is_finite()) {
                invalid_synthetic += 1;
                continue;
            }
            sample["inputs"]["observed"] = if scalar {
                json!(synthetic.into_iter().flatten().collect::<Vec<_>>())
            } else {
                json!(synthetic)
            };
        } else {
            let mut weights = vec![0; time.len()];
            for _ in time {
                weights[rng.sample(range)] += 1;
            }
            sample["fit_options"]["weights"] = json!(weights);
        }
        let Some(fit) = kernel
            .fit_model(model, sample.clone())
            .ok()
            .and_then(|fit| fit.result)
        else {
            fit_errors += 1;
            continue;
        };
        if fit["diagnostics"]["optimizer"]["converged"] != true {
            nonconverged += 1;
            continue;
        }
        if fit["diagnostics"]["identifiability"]["status"] == "rank_deficient" {
            rank_deficient += 1;
            continue;
        }
        let parameters = fit["parameters"]
            .as_object()
            .ok_or("fit has no parameter map")?;
        for (name, value) in parameters {
            estimates
                .entry(name.clone())
                .or_default()
                .push(value.as_f64().ok_or("fit parameter must be numeric")?);
        }
        if let Some((key, counts)) = profile.zip(event_counts.as_mut()) {
            let index = fit[key]["selected_index"]
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| counts.get_mut(index))
                .ok_or("bootstrap refit did not select a valid event candidate")?;
            *index += 1;
        }
        if let Some((inputs, (columns, baseline))) = forecast_inputs.zip(baseline_forecast.as_ref())
            && let Some((_, rows)) = predict(inputs, &fit).filter(|(names, rows)| {
                names == columns
                    && rows.len() == baseline.len()
                    && rows.iter().all(|row| row.len() == columns.len())
            })
        {
            forecast_draws.push(rows);
        }
        succeeded += 1;
    }
    let tail = (1. - interval_mass) / 2.;
    let parameters = baseline["parameters"]
        .as_object()
        .ok_or("baseline has no parameter map")?;
    let intervals: BTreeMap<_, _> = parameters
        .keys()
        .map(|name| {
            let bounds = estimates
                .get(name)
                .filter(|values| values.len() >= 2)
                .map(|values| {
                    let mut sorted = values.clone();
                    sorted.sort_unstable_by(f64::total_cmp);
                    [
                        crate::numerics::linear_quantile(&sorted, tail),
                        crate::numerics::linear_quantile(&sorted, 1. - tail),
                    ]
                });
            (name, bounds)
        })
        .collect();
    let standard_errors: BTreeMap<_, _> = parameters
        .keys()
        .map(|name| {
            let error = estimates
                .get(name)
                .filter(|values| values.len() >= 2)
                .map(|values| {
                    let scale = values.iter().map(|value| value.abs()).fold(0., f64::max);
                    if scale == 0. {
                        return 0.;
                    }
                    let mean =
                        values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64;
                    let squares = values
                        .iter()
                        .map(|value| (value / scale - mean).powi(2))
                        .sum::<f64>();
                    (squares / (values.len() - 1) as f64).sqrt() * scale
                });
            (name, error)
        })
        .collect();
    let mut result = json!({"method":method,"seed":seed,"draws":draws,
        "succeeded":succeeded,"failed":draws-succeeded,
        "failure_counts":{"nonfinite_synthetic":invalid_synthetic,"fit_error":fit_errors,"nonconverged":nonconverged,"rank_deficient":rank_deficient},
        "baseline_optimizer":baseline["diagnostics"]["optimizer"],
        "baseline_parameters":baseline["parameters"],"estimates":estimates,
        "interval_method":"percentile","interval_mass":interval_mass,"parameter_intervals":intervals,
        "parameter_standard_errors":standard_errors,
        "parameter_correlation":correlation(&estimates, succeeded)});
    if let Some((columns, baseline)) = baseline_forecast {
        let intervals: Vec<Vec<_>> = baseline
            .iter()
            .enumerate()
            .map(|(i, row)| {
                (0..row.len())
                    .map(|j| {
                        if forecast_draws.len() < 2 {
                            return None;
                        }
                        let mut values: Vec<_> =
                            forecast_draws.iter().map(|draw| draw[i][j]).collect();
                        values.sort_unstable_by(f64::total_cmp);
                        Some([
                            crate::numerics::linear_quantile(&values, tail),
                            crate::numerics::linear_quantile(&values, 1. - tail),
                        ])
                    })
                    .collect()
            })
            .collect();
        result["forecast"] = json!({"time":forecast_inputs.expect("forecast inputs")["time"],
            "columns":columns,"baseline_rows":baseline,"pointwise_intervals":intervals,
            "succeeded":forecast_draws.len(),"failed":succeeded-forecast_draws.len() as u64,
            "interval_method":"percentile_of_refitted_mean_paths","interval_mass":interval_mass});
    }
    if let Some((_, _, residuals, offset)) = &residual_design {
        result["residual_design"] = json!({"pool_rows":residuals.len(),"conditioning_rows":offset});
    }
    if let Some((key, counts)) = profile.zip(event_counts) {
        let profile = &baseline[key];
        let candidates = profile["candidates"]
            .as_array()
            .expect("profile candidates");
        let selections: Vec<_> = candidates
            .iter()
            .zip(counts)
            .map(|(candidate, count)| {
                let mut row = json!({"count":count,
                "frequency":(succeeded > 0).then(|| count as f64 / succeeded as f64)});
                for field in ["t_event", "launch_time", "override_value"] {
                    if let Some(value) = candidate.get(field) {
                        row[field] = value.clone();
                    }
                }
                row
            })
            .collect();
        let output_key = match key {
            "launch_profile" => "launch_selection",
            "joint_time_profile" => "joint_time_selection",
            _ => "event_selection",
        };
        result[output_key] = json!({
            "baseline":profile["selected"],
            "successful_refits":succeeded,"candidates":selections
        });
    }
    Ok(result)
}

fn correlation(estimates: &BTreeMap<String, Vec<f64>>, draws: u64) -> Value {
    if draws < 2 || estimates.len() < 2 {
        return Value::Null;
    }
    let columns: BTreeMap<_, _> = estimates
        .iter()
        .map(|(name, values)| {
            let scale = values.iter().map(|value| value.abs()).fold(0., f64::max);
            let centered = (scale > 0.)
                .then(|| {
                    let origin = values[0] / scale;
                    let mut column: Vec<_> = values.iter().map(|v| v / scale - origin).collect();
                    let mean = column.iter().sum::<f64>() / draws as f64;
                    column.iter_mut().for_each(|v| *v -= mean);
                    let norm = column.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
                    (norm > 0.).then(|| column.into_iter().map(|v| v / norm).collect::<Vec<_>>())
                })
                .flatten();
            (name, centered)
        })
        .collect();
    let matrix: BTreeMap<_, _> = columns
        .iter()
        .map(|(name, left)| {
            let row: BTreeMap<_, _> = columns
                .iter()
                .map(|(other, right)| {
                    let coefficient = left.as_ref().zip(right.as_ref()).map(|(left, right)| {
                        left.iter()
                            .zip(right)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            .clamp(-1., 1.)
                    });
                    (*other, coefficient)
                })
                .collect();
            (*name, row)
        })
        .collect();
    json!(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_handles_signed_and_constant_columns() {
        let draws = BTreeMap::from([
            ("a".into(), vec![1., 2., 3.]),
            ("b".into(), vec![3., 2., 1.]),
            ("constant".into(), vec![7.; 3]),
        ]);
        let matrix = correlation(&draws, 3);
        assert!((matrix["a"]["a"].as_f64().unwrap() - 1.).abs() < 1e-14);
        assert!((matrix["a"]["b"].as_f64().unwrap() + 1.).abs() < 1e-14);
        assert_eq!(matrix["a"]["b"], matrix["b"]["a"]);
        assert!(matrix["a"]["constant"].is_null());
        assert!(correlation(&draws, 1).is_null());
    }
}
