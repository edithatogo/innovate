//! Case and centered residual resampling for scalar model fits.
use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn fit(payload: &Value) -> Result<Value, &'static str> {
    let model = payload["fit_model"]
        .as_str()
        .ok_or("bootstrap requires fit_model")?;
    if !matches!(model, "bass" | "logistic" | "gompertz" | "fisher_pry") {
        return Err("bootstrap requires a supported scalar model");
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
    let residual_design = if method == "centered_residual_resampling" {
        if baseline["diagnostics"]["optimizer"]["converged"] != true {
            return Err("residual bootstrap requires a converged baseline fit");
        }
        let fitted: Vec<f64> = serde_json::from_value(baseline["predictions"]["values"].clone())
            .map_err(|_| "residual bootstrap requires scalar fitted values")?;
        let observed: Vec<f64> = serde_json::from_value(fit["inputs"]["observed"].clone())
            .map_err(|_| "residual bootstrap requires inputs.observed")?;
        if fitted.len() != time.len() || observed.len() != time.len() {
            return Err("residual bootstrap requires aligned fitted and observed values");
        }
        let mut residuals: Vec<_> = observed.iter().zip(&fitted).map(|(y, f)| y - f).collect();
        let mean = residuals.iter().map(|r| r / time.len() as f64).sum::<f64>();
        residuals.iter_mut().for_each(|r| *r -= mean);
        Some((fitted, residuals))
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
    for _ in 0..draws {
        if let Some((fitted, residuals)) = &residual_design {
            let synthetic: Vec<_> = fitted
                .iter()
                .map(|f| f + residuals[rng.sample(range)])
                .collect();
            if synthetic.iter().any(|value| !value.is_finite()) {
                invalid_synthetic += 1;
                continue;
            }
            sample["inputs"]["observed"] = json!(synthetic);
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
        let parameters = fit["parameters"]
            .as_object()
            .ok_or("fit has no parameter map")?;
        for (name, value) in parameters {
            estimates
                .entry(name.clone())
                .or_default()
                .push(value.as_f64().ok_or("fit parameter must be numeric")?);
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
    Ok(json!({"method":method,"seed":seed,"draws":draws,
        "succeeded":succeeded,"failed":draws-succeeded,
        "failure_counts":{"nonfinite_synthetic":invalid_synthetic,"fit_error":fit_errors,"nonconverged":nonconverged},
        "baseline_optimizer":baseline["diagnostics"]["optimizer"],
        "baseline_parameters":baseline["parameters"],"estimates":estimates,
        "interval_method":"percentile","interval_mass":interval_mass,"parameter_intervals":intervals}))
}
