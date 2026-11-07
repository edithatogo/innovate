//! Paired case resampling for weighted scalar model fits.
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
    let range = Uniform::new(0, time.len()).expect("validated nonempty time");
    let mut rng = DeterministicStream::from_run_seed(seed);
    let mut estimates = BTreeMap::<String, Vec<f64>>::new();
    let mut succeeded = 0;
    for _ in 0..draws {
        let mut weights = vec![0; time.len()];
        for _ in time {
            weights[rng.sample(range)] += 1;
        }
        sample["fit_options"]["weights"] = json!(weights);
        let Some(fit) = kernel
            .fit_model(model, sample.clone())
            .ok()
            .and_then(|fit| fit.result)
        else {
            continue;
        };
        if fit["diagnostics"]["optimizer"]["converged"] != true {
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
    Ok(
        json!({"method":"paired_case_resampling","seed":seed,"draws":draws,
        "succeeded":succeeded,"failed":draws-succeeded,
        "baseline_parameters":baseline["parameters"],"estimates":estimates,
        "interval_method":"percentile","interval_mass":interval_mass,"parameter_intervals":intervals}),
    )
}
