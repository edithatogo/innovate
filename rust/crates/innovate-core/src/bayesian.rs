//! Bounded posterior quadrature using the native predictor.
use serde_json::{Value, json};

pub(crate) fn grid(payload: &Value) -> Result<Value, &'static str> {
    let model = payload["forecast_model"]
        .as_str()
        .ok_or("forecast_model is required")?;
    let mut prediction = payload["prediction"].clone();
    let parameters = prediction["parameters"]
        .as_object()
        .ok_or("explicit parameters are required")?;
    let axes = match payload.get("axes") {
        Some(value) => {
            if ["parameter", "bounds", "points"]
                .iter()
                .any(|key| payload.get(key).is_some())
            {
                return Err("provide axes or a single parameter grid, not both");
            }
            let axes = value
                .as_array()
                .filter(|axes| axes.len() == 2)
                .ok_or("joint posterior requires two axes")?;
            axes.iter().map(axis).collect::<Result<Vec<_>, _>>()?
        }
        None => vec![axis(payload)?],
    };
    if axes.iter().any(|(name, _)| !parameters.contains_key(*name)) {
        return Err("unknown posterior parameter");
    }
    if axes.len() == 2 && axes[0].0 == axes[1].0 {
        return Err("posterior axes must name distinct parameters");
    }
    let cells = axes.iter().map(|(_, grid)| grid.len()).product::<usize>();
    if cells > 10_000 {
        return Err("posterior grid exceeds 10000 cells");
    }
    let credible_mass = payload
        .get("credible_mass")
        .map_or(Some(0.95), Value::as_f64)
        .filter(|mass| *mass > 0. && *mass < 1.)
        .ok_or("credible_mass must be between zero and one")?;
    let (sigma, observed) = observations(payload)?;
    let kernel = crate::KernelBinding::new();
    let mut log_weights = Vec::with_capacity(cells);
    let stride = axes.get(1).map_or(1, |(_, grid)| grid.len());
    for cell in 0..cells {
        let (name, grid) = &axes[0];
        prediction["parameters"][*name] = json!(grid[cell / stride]);
        if let Some((name, grid)) = axes.get(1) {
            prediction["parameters"][*name] = json!(grid[cell % stride]);
        }
        let log_weight = log_likelihood(&kernel, model, &prediction, &observed, sigma)?;
        log_weights.push(log_weight);
    }
    let peak = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return Err("all posterior grid likelihoods are nonfinite");
    }
    let mut weights: Vec<_> = log_weights
        .iter()
        .map(|log_weight| (log_weight - peak).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= total);
    let mut result = json!({"posterior_mass":weights,
        "method":"midpoint_grid_quadrature","prior":"uniform_within_bounds","likelihood":"independent_gaussian_known_sd"});
    let mut means = Vec::with_capacity(axes.len());
    let mut intervals = Vec::with_capacity(axes.len());
    let tail = (1. - credible_mass) / 2.;
    for (axis, (_, grid)) in axes.iter().enumerate() {
        let mut marginal = vec![0.; grid.len()];
        for (cell, weight) in weights.iter().enumerate() {
            let index = if axis == 0 {
                cell / stride
            } else {
                cell % stride
            };
            marginal[index] += weight;
        }
        means.push(
            grid.iter()
                .zip(&marginal)
                .map(|(value, mass)| value * mass)
                .sum::<f64>(),
        );
        let quantile = |probability| {
            let mut cumulative = 0.;
            grid.iter()
                .zip(&marginal)
                .find_map(|(value, mass)| {
                    cumulative += mass;
                    (cumulative >= probability).then_some(*value)
                })
                .unwrap_or(grid[grid.len() - 1])
        };
        intervals.push([quantile(tail), quantile(1. - tail)]);
    }
    result["credible_mass"] = json!(credible_mass);
    result["interval_method"] = json!("equal_tail_discrete_grid");
    if means.iter().any(|mean| !mean.is_finite()) {
        return Err("posterior mean exceeds finite numerical domain");
    }
    if axes.len() == 1 {
        result["parameter"] = json!(axes[0].0);
        result["grid"] = json!(axes[0].1);
        result["posterior_mean"] = json!(means[0]);
        result["credible_interval"] = json!(intervals[0]);
    } else {
        result["axes"] = json!(
            axes.iter()
                .map(|(name, grid)| json!({"parameter":name,"grid":grid}))
                .collect::<Vec<_>>()
        );
        result["shape"] = json!([axes[0].1.len(), stride]);
        result["posterior_means"] = json!(means);
        result["credible_intervals"] = json!(intervals);
    }
    Ok(result)
}

fn axis(payload: &Value) -> Result<(&str, Vec<f64>), &'static str> {
    let name = payload["parameter"]
        .as_str()
        .ok_or("parameter is required")?;
    let [lower, upper]: [f64; 2] = serde_json::from_value(payload["bounds"].clone())
        .map_err(|_| "bounds must contain two finite numbers")?;
    let width = upper - lower;
    if !width.is_finite() || width <= 0. {
        return Err("posterior bounds must be finite and increasing");
    }
    let points = payload["points"]
        .as_u64()
        .filter(|n| (3..=1001).contains(n))
        .ok_or("points must be between 3 and 1001")? as usize;
    let grid: Vec<_> = (0..points)
        .map(|index| lower + width * ((index as f64 + 0.5) / points as f64))
        .collect();
    if grid.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("posterior grid points must be distinct");
    }
    Ok((name, grid))
}

fn observations(payload: &Value) -> Result<(f64, Vec<f64>), &'static str> {
    let sigma = payload["noise_sd"]
        .as_f64()
        .filter(|sd| sd.is_finite() && *sd > 0.)
        .ok_or("noise_sd must be finite and positive")?;
    let observed: Vec<f64> = serde_json::from_value(payload["observed"].clone())
        .map_err(|_| "observed must be a finite numeric array")?;
    if observed.is_empty() || observed.len() > 500 || observed.iter().any(|y| !y.is_finite()) {
        return Err("posterior requires 1 to 500 finite observations");
    }
    Ok((sigma, observed))
}

fn log_likelihood(
    kernel: &crate::KernelBinding,
    model: &str,
    prediction: &Value,
    observed: &[f64],
    sigma: f64,
) -> Result<f64, &'static str> {
    let forecast = kernel
        .predict_model(model, prediction.clone())
        .map_err(|_| "posterior prediction failed")?
        .result
        .ok_or("posterior prediction returned no result")?;
    let values = forecast["values"]
        .as_array()
        .filter(|x| x.len() == observed.len())
        .ok_or("posterior requires aligned scalar predictions")?;
    let mut log_weight = 0.;
    for (value, y) in values.iter().zip(observed) {
        let value = value
            .as_f64()
            .ok_or("posterior prediction must be numeric")?;
        log_weight -= 0.5 * ((value - y) / sigma).powi(2);
    }
    Ok(log_weight)
}

pub(crate) fn sample(payload: &Value) -> Result<Value, &'static str> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};

    let model = payload["forecast_model"]
        .as_str()
        .ok_or("forecast_model is required")?;
    let mut prediction = payload["prediction"].clone();
    let joint = payload.get("axes").is_some();
    let axes = match payload.get("axes") {
        Some(axes) => {
            if ["parameter", "bounds", "proposal_step"]
                .iter()
                .any(|key| payload.get(key).is_some())
            {
                return Err("provide axes or a single sampling parameter");
            }
            axes.as_array()
                .filter(|axes| (2..=16).contains(&axes.len()))
                .ok_or("joint sampling requires 2 to 16 axes")?
                .iter()
                .collect::<Vec<_>>()
        }
        None => vec![payload],
    };
    let mut names = Vec::with_capacity(axes.len());
    let mut current = Vec::with_capacity(axes.len());
    let mut bounds = Vec::with_capacity(axes.len());
    let mut steps = Vec::with_capacity(axes.len());
    for axis in axes {
        let name = axis["parameter"].as_str().ok_or("parameter is required")?;
        if names.contains(&name) {
            return Err("sampling axes must name distinct parameters");
        }
        let initial = prediction["parameters"][name]
            .as_f64()
            .ok_or("initial parameter is required")?;
        let [lower, upper]: [f64; 2] = serde_json::from_value(axis["bounds"].clone())
            .map_err(|_| "bounds must contain two finite numbers")?;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower >= upper
            || !(lower..=upper).contains(&initial)
        {
            return Err("initial parameter must lie inside finite increasing bounds");
        }
        let step = axis["proposal_step"]
            .as_f64()
            .filter(|step| step.is_finite() && *step > 0.)
            .ok_or("proposal_step must be finite and positive")?;
        names.push(name);
        current.push(initial);
        bounds.push(lower..=upper);
        steps.push(step);
    }
    let draws = payload["draws"]
        .as_u64()
        .filter(|n| (2..=10_000).contains(n))
        .ok_or("draws must be between 2 and 10000")?;
    let warmup = payload["warmup"]
        .as_u64()
        .filter(|n| *n <= 10_000)
        .ok_or("warmup must be between 0 and 10000")?;
    let seed = payload["seed"]
        .as_u64()
        .ok_or("seed must be an unsigned integer")?;
    let (sigma, observed) = observations(payload)?;
    let kernel = crate::KernelBinding::new();
    let mut current_log = log_likelihood(&kernel, model, &prediction, &observed, sigma)?;
    if !current_log.is_finite() {
        return Err("initial log likelihood must be finite");
    }
    let mut rng = DeterministicStream::from_run_seed(seed);
    let unit = Uniform::new(0., 1.).expect("valid uniform range");
    let mut samples = Vec::with_capacity(draws as usize);
    let mut accepted = 0;
    let mut proposed = current.clone();
    for iteration in 0..(warmup + draws) {
        for ((candidate, value), step) in proposed.iter_mut().zip(&current).zip(&steps) {
            *candidate = value + step * (2. * rng.sample(unit) - 1.);
        }
        if proposed
            .iter()
            .zip(&bounds)
            .all(|(value, bound)| bound.contains(value))
        {
            for (name, value) in names.iter().zip(&proposed) {
                prediction["parameters"][*name] = json!(value);
            }
            let proposed_log = log_likelihood(&kernel, model, &prediction, &observed, sigma)?;
            let acceptance: f64 = rng.sample(unit);
            if proposed_log.is_finite() && acceptance.ln() < proposed_log - current_log {
                std::mem::swap(&mut current, &mut proposed);
                current_log = proposed_log;
                accepted += u64::from(iteration >= warmup);
            }
        }
        if iteration >= warmup {
            samples.push(current.clone());
        }
    }
    let means = (0..names.len())
        .map(|axis| {
            samples
                .iter()
                .map(|row| row[axis] / draws as f64)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    if means.iter().any(|mean| !mean.is_finite()) {
        return Err("posterior mean exceeds finite numerical domain");
    }
    let mut result = json!({
        "seed":seed,"draws":draws,"warmup":warmup,"accepted":accepted,
        "acceptance_rate":accepted as f64 / draws as f64,
        "method":"random_walk_metropolis","prior":"uniform_within_bounds",
        "likelihood":"independent_gaussian_known_sd"
    });
    if joint {
        result["parameters"] = json!(names);
        result["samples"] = json!(samples);
        result["posterior_means"] = json!(means);
    } else {
        result["parameter"] = json!(names[0]);
        result["samples"] = json!(samples.iter().map(|row| row[0]).collect::<Vec<_>>());
        result["posterior_mean"] = json!(means[0]);
    }
    Ok(result)
}

/// Classical split R-hat for equal-length chains of one posterior marginal.
pub(crate) fn split_rhat(payload: &Value) -> Result<Value, &'static str> {
    let chains = payload["chains"]
        .as_array()
        .filter(|chains| (2..=16).contains(&chains.len()))
        .ok_or("split R-hat requires 2 to 16 chains")?;
    let draws = chains[0]
        .as_array()
        .map(Vec::len)
        .filter(|draws| (4..=10_000).contains(draws))
        .ok_or("split R-hat requires 4 to 10000 draws per chain")?;
    let chains = chains
        .iter()
        .map(|chain| {
            chain
                .as_array()
                .filter(|chain| chain.len() == draws)
                .ok_or("chains must have equal draw counts")?
                .iter()
                .map(|value| value.as_f64().ok_or("chain draws must be finite numbers"))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let half = draws / 2;
    let splits: Vec<_> = chains
        .iter()
        .flat_map(|chain| [&chain[..half], &chain[draws - half..]])
        .collect();
    let method = payload
        .get("method")
        .map_or(Some("classical_split_rhat"), Value::as_str);
    let result = match method {
        Some("classical_split_rhat") => {
            let rhat = split_rhat_moments(&splits)?;
            json!({"method":"classical_split_rhat","rhat":rhat,
                "status": if rhat.is_some() {"computed"} else {"zero_within_split_variance"}})
        }
        Some("rank_normalized_split_rhat") => {
            let pooled: Vec<_> = splits
                .iter()
                .flat_map(|split| split.iter().copied())
                .collect();
            let bulk = ranked_rhat(&pooled, half)?;
            let mut sorted = pooled.clone();
            sorted.sort_unstable_by(f64::total_cmp);
            let middle = sorted.len() / 2;
            let median = sorted[middle - 1].midpoint(sorted[middle]);
            let folded: Vec<_> = pooled.iter().map(|value| (value - median).abs()).collect();
            if folded.iter().any(|value| !value.is_finite()) {
                return Err("folded draws exceed finite numerical domain");
            }
            let tail = ranked_rhat(&folded, half)?;
            let rhat = bulk.zip(tail).map(|(bulk, tail)| bulk.max(tail));
            json!({"method":"rank_normalized_split_rhat","rhat":rhat,
                "bulk_rhat":bulk,"folded_rhat":tail,
                "status":if rhat.is_some() {"computed"} else {"zero_within_split_variance"}})
        }
        _ => return Err("unknown split R-hat method"),
    };
    let mut result = result;
    result["chains"] = json!(chains.len());
    result["draws_per_chain"] = json!(draws);
    result["draws_per_split"] = json!(half);
    Ok(result)
}

fn ranked_rhat(values: &[f64], half: usize) -> Result<Option<f64>, &'static str> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_unstable_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut scores = vec![0.; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]] == values[order[end]] {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.;
        let probability = (rank - 0.375) / (values.len() as f64 + 0.25);
        let score = normal_quantile(probability);
        for index in &order[start..end] {
            scores[*index] = score;
        }
        start = end;
    }
    split_rhat_moments(&scores.chunks_exact(half).collect::<Vec<_>>())
}

fn normal_quantile(probability: f64) -> f64 {
    // The bounded draw counts keep the Blom probabilities inside Phi(±8).
    let (mut low, mut high) = (-8_f64, 8_f64);
    for _ in 0..48 {
        let middle = low.midpoint(high);
        if 0.5 * libm::erfc(-middle / std::f64::consts::SQRT_2) < probability {
            low = middle;
        } else {
            high = middle;
        }
    }
    low.midpoint(high)
}

fn split_rhat_moments(splits: &[&[f64]]) -> Result<Option<f64>, &'static str> {
    let n = splits[0].len() as f64;
    let mut means = Vec::with_capacity(splits.len());
    let mut variances = Vec::with_capacity(splits.len());
    // Shift before computing moments to retain precision for chains far from zero.
    let origin = splits[0][0];
    for split in splits {
        let mean = split.iter().map(|value| (value - origin) / n).sum::<f64>();
        let variance = split
            .iter()
            .map(|value| ((value - origin) - mean).powi(2) / (n - 1.))
            .sum::<f64>();
        means.push(mean);
        variances.push(variance);
    }
    let m = means.len() as f64;
    let within = variances.iter().sum::<f64>() / m;
    let center = means.iter().sum::<f64>() / m;
    let between_per_draw = means
        .iter()
        .map(|mean| (mean - center).powi(2) / (m - 1.))
        .sum::<f64>();
    if !within.is_finite() || !between_per_draw.is_finite() {
        return Err("split R-hat moments exceed finite numerical domain");
    }
    if within == 0. {
        return Ok(None);
    }
    let rhat = ((n - 1.) / n + between_per_draw / within).sqrt();
    if !rhat.is_finite() {
        return Err("split R-hat exceeds finite numerical domain");
    }
    Ok(Some(rhat))
}
