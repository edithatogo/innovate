//! Deterministic descriptive summaries for decision reports.
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    AdoptionDrivers {
        drivers: BTreeMap<String, f64>,
        baseline_adoption: f64,
        scenario_adoption: f64,
    },
    CompetitionEffects {
        product_shares: BTreeMap<String, f64>,
        focal_product: String,
    },
    SubstitutionThresholds {
        shares: Vec<f64>,
        thresholds: Option<Vec<f64>>,
    },
    PolicyComponents {
        components: BTreeMap<String, f64>,
        total_effect: Option<f64>,
    },
}

pub(crate) fn summarize(payload: &Value) -> Result<Value, &'static str> {
    let request = Request::deserialize(payload).map_err(|_| "invalid explainability request")?;
    match request {
        Request::AdoptionDrivers {
            drivers,
            baseline_adoption,
            scenario_adoption,
        } => {
            if drivers.is_empty() {
                return Err("drivers must be nonempty");
            }
            let delta = scenario_adoption - baseline_adoption;
            if !delta.is_finite() {
                return Err("adoption change must be finite");
            }
            let scale = drivers.values().map(|value| value.abs()).fold(0., f64::max);
            let total = drivers
                .values()
                .map(|value| value.abs() / scale.max(f64::MIN_POSITIVE))
                .sum::<f64>();
            let contributions: BTreeMap<_, _> = drivers
                .iter()
                .map(|(name, value)| {
                    let weight = if scale == 0. {
                        1. / drivers.len() as f64
                    } else {
                        (value.abs() / scale.max(f64::MIN_POSITIVE)) / total
                    };
                    (name, json!({"weight":weight,"contribution":delta * weight}))
                })
                .collect();
            Ok(
                json!({"kind":"adoption_drivers","baseline_adoption":baseline_adoption,
                "scenario_adoption":scenario_adoption,"delta":delta,"contributions":contributions,
                "deterministic":true}),
            )
        }
        Request::CompetitionEffects {
            mut product_shares,
            focal_product,
        } => {
            let focal = product_shares
                .remove(&focal_product)
                .ok_or("focal_product must be present")?;
            if focal < 0. || product_shares.values().any(|share| *share < 0.) {
                return Err("product shares must be nonnegative");
            }
            let pressure: f64 = product_shares.values().sum();
            let sum = focal + pressure;
            if sum > 1. + 1e-9 {
                return Err("product shares must sum to at most one");
            }
            let strongest = product_shares.values().copied().fold(0., f64::max);
            Ok(
                json!({"kind":"competition_effects","focal_product":focal_product,"focal_share":focal,
                "competitor_shares":product_shares,"competitive_pressure":pressure,"share_sum":sum,
                "lead":focal - strongest,"deterministic":true}),
            )
        }
        Request::SubstitutionThresholds { shares, thresholds } => {
            let final_share = shares.last().ok_or("shares must be nonempty")?;
            let thresholds = thresholds.unwrap_or_else(|| vec![0.1, 0.25, 0.5, 0.75]);
            let crossings: Vec<_> = thresholds
                .into_iter()
                .map(|threshold| {
                    let index = shares.iter().position(|share| *share >= threshold);
                    json!({"threshold":threshold,"first_index":index,"met":index.is_some()})
                })
                .collect();
            Ok(
                json!({"kind":"substitution_thresholds","crossings":crossings,
                "final_share":final_share,"deterministic":true}),
            )
        }
        Request::PolicyComponents {
            components,
            total_effect,
        } => {
            let dominant = components
                .iter()
                .max_by(|(left_name, left), (right_name, right)| {
                    left.abs()
                        .total_cmp(&right.abs())
                        .then_with(|| right_name.cmp(left_name))
                })
                .map(|(name, _)| name)
                .ok_or("components must be nonempty")?;
            let summed: f64 = components.values().sum();
            let total = total_effect.unwrap_or(summed);
            let residual = total - summed;
            if !summed.is_finite() || !residual.is_finite() {
                return Err("policy component totals must be finite");
            }
            Ok(
                json!({"kind":"policy_components","components":components,"sum_components":summed,
                "total_effect":total,"residual":residual,"dominant_component":dominant,"deterministic":true}),
            )
        }
    }
}
