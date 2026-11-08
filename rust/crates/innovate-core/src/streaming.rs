use serde_json::{Value, json};

pub(crate) fn update(payload: &Value) -> Result<Value, &'static str> {
    let series = |key: &str| {
        serde_json::from_value::<Vec<f64>>(payload[key].clone())
            .map_err(|_| "streaming update requires numeric arrays")
    };
    let mut time = series("previous_time")?;
    let mut observed = series("previous_observed")?;
    let new_time = series("new_time")?;
    let new_observed = series("new_observed")?;
    if time.is_empty() || time.len() != observed.len() {
        return Err("previous time and observations must be nonempty and aligned");
    }
    if new_time.is_empty() || new_time.len() != new_observed.len() {
        return Err("new time and observations must be nonempty and aligned");
    }
    let previous_count = time.len();
    let new_count = new_time.len();
    let previous_last = observed[previous_count - 1];
    time.extend(new_time);
    observed.extend(new_observed);
    if time.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("streaming time points must be sorted");
    }
    if observed.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("streaming observed values must be cumulative");
    }
    let total_count = time.len();
    let last_observed = observed[total_count - 1];
    let growth = last_observed - previous_last;
    let rate = if previous_last == 0. {
        0.
    } else {
        growth / previous_last
    };
    if !growth.is_finite() || !rate.is_finite() {
        return Err("streaming growth exceeds the finite numerical domain");
    }
    let assumptions: Option<Vec<String>> =
        serde_json::from_value(payload["assumptions"].clone())
            .map_err(|_| "assumptions must be an array of strings")?;
    Ok(
        json!({"workflow":"streaming_update","stability":"provisional","backend":"rust_native",
        "time":time,"mean":observed,"metadata":{"previous_count":previous_count,"new_count":new_count,
        "assumptions":assumptions.unwrap_or_default(),
        "state":{"last_time":time[total_count - 1],"last_observed":last_observed,"total_count":total_count}},
        "diagnostics":{"incremental_growth":growth,"growth_rate":rate}}),
    )
}
