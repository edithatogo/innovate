use chrono::{Datelike, Days, Months, NaiveDate};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn aggregate(payload: &Value) -> Result<Value, &'static str> {
    let dates: Vec<String> = serde_json::from_value(payload["dates"].clone())
        .map_err(|_| "dates must be an array of ISO calendar dates")?;
    let values: Vec<f64> = serde_json::from_value(payload["values"].clone())
        .map_err(|_| "values must be a numeric array")?;
    if dates.len() != values.len() {
        return Err("dates and values must be aligned");
    }
    let dates: Vec<_> = dates
        .iter()
        .map(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d"))
        .collect::<Result<_, _>>()
        .map_err(|_| "invalid ISO calendar date")?;
    let frequency = payload["frequency"].as_str().unwrap_or("");
    if !["D", "W", "M", "ME"].contains(&frequency) {
        return Err("frequency must be D, W, M or ME");
    }
    let monthly = matches!(frequency, "M" | "ME");
    let Some(&first) = dates.iter().min() else {
        return Ok(json!({"dates":[],"values":[]}));
    };
    let mut sums = BTreeMap::new();
    for (date, value) in dates.into_iter().zip(values) {
        let bucket = match frequency {
            "W" => first.checked_add_days(Days::new((date - first).num_days() as u64 / 7 * 7)),
            "M" | "ME" => date.with_day(1),
            _ => Some(date),
        }
        .ok_or("calendar bucket outside supported date range")?;
        let sum = sums.entry(bucket).or_insert(0.);
        *sum += value;
        if !sum.is_finite() {
            return Err("calendar sum exceeded finite numerical domain");
        }
    }
    let mut labels = Vec::new();
    let mut values = Vec::new();
    if frequency == "W" {
        for (date, value) in sums {
            labels.push(date.to_string());
            values.push(value);
        }
    } else {
        let (&last, _) = sums.last_key_value().unwrap();
        let (&mut_date, _) = sums.first_key_value().unwrap();
        let mut date = mut_date;
        loop {
            if labels.len() == 100_000 {
                return Err("calendar output exceeds 100000 periods");
            }
            let next = if monthly {
                date.checked_add_months(Months::new(1))
            } else {
                date.succ_opt()
            };
            let label = if monthly {
                next.and_then(|d| d.pred_opt())
                    .ok_or("month end outside supported date range")?
            } else {
                date
            };
            labels.push(label.to_string());
            values.push(sums.get(&date).copied().unwrap_or(0.));
            if date == last {
                break;
            }
            date = next.ok_or("calendar period outside supported date range")?;
        }
    }
    Ok(json!({"dates":labels,"values":values}))
}
