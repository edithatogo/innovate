use crate::calendar_time::{Date, Timestamp};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn aggregate(payload: &Value) -> Result<Value, &'static str> {
    let dates: Vec<String> = serde_json::from_value(payload["dates"].clone())
        .map_err(|_| "dates must be an array of ISO dates or timestamps")?;
    let values: Vec<f64> = serde_json::from_value(payload["values"].clone())
        .map_err(|_| "values must be a numeric array")?;
    if dates.len() != values.len() {
        return Err("dates and values must be aligned");
    }
    let frequency = payload["frequency"].as_str().unwrap_or("");
    if !["D", "W", "M", "ME"].contains(&frequency) {
        return Err("frequency must be D, W, M or ME");
    }
    if dates
        .iter()
        .any(|date| date.contains('T') || date.contains(' '))
    {
        let (timestamps, offsets): (Vec<_>, Vec<_>) = dates
            .iter()
            .map(|date| {
                Timestamp::parse_offset(date)
                    .map(|time| (time, true))
                    .or_else(|| Timestamp::parse_naive(date).map(|time| (time, false)))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("invalid ISO timestamp or mixed date and timestamp inputs")?
            .into_iter()
            .unzip();
        let aware = offsets[0];
        if offsets.iter().any(|&offset| offset != aware) {
            return Err("cannot mix offset timestamps and naive timestamps");
        }
        if frequency == "W" {
            let first = *timestamps
                .iter()
                .min()
                .expect("timestamp input is nonempty");
            let mut sums = BTreeMap::new();
            for (time, value) in timestamps.into_iter().zip(values) {
                let bucket = time
                    .weekly_bucket(first)
                    .ok_or("calendar bucket outside supported date range")?;
                add_value(&mut sums, bucket, value)?;
            }
            let (dates, values): (Vec<_>, Vec<_>) = sums
                .into_iter()
                .map(|(time, value)| {
                    let suffix = if aware { "Z" } else { "" };
                    (format!("{time}{suffix}"), value)
                })
                .unzip();
            return Ok(json!({"dates": dates, "values": values}));
        }
        return aggregate_dates(
            timestamps.into_iter().map(|time| time.date).collect(),
            values,
            frequency,
        );
    }
    let dates = dates
        .iter()
        .map(|date| Date::parse(date))
        .collect::<Option<_>>()
        .ok_or("invalid ISO calendar date")?;
    aggregate_dates(dates, values, frequency)
}

fn add_value<K: Ord>(
    sums: &mut BTreeMap<K, f64>,
    bucket: K,
    value: f64,
) -> Result<(), &'static str> {
    let sum = sums.entry(bucket).or_insert(0.);
    *sum += value;
    if !sum.is_finite() {
        return Err("calendar sum exceeded finite numerical domain");
    }
    Ok(())
}

fn aggregate_dates(
    dates: Vec<Date>,
    values: Vec<f64>,
    frequency: &str,
) -> Result<Value, &'static str> {
    let monthly = matches!(frequency, "M" | "ME");
    let Some(&first) = dates.iter().min() else {
        return Ok(json!({"dates":[],"values":[]}));
    };
    let mut sums = BTreeMap::new();
    for (date, value) in dates.into_iter().zip(values) {
        let bucket = match frequency {
            "W" => first.checked_add_days(date.days_since(first) / 7 * 7),
            "M" | "ME" => date.month_start(0),
            _ => Some(date),
        }
        .ok_or("calendar bucket outside supported date range")?;
        add_value(&mut sums, bucket, value)?;
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
                date.month_start(1)
            } else {
                date.checked_add_days(1)
            };
            let label = if monthly {
                next.and_then(|d| d.checked_add_days(-1))
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
