use chrono::{DateTime, Datelike, Days, Months, NaiveDate, NaiveDateTime, TimeDelta};
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
    if !["D", "W", "M", "ME", "MS"].contains(&frequency) {
        return Err("frequency must be D, W, M, ME or MS");
    }
    if dates
        .iter()
        .any(|date| date.contains('T') || date.contains(' '))
    {
        let (timestamps, offsets): (Vec<_>, Vec<_>) = dates
            .iter()
            .map(|date| {
                DateTime::parse_from_rfc3339(date)
                    .map(|time| (time.naive_utc(), true))
                    .or_else(|_| {
                        NaiveDateTime::parse_from_str(date, "%Y-%m-%dT%H:%M:%S%.f")
                            .or_else(|_| {
                                NaiveDateTime::parse_from_str(date, "%Y-%m-%d %H:%M:%S%.f")
                            })
                            .map(|time| (time, false))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "invalid ISO timestamp or mixed date and timestamp inputs")?
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
                let weeks = (time - first).num_days() / 7;
                let bucket = first
                    .checked_add_signed(TimeDelta::days(weeks * 7))
                    .ok_or("calendar bucket outside supported date range")?;
                add_value(&mut sums, bucket, value)?;
            }
            let (dates, values): (Vec<_>, Vec<_>) = sums
                .into_iter()
                .map(|(time, value)| {
                    (
                        time.format(if aware {
                            "%Y-%m-%dT%H:%M:%S%.fZ"
                        } else {
                            "%Y-%m-%dT%H:%M:%S%.f"
                        })
                        .to_string(),
                        value,
                    )
                })
                .unzip();
            return Ok(json!({"dates": dates, "values": values}));
        }
        return aggregate_dates(
            timestamps.into_iter().map(|time| time.date()).collect(),
            values,
            frequency,
        );
    }
    let dates = dates
        .iter()
        .map(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d"))
        .collect::<Result<_, _>>()
        .map_err(|_| "invalid ISO calendar date")?;
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
    dates: Vec<NaiveDate>,
    values: Vec<f64>,
    frequency: &str,
) -> Result<Value, &'static str> {
    let month_end = matches!(frequency, "M" | "ME");
    let monthly = month_end || frequency == "MS";
    let Some(&first) = dates.iter().min() else {
        return Ok(json!({"dates":[],"values":[]}));
    };
    let mut sums = BTreeMap::new();
    for (date, value) in dates.into_iter().zip(values) {
        let bucket = match frequency {
            "W" => first.checked_add_days(Days::new((date - first).num_days() as u64 / 7 * 7)),
            "M" | "ME" | "MS" => date.with_day(1),
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
                date.checked_add_months(Months::new(1))
            } else {
                date.succ_opt()
            };
            let label = if month_end {
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
