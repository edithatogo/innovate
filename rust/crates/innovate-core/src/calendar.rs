use chrono::{DateTime, Datelike, Days, Months, NaiveDate, NaiveDateTime, TimeDelta};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const MAX_PERIODS: usize = 100_000;

pub(crate) fn aggregate(payload: &Value) -> Result<Value, &'static str> {
    let dates: Vec<String> = serde_json::from_value(payload["dates"].clone())
        .map_err(|_| "dates must be an array of ISO dates or timestamps")?;
    let values: Vec<f64> = serde_json::from_value(payload["values"].clone())
        .map_err(|_| "values must be a numeric array")?;
    if dates.len() != values.len() {
        return Err("dates and values must be aligned");
    }
    let frequency = payload["frequency"].as_str().unwrap_or("");
    if !["D", "W", "M", "ME", "MS", "QS", "QE", "YS", "YE"].contains(&frequency) {
        return Err("frequency must be D, W, M, ME, MS, QS, QE, YS or YE");
    }
    let start_month = match payload.get("fiscal_start_month") {
        None => 1,
        Some(value) if matches!(frequency, "QS" | "QE" | "YS" | "YE") => value
            .as_u64()
            .filter(|month| (1..=12).contains(month))
            .ok_or("fiscal_start_month must be an integer from 1 through 12")?
            as u32,
        Some(_) => return Err("fiscal_start_month requires a quarter or year frequency"),
    };
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
            start_month,
        );
    }
    let dates = dates
        .iter()
        .map(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d"))
        .collect::<Result<_, _>>()
        .map_err(|_| "invalid ISO calendar date")?;
    aggregate_dates(dates, values, frequency, start_month)
}

fn add_value<K: Ord>(
    sums: &mut BTreeMap<K, f64>,
    bucket: K,
    value: f64,
) -> Result<(), &'static str> {
    if sums.len() == MAX_PERIODS && !sums.contains_key(&bucket) {
        return Err("calendar output exceeds 100000 periods");
    }
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
    start_month: u32,
) -> Result<Value, &'static str> {
    let period_end = matches!(frequency, "M" | "ME" | "QE" | "YE");
    let months = match frequency {
        "M" | "ME" | "MS" => 1,
        "QS" | "QE" => 3,
        "YS" | "YE" => 12,
        _ => 0,
    };
    let Some(&first) = dates.iter().min() else {
        return Ok(json!({"dates":[],"values":[]}));
    };
    let mut sums = BTreeMap::new();
    for (date, value) in dates.into_iter().zip(values) {
        let bucket = match frequency {
            "W" => first.checked_add_days(Days::new((date - first).num_days() as u64 / 7 * 7)),
            _ if months > 0 => {
                let offset = (date.month0() + 13 - start_month) % months;
                date.with_day(1)
                    .expect("first day exists")
                    .checked_sub_months(Months::new(offset))
            }
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
            if labels.len() == MAX_PERIODS {
                return Err("calendar output exceeds 100000 periods");
            }
            let next = if months > 0 {
                date.checked_add_months(Months::new(months))
            } else {
                date.succ_opt()
            };
            let label = if period_end {
                match next {
                    Some(boundary) => boundary
                        .pred_opt()
                        .ok_or("calendar period end outside supported date range")?,
                    None if date
                        .checked_add_months(Months::new(months - 1))
                        .is_some_and(|last_month| {
                            last_month.year() == NaiveDate::MAX.year()
                                && last_month.month() == NaiveDate::MAX.month()
                        }) =>
                    {
                        NaiveDate::MAX
                    }
                    None => return Err("calendar period end outside supported date range"),
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekly_buckets_obey_period_limit() {
        let first = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
        let dates: Vec<_> = (0..=MAX_PERIODS)
            .map(|week| first.checked_add_days(Days::new(week as u64 * 7)).unwrap())
            .collect();
        let mut at_limit = dates[..MAX_PERIODS].to_vec();
        at_limit.push(first);
        let result = aggregate_dates(at_limit, vec![1.; MAX_PERIODS + 1], "W", 1).unwrap();
        let values = result["values"].as_array().unwrap();
        assert_eq!(values.len(), MAX_PERIODS);
        assert_eq!(values[0], 2.);
        assert_eq!(values[MAX_PERIODS - 1], 1.);
        assert_eq!(
            aggregate_dates(dates, vec![1.; MAX_PERIODS + 1], "W", 1),
            Err("calendar output exceeds 100000 periods")
        );
    }

    #[test]
    fn end_label_can_be_the_last_supported_date() {
        for frequency in ["ME", "QE", "YE"] {
            let result = aggregate_dates(vec![NaiveDate::MAX], vec![1.], frequency, 1).unwrap();
            assert_eq!(
                result,
                json!({"dates":[NaiveDate::MAX.to_string()],"values":[1.]})
            );
        }
    }
}
