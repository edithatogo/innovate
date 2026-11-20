use crate::calendar_time::{self, MAX_DATE, Timestamp, supported_date};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use time::{Date, Duration, Month};

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
            timestamps.into_iter().map(|time| time.date()).collect(),
            values,
            frequency,
            start_month,
        );
    }
    let dates = dates
        .iter()
        .map(|date| calendar_time::date(date))
        .collect::<Option<_>>()
        .ok_or("invalid ISO calendar date")?;
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

fn month_start(date: Date, offset: i32) -> Option<Date> {
    let months = date.year() * 12 + i32::from(u8::from(date.month())) - 1 + offset;
    let month = Month::try_from(months.rem_euclid(12) as u8 + 1).ok()?;
    supported_date(Date::from_calendar_date(months.div_euclid(12), month, 1).ok()?)
}

fn aggregate_dates(
    dates: Vec<Date>,
    values: Vec<f64>,
    frequency: &str,
    start_month: u32,
) -> Result<Value, &'static str> {
    let period_end = matches!(frequency, "M" | "ME" | "QE" | "YE");
    let months: i32 = match frequency {
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
            "W" => first.checked_add(Duration::weeks((date - first).whole_weeks())),
            _ if months > 0 => {
                let offset = (i32::from(u8::from(date.month())) + 12 - start_month as i32) % months;
                month_start(date, -offset)
            }
            _ => Some(date),
        }
        .and_then(supported_date)
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
                month_start(date, months)
            } else {
                date.next_day().and_then(supported_date)
            };
            let label = if period_end {
                match next {
                    Some(boundary) => boundary
                        .previous_day()
                        .and_then(supported_date)
                        .ok_or("calendar period end outside supported date range")?,
                    None if month_start(date, months - 1).is_some_and(|last_month| {
                        last_month.year() == MAX_DATE.year()
                            && last_month.month() == MAX_DATE.month()
                    }) =>
                    {
                        MAX_DATE
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
        let first = Date::from_calendar_date(2000, Month::January, 1).unwrap();
        let dates: Vec<_> = (0..=MAX_PERIODS)
            .map(|week| first.checked_add(Duration::weeks(week as i64)).unwrap())
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
            let result = aggregate_dates(vec![MAX_DATE], vec![1.], frequency, 1).unwrap();
            assert_eq!(
                result,
                json!({"dates":[MAX_DATE.to_string()],"values":[1.]})
            );
        }
    }
}
