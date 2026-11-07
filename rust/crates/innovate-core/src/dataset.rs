use serde_json::{Value, json};

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    let kind = payload["kind"].as_str().unwrap_or("");
    if !matches!(kind, "adoption" | "substitution") {
        return Err("unsupported dataset diagnostic kind");
    }
    let values = |key: &str| {
        payload[key]
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or("dataset columns must be nonempty arrays")
    };
    let time = values("time")?;
    let observation_key = if kind == "adoption" {
        "adoption"
    } else {
        "share"
    };
    let observations = values(observation_key)?;
    let denominator = payload
        .get("denominator")
        .filter(|value| !value.is_null())
        .map(|_| values("denominator"))
        .transpose()?;
    let mut checks = Vec::new();
    let mut failed = Vec::new();
    let mut check = |name: &str, pass: bool, message: &str| {
        checks.push(json!({"name":name,"status":if pass {"pass"} else {"fail"},
            "message":message,"details":{}}));
        if !pass {
            failed.push(name.to_owned());
        }
    };
    let aligned =
        time.len() == observations.len() && denominator.is_none_or(|d| d.len() == time.len());
    check(
        "column_alignment",
        aligned,
        "dataset columns must have the same length",
    );
    for (key, column) in [
        ("time", Some(time)),
        (
            observation_key,
            (kind == "adoption").then_some(observations),
        ),
        ("denominator", denominator),
    ] {
        if let Some(column) = column {
            check(
                &format!("missingness:{key}"),
                column.iter().all(Value::is_number),
                "column must contain finite numeric observations",
            );
        }
    }
    let numeric = |column: &[Value]| column.iter().map(Value::as_f64).collect::<Option<Vec<_>>>();
    if let Some(time) = numeric(time) {
        check(
            "time_alignment",
            time.windows(2).all(|pair| pair[0] < pair[1]),
            "time values must be strictly increasing",
        );
        let mut sorted = time;
        sorted.sort_by(f64::total_cmp);
        check(
            "duplicate_keys:time",
            sorted.windows(2).all(|pair| pair[0] != pair[1]),
            "time keys must be unique",
        );
    }
    if kind == "substitution" {
        substitution(payload, observations, &mut check);
        return Ok(
            json!({"kind":kind,"ok":failed.is_empty(),"checks":checks,"failed":failed,"warnings":[]}),
        );
    }
    let unit = payload.get("unit").map_or(Some("count"), Value::as_str);
    check(
        "unit_compatibility:adoption",
        matches!(unit, Some("count" | "rate" | "share")),
        "adoption unit must be count, rate or share",
    );
    if let Some(adoption) = numeric(observations) {
        check(
            "monotonicity:adoption",
            adoption.windows(2).all(|pair| pair[1] - pair[0] >= -1e-12),
            "adoption must be nondecreasing",
        );
        if unit == Some("share") {
            check(
                "share_bounds:adoption",
                adoption
                    .iter()
                    .all(|value| (0. ..=1. + 1e-9).contains(value)),
                "adoption shares must be between zero and one",
            );
        }
        if let Some(denominator) = denominator.and_then(|column| numeric(column)) {
            check(
                "denominator_consistency",
                aligned
                    && adoption
                        .iter()
                        .zip(denominator)
                        .all(|(&level, limit)| limit > 0. && level <= limit + 1e-9),
                "denominators must be positive and at least as large as adoption",
            );
        }
    }
    Ok(
        json!({"kind":"adoption","ok":failed.is_empty(),"checks":checks,"failed":failed,"warnings":[]}),
    )
}

fn substitution(payload: &Value, shares: &[Value], check: &mut impl FnMut(&str, bool, &str)) {
    let unit = payload.get("unit").map_or(Some("share"), Value::as_str);
    check(
        "unit_compatibility:substitution",
        unit == Some("share"),
        "substitution unit must be share",
    );
    let width = shares[0].as_array().map_or(0, Vec::len);
    let rectangular = width > 0
        && shares
            .iter()
            .all(|row| row.as_array().is_some_and(|cells| cells.len() == width));
    check(
        "share_shape",
        rectangular,
        "shares must form a nonempty rectangular matrix",
    );
    if !rectangular {
        return;
    }
    let rows = shares.iter().filter_map(Value::as_array);
    let complete = rows.clone().flatten().all(Value::is_number);
    check(
        "missingness:share",
        complete,
        "shares must contain finite numeric observations",
    );
    if complete {
        check(
            "share_bounds",
            rows.clone().flatten().all(|value| {
                value
                    .as_f64()
                    .is_some_and(|v| (0. ..=1. + 1e-9).contains(&v))
            }),
            "individual shares must be between zero and one",
        );
        check(
            "share_sum",
            rows.into_iter()
                .all(|row| row.iter().filter_map(Value::as_f64).sum::<f64>() <= 1. + 1e-9),
            "row share sums must not exceed one",
        );
    }
}
