use serde_json::{Value, json};

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    if payload["kind"] != "adoption" {
        return Err("unsupported dataset diagnostic kind");
    }
    let values = |key: &str| {
        payload[key]
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or("dataset columns must be nonempty arrays")
    };
    let time = values("time")?;
    let adoption = values("adoption")?;
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
    let aligned = time.len() == adoption.len() && denominator.is_none_or(|d| d.len() == time.len());
    check(
        "column_alignment",
        aligned,
        "dataset columns must have the same length",
    );
    for (key, column) in [
        ("time", Some(time)),
        ("adoption", Some(adoption)),
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
    let unit = payload.get("unit").map_or(Some("count"), Value::as_str);
    check(
        "unit_compatibility:adoption",
        matches!(unit, Some("count" | "rate" | "share")),
        "adoption unit must be count, rate or share",
    );
    if let Some(adoption) = numeric(adoption) {
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
