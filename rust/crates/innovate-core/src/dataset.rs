use serde_json::{Value, json};

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    let kind = payload["kind"].as_str().unwrap_or("");
    if kind == "network_edges" {
        return validate_edges(payload);
    }
    if matches!(kind, "competition" | "policy_timing") {
        return validate_keyed(payload, kind);
    }
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

fn validate_keyed(payload: &Value, kind: &str) -> Result<Value, &'static str> {
    let competition = kind == "competition";
    let (time_key, value_key, id_keys, duplicate_key, units, default_unit) = if competition {
        (
            "time",
            "value",
            &["unit_id", "product_id"][..],
            "time_unit_product",
            &["count", "share", "rate", "currency"][..],
            "count",
        )
    } else {
        (
            "event_times",
            "event_effects",
            &["event_labels"][..],
            "event_time_label",
            &["effect", "rate", "count"][..],
            "effect",
        )
    };
    let column = |key: &str| {
        payload[key]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or("dataset columns must be nonempty arrays")
    };
    let time = column(time_key)?;
    let values = column(value_key)?;
    let ids: Vec<_> = id_keys
        .iter()
        .map(|key| column(key))
        .collect::<Result<_, _>>()?;
    let mut checks = Vec::new();
    let mut failed = Vec::new();
    let mut warnings = Vec::new();
    let mut check = |name: &str, status: &str, message: &str| {
        checks.push(json!({"name":name,"status":status,"message":message,"details":{}}));
        match status {
            "fail" => failed.push(name.to_owned()),
            "warn" => warnings.push(name.to_owned()),
            _ => {}
        }
    };
    let status = |pass| if pass { "pass" } else { "fail" };
    let aligned = time.len() == values.len() && ids.iter().all(|v| v.len() == time.len());
    check(
        "column_alignment",
        status(aligned),
        "dataset columns must have the same length",
    );
    let times = time.iter().map(Value::as_f64).collect::<Option<Vec<_>>>();
    check(
        &format!("missingness:{time_key}"),
        status(times.is_some()),
        "time observations must be numeric",
    );
    check(
        &format!("missingness:{value_key}"),
        status(values.iter().all(Value::is_number)),
        "outcome observations must be numeric",
    );
    let labels: Option<Vec<Vec<&str>>> = ids
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|v| v.as_str().filter(|s| !s.trim().is_empty()))
                .collect()
        })
        .collect();
    check(
        "identifier_labels",
        status(labels.is_some()),
        "identifiers must be nonempty strings",
    );
    let unit = payload
        .get("unit")
        .map_or(Some(default_unit), Value::as_str);
    check(
        &format!("unit_compatibility:{kind}"),
        status(unit.is_some_and(|u| units.contains(&u))),
        "dataset unit must be compatible with its kind",
    );
    if let Some(times) = times {
        if !competition {
            check(
                "event_time_order",
                if times.windows(2).all(|pair| pair[0] <= pair[1]) {
                    "pass"
                } else {
                    "warn"
                },
                "unsorted event times may be reordered by consumers",
            );
        }
        if let Some(labels) = labels.filter(|_| aligned) {
            let mut seen = std::collections::HashSet::new();
            let unique = times.iter().enumerate().all(|(i, &time)| {
                let bits = if time == 0. { 0 } else { time.to_bits() };
                seen.insert((bits, labels[0][i], competition.then(|| labels[1][i])))
            });
            check(
                &format!("duplicate_keys:{duplicate_key}"),
                status(unique),
                "dataset keys must be unique",
            );
        }
    }
    Ok(
        json!({"kind":kind,"ok":failed.is_empty(),"checks":checks,"failed":failed,"warnings":warnings}),
    )
}

fn validate_edges(payload: &Value) -> Result<Value, &'static str> {
    let column = |key: &str| {
        payload[key]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or("edge columns must be nonempty arrays")
    };
    let source = column("source")?;
    let target = column("target")?;
    let weights = column("weight")?;
    let aligned = source.len() == target.len() && source.len() == weights.len();
    let labels = source
        .iter()
        .chain(target)
        .all(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()));
    let mut seen = std::collections::HashSet::new();
    let unique = aligned
        && labels
        && source
            .iter()
            .zip(target)
            .all(|(a, b)| seen.insert((a.as_str(), b.as_str())));
    let unit = payload.get("unit").map_or(Some("weight"), Value::as_str);
    let checks = [
        (
            "column_alignment",
            aligned,
            "edge columns must have the same length",
        ),
        (
            "identifier_labels",
            labels,
            "edge endpoints must be nonempty strings",
        ),
        (
            "missingness:weight",
            weights.iter().all(Value::is_number),
            "edge weights must be numeric",
        ),
        (
            "weight_bounds",
            weights.iter().all(|v| v.as_f64().is_some_and(|w| w >= 0.)),
            "edge weights must be nonnegative",
        ),
        (
            "duplicate_keys:source_target",
            unique,
            "ordered source and target pairs must be unique",
        ),
        (
            "unit_compatibility:network_edges",
            matches!(unit, Some("weight" | "count" | "rate")),
            "network edge unit must be weight, count or rate",
        ),
    ];
    let failed: Vec<_> = checks
        .iter()
        .filter_map(|&(name, pass, _)| (!pass).then_some(name))
        .collect();
    let checks: Vec<_> = checks.into_iter().map(|(name, pass, message)| {
        json!({"name":name,"status":if pass {"pass"} else {"fail"},"message":message,"details":{}})
    }).collect();
    Ok(
        json!({"kind":"network_edges","ok":failed.is_empty(),"checks":checks,"failed":failed,"warnings":[]}),
    )
}

pub(crate) fn ingest(payload: &Value) -> Result<Value, &'static str> {
    let mut dataset = payload["dataset"].clone();
    if !dataset["provenance"].is_object() {
        return Err("ingestion requires dataset provenance");
    }
    let digest = crate::provenance::checksum(payload)?;
    let report = validate(&dataset)?;
    if report["ok"] != true {
        return Err("dataset diagnostics failed; inspect validate_dataset for details");
    }
    dataset["provenance"]["checksum"] = digest["checksum"].clone();
    Ok(json!({
        "artifact_kind":"innovate.dataset",
        "dataset":dataset,
        "validation":report,
        "checksum_encoding":digest["encoding"]
    }))
}
