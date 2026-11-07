use serde_json::{Value, json};

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    if payload
        .get("schema_version")
        .is_some_and(|version| version != "1.0")
    {
        return Err("unsupported dataset schema version");
    }
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
        if matches!(unit, Some("count" | "rate")) {
            check(
                "nonnegative:adoption",
                adoption.iter().all(|&value| value >= 0.),
                "adoption counts and rates must be nonnegative",
            );
        }
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
    if let Some(labels) = payload.get("product_labels") {
        let valid = labels.as_array().is_some_and(|labels| {
            labels.len() == width
                && labels
                    .iter()
                    .all(|label| label.as_str().is_some_and(|text| !text.trim().is_empty()))
        });
        check(
            "product_labels",
            valid,
            "product labels must align with share columns",
        );
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
    let column = |key: &str| payload[key].as_array().ok_or("edge columns must be arrays");
    let source = column("source")?;
    let target = column("target")?;
    let weights = column("weight")?;
    let aligned = source.len() == target.len() && source.len() == weights.len();
    let labels = source
        .iter()
        .chain(target)
        .all(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()));
    let directed = match payload.get("directed") {
        Some(value) => value.as_bool().ok_or("directed must be a boolean")?,
        None => true, // Preserve ordered keys for historical datasets without this field.
    };
    let mut seen = std::collections::HashSet::new();
    let unique = aligned
        && labels
        && source.iter().zip(target).all(|(a, b)| {
            let (a, b) = (a.as_str(), b.as_str());
            seen.insert(if directed || a <= b { (a, b) } else { (b, a) })
        });
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
            if directed {
                "ordered source and target pairs must be unique"
            } else {
                "undirected source and target pairs must be unique"
            },
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
    let warnings = if labels && source.iter().zip(target).any(|(a, b)| a == b) {
        vec!["self_loops:kairos_network"]
    } else {
        Vec::new()
    };
    Ok(
        json!({"kind":"network_edges","ok":failed.is_empty(),"checks":checks,"failed":failed,"warnings":warnings}),
    )
}

pub(crate) fn ingest_request(request: &Value) -> Value {
    let result = if request["schema_version"] == crate::KERNEL_SCHEMA_VERSION {
        ingest(&request["payload"])
    } else {
        Err(ingestion_error("ingest_dataset requires schema 2.0"))
    };
    let (result, error) = match result {
        Ok(artifact) => (artifact, Value::Null),
        Err(error) => (Value::Null, error),
    };
    json!({"schema_version":crate::KERNEL_SCHEMA_VERSION,"operation":"ingest_dataset",
        "model_key":null,"result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}

fn ingestion_error(message: &str) -> Value {
    json!({"code":"invalid_request","message":message})
}

fn ingest(payload: &Value) -> Result<Value, Value> {
    let (dataset, encoding) = if let Some(artifact) = payload.get("artifact") {
        if payload.get("dataset").is_some() || payload.get("encoding").is_some() {
            return Err(ingestion_error(
                "artifact cannot be combined with dataset or encoding",
            ));
        }
        if artifact["artifact_kind"] != "innovate.dataset"
            || !artifact["checksum_encoding"].is_string()
            || artifact["dataset"]["provenance"]["checksum"]
                .as_str()
                .is_none_or(str::is_empty)
        {
            return Err(ingestion_error(
                "dataset artifact requires its kind, checksum and checksum encoding",
            ));
        }
        (&artifact["dataset"], artifact.get("checksum_encoding"))
    } else {
        (&payload["dataset"], payload.get("encoding"))
    };
    if !dataset["provenance"].is_object() {
        return Err(ingestion_error("ingestion requires dataset provenance"));
    }
    let default = match (payload.get("artifact").is_none(), dataset["kind"].as_str()) {
        (true, Some("policy_timing")) if dataset.get("event_labels").is_none() => {
            dataset["event_times"].as_array().map(|times| {
                (
                    "event_labels",
                    json!(
                        (0..times.len())
                            .map(|i| format!("event_{i}"))
                            .collect::<Vec<_>>()
                    ),
                )
            })
        }
        (true, Some("network_edges")) if dataset.get("weight").is_none() => dataset["source"]
            .as_array()
            .map(|source| ("weight", json!(vec![1.0; source.len()]))),
        _ => None,
    };
    let normalized;
    let dataset = if let Some((field, values)) = default {
        let mut updated = dataset.clone();
        updated[field] = values;
        normalized = updated;
        &normalized
    } else {
        dataset
    };
    let digest = crate::provenance::checksum(dataset, encoding).map_err(ingestion_error)?;
    let report = validate(dataset).map_err(ingestion_error)?;
    if report["ok"] != true {
        let mut error = ingestion_error("dataset diagnostics failed");
        error["validation"] = report;
        return Err(error);
    }
    let mut dataset = dataset.clone();
    dataset["provenance"]["checksum"] = digest["checksum"].clone();
    Ok(json!({
        "artifact_kind":"innovate.dataset",
        "dataset":dataset,
        "validation":report,
        "checksum_encoding":digest["encoding"]
    }))
}
