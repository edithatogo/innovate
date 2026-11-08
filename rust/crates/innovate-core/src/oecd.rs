use serde_json::{Value, json};

pub(crate) fn request(payload: &Value) -> Result<Value, &'static str> {
    let dataset = payload["dataset_id"]
        .as_str()
        .ok_or("dataset_id is required")?;
    let valid_code = |code: &str| {
        !code.is_empty()
            && code
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-@".contains(&b))
    };
    let components: Vec<_> = dataset.split(',').collect();
    if !(2..=3).contains(&components.len()) || !components.iter().all(|part| valid_code(part)) {
        return Err("dataset_id must be an OECD agency,dataflow[,version] reference");
    }
    let dimensions = payload["dimension_key"]
        .as_str()
        .ok_or("dimension_key is required")?;
    if dimensions.is_empty()
        || !dimensions
            .split('.')
            .all(|part| part.is_empty() || part.split('+').all(valid_code))
    {
        return Err("dimension_key must be an ordered SDMX dimension key");
    }
    let period = |field: &str, default: &'static str| {
        if payload[field].is_null() {
            return Ok(default);
        }
        payload[field]
            .as_str()
            .filter(|period| {
                !period.is_empty()
                    && period
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            })
            .ok_or("periods must be SDMX date codes")
    };
    let start = period("start_date", "1960")?;
    let end = period("end_date", "2020")?;
    let url = format!(
        "https://sdmx.oecd.org/public/rest/data/{dataset}/{dimensions}?startPeriod={start}&endPeriod={end}&dimensionAtObservation=AllDimensions&format=csvfile"
    );
    Ok(json!({"url":url,"provider":"OECD","format":"sdmx_csv","start_date":start,"end_date":end}))
}
