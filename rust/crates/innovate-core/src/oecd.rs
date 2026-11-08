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

pub(crate) fn fetch(value: &Value) -> Value {
    let (result, error) = if value["schema_version"] != crate::KERNEL_SCHEMA_VERSION {
        (
            Value::Null,
            json!({"code":"invalid_request","message":"OECD requests require schema 2.0"}),
        )
    } else {
        match request(&value["payload"]) {
            Err(message) => (
                Value::Null,
                json!({"code":"invalid_request","message":message}),
            ),
            Ok(metadata) => match download(metadata["url"].as_str().unwrap()) {
                Ok(csv) => (json!({"request":metadata,"csv":csv}), Value::Null),
                Err(message) => (
                    Value::Null,
                    json!({"code":"upstream_error","message":message}),
                ),
            },
        }
    };
    json!({"schema_version":crate::KERNEL_SCHEMA_VERSION,"operation":"fetch_oecd_dataset",
        "model_key":null,"result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}

fn download(url: &str) -> Result<String, String> {
    use std::{io::Read, time::Duration};

    const MAX_BYTES: u64 = 16 * 1024 * 1024;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())?;
    let response = client.get(url).send().map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("OECD returned HTTP {}", response.status()));
    }
    if response
        .content_length()
        .is_some_and(|size| size > MAX_BYTES)
    {
        return Err("OECD response exceeds 16 MiB".into());
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("OECD response exceeds 16 MiB".into());
    }
    String::from_utf8(bytes).map_err(|_| "OECD response is not UTF-8".into())
}

#[cfg(test)]
mod tests {
    use super::download;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    fn serve(response: &'static [u8]) -> Result<String, String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("OECD fixture connection failed: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            stream.write_all(response).unwrap();
        });
        let result = download(&format!("http://{address}/data"));
        server.join().unwrap();
        result
    }

    #[test]
    fn reports_upstream_http_errors_and_does_not_follow_redirects() {
        for (response, status) in [
            (&b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\n\r\n"[..], "429"),
            (&b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/\r\nContent-Length: 0\r\n\r\n"[..], "302"),
        ] {
            assert!(serve(response).unwrap_err().contains(status));
        }
    }

    #[test]
    fn rejects_oversized_and_invalid_utf8_responses() {
        assert_eq!(
            serve(b"HTTP/1.1 200 OK\r\nContent-Length: 16777217\r\n\r\n").unwrap_err(),
            "OECD response exceeds 16 MiB"
        );
        assert_eq!(
            serve(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n\xff").unwrap_err(),
            "OECD response is not UTF-8"
        );
    }
}
