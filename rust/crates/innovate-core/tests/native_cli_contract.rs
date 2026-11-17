use serde_json::{Value, json};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("innovate-native-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn invoke<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(env!("CARGO_BIN_EXE_innovate-native"))
        .args(args)
        .output()
        .expect("run the built native CLI")
}

fn request(command: &str, input: &Path, output: &Path) -> Output {
    invoke([
        OsStr::new(command),
        OsStr::new("--request"),
        input.as_os_str(),
        OsStr::new("--output"),
        output.as_os_str(),
    ])
}

fn assert_failure(output: &Output, code: i32, message: &str) {
    assert_eq!(output.status.code(), Some(code));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(message),
        "expected {message:?} in {stderr:?}"
    );
}

#[test]
fn native_cli_validates_requests_and_returns_native_results() {
    let dir = TestDirectory::new();
    let input = dir.0.join("request.json");
    let output = dir.0.join("response.json");

    assert_failure(&invoke::<_, &str>([]), 2, "usage: innovate-native");
    assert_failure(
        &invoke(["predict", "--unknown", "x", "--output", "y"]),
        2,
        "unknown or incomplete argument",
    );
    assert_failure(
        &invoke(["predict", "--request", "x", "--request", "y"]),
        2,
        "missing --output",
    );
    assert_failure(&request("predict", &input, &output), 1, "failed to read");

    fs::write(&input, "{").unwrap();
    assert_failure(&request("predict", &input, &output), 1, "malformed JSON");
    fs::write(&input, "[]").unwrap();
    assert_failure(&request("predict", &input, &output), 1, "must be an object");
    fs::write(&input, json!({"operation": "fit_model"}).to_string()).unwrap();
    assert_failure(
        &request("predict", &input, &output),
        1,
        "does not allow operation",
    );
    let marker = "PRIVATE_REQUEST_OPERATION_83972";
    fs::write(&input, json!({"operation": marker}).to_string()).unwrap();
    let rejected = request("predict", &input, &output);
    assert_failure(&rejected, 1, "predict_model");
    assert!(!String::from_utf8_lossy(&rejected.stderr).contains(marker));
    assert!(!output.exists());

    let prediction = json!({
        "model_key": "bass",
        "payload": {
            "inputs": {"time": [0.0, 1.0]},
            "parameters": {"p": 0.03, "q": 0.38, "m": 1000.0}
        }
    });
    fs::write(&input, prediction.to_string()).unwrap();
    assert_failure(&request("predict", &input, &dir.0), 1, "failed to write");
    let result = request("predict", &input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let response: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    let values = response["result"]["values"].as_array().unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].as_f64().unwrap(), 0.0);
    let decay = (-0.41_f64).exp();
    let expected = 1000.0 * (1.0 - decay) / (1.0 + (0.38 / 0.03) * decay);
    assert!((values[1].as_f64().unwrap() - expected).abs() < 1e-9);

    fs::write(
        &input,
        json!({"model_key": "nonexistent_model", "payload": {}}).to_string(),
    )
    .unwrap();
    let result = request("predict", &input, &output);
    assert_eq!(result.status.code(), Some(1));
    assert!(!result.stderr.is_empty());
    let response: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert!(!response["error"].is_null());

    fs::write(&input, json!({
        "model_key": "bass",
        "payload": {"inputs": {"time": [0.0, 1.0, 2.0, 3.0, 4.0], "observed": [0.0, 10.0, 30.0, 60.0, 85.0]}}
    }).to_string()).unwrap();
    let result = request("fit", &input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let response: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert!(
        response["result"]["parameters"]["p"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
}
