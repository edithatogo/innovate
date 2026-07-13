# Rust Binding Snippet

```rust
use innovate::{fit_model, predict_model, PredictOptions, TablePayload};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = fit_model(
        "bass",
        TablePayload::from_columns([
            ("time", vec![1.0, 2.0, 3.0, 4.0]),
            ("adoption", vec![3.0, 8.0, 15.0, 25.0]),
        ]),
    )?;

    let _predictions = predict_model(
        &model,
        PredictOptions {
            horizon: 6,
            schema_version: "1.0".to_string(),
        },
    )?;

    Ok(())
}
```
