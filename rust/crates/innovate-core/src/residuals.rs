use serde_json::{Value, json};

pub(crate) fn residual_series(payload: &Value) -> Result<Value, &'static str> {
    let series = |key: &str| {
        serde_json::from_value::<Vec<f64>>(payload[key].clone())
            .map_err(|_| "residual diagnostics require numeric time and residual arrays")
    };
    let time = series("time")?;
    let residuals = series("residuals")?;
    if time.is_empty() || time.len() != residuals.len() {
        return Err("time and residuals must be nonempty and aligned");
    }
    let lags = if payload["lags"].is_null() {
        30
    } else {
        payload["lags"]
            .as_u64()
            .filter(|lag| *lag <= 1024)
            .ok_or("lags must be an integer between 0 and 1024")? as usize
    };
    let lags = lags.min(residuals.len() - 1);
    let scale = residuals.iter().map(|v| v.abs()).fold(0., f64::max);
    let count = residuals.len() as f64;
    let scaled: Vec<_> = residuals
        .iter()
        .map(|v| if scale == 0. { 0. } else { v / scale })
        .collect();
    let mean = scaled.iter().sum::<f64>() / count;
    let centered: Vec<_> = scaled.iter().map(|v| v - mean).collect();
    let variance: f64 = centered.iter().map(|v| v * v).sum();
    let mut acf = vec![Some(1.)];
    let mut pacf = vec![Some(1.)];
    let mut coefficients: Vec<f64> = Vec::new();
    let mut prediction_variance = 1.;
    for lag in 1..=lags {
        let correlation = (variance > 0.).then(|| {
            centered
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / variance
        });
        acf.push(correlation);
        let partial = correlation
            .filter(|_| prediction_variance > f64::EPSILON)
            .map(|correlation| {
                let explained = coefficients
                    .iter()
                    .zip(acf[1..lag].iter().rev())
                    .map(|(coefficient, correlation)| coefficient * correlation.unwrap_or_default())
                    .sum::<f64>();
                ((correlation - explained) / prediction_variance).clamp(-1., 1.)
            });
        pacf.push(partial);
        if let Some(partial) = partial {
            coefficients = coefficients
                .iter()
                .zip(coefficients.iter().rev())
                .map(|(forward, reverse)| forward - partial * reverse)
                .chain(std::iter::once(partial))
                .collect();
            prediction_variance *= 1. - partial * partial;
        }
    }
    Ok(
        json!({"time":time,"residuals":residuals,"lags":(0..=lags).collect::<Vec<_>>(),
        "acf":acf,"pacf":pacf,"pacf_method":"durbin_levinson_biased_autocovariance"}),
    )
}

pub(crate) fn render_svg(payload: &Value) -> Result<Value, &'static str> {
    use std::fmt::Write;
    let config = &payload["config"];
    let mut request = payload.clone();
    if request["lags"].is_null() && !config["lags"].is_null() {
        request["lags"] = config["lags"].clone();
    }
    let series = residual_series(&request)?;
    let acf_only = config["acf_only"].as_bool().unwrap_or(false);
    let height = if acf_only { 320 } else { 460 };
    let (display_width, display_height) = if config["figsize"].is_null() {
        (640., f64::from(height))
    } else {
        let (width, explicit_height): (f64, Option<f64>) =
            serde_json::from_value(config["figsize"].clone())
                .map_err(|_| "figsize must contain width and optional height in inches")?;
        let display_width = width * 96.;
        let display_height =
            explicit_height.map_or(display_width * f64::from(height) / 640., |h| h * 96.);
        if [display_width, display_height]
            .iter()
            .any(|value| !(1. ..=16384.).contains(value))
        {
            return Err("figure dimensions must be between 1 and 16384 pixels");
        }
        (display_width, display_height)
    };
    let title = config["title"]
        .as_str()
        .unwrap_or("Residual diagnostics")
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 640 {height}" role="img"><title>{title}</title><text x="32" y="24">{title}</text>"#
    );
    for (panel, (key, label, x_key)) in [
        ("residuals", "Residuals over time", "time"),
        ("acf", "Autocorrelation", "lags"),
        ("pacf", "Partial autocorrelation", "lags"),
    ]
    .into_iter()
    .take(if acf_only { 2 } else { 3 })
    .enumerate()
    {
        let color_key = ["color_residuals", "color_acf", "color_pacf"][panel];
        let color = if config[color_key].is_null() {
            ["#1f77b4", "#ff7f0e", "#2ca02c"][panel]
        } else {
            plot_color(
                config[color_key]
                    .as_str()
                    .ok_or("plot colors must be strings")?,
            )?
        };
        let top = 48. + 140. * panel as f64;
        let xs = series[x_key].as_array().unwrap();
        let ys = series[key].as_array().unwrap();
        let x_scale = xs
            .iter()
            .filter_map(Value::as_f64)
            .map(f64::abs)
            .fold(1., f64::max);
        let (min_x, max_x) = xs
            .iter()
            .filter_map(Value::as_f64)
            .map(|x| x / x_scale)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        let y_scale = ys
            .iter()
            .filter_map(Value::as_f64)
            .map(f64::abs)
            .fold(1., f64::max);
        let mid = top + 60.;
        write!(svg, r#"<g aria-label="{label}"><text x="32" y="{top}">{label}</text><path d="M32 {mid}H608" stroke="gray"/>"#).unwrap();
        for (x, y) in xs.iter().zip(ys) {
            if let (Some(x), Some(y)) = (x.as_f64(), y.as_f64()) {
                let x = if max_x > min_x {
                    32. + 576. * ((x / x_scale - min_x) / (max_x - min_x))
                } else {
                    320.
                };
                let y = mid - 44. * (y / y_scale);
                if panel == 0 {
                    write!(
                        svg,
                        r#"<circle cx="{x:.2}" cy="{y:.2}" r="2" fill="{color}"/>"#
                    )
                    .unwrap();
                } else {
                    write!(svg, r#"<path d="M{x:.2} {mid}V{y:.2}" stroke="{color}"/>"#).unwrap();
                }
            }
        }
        svg.push_str("</g>");
    }
    svg.push_str("</svg>");
    Ok(json!({"format":"svg","svg":svg,"series":series}))
}

fn plot_color(color: &str) -> Result<&str, &'static str> {
    if let [b'C', index @ b'0'..=b'9'] = color.as_bytes() {
        return Ok([
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ][usize::from(index - b'0')]);
    }
    if color.strip_prefix('#').is_some_and(|hex| {
        matches!(hex.len(), 3 | 4 | 6 | 8) && hex.bytes().all(|digit| digit.is_ascii_hexdigit())
    }) || (!color.is_empty() && color.bytes().all(|letter| letter.is_ascii_alphabetic()))
    {
        return Ok(color);
    }
    Err("plot colors must be SVG color names, hex codes or palette indices C0 through C9")
}
