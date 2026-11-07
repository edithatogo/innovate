import json
import sys
from pathlib import Path

import innovate

model, filename, iterations = sys.argv[1:]
fixture = json.loads(Path(f"rust/crates/innovate-core/tests/fixtures/{filename}_scipy.json").read_text())
case = fixture.get("cases", {}).get("corrected", fixture)
options = {"covariates": ["price"]}
if model == "lock_in":
    options["initial_values"] = fixture["initial"]
else:
    options["n_products"] = 2
fit = innovate.execute(
    "fit_model", model,
    inputs={"time": fixture["time"], "observed": case["rows"], "covariates": fixture["covariates"]},
    constructor_kwargs=options, fit_options={"max_iterations": int(iterations)},
)["result"]
forecast = innovate.execute("predict_model", model, inputs={"time": fixture["forecast_time"]}, state=fit["state"])["result"]
error = max(abs(a-b) for row, expected in zip(forecast["rows"], case["forecast_rows"], strict=True)
            for a,b in zip(row, expected, strict=True))
print(json.dumps({"model": model, "optimizer": fit["diagnostics"]["optimizer"],
                  "rss": fit["diagnostics"]["metrics"]["RSS"], "forecast_max_absolute_error": error}, indent=2))
