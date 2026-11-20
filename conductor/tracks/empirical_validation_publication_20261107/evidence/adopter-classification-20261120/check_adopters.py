import json
import sys
from pathlib import Path

import polars as pl

import innovate

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in report["valid"]:
    time, rates = case["time"], case["rates"]
    frame = pl.DataFrame({"row_id": list(range(len(time))), "time": time, "adoption_rate": rates})
    result = innovate.categorize_adopters(frame)
    assert result.select(frame.columns).equals(frame), case["name"]
    scalar = innovate.execute("categorize_adopters", time=time, adoption_rate=rates)["result"]
    assert scalar["time"] == time and scalar["adoption_rate"] == rates, case["name"]
    assert result["category"].to_list() == scalar["category"] == case["categories"], case["name"]

for case in report["invalid"]:
    try:
        innovate.execute("categorize_adopters", time=case["time"], adoption_rate=case["rates"])
    except innovate.NativeError as error:
        assert case["error"] in str(error), (case["name"], str(error))
    else:
        raise AssertionError(f"{case['name']} returned a successful response")

case = report["logistic_fit"]
fit = innovate.fit(
    "logistic",
    pl.DataFrame({"time": case["time"], "observed": case["observed"]}),
    fit_options={"initial_parameters": case["initial"]},
)
frame = pl.DataFrame({"time": case["prediction_time"]})
classified = fit.categorize_adopters(frame)
assert all(abs(a - b) <= 1e-8 for a, b in zip(classified["rate"], case["analytic_rates"], strict=True))
assert classified["category"].to_list() == case["categories"]
print(
    f"Python {sys.version.split()[0]}: {len(report['valid'])} reference cases, "
    f"{len(report['invalid'])} error cases and the fitted Logistic case passed"
)
