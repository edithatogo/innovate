"""Generate the historical SARIMAX initialization oracle in an isolated environment."""

import json
import sys

import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

values = [2.0, 5.0, 3.0, 5.0, 8.0, 6.0, 8.0, 21.0, 9.0, 11.0, 14.0, 12.0]
seasonal_order = (0, 1, 0, 3)
cases = []
for order, params in [((0, 1, 0), [1.0]), ((1, 1, 0), [0.45, 1.0])]:
    result = SARIMAX(values, order=order, seasonal_order=seasonal_order).filter(params)
    cases.append({
        "order": order,
        "seasonal_order": seasonal_order,
        "coefficients": params[:-1],
        "innovation_variance": params[-1],
        "burn": int(result.loglikelihood_burn),
        "fitted": result.fittedvalues.tolist(),
    })

with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump({"statsmodels": statsmodels.__version__, "values": values, "cases": cases},
              output, indent=2, allow_nan=False)
    output.write("\n")
