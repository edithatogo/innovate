import json
import math
import random
import runpy
import struct
import sys
from itertools import product
from pathlib import Path

repo = Path(sys.argv[1])
reference = runpy.run_path(repo / "tests/native/test_python_consumer.py")["_exact_adopter_categories"]
smallest = math.ulp(0.0)
largest = float.fromhex("0x1.fffffffffffffp+1023")
weights = [0.0, smallest, 1.0, largest]
cases = []


def add_case(group, time, rates):
    cases.append(
        {
            "name": f"{group}-{len(cases) + 1}",
            "time": time,
            "rates": rates,
            "time_bits": [struct.pack(">d", value).hex() for value in time],
            "rate_bits": [struct.pack(">d", value).hex() for value in rates],
            "categories": reference(time, rates),
        }
    )


for time in ([-3.0, 0.0, 1.0, 4.0], [0.0, 1.0, 3.0, 8.0], [-largest, 0.0, smallest, largest]):
    for rates in product(weights, repeat=len(time)):
        if any(rates):
            add_case("signed-wheel", time, rates)
assert len(cases) == 765

rng = random.Random(401)
rate_scales = [0.0, smallest, 2.0**-1022, 0.125, 1.0, 31.0, 2.0**+1020, largest]
for length in (1, 2, 3, 7, 16, 31, 65):
    for scale in (2.0**-1074, 2.0**-53, 1.0, 2.0**+1000):
        for _ in range(12):
            time = [sample * scale for sample in sorted(rng.sample(range(-300, 301), length))]
            rates = [rng.choice(rate_scales) for _ in time]
            if not any(rates):
                rates[-1] = 1.0
            add_case("irregular-grid", time, rates)

invalid = [
    {"name": "empty", "time": [], "rates": [], "error": "nonempty, finite and aligned"},
    {"name": "misaligned", "time": [1, 2], "rates": [1], "error": "nonempty, finite and aligned"},
    {"name": "unordered", "time": [2, 1], "rates": [1, 1], "error": "strictly increasing"},
    {"name": "duplicate", "time": [2, 2], "rates": [1, 1], "error": "strictly increasing"},
    {"name": "negative", "time": [1, 2], "rates": [1, -1], "error": "nonnegative with positive total"},
    {"name": "zero-weight", "time": [1, 2], "rates": [0, 0], "error": "nonnegative with positive total"},
]
fit_time = list(range(21))
sample_time = [-10, 0, 5, 9, 11, 15, 20, 30]
exact_levels = [100 / (1 + math.exp(10 - t)) for t in fit_time]
exact_rates = [100 * math.exp(10 - t) / (1 + math.exp(10 - t)) ** 2 for t in sample_time]
report = {
    "valid": cases,
    "invalid": invalid,
    "logistic_fit": {
        "time": fit_time,
        "observed": exact_levels,
        "initial": {"L": 98, "k": 0.95, "x0": 9.75},
        "prediction_time": sample_time,
        "analytic_rates": exact_rates,
        "categories": reference(sample_time, exact_rates),
    },
}
Path(sys.argv[2]).write_text(json.dumps(report, allow_nan=False) + "\n", encoding="utf-8")
print(f"{len(cases)} reference cases and {len(invalid)} invalid inputs")
