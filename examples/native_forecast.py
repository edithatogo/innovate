"""Native Polars prediction with explicit model parameters."""
import polars as pl
from innovate import predict

print(predict("logistic", pl.DataFrame({"time": [0.0, 1.0, 2.0]}), {"L": 100.0, "k": 1.0, "x0": 1.0}))
