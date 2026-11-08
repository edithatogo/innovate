"""Generate equivalent Polars IPC file and stream fixtures for Julia."""

import argparse
from pathlib import Path

import polars as pl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

frame = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "values": [[1.5, None], None, [], [2.5]],
        "label": ["β", None, "", "policy"],
    },
    schema={"id": pl.Int32, "values": pl.List(pl.Float64), "label": pl.String},
)
frame = pl.concat([frame.slice(0, 2), frame.slice(2)], rechunk=False)
compatibility = pl.CompatLevel.oldest()
frame.write_ipc(args.output / "nested.arrow", compression="uncompressed", compat_level=compatibility)
frame.write_ipc_stream(
    args.output / "nested.stream", compression="uncompressed", compat_level=compatibility
)
