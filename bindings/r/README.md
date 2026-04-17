# innovate R bindings

This package provides a thin R-facing adapter over the Python `innovate` functional kernel.

## Invocation path

The package shells out to the kernel bridge script at `inst/python/kernel_bridge.py` and passes
JSON request/response envelopes between R and Python.

The bridge expects the Python source tree to be available at `../src` relative to the repository
root and uses `uv run python` by default. Set `INNOVATE_PYTHON_COMMAND=python3` if you want to
override the launcher.

## Current scope

- Stable kernel discovery
- Versioned request construction
- Thin operation wrappers for the functional kernel

The R layer intentionally avoids duplicating any model logic.
