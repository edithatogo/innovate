---
title: Benchmark Workflows and Model Cards
description: Reproducible benchmark corpus, model cards, promotion dossiers, and opt-in timing evidence.
---

The benchmark suite provides a stable, machine-readable way to compare the
canonical model families shipped with `innovate`. It keeps the benchmark corpus,
model cards, and evaluation outputs synchronized so the library can be used for
scientific comparison and release validation.

## What the suite includes

- a reproducible benchmark corpus with stable case identifiers
- synchronized model cards for stable model families
- fast metadata checks for benchmark contribution review
- a canonical runner that emits comparable metrics, diagnostics, and
  uncertainty summaries
- JSON-friendly artifacts that can be saved and diffed in CI

## Fast validation

Run the fast validation gate before adding or changing benchmark cases:

```python
from innovate.benchmarks import (
    refresh_model_card_summaries,
    validate_benchmark_corpus,
)

report = validate_benchmark_corpus()
report.assert_valid()

summaries = refresh_model_card_summaries()
print(summaries["bass"]["freshness"]["status"])
```

This gate checks required metadata, model-card freshness, and CI policy. It is
intended for normal pull request CI and does not execute timing benchmarks.

## Running the stable suite

```python
from innovate.benchmarks import run_stable_benchmark_suite

suite = run_stable_benchmark_suite()
print(suite.to_dict()["run_count"])

for run in suite.runs:
    print(run.case_id, run.model_key, run.metrics["RMSE"])
```

## Saving benchmark artifacts

```python
from pathlib import Path
from innovate.benchmarks import run_stable_benchmark_suite

output_dir = Path("benchmark-results")
output_dir.mkdir(exist_ok=True)

suite = run_stable_benchmark_suite(model_keys=("bass", "fisher_pry"))
suite.write_json(output_dir / "stable-suite.json")
```

## Interpreting outputs

- `metrics` contains the comparable fit measures for each benchmark run.
- `diagnostics` records the standardized diagnostics contract, including
  support level, warnings, and residual analysis.
- `uncertainty` describes whether the result is deterministic, bootstrap, or
  Bayesian and includes the provenance required to compare runs safely.
- `metadata` captures the stable model card and benchmark case identity so
  artifacts can be traced back to the corpus version used for the run.

## Model-card synchronization

The model-card registry is generated from the stable capability registry, so
each stable family has a consistent machine-readable description.

```python
from innovate.benchmarks import get_model_card, list_model_cards

cards = list_model_cards()
bass = get_model_card("bass")

print(sorted(cards))
print(bass.summary)
print(bass.benchmark_case_ids)
```

## Recommended workflow

1. Run `validate_benchmark_corpus` after editing cases or model cards.
2. Use `workflow_dispatch` for opt-in timing runs.
3. Save the JSON artifact as a release or CI output.
4. Use the model cards to confirm assumptions, outputs, diagnostics, and
   limitations before interpreting the scores.
5. Keep documentation synchronized with code changes so the suite stays
   reproducible and auditable.

## Promotion metadata

Backend and Rust-core promotion candidates must report reference backend timing
separately from accelerated results. XLA compilation cost and XLA steady-state
runtime should be recorded independently so first-call compilation does not get
confused with repeated execution. Cases that require expensive accelerator
timing should use `workflow_dispatch` or scheduled CI instead of the fast
default test path.

## Promotion dossier capture

Each promotion dossier should store the raw artifacts and a short manifest under
one candidate-specific directory, for example
`benchmark-results/promotion/logistic-native/`. Record the candidate operation,
commit, command, hardware target, backend versions, and promotion decision next
to the artifacts so a reviewer can reproduce the run.

Capture Rust-native CPU flamegraph evidence with the packaged profiling script.
The script writes the SVG and a `.metadata.txt` file that records the Rust
toolchain, `cargo-flamegraph` version, git revision, profile environment, and
command:

```bash
INNOVATE_RUST_CPU_PROFILE_OUTPUT=../../benchmark-results/promotion/logistic-native/flamegraph-native-kernels.svg \
  bindings/rust/scripts/profile_native_kernels.sh
```

Capture Rust-native memory evidence with the DHAT wrapper. Use a fixed
iteration count in the dossier so allocation profiles can be compared between
runs:

```bash
INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS=10000 \
INNOVATE_RUST_MEMORY_PROFILE_OUTPUT="$PWD/benchmark-results/promotion/logistic-native/dhat-native-kernels-heap.json" \
  bindings/rust/scripts/profile_memory_native_kernels.sh
```

Capture XLA CPU and GPU benchmark JSON separately. The CPU file is the portable
baseline; the GPU file should come from an accelerator runner with a JAX GPU
install:

```bash
JAX_PLATFORM_NAME=cpu uv run pytest --benchmark-only --benchmark-json=benchmark-results/promotion/logistic-native/benchmark-xla-cpu.json
JAX_PLATFORM_NAME=gpu uv run pytest --benchmark-only --benchmark-json=benchmark-results/promotion/logistic-native/benchmark-xla-gpu.json
```

The dossier should reference these files explicitly:

| Artifact | Example file | Dossier field |
| --- | --- | --- |
| Rust CPU flamegraph | `flamegraph-native-kernels.svg` and `flamegraph-native-kernels.svg.metadata.txt` | Rust-native CPU runtime, toolchain, command, and git revision |
| Rust DHAT memory profile | `dhat-native-kernels-heap.json` | Allocation-sensitive memory behavior and iteration count |
| XLA CPU benchmark | `benchmark-xla-cpu.json` | XLA compile cost and steady-state runtime for the CPU baseline |
| XLA GPU benchmark | `benchmark-xla-gpu.json` | Accelerator target, XLA compile cost, and steady-state runtime |

## GPU and XLA profiling boundary

GPU profiling belongs to the optional JAX/XLA backend today. Rust profiling
covers native CPU hot paths and Rust heap behavior; it should not be used to
claim GPU coverage until Rust owns a promoted GPU execution backend behind the
kernel contract.

Use the existing optional-backend setup before collecting XLA evidence:

```bash
uv sync --extra jax
```

Use a CPU-only XLA baseline when comparing compilation and steady-state costs in
portable CI:

```bash
JAX_PLATFORM_NAME=cpu uv run pytest --benchmark-only --benchmark-json=benchmark-xla-cpu.json
```

Use the same benchmark command on an accelerator runner for GPU evidence, with a
JAX GPU install available in that environment:

```bash
JAX_PLATFORM_NAME=gpu uv run pytest --benchmark-only --benchmark-json=benchmark-xla-gpu.json
```

For Python-side CPU and memory profiling of the optional backend path, use the
same Scalene command as the opt-in benchmark workflow:

```bash
uv run scalene src/innovate --cli --reduced-profile
```

Record the active `JAX_PLATFORM_NAME`, accelerator model, XLA compilation time,
XLA steady-state runtime, and memory behavior where measurable. Keep Rust CPU
and DHAT memory profiles in separate artifacts so Rust-native promotion and
JAX/XLA accelerator promotion are not conflated.

## MARS surrogate benchmark gate

The MARS surrogate benchmark gate is a metadata-first decision gate for the
ecosystem `mars` surrogate package. It keeps `mars` out of base and optional
package metadata while the current outcome is `defer`.

Fast validation checks candidate scenarios, correctness tolerances, promotion
thresholds, failure modes, and the eligible `jax_xla_surrogate_candidate`
without importing or running `mars`. The opt-in dry run writes a small JSON
artifact:

```bash
uv run python -m innovate.benchmarks.mars_surrogate --write-json benchmark-results/mars-surrogate-gate.json
```

Future recorded benchmark evidence must compare NumPy/SciPy reference behavior,
MARS surrogate behavior, and eligible XLA-backed alternatives before any adapter
promotion decision changes.
