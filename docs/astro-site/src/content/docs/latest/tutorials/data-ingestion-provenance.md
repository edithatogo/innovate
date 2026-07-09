---
title: Data Ingestion and Provenance
description: Dataset contracts, validation diagnostics, local ingestion, and public-data adapter safeguards.
---

# Data Ingestion and Provenance

Innovate provides fail-closed dataset contracts so adoption, substitution,
competition, policy timing, and network edge tables enter modeling workflows
with validated schemas and explicit provenance.

## Policy

| Rule | Behavior |
|------|----------|
| Unknown license | **Rejected** |
| Missing provenance on ingest | **Rejected** |
| Restricted external datasets | **Not bundled** |
| Local formats | CSV, Parquet, Arrow/Feather (pandas+pyarrow) |
| Polars | Optional; supported when installed |

## Local ingestion

```python
from innovate.data import DatasetProvenance, ingest_local

provenance = DatasetProvenance.create(
    source="local-lab-export",
    license="CC-BY-4.0",
    citation="Lab notebook export 2026-07-01",
    transform_steps=("export",),
)

dataset, report = ingest_local(
    "adoption.csv",
    "adoption",
    provenance=provenance,
)
assert report.ok
print(dataset.provenance.checksum)
```

### Expected columns

| Kind | Required columns |
|------|------------------|
| `adoption` | `time`, `adoption` (+ optional `denominator`) |
| `substitution` | `time`, `share` or `share_*` |
| `competition` | `time`, `unit_id`, `product_id`, `value` |
| `policy_timing` | `event_times`, `event_effects` (+ optional `event_labels`) |
| `network_edges` | `source`, `target` (+ optional `weight`) |

## Public-data-style adapter

```python
from innovate.data import get_builtin_adapter, list_builtin_adapters

print([m.adapter_id for m in list_builtin_adapters()])
adapter = get_builtin_adapter("synthetic_adoption_v1")
dataset, report = adapter.ingest(periods=12, seed=1)
assert report.ok
```

Adapters must declare license and citation in a manifest. The built-in
synthetic adapter only generates open fixture rows; it does not scrape or
bundle restricted sources.

## Validation diagnostics

```python
from innovate.data import validate_dataset

report = validate_dataset(dataset)
print(report.to_dict()["failed"])
```

Checks include missingness, monotonicity (adoption), time alignment,
denominator consistency, duplicate keys, and unit compatibility.

## Benchmarks and scenarios

```python
from innovate.data import integration_bundle, dataset_to_baseline_scenario

bundle = integration_bundle(dataset, report)
scenario = dataset_to_baseline_scenario(dataset, name="baseline-from-data")
```

## Limitations

- Polars is optional; base installs use pandas + pyarrow.
- OECD helper (`get_oecd_dataset`) remains a separate legacy utility and still
  requires its own licensing review before production use.
- Network and policy contracts integrate with existing model input types but do
  not download external graphs or event calendars.
