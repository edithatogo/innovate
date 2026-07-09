# Data Ingestion and Provenance Connectors

## Overview

Add robust data ingestion and provenance workflows so users can bring adoption,
market-share, policy, and network datasets into `innovate` with validated
schemas, reproducible metadata, and clear licensing/provenance records.

## Functional Requirements

- Define dataset contracts for adoption curves, substitution shares,
  competition panels, policy timing, and network edges.
- Add ingestion helpers for CSV, Parquet/Arrow, Polars frames, and documented
  external dataset adapters.
- Record provenance metadata: source, license, extraction time, transform steps,
  schema version, checksum, and citation.
- Add validation reports for missingness, monotonicity, time alignment,
  denominator consistency, duplicate keys, and unit compatibility.
- Integrate data contracts with benchmark cases, scenario workflows, and model
  cards.

## Non-Functional Requirements

- Prefer Polars and Arrow-compatible interfaces.
- Do not bundle restricted external datasets.
- Fail closed when provenance or licensing is unknown.

## Acceptance Criteria

- Dataset contracts and validation reports are tested.
- Ingestion examples exist for local files and at least one public-data-style
  adapter.
- Provenance artifacts are schema-tested and documented.
- Release evidence records supported data ingestion formats and limitations.

## Out Of Scope

- Hosting datasets.
- Scraping sources without explicit permission.
