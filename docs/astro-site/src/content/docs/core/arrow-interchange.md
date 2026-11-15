---
title: Arrow and Polars interchange
description: Understand the current table boundary, IPC file and stream formats, metadata preservation and round-trip checks.
---

The current Python table interface accepts Polars `DataFrame` and `LazyFrame`
objects. It serializes them to Arrow IPC before native execution and reads native
IPC output back into Polars. A lazy input is collected at that boundary. The
current implementation copies and re-encodes buffers; a columnar format alone
does not make this particular call zero-copy.

## Choose the representation deliberately

| Representation | Purpose in Innovate | Important boundary |
| --- | --- | --- |
| Polars table | Convenient input and output for Python table operations | Only metadata represented in the materialized table is available for later export |
| IPC file bytes | Store or exchange a complete table with a file footer | This is a serialization format, not an analysis-provenance record |
| IPC stream bytes | Exchange a schema followed by record batches | The consumer must understand stream framing and schema metadata |
| Native JSON envelope | Retain operation results, errors and structured metadata | Preserve the complete required artifact when replay is intended |

The distinction between IPC files and streams follows the
[Apache Arrow format specification](https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc).
Polars documents its file writer as
[`DataFrame.write_ipc`](https://docs.pola.rs/api/python/stable/reference/api/polars.DataFrame.write_ipc.html).

## Current export functions

[export_ipc_file](/innovate/api/generated/export-ipc-file/) and
[export_ipc_stream](/innovate/api/generated/export-ipc-stream/) accept a Polars
frame or complete IPC file/stream bytes and return bytes in the requested
encoding. They use the native writer and preserve compatible field annotations
and schema metadata present in IPC input. They cannot reconstruct metadata that
was already discarded by an earlier table conversion. Re-encoding preserves
the supported data meaning, not byte identity with the input.

The Julia and R export functions take byte vectors and raw vectors respectively;
their signatures and ownership behavior are in the
[Julia](/innovate/api/generated/julia/) and [R](/innovate/api/generated/r/)
references. Native output buffers are copied into the host language and released
through the binding's ownership rules.

## Row identity and scientific meaning

Prediction retains input columns and ordering and appends output columns. Check
identifiers and time values when joining a returned table to an external table.
For multi-series models, output names such as `prediction_1` follow the model's
series order; they do not independently identify product names, units or a
causal estimand.

Normalized scenario tables and their retained JSON fields serve complementary
purposes. Columns support table operations; the JSON record preserves the
structured execution information required by the checked replay path. The
[round-trip tutorial](/innovate/tutorials/arrow-interchange/) checks both and then
re-executes each saved scenario.

## Failure behavior and limitations

Malformed IPC, unsupported data, output-name collisions, invalid numerical
inputs and incompatible payloads fail at the relevant boundary. Preserve the
exception or structured native error rather than replacing a failure with an
empty table. Exact exception types are documented for each wrapper.

Successful serialization does not establish provenance, licensing, calibration
or cross-version replay compatibility. A table also cannot recover fields omitted
before export. Use the [dataset-artifact tutorial](/innovate/tutorials/data-ingestion-provenance/)
for the separately checked ingestion and checksum contract.

The [earlier interchange page](/innovate/latest/core/arrow-interchange/) remains
available for interpreting historical artifacts; its earlier Python payload
classes are not the current interface.
