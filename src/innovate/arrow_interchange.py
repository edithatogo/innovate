"""Arrow-compatible interchange helpers for kernel payloads and metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pyarrow as pa

from .kernel import (
    KERNEL_SCHEMA_VERSION,
    KernelArrayPayload,
    KernelDiscoveryRecord,
    KernelDiscoveryResponse,
    KernelTablePayload,
)

ARROW_INTERCHANGE_SCHEMA_VERSION = KERNEL_SCHEMA_VERSION
ARROW_INTERCHANGE_KIND_KEY = "innovate.kind"
ARROW_INTERCHANGE_SCHEMA_VERSION_KEY = "innovate.schema_version"
ARROW_INTERCHANGE_SHAPE_KEY = "innovate.shape"
ARROW_INTERCHANGE_DTYPE_KEY = "innovate.dtype"
ARROW_INTERCHANGE_COLUMNS_KEY = "innovate.columns"
ARROW_INTERCHANGE_ARRAY_KIND = "array"
ARROW_INTERCHANGE_TABLE_KIND = "table"
ARROW_INTERCHANGE_DISCOVERY_KIND = "discovery"
ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN = "value"

_CONTRACT_KEYS = {
    ARROW_INTERCHANGE_KIND_KEY,
    ARROW_INTERCHANGE_SCHEMA_VERSION_KEY,
    ARROW_INTERCHANGE_SHAPE_KEY,
    ARROW_INTERCHANGE_DTYPE_KEY,
    ARROW_INTERCHANGE_COLUMNS_KEY,
}
_IMMUTABLE_KEYS = {
    ARROW_INTERCHANGE_KIND_KEY,
    ARROW_INTERCHANGE_SCHEMA_VERSION_KEY,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _decode_metadata(metadata: Mapping[str, bytes] | None) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        key_text = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        if key_text in _CONTRACT_KEYS:
            continue
        text = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
        decoded[key_text] = json.loads(text)
    return decoded


def _schema_metadata(
    kind: str, *, metadata: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None
) -> dict[str, bytes]:
    encoded = {
        ARROW_INTERCHANGE_KIND_KEY: kind.encode("utf-8"),
        ARROW_INTERCHANGE_SCHEMA_VERSION_KEY: ARROW_INTERCHANGE_SCHEMA_VERSION.encode("utf-8"),
    }
    for source in (extra, metadata):
        for key, value in (source or {}).items():
            key_text = str(key)
            if key_text in _IMMUTABLE_KEYS:
                continue
            encoded[key_text] = json.dumps(_json_ready(value), sort_keys=True).encode("utf-8")
    return encoded


def describe_arrow_interchange() -> dict[str, object]:
    """Describe the durable Arrow-compatible interchange contract."""
    return {
        "schema_version": ARROW_INTERCHANGE_SCHEMA_VERSION,
        "payloads": {
            "array": {
                "arrow_columns": (ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN,),
                "metadata_keys": (
                    ARROW_INTERCHANGE_KIND_KEY,
                    ARROW_INTERCHANGE_SCHEMA_VERSION_KEY,
                    ARROW_INTERCHANGE_SHAPE_KEY,
                    ARROW_INTERCHANGE_DTYPE_KEY,
                ),
                "pandas": {
                    "value_column": ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN,
                    "preserve_index": False,
                },
            },
            "table": {
                "arrow_columns": "preserve payload columns",
                "metadata_keys": (
                    ARROW_INTERCHANGE_KIND_KEY,
                    ARROW_INTERCHANGE_SCHEMA_VERSION_KEY,
                    ARROW_INTERCHANGE_COLUMNS_KEY,
                ),
                "pandas": {
                    "preserve_columns": True,
                    "preserve_index": False,
                },
            },
            "discovery": {
                "arrow_columns": (
                    "key",
                    "family",
                    "import_path",
                    "stability",
                    "supports_covariates",
                    "supports_multivariate_output",
                    "supported_backends",
                    "optional_dependencies",
                    "supports_simulation",
                    "supports_summarize",
                ),
                "metadata_keys": (
                    ARROW_INTERCHANGE_KIND_KEY,
                    ARROW_INTERCHANGE_SCHEMA_VERSION_KEY,
                ),
            },
        },
    }


def kernel_array_payload_to_table(payload: KernelArrayPayload) -> pa.Table:
    """Convert a kernel array payload into a single-column Arrow table."""
    table = pa.Table.from_pydict({ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN: list(payload.values)})
    metadata = _schema_metadata(
        ARROW_INTERCHANGE_ARRAY_KIND,
        metadata=payload.metadata,
        extra={
            ARROW_INTERCHANGE_SHAPE_KEY: list(payload.shape),
            ARROW_INTERCHANGE_DTYPE_KEY: payload.dtype,
        },
    )
    return table.replace_schema_metadata(metadata)


def kernel_array_payload_from_table(table: pa.Table) -> KernelArrayPayload:
    """Convert a contract Arrow table back into a kernel array payload."""
    metadata = table.schema.metadata or {}
    if metadata.get(ARROW_INTERCHANGE_KIND_KEY.encode("utf-8")) != ARROW_INTERCHANGE_ARRAY_KIND.encode("utf-8"):
        raise ValueError("Arrow interchange metadata is missing or invalid for array payloads")
    shape_value = metadata.get(ARROW_INTERCHANGE_SHAPE_KEY.encode("utf-8"))
    dtype_value = metadata.get(ARROW_INTERCHANGE_DTYPE_KEY.encode("utf-8"))
    if shape_value is None or dtype_value is None:
        raise ValueError("Arrow interchange metadata is missing or invalid for array payloads")

    values = table.column(ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN).to_pylist()
    shape = tuple(int(dimension) for dimension in json.loads(shape_value.decode("utf-8")))
    dtype = str(json.loads(dtype_value.decode("utf-8")))
    return KernelArrayPayload.from_values(
        values=values,
        shape=shape,
        dtype=dtype,
        metadata=_decode_metadata(metadata),
    )


def kernel_array_payload_to_dataframe(payload: KernelArrayPayload) -> pd.DataFrame:
    """Convert a kernel array payload into a pandas DataFrame."""
    frame = pd.DataFrame({ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN: list(payload.values)})
    frame.attrs["innovate.kind"] = ARROW_INTERCHANGE_ARRAY_KIND
    frame.attrs["innovate.schema_version"] = ARROW_INTERCHANGE_SCHEMA_VERSION
    frame.attrs["innovate.shape"] = list(payload.shape)
    frame.attrs["innovate.dtype"] = payload.dtype
    frame.attrs["innovate.metadata"] = dict(payload.metadata)
    return frame


def kernel_array_payload_from_dataframe(
    frame: pd.DataFrame,
    *,
    shape: Sequence[int],
    dtype: str,
    metadata: Mapping[str, Any] | None = None,
    value_column: str = ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN,
) -> KernelArrayPayload:
    """Convert a pandas DataFrame into a kernel array payload."""
    if value_column not in frame.columns:
        raise ValueError(f"DataFrame must contain a '{value_column}' column")
    return KernelArrayPayload.from_values(
        values=frame[value_column].tolist(),
        shape=shape,
        dtype=dtype,
        metadata=metadata if metadata is not None else frame.attrs.get("innovate.metadata", {}),
    )


def kernel_table_payload_to_table(payload: KernelTablePayload) -> pa.Table:
    """Convert a kernel table payload into an Arrow table."""
    rows = [dict(zip(payload.columns, row, strict=True)) for row in payload.rows]
    table = pa.Table.from_pylist(rows)
    metadata = _schema_metadata(
        ARROW_INTERCHANGE_TABLE_KIND,
        metadata=payload.metadata,
        extra={ARROW_INTERCHANGE_COLUMNS_KEY: list(payload.columns)},
    )
    return table.replace_schema_metadata(metadata)


def kernel_table_payload_from_table(table: pa.Table) -> KernelTablePayload:
    """Convert a contract Arrow table back into a kernel table payload."""
    metadata = table.schema.metadata or {}
    if metadata.get(ARROW_INTERCHANGE_KIND_KEY.encode("utf-8")) != ARROW_INTERCHANGE_TABLE_KIND.encode("utf-8"):
        raise ValueError("Arrow interchange metadata is missing or invalid for table payloads")

    columns = tuple(str(name) for name in table.column_names)
    rows = tuple(tuple(row[column] for column in columns) for row in table.to_pylist())
    return KernelTablePayload.from_rows(
        columns=columns,
        rows=rows,
        metadata=_decode_metadata(metadata),
    )


def kernel_table_payload_to_dataframe(payload: KernelTablePayload) -> pd.DataFrame:
    """Convert a kernel table payload into a pandas DataFrame."""
    frame = pd.DataFrame.from_records(list(payload.rows), columns=list(payload.columns))
    frame.attrs["innovate.kind"] = ARROW_INTERCHANGE_TABLE_KIND
    frame.attrs["innovate.schema_version"] = ARROW_INTERCHANGE_SCHEMA_VERSION
    frame.attrs["innovate.columns"] = list(payload.columns)
    frame.attrs["innovate.metadata"] = dict(payload.metadata)
    return frame


def kernel_table_payload_from_dataframe(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> KernelTablePayload:
    """Convert a pandas DataFrame into a kernel table payload."""
    return KernelTablePayload.from_rows(
        columns=tuple(str(column) for column in frame.columns),
        rows=tuple(tuple(row) for row in frame.itertuples(index=False, name=None)),
        metadata=metadata if metadata is not None else frame.attrs.get("innovate.metadata", {}),
    )


def kernel_discovery_response_to_table(response: KernelDiscoveryResponse) -> pa.Table:
    """Convert kernel discovery metadata into an Arrow table."""
    rows = [
        {
            "key": record.key,
            "family": record.family,
            "import_path": record.import_path,
            "stability": record.stability,
            "supports_covariates": record.supports_covariates,
            "supports_multivariate_output": record.supports_multivariate_output,
            "supported_backends": list(record.supported_backends),
            "optional_dependencies": list(record.optional_dependencies),
            "supports_simulation": record.supports_simulation,
            "supports_summarize": record.supports_summarize,
        }
        for record in response.models
    ]
    table = pa.Table.from_pylist(rows)
    metadata = _schema_metadata(ARROW_INTERCHANGE_DISCOVERY_KIND, metadata=response.metadata)
    return table.replace_schema_metadata(metadata)


def kernel_discovery_response_from_table(table: pa.Table) -> KernelDiscoveryResponse:
    """Convert an Arrow table back into kernel discovery metadata."""
    metadata = table.schema.metadata or {}
    if metadata.get(ARROW_INTERCHANGE_KIND_KEY.encode("utf-8")) != ARROW_INTERCHANGE_DISCOVERY_KIND.encode("utf-8"):
        raise ValueError("Arrow interchange metadata is missing or invalid for discovery payloads")

    models = tuple(
        KernelDiscoveryRecord(
            key=str(record["key"]),
            family=str(record["family"]),
            import_path=str(record["import_path"]),
            stability=str(record["stability"]),
            supports_covariates=bool(record["supports_covariates"]),
            supports_multivariate_output=bool(record["supports_multivariate_output"]),
            supported_backends=tuple(str(value) for value in record["supported_backends"]),
            optional_dependencies=tuple(str(value) for value in record.get("optional_dependencies", ())),
            supports_simulation=bool(record.get("supports_simulation", False)),
            supports_summarize=bool(record.get("supports_summarize", False)),
        )
        for record in table.to_pylist()
    )
    return KernelDiscoveryResponse(models=models, metadata=_decode_metadata(metadata))


__all__ = [
    "ARROW_INTERCHANGE_ARRAY_KIND",
    "ARROW_INTERCHANGE_ARRAY_VALUE_COLUMN",
    "ARROW_INTERCHANGE_COLUMNS_KEY",
    "ARROW_INTERCHANGE_DISCOVERY_KIND",
    "ARROW_INTERCHANGE_DTYPE_KEY",
    "ARROW_INTERCHANGE_KIND_KEY",
    "ARROW_INTERCHANGE_SCHEMA_VERSION",
    "ARROW_INTERCHANGE_SCHEMA_VERSION_KEY",
    "ARROW_INTERCHANGE_SHAPE_KEY",
    "ARROW_INTERCHANGE_TABLE_KIND",
    "describe_arrow_interchange",
    "kernel_array_payload_from_dataframe",
    "kernel_array_payload_from_table",
    "kernel_array_payload_to_dataframe",
    "kernel_array_payload_to_table",
    "kernel_discovery_response_from_table",
    "kernel_discovery_response_to_table",
    "kernel_table_payload_from_dataframe",
    "kernel_table_payload_from_table",
    "kernel_table_payload_to_dataframe",
    "kernel_table_payload_to_table",
]
