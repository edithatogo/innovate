"""Fail-closed provenance metadata for ingested datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

DATASET_PROVENANCE_SCHEMA_VERSION = "1.0"
UNKNOWN_LICENSE_MARKERS = frozenset({"", "unknown", "n/a", "na", "none", "unspecified"})


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def compute_payload_checksum(payload: Mapping[str, Any] | Sequence[Any] | str | bytes) -> str:
    """Return a stable SHA-256 checksum for JSON-serializable or raw payloads."""
    if isinstance(payload, (bytes, bytearray)):
        digest = hashlib.sha256(payload)
    elif isinstance(payload, str):
        digest = hashlib.sha256(payload.encode("utf-8"))
    else:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def compute_dataset_content_checksum(dataset_payload: Mapping[str, Any]) -> str:
    """Checksum dataset content excluding the mutable provenance.checksum field.

    Hashing the full ``to_dict()`` output would bake an empty checksum into the
    digest and then change the payload when the checksum is filled in, making
    re-validation of the digest ambiguous.
    """
    payload = dict(dataset_payload)
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_copy = dict(provenance)
        provenance_copy.pop("checksum", None)
        # extraction_time may vary by wall clock; content identity is data rows.
        provenance_copy.pop("extraction_time", None)
        payload["provenance"] = provenance_copy
    return compute_payload_checksum(payload)


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Reproducible provenance record for an ingested dataset.

    Fail closed: unknown/missing license is rejected so restricted data cannot
    silently enter supported workflows.
    """

    source: str
    license: str
    extraction_time: str
    transform_steps: tuple[str, ...]
    schema_version: str = DATASET_PROVENANCE_SCHEMA_VERSION
    checksum: str = ""
    citation: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _require_non_empty(self.source, "source"))
        license_value = _require_non_empty(self.license, "license")
        if license_value.lower() in UNKNOWN_LICENSE_MARKERS:
            raise ValueError(
                "license is unknown; refuse to ingest without an explicit license "
                "(fail closed for provenance/licensing safeguards)"
            )
        object.__setattr__(self, "license", license_value)
        object.__setattr__(self, "extraction_time", _require_non_empty(self.extraction_time, "extraction_time"))
        object.__setattr__(
            self,
            "transform_steps",
            tuple(_require_non_empty(step, "transform_steps entry") for step in self.transform_steps),
        )
        object.__setattr__(self, "schema_version", _require_non_empty(self.schema_version, "schema_version"))
        if self.schema_version != DATASET_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported provenance schema_version: {self.schema_version}")
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def create(
        cls,
        *,
        source: str,
        license: str,
        transform_steps: Sequence[str] = (),
        checksum: str = "",
        citation: str = "",
        extraction_time: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> DatasetProvenance:
        """Build provenance with UTC extraction time when not provided."""
        stamp = extraction_time or datetime.now(tz=UTC).replace(microsecond=0).isoformat()
        return cls(
            source=source,
            license=license,
            extraction_time=stamp,
            transform_steps=tuple(transform_steps),
            checksum=checksum,
            citation=citation,
            extra=dict(extra or {}),
        )

    def with_checksum(self, checksum: str) -> DatasetProvenance:
        """Return a copy with an updated content checksum."""
        return DatasetProvenance(
            source=self.source,
            license=self.license,
            extraction_time=self.extraction_time,
            transform_steps=self.transform_steps,
            schema_version=self.schema_version,
            checksum=_require_non_empty(checksum, "checksum"),
            citation=self.citation,
            extra=self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "license": self.license,
            "extraction_time": self.extraction_time,
            "transform_steps": list(self.transform_steps),
            "checksum": self.checksum,
            "citation": self.citation,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetProvenance:
        return cls(
            source=str(data["source"]),
            license=str(data["license"]),
            extraction_time=str(data["extraction_time"]),
            transform_steps=tuple(str(step) for step in data.get("transform_steps", ())),
            schema_version=str(data.get("schema_version", DATASET_PROVENANCE_SCHEMA_VERSION)),
            checksum=str(data.get("checksum", "")),
            citation=str(data.get("citation", "")),
            extra=dict(data.get("extra", {})),
        )
