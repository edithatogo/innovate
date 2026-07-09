"""Public-data-style adapter pattern with licensing/provenance safeguards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from innovate.data.contracts import DatasetContract, DatasetKind
from innovate.data.ingestion import frame_to_dataset
from innovate.data.provenance import DatasetProvenance
from innovate.data.validation import ValidationReport, validate_dataset


class PublicDataAdapterError(ValueError):
    """Raised when a public-data adapter cannot safely produce a dataset."""


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    """Documented adapter metadata for discovery and release evidence."""

    adapter_id: str
    title: str
    source_name: str
    license: str
    citation: str
    dataset_kind: DatasetKind
    homepage: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter_id": self.adapter_id,
            "title": self.title,
            "source_name": self.source_name,
            "license": self.license,
            "citation": self.citation,
            "dataset_kind": self.dataset_kind,
            "homepage": self.homepage,
            "notes": self.notes,
        }


class PublicDataAdapter(ABC):
    """Base adapter for external/public datasets.

    Subclasses must declare license/citation up front. Fetching is optional;
    many adapters only transform caller-supplied tables while recording
    provenance. Restricted datasets must not be bundled in the repository.
    """

    manifest: AdapterManifest

    def __init__(self) -> None:
        if not getattr(self, "manifest", None):
            raise PublicDataAdapterError("adapter must define a manifest")
        # Fail closed at construction if license is unknown.
        DatasetProvenance.create(
            source=self.manifest.source_name,
            license=self.manifest.license,
            citation=self.manifest.citation,
            transform_steps=("adapter:init",),
        )

    @abstractmethod
    def load_frame(self, **kwargs: Any) -> pd.DataFrame:
        """Return a tabular frame ready for contract conversion."""

    def provenance(
        self,
        *,
        transform_steps: tuple[str, ...] = (),
        extra: Mapping[str, Any] | None = None,
        extraction_time: str | None = None,
    ) -> DatasetProvenance:
        steps = ("adapter:" + self.manifest.adapter_id, *transform_steps)
        return DatasetProvenance.create(
            source=self.manifest.source_name,
            license=self.manifest.license,
            citation=self.manifest.citation,
            transform_steps=steps,
            extraction_time=extraction_time,
            extra={
                "adapter_id": self.manifest.adapter_id,
                "homepage": self.manifest.homepage,
                **dict(extra or {}),
            },
        )

    def ingest(self, **kwargs: Any) -> tuple[DatasetContract, ValidationReport]:
        """Load, attach provenance, convert, and validate."""
        extraction_time = kwargs.pop("extraction_time", None)
        frame = self.load_frame(**kwargs)
        dataset = frame_to_dataset(
            frame,
            self.manifest.dataset_kind,
            provenance=self.provenance(
                transform_steps=("load_frame", "frame_to_dataset"),
                extraction_time=extraction_time,
            ),
            validate=True,
        )
        return dataset, validate_dataset(dataset)


class SyntheticAdoptionPublicAdapter(PublicDataAdapter):
    """Documented public-data-style adapter using synthetic open data only.

    This demonstrates the adapter pattern without bundling restricted sources.
    The synthetic series is CC0-like (explicit public-domain dedication for
    generated fixture data).
    """

    manifest = AdapterManifest(
        adapter_id="synthetic_adoption_v1",
        title="Synthetic Adoption Fixture Adapter",
        source_name="innovate.synthetic.adoption",
        license="CC0-1.0",
        citation="Innovate project synthetic adoption fixture (generated, not observed).",
        dataset_kind="adoption",
        homepage="https://github.com/edithatogo/innovate",
        notes="Does not fetch external APIs; generates reproducible open fixture rows.",
    )

    def load_frame(self, *, periods: int = 10, seed: int = 0) -> pd.DataFrame:
        if periods < 2:
            raise PublicDataAdapterError("periods must be >= 2")
        # Deterministic logistic-like cumulative series for demos/tests.
        time = list(range(periods))
        base = 1.0 / (1.0 + 2.718281828 ** (-0.45 * (seed % 7 + 1)))
        adoption = []
        level = 1.0
        for step in time:
            level = min(100.0, level * (1.0 + 0.35 / (1.0 + step * 0.1)) + base)
            adoption.append(level)
        return pd.DataFrame({"time": time, "adoption": adoption, "denominator": [100.0] * periods})

    def ingest(self, **kwargs: Any) -> tuple[DatasetContract, ValidationReport]:
        """Ingest with a deterministic extraction_time when callers omit one."""
        kwargs.setdefault("extraction_time", "1970-01-01T00:00:00+00:00")
        return super().ingest(**kwargs)


def list_builtin_adapters() -> tuple[AdapterManifest, ...]:
    """Return manifests for built-in public-data-style adapters."""
    return (SyntheticAdoptionPublicAdapter.manifest,)


def get_builtin_adapter(adapter_id: str) -> PublicDataAdapter:
    """Instantiate a built-in adapter by id."""
    registry: dict[str, Callable[[], PublicDataAdapter]] = {
        SyntheticAdoptionPublicAdapter.manifest.adapter_id: SyntheticAdoptionPublicAdapter,
    }
    try:
        return registry[adapter_id]()
    except KeyError as exc:
        raise KeyError(f"unknown public-data adapter: {adapter_id}") from exc
