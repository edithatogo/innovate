"""Data ingestion, provenance, and dataset contracts for innovate."""

from innovate.data.adapters import (
    AdapterManifest,
    PublicDataAdapter,
    PublicDataAdapterError,
    SyntheticAdoptionPublicAdapter,
    get_builtin_adapter,
    list_builtin_adapters,
)
from innovate.data.contracts import (
    DATASET_CONTRACT_SCHEMA_VERSION,
    AdoptionDataset,
    CompetitionDataset,
    DatasetContract,
    DatasetKind,
    NetworkEdgeDataset,
    PolicyTimingDataset,
    SubstitutionDataset,
    attach_provenance,
    dataset_from_dict,
)
from innovate.data.ingestion import (
    SUPPORTED_LOCAL_FORMATS,
    frame_to_dataset,
    ingest_local,
    ingest_polars,
    load_table,
    polars_available,
    reproducible_artifact,
)
from innovate.data.integration import (
    DatasetBenchmarkLink,
    dataset_to_baseline_scenario,
    get_dataset_benchmark_link,
    integration_bundle,
    list_dataset_benchmark_links,
    resolve_model_cards_for_dataset,
)
from innovate.data.provenance import (
    DATASET_PROVENANCE_SCHEMA_VERSION,
    DatasetProvenance,
    compute_payload_checksum,
)
from innovate.data.validation import (
    ValidationCheck,
    ValidationReport,
    require_valid,
    validate_dataset,
)

# Preserve legacy OECD helper if present.
try:
    from innovate.data.oecd import get_dataset as get_oecd_dataset
except ImportError:  # pragma: no cover
    get_oecd_dataset = None  # type: ignore[assignment]

__all__ = [
    "DATASET_CONTRACT_SCHEMA_VERSION",
    "DATASET_PROVENANCE_SCHEMA_VERSION",
    "SUPPORTED_LOCAL_FORMATS",
    "AdapterManifest",
    "AdoptionDataset",
    "CompetitionDataset",
    "DatasetBenchmarkLink",
    "DatasetContract",
    "DatasetKind",
    "DatasetProvenance",
    "NetworkEdgeDataset",
    "PolicyTimingDataset",
    "PublicDataAdapter",
    "PublicDataAdapterError",
    "SubstitutionDataset",
    "SyntheticAdoptionPublicAdapter",
    "ValidationCheck",
    "ValidationReport",
    "attach_provenance",
    "compute_payload_checksum",
    "dataset_from_dict",
    "dataset_to_baseline_scenario",
    "frame_to_dataset",
    "get_builtin_adapter",
    "get_dataset_benchmark_link",
    "get_oecd_dataset",
    "ingest_local",
    "ingest_polars",
    "integration_bundle",
    "list_builtin_adapters",
    "list_dataset_benchmark_links",
    "load_table",
    "polars_available",
    "reproducible_artifact",
    "require_valid",
    "resolve_model_cards_for_dataset",
    "validate_dataset",
]
