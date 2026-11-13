import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from innovate.scenario.schemas import (
    ArtifactEnvelope,
    BaselineScenario,
    CompetitionScenario,
    InterventionScenario,
    NetworkScenario,
    SubstitutionScenario,
)

common = dict(
    name="Historical",
    description="v0.6 tagged schema",
    time_horizon=5,
    time_unit="years",
    reference_year=2026,
    market_size=200,
    initial_adoption=0.25,
)
scenarios = [
    BaselineScenario(**common),
    InterventionScenario(**common, intervention_type="subsidy", intervention_start_time=2, intervention_magnitude=0.1),
    SubstitutionScenario(**common, incumbent_name="A", entrant_name="B", substitution_rate=0.5),
    CompetitionScenario(**common, num_competitors=2, competitor_names=["A", "B"], market_share_initial=[0.25, 0.75]),
    NetworkScenario(**common, network_type="random", num_nodes=4, average_degree=1),
]
artifacts = [
    ArtifactEnvelope(
        scenario=scenario,
        seed=2**64 - 1,
        model_type="bass",
        version="0.6.0",
        notes="tagged source",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    ).to_dict()
    for scenario in scenarios
]
Path(sys.argv[1]).write_text(json.dumps(artifacts, indent=2, allow_nan=False), encoding="utf-8")
print("Generated", len(artifacts), "artifacts from tagged scenario classes.")
