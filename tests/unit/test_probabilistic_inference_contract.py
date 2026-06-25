"""Tests for the probabilistic inference payload contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_probabilistic_backend_statuses_are_xla_first_and_optional() -> None:
    """Probabilistic engines should be discoverable without importing optional extras."""
    from innovate.probabilistic import list_probabilistic_backend_statuses

    statuses = {status.engine: status for status in list_probabilistic_backend_statuses()}

    assert statuses["numpyro"].role == "probabilistic_programming"
    assert statuses["blackjax"].role == "sampler"
    assert statuses["tensorflow_probability_jax"].role == "distribution_bijector"
    assert statuses["numpyro"].xla_eligible is True
    assert statuses["blackjax"].xla_eligible is True
    assert statuses["tensorflow_probability_jax"].xla_eligible is True

    for status in statuses.values():
        payload = status.to_dict()
        assert payload["backend"] == "jax"
        assert payload["optional_dependencies"]
        assert payload["install_hint"] == "innovate[bayesian]"


def test_posterior_samples_payload_round_trips_and_summarizes() -> None:
    """Posterior draws should round-trip through a stable schema payload."""
    from innovate.probabilistic import PosteriorSamplesPayload

    payload = PosteriorSamplesPayload.from_samples(
        model_key="bass",
        samples={
            "p": np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]]),
            "q": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]),
        },
        engine="blackjax",
        seed=123,
        metadata={"xla_eligible": True},
    )

    restored = PosteriorSamplesPayload.from_dict(payload.to_dict())
    summary = restored.to_uncertainty_summary(level=0.8)

    assert restored == payload
    assert restored.schema_version == "1.0"
    assert restored.draw_shape == (2, 3)
    assert restored.parameter_names == ("p", "q")
    assert summary.report_type == "posterior_summary"
    assert summary.provenance == "bayesian"
    assert summary.level == 0.8
    assert summary.median["p"] == pytest.approx(0.025)
    assert summary.lower["q"] < summary.median["q"] < summary.upper["q"]


def test_posterior_samples_payload_rejects_inconsistent_draw_shapes() -> None:
    """All posterior parameters must share the same chain/draw shape."""
    from innovate.probabilistic import PosteriorSamplesPayload

    with pytest.raises(ValueError, match="same 2D draw shape"):
        PosteriorSamplesPayload.from_samples(
            model_key="bass",
            samples={
                "p": np.array([[0.01, 0.02]]),
                "q": np.array([[0.2], [0.3]]),
            },
        )


def test_missing_probabilistic_backend_error_is_structured() -> None:
    """Missing optional probabilistic dependencies should produce structured errors."""
    from innovate.probabilistic import ProbabilisticBackendUnavailable, require_probabilistic_backend

    with pytest.raises(ProbabilisticBackendUnavailable) as exc_info:
        require_probabilistic_backend(
            engine="example_missing_engine",
            optional_dependencies=("definitely_missing_innovate_backend",),
        )

    error = exc_info.value.to_dict()
    assert error["code"] == "probabilistic_backend_unavailable"
    assert error["engine"] == "example_missing_engine"
    assert error["install_hint"] == "innovate[bayesian]"
    assert "definitely_missing_innovate_backend" in error["missing_dependencies"]


def test_probabilistic_inference_docs_are_linked() -> None:
    """Probabilistic payload documentation should be reachable from docs navigation."""
    docs = {
        "strategy": Path("docs/astro-site/src/content/docs/roadmap/probabilistic-inference.md"),
        "index": Path("docs/astro-site/starlight.config.mjs"),
        "innovate": Path("docs/source/innovate.rst"),
        "roadmap": Path("docs/architecture_modernization_roadmap.md"),
    }
    text = {name: path.read_text() for name, path in docs.items()}

    assert "PosteriorSamplesPayload" in text["strategy"]
    assert "NumPyro" in text["strategy"]
    assert "BlackJAX" in text["strategy"]
    assert "TensorFlow Probability's JAX substrate" in text["strategy"]
    assert "structured errors for missing optional dependencies" in text["strategy"]
    assert "/roadmap/probabilistic-inference/" in text["index"]
    assert "roadmap/probabilistic-inference.md" in text["innovate"]
    assert "roadmap/probabilistic-inference.md" in text["roadmap"]
