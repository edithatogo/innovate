import numpy as np
import pytest

from innovate.probabilistic import (
    PROBABILISTIC_INSTALL_HINT,
    PROBABILISTIC_SCHEMA_VERSION,
    PosteriorConfig,
    PosteriorSamplesPayload,
    ProbabilisticBackendStatus,
    ProbabilisticBackendUnavailableError,
    _module_available,
    _validate_schema_version,
    list_probabilistic_backend_statuses,
    require_probabilistic_backend,
)


def test_probabilistic_backend_status_dataclass() -> None:
    status = ProbabilisticBackendStatus(
        engine="test_engine",
        role="test_role",
    )
    assert status.engine == "test_engine"
    assert status.role == "test_role"
    assert status.backend == "jax"
    assert status.xla_eligible is True


def test_require_probabilistic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("innovate.probabilistic._module_available", lambda x: x == "present_dep")  # type: ignore[no-any-return]

    # Should not raise
    require_probabilistic_backend(engine="test", optional_dependencies=("present_dep",))

    # Should raise
    with pytest.raises(ProbabilisticBackendUnavailableError) as exc_info:
        require_probabilistic_backend(engine="test", optional_dependencies=("present_dep", "missing_dep"))

    err = exc_info.value
    assert err.engine == "test"
    assert err.missing_dependencies == ("missing_dep",)

    err_dict = err.to_dict()
    assert err_dict["code"] == "probabilistic_backend_unavailable"
    assert err_dict["engine"] == "test"
    assert err_dict["missing_dependencies"] == ["missing_dep"]


def test_list_probabilistic_backend_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("innovate.probabilistic._module_available", lambda x: x in ("jax", "jaxlib"))  # type: ignore[no-any-return]

    statuses = list_probabilistic_backend_statuses()

    assert len(statuses) == 3
    # numpyro requires numpyro and arviz, so available should be False
    assert statuses[0].engine == "numpyro"
    assert statuses[0].available is False

    # tensorflow_probability_jax requires tensorflow_probability, so available should be False
    assert statuses[2].engine == "tensorflow_probability_jax"
    assert statuses[2].available is False


def test_posterior_payload_validation() -> None:
    # Model key empty
    with pytest.raises(ValueError, match="model_key must be non-empty"):
        PosteriorSamplesPayload(
            model_key="", parameter_names=("a",), draw_shape=(2, 10), samples={"a": tuple([1.0] * 20)}
        )

    # No parameter names
    with pytest.raises(ValueError, match="must include parameter names"):
        PosteriorSamplesPayload(model_key="test", parameter_names=(), draw_shape=(2, 10), samples={})

    # Invalid draw shape
    with pytest.raises(ValueError, match="must be positive"):
        PosteriorSamplesPayload(model_key="test", parameter_names=("a",), draw_shape=(0, 10), samples={"a": ()})

    # Mismatched sample sizes
    with pytest.raises(ValueError, match="must match draw_shape"):
        PosteriorSamplesPayload(
            model_key="test",
            parameter_names=("a",),
            draw_shape=(2, 10),  # expects 20
            samples={"a": tuple([1.0] * 15)},  # gives 15
        )


def test_posterior_payload_from_samples() -> None:
    samples = {
        "alpha": np.ones((2, 10)),
        "beta": np.zeros((2, 10)),
    }

    config = PosteriorConfig(engine="test_engine", seed=42)

    payload = PosteriorSamplesPayload.from_samples(
        model_key="test_model",
        samples=samples,
        config=config,
    )

    assert payload.model_key == "test_model"
    assert payload.draw_shape == (2, 10)
    assert payload.parameter_names == ("alpha", "beta")
    assert payload.engine == "test_engine"
    assert payload.seed == 42
    assert len(payload.samples["alpha"]) == 20


def test_posterior_payload_from_samples_validation() -> None:
    # Empty samples
    with pytest.raises(ValueError, match="requires at least one parameter sample"):
        PosteriorSamplesPayload.from_samples(model_key="test", samples={})

    # Mismatched shapes
    samples = {
        "a": np.ones((2, 10)),
        "b": np.zeros((3, 10)),
    }
    with pytest.raises(ValueError, match="must share the same 2D draw shape"):
        PosteriorSamplesPayload.from_samples(model_key="test", samples=samples)


def test_posterior_payload_serialization() -> None:
    original = PosteriorSamplesPayload(
        model_key="test_model",
        parameter_names=("a",),
        draw_shape=(2, 5),
        samples={"a": tuple(range(10))},
        seed=123,
    )

    serialized = original.to_dict()
    assert serialized["model_key"] == "test_model"
    assert serialized["draw_shape"] == [2, 5]

    deserialized = PosteriorSamplesPayload.from_dict(serialized)

    assert deserialized.model_key == original.model_key
    assert deserialized.parameter_names == original.parameter_names
    assert deserialized.draw_shape == original.draw_shape
    assert deserialized.samples == original.samples
    assert deserialized.seed == original.seed


def test_schema_version_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported probabilistic schema version"):
        PosteriorSamplesPayload.from_dict(
            {
                "schema_version": "999.0",
                "model_key": "test",
                "parameter_names": ["a"],
                "draw_shape": [2, 5],
                "samples": {"a": [1.0] * 10},
            }
        )


def test_posterior_payload_methods() -> None:
    payload = PosteriorSamplesPayload(
        model_key="test_model",
        parameter_names=("a", "b"),
        draw_shape=(2, 5),
        samples={"a": tuple(range(10)), "b": tuple(range(10, 20))},
    )

    arr_a = payload.sample_array("a")
    assert arr_a.shape == (2, 5)
    assert arr_a[0, 0] == 0
    assert arr_a[1, 4] == 9

    summary = payload.to_uncertainty_summary(level=0.9)
    # 5th and 95th percentiles
    assert "a" in summary.lower
    assert "b" in summary.upper
    assert "a" in summary.median

    assert summary.level == 0.9
    assert summary.report_type == "posterior_summary"


def test_module_available() -> None:
    # Test real module
    assert _module_available("numpy") is True
    # Test fake module
    assert _module_available("non_existent_module_123") is False


def test_probabilistic_backend_status_to_dict() -> None:
    status = ProbabilisticBackendStatus(
        engine="test_engine",
        role="test_role",
        optional_dependencies=("dep1", "dep2"),
    )

    serialized = status.to_dict()
    assert serialized["engine"] == "test_engine"
    assert serialized["role"] == "test_role"
    assert serialized["backend"] == "jax"
    assert serialized["xla_eligible"] is True
    assert serialized["optional_dependencies"] == ["dep1", "dep2"]
    assert serialized["available"] is False
    assert "install_hint" in serialized


def test_probabilistic_backend_unavailable_error_str() -> None:
    error = ProbabilisticBackendUnavailableError(
        engine="test_engine",
        missing_dependencies=("dep1", "dep2"),
    )
    assert (
        str(error)
        == f"test_engine is unavailable; install {PROBABILISTIC_INSTALL_HINT}. Missing dependencies: dep1, dep2"
    )


def test_validate_schema_version_valid() -> None:
    # Should just return the exact same string
    result = _validate_schema_version(PROBABILISTIC_SCHEMA_VERSION)
    assert result == PROBABILISTIC_SCHEMA_VERSION


def test_posterior_payload_from_dict_missing_fields() -> None:
    payload = {
        "schema_version": PROBABILISTIC_SCHEMA_VERSION,
        # model_key is missing
        "parameter_names": ["a"],
        "draw_shape": [2, 5],
        "samples": {"a": [1.0] * 10},
    }
    with pytest.raises(KeyError, match="model_key"):
        PosteriorSamplesPayload.from_dict(payload)
