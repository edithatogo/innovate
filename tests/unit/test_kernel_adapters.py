"""Tests for the functional kernel adapter surface."""

from __future__ import annotations

import numpy as np


def _payload_to_ndarray(result: object) -> np.ndarray:
    from innovate import kernel

    if isinstance(result, kernel.KernelArrayPayload):
        return np.asarray(result.values, dtype=float).reshape(result.shape)
    if isinstance(result, kernel.KernelTablePayload):
        return np.asarray(result.rows, dtype=float)
    raise AssertionError(f"Unexpected kernel payload type: {type(result)!r}")


def _fit_request(
    model_key: str,
    time: np.ndarray,
    observed: np.ndarray,
    *,
    inputs: dict[str, object] | None = None,
    **payload: object,
):
    from innovate import kernel

    request_payload: dict[str, object] = {
        "inputs": {
            "time": time.tolist(),
            "observed": observed.tolist(),
        },
    }
    if inputs:
        request_payload["inputs"].update(inputs)
    request_payload.update(payload)
    return kernel.KernelRequest(
        operation="fit_model",
        model_key=model_key,
        payload=request_payload,
    )


def _state_request(operation: str, model_key: str, state: dict[str, object], time: np.ndarray, **inputs: object):
    from innovate import kernel

    request_payload: dict[str, object] = {
        "state": state,
        "inputs": {
            "time": time.tolist(),
            **inputs,
        },
    }
    return kernel.KernelRequest(
        operation=operation,
        model_key=model_key,
        payload=request_payload,
    )


def test_kernel_adapter_round_trip_for_diffusion_family() -> None:
    """Diffusion adapters should fit, predict, and simulate from a fitted state."""
    from innovate import kernel
    from innovate.diffuse.bass import BassModel

    time = np.linspace(0.0, 4.0, 6)
    fitted_model = BassModel()
    fitted_model.params_ = {"p": 0.03, "q": 0.38, "m": 120.0}
    observed = np.asarray(fitted_model.predict(time), dtype=float)

    fit_response = kernel.fit_model(_fit_request("bass", time, observed))

    assert fit_response.error is None
    assert fit_response.result is not None
    assert fit_response.result["model_key"] == "bass"
    assert fit_response.result["family"] == "diffusion"
    assert fit_response.result["state"]["model_key"] == "bass"
    assert fit_response.result["state"]["parameters"]

    fitted_state = fit_response.result["state"]

    predict_response = kernel.predict_model(_state_request("predict_model", "bass", fitted_state, time))
    simulate_response = kernel.simulate_model(_state_request("simulate_model", "bass", fitted_state, time))

    predicted = _payload_to_ndarray(predict_response.result)
    simulated = _payload_to_ndarray(simulate_response.result)

    np.testing.assert_allclose(predicted, observed, rtol=0.25, atol=5.0)
    np.testing.assert_allclose(simulated, predicted)


def test_kernel_adapter_round_trip_for_substitution_family() -> None:
    """Substitution adapters should summarize fitted state and preserve predictions."""
    from innovate import kernel
    from innovate.substitute.fisher_pry import FisherPryModel

    time = np.linspace(0.0, 5.0, 7)
    fitted_model = FisherPryModel()
    fitted_model.params_ = {"alpha": 1.6, "t0": 2.0}
    observed = np.asarray(fitted_model.predict(time), dtype=float)

    fit_response = kernel.fit_model(_fit_request("fisher_pry", time, observed))
    assert fit_response.error is None

    fitted_state = fit_response.result["state"]
    predict_response = kernel.predict_model(_state_request("predict_model", "fisher_pry", fitted_state, time))
    summary_response = kernel.summarize_model(
        kernel.KernelRequest(
            operation="summarize_model",
            model_key="fisher_pry",
            payload={
                "state": fitted_state,
                "inputs": {
                    "time": time.tolist(),
                    "observed": observed.tolist(),
                },
            },
        ),
    )

    predicted = _payload_to_ndarray(predict_response.result)
    np.testing.assert_allclose(predicted, observed, rtol=0.2, atol=0.05)

    assert summary_response.error is None
    assert summary_response.result is not None
    assert summary_response.result["family"] == "substitution"
    assert summary_response.result["parameter_names"] == ["alpha", "t0"]
    assert summary_response.result["diagnostics"]["uncertainty"]["report_type"] == "point_estimate"


def test_kernel_adapter_round_trip_for_competition_family() -> None:
    """Competition adapters should fit native models and emit diagnostics contracts."""
    from innovate import kernel
    from innovate.compete.lotka_volterra import LotkaVolterraModel

    time = np.linspace(0.0, 3.0, 6)
    fitted_model = LotkaVolterraModel()
    fitted_model.params_ = {
        "alpha1": 0.7,
        "beta1": 0.15,
        "alpha2": 0.5,
        "beta2": 0.08,
    }
    y0 = np.array([0.12, 0.08], dtype=float)
    observed = np.asarray(fitted_model.predict(time, y0.tolist()), dtype=float)

    fit_response = kernel.fit_model(
        _fit_request("lotka_volterra", time, observed, inputs={"y0": y0.tolist()}),
    )
    assert fit_response.error is None

    fitted_state = fit_response.result["state"]
    predict_response = kernel.predict_model(
        _state_request("predict_model", "lotka_volterra", fitted_state, time, y0=y0.tolist()),
    )
    diagnose_response = kernel.diagnose_model(
        kernel.KernelRequest(
            operation="diagnose_model",
            model_key="lotka_volterra",
            payload={
                "state": fitted_state,
                "inputs": {
                    "time": time.tolist(),
                    "observed": observed.tolist(),
                    "y0": y0.tolist(),
                },
            },
        ),
    )

    predicted = _payload_to_ndarray(predict_response.result)
    np.testing.assert_allclose(predicted, observed, rtol=0.35, atol=0.08)

    assert diagnose_response.error is None
    assert diagnose_response.result is not None
    assert diagnose_response.result["diagnostics"]["provenance"] == "deterministic"
    assert diagnose_response.result["diagnostics"]["support_level"] == "supported"
