from __future__ import annotations

import numpy as np
import pytest

import innovate
from innovate.capabilities import ModelCapability


def test_network_input_contracts_validate_and_adapt():
    """Network diffusion inputs should validate adjacency, edges, and coordinates."""
    from innovate.models.contracts import NetworkDiffusionInputs

    inputs = NetworkDiffusionInputs.from_adjacency(
        [[0.0, 1.0], [1.0, 0.0]],
        node_labels=["north", "south"],
    )

    assert inputs.adjacency.shape == (2, 2)
    assert inputs.node_labels == ("north", "south")
    assert np.allclose(inputs.row_normalized_adjacency(), [[0.0, 1.0], [1.0, 0.0]])
    assert inputs.to_dict()["node_labels"] == ["north", "south"]

    edge_inputs = NetworkDiffusionInputs.from_edge_list(
        [("north", "south")],
        node_labels=["north", "south"],
    )
    assert edge_inputs.adjacency.shape == (2, 2)

    coord_inputs = NetworkDiffusionInputs.from_coordinates(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        node_labels=["a", "b", "c"],
    )
    assert coord_inputs.adjacency.shape == (3, 3)
    assert np.all(np.diag(coord_inputs.adjacency) == 0.0)

    with pytest.raises(ValueError, match="square"):
        NetworkDiffusionInputs.from_adjacency([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])


def test_policy_input_contracts_validate_and_adapt():
    """Policy timing inputs should validate ordered timing data and serialize cleanly."""
    from innovate.models.contracts import PolicyTimingInputs

    inputs = PolicyTimingInputs.from_events(
        event_times=[3.0, 8.0],
        event_effects=[0.2, -0.1],
        event_labels=["policy_a", "policy_b"],
    )

    assert inputs.event_times == (3.0, 8.0)
    assert inputs.event_effects == (0.2, -0.1)
    assert inputs.to_dict()["event_labels"] == ["policy_a", "policy_b"]

    with pytest.raises(ValueError, match="same length"):
        PolicyTimingInputs.from_events(event_times=[1.0], event_effects=[0.1, 0.2])


def test_network_policy_models_expose_canonical_surface():
    """The canonical network and policy models should fit, predict, simulate, and summarize."""
    from innovate.diffuse.bass import BassModel
    from innovate.models.contracts import NetworkDiffusionInputs, PolicyTimingInputs
    from innovate.models.network import NetworkDiffusionModel
    from innovate.models.policy import PolicyHazardDiffusionModel

    t = np.arange(1, 9, dtype=float)
    node_0 = np.array([2.0, 5.0, 9.0, 16.0, 28.0, 42.0, 57.0, 70.0])
    node_1 = np.array([1.0, 4.0, 8.0, 13.0, 21.0, 31.0, 44.0, 58.0])
    y_network = np.vstack([node_0, node_1])

    network_inputs = NetworkDiffusionInputs.from_adjacency(
        [[0.0, 1.0], [1.0, 0.0]],
        node_labels=["north", "south"],
    )
    network_model = NetworkDiffusionModel(network_inputs, BassModel(), spillover_strength=0.2)
    network_model.fit(t, y_network)
    network_prediction = network_model.predict(t)
    network_draws = network_model.simulate(t, n_draws=3, random_state=5)
    network_summary = network_model.summarize(t)

    assert network_prediction.shape == (2, 8)
    assert network_draws.shape == (3, 2, 8)
    assert network_summary.family == "network_diffusion"
    assert network_summary.details["node_count"] == 2
    assert network_summary.details["spillover_strength"] == pytest.approx(0.2)

    y_policy = np.array([1.0, 3.0, 7.0, 14.0, 25.0, 39.0, 56.0, 76.0])
    policy_inputs = PolicyTimingInputs.from_events(
        event_times=[3.0, 6.0],
        event_effects=[0.15, -0.05],
        event_labels=["subsidy", "rollback"],
    )
    policy_model = PolicyHazardDiffusionModel(policy_inputs, BassModel(), decay=0.4)
    policy_model.fit(t, y_policy)
    policy_prediction = policy_model.predict(t)
    policy_draws = policy_model.simulate(t, n_draws=2, random_state=11)
    policy_summary = policy_model.summarize(t)

    assert policy_prediction.shape == (8,)
    assert policy_draws.shape == (2, 8)
    assert policy_summary.family == "policy_hazard"
    assert policy_summary.details["event_count"] == 2
    assert policy_summary.details["decay"] == pytest.approx(0.4)


def test_canonical_exports_and_capabilities_reflect_new_model_families():
    """The public API should surface the new model families and capability metadata."""
    assert hasattr(innovate.models, "NetworkDiffusionModel")
    assert hasattr(innovate.models, "PolicyHazardDiffusionModel")
    assert innovate.NetworkDiffusionModel is innovate.models.NetworkDiffusionModel
    assert innovate.PolicyHazardDiffusionModel is innovate.models.PolicyHazardDiffusionModel

    registry = innovate.get_model_registry()
    assert isinstance(registry["network_diffusion"], ModelCapability)
    assert isinstance(registry["policy_hazard"], ModelCapability)
    assert registry["network_diffusion"].supports_multivariate_output is True
    assert registry["network_diffusion"].supports_simulation is True
    assert registry["policy_hazard"].supports_summarize is True
