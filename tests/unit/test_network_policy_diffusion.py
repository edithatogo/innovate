from __future__ import annotations

import pytest


def test_network_input_contracts_validate_and_adapt(network_diffusion_scenario):
    """Network diffusion inputs should validate adjacency, edges, and coordinates."""
    from innovate.models.contracts import NetworkDiffusionInputs

    inputs = NetworkDiffusionInputs.from_adjacency(
        network_diffusion_scenario["adjacency"],
        node_labels=network_diffusion_scenario["node_labels"],
    )

    assert inputs.node_labels == ("north", "south")
    assert inputs.row_normalized_adjacency().tolist() == [[0.0, 1.0], [1.0, 0.0]]
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
    assert all(coord_inputs.adjacency[index, index] == 0.0 for index in range(3))

    with pytest.raises(ValueError, match="square"):
        NetworkDiffusionInputs.from_adjacency([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])


def test_policy_input_contracts_validate_and_adapt(policy_timing_scenario):
    """Policy timing inputs should validate ordered timing data and serialize cleanly."""
    from innovate.models.contracts import PolicyTimingInputs

    inputs = PolicyTimingInputs.from_events(
        event_times=policy_timing_scenario["event_times"],
        event_effects=policy_timing_scenario["event_effects"],
        event_labels=policy_timing_scenario["event_labels"],
    )

    assert inputs.event_times == (3.0, 6.0)
    assert inputs.event_effects == (0.15, -0.05)
    assert inputs.to_dict()["event_labels"] == ["subsidy", "rollback"]

    with pytest.raises(ValueError, match="same length"):
        PolicyTimingInputs.from_events(event_times=[1.0], event_effects=[0.1, 0.2])


def test_network_policy_models_expose_canonical_surface(
    network_diffusion_scenario,
    policy_timing_scenario,
):
    """The canonical network and policy models should fit, predict, simulate, and summarize."""
    from innovate.diffuse.bass import BassModel
    from innovate.models.contracts import NetworkDiffusionInputs, PolicyTimingInputs
    from innovate.models.network import NetworkDiffusionModel
    from innovate.models.policy import PolicyHazardDiffusionModel

    t = network_diffusion_scenario["time_points"]
    y_network = network_diffusion_scenario["observations"]
    network_inputs = NetworkDiffusionInputs.from_adjacency(
        network_diffusion_scenario["adjacency"],
        node_labels=network_diffusion_scenario["node_labels"],
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

    y_policy = policy_timing_scenario["observations"]
    policy_inputs = PolicyTimingInputs.from_events(
        event_times=policy_timing_scenario["event_times"],
        event_effects=policy_timing_scenario["event_effects"],
        event_labels=policy_timing_scenario["event_labels"],
    )
    policy_model = PolicyHazardDiffusionModel(policy_inputs, BassModel(), decay=0.4)
    policy_model.fit(policy_timing_scenario["time_points"], y_policy)
    policy_prediction = policy_model.predict(policy_timing_scenario["time_points"])
    policy_draws = policy_model.simulate(policy_timing_scenario["time_points"], n_draws=2, random_state=11)
    policy_summary = policy_model.summarize(policy_timing_scenario["time_points"])

    assert policy_prediction.shape == (8,)
    assert policy_draws.shape == (2, 8)
    assert policy_summary.family == "policy_hazard"
    assert policy_summary.details["event_count"] == 2
    assert policy_summary.details["decay"] == pytest.approx(0.4)


def test_canonical_exports_and_capabilities_reflect_new_model_families():
    """The public API should surface the new model families and capability metadata."""
    import innovate
    from innovate.capabilities import ModelCapability

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
