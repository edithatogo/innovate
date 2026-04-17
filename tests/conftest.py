"""Configuration for pytest with faulthandler enabled to catch segmentation faults."""

import faulthandler

import pytest

# Enable faulthandler to get Python tracebacks from segmentation faults
faulthandler.enable()

print("Faulthandler enabled for all tests.")
print("This will help identify if segfaults are from your code or dependencies.")

@pytest.fixture
def network_diffusion_scenario():
    """Representative network spillover data for diffusion tests."""
    return {
        "time_points": [float(value) for value in range(1, 9)],
        "observations": [
            [2.0, 5.0, 9.0, 16.0, 28.0, 42.0, 57.0, 70.0],
            [1.0, 4.0, 8.0, 13.0, 21.0, 31.0, 44.0, 58.0],
        ],
        "adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "node_labels": ["north", "south"],
    }


@pytest.fixture
def policy_timing_scenario():
    """Representative policy timing data for hazard diffusion tests."""
    return {
        "time_points": [float(value) for value in range(1, 9)],
        "observations": [1.0, 3.0, 7.0, 14.0, 25.0, 39.0, 56.0, 76.0],
        "event_times": [3.0, 6.0],
        "event_effects": [0.15, -0.05],
        "event_labels": ["subsidy", "rollback"],
    }
