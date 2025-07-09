import pytest
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from innovate.plots.network import plot_network_diffusion
import matplotlib.pyplot as plt
import os

def test_plot_network_diffusion_basic(tmp_path):
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])

    # Define dummy node states over time
    node_states_over_time = [
        {1: False, 2: False, 3: False, 4: False}, # t=0
        {1: True, 2: False, 3: False, 4: False},  # t=1: Node 1 adopted
        {1: True, 2: True, 3: False, 4: False},   # t=2: Node 2 adopted
        {1: True, 2: True, 3: True, 4: False},    # t=3: Node 3 adopted
        {1: True, 2: True, 3: True, 4: True},     # t=4: Node 4 adopted
    ]

    # Define a temporary path for saving plots
    save_prefix = os.path.join(tmp_path, "network_snapshot")

    # Call the plotting function
    try:
        plot_network_diffusion(
            graph=G,
            node_states_over_time=node_states_over_time,
            title="Test Network Diffusion",
            save_path_prefix=save_prefix,
            snapshot_interval=1
        )
        # Check if files were created
        assert os.path.exists(f'{save_prefix}_step_000.png')
        assert os.path.exists(f'{save_prefix}_step_001.png')
        assert os.path.exists(f'{save_prefix}_step_002.png')
        assert os.path.exists(f'{save_prefix}_step_003.png')
        assert os.path.exists(f'{save_prefix}_step_004.png')

    except Exception as e:
        pytest.fail(f"plot_network_diffusion raised an exception: {e}")

    finally:
        # Clean up any open matplotlib figures to prevent them from showing up unexpectedly
        plt.close('all')

def test_plot_network_diffusion_no_save():
    # Test without saving to ensure it runs without error (will display if not in headless env)
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    node_states = [{1: False, 2: False, 3: False}, {1: True, 2: False, 3: False}]
    try:
        plot_network_diffusion(graph=G, node_states_over_time=node_states, snapshot_interval=1)
    except Exception as e:
        pytest.fail(f"plot_network_diffusion (no save) raised an exception: {e}")
    finally:
        plt.close('all')
