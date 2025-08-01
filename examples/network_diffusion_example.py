"""Example of a network diffusion model."""
import networkx as nx
import numpy as np
from innovate.plots.network import plot_network_diffusion


def run_network_diffusion_example() -> None:
    """Run a simplified simulation of product adoption to identify failed products."""
    # 1. Create a sample networkx graph (e.g., a small-world network)
    num_nodes = 20
    g = nx.watts_strogatz_graph(num_nodes, k=4, p=0.3, seed=42)

    # For consistent layout across plots
    pos = nx.spring_layout(g, seed=42)

    # 2. Simulate a simple diffusion process
    # Start with one random adopted node
    rng = np.random.default_rng(42)
    adopted_nodes = {rng.choice(list(g.nodes())): True}

    # Initialize all other nodes as not adopted
    initial_states = {node: adopted_nodes.get(node, False) for node in g.nodes()}
    node_states_over_time = [initial_states]

    # Simulate for a few steps
    num_steps = 10
    for _ in range(num_steps - 1):
        current_states = node_states_over_time[-1].copy()
        newly_adopted_this_step = set()

        for node in g.nodes():
            if not current_states[node]:  # If not yet adopted
                # Check if any adopted neighbors exist
                adopted_neighbors = [
                    n for n in g.neighbors(node) if current_states.get(n, False)
                ]
                if (
                    adopted_neighbors
                ):  # Simple rule: adopt if at least one neighbor adopted
                    newly_adopted_this_step.add(node)

        # Update states for the next step
        next_states = current_states.copy()
        for node in newly_adopted_this_step:
            next_states[node] = True

        node_states_over_time.append(next_states)

    # 3. Call plot_network_diffusion to visualize the process
    plot_network_diffusion(
        graph=g,
        node_states_over_time=node_states_over_time,
        title="Simple Network Diffusion",
        node_color_map={False: "skyblue", True: "red"},
        pos=pos,  # Use pre-computed positions
        snapshot_interval=1,  # Plot every step
    )


if __name__ == "__main__":
    run_network_diffusion_example()
