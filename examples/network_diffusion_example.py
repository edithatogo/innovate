import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from innovate.plots.network import plot_network_diffusion

def run_network_diffusion_example():
    print("--- Running Network Diffusion Example ---")

    # 1. Create a sample networkx graph (e.g., a small-world network)
    num_nodes = 20
    G = nx.watts_strogatz_graph(num_nodes, k=4, p=0.3, seed=42)

    # For consistent layout across plots
    pos = nx.spring_layout(G, seed=42)

    # 2. Simulate a simple diffusion process
    # Start with one random adopted node
    adopted_nodes = {np.random.choice(list(G.nodes())): True}
    
    # Initialize all other nodes as not adopted
    initial_states = {node: adopted_nodes.get(node, False) for node in G.nodes()}
    node_states_over_time = [initial_states]

    # Simulate for a few steps
    num_steps = 10
    for step in range(num_steps - 1):
        current_states = node_states_over_time[-1].copy()
        newly_adopted_this_step = set()

        for node in G.nodes():
            if current_states[node] == False: # If not yet adopted
                # Check if any adopted neighbors exist
                adopted_neighbors = [n for n in G.neighbors(node) if current_states.get(n, False) == True]
                if adopted_neighbors: # Simple rule: adopt if at least one neighbor adopted
                    newly_adopted_this_step.add(node)
        
        # Update states for the next step
        next_states = current_states.copy()
        for node in newly_adopted_this_step:
            next_states[node] = True
        
        node_states_over_time.append(next_states)

    # 3. Call plot_network_diffusion to visualize the process
    print(f"Simulated {num_steps} steps of diffusion. Generating plots...")
    plot_network_diffusion(
        graph=G,
        node_states_over_time=node_states_over_time,
        title="Simple Network Diffusion",
        node_color_map={False: 'skyblue', True: 'red'},
        pos=pos, # Use pre-computed positions
        snapshot_interval=1, # Plot every step
        # save_path_prefix="network_diffusion_snapshot" # Uncomment to save images
    )
    print("Network diffusion visualization complete.")

if __name__ == "__main__":
    run_network_diffusion_example()
