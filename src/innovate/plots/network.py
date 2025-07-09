import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Optional

def plot_network_diffusion(
    graph: nx.Graph,
    node_states_over_time: List[Dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: Dict[Any, str] = {False: 'skyblue', True: 'red'},
    pos: Optional[Dict[Any, Any]] = None,
    snapshot_interval: int = 1,
    save_path_prefix: Optional[str] = None,
):
    """
    Plots snapshots of a network diffusion process over time.

    Args:
        graph: The networkx graph representing the network.
        node_states_over_time: A list of dictionaries, where each dictionary
                               represents the state of nodes at a given time step.
                               Keys are node IDs, values are their states (e.g., True/False for adopted/not adopted).
        title: The base title for the plots.
        node_color_map: A dictionary mapping node states to colors.
        pos: Optional. A dictionary of node positions for consistent layout.
             If None, a spring layout will be computed.
        snapshot_interval: How often to save/display a snapshot (e.g., 1 for every step).
        save_path_prefix: Optional. If provided, plots will be saved as
                          '<save_path_prefix>_step_<step_number>.png'.
    """
    if not pos:
        pos = nx.spring_layout(graph, seed=42) # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))
            
            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), 'gray') for node in graph.nodes()]

            nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=200, alpha=0.9)
            nx.draw_networkx_edges(graph, pos, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black')

            plt.title(f'{title} - Time Step {i}')
            plt.axis('off')
            
            if save_path_prefix:
                plt.savefig(f'{save_path_prefix}_step_{i:03d}.png')
                plt.close() # Close plot to prevent display if saving
            else:
                plt.show()
