from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


def plot_network_diffusion(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] | None = None,
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph representing the network.
    node_states_over_time : list[dict[Any, Any]]
        State snapshots for each time step. Keys are node IDs and values are
        their states, for example adopted or not adopted.
    title : str, default="Network Diffusion"
        The base title for the plots.
    node_color_map : dict[Any, str] | None, optional
        Mapping from node states to colors.
    pos : dict[Any, Any] | None, optional
        Node positions for a consistent layout. If omitted, a spring layout is
        computed.
    snapshot_interval : int, default=1
        How often to save or display a snapshot.
    save_path_prefix : str | None, optional
        If provided, plots are saved as ``"<save_path_prefix>_step_<step>.png"``.
    """
    if node_color_map is None:
        node_color_map = {False: "skyblue", True: "red"}
    if not pos:
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=colors,
                node_size=200,
                alpha=0.9,
            )
            nx.draw_networkx_edges(graph, pos, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()
