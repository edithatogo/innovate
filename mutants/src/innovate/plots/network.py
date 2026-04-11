from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


def plot_network_diffusion(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    args = [graph, node_states_over_time, title, node_color_map, pos, snapshot_interval, save_path_prefix]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_network_diffusion__mutmut_orig, x_plot_network_diffusion__mutmut_mutants, args, kwargs, None)


def x_plot_network_diffusion__mutmut_orig(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_1(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "XXNetwork DiffusionXX",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_2(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "network diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_3(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "NETWORK DIFFUSION",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_4(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 2,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_5(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
    if pos:
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


def x_plot_network_diffusion__mutmut_6(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = None  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_7(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(None, seed=42)  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_8(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=None)  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_9(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(seed=42)  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_10(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, )  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_11(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=43)  # For reproducible layout

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


def x_plot_network_diffusion__mutmut_12(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(None):
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


def x_plot_network_diffusion__mutmut_13(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i / snapshot_interval == 0:
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


def x_plot_network_diffusion__mutmut_14(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval != 0:
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


def x_plot_network_diffusion__mutmut_15(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 1:
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


def x_plot_network_diffusion__mutmut_16(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=None)

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


def x_plot_network_diffusion__mutmut_17(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(11, 8))

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


def x_plot_network_diffusion__mutmut_18(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 9))

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


def x_plot_network_diffusion__mutmut_19(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = None

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


def x_plot_network_diffusion__mutmut_20(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(None, "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_21(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), None) for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_22(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get("gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_23(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), ) for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_24(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(None, False), "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_25(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, None), "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_26(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(False), "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_27(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, ), "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_28(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, True), "gray") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_29(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "XXgrayXX") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_30(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "GRAY") for node in graph.nodes()]

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


def x_plot_network_diffusion__mutmut_31(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                None,
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


def x_plot_network_diffusion__mutmut_32(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph,
                None,
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


def x_plot_network_diffusion__mutmut_33(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=None,
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


def x_plot_network_diffusion__mutmut_34(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
                node_size=None,
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


def x_plot_network_diffusion__mutmut_35(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
                alpha=None,
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


def x_plot_network_diffusion__mutmut_36(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
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


def x_plot_network_diffusion__mutmut_37(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph,
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


def x_plot_network_diffusion__mutmut_38(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
        pos = nx.spring_layout(graph, seed=42)  # For reproducible layout

    for i, current_states in enumerate(node_states_over_time):
        if i % snapshot_interval == 0:
            plt.figure(figsize=(10, 8))

            # Get colors for nodes based on their current state
            colors = [node_color_map.get(current_states.get(node, False), "gray") for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph,
                pos,
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


def x_plot_network_diffusion__mutmut_39(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_40(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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


def x_plot_network_diffusion__mutmut_41(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
                node_size=201,
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


def x_plot_network_diffusion__mutmut_42(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
                alpha=1.9,
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


def x_plot_network_diffusion__mutmut_43(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(None, pos, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_44(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(graph, None, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_45(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(graph, pos, alpha=None)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_46(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(pos, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_47(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(graph, alpha=0.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_48(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(graph, pos, )
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_49(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_edges(graph, pos, alpha=1.3)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_50(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(None, pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_51(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, None, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_52(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=None, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_53(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color=None)

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_54(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(pos, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_55(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, font_size=8, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_56(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_57(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=8, )

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_58(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=9, font_color="black")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_59(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="XXblackXX")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_60(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color="BLACK")

            plt.title(f"{title} - Time Step {i}")
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_61(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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

            plt.title(None)
            plt.axis("off")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_62(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            plt.axis(None)

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_63(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            plt.axis("XXoffXX")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_64(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
            plt.axis("OFF")

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_step_{i:03d}.png")
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()


def x_plot_network_diffusion__mutmut_65(
    graph: nx.Graph,
    node_states_over_time: list[dict[Any, Any]],
    title: str = "Network Diffusion",
    node_color_map: dict[Any, str] = {False: "skyblue", True: "red"},
    pos: dict[Any, Any] | None = None,
    snapshot_interval: int = 1,
    save_path_prefix: str | None = None,
):
    """Plots snapshots of a network diffusion process over time.

    Args:
    ----
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
                plt.savefig(None)
                plt.close()  # Close plot to prevent display if saving
            else:
                plt.show()

x_plot_network_diffusion__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_network_diffusion__mutmut_1': x_plot_network_diffusion__mutmut_1, 
    'x_plot_network_diffusion__mutmut_2': x_plot_network_diffusion__mutmut_2, 
    'x_plot_network_diffusion__mutmut_3': x_plot_network_diffusion__mutmut_3, 
    'x_plot_network_diffusion__mutmut_4': x_plot_network_diffusion__mutmut_4, 
    'x_plot_network_diffusion__mutmut_5': x_plot_network_diffusion__mutmut_5, 
    'x_plot_network_diffusion__mutmut_6': x_plot_network_diffusion__mutmut_6, 
    'x_plot_network_diffusion__mutmut_7': x_plot_network_diffusion__mutmut_7, 
    'x_plot_network_diffusion__mutmut_8': x_plot_network_diffusion__mutmut_8, 
    'x_plot_network_diffusion__mutmut_9': x_plot_network_diffusion__mutmut_9, 
    'x_plot_network_diffusion__mutmut_10': x_plot_network_diffusion__mutmut_10, 
    'x_plot_network_diffusion__mutmut_11': x_plot_network_diffusion__mutmut_11, 
    'x_plot_network_diffusion__mutmut_12': x_plot_network_diffusion__mutmut_12, 
    'x_plot_network_diffusion__mutmut_13': x_plot_network_diffusion__mutmut_13, 
    'x_plot_network_diffusion__mutmut_14': x_plot_network_diffusion__mutmut_14, 
    'x_plot_network_diffusion__mutmut_15': x_plot_network_diffusion__mutmut_15, 
    'x_plot_network_diffusion__mutmut_16': x_plot_network_diffusion__mutmut_16, 
    'x_plot_network_diffusion__mutmut_17': x_plot_network_diffusion__mutmut_17, 
    'x_plot_network_diffusion__mutmut_18': x_plot_network_diffusion__mutmut_18, 
    'x_plot_network_diffusion__mutmut_19': x_plot_network_diffusion__mutmut_19, 
    'x_plot_network_diffusion__mutmut_20': x_plot_network_diffusion__mutmut_20, 
    'x_plot_network_diffusion__mutmut_21': x_plot_network_diffusion__mutmut_21, 
    'x_plot_network_diffusion__mutmut_22': x_plot_network_diffusion__mutmut_22, 
    'x_plot_network_diffusion__mutmut_23': x_plot_network_diffusion__mutmut_23, 
    'x_plot_network_diffusion__mutmut_24': x_plot_network_diffusion__mutmut_24, 
    'x_plot_network_diffusion__mutmut_25': x_plot_network_diffusion__mutmut_25, 
    'x_plot_network_diffusion__mutmut_26': x_plot_network_diffusion__mutmut_26, 
    'x_plot_network_diffusion__mutmut_27': x_plot_network_diffusion__mutmut_27, 
    'x_plot_network_diffusion__mutmut_28': x_plot_network_diffusion__mutmut_28, 
    'x_plot_network_diffusion__mutmut_29': x_plot_network_diffusion__mutmut_29, 
    'x_plot_network_diffusion__mutmut_30': x_plot_network_diffusion__mutmut_30, 
    'x_plot_network_diffusion__mutmut_31': x_plot_network_diffusion__mutmut_31, 
    'x_plot_network_diffusion__mutmut_32': x_plot_network_diffusion__mutmut_32, 
    'x_plot_network_diffusion__mutmut_33': x_plot_network_diffusion__mutmut_33, 
    'x_plot_network_diffusion__mutmut_34': x_plot_network_diffusion__mutmut_34, 
    'x_plot_network_diffusion__mutmut_35': x_plot_network_diffusion__mutmut_35, 
    'x_plot_network_diffusion__mutmut_36': x_plot_network_diffusion__mutmut_36, 
    'x_plot_network_diffusion__mutmut_37': x_plot_network_diffusion__mutmut_37, 
    'x_plot_network_diffusion__mutmut_38': x_plot_network_diffusion__mutmut_38, 
    'x_plot_network_diffusion__mutmut_39': x_plot_network_diffusion__mutmut_39, 
    'x_plot_network_diffusion__mutmut_40': x_plot_network_diffusion__mutmut_40, 
    'x_plot_network_diffusion__mutmut_41': x_plot_network_diffusion__mutmut_41, 
    'x_plot_network_diffusion__mutmut_42': x_plot_network_diffusion__mutmut_42, 
    'x_plot_network_diffusion__mutmut_43': x_plot_network_diffusion__mutmut_43, 
    'x_plot_network_diffusion__mutmut_44': x_plot_network_diffusion__mutmut_44, 
    'x_plot_network_diffusion__mutmut_45': x_plot_network_diffusion__mutmut_45, 
    'x_plot_network_diffusion__mutmut_46': x_plot_network_diffusion__mutmut_46, 
    'x_plot_network_diffusion__mutmut_47': x_plot_network_diffusion__mutmut_47, 
    'x_plot_network_diffusion__mutmut_48': x_plot_network_diffusion__mutmut_48, 
    'x_plot_network_diffusion__mutmut_49': x_plot_network_diffusion__mutmut_49, 
    'x_plot_network_diffusion__mutmut_50': x_plot_network_diffusion__mutmut_50, 
    'x_plot_network_diffusion__mutmut_51': x_plot_network_diffusion__mutmut_51, 
    'x_plot_network_diffusion__mutmut_52': x_plot_network_diffusion__mutmut_52, 
    'x_plot_network_diffusion__mutmut_53': x_plot_network_diffusion__mutmut_53, 
    'x_plot_network_diffusion__mutmut_54': x_plot_network_diffusion__mutmut_54, 
    'x_plot_network_diffusion__mutmut_55': x_plot_network_diffusion__mutmut_55, 
    'x_plot_network_diffusion__mutmut_56': x_plot_network_diffusion__mutmut_56, 
    'x_plot_network_diffusion__mutmut_57': x_plot_network_diffusion__mutmut_57, 
    'x_plot_network_diffusion__mutmut_58': x_plot_network_diffusion__mutmut_58, 
    'x_plot_network_diffusion__mutmut_59': x_plot_network_diffusion__mutmut_59, 
    'x_plot_network_diffusion__mutmut_60': x_plot_network_diffusion__mutmut_60, 
    'x_plot_network_diffusion__mutmut_61': x_plot_network_diffusion__mutmut_61, 
    'x_plot_network_diffusion__mutmut_62': x_plot_network_diffusion__mutmut_62, 
    'x_plot_network_diffusion__mutmut_63': x_plot_network_diffusion__mutmut_63, 
    'x_plot_network_diffusion__mutmut_64': x_plot_network_diffusion__mutmut_64, 
    'x_plot_network_diffusion__mutmut_65': x_plot_network_diffusion__mutmut_65
}
x_plot_network_diffusion__mutmut_orig.__name__ = 'x_plot_network_diffusion'
