import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
from mesa import Model

from .agent import InnovationAgent
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


class NDlibModel(Model):
    """An innovation diffusion model using ndlib for network-based simulations."""

    def __init__(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        args = [num_agents, graph, model_name]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNDlibModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁNDlibModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁNDlibModelǁ__init____mutmut_orig(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_1(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "XXicXX",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_2(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "IC",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_3(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = None
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_4(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = None

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_5(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = False

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_6(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is not None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_7(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = None
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_8(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=None, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_9(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=None)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_10(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_11(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, )
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_12(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=1.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_13(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = None

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_14(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(None):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_15(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = None
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_16(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=None, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_17(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=None)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_18(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_19(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, )
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_20(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = None

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_21(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["XXagentXX"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_22(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["AGENT"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_23(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name != "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_24(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "XXicXX":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_25(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "IC":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_26(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = None
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_27(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(None)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_28(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name != "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_29(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "XXltXX":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_30(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "LT":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_31(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = None
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_32(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(None)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_33(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name != "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_34(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "XXsirXX":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_35(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "SIR":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_36(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = None
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_37(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(None)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_38(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name != "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_39(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "XXsisXX":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_40(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "SIS":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_41(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = None
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_42(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(None)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_43(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(None)

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_44(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = None
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_45(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration(None, [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_46(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", None)
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_47(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration([0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_48(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", )
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_49(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("XXInfectedXX", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_50(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("infected", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_51(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("INFECTED", [0])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_52(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [1])
        self.diffusion_model.set_initial_status(config)

    def xǁNDlibModelǁ__init____mutmut_53(
        self,
        num_agents,
        graph: nx.Graph | None = None,
        model_name: str = "ic",
    ):
        """Initialize the NDlibModel with a specified number of agents, network graph, and diffusion model type.

        Creates a network-based innovation diffusion simulation by assigning agents to nodes in the provided or generated graph and initializing the chosen NDlib diffusion model. The initial state marks node 0 as 'Infected' (adopted). Supported diffusion models are Independent Cascades ("ic"), Linear Threshold ("lt"), SIR ("sir"), and SIS ("sis").

        Parameters
        ----------
            num_agents (int): Number of agents (nodes) in the simulation.
            graph (nx.Graph, optional): NetworkX graph to use for the simulation. If None, an Erdős-Rényi random graph is generated.
            model_name (str, optional): Name of the diffusion model to use ("ic", "lt", "sir", or "sis"). Defaults to "ic".

        Raises
        ------
            ValueError: If an unsupported model_name is provided.
        """
        super().__init__()
        self.num_agents = num_agents
        self.running = True

        # Create a networkx graph if one is not provided
        if graph is None:
            self.graph = nx.erdos_renyi_graph(n=self.num_agents, p=0.1)
        else:
            self.graph = graph

        # Create agents and add them as nodes to the graph
        for i in range(self.num_agents):
            agent = InnovationAgent(unique_id=i, model=self)
            self.graph.nodes[i]["agent"] = agent

        # Initialize the ndlib diffusion model
        if model_name == "ic":
            self.diffusion_model = ep.IndependentCascadesModel(self.graph)
        elif model_name == "lt":
            self.diffusion_model = ep.ThresholdModel(self.graph)
        elif model_name == "sir":
            self.diffusion_model = ep.SIRModel(self.graph)
        elif model_name == "sis":
            self.diffusion_model = ep.SISModel(self.graph)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Set up the initial state of the diffusion model using a
        # Configuration object. Infect a single node to start the cascade.
        config = mc.Configuration()
        config.add_model_initial_configuration("Infected", [0])
        self.diffusion_model.set_initial_status(None)
    
    xǁNDlibModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNDlibModelǁ__init____mutmut_1': xǁNDlibModelǁ__init____mutmut_1, 
        'xǁNDlibModelǁ__init____mutmut_2': xǁNDlibModelǁ__init____mutmut_2, 
        'xǁNDlibModelǁ__init____mutmut_3': xǁNDlibModelǁ__init____mutmut_3, 
        'xǁNDlibModelǁ__init____mutmut_4': xǁNDlibModelǁ__init____mutmut_4, 
        'xǁNDlibModelǁ__init____mutmut_5': xǁNDlibModelǁ__init____mutmut_5, 
        'xǁNDlibModelǁ__init____mutmut_6': xǁNDlibModelǁ__init____mutmut_6, 
        'xǁNDlibModelǁ__init____mutmut_7': xǁNDlibModelǁ__init____mutmut_7, 
        'xǁNDlibModelǁ__init____mutmut_8': xǁNDlibModelǁ__init____mutmut_8, 
        'xǁNDlibModelǁ__init____mutmut_9': xǁNDlibModelǁ__init____mutmut_9, 
        'xǁNDlibModelǁ__init____mutmut_10': xǁNDlibModelǁ__init____mutmut_10, 
        'xǁNDlibModelǁ__init____mutmut_11': xǁNDlibModelǁ__init____mutmut_11, 
        'xǁNDlibModelǁ__init____mutmut_12': xǁNDlibModelǁ__init____mutmut_12, 
        'xǁNDlibModelǁ__init____mutmut_13': xǁNDlibModelǁ__init____mutmut_13, 
        'xǁNDlibModelǁ__init____mutmut_14': xǁNDlibModelǁ__init____mutmut_14, 
        'xǁNDlibModelǁ__init____mutmut_15': xǁNDlibModelǁ__init____mutmut_15, 
        'xǁNDlibModelǁ__init____mutmut_16': xǁNDlibModelǁ__init____mutmut_16, 
        'xǁNDlibModelǁ__init____mutmut_17': xǁNDlibModelǁ__init____mutmut_17, 
        'xǁNDlibModelǁ__init____mutmut_18': xǁNDlibModelǁ__init____mutmut_18, 
        'xǁNDlibModelǁ__init____mutmut_19': xǁNDlibModelǁ__init____mutmut_19, 
        'xǁNDlibModelǁ__init____mutmut_20': xǁNDlibModelǁ__init____mutmut_20, 
        'xǁNDlibModelǁ__init____mutmut_21': xǁNDlibModelǁ__init____mutmut_21, 
        'xǁNDlibModelǁ__init____mutmut_22': xǁNDlibModelǁ__init____mutmut_22, 
        'xǁNDlibModelǁ__init____mutmut_23': xǁNDlibModelǁ__init____mutmut_23, 
        'xǁNDlibModelǁ__init____mutmut_24': xǁNDlibModelǁ__init____mutmut_24, 
        'xǁNDlibModelǁ__init____mutmut_25': xǁNDlibModelǁ__init____mutmut_25, 
        'xǁNDlibModelǁ__init____mutmut_26': xǁNDlibModelǁ__init____mutmut_26, 
        'xǁNDlibModelǁ__init____mutmut_27': xǁNDlibModelǁ__init____mutmut_27, 
        'xǁNDlibModelǁ__init____mutmut_28': xǁNDlibModelǁ__init____mutmut_28, 
        'xǁNDlibModelǁ__init____mutmut_29': xǁNDlibModelǁ__init____mutmut_29, 
        'xǁNDlibModelǁ__init____mutmut_30': xǁNDlibModelǁ__init____mutmut_30, 
        'xǁNDlibModelǁ__init____mutmut_31': xǁNDlibModelǁ__init____mutmut_31, 
        'xǁNDlibModelǁ__init____mutmut_32': xǁNDlibModelǁ__init____mutmut_32, 
        'xǁNDlibModelǁ__init____mutmut_33': xǁNDlibModelǁ__init____mutmut_33, 
        'xǁNDlibModelǁ__init____mutmut_34': xǁNDlibModelǁ__init____mutmut_34, 
        'xǁNDlibModelǁ__init____mutmut_35': xǁNDlibModelǁ__init____mutmut_35, 
        'xǁNDlibModelǁ__init____mutmut_36': xǁNDlibModelǁ__init____mutmut_36, 
        'xǁNDlibModelǁ__init____mutmut_37': xǁNDlibModelǁ__init____mutmut_37, 
        'xǁNDlibModelǁ__init____mutmut_38': xǁNDlibModelǁ__init____mutmut_38, 
        'xǁNDlibModelǁ__init____mutmut_39': xǁNDlibModelǁ__init____mutmut_39, 
        'xǁNDlibModelǁ__init____mutmut_40': xǁNDlibModelǁ__init____mutmut_40, 
        'xǁNDlibModelǁ__init____mutmut_41': xǁNDlibModelǁ__init____mutmut_41, 
        'xǁNDlibModelǁ__init____mutmut_42': xǁNDlibModelǁ__init____mutmut_42, 
        'xǁNDlibModelǁ__init____mutmut_43': xǁNDlibModelǁ__init____mutmut_43, 
        'xǁNDlibModelǁ__init____mutmut_44': xǁNDlibModelǁ__init____mutmut_44, 
        'xǁNDlibModelǁ__init____mutmut_45': xǁNDlibModelǁ__init____mutmut_45, 
        'xǁNDlibModelǁ__init____mutmut_46': xǁNDlibModelǁ__init____mutmut_46, 
        'xǁNDlibModelǁ__init____mutmut_47': xǁNDlibModelǁ__init____mutmut_47, 
        'xǁNDlibModelǁ__init____mutmut_48': xǁNDlibModelǁ__init____mutmut_48, 
        'xǁNDlibModelǁ__init____mutmut_49': xǁNDlibModelǁ__init____mutmut_49, 
        'xǁNDlibModelǁ__init____mutmut_50': xǁNDlibModelǁ__init____mutmut_50, 
        'xǁNDlibModelǁ__init____mutmut_51': xǁNDlibModelǁ__init____mutmut_51, 
        'xǁNDlibModelǁ__init____mutmut_52': xǁNDlibModelǁ__init____mutmut_52, 
        'xǁNDlibModelǁ__init____mutmut_53': xǁNDlibModelǁ__init____mutmut_53
    }
    xǁNDlibModelǁ__init____mutmut_orig.__name__ = 'xǁNDlibModelǁ__init__'

    def step(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNDlibModelǁstep__mutmut_orig'), object.__getattribute__(self, 'xǁNDlibModelǁstep__mutmut_mutants'), args, kwargs, self)

    def xǁNDlibModelǁstep__mutmut_orig(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "Infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_1(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = None
            if status == "Infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_2(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["XXagentXX"]
            if status == "Infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_3(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["AGENT"]
            if status == "Infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_4(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status != "Infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_5(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "XXInfectedXX":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_6(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "infected":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_7(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "INFECTED":
                agent.adopted = True

    def xǁNDlibModelǁstep__mutmut_8(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "Infected":
                agent.adopted = None

    def xǁNDlibModelǁstep__mutmut_9(self):
        """Run one step of the diffusion model."""
        self.diffusion_model.iteration()

        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]["agent"]
            if status == "Infected":
                agent.adopted = False
    
    xǁNDlibModelǁstep__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNDlibModelǁstep__mutmut_1': xǁNDlibModelǁstep__mutmut_1, 
        'xǁNDlibModelǁstep__mutmut_2': xǁNDlibModelǁstep__mutmut_2, 
        'xǁNDlibModelǁstep__mutmut_3': xǁNDlibModelǁstep__mutmut_3, 
        'xǁNDlibModelǁstep__mutmut_4': xǁNDlibModelǁstep__mutmut_4, 
        'xǁNDlibModelǁstep__mutmut_5': xǁNDlibModelǁstep__mutmut_5, 
        'xǁNDlibModelǁstep__mutmut_6': xǁNDlibModelǁstep__mutmut_6, 
        'xǁNDlibModelǁstep__mutmut_7': xǁNDlibModelǁstep__mutmut_7, 
        'xǁNDlibModelǁstep__mutmut_8': xǁNDlibModelǁstep__mutmut_8, 
        'xǁNDlibModelǁstep__mutmut_9': xǁNDlibModelǁstep__mutmut_9
    }
    xǁNDlibModelǁstep__mutmut_orig.__name__ = 'xǁNDlibModelǁstep'
