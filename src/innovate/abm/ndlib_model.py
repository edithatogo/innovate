
import networkx as nx
import ndlib.models.epidemics as ep
from mesa import Model
from .agent import InnovationAgent

class NDlibModel(Model):
    """
    An innovation diffusion model using ndlib for network-based simulations.
    """
    def __init__(self, num_agents, graph: nx.Graph = None, model_name: str = "ic"):
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
            self.graph.nodes[i]['agent'] = agent

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

        # Set up the initial state of the diffusion model
        # For example, infect a single node to start the cascade
        self.diffusion_model.set_initial_status({
            'Infected': [0] # Start with agent 0 as 'Infected' (adopted)
        })

    def step(self):
        """
        Run one step of the diffusion model.
        """
        self.diffusion_model.iteration()
        
        # Update the state of the Mesa agents based on the ndlib model
        for node_id, status in self.diffusion_model.status.items():
            agent = self.graph.nodes[node_id]['agent']
            if status == 'Infected':
                agent.adopted = True
