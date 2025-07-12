
from mesa import Model
from mesa.agent import AgentSet
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from .agent import InnovationAgent

class CompetitiveDiffusionAgent(InnovationAgent):
    """
    An agent in a competitive diffusion model.
    The agent can adopt one of several competing innovations.
    """
    def __init__(self, unique_id, model, num_innovations):
        super().__init__(unique_id, model)
        self.adopted_innovation = -1  # -1 means no adoption, 0, 1, ... for innovations

    def step(self):
        """
        The agent's step function.
        The agent's decision to adopt is based on its neighbors' adoptions.
        """
        if self.adopted_innovation != -1:
            return  # Already adopted

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            return

        # Check neighbors' adoptions
        adoptions = [n.adopted_innovation for n in neighbors if n.adopted_innovation != -1]
        if not adoptions:
            return

        # Simple adoption rule: adopt the most popular innovation among neighbors
        # More complex rules can be added here (e.g., based on influence, susceptibility)
        most_popular = max(set(adoptions), key=adoptions.count)
        self.adopted_innovation = most_popular


class CompetitiveDiffusionModel(Model):
    """
    A model for competitive diffusion of multiple innovations.
    """
    def __init__(self, num_agents, width, height, num_innovations):
        self.num_agents = num_agents
        self.num_innovations = num_innovations
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        self.agents = AgentSet(self, CompetitiveDiffusionAgent, {"num_innovations": num_innovations})
        for i in range(self.num_agents):
            agent = self.agents.create_agent(unique_id=i)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Seed initial adopters for each innovation
        for i in range(self.num_innovations):
            agent = self.random.choice(list(self.agents))
            agent.adopted_innovation = i

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={"AdoptionCounts": lambda m: [sum(1 for a in m.agents if a.adopted_innovation == i) for i in range(m.num_innovations)]}
        )

    def step(self):
        """
        Run one step of the model.
        """
        self.datacollector.collect(self)
        self.agents.step()

    def run_model(self, n_steps):
        """
        Run the model for a specified number of steps.
        """
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
