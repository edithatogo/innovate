from mesa import Model
from mesa.agent import AgentSet
from mesa.space import MultiGrid
from .agent import InnovationAgent

class InnovationModel(Model):
    """A model for innovation diffusion."""

    def __init__(self, num_agents, width, height):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True  # For visualization/interactive mode

        self.agents = AgentSet(self)
        # Create agents
        for i in range(self.num_agents):
            agent = InnovationAgent(i, self)
            self.agents.add(agent)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        """Run one step of the model."""
        self.agents.step()
