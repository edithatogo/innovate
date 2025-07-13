
from mesa import Model
from mesa.agent import AgentSet
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from .agent import InnovationAgent

class DisruptiveInnovationAgent(InnovationAgent):
    """
    An agent in a disruptive innovation model.
    The agent can choose between an incumbent and a disruptive product.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.choice = None  # 'incumbent' or 'disruptive'

    def step(self):
        """
        The agent's step function.
        The agent's choice is based on the perceived value of each product.
        """
        incumbent_value = self.model.incumbent_performance - self.model.incumbent_price
        disruptive_value = self.model.disruptive_performance - self.model.disruptive_price

        if disruptive_value > incumbent_value:
            self.choice = 'disruptive'
        else:
            self.choice = 'incumbent'


class DisruptiveInnovationModel(Model):
    """
    A model for disruptive innovation.
    """
    def __init__(self, num_agents, width, height, initial_disruptive_performance, disruptive_performance_improvement):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.running = True

        self.incumbent_performance = 1.0
        self.incumbent_price = 0.5
        self.disruptive_performance = initial_disruptive_performance
        self.disruptive_price = 0.2
        self.disruptive_performance_improvement = disruptive_performance_improvement

        # Create agents
        self.agents = AgentSet(self, DisruptiveInnovationAgent)
        for i in range(self.num_agents):
            agent = self.agents.create_agent(unique_id=i)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "IncumbentAdopters": lambda m: sum([1 for a in m.agents if a.choice == 'incumbent']),
                "DisruptiveAdopters": lambda m: sum([1 for a in m.agents if a.choice == 'disruptive']),
            }
        )

    def step(self):
        """
        Run one step of the model.
        """
        self.disruptive_performance += self.disruptive_performance_improvement
        self.datacollector.collect(self)
        self.agents.step()

    def run_model(self, n_steps):
        """
        Run the model for a specified number of steps.
        """
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
