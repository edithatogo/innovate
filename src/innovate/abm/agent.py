from mesa import Agent

class InnovationAgent(Agent):
    """An agent in the innovation diffusion model."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Add agent-specific attributes here, e.g.,
        self.adopted = False
        self.awareness = 0.0 # 0.0 to 1.0
        self.influence = 0.0 # How much this agent influences others
        self.susceptibility = 0.0 # How susceptible this agent is to influence

    def step(self):
        """Agent's behavior at each step."""
        # Implement agent's decision-making process here
        pass
