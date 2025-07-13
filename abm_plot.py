import matplotlib.pyplot as plt
from innovate.abm.model import InnovationModel

# Model parameters
num_agents = 1000
width = 20
height = 20

# Run the model without the intervention
model_no_intervention = InnovationModel(num_agents, width, height)
for i in range(100):
    model_no_intervention.step()

# Run the model with the intervention
model_intervention = InnovationModel(num_agents, width, height)
# Vaccinate 50% of the population
for agent in model_intervention.agents:
    if agent.random.random() < 0.5:
        agent.adopted = True
for i in range(100):
    model_intervention.step()

# Get the data
data_no_intervention = [agent.adopted for agent in model_no_intervention.agents]
data_intervention = [agent.adopted for agent in model_intervention.agents]

# Plot the results
plt.figure(figsize=(10, 6))
plt.hist(data_no_intervention, bins=2, label='No Intervention', alpha=0.5)
plt.hist(data_intervention, bins=2, label='Intervention', alpha=0.5)
plt.title("Agent-Based Model of a Public Health Intervention")
plt.xlabel("Adopted")
plt.ylabel("Number of Agents")
plt.legend()
plt.grid(True)
plt.savefig("docs/images/abm.png")
plt.show()
