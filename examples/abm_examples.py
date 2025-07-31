import matplotlib.pyplot as plt
from innovate.abm import (
    CompetitiveDiffusionModel,
    SentimentHypeModel,
    DisruptiveInnovationModel,
)


def plot_results(df, title):
    """Helper function to plot model results."""
    df.plot()
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Number of Adopters")
    plt.show()


# 1. Competitive Diffusion Example
print("Running Competitive Diffusion Model...")
competitive_model = CompetitiveDiffusionModel(
    num_agents=100,
    width=10,
    height=10,
    num_innovations=3,
)
competitive_results = competitive_model.run_model(n_steps=50)
plot_results(competitive_results, "Competitive Diffusion")

# 2. Sentiment-Driven Hype Cycle Example
print("Running Sentiment-Driven Hype Cycle Model...")
sentiment_model = SentimentHypeModel(
    num_agents=100,
    width=10,
    height=10,
    adoption_threshold=5,
    sentiment_threshold=3,
)
sentiment_results = sentiment_model.run_model(n_steps=50)
plot_results(sentiment_results, "Sentiment-Driven Hype Cycle")

# 3. Disruptive Innovation Example
print("Running Disruptive Innovation Model...")
disruptive_model = DisruptiveInnovationModel(
    num_agents=100,
    width=10,
    height=10,
    initial_disruptive_performance=0.1,
    disruptive_performance_improvement=0.02,
)
disruptive_results = disruptive_model.run_model(n_steps=50)
plot_results(disruptive_results, "Disruptive Innovation")
