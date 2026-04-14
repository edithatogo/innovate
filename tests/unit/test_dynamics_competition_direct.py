"""Tests for the main dynamics competition module (not the subdirectory) to improve coverage to >90%."""

import importlib.util
import os
import sys

import pytest

# Add the src directory to the path to allow importing the main competition.py file
# __file__ is this test file's actual path; tests/unit/ -> ../.. -> project root
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

# Import the main competition.py module (not the competition/ directory) directly
competition_path = os.path.join(PROJECT_ROOT, "src", "innovate", "dynamics", "competition.py")

spec = importlib.util.spec_from_file_location("competition_main", competition_path)
competition_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(competition_module)

# Get the classes from the imported module
CompetitiveInteraction = competition_module.CompetitiveInteraction
LotkaVolterra = competition_module.LotkaVolterra
MarketShareAttraction = competition_module.MarketShareAttraction
ReplicatorDynamics = competition_module.ReplicatorDynamics


class TestCompetitiveInteraction:
    """Test the CompetitiveInteraction abstract base class."""

    def test_competitive_interaction_is_abstract(self):
        """Test that CompetitiveInteraction is an abstract class that can't be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CompetitiveInteraction()

    def test_competitive_interaction_subclass_must_implement_compute_interaction_rate(self):
        """Test that subclasses must implement compute_interaction_rate method."""

        class IncompleteCompetitiveInteraction(CompetitiveInteraction):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCompetitiveInteraction()

        class CompleteCompetitiveInteraction(CompetitiveInteraction):
            def compute_interaction_rate(
                self,
                population1: float,
                population2: float,
                **params,
            ):
                return 0.0

        # This should work without error
        instance = CompleteCompetitiveInteraction()
        assert instance is not None

        # Test the implemented method
        result = instance.compute_interaction_rate(10, 20, param1=1.0)
        assert result == 0.0


class TestLotkaVolterra:
    """Test the LotkaVolterra class from the main competition module."""

    def test_lotka_volterra_compute_interaction_rate_default(self):
        """Test compute_interaction_rate with default parameters."""
        model = LotkaVolterra()
        result = model.compute_interaction_rate(10, 20)  # population1=10, population2=20
        expected = 0.1 * 10 * 20  # alpha (default 0.1) * population1 * population2
        assert result == expected

    def test_lotka_volterra_compute_interaction_rate_with_params(self):
        """Test compute_interaction_rate with custom alpha parameter."""
        model = LotkaVolterra()
        result = model.compute_interaction_rate(10, 20, alpha=0.5)
        expected = 0.5 * 10 * 20  # alpha * population1 * population2
        assert result == expected

    def test_lotka_volterra_compute_interaction_rate_with_zero_population(self):
        """Test compute_interaction_rate with zero population."""
        model = LotkaVolterra()
        result = model.compute_interaction_rate(0, 20, alpha=0.5)
        assert result == 0.0  # Should be 0 when one population is 0

    def test_lotka_volterra_compute_interaction_rate_edge_cases(self):
        """Test compute_interaction_rate with edge cases."""
        model = LotkaVolterra()
        # Both populations are zero
        result = model.compute_interaction_rate(0, 0, alpha=0.5)
        assert result == 0.0

        # Very small populations
        result = model.compute_interaction_rate(0.001, 0.001, alpha=1.0)
        assert result == 0.000001  # 1.0 * 0.001 * 0.001


class TestMarketShareAttraction:
    """Test the MarketShareAttraction class from the main competition module."""

    def test_market_share_attraction_compute_interaction_rate_default(self):
        """Test compute_interaction_rate with default parameters."""
        model = MarketShareAttraction()
        result = model.compute_interaction_rate(10, 20)  # population1=10, population2=20
        expected = 0.1 * 10 - 0.1 * 20  # attraction1 * population1 - attraction2 * population2
        assert result == expected

    def test_market_share_attraction_compute_interaction_rate_with_params(self):
        """Test compute_interaction_rate with custom attraction parameters."""
        model = MarketShareAttraction()
        result = model.compute_interaction_rate(10, 20, attraction1=0.3, attraction2=0.2)
        expected = 0.3 * 10 - 0.2 * 20  # attraction1 * population1 - attraction2 * population2
        assert result == expected

    def test_market_share_attraction_compute_interaction_rate_zero_attraction(self):
        """Test compute_interaction_rate with zero attraction."""
        model = MarketShareAttraction()
        result = model.compute_interaction_rate(10, 20, attraction1=0.0, attraction2=0.0)
        assert result == 0.0  # Should be 0 when attractions are 0

    def test_market_share_attraction_compute_interaction_rate_edge_cases(self):
        """Test compute_interaction_rate with edge cases."""
        model = MarketShareAttraction()
        # Both attractions are 0
        result = model.compute_interaction_rate(0, 0, attraction1=0.5, attraction2=0.3)
        assert result == 0.0  # 0.5 * 0 - 0.3 * 0 = 0

        # Very small attractions
        result = model.compute_interaction_rate(10, 20, attraction1=0.001, attraction2=0.002)
        expected = 0.001 * 10 - 0.002 * 20
        assert result == expected


class TestReplicatorDynamics:
    """Test the ReplicatorDynamics class from the main competition module."""

    def test_replicator_dynamics_compute_interaction_rate_default(self):
        """Test compute_interaction_rate with default parameters."""
        model = ReplicatorDynamics()
        # population1=10, population2=20
        # fitness1=0.1, fitness2=0.1 (defaults)
        # average_fitness = (0.1 * 10 + 0.1 * 20) / (10 + 20) = (1 + 2) / 30 = 3/30 = 0.1
        # result = 10 * (0.1 - 0.1) = 0
        result = model.compute_interaction_rate(10, 20)
        expected = 10 * (0.1 - 0.1)  # population1 * (fitness1 - average_fitness)
        assert result == expected

    def test_replicator_dynamics_compute_interaction_rate_with_params(self):
        """Test compute_interaction_rate with custom fitness parameters."""
        model = ReplicatorDynamics()
        # population1=10, population2=20
        # fitness1=0.5, fitness2=0.3
        # average_fitness = (0.5 * 10 + 0.3 * 20) / (10 + 20) = (5 + 6) / 30 = 11/30
        # result = 10 * (0.5 - 11/30) = 10 * (15/30 - 11/30) = 10 * 4/30 = 40/30 = 4/3
        result = model.compute_interaction_rate(10, 20, fitness1=0.5, fitness2=0.3)
        average_fitness = (0.5 * 10 + 0.3 * 20) / (10 + 20)
        expected = 10 * (0.5 - average_fitness)
        assert abs(result - expected) < 1e-10  # Use approximate comparison due to float precision

    def test_replicator_dynamics_compute_interaction_rate_equal_fitness(self):
        """Test compute_interaction_rate when both fitness values are equal."""
        model = ReplicatorDynamics()
        # When fitness1 == fitness2, the result should be 0 regardless of populations
        result = model.compute_interaction_rate(10, 20, fitness1=0.5, fitness2=0.5)
        assert abs(result) < 1e-10  # Should be very close to 0 due to float precision

    def test_replicator_dynamics_compute_interaction_rate_edge_cases(self):
        """Test compute_interaction_rate with edge cases."""
        model = ReplicatorDynamics()
        # When population1 is 0, result should be 0
        result = model.compute_interaction_rate(0, 20, fitness1=0.5, fitness2=0.3)
        assert abs(result) < 1e-10  # Should be 0 (or very close due to floating point)

        # Test with very small populations close to 0
        result = model.compute_interaction_rate(0.0001, 20, fitness1=0.5, fitness2=0.3)
        # This will be a very small number since it's 0.0001 * (0.5 - avg_fitness)
        average_fitness = (0.5 * 0.0001 + 0.3 * 20) / (0.0001 + 20)
        expected = 0.0001 * (0.5 - average_fitness)
        assert abs(result - expected) < 1e-10


def test_dynamics_competition_comprehensive():
    """Integration test for all dynamics competition functionality."""
    # Test all three models with various parameter combinations

    # Lotka-Volterra
    lotka = LotkaVolterra()
    assert lotka.compute_interaction_rate(10, 15) == 0.1 * 10 * 15

    # Market Share Attraction
    msa = MarketShareAttraction()
    result = msa.compute_interaction_rate(10, 15, attraction1=0.2, attraction2=0.1)
    expected = 0.2 * 10 - 0.1 * 15  # 2 - 1.5 = 0.5
    assert result == expected

    # Replicator Dynamics
    replicator = ReplicatorDynamics()
    result = replicator.compute_interaction_rate(10, 10, fitness1=0.6, fitness2=0.4)
    # average_fitness = (0.6 * 10 + 0.4 * 10) / (10 + 10) = (6 + 4) / 20 = 0.5
    # result = 10 * (0.6 - 0.5) = 10 * 0.1 = 1
    expected = 10 * (0.6 - 0.5)  # 1.0
    assert abs(result - expected) < 1e-10


if __name__ == "__main__":
    # Run the tests individually to ensure they work
    test_instance = TestCompetitiveInteraction()
    test_instance_lotka = TestLotkaVolterra()
    test_instance_msa = TestMarketShareAttraction()
    test_instance_replicator = TestReplicatorDynamics()

    print("Running dynamics competition comprehensive tests (direct import)...")

    test_instance.test_competitive_interaction_is_abstract()
    print("✓ CompetitiveInteraction is abstract test passed")

    test_instance.test_competitive_interaction_subclass_must_implement_compute_interaction_rate()
    print("✓ CompetitiveInteraction subclass implementation test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rate_default()
    print("✓ LotkaVolterra default params test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rate_with_params()
    print("✓ LotkaVolterra custom params test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rate_with_zero_population()
    print("✓ LotkaVolterra zero population test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rate_edge_cases()
    print("✓ LotkaVolterra edge cases test passed")

    test_instance_msa.test_market_share_attraction_compute_interaction_rate_default()
    print("✓ MarketShareAttraction default params test passed")

    test_instance_msa.test_market_share_attraction_compute_interaction_rate_with_params()
    print("✓ MarketShareAttraction custom params test passed")

    test_instance_msa.test_market_share_attraction_compute_interaction_rate_zero_attraction()
    print("✓ MarketShareAttraction zero attraction test passed")

    test_instance_msa.test_market_share_attraction_compute_interaction_rate_edge_cases()
    print("✓ MarketShareAttraction edge cases test passed")

    test_instance_replicator.test_replicator_dynamics_compute_interaction_rate_default()
    print("✓ ReplicatorDynamics default params test passed")

    test_instance_replicator.test_replicator_dynamics_compute_interaction_rate_with_params()
    print("✓ ReplicatorDynamics custom params test passed")

    test_instance_replicator.test_replicator_dynamics_compute_interaction_rate_equal_fitness()
    print("✓ ReplicatorDynamics equal fitness test passed")

    test_instance_replicator.test_replicator_dynamics_compute_interaction_rate_edge_cases()
    print("✓ ReplicatorDynamics edge cases test passed")

    test_dynamics_competition_comprehensive()
    print("✓ Integration test passed")

    print("\nAll comprehensive dynamics competition tests (direct import) passed! 🎉")
