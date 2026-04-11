"""Comprehensive tests for dynamics competition module to improve coverage to >90%."""

import pytest

from innovate.dynamics.competition import CompetitiveInteraction, MarketShareAttraction, ReplicatorDynamics
from innovate.dynamics.competition import LotkaVolterraCompetition as LotkaVolterra


class TestCompetitiveInteraction:
    """Test the CompetitiveInteraction abstract base class."""

    def test_competitive_interaction_is_abstract(self):
        """Test that CompetitiveInteraction is an abstract class that can't be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CompetitiveInteraction()

    def test_competitive_interaction_subclass_must_implement_all_abstract_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteCompetitiveInteraction(CompetitiveInteraction):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCompetitiveInteraction()

        class CompleteCompetitiveInteraction(CompetitiveInteraction):
            def compute_interaction_rates(self, **params):
                return 0.0

            def predict_states(self, time_points, **params):
                # Return dummy states for testing
                return [0.0] * len(time_points)

            def get_parameters_schema(self):
                # Return dummy schema for testing
                return {}

        # This should work without error
        instance = CompleteCompetitiveInteraction()
        assert instance is not None

        # Test the implemented method
        result = instance.compute_interaction_rates(param1=1.0)
        assert result == 0.0


class TestLotkaVolterra:
    """Test the LotkaVolterra class."""

    def test_lotka_volterra_compute_interaction_rates_default(self):
        """Test compute_interaction_rates with default parameters."""
        model = LotkaVolterra()
        # Using the actual parameters for the LotkaVolterraCompetition model
        result = model.compute_interaction_rates(N1=10, N2=20)
        # Expected: r1*N1*(1-(N1+alpha12*N2)/K1), r2*N2*(1-(N2+alpha21*N1)/K2)
        expected_dN1dt = 0.1 * 10 * (1 - (10 + 1.0 * 20) / 1000)  # growth_rate_1*N1*(1-(N1+competition_coeff_12*N2)/K1)
        expected_dN2dt = 0.1 * 20 * (1 - (20 + 1.0 * 10) / 1000)  # growth_rate_2*N2*(1-(N2+competition_coeff_21*N1)/K2)
        assert len(result) == 2
        assert abs(result[0] - expected_dN1dt) < 0.01
        assert abs(result[1] - expected_dN2dt) < 0.01

    def test_lotka_volterra_compute_interaction_rates_with_params(self):
        """Test compute_interaction_rates with custom parameters."""
        model = LotkaVolterra()
        result = model.compute_interaction_rates(
            N1=10, N2=20, growth_rate_1=0.2, growth_rate_2=0.3, competition_coeff_12=0.5, competition_coeff_21=0.8
        )
        # Expected: r1*N1*(1-(N1+alpha12*N2)/K1), r2*N2*(1-(N2+alpha21*N1)/K2)
        expected_dN1dt = 0.2 * 10 * (1 - (10 + 0.5 * 20) / 1000)  # growth_rate_1*N1*(1-(N1+competition_coeff_12*N2)/K1)
        expected_dN2dt = 0.3 * 20 * (1 - (20 + 0.8 * 10) / 1000)  # growth_rate_2*N2*(1-(N2+competition_coeff_21*N1)/K2)
        assert len(result) == 2
        assert abs(result[0] - expected_dN1dt) < 0.01
        assert abs(result[1] - expected_dN2dt) < 0.01

    def test_lotka_volterra_compute_interaction_rates_with_zero_population(self):
        """Test compute_interaction_rates with zero population."""
        model = LotkaVolterra()
        result = model.compute_interaction_rates(N1=0, N2=20, growth_rate_1=0.2, growth_rate_2=0.3)
        # When N1 is 0: 0.2*0*(1-(0+alpha12*20)/K1) = 0
        expected_dN1dt = 0.0
        expected_dN2dt = 0.3 * 20 * (1 - 20 / 1000)  # growth_rate_2*N2*(1-N2/K2)
        assert len(result) == 2
        assert abs(result[0] - expected_dN1dt) < 0.01
        assert abs(result[1] - expected_dN2dt) < 0.01

    def test_lotka_volterra_compute_interaction_rates_edge_cases(self):
        """Test compute_interaction_rates with edge cases."""
        model = LotkaVolterra()
        # Both populations are zero
        result = model.compute_interaction_rates(N1=0, N2=0, growth_rate_1=0.2, growth_rate_2=0.3)
        expected_dN1dt = 0.0  # When N1 is 0, growth_rate_1 * N1 * (1 - N1/K1) = 0
        expected_dN2dt = 0.0  # When N2 is 0, growth_rate_2 * N2 * (1 - N2/K2) = 0
        assert len(result) == 2
        assert abs(result[0] - expected_dN1dt) < 0.01
        assert abs(result[1] - expected_dN2dt) < 0.01


class TestMarketShareAttraction:
    """Test the MarketShareAttraction class."""

    def test_market_share_attraction_compute_interaction_rates(self):
        """Test that the compute_interaction_rates method exists and is properly defined."""
        model = MarketShareAttraction()
        # This model has a special implementation that indicates the method is not applicable
        # The test ensures the abstract method is implemented (even if just to indicate non-applicability)
        assert hasattr(model, "compute_interaction_rates")


class TestReplicatorDynamics:
    """Test the ReplicatorDynamics class."""

    def test_replicator_dynamics_compute_interaction_rates_default(self):
        """Test compute_interaction_rates with default parameters."""
        model = ReplicatorDynamics()
        # Example: 2 strategies with equal proportions and identity payoff matrix
        x = [0.5, 0.5]
        payoff_matrix = [[1, 0], [0, 1]]  # Identity matrix
        result = model.compute_interaction_rates(x=x, payoff_matrix=payoff_matrix)

        # Check that the output has the correct shape (same as input x)
        assert len(result) == len(x)


def test_dynamics_competition_comprehensive():
    """Integration test for all dynamics competition functionality."""
    # Test all three models can be instantiated and have required methods
    lotka = LotkaVolterra()
    assert hasattr(lotka, "compute_interaction_rates")
    assert hasattr(lotka, "predict_states")
    assert hasattr(lotka, "get_parameters_schema")

    msa = MarketShareAttraction()
    assert hasattr(msa, "compute_interaction_rates")
    assert hasattr(msa, "predict_states")
    assert hasattr(msa, "get_parameters_schema")

    replicator = ReplicatorDynamics()
    assert hasattr(replicator, "compute_interaction_rates")
    assert hasattr(replicator, "predict_states")
    assert hasattr(replicator, "get_parameters_schema")


if __name__ == "__main__":
    # Run the tests individually to ensure they work
    test_instance = TestCompetitiveInteraction()
    test_instance_lotka = TestLotkaVolterra()
    test_instance_msa = TestMarketShareAttraction()
    test_instance_replicator = TestReplicatorDynamics()

    print("Running dynamics competition comprehensive tests...")

    test_instance.test_competitive_interaction_is_abstract()
    print("✓ CompetitiveInteraction is abstract test passed")

    test_instance.test_competitive_interaction_subclass_must_implement_all_abstract_methods()
    print("✓ CompetitiveInteraction subclass implementation test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rates_default()
    print("✓ LotkaVolterra default params test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rates_with_params()
    print("✓ LotkaVolterra custom params test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rates_with_zero_population()
    print("✓ LotkaVolterra zero population test passed")

    test_instance_lotka.test_lotka_volterra_compute_interaction_rates_edge_cases()
    print("✓ LotkaVolterra edge cases test passed")

    test_instance_msa.test_market_share_attraction_compute_interaction_rates()
    print("✓ MarketShareAttraction method exists test passed")

    test_instance_replicator.test_replicator_dynamics_compute_interaction_rates_default()
    print("✓ ReplicatorDynamics default params test passed")

    test_dynamics_competition_comprehensive()
    print("✓ Integration test passed")

    print("\nAll comprehensive dynamics competition tests passed! 🎉")
