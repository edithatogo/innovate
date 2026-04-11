"""Tests for the main dynamics contagion module (not the subdirectory) to improve coverage to >90%."""

import os
import sys

# Add the src directory to sys.path so we can import modules properly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Now import the contagion models properly
from innovate.dynamics.contagion import SEIRModel as SEIR
from innovate.dynamics.contagion import SIRModel as SIR
from innovate.dynamics.contagion import SISModel as SIS


class TestSIR:
    """Test the SIR class from the main contagion module."""

    def test_sir_initialization(self):
        """Test SIR initialization - SIRModel doesn't take parameters in constructor."""
        model = SIR()
        # SIRModel doesn't have direct access to beta and gamma like the test expects
        # So we just test that it initializes without error and has necessary methods
        assert model is not None
        assert hasattr(model, "compute_spread_rate")
        assert hasattr(model, "predict_states")
        assert hasattr(model, "get_parameters_schema")

    def test_sir_compute_spread_rate_default(self):
        """Test the compute_spread_rate method with default parameters."""
        model = SIR()
        result = model.compute_spread_rate(S=990.0, I=10.0)
        # Default transmission_rate=0.1, recovery_rate=0.01
        # dSdt = -beta*S*I = -0.1*990*10 = -990
        # dIdt = beta*S*I - gamma*I = 0.1*990*10 - 0.01*10 = 990 - 0.1 = 989.9
        # dRdt = gamma*I = 0.01*10 = 0.1
        expected_dSdt = -0.1 * 990 * 10
        expected_dIdt = 0.1 * 990 * 10 - 0.01 * 10
        expected_dRdt = 0.01 * 10
        assert len(result) == 3
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dIdt) < 0.1
        assert abs(result[2] - expected_dRdt) < 0.1

    def test_sir_compute_spread_rate_custom_params(self):
        """Test the compute_spread_rate method with custom parameters."""
        model = SIR()
        result = model.compute_spread_rate(S=800.0, I=150.0, transmission_rate=0.15, recovery_rate=0.05)
        # dSdt = -beta*S*I = -0.15*800*150 = -18000
        # dIdt = beta*S*I - gamma*I = 0.15*800*150 - 0.05*150 = 18000 - 7.5 = 17992.5
        # dRdt = gamma*I = 0.05*150 = 7.5
        expected_dSdt = -0.15 * 800 * 150
        expected_dIdt = 0.15 * 800 * 150 - 0.05 * 150
        expected_dRdt = 0.05 * 150
        assert len(result) == 3
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dIdt) < 0.1
        assert abs(result[2] - expected_dRdt) < 0.1


class TestSIS:
    """Test the SIS class from the main contagion module."""

    def test_sis_initialization(self):
        """Test SIS initialization - SISModel doesn't take parameters in constructor."""
        model = SIS()
        assert model is not None
        assert hasattr(model, "compute_spread_rate")
        assert hasattr(model, "predict_states")
        assert hasattr(model, "get_parameters_schema")

    def test_sis_compute_spread_rate_default(self):
        """Test the compute_spread_rate method with default parameters."""
        model = SIS()
        result = model.compute_spread_rate(S=990.0, I=10.0)
        # Default transmission_rate=0.1, recovery_rate=0.01
        # dSdt = -beta*S*I + gamma*I = -0.1*990*10 + 0.01*10 = -990 + 0.1 = -989.9
        # dIdt = beta*S*I - gamma*I = 0.1*990*10 - 0.01*10 = 990 - 0.1 = 989.9
        expected_dSdt = -0.1 * 990 * 10 + 0.01 * 10
        expected_dIdt = 0.1 * 990 * 10 - 0.01 * 10
        assert len(result) == 2
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dIdt) < 0.1

    def test_sis_compute_spread_rate_custom_params(self):
        """Test the compute_spread_rate method with custom parameters."""
        model = SIS()
        result = model.compute_spread_rate(S=800.0, I=200.0, transmission_rate=0.12, recovery_rate=0.08)
        # dSdt = -beta*S*I + gamma*I = -0.12*800*200 + 0.08*200 = -19200 + 16 = -19184
        # dIdt = beta*S*I - gamma*I = 0.12*800*200 - 0.08*200 = 19200 - 16 = 19184
        expected_dSdt = -0.12 * 800 * 200 + 0.08 * 200
        expected_dIdt = 0.12 * 800 * 200 - 0.08 * 200
        assert len(result) == 2
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dIdt) < 0.1


class TestSEIR:
    """Test the SEIR class from the main contagion module."""

    def test_seir_initialization(self):
        """Test SEIR initialization - SEIRModel doesn't take parameters in constructor."""
        model = SEIR()
        assert model is not None
        assert hasattr(model, "compute_spread_rate")
        assert hasattr(model, "predict_states")
        assert hasattr(model, "get_parameters_schema")

    def test_seir_compute_spread_rate_default(self):
        """Test the compute_spread_rate method with default parameters."""
        model = SEIR()
        result = model.compute_spread_rate(S=980.0, E=10.0, I=5.0)
        # Default transmission_rate=0.1, incubation_rate=0.1, recovery_rate=0.01
        # dSdt = -beta*S*I = -0.1*980*5 = -490
        # dEdt = beta*S*I - alpha*E = 0.1*980*5 - 0.1*10 = 490 - 1 = 489
        # dIdt = alpha*E - gamma*I = 0.1*10 - 0.01*5 = 1 - 0.05 = 0.95
        # dRdt = gamma*I = 0.01*5 = 0.05
        expected_dSdt = -0.1 * 980 * 5
        expected_dEdt = 0.1 * 980 * 5 - 0.1 * 10
        expected_dIdt = 0.1 * 10 - 0.01 * 5
        expected_dRdt = 0.01 * 5
        assert len(result) == 4
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dEdt) < 0.1
        assert abs(result[2] - expected_dIdt) < 0.1
        assert abs(result[3] - expected_dRdt) < 0.1

    def test_seir_compute_spread_rate_custom_params(self):
        """Test the compute_spread_rate method with custom parameters."""
        model = SEIR()
        result = model.compute_spread_rate(
            S=700.0, E=150.0, I=100.0, transmission_rate=0.08, incubation_rate=0.15, recovery_rate=0.05
        )
        # dSdt = -beta*S*I = -0.08*700*100 = -5600
        # dEdt = beta*S*I - alpha*E = 0.08*700*100 - 0.15*150 = 5600 - 22.5 = 5577.5
        # dIdt = alpha*E - gamma*I = 0.15*150 - 0.05*100 = 22.5 - 5 = 17.5
        # dRdt = gamma*I = 0.05*100 = 5
        expected_dSdt = -0.08 * 700 * 100
        expected_dEdt = 0.08 * 700 * 100 - 0.15 * 150
        expected_dIdt = 0.15 * 150 - 0.05 * 100
        expected_dRdt = 0.05 * 100
        assert len(result) == 4
        assert abs(result[0] - expected_dSdt) < 0.1
        assert abs(result[1] - expected_dEdt) < 0.1
        assert abs(result[2] - expected_dIdt) < 0.1
        assert abs(result[3] - expected_dRdt) < 0.1


def test_contagion_comprehensive():
    """Integration test for all dynamics contagion functionality."""
    # Test all three models with various parameter combinations

    # SIR model
    sir = SIR()
    result_sir = sir.compute_spread_rate(S=900, I=100)
    assert len(result_sir) == 3

    # SIS model
    sis = SIS()
    result_sis = sis.compute_spread_rate(S=800, I=200)
    assert len(result_sis) == 2

    # SEIR model
    seir = SEIR()
    result_seir = seir.compute_spread_rate(S=700, E=150, I=100)
    assert len(result_seir) == 4


if __name__ == "__main__":
    # Run the tests individually to ensure they work
    test_instance_sir = TestSIR()
    test_instance_sis = TestSIS()
    test_instance_seir = TestSEIR()

    print("Running dynamics contagion comprehensive tests...")

    test_instance_sir.test_sir_initialization()
    print("✓ SIR initialization test passed")

    test_instance_sir.test_sir_compute_spread_rate_default()
    print("✓ SIR compute spread rate default test passed")

    test_instance_sir.test_sir_compute_spread_rate_custom_params()
    print("✓ SIR compute spread rate custom params test passed")

    test_instance_sis.test_sis_initialization()
    print("✓ SIS initialization test passed")

    test_instance_sis.test_sis_compute_spread_rate_default()
    print("✓ SIS compute spread rate default test passed")

    test_instance_sis.test_sis_compute_spread_rate_custom_params()
    print("✓ SIS compute spread rate custom params test passed")

    test_instance_seir.test_seir_initialization()
    print("✓ SEIR initialization test passed")

    test_instance_seir.test_seir_compute_spread_rate_default()
    print("✓ SEIR compute spread rate default test passed")

    test_instance_seir.test_seir_compute_spread_rate_custom_params()
    print("✓ SEIR compute spread rate custom params test passed")

    test_contagion_comprehensive()
    print("✓ Integration test passed")

    print("\nAll comprehensive dynamics contagion tests passed! 🎉")
