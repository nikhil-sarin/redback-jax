"""
Tests for redback_jax.phenomenological_models module.
"""
import jax.numpy as jnp

from redback_jax import phenomenological_models


class TestPhenomenologicalModelsModule:
    """Test class for phenomenological_models module functionality."""

    def test_smooth_exponential_powerlaw(self):
        """Test the smooth_exponential_powerlaw function."""
        time = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = phenomenological_models.smooth_exponential_powerlaw(
            time, 
            1.0,  # a_1
            3.0,  # tpeak
            2.0,  # alpha_1
            -1.0,  # alpha_2
            1.0,  # smoothing_factor
        )
        assert jnp.allclose(result, jnp.array([0.4, 0.76923077, 1.0, 1.12, 1.17647059]), atol=1e-5)

        # Test a few different settings against ground truth from the non-JAX implementation.
        result = phenomenological_models.smooth_exponential_powerlaw(
            time, 
            2.0,  # a_1
            2.5,  # tpeak
            1.5,  # alpha_1
            -1.2,  # alpha_2
            5.0,  # smoothing_factor
        )
        assert jnp.allclose(result, jnp.array([0.50654, 1.54579, 1.74911, 1.16408, 0.87522]), atol=1e-5)
