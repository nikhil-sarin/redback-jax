import jax.numpy as jnp

from redback_jax.conversions import calc_kcorrected_properties


def test_calc_kcorrected_properties():
    """Test the calc_kcorrected_properties function."""
    frequency = jnp.array([1e14, 2e14, 3e14])  # Hz
    redshift = 0.5
    time = jnp.array([10.0, 20.0, 30.0])  # days

    k_freq, k_time = calc_kcorrected_properties(frequency, redshift, time)

    expected_k_freq = frequency * (1 + redshift)
    expected_k_time = time / (1 + redshift)

    assert jnp.allclose(k_freq, expected_k_freq)
    assert jnp.allclose(k_time, expected_k_time)