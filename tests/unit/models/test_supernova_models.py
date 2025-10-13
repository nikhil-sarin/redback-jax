"""
Tests for redback_jax.models.supernova_models module.
"""
import jax.numpy as jnp
import pytest

from redback_jax.models.supernova_models import _nickelcobalt_engine


def test_nickelcobalt_engine():
    """Test the _nickelcobalt_engine function."""
    time = jnp.array([0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0])  # days
    f_nickel = 0.1  # fraction of nickel mass
    mej = 1.0  # solar masses

    lbol = _nickelcobalt_engine(time, f_nickel, mej)
    expected = jnp.array([7.82582e+42, 7.19419e+42, 3.39575e+42, 1.876061e+42, 9.47245e+41, 5.90502e+41, 2.40417e+41])
    assert jnp.allclose(lbol, expected, rtol=1e-4)

    f_nickel = 0.5  # fraction of nickel mass
    mej = 1.5  # solar masses

    lbol = _nickelcobalt_engine(time, f_nickel, mej)
    expected = jnp.array([5.86936292e+43, 5.39564058e+43, 2.54681265e+43, 1.40704539e+43, 7.10433476e+42, 4.42876518e+42, 1.80312521e+42])
    assert jnp.allclose(lbol, expected, rtol=1e-4)
