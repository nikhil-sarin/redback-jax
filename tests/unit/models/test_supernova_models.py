"""
Tests for redback_jax.models.supernova_models module.
"""
import jax.numpy as jnp

from redback_jax.models.supernova_models import _nickelcobalt_engine, arnett_bolometric

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


def test_arnett_bolometric():
    """Test the arnett_bolometric function."""
    times = jnp.array([0.1, 1.0, 5.0, 10.0, 20.0, 50.0])  # days
    f_nickel = 0.1  # fraction of nickel mass
    mej = 1.0  # solar masses

    lbol = arnett_bolometric(times, f_nickel, mej, kappa=0.07, kappa_gamma=0.1, vej=5000)
    expected = jnp.array([4.32312e+38, 4.07620e+40, 7.51038e+41, 1.85010e+42, 2.17230e+42, 9.74861e+41])
    assert jnp.allclose(lbol, expected, rtol=1e-4)

    f_nickel = 0.5  # fraction of nickel mass
    mej = 1.5  # solar masses
    lbol = arnett_bolometric(times, f_nickel, mej, kappa=0.03, kappa_gamma=0.15, vej=7000)
    expected = jnp.array([7.06086e+39, 6.63590e+41, 1.12921e+43, 2.21348e+43, 1.59988e+43, 7.22212e+42])
    assert jnp.allclose(lbol, expected, rtol=1e-4)

    f_nickel = 0.3  # fraction of nickel mass
    mej = 2.5  # solar masses
    lbol = arnett_bolometric(times, f_nickel, mej, kappa=0.04, kappa_gamma=0.11, vej=6500)
    expected = jnp.array([2.95054e+39, 2.78271e+41, 5.15889e+42, 1.29505e+43, 1.60733e+43, 7.32969e+42])
    assert jnp.allclose(lbol, expected, rtol=1e-4)
