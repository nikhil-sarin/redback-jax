"""Independent cgs evaluation of the Piro cooling-envelope equations."""
import numpy as np
import jax.numpy as jnp
import pytest

from redback_jax.models.shock_powered_models import shock_cooling_bolometric


@pytest.mark.parametrize("kappa", [0.1, 0.2, 0.4])
def test_piro_cgs_reference(kappa):
    mass, radius, energy = 0.1 * 1.989e33, 10**13.5, 1e50
    n, delta = 10., 1.1
    density_norm = (n - 3) * (3 - delta) / (4 * np.pi * (n - delta))
    velocity = np.sqrt(2 * (n - 5) * (5 - delta) * energy /
                       ((n - 3) * (3 - delta) * mass))
    td = np.sqrt(3 * kappa * density_norm * mass / ((n - 1) * velocity * 2.998e10))
    seconds = np.array([0.1, 0.9, 1.1, 3., 20.]) * td
    prefactor = np.pi * (n - 1) * 2.998e10 * radius * velocity**2 / (3 * (n - 5) * kappa)
    expected = np.log10(prefactor) + np.where(
        seconds < td, 4 / (n - 2) * np.log10(td / seconds),
        -0.5 * ((seconds / td)**2 - 1) / np.log(10))
    actual = shock_cooling_bolometric(jnp.asarray(seconds / 86400), -1., 13.5, 50., n, delta, kappa)
    np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0)


def test_combined_cooling_and_arnett_uses_corrected_cooling():
    from redback_jax.models import (
        arnett_bolometric, shock_cooling_and_arnett_bolometric,
    )
    time = jnp.array([1., 3., 11., 30.])
    shock = shock_cooling_bolometric(time, -1., 13.5, 50., 10., 1.1, 0.2)
    arnett = arnett_bolometric(time, 0.1, 1., vej=10000., kappa=0.2, kappa_gamma=0.1)
    expected = jnp.logaddexp(shock*jnp.log(10.), arnett*jnp.log(10.))/jnp.log(10.)
    actual = shock_cooling_and_arnett_bolometric(
        time, -1., 13.5, 50., 10., 1.1, 0.1, 1., 10000., 0.2, 0.1)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
