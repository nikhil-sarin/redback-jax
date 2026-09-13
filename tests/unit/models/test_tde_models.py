"""Tests for JAX tidal-disruption-event models."""

import jax
import jax.numpy as jnp
import numpy as np

from redback_jax.models import MODEL_REGISTRY, tde_fallback_bolometric


def test_tde_fallback_matches_redback_reference_curve():
    """The rectangular JAX tables reproduce Redback to better than 0.015 dex."""
    time = jnp.geomspace(0.01, 300.0, 25)
    result = np.asarray(tde_fallback_bolometric(
        time, mbh6=1.0, mstar=1.0, tvisc=10.0, bb=1.0,
        eta=0.1, leddlimit=1.0))

    # Generated with redback 1.20.0 tde_fallback_bolometric using the same
    # parameters, then converted from linear luminosity to log10(erg/s).
    reference = np.array([
        39.2228, 39.9956, 40.8953, 42.6429,
        43.8803, 44.1439, 44.0119,
    ])
    np.testing.assert_allclose(result[[0, 4, 8, 12, 16, 20, 24]],
                               reference, atol=0.015, rtol=0.0)


def test_tde_fallback_is_finite_and_differentiable():
    time = jnp.geomspace(0.01, 300.0, 16)

    def total_luminosity(mbh6):
        return jnp.sum(tde_fallback_bolometric(
            time, mbh6=mbh6, mstar=0.5, tvisc=3.0, bb=0.5,
            eta=0.1, leddlimit=2.0))

    result = tde_fallback_bolometric(
        time, mbh6=0.1, mstar=0.5, tvisc=3.0, bb=0.5,
        eta=0.1, leddlimit=2.0)
    assert result.shape == time.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert bool(jnp.isfinite(jax.grad(total_luminosity)(0.1)))


def test_tde_fallback_is_registered():
    assert MODEL_REGISTRY["tde_fallback_bolometric"] is tde_fallback_bolometric
