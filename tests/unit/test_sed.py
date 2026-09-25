"""
Tests for redback_jax.sed — numerical match against redback's CutoffBlackbody.
"""
import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from redback_jax.sed import (
    cutoff_blackbody_flux_density,
    cutoff_blackbody_flux_density_with_norm,
    cutoff_blackbody_norm,
)
from redback_jax.models.supernova_models import blackbody_to_flux_density


# ---------------------------------------------------------------------------
# Helpers: build redback reference values
# ---------------------------------------------------------------------------

def _redback_cutoff_bb(time, temperature, luminosity, r_photosphere,
                        frequency, dl, cutoff_wl, alpha=1.0):
    """Return flux density in mJy from redback's CutoffBlackbody."""
    pytest.importorskip("redback")
    from redback.sed import CutoffBlackbody
    sed = CutoffBlackbody(
        time=time,
        temperature=temperature,
        luminosity=luminosity,
        r_photosphere=r_photosphere,
        frequency=frequency,
        luminosity_distance=dl,
        cutoff_wavelength=cutoff_wl,
        absorption_index=alpha,
    )
    import astropy.units as uu
    return sed.flux_density.to(uu.mJy).value


# ---------------------------------------------------------------------------
# Fixtures: a realistic SLSN-like set of parameters
# ---------------------------------------------------------------------------

N = 30  # number of data points

@pytest.fixture
def slsn_params():
    rng = np.random.default_rng(42)
    time = np.linspace(1.0, 200.0, N)  # days

    T0  = 15000.0  # K
    T   = T0 * (1.0 + time / 60.0) ** -0.5
    R0  = 1e14     # cm
    R   = R0 * (1.0 + time / 20.0)
    L   = 4.0 * math.pi * R**2 * 5.6704e-5 * T**4   # Stefan–Boltzmann

    # Multi-band: cycle through g, r, i, z approximate effective frequencies
    band_freqs = np.array([6.3e14, 4.9e14, 3.9e14, 3.3e14])  # Hz
    freq = band_freqs[np.arange(N) % 4]

    dl  = 3.0856e26   # 100 Mpc in cm
    return time, T, L, R, freq, dl


# ---------------------------------------------------------------------------
# Test 1: 1-D flux density matches redback to <0.01 % (relative)
# ---------------------------------------------------------------------------

def test_cutoff_bb_1d_matches_redback(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    cutoff_wl = 3000.0
    alpha     = 1.0

    ref = _redback_cutoff_bb(time, T, L, R, freq, dl, cutoff_wl, alpha)
    jax_out = np.array(cutoff_blackbody_flux_density(
        jnp.array(freq), jnp.array(L), jnp.array(T), jnp.array(R), dl, cutoff_wl, alpha,
    ))

    # Strip astropy shape quirks from redback output
    ref = np.asarray(ref).flatten()
    assert ref.shape == jax_out.shape, f"Shape mismatch: {ref.shape} vs {jax_out.shape}"
    rel_err = np.abs((jax_out - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel_err.max() < 1e-4, f"Max relative error {rel_err.max():.2e} > 1e-4"


# ---------------------------------------------------------------------------
# Test 2: different absorption_index values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.0])
def test_cutoff_bb_absorption_index(slsn_params, alpha):
    time, T, L, R, freq, dl = slsn_params
    cutoff_wl = 3000.0

    ref = _redback_cutoff_bb(time, T, L, R, freq, dl, cutoff_wl, alpha)
    jax_out = np.array(cutoff_blackbody_flux_density(
        jnp.array(freq), jnp.array(L), jnp.array(T), jnp.array(R), dl, cutoff_wl, alpha,
    ))
    ref = np.asarray(ref).flatten()
    rel_err = np.abs((jax_out - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel_err.max() < 1e-4, f"alpha={alpha}: max rel error {rel_err.max():.2e}"


# ---------------------------------------------------------------------------
# Test 3: precomputed-norm variant matches the direct evaluation
# ---------------------------------------------------------------------------

def test_cutoff_bb_with_norm_matches_direct(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    cutoff_wl, alpha = 3000.0, 1.0
    T, L, R, freq = (jnp.array(x) for x in (T, L, R, freq))

    direct = np.array(cutoff_blackbody_flux_density(freq, L, T, R, dl, cutoff_wl, alpha))
    norm = cutoff_blackbody_norm(L, T, R, cutoff_wl, alpha)
    via_norm = np.array(cutoff_blackbody_flux_density_with_norm(
        freq, T, R, dl, norm, cutoff_wl, alpha,
    ))
    rel = np.abs(direct - via_norm) / np.maximum(np.abs(direct), 1e-30)
    assert rel.max() < 1e-10, f"with_norm/direct mismatch: {rel.max():.2e}"


# ---------------------------------------------------------------------------
# Test 4: flux is positive and finite
# ---------------------------------------------------------------------------

def test_cutoff_bb_positive_finite(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    out = np.array(cutoff_blackbody_flux_density(
        jnp.array(freq), jnp.array(L), jnp.array(T), jnp.array(R), dl, 3000.0,
    ))
    assert np.all(np.isfinite(out)), "Non-finite values in output"
    assert np.all(out > 0), "Non-positive flux density"


# ---------------------------------------------------------------------------
# Test 5: JIT is stable — two calls with same inputs give same result
# ---------------------------------------------------------------------------

def test_cutoff_bb_jit_stable(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    a = jnp.array(cutoff_blackbody_flux_density(
        jnp.array(freq), jnp.array(L), jnp.array(T), jnp.array(R), dl, 3000.0,
    ))
    b = jnp.array(cutoff_blackbody_flux_density(
        jnp.array(freq), jnp.array(L), jnp.array(T), jnp.array(R), dl, 3000.0,
    ))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Test 6: simple blackbody matches redback's blackbody_to_flux_density
# ---------------------------------------------------------------------------

def test_blackbody_matches_redback(slsn_params):
    pytest.importorskip("redback")
    time, T, L, R, freq, dl = slsn_params
    from redback.sed import blackbody_to_flux_density as rb_blackbody
    import astropy.units as uu
    ref = rb_blackbody(T, R, dl, freq).to(uu.mJy).value
    # supernova_models version returns erg/s/Hz/cm^2; convert to mJy
    out = 1e26 * np.array(blackbody_to_flux_density(
        jnp.array(T), jnp.array(R), dl, jnp.array(freq),
    ))
    ref = np.asarray(ref).flatten()
    rel_err = np.abs((out - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel_err.max() < 1e-5, f"Max rel error {rel_err.max():.2e}"
