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
    cutoff_blackbody_sed,
    blackbody_flux_density,
)


# ---------------------------------------------------------------------------
# Helpers: build redback reference values
# ---------------------------------------------------------------------------

def _redback_cutoff_bb(time, temperature, luminosity, r_photosphere,
                        frequency, dl, cutoff_wl, alpha=1.0):
    """Return flux density in mJy from redback's CutoffBlackbody."""
    import sys
    sys.path.insert(0, "/home/nikhilsarin/projects/redback")
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
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, cutoff_wl, alpha,
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
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, cutoff_wl, alpha,
    ))
    ref = np.asarray(ref).flatten()
    rel_err = np.abs((jax_out - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel_err.max() < 1e-4, f"alpha={alpha}: max rel error {rel_err.max():.2e}"


# ---------------------------------------------------------------------------
# Test 3: 2-D (spectral) mode produces consistent results
# ---------------------------------------------------------------------------

def test_cutoff_bb_2d_mode(slsn_params):
    """2-D mode evaluated at the data-point frequencies should match 1-D."""
    time, T, L, R, freq, dl = slsn_params
    cutoff_wl = 3000.0
    alpha     = 1.0

    # Unique frequencies
    unique_freq = np.unique(freq)

    out_2d = np.array(cutoff_blackbody_sed(
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(unique_freq), dl, cutoff_wl, alpha,
    ))  # (n_f, n_t)

    out_1d = np.array(cutoff_blackbody_flux_density(
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, cutoff_wl, alpha,
    ))  # (n,)

    # For each data point, cross-check via 2D output
    freq_idx = {f: i for i, f in enumerate(unique_freq)}
    for i, (fi, ti) in enumerate(zip(freq, range(N))):
        fi_idx = freq_idx[fi]
        rel = abs(out_2d[fi_idx, ti] - out_1d[i]) / max(abs(out_1d[i]), 1e-30)
        assert rel < 1e-10, f"2D/1D mismatch at i={i}: {rel:.2e}"


# ---------------------------------------------------------------------------
# Test 4: flux is positive and finite
# ---------------------------------------------------------------------------

def test_cutoff_bb_positive_finite(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    out = np.array(cutoff_blackbody_flux_density(
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, 3000.0,
    ))
    assert np.all(np.isfinite(out)), "Non-finite values in output"
    assert np.all(out > 0), "Non-positive flux density"


# ---------------------------------------------------------------------------
# Test 5: JIT is stable — two calls with same inputs give same result
# ---------------------------------------------------------------------------

def test_cutoff_bb_jit_stable(slsn_params):
    time, T, L, R, freq, dl = slsn_params
    a = jnp.array(cutoff_blackbody_flux_density(
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, 3000.0,
    ))
    b = jnp.array(cutoff_blackbody_flux_density(
        jnp.array(T), jnp.array(R), jnp.array(L),
        jnp.array(freq), dl, 3000.0,
    ))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Test 6: simple blackbody matches redback's blackbody_to_flux_density
# ---------------------------------------------------------------------------

def test_blackbody_matches_redback(slsn_params):
    import sys
    sys.path.insert(0, "/home/nikhilsarin/projects/redback")
    time, T, L, R, freq, dl = slsn_params
    from redback.sed import blackbody_to_flux_density
    import astropy.units as uu
    ref = blackbody_to_flux_density(T, R, dl, freq).to(uu.mJy).value
    out = np.array(blackbody_flux_density(
        jnp.array(T), jnp.array(R), jnp.array(freq), dl,
    ))
    ref = np.asarray(ref).flatten()
    rel_err = np.abs((out - ref) / np.maximum(np.abs(ref), 1e-30))
    assert rel_err.max() < 1e-5, f"Max rel error {rel_err.max():.2e}"
