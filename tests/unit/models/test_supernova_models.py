"""
Tests for redback_jax.models.supernova_models module.
"""
import jax.numpy as jnp
import numpy as np

from redback_jax.models.sed_features import SEDFeatures
from redback_jax.models.supernova_models import (
    _nickelcobalt_log10_engine,
    arnett_bolometric,
    arnett_with_features_cosmology,
    arnett_with_features_lum_dist,
    PLANCK18_H0,
    PLANCK18_OM0,
)


def test_nickelcobalt_log10_engine():
    """Test the _nickelcobalt_log10_engine function returns log10(L)."""
    time = jnp.array([0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    f_nickel = 0.1
    mej = 1.0

    log10_lbol = _nickelcobalt_log10_engine(time, f_nickel, mej)

    # Returns log10(L): physical range ~42-44 erg/s for these parameters
    assert log10_lbol.shape == time.shape
    assert not jnp.any(jnp.isnan(log10_lbol))
    assert not jnp.any(jnp.isinf(log10_lbol))
    assert jnp.all(log10_lbol > 40.0) and jnp.all(log10_lbol < 50.0)

    # Monotonically decreasing after early times (Co decay dominates late)
    assert log10_lbol[0] > log10_lbol[-1]

    # Verify log10 values match the analytic formula computed in float64
    import numpy as np
    ni56_lum = 6.45e43; co56_lum = 1.45e43
    ni56_life = 8.8;    co56_life = 111.3
    t_np = np.array(time, dtype=np.float64)
    expected_log10 = np.log10(
        f_nickel * mej * (ni56_lum * np.exp(-t_np / ni56_life)
                          + co56_lum * np.exp(-t_np / co56_life)))
    assert jnp.allclose(log10_lbol, jnp.array(expected_log10, dtype=jnp.float32),
                        rtol=1e-4, atol=1e-3)


def test_arnett_bolometric():
    """Test arnett_bolometric returns log10(L) in the correct range."""
    times = jnp.array([0.1, 1.0, 5.0, 10.0, 20.0, 50.0])

    log10_lbol = arnett_bolometric(times, f_nickel=0.1, mej=1.0,
                                    kappa=0.07, kappa_gamma=0.1, vej=5000)

    assert log10_lbol.shape == times.shape
    assert not jnp.any(jnp.isnan(log10_lbol))
    assert not jnp.any(jnp.isinf(log10_lbol))
    # Physical range for these parameters
    assert jnp.all(log10_lbol > 38.0) and jnp.all(log10_lbol < 45.0)

    # Peak should occur near middle of the light curve (diffusion broadens it)
    peak_idx = int(jnp.argmax(log10_lbol))
    assert 1 <= peak_idx <= len(times) - 2

    # Verify log10 values match expected (compare in log10 space to avoid float32 overflow)
    import numpy as np
    expected_log10 = np.log10(np.array([4.32312e+38, 4.07620e+40, 7.51038e+41,
                                         1.85010e+42, 2.17230e+42, 9.74861e+41]))
    assert jnp.allclose(log10_lbol, jnp.array(expected_log10, dtype=jnp.float32),
                        rtol=1e-3, atol=0.01)


def test_arnett_bolometric_parameter_scaling():
    """Higher nickel fraction and mass give higher peak luminosity."""
    times = jnp.array([5.0, 10.0, 20.0, 30.0])
    kwargs = dict(kappa=0.1, kappa_gamma=10.0, vej=10000.0)

    log10_lo = arnett_bolometric(times, f_nickel=0.1, mej=0.5, **kwargs)
    log10_hi = arnett_bolometric(times, f_nickel=0.5, mej=2.0, **kwargs)

    assert jnp.max(log10_hi) > jnp.max(log10_lo)


def test_arnett_models():
    """Test the arnett models specified by cosmology and luminosity distance."""
    f_nickel = 0.1
    mej = 1.0

    val_cosmo = arnett_with_features_cosmology(
        f_nickel, mej, redshift=0.1,
        kappa=0.07, kappa_gamma=0.1, vej=5000,
        cosmo_H0=PLANCK18_H0, cosmo_Om0=PLANCK18_OM0,
        temperature_floor=1000,
    )
    assert val_cosmo.time.shape == (3000,)
    assert val_cosmo.lambdas.shape == (100,)
    assert val_cosmo.spectra.shape == (3000, 100)

    # Each epoch should have a single spectral peak (blackbody).
    # Allow a tiny tolerance for floating-point noise near zero at the spectral edges.
    for i in range(len(val_cosmo.time)):
        max_t = jnp.argmax(val_cosmo.spectra[i, :])
        peak = float(jnp.max(val_cosmo.spectra[i, :]))
        atol = peak * 1e-6
        assert jnp.all(jnp.diff(val_cosmo.spectra[i, :max_t]) >= -atol)
        assert jnp.all(jnp.diff(val_cosmo.spectra[i, max_t:]) <= atol)

    val_dl = arnett_with_features_lum_dist(
        f_nickel, mej, redshift=0.1,
        lum_dist=1.4684007701387617e+27,
        kappa=0.07, kappa_gamma=0.1, vej=5000,
        temperature_floor=1000,
    )
    assert val_dl.time.shape == (3000,)
    assert val_dl.lambdas.shape == (100,)
    assert val_dl.spectra.shape == (3000, 100)

    assert jnp.allclose(val_cosmo.time, val_dl.time, rtol=1e-5)
    assert jnp.allclose(val_cosmo.lambdas, val_dl.lambdas, rtol=1e-5)
    assert jnp.allclose(val_cosmo.spectra, val_dl.spectra, rtol=1e-5)


def test_arnett_with_features():
    """Test the arnett model with SED features."""
    f_nickel = 0.1
    mej = 1.0

    val_no_features = arnett_with_features_lum_dist(
        f_nickel, mej, redshift=0.0,
        lum_dist=1.4684007701387617e+27,
        kappa=0.07, kappa_gamma=0.1, vej=5000,
        temperature_floor=1000,
    )

    features = SEDFeatures(
        rest_wavelengths=jnp.array([5000.0, 6000.0]),
        sigmas=jnp.array([1000.0, 2000.0]),
        amplitudes=jnp.array([100.0, 200.0]),
        t_starts=jnp.array([0.0, 100.0]),
        t_ends=jnp.array([1000.0, 2000.0]),
    )
    val_features = arnett_with_features_lum_dist(
        f_nickel, mej, redshift=0.0,
        lum_dist=1.4684007701387617e+27,
        kappa=0.07, kappa_gamma=0.1, vej=5000,
        temperature_floor=1000, features=features,
    )

    # Time and wavelength grids unchanged, only spectra differ
    assert jnp.allclose(val_no_features.time, val_features.time, rtol=1e-5)
    assert jnp.allclose(val_no_features.lambdas, val_features.lambdas, rtol=1e-5)
    assert not jnp.allclose(val_no_features.spectra, val_features.spectra,
                             rtol=1e-5, atol=1e-20)
