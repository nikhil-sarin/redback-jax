"""
Tests for redback_jax.models.supernova_models module.
"""
import jax.numpy as jnp
import numpy as np

from redback_jax.models.sed_features import SEDFeatures
from redback_jax.models.supernova_models import (
    _compute_mass_and_nickel,
    _nickelcobalt_engine,
    arnett_bolometric,
    arnett_with_features_cosmology,
    arnett_with_features_lum_dist,
    PLANCK18_H0,
    PLANCK18_OM0,
)


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


def test_compute_mass_and_nickel():
    """Test the _compute_mass_and_nickel function."""
    vel, v_m, m_array, ni_array = _compute_mass_and_nickel(
        vmin=500.0,
        esn=1.0,
        mej=2.0,
        f_nickel=0.1,
        f_mixing=0.25,
        vmax=10000,
        delta=0.0,
        n=12.0,
    )

    # Test the shape is correct and the first few values match the non-JAX implementation.
    assert vel.shape == (200,)
    assert jnp.allclose(vel[0:3], jnp.array([500., 507.58390609, 515.28284344]), rtol=1e-8)

    assert v_m.shape == (200,)
    assert jnp.allclose(v_m[0:3], jnp.array([5.00000000e+07, 5.07583906e+07, 5.15282843e+07]), rtol=1e-8)

    assert m_array.shape == (200,)
    assert jnp.allclose(m_array[0:3], jnp.array([0.00019662, 0.00020263, 0.00020883]), rtol=1e-8)

    ni_array = np.array(ni_array)
    assert ni_array.shape == (200,)
    assert np.allclose(ni_array[0:3], [0.00174366, 0.00179696, 0.00185188], rtol=1e-8)
    assert np.all(ni_array[50:] == 0.0)  # Only the inner 25% should have nickel.


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


def test_arnett_models():
    """Test the arnett models specified by cosmology and luminosity distance."""
    f_nickel = 0.1  # fraction of nickel mass
    mej = 1.0  # solar masses

    val_cosmo = arnett_with_features_cosmology(
        f_nickel,
        mej,
        redshift=0.1,
        kappa=0.07,
        kappa_gamma=0.1,
        vej=5000,
        cosmo_H0=PLANCK18_H0,
        cosmo_Om0=PLANCK18_OM0,
        temperature_floor=1000,
    )
    assert(val_cosmo.time.shape == (3000,))
    assert(val_cosmo.lambdas.shape == (100,))
    assert(val_cosmo.spectra.shape == (3000, 100))

    # Check for a single peak at each time.
    for i in range(len(val_cosmo.time)):
        max_t = jnp.argmax(val_cosmo.spectra[i, :])
        assert jnp.all(jnp.diff(val_cosmo.spectra[i, :max_t]) >= 0)
        assert jnp.all(jnp.diff(val_cosmo.spectra[i, max_t:]) <= 0)

    val_dl = arnett_with_features_lum_dist(
        f_nickel,
        mej,
        redshift=0.1,
        lum_dist=1.4684007701387617e+27,  # cm
        kappa=0.07,
        kappa_gamma=0.1,
        vej=5000,
        temperature_floor=1000,
    )
    assert(val_dl.time.shape == (3000,))
    assert(val_dl.lambdas.shape == (100,))
    assert(val_dl.spectra.shape == (3000, 100))

    assert jnp.allclose(val_cosmo.time, val_dl.time, rtol=1e-5)
    assert jnp.allclose(val_cosmo.lambdas, val_dl.lambdas, rtol=1e-5)
    assert jnp.allclose(val_cosmo.spectra, val_dl.spectra, rtol=1e-5)


def test_arnett_with_features():
    """Test the arnett model with features."""
    f_nickel = 0.1  # fraction of nickel mass
    mej = 1.0  # solar masses

    # Compute a reference without features.
    val_no_features = arnett_with_features_lum_dist(
        f_nickel,
        mej,
        redshift=0.0,
        lum_dist=1.4684007701387617e+27,  # cm
        kappa=0.07,
        kappa_gamma=0.1,
        vej=5000,
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
        f_nickel,
        mej,
        redshift=0.0,
        lum_dist=1.4684007701387617e+27,  # cm
        kappa=0.07,
        kappa_gamma=0.1,
        vej=5000,
        temperature_floor=1000,
        features=features,
    )

    # Check that the features have modified the output of the spectra only.
    assert jnp.allclose(val_no_features.time, val_features.time, rtol=1e-5)
    assert jnp.allclose(val_no_features.lambdas, val_features.lambdas, rtol=1e-5)
    assert not jnp.allclose(val_no_features.spectra, val_features.spectra, rtol=1e-5, atol=1e-20)
