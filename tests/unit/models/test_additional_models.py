"""Behavioural tests for magnetar / CSM / nickel models and the diffrax variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants as sc

jax.config.update("jax_enable_x64", True)

from redback_jax.models import supernova_models as sm

TIMES = jnp.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# blackbody_to_flux_density
# ---------------------------------------------------------------------------

def test_blackbody_to_flux_density_matches_planck():
    T = np.array([5e3, 1e4, 2e4])
    R = np.array([1e14, 3e14, 5e13])
    nu = np.array([4e14, 6e14, 8e14])
    dl = 3.0857e26

    h, c, k = sc.h * 1e7, sc.c * 1e2, sc.k * 1e7   # cgs
    planck = 2.0 * np.pi * h * nu**3 / c**2 / np.expm1(h * nu / (k * T))
    expected = planck * R**2 / dl**2

    out = np.asarray(sm.blackbody_to_flux_density(jnp.asarray(T), jnp.asarray(R), dl, jnp.asarray(nu)))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# magnetar_powered_bolometric / magnetar_nickel_bolometric
# ---------------------------------------------------------------------------

MAG = dict(mej=5.0, kappa=0.1, kappa_gamma=0.1, vej=8000.0,
           mass_ns=1.4, theta_pb=1.0, bp=1.0)


def test_magnetar_powered_bolometric_physical_trends():
    fast = np.asarray(sm.magnetar_powered_bolometric(TIMES, p0=2.0, **MAG))
    slow = np.asarray(sm.magnetar_powered_bolometric(TIMES, p0=10.0, **MAG))
    assert fast.shape == slow.shape == TIMES.shape
    assert np.all(np.isfinite(fast)) and np.all(np.isfinite(slow))
    assert 42.0 < fast.max() < 46.0                      # log10 erg/s, SLSN-like
    assert fast.max() > slow.max()                       # more rotational energy -> brighter


def test_magnetar_nickel_adds_luminosity_at_late_times():
    base = np.asarray(sm.magnetar_powered_bolometric(TIMES, p0=10.0, **MAG))
    with_ni = np.asarray(sm.magnetar_nickel_bolometric(TIMES, f_nickel=0.1, p0=10.0, **MAG))
    no_ni = np.asarray(sm.magnetar_nickel_bolometric(TIMES, f_nickel=0.0, p0=10.0, **MAG))
    assert np.all(np.isfinite(with_ni))
    np.testing.assert_allclose(no_ni, base, atol=0.02)   # f_nickel=0 reduces to magnetar-only
    assert np.all(with_ni[2:] > base[2:])


# ---------------------------------------------------------------------------
# csm_interaction_bolometric
# ---------------------------------------------------------------------------

CSM = dict(mej=5.0, csm_mass=2.0, vej=10000.0, rho=1e-12, kappa=0.1, r0=100.0)
CSM_TIMES = jnp.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=jnp.float64)


@pytest.mark.parametrize("eta", [0.0, 2.0])
def test_csm_interaction_matches_redback(eta):
    pytest.importorskip("redback")
    from redback.transient_models.supernova_models import csm_interaction_bolometric as rb

    ref = np.log10(np.asarray(rb(np.asarray(CSM_TIMES), eta=eta, **CSM)))
    out = np.asarray(sm.csm_interaction_bolometric(CSM_TIMES, eta=eta, **CSM))
    np.testing.assert_allclose(out, ref, atol=0.1)


def test_csm_interaction_finite_and_smooth_through_eta_one():
    """eta == 1 is singular in the photosphere formulae; it used to return ~1e29 erg/s."""
    t = jnp.array([5.0, 20.0, 50.0], dtype=jnp.float64)
    at_one = np.asarray(sm.csm_interaction_bolometric(t, eta=1.0, **CSM))
    below = np.asarray(sm.csm_interaction_bolometric(t, eta=0.97, **CSM))
    above = np.asarray(sm.csm_interaction_bolometric(t, eta=1.03, **CSM))
    assert np.all(np.isfinite(at_one))
    assert at_one.min() > 40.0
    np.testing.assert_allclose(at_one, below, atol=0.05)
    np.testing.assert_allclose(at_one, above, atol=0.05)


# ---------------------------------------------------------------------------
# General magnetar: diffrax variants
# ---------------------------------------------------------------------------

GEN = dict(mej=1.0, log10_E_sn=51.0, kappa=0.1, log10_l0=45.0, tau_sd=1e6,
           nn=3.0, kappa_gamma=1.0, f_nickel=0.1)
FLUX_EXTRA = dict(temperature_floor=3000.0, luminosity_distance=3e26, redshift=0.05)


def test_general_magnetar_bolometric_diffrax_matches_scan():
    scan = np.asarray(sm.general_magnetar_driven_supernova_bolometric(TIMES, **GEN))
    dfx = np.asarray(sm.general_magnetar_driven_supernova_bolometric_diffrax(TIMES, **GEN))
    assert np.all(np.isfinite(dfx))
    np.testing.assert_allclose(dfx, scan, atol=0.02)


def test_general_magnetar_bolometric_and_vej_diffrax():
    lbol, vej = sm.general_magnetar_driven_supernova_bolometric_and_vej_diffrax(TIMES, **GEN)
    lbol, vej = np.asarray(lbol), np.asarray(vej)
    assert lbol.shape == vej.shape == TIMES.shape
    assert np.all(np.isfinite(lbol)) and np.all(vej > 0)
    # ejecta are accelerated by the magnetar: velocity is non-decreasing
    assert np.all(np.diff(vej) >= -1.0)


def test_general_magnetar_flux_density_diffrax_matches_scan():
    nu = jnp.full(TIMES.shape, 5e14)
    scan = np.asarray(sm.general_magnetar_driven_supernova(TIMES, nu, **GEN, **FLUX_EXTRA))
    dfx = np.asarray(sm.general_magnetar_driven_supernova_diffrax(TIMES, nu, **GEN, **FLUX_EXTRA))
    assert np.all(np.isfinite(dfx)) and np.all(dfx > 0)
    np.testing.assert_allclose(dfx, scan, rtol=1e-3)


# ---------------------------------------------------------------------------
# csm_nickel_flux_density
# ---------------------------------------------------------------------------

def _csm_nickel_flux(f_nickel=0.1, dl=7e26, eta=2.0, redshift=0.05, t=None):
    t = jnp.array([5.0, 20.0, 50.0, 100.0, 200.0]) if t is None else t
    nu = jnp.full(t.shape, 5e14)
    return np.asarray(sm.csm_nickel_flux_density(
        t, nu, redshift, dl, 5.0, f_nickel, 2.0, 10000.0, eta, 1e-12, 0.1, 1.0,
        100.0, 3000.0, 1.0, 3000.0,
    ))


def test_csm_nickel_flux_density_positive_and_scales_with_distance():
    near, far = _csm_nickel_flux(dl=7e26), _csm_nickel_flux(dl=14e26)
    assert np.all(np.isfinite(near)) and np.all(near > 0)
    np.testing.assert_allclose(near / far, 4.0, rtol=1e-6)      # F ∝ d_L^-2


def test_csm_nickel_flux_density_nickel_powers_late_times():
    none = _csm_nickel_flux(f_nickel=0.0)
    lots = _csm_nickel_flux(f_nickel=0.3)
    assert np.all(np.isfinite(none))
    assert lots[-1] > 100.0 * none[-1]                # nickel tail dominates at 200 d
    np.testing.assert_allclose(lots[0], none[0], rtol=0.05)   # early CSM phase unaffected


def test_csm_nickel_flux_density_finite_at_eta_one():
    assert np.all(np.isfinite(_csm_nickel_flux(eta=1.0)))
    np.testing.assert_allclose(_csm_nickel_flux(eta=1.0)[1:], _csm_nickel_flux(eta=1.03)[1:], rtol=0.15)
