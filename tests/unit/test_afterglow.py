"""
Tests for redback_jax.models.afterglow_models — numerical match
against redback's tophat_redback.
"""
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/home/nikhilsarin/projects/redback")

from redback_jax.models.afterglow_models import tophat_redback


# ---------------------------------------------------------------------------
# Reference: run redback's tophat_redback
# ---------------------------------------------------------------------------

def _redback_tophat(time, redshift, thv, loge0, thc, logn0, p,
                    logepse, logepsb, g0, xiN, frequency, dl=None,
                    res=100, steps=250):
    """Return flux in mJy from redback's tophat_redback.

    redback ignores any dl kwarg and computes it internally from Planck18.
    """
    from redback.transient_models.afterglow_models.base_models import tophat_redback as rb_tophat

    result = rb_tophat(
        time=time, redshift=redshift, thv=thv, loge0=loge0, thc=thc,
        logn0=logn0, p=p, logepse=logepse, logepsb=logepsb,
        g0=g0, xiN=xiN,
        frequency=frequency,
        output_format='flux_density',
        res=res, steps=steps,
    )
    return np.asarray(result)   # already in mJy (fmjy)


# ---------------------------------------------------------------------------
# Canonical parameter set
# ---------------------------------------------------------------------------

PARAMS = dict(
    redshift=0.01,
    thv=0.05,
    loge0=53.0,
    thc=0.1,
    logn0=-2.0,
    p=2.2,
    logepse=-0.7,
    logepsb=-2.0,
    g0=100.0,
    xiN=1.0,
)

def _planck18_dl(z):
    """Planck18 luminosity distance in cm."""
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(z).cgs.value

DL_CM = _planck18_dl(PARAMS['redshift'])  # exact Planck18 value for z=0.01


# ---------------------------------------------------------------------------
# Test 1: On-axis radio — single frequency
# ---------------------------------------------------------------------------

def test_tophat_onaxis_radio():
    time = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    freq = np.full(len(time), 5e9)   # 5 GHz

    ref = _redback_tophat(time, **PARAMS, frequency=freq, dl=DL_CM,
                           res=100, steps=250)
    out = np.array(tophat_redback(
        jnp.array(time), **PARAMS, frequency=jnp.array(freq), dl=DL_CM,
        res=100, steps=250,
    ))

    assert out.shape == ref.shape
    # ~5-6% tolerance: residual after matching dl is fastmath accumulation error
    good = np.abs(ref) > 1e-6 * np.abs(ref).max()
    rel = np.abs((out[good] - ref[good]) / ref[good])
    assert rel.max() < 0.08, f"Max relative error {rel.max():.3f} > 8%"


# ---------------------------------------------------------------------------
# Test 2: Off-axis optical — verify same ordering / sign
# ---------------------------------------------------------------------------

def test_tophat_offaxis_optical():
    time = np.geomspace(0.5, 200.0, 8)
    freq = np.full(len(time), 3e14)   # ~r-band

    params_offaxis = {**PARAMS, 'thv': 0.4}   # observer outside jet

    ref = _redback_tophat(time, **params_offaxis, frequency=freq, dl=DL_CM,
                           res=100, steps=250)
    out = np.array(tophat_redback(
        jnp.array(time), **params_offaxis,
        frequency=jnp.array(freq), dl=DL_CM,
        res=100, steps=250,
    ))

    assert out.shape == ref.shape
    good = np.abs(ref) > 1e-6 * np.abs(ref).max()
    rel = np.abs((out[good] - ref[good]) / ref[good])
    assert rel.max() < 0.08, f"Max relative error {rel.max():.3f} > 8%"


# ---------------------------------------------------------------------------
# Test 3: Multi-band — two frequencies interleaved
# ---------------------------------------------------------------------------

def test_tophat_multiband():
    n = 10
    time = np.geomspace(1.0, 100.0, n)
    freq = np.where(np.arange(n) % 2 == 0, 5e9, 3e14)

    ref = _redback_tophat(time, **PARAMS, frequency=freq, dl=DL_CM,
                           res=100, steps=250)
    out = np.array(tophat_redback(
        jnp.array(time), **PARAMS, frequency=jnp.array(freq), dl=DL_CM,
        res=100, steps=250,
    ))

    assert out.shape == ref.shape
    good = np.abs(ref) > 1e-6 * np.abs(ref).max()
    rel = np.abs((out[good] - ref[good]) / ref[good])
    assert rel.max() < 0.08, f"Max relative error {rel.max():.3f} > 8%"


# ---------------------------------------------------------------------------
# Test 4: Output is positive and finite
# ---------------------------------------------------------------------------

def test_tophat_positive_finite():
    time = np.geomspace(0.1, 100.0, 10)
    freq = np.full(len(time), 5e9)
    out = np.array(tophat_redback(
        jnp.array(time), **PARAMS, frequency=jnp.array(freq), dl=DL_CM,
        res=30, steps=100,
    ))
    assert np.all(np.isfinite(out)), "Non-finite values"
    assert np.all(out >= 0), "Negative flux"


# ---------------------------------------------------------------------------
# Test 5: JIT stability
# ---------------------------------------------------------------------------

def test_tophat_jit_stable():
    time = np.array([1.0, 10.0, 100.0])
    freq = np.full(3, 5e9)
    a = np.array(tophat_redback(
        jnp.array(time), **PARAMS, frequency=jnp.array(freq), dl=DL_CM,
        res=30, steps=100,
    ))
    b = np.array(tophat_redback(
        jnp.array(time), **PARAMS, frequency=jnp.array(freq), dl=DL_CM,
        res=30, steps=100,
    ))
    np.testing.assert_array_equal(a, b)
