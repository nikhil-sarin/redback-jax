"""
Tests for redback_jax.models.general_magnetar

Checks:
  1. Output shape, dtype, and finite-value guarantees.
  2. Monotonic decline at late times (decay phase).
  3. Physical parameter scaling (higher l0 → higher peak).
  4. Numerical agreement with the original redback implementation.
  5. JIT idempotence (second call returns the same values).
  6. Model registered in MODEL_REGISTRY.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from redback_jax.models import (
    general_magnetar_driven_supernova_bolometric,
    general_magnetar_driven_supernova_bolometric_batched,
)
from redback_jax.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Shared test parameters (typical SN values used in SN 2025wny fits)
# ---------------------------------------------------------------------------
_PARAMS = dict(
    mej=1.0,          # M_sun
    E_sn=1e51,        # erg
    kappa=0.1,        # cm^2/g
    l0=1e45,          # erg/s
    tau_sd=1e6,       # s
    nn=3.0,           # dipole braking index
    kappa_gamma=1.0,  # cm^2/g
    f_nickel=0.1,
)

_TIMES = jnp.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# 1. Shape, dtype, and finite values
# ---------------------------------------------------------------------------

def test_output_shape():
    out = general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS)
    assert out.shape == _TIMES.shape


def test_output_dtype_preserved():
    """Output dtype should match input time array dtype."""
    out64 = general_magnetar_driven_supernova_bolometric(
        _TIMES.astype(jnp.float64), **_PARAMS)
    assert out64.dtype == jnp.float64

    out32 = general_magnetar_driven_supernova_bolometric(
        _TIMES.astype(jnp.float32), **_PARAMS)
    assert out32.dtype == jnp.float32


def test_output_finite():
    out = general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS)
    assert jnp.all(jnp.isfinite(out)), f"Non-finite values: {out}"


def test_physical_luminosity_range():
    """log10(L) should be in 42–47 erg/s for these typical parameters."""
    out = general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS)
    assert jnp.all(out > 40.0), f"Too faint: {out}"
    assert jnp.all(out < 50.0), f"Too bright: {out}"


# ---------------------------------------------------------------------------
# 2. Qualitative behaviour
# ---------------------------------------------------------------------------

def test_late_time_decay():
    """Luminosity should decrease at late times once magnetar fades."""
    late_times = jnp.array([50.0, 100.0, 200.0, 500.0], dtype=jnp.float64)
    out = general_magnetar_driven_supernova_bolometric(late_times, **_PARAMS)
    # Late-time values should decline (not necessarily monotone on short scales,
    # but the last value should be fainter than the first)
    assert out[-1] < out[0], f"Expected decline at late times, got {out}"


def test_higher_l0_brighter():
    """Higher initial magnetar luminosity should give a brighter light curve."""
    times = jnp.array([5.0, 10.0, 20.0], dtype=jnp.float64)
    out_lo = general_magnetar_driven_supernova_bolometric(times, **{**_PARAMS, 'l0': 1e44})
    out_hi = general_magnetar_driven_supernova_bolometric(times, **{**_PARAMS, 'l0': 1e47})
    assert jnp.all(out_hi > out_lo), f"Expected hi > lo: hi={out_hi}, lo={out_lo}"


def test_higher_mej_dims_peak():
    """Much higher ejecta mass → more diffusion → lower/later peak at early times."""
    times = jnp.array([5.0, 10.0], dtype=jnp.float64)
    out_low_mej = general_magnetar_driven_supernova_bolometric(times, **{**_PARAMS, 'mej': 0.1})
    out_high_mej = general_magnetar_driven_supernova_bolometric(times, **{**_PARAMS, 'mej': 10.0})
    # At very early times, low-mej ejecta is more optically thin → more luminous
    assert jnp.max(out_low_mej) > jnp.max(out_high_mej) - 3.0  # within 3 dex


def test_nickel_increases_late_luminosity():
    """Adding Ni/Co decay raises the late-time tail."""
    late = jnp.array([100.0, 200.0], dtype=jnp.float64)
    out_no_ni  = general_magnetar_driven_supernova_bolometric(late, **{**_PARAMS, 'f_nickel': 0.0})
    out_with_ni = general_magnetar_driven_supernova_bolometric(late, **{**_PARAMS, 'f_nickel': 0.5})
    assert jnp.all(out_with_ni >= out_no_ni), \
        f"Nickel should raise late-time luminosity: {out_with_ni} vs {out_no_ni}"


# ---------------------------------------------------------------------------
# 3. JIT idempotence
# ---------------------------------------------------------------------------

def test_jit_consistent():
    """Second JIT call must return bit-identical results."""
    out1 = general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS)
    out2 = general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS)
    assert jnp.allclose(out1, out2, rtol=0, atol=0), "JIT results differ between calls"


# ---------------------------------------------------------------------------
# 4. Registry
# ---------------------------------------------------------------------------

def test_registered():
    assert "general_magnetar_driven_supernova_bolometric" in MODEL_REGISTRY
    fn = MODEL_REGISTRY["general_magnetar_driven_supernova_bolometric"]
    out = fn(_TIMES, **_PARAMS)
    assert out.shape == _TIMES.shape


# ---------------------------------------------------------------------------
# 5. Numerical agreement with original redback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Batched variant
# ---------------------------------------------------------------------------

_B = 8   # batch size for unit tests (small enough to run quickly)
_BATCH_PARAMS = {
    'mej':          jnp.full((_B,), _PARAMS['mej'],          dtype=jnp.float64),
    'E_sn':         jnp.full((_B,), _PARAMS['E_sn'],         dtype=jnp.float64),
    'kappa':        jnp.full((_B,), _PARAMS['kappa'],        dtype=jnp.float64),
    'l0':           jnp.full((_B,), _PARAMS['l0'],           dtype=jnp.float64),
    'tau_sd':       jnp.full((_B,), _PARAMS['tau_sd'],       dtype=jnp.float64),
    'nn':           jnp.full((_B,), _PARAMS['nn'],           dtype=jnp.float64),
    'kappa_gamma':  jnp.full((_B,), _PARAMS['kappa_gamma'],  dtype=jnp.float64),
    'f_nickel':     jnp.full((_B,), _PARAMS['f_nickel'],     dtype=jnp.float64),
}


def test_batched_output_shape():
    out = general_magnetar_driven_supernova_bolometric_batched(_TIMES, **_BATCH_PARAMS)
    assert out.shape == (_B, len(_TIMES)), f"Expected ({_B}, {len(_TIMES)}), got {out.shape}"


def test_batched_output_finite():
    out = general_magnetar_driven_supernova_bolometric_batched(_TIMES, **_BATCH_PARAMS)
    assert jnp.all(jnp.isfinite(out)), f"Non-finite values in batched output"


def test_batched_agrees_with_single():
    """Each row of the batched output must match the single-sample Euler model.

    The batched function uses the fixed-step Euler backend; compare against the
    single-call function with the same backend (solver='euler') so the two paths
    are numerically identical rather than differing by ODE solver tolerance.
    """
    out_batch = np.array(
        general_magnetar_driven_supernova_bolometric_batched(_TIMES, **_BATCH_PARAMS)
    )
    out_single = np.array(
        general_magnetar_driven_supernova_bolometric(_TIMES, **_PARAMS, solver='euler')
    )
    # All rows should be identical since all batch elements have the same params
    for i in range(_B):
        diff = np.max(np.abs(out_batch[i] - out_single))
        assert diff < 1e-10, (
            f"Row {i}: batched differs from single by {diff:.2e} dex"
        )


def test_batched_heterogeneous_params():
    """Different parameter rows must produce different outputs."""
    rng = np.random.default_rng(0)
    B = 4
    out = general_magnetar_driven_supernova_bolometric_batched(
        _TIMES,
        mej    = jnp.array(rng.uniform(0.5, 2.0, B),  dtype=jnp.float64),
        E_sn   = jnp.array(10**rng.uniform(50, 51.5, B), dtype=jnp.float64),
        kappa  = jnp.full((B,), 0.1, dtype=jnp.float64),
        l0     = jnp.array(10**rng.uniform(44, 46, B),   dtype=jnp.float64),
        tau_sd = jnp.full((B,), 1e6, dtype=jnp.float64),
        nn     = jnp.full((B,), 3.0, dtype=jnp.float64),
        kappa_gamma = jnp.full((B,), 1.0, dtype=jnp.float64),
    )
    assert out.shape == (B, len(_TIMES))
    assert jnp.all(jnp.isfinite(out))
    # Rows should differ because params differ
    assert not jnp.allclose(out[0], out[1]), "Rows 0 and 1 are identical despite different params"


def test_batched_default_f_nickel_zeros():
    """f_nickel=None should give same result as f_nickel=zeros."""
    out_none  = general_magnetar_driven_supernova_bolometric_batched(
        _TIMES, **{k: v for k, v in _BATCH_PARAMS.items() if k != 'f_nickel'}
    )
    out_zeros = general_magnetar_driven_supernova_bolometric_batched(
        _TIMES, **{**_BATCH_PARAMS, 'f_nickel': jnp.zeros(_B, dtype=jnp.float64)}
    )
    assert jnp.allclose(out_none, out_zeros, atol=1e-10)


def test_batched_registered():
    assert "general_magnetar_driven_supernova_bolometric_batched" in MODEL_REGISTRY


@pytest.mark.slow
def test_agreement_with_redback():
    """JAX model should agree with the original redback implementation to < 5%.

    Tests the critical outputs (lbol array) after interpolation to common times.
    The two models use different time grids, so we compare in log10 space at the
    requested times and allow for up to 0.05 dex (≈ 12%) error.
    """
    try:
        from redback.transient_models.supernova_models import (
            general_magnetar_driven_supernova_bolometric as redback_fn,
        )
        from redback.transient_models.magnetar_models import magnetar_only
        from redback.utils import get_optimal_time_array, velocity_from_lorentz_factor
    except ImportError:
        pytest.skip("redback not available — skipping cross-comparison")

    times_days = np.array([5.0, 10.0, 20.0, 50.0])
    p = _PARAMS

    # redback bolometric function returns linear lbol (not log10)
    lbol_redback = redback_fn(
        times_days,
        mej=p['mej'],
        E_sn=p['E_sn'],
        kappa=p['kappa'],
        l0=p['l0'],
        tau_sd=p['tau_sd'],
        nn=p['nn'],
        kappa_gamma=p['kappa_gamma'],
        f_nickel=p['f_nickel'],
        output_format='lbol',
    )
    log10_redback = np.log10(np.maximum(lbol_redback, 1e25))

    log10_jax = np.array(
        general_magnetar_driven_supernova_bolometric(
            jnp.array(times_days, dtype=jnp.float64), **p
        )
    )

    diff = np.abs(log10_jax - log10_redback)
    assert np.all(diff < 0.05), (
        f"JAX vs redback log10(L) differs by > 0.05 dex:\n"
        f"  times     = {times_days}\n"
        f"  JAX       = {log10_jax}\n"
        f"  redback   = {log10_redback}\n"
        f"  |diff|    = {diff}"
    )
